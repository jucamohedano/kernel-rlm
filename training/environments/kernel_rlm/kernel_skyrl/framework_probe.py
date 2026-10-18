"""One bounded native-framework update on two GPUs; no inference service or task training."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
from datetime import timedelta
from pathlib import Path

from kernel_skyrl import framework_hooks, laguna_policy, laguna_probe
from kernel_skyrl.framework_hooks import (
    materialize_meta_adapters,
    materialize_nonpersistent_buffers,
    skyrl_muon_strategy,
    trl_muon_trainer,
)
from kernel_skyrl.laguna_policy import attach_adapters, audit_storage, load_base, load_config
from kernel_skyrl.laguna_probe import event, tiny_model


def local_tensor(parameter):
    return parameter.to_local() if hasattr(parameter, "to_local") else parameter


def observe_updates(optimizer, model, directory: Path, records: list):
    import torch
    from torch.distributed.tensor import DTensor

    snapshot = {}

    def before_step(current, args, kwargs):
        parameters = current.param_groups[0]["params"]
        if not all(isinstance(parameter, DTensor) for parameter in parameters):
            raise ValueError("Muon must update actual FSDP2 DTensors")
        expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        if len({id(parameter) for parameter in parameters}) != len(parameters) or expected != {
            id(parameter) for parameter in parameters
        }:
            raise ValueError("Optimizer parameter remapping lost or duplicated trainable adapters")
        nonzero = sum(
            parameter.grad is not None and bool(torch.count_nonzero(local_tensor(parameter.grad)))
            for parameter in parameters
        )
        if nonzero == 0 or not all(
            parameter.grad is None or bool(torch.isfinite(local_tensor(parameter.grad)).all())
            for parameter in parameters
        ):
            raise ValueError("Expected finite, nonzero autograd gradients")
        if any(
            parameter.grad is not None
            for parameter in model.parameters()
            if not parameter.requires_grad
        ):
            raise ValueError("A frozen base parameter received gradients")
        snapshot.update(
            {id(parameter): local_tensor(parameter).detach().clone() for parameter in parameters}
        )
        records.append({"nonzero_gradient_tensors": nonzero, "adapter_tensors": len(parameters)})
        event(directory, "autograd_before_muon", **records[-1])

    def after_step(current, args, kwargs):
        parameters = current.param_groups[0]["params"]
        changed = sum(
            not torch.equal(snapshot[id(parameter)], local_tensor(parameter))
            for parameter in parameters
        )
        if changed == 0 or not all(
            bool(torch.isfinite(local_tensor(parameter)).all()) for parameter in parameters
        ):
            raise ValueError("Muon failed to make a finite adapter update")
        records[-1]["changed_adapter_tensors"] = changed
        records[-1]["optimizer_class"] = type(current).__module__ + "." + type(current).__name__
        event(directory, "native_optimizer_step", **records[-1])
        snapshot.clear()

    optimizer.register_step_pre_hook(before_step)
    optimizer.register_step_post_hook(after_step)


def skyrl_update(model, directory: Path, records: list) -> dict:
    import numpy as np
    import torch
    import torch.distributed as dist
    from skyrl.backends.skyrl_train.utils.ppo_utils import (
        compute_grpo_outcome_advantage,
        ppo_policy_loss,
    )
    from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
    from skyrl.train.config import AlgorithmConfig, FSDPConfig, ModelConfig, OptimizerConfig

    configuration = ModelConfig()
    configuration.lora.rank = 8
    configuration.lora.alpha = 16
    strategy = skyrl_muon_strategy()(
        FSDPConfig(
            fsdp_size=2, wrap_policy={"transformer_layer_cls_to_wrap": ["LagunaDecoderLayer"]}
        ),
        optimizer_config=OptimizerConfig(lr=1e-5, weight_decay=0.01, offload_after_step=False),
        model_config=configuration,
        num_training_steps=1,
    )
    strategy.setup_distributed()
    wrapper = HFModelWrapper(model, use_flash_attention_2=False, remove_microbatch_padding=False)
    wrapper.logprobs_chunk_size = 16
    wrapper, optimizer, scheduler = strategy.prepare((wrapper, None, None))
    observe_updates(optimizer, wrapper, directory, records)
    wrapper.train()
    mask = torch.tensor([[1, 0, 1]] * 4, device="cuda", dtype=torch.float32)
    rewards = torch.zeros_like(mask)
    rewards[:, -1] = torch.tensor([0, 1 / 3, 2 / 3, 1], device="cuda")
    advantages, returns = compute_grpo_outcome_advantage(rewards, mask, np.array([0] * 4))
    losses = []
    algorithm = AlgorithmConfig(use_kl_loss=False)
    for microbatch in range(2):
        sample = dist.get_rank() * 2 + microbatch
        sequences = torch.tensor([[2, 10, 11 + sample % 2, 7, 2]], device="cuda")
        logprobs = wrapper(sequences, num_actions=3, attention_mask=torch.ones_like(sequences))
        loss, metrics = ppo_policy_loss(
            logprobs,
            logprobs.detach(),
            advantages[sample : sample + 1],
            algorithm,
            mask[sample : sample + 1],
        )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite native SkyRL policy loss")
        strategy.backward(loss / 2, wrapper, optimizer)
        losses.append(float(loss.detach()))
    norm = strategy.optimizer_step(optimizer, wrapper, scheduler)
    if norm is None or not bool(torch.isfinite(norm)):
        raise ValueError("Native SkyRL gradient clipping did not return a finite norm")
    return {
        "native_path": "FSDPStrategy.prepare/backward/optimizer_step + HFModelWrapper + GRPO advantages/PPO loss",
        "losses": losses,
        "gradient_norm": float(local_tensor(norm)),
        "optimizer_steps": len(records),
        "ray_orchestration_tested": False,
        "model_inventory_after": audit_storage(wrapper),
    }


def trl_update(model, mesh, directory: Path, records: list) -> dict:
    import torch
    import torch.distributed as dist
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from trl import GRPOConfig

    vocabulary = {f"token{index}": index for index in range(128)}
    tokenizer_backend = Tokenizer(WordLevel(vocabulary, unk_token="token3"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        pad_token="token9",
        bos_token="token2",
        eos_token="token2",
        unk_token="token3",
    )

    def rollout(prompts, trainer):
        samples = [dist.get_rank() * 2 + index for index in range(len(prompts))]
        sequences = torch.tensor(
            [[2, 10, 11 + sample % 2, 7, 2] for sample in samples], device="cuda"
        )
        with torch.no_grad():
            scores = (
                trainer.model(sequences, use_cache=False).logits[:, 1:-1].float().log_softmax(-1)
            )
            logprobs = scores.gather(-1, sequences[:, 2:].unsqueeze(-1)).squeeze(-1).tolist()
        return {
            "prompt_ids": [[2, 10] for sample in samples],
            "completion_ids": [[11 + sample % 2, 7, 2] for sample in samples],
            "logprobs": logprobs,
            "env_mask": [[1, 0, 1] for sample in samples],
            "fixture_reward": [sample / 3 for sample in samples],
        }

    def reward(fixture_reward, **kwargs):
        return fixture_reward

    arguments = GRPOConfig(
        output_dir=str(directory / "trainer-output"),
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=4,
        max_steps=1,
        max_completion_length=3,
        learning_rate=1e-5,
        weight_decay=0.01,
        lr_scheduler_type="constant",
        beta=0.0,
        report_to="none",
        save_strategy="no",
        logging_steps=1,
        disable_tqdm=True,
        gradient_checkpointing=False,
        use_vllm=False,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "version": 2,
            "transformer_layer_cls_to_wrap": ["LagunaDecoderLayer"],
            "cpu_ram_efficient_loading": True,
            "activation_checkpointing": True,
            "reshard_after_forward": True,
        },
    )
    trainer_class = trl_muon_trainer(
        mesh, lambda optimizer: observe_updates(optimizer, model, directory, records)
    )
    trainer = trainer_class(
        model=model,
        reward_funcs=reward,
        args=arguments,
        train_dataset=Dataset.from_dict({"prompt": ["token10"] * 4}),
        processing_class=tokenizer,
        rollout_func=rollout,
    )
    result = trainer.train()
    if trainer.state.global_step != 1 or len(records) != 1:
        raise ValueError("Expected exactly one native TRL optimizer update")
    return {
        "native_path": "GRPOTrainer.train + Accelerate FSDP2 + custom rollout/env_mask + Muon optimizer hook",
        "global_step": trainer.state.global_step,
        "metrics": result.metrics,
        "log_history": trainer.state.log_history,
        "model_inventory_after": audit_storage(trainer.model),
        "synthetic_rollouts_not_kernel_tasks": True,
        "rollout_logprobs_scored_by_current_policy": True,
    }


def run(arguments, directory: Path) -> dict:
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoModelForCausalLM

    if int(os.environ["WORLD_SIZE"]) != 2 or torch.cuda.device_count() != 2:
        raise ValueError("This bounded gate requires exactly two GPUs")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        "nccl",
        timeout=timedelta(seconds=300),
        device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])),
    )
    failed = False
    try:
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("fsdp",))
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        event(
            directory,
            "load_started",
            backend=arguments.backend,
            full_checkpoint=bool(arguments.checkpoint),
        )
        if arguments.checkpoint is None:
            with torch.device("cpu" if rank == 0 else "meta"):
                model = tiny_model().to(torch.bfloat16)
        elif rank == 0:
            model, information = load_base(arguments.checkpoint)
            (directory / "loading-information.json").write_text(json.dumps(information, indent=2))
        else:
            configuration = load_config(arguments.checkpoint)
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(
                    configuration, dtype=torch.bfloat16, attn_implementation="sdpa"
                )
        model.config.use_cache = False
        model = attach_adapters(model)
        inventory = audit_storage(model)
        event(directory, "cpu_model_ready", inventory=inventory)
        buffers = []
        materialized_adapters = 0
        if arguments.backend == "trl":
            materialized_adapters = materialize_meta_adapters(model)
            buffers = materialize_nonpersistent_buffers(model)
            event(
                directory,
                "nonpersistent_buffers_materialized",
                buffers=buffers,
                meta_adapters_materialized_on_cpu=materialized_adapters,
            )
        records = []
        result = (
            skyrl_update(model, directory, records)
            if arguments.backend == "skyrl"
            else trl_update(model, mesh, directory, records)
        )
        torch.cuda.synchronize()
        return {
            "status": "passed",
            "kind": "backend-compatibility-diagnostic-not-quality-experiment",
            "backend": arguments.backend,
            "full_pretrained_checkpoint": bool(arguments.checkpoint),
            "inventory_before": inventory,
            "updates": records,
            "nonpersistent_buffers_materialized_before_accelerate": buffers,
            "meta_adapters_materialized_on_cpu": materialized_adapters,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "peft", "accelerate", "trl", "dion")
            },
            "not_tested": [
                "vLLM generation/colocation",
                "RLM trajectory credit across changing context",
                "CUDA verifier",
                "checkpoint/resume",
                "realistic context memory",
                "model quality",
            ],
            **result,
        }
    except Exception as error:
        failed = True
        event(
            directory,
            "exception_before_distributed_cleanup",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        if not failed:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("skyrl", "trl"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    directory = arguments.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    rank = os.environ["RANK"]
    snapshot = directory / "source" / f"rank-{rank}"
    snapshot.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for module in (framework_hooks, laguna_policy, laguna_probe):
        path = Path(module.__file__)
        shutil.copy2(path, snapshot / path.name)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    started = time.monotonic()
    try:
        result = run(arguments, directory)
    except Exception as error:
        (directory / f"failure-rank-{rank}.json").write_text(
            json.dumps(
                {
                    "type": type(error).__name__,
                    "error": str(error),
                    "elapsed_seconds": time.monotonic() - started,
                },
                indent=2,
            )
        )
        raise
    result.update(elapsed_seconds=time.monotonic() - started, source_hashes=hashes, rank=int(rank))
    (directory / f"report-rank-{rank}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
