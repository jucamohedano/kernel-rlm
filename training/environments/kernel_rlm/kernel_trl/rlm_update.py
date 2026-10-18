"""One real-data TRL/Muon update; input group is generated and verified beforehand."""

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path


def export_adapter(model, optimizer, directory: Path) -> None:
    import torch
    import torch.distributed as dist
    from kernel_skyrl.framework_probe import local_tensor
    from safetensors.torch import save_file

    rank = dist.get_rank()
    full = {}
    local = {}
    momentum = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        local[name] = local_tensor(parameter).detach().cpu().clone()
        state = optimizer.state.get(parameter, {})
        if "momentum" in state:
            momentum[name] = local_tensor(state["momentum"]).detach().cpu().clone()
        value = parameter.full_tensor().detach().cpu()
        if rank == 0:
            full[name.replace(".default.", ".")] = value.contiguous()
    output = directory / "checkpoint-step-1"
    output.mkdir(exist_ok=True)
    torch.save(
        {
            "adapters": local,
            "momentum": momentum,
            "step": 1,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        },
        output / f"state-rank-{rank}.pt",
    )
    if rank == 0:
        save_file(full, output / "adapter_model.safetensors")
        model.peft_config["default"].save_pretrained(output)
        (output / "metadata.json").write_text(
            json.dumps(
                {
                    "step": 1,
                    "world_size": 2,
                    "adapter_tensors": len(full),
                    "frozen_weights_saved": False,
                    "optimizer_resume_tested": False,
                    "muon_groups": [
                        {name: value for name, value in group.items() if name != "params"}
                        for group in optimizer.param_groups
                    ],
                },
                indent=2,
            )
        )
    dist.barrier()


def run(arguments) -> dict:
    import torch
    import torch.distributed as dist
    from datasets import Dataset
    from kernel_skyrl.framework_hooks import (
        materialize_meta_adapters,
        materialize_nonpersistent_buffers,
    )
    from kernel_skyrl.framework_probe import observe_updates
    from kernel_skyrl.laguna_policy import attach_adapters, audit_storage, load_base
    from kernel_skyrl.laguna_probe import event
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig

    from kernel_trl.replay_trainer import rlm_grpo_trainer

    group = json.loads(arguments.group.read_text())
    if group["policy_version"] != 0 or len(group["rows"]) % 4 or group["trainable_tokens"] <= 0:
        raise ValueError("Invalid or stale one-update group")
    if int(os.environ["WORLD_SIZE"]) != 2 or torch.cuda.device_count() != 2:
        raise ValueError("Exactly two allocated GPUs are required")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        "nccl",
        timeout=timedelta(seconds=240),
        device_id=torch.device("cuda", torch.cuda.current_device()),
    )
    failed = False
    directory = arguments.output_dir
    rank = dist.get_rank()
    started = time.monotonic()
    try:
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("fsdp",))
        torch.manual_seed(42)
        event(directory, "load_started")
        if rank == 0:
            model, information = load_base(arguments.checkpoint)
            (directory / "loading-information.json").write_text(json.dumps(information, indent=2))
        else:
            configuration = AutoConfig.from_pretrained(
                str(arguments.checkpoint), trust_remote_code=False
            )
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(
                    configuration, dtype=torch.bfloat16, attn_implementation="sdpa"
                )
        model = attach_adapters(model)
        model.config.use_cache = False
        materialize_meta_adapters(model)
        materialize_nonpersistent_buffers(model)
        tokenizer = AutoTokenizer.from_pretrained(
            str(arguments.checkpoint), trust_remote_code=False
        )
        updates = []
        gas = len(group["rows"]) // 2
        settings = GRPOConfig(
            output_dir=str(directory / "trainer-output"),
            bf16=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=gas,
            steps_per_generation=gas,
            num_generations=4,
            max_steps=1,
            max_completion_length=4096,
            loss_type="dapo",
            router_aux_loss_coef=0.0,
            learning_rate=1e-5,
            weight_decay=0.01,
            lr_scheduler_type="constant",
            beta=0.0,
            temperature=1.0,
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
        trainer = rlm_grpo_trainer(
            mesh, lambda optimizer: observe_updates(optimizer, model, directory, updates), group
        )(
            model=model,
            args=settings,
            train_dataset=Dataset.from_dict(
                {"prompt": ["unused-exact-token-replay"] * len(group["rows"])}
            ),
            reward_funcs=lambda completions, **kwargs: [0.0] * len(completions),
            processing_class=tokenizer,
        )
        result = trainer.train()
        if trainer.state.global_step != 1 or len(updates) != 1:
            raise ValueError("Expected one actual optimizer update")
        export_adapter(trainer.model, trainer.optimizer.optimizer, directory)
        return {
            "status": "passed",
            "global_step": 1,
            "rewards": group["rewards"],
            "advantages": group["advantages"],
            "root_turns": group["real_rows"],
            "trainable_tokens": group["trainable_tokens"],
            "padded_rows": len(group["rows"]) - group["real_rows"],
            "updates": updates,
            "inventory": audit_storage(trainer.model),
            "metrics": result.metrics,
            "native_loss": "TRL DAPO clipped PPO, externally computed trajectory GRPO advantages",
            "auxiliary_router_loss": False,
            "padding_loss": 0,
            "quality_improvement_claimed": False,
            "checkpoint": "checkpoint-step-1",
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "elapsed_seconds": time.monotonic() - started,
        }
    except Exception as error:
        failed = True
        event(directory, "failure", type=type(error).__name__, error=str(error))
        raise
    finally:
        if not failed:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--group", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    result = run(arguments)
    (arguments.output_dir / f"report-rank-{os.environ['RANK']}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
