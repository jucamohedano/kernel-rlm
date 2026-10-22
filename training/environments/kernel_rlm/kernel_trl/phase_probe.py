"""Bounded native TRL generation, Muon updates and single-policy phase sharing."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
from datetime import timedelta
from pathlib import Path

from kernel_training import diagnostics, laguna_policy
from kernel_training.diagnostics import event, observe_updates, tiny_model
from kernel_training.laguna_policy import attach_adapters, audit_storage, load_base, load_config

from kernel_trl import framework_hooks, phase_state
from kernel_trl.framework_hooks import (
    materialize_meta_adapters,
    materialize_nonpersistent_buffers,
    trl_muon_trainer,
)
from kernel_trl.phase_state import phase_roundtrip


def tokenizer_and_prompt(checkpoint: Path | None):
    if checkpoint is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=False)
        messages = [
            {"role": "user", "content": "Write the CUDA expression for a thread's global 1D index."}
        ]
        return tokenizer, messages

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(
        WordLevel({f"token{index}": index for index in range(128)}, unk_token="token3")
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="token9",
        bos_token="token2",
        eos_token="token2",
        unk_token="token3",
    ), "token10"


def train_phases(model, mesh, checkpoint: Path | None, directory: Path) -> dict:
    import torch
    import torch.distributed as dist
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig

    tokenizer, prompt = tokenizer_and_prompt(checkpoint)
    updates = []
    generations = []
    phase_moves = []

    class PhaseObserver(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            original = model.generate

            def traced_generate(*positional, **keywords):
                started = time.monotonic()
                generated = original(*positional, **keywords)
                torch.cuda.synchronize()
                inputs = keywords["input_ids"].detach().cpu().tolist()
                outputs = generated.detach().cpu().tolist()
                entry = {
                    "optimizer_step_before_generation": state.global_step,
                    "elapsed_seconds": time.monotonic() - started,
                    "prompt_ids": inputs,
                    "completion_ids": [
                        output[len(prefix) :]
                        for prefix, output in zip(inputs, outputs, strict=True)
                    ],
                }
                entry["completion_text"] = tokenizer.batch_decode(
                    entry["completion_ids"], skip_special_tokens=False
                )
                generations.append(entry)
                event(directory, "native_generation", **entry)
                return generated

            model.generate = traced_generate

        def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
            if state.global_step == 1:
                actual_optimizer = optimizer.optimizer
                event(directory, "offload_started")
                result = phase_roundtrip(model, actual_optimizer)
                phase_moves.append(result)
                event(directory, "offload_reload_complete", **result)

    def fixture_reward(completions, **kwargs):
        return [
            (dist.get_rank() * len(completions) + index) / 3 for index in range(len(completions))
        ]

    arguments = GRPOConfig(
        output_dir=str(directory / "trainer-output"),
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=4,
        max_steps=2,
        max_completion_length=8,
        generation_kwargs={"min_new_tokens": 8, "use_cache": True},
        chat_template_kwargs={"enable_thinking": True},
        mask_truncated_completions=False,
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
        mesh, lambda optimizer: observe_updates(optimizer, model, directory, updates)
    )
    trainer = trainer_class(
        model=model,
        reward_funcs=fixture_reward,
        args=arguments,
        train_dataset=Dataset.from_dict({"prompt": [prompt] * 8}),
        processing_class=tokenizer,
        callbacks=[PhaseObserver()],
    )
    result = trainer.train()
    if trainer.state.global_step != 2 or len(updates) != 2 or len(phase_moves) != 1:
        raise ValueError("Expected two native Muon updates with one phase roundtrip")
    if [record["optimizer_step_before_generation"] for record in generations] != [0, 1]:
        raise ValueError("Expected native generations both before and after the first update")
    optimizer = trainer.optimizer.optimizer
    if [group["step"] for group in optimizer.param_groups] != [2]:
        raise ValueError("Muon step state did not survive phase moves")
    return {
        "status": "passed",
        "native_path": "GRPOTrainer native Transformers generation and two Muon updates, same FSDP2 policy",
        "scope": "Eight-token synthetic-reward phase diagnostic, not an RLM curriculum run",
        "full_pretrained_checkpoint": checkpoint is not None,
        "updates": updates,
        "phase_roundtrips": phase_moves,
        "generations": generations,
        "global_step": trainer.state.global_step,
        "muon_step": 2,
        "inventory_after": audit_storage(trainer.model),
        "metrics": result.metrics,
        "log_history": trainer.state.log_history,
        "single_policy_no_second_model_copy": True,
        "use_vllm": False,
        "no_custom_rollout_function": True,
        "synthetic_rewards_not_kernel_correctness": True,
        "thinking_enabled_for_full_prompt": checkpoint is not None,
        "eight_token_cap_is_deliberate_diagnostic_not_curriculum_setting": True,
        "not_tested": [
            "vLLM colocation/sleep/reload",
            "RLM changing-prefix trajectory credit",
            "CUDA verifier sharing",
            "compact checkpoint restart",
            "realistic context and throughput",
            "quality improvement",
        ],
    }


def run(checkpoint: Path | None, directory: Path) -> dict:
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoModelForCausalLM

    if int(os.environ["WORLD_SIZE"]) != 2 or torch.cuda.device_count() != 2:
        raise ValueError("This gate requires exactly two allocated GPUs")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("nccl", timeout=timedelta(seconds=180), device_id=device)
    failed = False
    try:
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("fsdp",))
        event(directory, "load_started", full_checkpoint=checkpoint is not None)
        if checkpoint is None:
            with torch.device("cpu" if dist.get_rank() == 0 else "meta"):
                model = tiny_model().to(torch.bfloat16)
        elif dist.get_rank() == 0:
            model, information = load_base(checkpoint)
            (directory / "loading-information.json").write_text(json.dumps(information, indent=2))
        else:
            configuration = load_config(checkpoint)
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(
                    configuration, dtype=torch.bfloat16, attn_implementation="sdpa"
                )
        model.config.use_cache = False
        model = attach_adapters(model)
        materialize_meta_adapters(model)
        materialize_nonpersistent_buffers(model)
        event(directory, "cpu_model_ready", inventory=audit_storage(model))
        result = train_phases(model, mesh, checkpoint, directory)
        torch.cuda.synchronize()
        result.update(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        return result
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
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    directory = arguments.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    rank = os.environ["RANK"]
    snapshot = directory / "source" / f"rank-{rank}"
    snapshot.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for module in (framework_hooks, diagnostics, laguna_policy, phase_state):
        path = Path(module.__file__)
        shutil.copy2(path, snapshot / path.name)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    started = time.monotonic()
    try:
        result = run(arguments.checkpoint, directory)
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
    result.update(
        rank=int(rank),
        elapsed_seconds=time.monotonic() - started,
        source_hashes=hashes,
        versions={
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "accelerate", "trl", "dion")
        },
    )
    (directory / f"report-rank-{rank}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
