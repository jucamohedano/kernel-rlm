"""One phase-shared TRL/Muon update with compact policy continuation, or audit only."""

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path


def run(arguments) -> dict:
    import torch
    import torch.distributed as dist
    from datasets import Dataset
    from kernel_skyrl.framework_hooks import (
        materialize_meta_adapters,
        materialize_nonpersistent_buffers,
        trl_muon_trainer,
    )
    from kernel_skyrl.framework_probe import observe_updates
    from kernel_skyrl.laguna_policy import attach_adapters, audit_storage, load_base
    from kernel_skyrl.laguna_probe import event
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig

    from kernel_trl.checkpoint_state import (
        checkpoint_metadata,
        load_adapters,
        restore_training_state,
        save_training_state,
    )
    from kernel_trl.probability_audit import AuditComplete, audit_group
    from kernel_trl.replay_trainer import rlm_grpo_trainer

    group = json.loads(arguments.group.read_text()) if arguments.group else None
    if arguments.readback_only:
        if group is not None or arguments.resume_from is None:
            raise ValueError("Readback requires only a checkpoint, not sampled trajectories")
        version = json.loads((arguments.resume_from / "metadata.json").read_text())["step"]
    else:
        if group is None:
            raise ValueError("A training or probability audit group is required")
        version = group["policy_version"]
        if type(version) is not int or version < 0:
            raise ValueError("Invalid sampled policy version")
        if len(group["rewards"]) not in {4, 8} or len(group["rows"]) % len(group["rewards"]):
            raise ValueError("Expected four or eight complete trajectories with aligned rows")
        if group["trainable_tokens"] <= 0:
            raise ValueError("No real trainable tokens")
    if version and arguments.resume_from is None:
        raise ValueError("Nonzero policy version requires its checkpoint")
    if arguments.resume_from:
        checkpoint_metadata(arguments.resume_from, version)
    if group and not arguments.audit_only and max(group["rewards"]) <= 0.1:
        raise ValueError("No reward above 0.1: inspect trajectories before further updates")
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
        if rank == 0 and arguments.resume_from:
            load_adapters(model, arguments.resume_from)
        model.config.use_cache = False
        materialize_meta_adapters(model)
        materialize_nonpersistent_buffers(model)
        tokenizer = AutoTokenizer.from_pretrained(
            str(arguments.checkpoint), trust_remote_code=False
        )
        updates = []
        row_count = len(group["rows"]) if group else 4
        gas = row_count // 2
        settings = GRPOConfig(
            output_dir=str(directory / "trainer-output"),
            bf16=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=gas,
            steps_per_generation=gas,
            num_generations=len(group["rewards"]) if group else 4,
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

        def observe_optimizer(optimizer):
            observe_updates(optimizer, model, directory, updates)

        trainer_class = (
            rlm_grpo_trainer(mesh, observe_optimizer, group)
            if group
            else trl_muon_trainer(mesh, observe_optimizer)
        )
        trainer = trainer_class(
            model=model,
            args=settings,
            train_dataset=Dataset.from_dict({"prompt": ["unused-exact-token-replay"] * row_count}),
            reward_funcs=lambda completions, **kwargs: [0.0] * len(completions),
            processing_class=tokenizer,
        )
        resumed = {}
        audit = {}
        readback = {}

        class PhaseStart(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                if arguments.resume_from:
                    resumed.update(restore_training_state(trainer, arguments.resume_from, version))
                    event(directory, "compact_resume_verified", **resumed)
                if arguments.readback_only:
                    output = save_training_state(trainer, directory, version)
                    readback.update(checkpoint=output.name)
                    raise AuditComplete()
                if arguments.audit_only:
                    audit.update(audit_group(trainer, group, directory))
                    event(directory, "probability_audit_complete", **audit)
                    raise AuditComplete()

        trainer.add_callback(PhaseStart())
        try:
            result = trainer.train()
        except AuditComplete:
            if updates:
                raise ValueError("Audit-only execution performed an optimizer update") from None
            return {
                "status": "readback" if arguments.readback_only else "audited",
                "global_step": version,
                "optimizer_steps": 0,
                "probability_audit": audit,
                "resume": resumed,
                **readback,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "elapsed_seconds": time.monotonic() - started,
            }
        if trainer.state.global_step != 1 or len(updates) != 1:
            raise ValueError("Expected one actual optimizer update")
        output = save_training_state(trainer, directory, version + 1)
        return {
            "status": "passed",
            "global_step": version + 1,
            "local_optimizer_steps": 1,
            "sampled_policy_version": version,
            "resume": resumed,
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
            "checkpoint": output.name,
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
    parser.add_argument("--group", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--readback-only", action="store_true")
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    result = run(arguments)
    (arguments.output_dir / f"report-rank-{os.environ['RANK']}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
