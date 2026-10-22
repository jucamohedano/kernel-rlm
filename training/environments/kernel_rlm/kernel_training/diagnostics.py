"""Trainer-neutral model fixtures and bounded distributed update diagnostics."""

import importlib.metadata
import json
import os
import time
from datetime import timedelta
from pathlib import Path

from kernel_training.laguna_policy import attach_adapters, audit_storage, load_base, load_config


def event(directory: Path, phase: str, **fields) -> None:
    with (directory / f"events-rank-{os.environ.get('RANK', 'cpu')}.jsonl").open("a") as stream:
        stream.write(json.dumps({"phase": phase, "monotonic": time.monotonic(), **fields}) + "\n")


def tiny_model(num_experts: int = 4):
    from transformers import AutoModelForCausalLM, LagunaConfig

    config = LagunaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=num_experts,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_attention_heads_per_layer=[4, 4],
        mlp_layer_types=["dense", "sparse"],
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=16,
        max_position_embeddings=128,
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
        bos_token_id=2,
        eos_token_id=2,
        pad_token_id=9,
    )
    return AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")


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


def run_framework_probe(
    arguments, directory: Path, *, backend: str, update, prepare=None, versions: tuple[str, ...]
) -> dict:
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
            backend=backend,
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
        if prepare is not None:
            materialized_adapters, buffers = prepare(model)
            event(
                directory,
                "nonpersistent_buffers_materialized",
                buffers=buffers,
                meta_adapters_materialized_on_cpu=materialized_adapters,
            )
        records = []
        result = update(model, mesh, directory, records)
        torch.cuda.synchronize()
        return {
            "status": "passed",
            "kind": "backend-compatibility-diagnostic-not-quality-experiment",
            "backend": backend,
            "full_pretrained_checkpoint": bool(arguments.checkpoint),
            "inventory_before": inventory,
            "updates": records,
            "nonpersistent_buffers_materialized_before_accelerate": buffers,
            "meta_adapters_materialized_on_cpu": materialized_adapters,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "versions": {name: importlib.metadata.version(name) for name in versions},
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
