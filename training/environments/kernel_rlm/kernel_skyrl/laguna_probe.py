"""Tiny conversion and distributed Muon gates before a full checkpoint download."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
from datetime import timedelta
from pathlib import Path

from kernel_training import diagnostics, laguna_policy
from kernel_training.diagnostics import event, tiny_model
from kernel_training.laguna_policy import (
    MODEL_REVISION,
    attach_adapters,
    audit_storage,
    build_muon,
    load_base,
    load_config,
)


def legacy_state(state: dict) -> dict:
    legacy = {}
    for name, tensor in state.items():
        if name.endswith(".experts.gate_up_proj"):
            prefix = name.removesuffix("gate_up_proj")
            for expert in range(tensor.shape[0]):
                gate, up = tensor[expert].chunk(2, dim=0)
                legacy[f"{prefix}{expert}.gate_proj.weight"] = gate.contiguous()
                legacy[f"{prefix}{expert}.up_proj.weight"] = up.contiguous()
        elif name.endswith(".experts.down_proj"):
            prefix = name.removesuffix("down_proj")
            for expert in range(tensor.shape[0]):
                legacy[f"{prefix}{expert}.down_proj.weight"] = tensor[expert].contiguous()
        elif name.endswith(".gate.e_score_correction_bias"):
            legacy[
                name.replace(".gate.e_score_correction_bias", ".experts.e_score_correction_bias")
            ] = tensor
        else:
            legacy[name.replace(".shared_experts.", ".shared_expert.")] = tensor
    return legacy


def conversion_fixture(directory: Path) -> dict:
    import torch
    from safetensors.torch import save_file

    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = tiny_model(num_experts=12).to(torch.bfloat16)
    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    fixture = directory / "legacy-fixture"
    fixture.mkdir()
    model.config.save_pretrained(fixture)
    save_file(legacy_state(state), fixture / "model.safetensors")
    restored, information = load_base(fixture)
    restored_state = restored.state_dict()
    assert set(state) == set(restored_state)
    assert all(torch.equal(tensor, restored_state[name]) for name, tensor in state.items())
    restored = attach_adapters(restored)
    inventory = audit_storage(restored)
    inputs = torch.tensor([[2, 10, 11, 12]])
    with torch.no_grad():
        logits = restored(inputs, use_cache=False).logits
    assert torch.isfinite(logits).all()
    return {
        "status": "passed",
        "scope": "Tiny random CPU Laguna, exact legacy-to-native tensor check",
        "native_class": type(model).__module__,
        "checkpoint_tensors": len(state),
        "all_tensors_bitwise_equal": True,
        "expert_count": 12,
        "double_digit_expert_order_checked": True,
        "loading_information": information,
        "inventory": inventory,
        "logits_finite": True,
        "gpu_allocated": False,
    }


def shard_model(model, full_state: dict, snapshots: dict, mesh):
    import torch
    import torch.distributed as dist
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    model.to("meta")
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for module in model.modules():
        if type(module).__name__ == "LagunaDecoderLayer" or isinstance(module, torch.nn.Embedding):
            fully_shard(module, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    fully_shard(model, mesh=mesh, mp_policy=policy)
    set_model_state_dict(
        model, full_state, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
    )
    for name, buffer in list(model.named_buffers()):
        if name in model.state_dict():
            continue
        value = (
            snapshots[name].cuda()
            if dist.get_rank() == 0
            else torch.empty(buffer.shape, dtype=buffer.dtype, device="cuda")
        )
        dist.broadcast(value, src=0)
        owner, leaf = name.rsplit(".", 1)
        model.get_submodule(owner).register_buffer(leaf, value, persistent=False)
    return model


def distributed_check(directory: Path, checkpoint: Path | None) -> dict:
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor
    from transformers import AutoModelForCausalLM

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2 or torch.cuda.device_count() != 2:
        raise ValueError("This bounded gate requires exactly two allocated GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        "nccl", timeout=timedelta(seconds=300 if checkpoint is not None else 180), device_id=device
    )
    try:
        torch.manual_seed(42)
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("fsdp",))
        torch.cuda.reset_peak_memory_stats()
        event(directory, "building_cpu_model", full_checkpoint=checkpoint is not None)
        if checkpoint is None:
            model = tiny_model().to(torch.bfloat16)
        elif rank == 0:
            model, information = load_base(checkpoint)
            (directory / "loading-information.json").write_text(json.dumps(information, indent=2))
        else:
            config = load_config(checkpoint)
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(
                    config,
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                    trust_remote_code=False,
                )
        model = attach_adapters(model)
        inventory = audit_storage(model)
        if checkpoint is not None:
            assert inventory["frozen_parameters"] == 33442617088
            assert inventory["adapter_tensors"] == 320
            assert inventory["adapter_parameters"] == 8192000
        full_state = model.state_dict() if rank == 0 else {}
        snapshots = (
            {
                name: buffer.detach().clone()
                for name, buffer in model.named_buffers()
                if name not in model.state_dict()
            }
            if rank == 0
            else {}
        )
        event(directory, "sharding", inventory=inventory)
        model = shard_model(model, full_state, snapshots, mesh)
        full_state.clear()
        event(
            directory,
            "loaded",
            allocated_bytes=torch.cuda.memory_allocated(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        inventory_after = audit_storage(model)
        optimizer = build_muon(model, mesh)
        trainable = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        assert len(optimizer.param_groups[0]["params"]) == len(trainable)
        before = {
            name: parameter.to_local().detach().clone() for name, parameter in trainable.items()
        }
        diagnostic_loss = None
        if checkpoint is None:
            diagnostic_loss = model(
                torch.tensor([[2, 10, 11, 12]], device=device),
                labels=torch.tensor([[2, 10, 11, 12]], device=device),
                use_cache=False,
            ).loss
            assert torch.isfinite(diagnostic_loss)
            diagnostic_loss.backward()
        else:
            for parameter in trainable.values():
                parameter.grad = torch.ones_like(parameter) * 0.001
        nonzero_gradients = sum(
            parameter.grad is not None and bool(torch.count_nonzero(parameter.grad.to_local()))
            for parameter in trainable.values()
        )
        assert nonzero_gradients > 0
        assert all(
            parameter.grad is None
            for parameter in model.parameters()
            if not parameter.requires_grad
        )
        optimizer.step()
        torch.cuda.synchronize()
        changed = sum(
            not torch.equal(before[name], parameter.to_local())
            for name, parameter in trainable.items()
        )
        assert changed == len(trainable)
        assert all(torch.isfinite(parameter.to_local()).all() for parameter in trainable.values())
        event(directory, "muon_step", changed_adapter_tensors=changed)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            logits = model(torch.tensor([[2, 10, 11, 12]], device=device), use_cache=False).logits
        assert torch.isfinite(logits).all()
        torch.cuda.synchronize()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "status": "passed",
            "scope": "Full pretrained BF16 load, synthetic-gradient Muon wiring and four-token forward"
            if checkpoint is not None
            else "Tiny random two-rank Laguna backward and Muon update",
            "framework_driver": "Torch FSDP2 component probe, not the SkyRL/Ray trainer",
            "rank": rank,
            "full_checkpoint": checkpoint is not None,
            "model_revision_requested": MODEL_REVISION if checkpoint is not None else None,
            "inventory": inventory_after,
            "changed_adapter_tensors": changed,
            "synthetic_muon_steps": int(checkpoint is not None),
            "tiny_backward_muon_steps": int(checkpoint is None),
            "diagnostic_loss": float(diagnostic_loss.detach())
            if diagnostic_loss is not None
            else None,
            "nonzero_adapter_gradient_tensors": nonzero_gradients,
            "grpo_steps": 0,
            "logits_finite": True,
            "optimizer_class": type(optimizer).__module__ + "." + type(optimizer).__name__,
            "adjust_lr": "rms_norm",
            "lr": 1e-5,
            "weight_decay": 0.01,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "device_free_bytes_after_forward": free_bytes,
            "device_total_bytes": total_bytes,
            "local_frozen_bytes": sum(
                (parameter.to_local() if isinstance(parameter, DTensor) else parameter).numel()
                * parameter.element_size()
                for parameter in model.parameters()
                if not parameter.requires_grad
            ),
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "transformers",
                    "peft",
                    "triton",
                    "dion",
                    "huggingface-hub",
                    "safetensors",
                )
            },
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu-fixture", "distributed"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("RANK", "cpu")
    snapshots = arguments.output_dir / "source" / f"rank-{rank}"
    snapshots.mkdir(parents=True, exist_ok=False)
    for source in (Path(__file__), Path(laguna_policy.__file__), Path(diagnostics.__file__)):
        shutil.copy2(source, snapshots / source.name)
    try:
        report = (
            conversion_fixture(arguments.output_dir)
            if arguments.mode == "cpu-fixture"
            else distributed_check(arguments.output_dir, arguments.checkpoint)
        )
    except BaseException as error:
        (arguments.output_dir / f"failure-{rank}.json").write_text(
            json.dumps({"error": f"{type(error).__name__}: {error}", "status": "failed"}, indent=2)
        )
        raise
    report["policy_sha256"] = hashlib.sha256(Path(laguna_policy.__file__).read_bytes()).hexdigest()
    (arguments.output_dir / f"report-{rank}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
