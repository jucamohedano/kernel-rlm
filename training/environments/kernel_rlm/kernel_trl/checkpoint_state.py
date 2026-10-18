"""Compact phase-boundary checkpoint state for two-rank attention LoRA/Muon."""

import hashlib
import json
import random
from pathlib import Path


def portable_adapter_name(name: str) -> str:
    return name.replace("._checkpoint_wrapped_module.", ".").replace(".default.", ".")


def checkpoint_metadata(directory: Path, expected_step: int) -> dict:
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["step"] != expected_step or metadata["world_size"] != 2:
        raise ValueError("Checkpoint step or world size does not match the sampled policy")
    if metadata.get("frozen_weights_saved") is not False:
        raise ValueError("Expected a compact adapter-only checkpoint")
    for name, expected in metadata.get("sha256", {}).items():
        if Path(name).name != name:
            raise ValueError("Checkpoint manifest contains a nonlocal path")
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    return metadata


def load_adapters(model, directory: Path) -> int:
    import torch
    from safetensors.torch import load_file

    saved = load_file(directory / "adapter_model.safetensors")
    weights = {portable_adapter_name(name): value for name, value in saved.items()}
    if len(weights) != len(saved):
        raise ValueError("Checkpoint adapter names collide after removing framework wrappers")
    parameters = {
        portable_adapter_name(name): parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(weights) != set(parameters):
        raise ValueError("Checkpoint does not contain exactly the current LoRA parameters")
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = weights[name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError(f"Adapter shape or dtype mismatch: {name}")
            parameter.copy_(value)
    return len(weights)


def restore_training_state(trainer, directory: Path, step: int) -> dict:
    import numpy as np
    import torch
    import torch.distributed as dist
    from kernel_skyrl.framework_probe import local_tensor
    from torch.distributed.tensor import DTensor

    metadata = checkpoint_metadata(directory, step)
    state = torch.load(
        directory / f"state-rank-{dist.get_rank()}.pt", map_location="cpu", weights_only=True
    )
    optimizer = trainer.optimizer.optimizer
    parameters = {
        name: parameter
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad
    }
    if state["step"] != step or set(state["adapters"]) != set(parameters):
        raise ValueError("Local checkpoint parameter identities or step differ")
    if set(state["momentum"]) != set(parameters):
        raise ValueError("Checkpoint lacks Muon momentum for an adapter")
    for name, parameter in parameters.items():
        current = local_tensor(parameter).detach().cpu()
        if not torch.equal(current, state["adapters"][name]):
            raise ValueError(f"Restored adapter shard differs: {name}")
        value = state["momentum"][name]
        if value.shape != current.shape or value.dtype != current.dtype:
            raise ValueError(f"Muon momentum shape or dtype mismatch: {name}")
        optimizer.state[parameter]["momentum"] = DTensor.from_local(
            value.to(parameter.device),
            device_mesh=parameter.device_mesh,
            placements=parameter.placements,
            shape=parameter.shape,
            stride=parameter.stride(),
        )
    if len(metadata["muon_groups"]) != len(optimizer.param_groups):
        raise ValueError("Optimizer group count changed")
    for current, previous in zip(optimizer.param_groups, metadata["muon_groups"], strict=True):
        for name, value in previous.items():
            if name != "step" and json.loads(json.dumps(current.get(name))) != value:
                raise ValueError(f"Optimizer setting changed: {name}")
        if previous.get("step") != step:
            raise ValueError("Muon optimizer counter disagrees with policy version")
        current["step"] = step
    if "scheduler" in state:
        trainer.lr_scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["python_rng"])
        saved_numpy = state["numpy_rng"]
        np.random.set_state(
            (saved_numpy[0], np.array(saved_numpy[1], dtype="uint32"), *saved_numpy[2:])
        )
    torch.set_rng_state(state["rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    return {
        "step": step,
        "adapter_shards_bitwise_equal": len(parameters),
        "momentum_shards_restored": len(state["momentum"]),
        "muon_step_restored": True,
        "full_rng_and_scheduler_restored": "scheduler" in state,
        "legacy_checkpoint": "sha256" not in metadata,
    }


def save_training_state(trainer, directory: Path, step: int) -> Path:
    import numpy as np
    import torch
    import torch.distributed as dist
    from kernel_skyrl.framework_probe import local_tensor
    from safetensors.torch import save_file

    rank = dist.get_rank()
    output = directory / f"checkpoint-step-{step}"
    if rank == 0:
        output.mkdir(exist_ok=False)
    dist.barrier()
    model = trainer.model
    optimizer = trainer.optimizer.optimizer
    full, local, momentum = {}, {}, {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        local[name] = local_tensor(parameter).detach().cpu().clone()
        parameter_state = optimizer.state[parameter]
        if set(parameter_state) != {"momentum"}:
            raise ValueError(f"Unexpected Muon state fields for {name}")
        momentum[name] = local_tensor(parameter_state["momentum"]).detach().cpu().clone()
        value = parameter.full_tensor().detach().cpu()
        if rank == 0:
            key = portable_adapter_name(name)
            if key in full:
                raise ValueError("Adapter names collide after removing framework wrappers")
            full[key] = value.contiguous()
    numpy_rng = np.random.get_state()
    state = {
        "adapters": local,
        "momentum": momentum,
        "step": step,
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
        "python_rng": random.getstate(),
        "numpy_rng": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
        "scheduler": trainer.lr_scheduler.state_dict(),
    }
    torch.save(state, output / f"state-rank-{rank}.pt")
    if rank == 0:
        save_file(full, output / "adapter_model.safetensors")
        model.peft_config["default"].save_pretrained(output)
    dist.barrier()
    if rank == 0:
        metadata = {
            "step": step,
            "world_size": 2,
            "adapter_tensors": len(full),
            "frozen_weights_saved": False,
            "muon_groups": [
                {name: value for name, value in group.items() if name != "params"}
                for group in optimizer.param_groups
            ],
            "sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(output.iterdir())
                if path.is_file()
            },
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    dist.barrier()
    return output
