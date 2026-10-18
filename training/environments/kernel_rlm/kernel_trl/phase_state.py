"""Manual phase-boundary moves for a single FSDP2 policy and its Muon state."""

import time


def move_optimizer_state(optimizer, device) -> int:
    import torch

    moved = 0
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                state[name] = value.to(device)
                moved += 1
            elif not isinstance(value, (int, float, bool)):
                raise ValueError(f"Unexpected optimizer state entry: {name}")
    return moved


def phase_roundtrip(model, optimizer) -> dict:
    import torch
    from kernel_skyrl.framework_probe import local_tensor
    from torch.distributed.fsdp import FSDPModule

    started = time.monotonic()
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()
    torch.cuda.synchronize()
    device = torch.device("cuda", torch.cuda.current_device())
    identities = {name: id(parameter) for name, parameter in model.named_parameters()}
    adapters = {
        name: local_tensor(parameter).detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    momentum = {
        id(parameter): {
            name: local_tensor(value).detach().cpu().clone()
            for name, value in state.items()
            if isinstance(value, torch.Tensor)
        }
        for parameter, state in optimizer.state.items()
    }
    if not momentum:
        raise ValueError("Expected live Muon state after the first update")
    allocated_before = torch.cuda.memory_allocated()
    model.to("cpu")
    state_tensors = move_optimizer_state(optimizer, "cpu")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    offloaded_at = time.monotonic()
    allocated_offloaded = torch.cuda.memory_allocated()
    if not all(local_tensor(parameter).device.type == "cpu" for parameter in model.parameters()):
        raise ValueError("Model was not fully offloaded to CPU")
    if allocated_before > 2**30 and allocated_offloaded > 2**30:
        raise ValueError("Offload left more than 1 GiB allocated; a second model is not safe")
    model.to(device)
    move_optimizer_state(optimizer, device)
    torch.cuda.synchronize()
    if identities != {name: id(parameter) for name, parameter in model.named_parameters()}:
        raise ValueError("Phase moves replaced Parameter identities used by the optimizer")
    for name, parameter in model.named_parameters():
        if name in adapters and not torch.equal(adapters[name], local_tensor(parameter).cpu()):
            raise ValueError(f"Adapter changed during phase move: {name}")
    for parameter, state in optimizer.state.items():
        for name, previous in momentum[id(parameter)].items():
            if not torch.equal(previous, local_tensor(state[name]).cpu()):
                raise ValueError("Muon state changed during phase move")
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    actual = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Optimizer no longer covers exactly the adapters")
    return {
        "allocated_before_bytes": allocated_before,
        "allocated_while_offloaded_bytes": allocated_offloaded,
        "allocated_after_reload_bytes": torch.cuda.memory_allocated(),
        "offload_seconds": offloaded_at - started,
        "roundtrip_seconds": time.monotonic() - started,
        "optimizer_state_tensors_moved": state_tensors,
        "adapter_tensors_bitwise_preserved": len(adapters),
        "momentum_bitwise_preserved": True,
        "parameter_identities_preserved": True,
    }
