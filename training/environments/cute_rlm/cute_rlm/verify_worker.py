"""GPU worker entry point. Run only through the configured external sandbox launcher.

Reference and candidate phases run in distinct processes. The candidate receives
inputs and initial model state, not reference outputs. Numerical results do not
prove CuTe use; a protected external observer must supply compile/launch evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path


def seed_all(torch, np, seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_arrays(torch, value, storages=None):
    if storages is None:
        storages = set()
    if isinstance(value, torch.Tensor):
        if not value.is_contiguous():
            raise NotImplementedError("noncontiguous tensors require an explicit stride contract")
        key = (str(value.device), value.untyped_storage().data_ptr())
        if value.numel() and key in storages:
            raise NotImplementedError("aliased tensor storage requires an explicit alias contract")
        if value.numel():
            storages.add(key)
        if value.is_complex() or value.is_quantized:
            raise NotImplementedError(
                "complex and quantized tensors need an explicit comparison contract"
            )
        cpu = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            array = cpu.float().numpy()
        else:
            array = cpu.numpy()
        return {"tensor_dtype": str(value.dtype), "device": str(value.device), "array": array}
    if isinstance(value, dict):
        return {k: to_arrays(torch, v, storages) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_arrays(torch, v, storages) for v in value)
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise NotImplementedError(f"unsupported value type: {type(value).__name__}")


def from_arrays(torch, value):
    if isinstance(value, dict) and set(value) == {"tensor_dtype", "device", "array"}:
        name = value["tensor_dtype"].removeprefix("torch.")
        return torch.from_numpy(value["array"].copy()).to(
            device=value["device"], dtype=getattr(torch, name)
        )
    if isinstance(value, dict):
        return {k: from_arrays(torch, v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(from_arrays(torch, v) for v in value)
    return value


def reject_cross_tree_aliases(torch, *trees):
    """Fail closed where independent serialization would erase an alias contract."""
    seen = set()

    def visit(value):
        if isinstance(value, torch.Tensor):
            key = (str(value.device), value.untyped_storage().data_ptr())
            if value.numel() and key in seen:
                raise NotImplementedError(
                    "cross-input/output/state aliases need an explicit alias contract"
                )
            if value.numel():
                seen.add(key)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    for tree in trees:
        visit(tree)


def place_inputs(tree, device):
    if isinstance(tree, dict) and set(tree) == {"tensor_dtype", "device", "array"}:
        return {**tree, "device": device}
    if isinstance(tree, dict):
        return {key: place_inputs(value, device) for key, value in tree.items()}
    if isinstance(tree, (tuple, list)):
        return type(tree)(place_inputs(value, device) for value in tree)
    return tree


def load_module(source, name):
    import types

    module = types.ModuleType(name)
    exec(compile(source, name + ".py", "exec"), module.__dict__)
    return module


def run(request):
    # Optional GPU dependencies are imported only inside the isolated worker.
    import numpy as np
    import torch

    from cute_rlm.contracts import VerificationConfig
    from cute_rlm.outputs import load_tree, save_tree

    config = VerificationConfig(**request["config"])
    role = request["role"]
    source = request["source"]
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    if source_hash != request["source_sha256"]:
        raise ValueError("worker source hash mismatch")
    output = Path(request["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    seed_all(torch, np, config.seed)
    module = load_module(source, "task_reference" if role == "reference" else "task_candidate")
    compile_wall_started = time.monotonic()
    if role == "reference":
        if not hasattr(module, "Model") or not hasattr(module, "get_inputs"):
            raise NotImplementedError("reference requires Model and get_inputs")
        init_args = module.get_init_inputs() if hasattr(module, "get_init_inputs") else []
        model_type = module.Model
        baseline = model_type(*copy.deepcopy(init_args)).cuda()
        if config.mode == "eval":
            baseline.eval()
        initial_state = to_arrays(torch, baseline.state_dict())
        init_tree = to_arrays(torch, init_args)
        save_tree(output / "initial_state.npz", initial_state)
        save_tree(output / "init_args.npz", init_tree)
        inputs = []
        for trial in range(config.correctness_runs):
            seed_all(torch, np, config.seed + trial)
            raw = module.get_inputs()
            if not isinstance(raw, (list, tuple)):
                raise NotImplementedError("get_inputs must return positional list/tuple inputs")
            tree = place_inputs(to_arrays(torch, raw), config.input_device)
            save_tree(output / f"inputs-{trial}.npz", tree)
            inputs.append(tree)
        training = baseline.training
        modes = {name: child.training for name, child in baseline.named_modules()}
    else:
        if not hasattr(module, "ModelNew"):
            raise NotImplementedError("candidate must define ModelNew")
        bundle = Path(request["input_dir"])
        initial_state = load_tree(bundle / "initial_state.npz")
        init_tree = load_tree(bundle / "init_args.npz")
        inputs = [
            load_tree(bundle / f"inputs-{trial}.npz") for trial in range(config.correctness_runs)
        ]
        model_type = module.ModelNew
        training = request["training"]
        modes = request["module_training"]

    def fresh(trial):
        seed_all(torch, np, config.seed + trial)
        args = from_arrays(torch, init_tree)
        model = model_type(*args).cuda()
        if config.mode == "eval":
            model.eval()
        elif {name: child.training for name, child in model.named_modules()} != modes:
            raise NotImplementedError(
                "candidate module modes differ from preserved reference modes"
            )
        state = from_arrays(torch, initial_state)
        # Exact state keys/shapes are a declared supported contract, not a silent repair.
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise NotImplementedError(
                "candidate parameter/buffer state contract differs"
            ) from error
        args = from_arrays(torch, inputs[trial])
        # Constructor RNG consumption may differ in an optimized candidate.
        seed_all(torch, np, config.seed + trial)
        return model, args

    for trial in range(config.correctness_runs):
        model, args = fresh(trial)
        with torch.no_grad():
            actual = model(*args)
        torch.cuda.synchronize()
        reject_cross_tree_aliases(torch, args, actual, model.state_dict())
        # Cross-tree alias cases were rejected before independent serialization.
        save_tree(output / f"outputs-{trial}.npz", to_arrays(torch, actual))
        save_tree(output / f"mutated-inputs-{trial}.npz", to_arrays(torch, args))
        save_tree(output / f"post-state-{trial}.npz", to_arrays(torch, model.state_dict()))

    model, _ = fresh(0)

    def reset_timing_inputs():
        seed_all(torch, np, config.seed)
        model.load_state_dict(from_arrays(torch, initial_state), strict=True)
        return from_arrays(torch, inputs[0])

    # Compile and initialize this instance before event timing. Reset mutable
    # tensor state and inputs between every forward, including warmups.
    with torch.no_grad():
        model(*reset_timing_inputs())
    torch.cuda.synchronize()
    samples = []
    for iteration in range(config.warmup_runs + config.timed_runs):
        args = reset_timing_inputs()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            start.record()
            model(*args)
            torch.cuda.synchronize()
            end.record()
        torch.cuda.synchronize()
        if iteration >= config.warmup_runs:
            samples.append(float(start.elapsed_time(end)))
    return {
        "status": "executed",
        "role": role,
        "source_sha256": source_hash,
        "training": training,
        "module_training": modes,
        "device_time_ms": samples,
        "median_ms": statistics.median(samples),
        "worker_compute_wall_s": time.monotonic() - compile_wall_started,
        "cute_verified": None,
    }


def main():
    request = json.load(sys.stdin)
    try:
        from contextlib import redirect_stdout

        with redirect_stdout(sys.stderr):
            result = run(request)
    except SyntaxError as error:
        result = {"status": "compile_error", "detail": str(error)}
    except NotImplementedError as error:
        result = {"status": "unsupported", "detail": str(error)}
    except Exception as error:
        result = {"status": "runtime_error", "detail": repr(error)}
    json.dump(result, sys.stdout, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
