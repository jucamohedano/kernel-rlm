"""One correctness-only GPU phase, invoked through the explicit sandbox launcher.

Reference and candidate regenerate the same seeded cases in separate processes.
Compile observations are not proof that CuTe performs the required computation.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from contextlib import contextmanager
from pathlib import Path

from cute_rlm.verify_worker import (
    InitializationError,
    initialization,
    load_module,
    prepare_output_directory,
    reject_cross_tree_aliases,
    seed_all,
    to_arrays,
)

CASE_SEED = 20261101


class WorkerInfrastructureError(RuntimeError):
    """Trusted worker setup, input construction, or persistence failed."""


@contextmanager
def trusted_worker_stage():
    try:
        yield
    except Exception as error:
        raise WorkerInfrastructureError(str(error)) from error


def input_contract(task):
    return {
        "task_id": task["id"],
        "cases": [
            {"case": case, "seed": case.get("seed", CASE_SEED + i)}
            for i, case in enumerate(task["cases"])
        ],
        "input_generation": "trusted build_inputs; seed reset before each case in each process",
    }


def check_model(model, task):
    if not callable(model) or not callable(getattr(model, "forward", None)):
        raise NotImplementedError("candidate must define callable ModelNew.forward")
    signature = inspect.signature(model.forward)
    parameters = list(signature.parameters.values())
    if any(p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in parameters):
        raise NotImplementedError("forward must have an explicit positional signature")
    # The prompt interface is frozen; binding is also checked against every real case.
    interface = task["interface"]
    if isinstance(interface, str):
        import ast

        declaration = interface.strip()
        if declaration.startswith("def "):
            declaration = declaration.removesuffix(":") + ":\n    pass"
        elif declaration.startswith("forward("):
            declaration = "def " + declaration.removesuffix(":") + ":\n    pass"
        else:
            raise ValueError("task interface must be an explicit forward signature")
        expected = ast.parse(declaration).body[0].args
        names = [p.arg for p in [*expected.posonlyargs, *expected.args] if p.arg != "self"]
    else:
        names = interface["forward_args"]
    if [p.name for p in parameters] != names:
        raise NotImplementedError("candidate forward signature differs from task interface")


def cuda_inputs(torch, inputs):
    """Copy tensor values without silently making gapped/transposed cases contiguous."""
    result = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved = torch.empty_strided(
                value.shape, value.stride(), dtype=value.dtype, device="cuda"
            )
            moved.copy_(value)
            result.append(moved)
        else:
            result.append(value)
    return result


def run(request):
    with trusted_worker_stage():
        import numpy as np
        import torch

        from cute_rlm.icl.inputs import build_inputs
        from cute_rlm.outputs import save_tree

    role = request["role"]
    if role not in {"reference", "candidate"}:
        raise WorkerInfrastructureError("invalid worker phase")
    source = request["source"]
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    if source_hash != request["source_sha256"]:
        raise WorkerInfrastructureError("worker source hash mismatch")
    task = request["icl_task"]
    contract = input_contract(task)
    with trusted_worker_stage():
        output = prepare_output_directory(request["output_dir"])
        if role == "candidate":
            provided = json.loads(
                (Path(request["input_dir"]) / "test-input-contract.json").read_text()
            )
            if provided != contract:
                raise WorkerInfrastructureError("candidate input contract mismatch")
        else:
            (output / "test-input-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    trace = None
    provenance = None
    try:
        if role == "candidate" and request.get("profile_compile"):
            with trusted_worker_stage():
                import cutlass.cute as cute

                from cute_rlm.compile_trace import install_compile_trace

                trace = install_compile_trace(cute, torch, output)
        model = None
        if role == "candidate":
            with trusted_worker_stage():
                seed_all(torch, np, CASE_SEED)
            module = load_module(source, "icl_candidate")
            with initialization():
                if not hasattr(module, "ModelNew"):
                    raise NotImplementedError("candidate must define ModelNew")
                model = module.ModelNew()
                if not isinstance(model, torch.nn.Module):
                    raise NotImplementedError("ModelNew must be a torch.nn.Module")
                model = model.cuda().eval()
                check_model(model, task)
                if model.state_dict():
                    raise NotImplementedError("ICL tasks require stateless ModelNew")
        for i, entry in enumerate(contract["cases"]):
            with trusted_worker_stage():
                seed_all(torch, np, entry["seed"])
                inputs = build_inputs(task, entry["case"], torch)
                if not isinstance(inputs, (list, tuple)):
                    raise TypeError("build_inputs must return positional inputs")
                inputs = cuda_inputs(torch, inputs)
            with torch.no_grad():
                if role == "reference":
                    with trusted_worker_stage():
                        from cute_rlm.icl.tasks import expected_output

                        value = expected_output(task, inputs, torch)
                else:
                    inspect.signature(model.forward).bind(*inputs)
                    value = model(*inputs)
            torch.cuda.synchronize()
            state = model.state_dict() if model is not None else {}
            reject_cross_tree_aliases(torch, inputs, value, state)
            # Output contiguity is checked by to_arrays, never silently repaired.
            name = "expected" if role == "reference" else "outputs"
            arrays = to_arrays(torch, value)
            with trusted_worker_stage():
                save_tree(output / f"{name}-{i}.npz", arrays)
    finally:
        if trace is not None:
            with trusted_worker_stage():
                try:
                    provenance = trace.snapshot()
                finally:
                    trace.close()
    return {
        "status": "executed",
        "role": role,
        "source_sha256": source_hash,
        "case_count": len(contract["cases"]),
        "compile_forward": role == "candidate",
        "compile_provenance": provenance,
        "valid_solution": None,
        "source_review_required": True,
    }


def main():
    import io
    import traceback
    from contextlib import redirect_stderr, redirect_stdout

    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            request = json.load(sys.stdin)
            result = run(request)
        except Exception as error:
            from cute_rlm.compile_trace import CompileTracePersistenceError

            if isinstance(error, (WorkerInfrastructureError, CompileTracePersistenceError)):
                status = "infrastructure"
            elif isinstance(error, SyntaxError):
                status = "compile_error"
            elif isinstance(error, NotImplementedError):
                status = "unsupported"
            elif isinstance(error, InitializationError):
                status = "initialization_error"
            else:
                status = "runtime_error"
            result = {
                "status": status,
                "detail": str(error),
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc(),
            }
    result.update(stdout=stdout.getvalue(), stderr=stderr.getvalue())
    json.dump(result, sys.stdout, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
