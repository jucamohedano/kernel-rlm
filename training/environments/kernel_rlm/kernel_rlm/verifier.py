"""Fresh-process CUDA compile, correctness, and benchmark verifier."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

_RUNNER = textwrap.dedent(
    """
    import importlib.util
    import json
    import statistics
    import sys

    import torch
    import torch.utils.cpp_extension

    def load_module(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def move_to_device(value, device):
        return value.to(device) if torch.is_tensor(value) else value

    def benchmark(function, warmup=10, repetitions=30):
        for _ in range(warmup):
            function()
        torch.cuda.synchronize()
        timings = []
        for _ in range(repetitions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            function()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
        return statistics.median(timings)

    reference_path, submission_path = sys.argv[1:3]
    trials = int(sys.argv[3])
    atol = float(sys.argv[4])
    rtol = float(sys.argv[5])
    benchmark_enabled = sys.argv[6] == "1"
    result = {
        "compiled": False,
        "extension_used": False,
        "correct": False,
        "max_diff": None,
        "speedup": None,
        "ref_ms": None,
        "kernel_ms": None,
        "error": "",
        "trials": trials,
        "device": "cuda",
    }

    if not torch.cuda.is_available():
        result["error"] = "CUDA is required for kernel_rlm verification"
        print(json.dumps(result))
        raise SystemExit(0)

    extension_builds = []
    extension_calls = []

    class TrackedExtension:
        def __init__(self, module):
            self._module = module

        def __getattr__(self, name):
            attribute = getattr(self._module, name)
            if not callable(attribute):
                return attribute

            def call(*args, **kwargs):
                extension_calls.append(name)
                return attribute(*args, **kwargs)

            return call

    original_load_inline = torch.utils.cpp_extension.load_inline

    def tracked_load_inline(*args, **kwargs):
        module = original_load_inline(*args, **kwargs)
        extension_builds.append(getattr(module, "__name__", "anonymous"))
        return TrackedExtension(module)

    torch.utils.cpp_extension.load_inline = tracked_load_inline

    try:
        reference = load_module(reference_path, "kernel_rlm_reference")
        submission = load_module(submission_path, "kernel_rlm_submission")
        if not isinstance(getattr(submission, "CPP_SOURCE", None), str):
            raise ValueError("submission must define CPP_SOURCE")
        if not isinstance(getattr(submission, "CUDA_SOURCE", None), str):
            raise ValueError("submission must define CUDA_SOURCE")
        cuda_forward = submission.cuda_forward
        model_class = reference.Model
        get_inputs = reference.get_inputs
        get_init_inputs = getattr(reference, "get_init_inputs", lambda: [])
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        print(json.dumps(result))
        raise SystemExit(0)

    device = torch.device("cuda")
    max_diff = 0.0
    correct = True
    try:
        for _ in range(trials):
            model = model_class(*get_init_inputs()).to(device)
            model.eval()
            inputs = [move_to_device(value, device) for value in get_inputs()]
            calls_before = len(extension_calls)
            with torch.no_grad():
                reference_output = model(*inputs)
                candidate_output = cuda_forward(*inputs)
            if not extension_builds:
                raise ValueError(
                    "submission must build a CUDA extension with torch.utils.cpp_extension.load_inline"
                )
            if len(extension_calls) == calls_before:
                raise ValueError("cuda_forward did not invoke the compiled CUDA extension")
            if tuple(reference_output.shape) != tuple(candidate_output.shape):
                result["error"] = (
                    f"shape mismatch: reference {tuple(reference_output.shape)} "
                    f"vs candidate {tuple(candidate_output.shape)}"
                )
                correct = False
                break
            difference = (reference_output.float() - candidate_output.float()).abs().max().item()
            max_diff = max(max_diff, float(difference))
            if not torch.allclose(reference_output, candidate_output, atol=atol, rtol=rtol):
                correct = False
                break
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        correct = False

    result["compiled"] = bool(extension_builds)
    result["extension_used"] = bool(correct and extension_calls)
    result["correct"] = correct
    result["max_diff"] = max_diff

    if correct and benchmark_enabled:
        try:
            inputs = [move_to_device(value, device) for value in get_inputs()]
            model = model_class(*get_init_inputs()).to(device).eval()
            reference_ms = benchmark(lambda: model(*inputs))
            candidate_ms = benchmark(lambda: cuda_forward(*inputs))
            result["ref_ms"] = reference_ms
            result["kernel_ms"] = candidate_ms
            result["speedup"] = reference_ms / max(candidate_ms, 1e-9)
        except Exception as error:
            result["error"] = f"benchmark failed: {type(error).__name__}: {error}"

    print(json.dumps(result))
    """
)


def verify_cuda(
    reference_code: str,
    submission_code: str,
    *,
    n_trials: int = 3,
    atol: float = 1e-3,
    rtol: float = 1e-4,
    timeout_s: float = 240.0,
    python: str | None = None,
    cwd: str | None = None,
    benchmark: bool = True,
) -> dict[str, Any]:
    """Compile, run, and benchmark a CUDA submission in a subprocess."""
    python = python or os.environ.get("KERNEL_VERIFY_PYTHON") or sys.executable
    failure = {
        "compiled": False,
        "extension_used": False,
        "correct": False,
        "max_diff": None,
        "speedup": None,
        "ref_ms": None,
        "kernel_ms": None,
        "error": "",
        "trials": n_trials,
        "device": "?",
    }

    with tempfile.TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)
        reference_path = directory / "reference.py"
        submission_path = directory / "submission.py"
        runner_path = directory / "runner.py"
        reference_path.write_text(reference_code, encoding="utf-8")
        submission_path.write_text(submission_code, encoding="utf-8")
        runner_path.write_text(_RUNNER, encoding="utf-8")
        command = [
            python,
            str(runner_path),
            str(reference_path),
            str(submission_path),
            str(n_trials),
            repr(float(atol)),
            repr(float(rtol)),
            "1" if benchmark else "0",
        ]
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=cwd,
            )
        except subprocess.TimeoutExpired:
            failure["error"] = f"verification timed out after {timeout_s}s"
            return failure
        if process.returncode != 0:
            failure["error"] = (
                f"verifier crashed (rc={process.returncode}): "
                f"{(process.stderr or '')[-1500:]}"
            )
            return failure
        try:
            return json.loads(process.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            failure["error"] = f"unparseable verifier output: {(process.stdout or '')[-500:]}"
            return failure
