"""Fresh-process CUDA compile, correctness, and benchmark verifier."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

from kernel_rlm.gpu_sharing import paused_inference
from kernel_rlm.verifier_slot import compile_slot, verifier_slot


def _run_verifier_process(
    command: list[str], *, timeout_s: float, cwd: str | None, env: dict[str, str] | None
) -> subprocess.CompletedProcess:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as timeout_error:
        stdout, stderr = _terminate_process_group(process)
        raise subprocess.TimeoutExpired(
            command, timeout_s, output=stdout, stderr=stderr
        ) from timeout_error
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

_RUNNER = textwrap.dedent(
    """
    import importlib.util
    import json
    import statistics
    import sys
    import time
    from pathlib import Path

    runner_started = time.monotonic()
    import torch
    import torch.utils.cpp_extension

    status_path = Path(sys.argv[7])
    phase_events = []

    def mark_phase(phase):
        event = {
            "phase": phase,
            "elapsed_seconds": time.monotonic() - runner_started,
            "monotonic_seconds": time.monotonic(),
        }
        phase_events.append(event)
        status_path.write_text(json.dumps(event))

    mark_phase("torch_imported")

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
    mode = sys.argv[8]
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
        "phase_events": phase_events,
    }
    if mode == "compile":
        result["device"] = "compile_only"

    if mode == "verify" and not torch.cuda.is_available():
        result["error"] = "CUDA is required for kernel_rlm verification"
        print(json.dumps(result))
        raise SystemExit(0)

    extension_builds = []
    extension_calls = []
    if mode == "verify":
        result["free_memory_before_gib"] = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        torch.cuda.reset_peak_memory_stats()

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
        mark_phase("extension_compile")
        module = original_load_inline(*args, **kwargs)
        extension_builds.append(getattr(module, "__name__", "anonymous"))
        return TrackedExtension(module)

    torch.utils.cpp_extension.load_inline = tracked_load_inline

    try:
        if mode == "compile":
            mark_phase("load_candidate_for_compile")
            submission = load_module(submission_path, "kernel_rlm_submission")
            if not isinstance(getattr(submission, "CPP_SOURCE", None), str):
                raise ValueError("submission must define CPP_SOURCE")
            if not isinstance(getattr(submission, "CUDA_SOURCE", None), str):
                raise ValueError("submission must define CUDA_SOURCE")
            if not extension_builds:
                raise ValueError("submission must build a CUDA extension with load_inline")
            result.update(
                compiled=True,
                extension_used=None,
                correct=None,
                verification_status="compile_completed",
            )
            mark_phase("compile_complete")
            print(json.dumps(result))
            raise SystemExit(0)

        mark_phase("load_reference_and_submission")
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
        if mode == "compile":
            result.update(
                compiled=bool(extension_builds),
                extension_used=None,
                correct=None,
                verification_status="compile_failed",
            )
        else:
            result["verification_status"] = "runner_error"
        print(json.dumps(result))
        raise SystemExit(0)

    device = torch.device("cuda")
    max_diff = 0.0
    correct = True
    try:
        for trial in range(trials):
            mark_phase(f"correctness_trial_{trial + 1}")
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
            reference_flat = reference_output.reshape(-1)
            candidate_flat = candidate_output.reshape(-1)
            reference_chunk = candidate_chunk = None
            for start in range(0, reference_flat.numel(), 1048576):
                reference_chunk = reference_flat[start:start + 1048576]
                candidate_chunk = candidate_flat[start:start + 1048576]
                difference = (reference_chunk.float() - candidate_chunk.float()).abs().max().item()
                max_diff = max(max_diff, float(difference))
                if not torch.allclose(reference_chunk, candidate_chunk, atol=atol, rtol=rtol):
                    correct = False
            del reference_flat, candidate_flat, reference_output, candidate_output
            del reference_chunk, candidate_chunk
            if inputs:
                del inputs
            del model
            if not correct:
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
            mark_phase("benchmark")
            inputs = [move_to_device(value, device) for value in get_inputs()]
            model = model_class(*get_init_inputs()).to(device).eval()
            reference_ms = benchmark(lambda: model(*inputs))
            candidate_ms = benchmark(lambda: cuda_forward(*inputs))
            result["ref_ms"] = reference_ms
            result["kernel_ms"] = candidate_ms
            result["speedup"] = reference_ms / max(candidate_ms, 1e-9)
        except Exception as error:
            result["error"] = f"benchmark failed: {type(error).__name__}: {error}"

    result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
    mark_phase("complete")
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
    collect_phase_events: bool = False,
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
    timings: dict[str, float] = {}
    phase = "verifier_slot_queue"
    verification_started = time.monotonic()

    with tempfile.TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)
        reference_path = directory / "reference.py"
        submission_path = directory / "submission.py"
        runner_path = directory / "runner.py"
        status_path = directory / "phase.json"
        compile_status_path = directory / "compile-phase.json"
        reference_path.write_text(reference_code, encoding="utf-8")
        submission_path.write_text(submission_code, encoding="utf-8")
        runner_path.write_text(_RUNNER, encoding="utf-8")
        command_prefix = [
            python,
            str(runner_path),
            str(reference_path),
            str(submission_path),
            str(n_trials),
            repr(float(atol)),
            repr(float(rtol)),
        ]
        compile_budget = min(90.0, timeout_s * 0.75)
        compile_queue_started = time.monotonic()
        compile_command = [
            *command_prefix,
            "0",
            str(compile_status_path),
            "compile",
        ]
        try:
            with compile_slot(compile_budget) as compile_remaining:
                timings["compile_queue_wait_seconds"] = (
                    time.monotonic() - compile_queue_started
                )
                compile_started = time.monotonic()
                compile_process = _run_verifier_process(
                    compile_command,
                    timeout_s=compile_remaining,
                    cwd=cwd,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                )
                timings["compile_elapsed_seconds"] = time.monotonic() - compile_started
        except (subprocess.TimeoutExpired, TimeoutError) as error:
            phase_details = _read_phase(compile_status_path)
            failure.update(
                compiled=None,
                extension_used=None,
                correct=None,
                verification_status="timeout",
                timeout_phase=phase_details.get("phase", "extension_compile"),
                timeout_phase_elapsed_seconds=phase_details.get("elapsed_seconds"),
                timings=timings,
                error=f"extension compile exceeded its {compile_budget:g}s budget: {error}",
            )
            return failure
        if compile_process.returncode != 0:
            failure.update(
                compiled=None,
                extension_used=None,
                correct=None,
                verification_status="compile_runner_crash",
                error=f"compile runner crashed (rc={compile_process.returncode}): "
                f"{(compile_process.stderr or '')[-1500:]}",
                timings=timings,
            )
            return failure
        try:
            compile_result = json.loads(compile_process.stdout.strip().splitlines()[-1])
            if not collect_phase_events:
                compile_result.pop("phase_events", None)
        except (IndexError, json.JSONDecodeError):
            failure.update(
                compiled=None,
                extension_used=None,
                correct=None,
                verification_status="invalid_compile_output",
                error=f"unparseable compile output: {(compile_process.stdout or '')[-500:]}",
                timings=timings,
            )
            return failure
        if not compile_result.get("compiled"):
            compile_result["verification_status"] = "compile_failed"
            compile_result["timings"] = timings
            compile_phase = _read_phase(compile_status_path)
            compile_result["verifier_phase"] = compile_phase.get("phase")
            return compile_result
        timings["total_compile_seconds"] = time.monotonic() - verification_started
        remaining_total = timeout_s - (time.monotonic() - verification_started)
        if remaining_total <= 0:
            failure.update(
                compiled=True,
                extension_used=None,
                correct=None,
                verification_status="timeout",
                timeout_phase="gpu_verification_budget",
                timings=timings,
                error="no verifier budget remained after extension compilation",
            )
            return failure
        command = [
            *command_prefix,
            "1" if benchmark else "0",
            str(status_path),
            "verify",
        ]
        queue_started = time.monotonic()
        try:
            with verifier_slot(remaining_total) as remaining:
                timings["verifier_queue_wait_seconds"] = time.monotonic() - queue_started
                phase = "pause_inference"
                pause_started = time.monotonic()
                try:
                    with paused_inference(remaining) as execution_budget:
                        phase = "runner_startup"
                        runner_started = time.monotonic()
                        try:
                            process = _run_verifier_process(
                                command,
                                timeout_s=execution_budget,
                                cwd=cwd,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
                                if os.environ.get("KERNEL_INFERENCE_ADMIN_URL")
                                else None,
                            )
                        finally:
                            timings["runner_elapsed_seconds"] = time.monotonic() - runner_started
                        phase = "result_parse"
                finally:
                    timings["inference_paused_seconds"] = (
                        time.monotonic() - pause_started
                        if os.environ.get("KERNEL_INFERENCE_ADMIN_URL")
                        else 0.0
                    )
        except (subprocess.TimeoutExpired, TimeoutError) as error:
            timings.setdefault("verifier_queue_wait_seconds", time.monotonic() - queue_started)
            phase_details = _read_phase(status_path)
            phase = phase_details.get("phase", phase)
            failure.update(
                compiled=True,
                extension_used=None,
                correct=None,
                verification_status="timeout",
            )
            failure["error"] = (
                f"verification timed out within {timeout_s}s (compile + queue + execution): {error}"
            )
            failure["timeout_phase"] = phase
            failure["timeout_phase_elapsed_seconds"] = phase_details.get("elapsed_seconds")
            failure["timings"] = timings
            return failure
        if process.returncode != 0:
            failure.update(
                compiled=True,
                extension_used=None,
                correct=None,
                verification_status="runner_crash",
            )
            failure["error"] = (
                f"verifier crashed (rc={process.returncode}): {(process.stderr or '')[-1500:]}"
            )
            failure["timings"] = timings
            phase_details = _read_phase(status_path)
            failure["verifier_phase"] = phase_details.get("phase")
            failure["verifier_phase_elapsed_seconds"] = phase_details.get("elapsed_seconds")
            return failure
        try:
            result = json.loads(process.stdout.strip().splitlines()[-1])
            if not collect_phase_events:
                result.pop("phase_events", None)
            result["timings"] = {**timings, **(result.get("timings") or {})}
            result["verification_status"] = "completed"
            phase_details = _read_phase(status_path)
            result["verifier_phase"] = phase_details.get("phase")
            result["verifier_phase_elapsed_seconds"] = phase_details.get("elapsed_seconds")
            return result
        except (IndexError, json.JSONDecodeError):
            failure.update(
                compiled=True,
                extension_used=None,
                correct=None,
                verification_status="invalid_runner_output",
            )
            failure["error"] = f"unparseable verifier output: {(process.stdout or '')[-500:]}"
            failure["timings"] = timings
            phase_details = _read_phase(status_path)
            failure["verifier_phase"] = phase_details.get("phase")
            failure["verifier_phase_elapsed_seconds"] = phase_details.get("elapsed_seconds")
            return failure


def _read_phase(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _terminate_process_group(
    process: subprocess.Popen, *, grace_s: float = 5.0
) -> tuple[str, str]:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.communicate(timeout=grace_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.communicate()


def _run_verifier_process(
    command: list[str], *, timeout_s: float, cwd: str | None, env: dict[str, str] | None
) -> subprocess.CompletedProcess:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as timeout_error:
        stdout, stderr = _terminate_process_group(process)
        raise subprocess.TimeoutExpired(
            command, timeout_s, output=stdout, stderr=stderr
        ) from timeout_error
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def summarize_verification_error(error: str, *, max_chars: int = 1400) -> str:
    """Keep actionable compiler diagnostics without flooding the model context."""
    lines = [line.strip() for line in error.splitlines() if line.strip()]
    timeout_line = next((line for line in lines if "verification timed out" in line.lower()), None)
    if timeout_line:
        return timeout_line.split("Command:", 1)[0].strip()[:max_chars]
    diagnostic_lines = [
        line
        for line in lines
        if any(
            marker in line.lower()
            for marker in ("error:", "fatal error:", "undefined reference", "ninja: build stopped")
        )
    ]
    selected = diagnostic_lines[-5:] if diagnostic_lines else lines[-5:]
    summary = "\n".join(selected)
    return summary[-max_chars:]
