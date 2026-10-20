"""Fresh-process CUDA compile, correctness, and benchmark verifier."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from kernel_rlm.feedback_overlap import RESERVE_BYTES, overlap_budget
from kernel_rlm.gpu_sharing import paused_inference
from kernel_rlm.verification_state import VERIFICATION_PROTOCOL
from kernel_rlm.verifier_slot import (
    COMPILE_LOCK_PATH,
    compile_slot,
    feedback_slot,
    inference_drain,
    verifier_slot,
)


def submission_extension_root(submission_code: str) -> Path:
    root = Path(
        os.environ.get(
            "TORCH_EXTENSIONS_DIR", str(Path(tempfile.gettempdir()) / "torch_extensions")
        )
    )
    digest = hashlib.sha256(submission_code.encode()).hexdigest()
    return root / "kernel_rlm_candidates" / digest


def submission_compile_lock(submission_code: str) -> Path:
    digest = hashlib.sha256(submission_code.encode()).hexdigest()
    return COMPILE_LOCK_PATH.with_name(f"{COMPILE_LOCK_PATH.name}-{digest}")


def candidate_fault_boundary(returncode: int, phase: dict[str, Any]) -> bool:
    return (
        returncode in {-signal.SIGSEGV, -signal.SIGBUS, -signal.SIGFPE}
        and phase.get("boundary_version") == 1
        and phase.get("owner") == "candidate"
        and phase.get("reference_completed") is True
        and phase.get("phase")
        in {
            "candidate_setup",
            "candidate_forward",
            "candidate_extension",
            "candidate_synchronize",
            "candidate_benchmark",
        }
    )


_DEVICE_HEALTH_RUNNER = textwrap.dedent(
    """
    import json
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable during independent device health check")
    values = torch.arange(64, device="cuda", dtype=torch.float32)
    result = values.square() + 1
    torch.cuda.synchronize()
    expected = torch.arange(64, dtype=torch.float32).square() + 1
    if not torch.equal(result.cpu(), expected):
        raise RuntimeError("Independent CUDA arithmetic check failed")
    print(json.dumps({"status": "healthy", "device": torch.cuda.get_device_name()}))
    """
)


def check_device_health(
    python: str, *, timeout_s: float, cwd: str | None, env: dict[str, str]
) -> dict[str, Any]:
    if timeout_s <= 0:
        return {"status": "unknown", "error": "No original verification budget remained"}
    try:
        process = _run_verifier_process(
            [python, "-c", _DEVICE_HEALTH_RUNNER], timeout_s=min(20.0, timeout_s), cwd=cwd, env=env
        )
        result = json.loads(process.stdout.strip().splitlines()[-1])
        if (
            process.returncode
            or not isinstance(result, dict)
            or result.get("status") != "healthy"
            or not isinstance(result.get("device"), str)
            or not result["device"]
        ):
            raise ValueError(
                f"Device check failed (rc={process.returncode}): {(process.stderr or '')[-500:]}"
            )
        return result
    except (
        subprocess.TimeoutExpired,
        OSError,
        IndexError,
        json.JSONDecodeError,
        ValueError,
    ) as error:
        return {"status": "unknown", "error": f"{type(error).__name__}: {error}"}


_RUNNER = textwrap.dedent(
    """
    import importlib.util
    import hashlib
    import json
    import os
    import resource
    import statistics
    import sys
    import time
    from pathlib import Path

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    runner_started = time.monotonic()
    import torch
    import torch.utils.cpp_extension
    from kernel_rlm.candidate import candidate_forward, clone_inputs, compare_outputs
    from kernel_rlm.verification_state import VERIFICATION_PROTOCOL, prepare_batchnorm_state

    status_path = Path(sys.argv[-2])
    extension_root = Path(os.environ.get("KERNEL_RLM_EXTENSION_BUILD_ROOT", "/tmp/kernel-rlm-extensions"))
    guard = json.loads(sys.argv[7]) if len(sys.argv) > 9 else {}
    phase_events = []
    reference_completed = False
    benchmark_active = False

    def mark_phase(phase, owner="harness"):
        event = {
            "phase": phase,
            "owner": owner,
            "boundary_version": 1,
            "reference_completed": reference_completed,
            "elapsed_seconds": time.monotonic() - runner_started,
            "monotonic_seconds": time.monotonic(),
        }
        phase_events.append(event)
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(event))
        temporary.replace(status_path)

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
    mode = sys.argv[-1]
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
        "verification_protocol": VERIFICATION_PROTOCOL,
        "batchnorm_state_cases": [],
    }

    def defer_overlap(reason):
        result.update(compiled=True, correct=None, extension_used=None,
                      verification_status="overlap_deferred", overlap_deferred=reason)
        mark_phase("overlap_deferred")
        print(json.dumps(result))
        raise SystemExit(0)

    if mode == "compile":
        result["device"] = "compile_only"

    if mode == "verify" and not torch.cuda.is_available():
        result["error"] = "CUDA is required for kernel_rlm verification"
        result.update(correct=None, verification_status="infrastructure_error", failure_kind="cuda_unavailable")
        print(json.dumps(result))
        raise SystemExit(0)

    extension_builds = []
    extension_calls = []
    if mode == "verify":
        free_memory, total_memory = torch.cuda.mem_get_info()
        result["free_memory_before_gib"] = free_memory / (1024 ** 3)
        if guard:
            if "A100" not in torch.cuda.get_device_name():
                defer_overlap("unprofiled_gpu")
            if free_memory < guard["allocator_limit_bytes"] + guard["reserve_bytes"]:
                defer_overlap("insufficient_free_memory")
            torch.cuda.set_per_process_memory_fraction(
                guard["allocator_limit_bytes"] / total_memory, device=0
            )
            result["allocator_limit_gib"] = guard["allocator_limit_bytes"] / (1024 ** 3)
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
                if mode == "verify" and not benchmark_active:
                    mark_phase("candidate_extension", owner="candidate")
                return attribute(*args, **kwargs)

            return call

    original_load_inline = torch.utils.cpp_extension.load_inline

    def tracked_load_inline(*args, **kwargs):
        mark_phase("extension_compile", owner=phase_events[-1]["owner"])
        extension_name = kwargs.get("name", args[0] if args else "anonymous")
        build_options = {
            key: value for key, value in kwargs.items() if key not in {"build_directory", "verbose"}
        }
        build_identity = repr((extension_name, args, build_options))
        module_digest = hashlib.sha256(build_identity.encode()).hexdigest()
        build_directory = extension_root / module_digest
        build_directory.mkdir(parents=True, exist_ok=True)
        kwargs["build_directory"] = str(build_directory)
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

        mark_phase("load_reference", owner="reference")
        reference = load_module(reference_path, "kernel_rlm_reference")
        model_class = reference.Model
        get_inputs = reference.get_inputs
        get_init_inputs = getattr(reference, "get_init_inputs", lambda: [])
        mark_phase("load_candidate", owner="candidate")
        submission = load_module(submission_path, "kernel_rlm_submission")
        if not isinstance(getattr(submission, "CPP_SOURCE", None), str):
            raise ValueError("submission must define CPP_SOURCE")
        if not isinstance(getattr(submission, "CUDA_SOURCE", None), str):
            raise ValueError("submission must define CUDA_SOURCE")
    except Exception as error:
        if guard and isinstance(error, torch.OutOfMemoryError):
            defer_overlap("allocator_limit_or_oom")
        result["error"] = f"{type(error).__name__}: {error}"
        if mode == "compile":
            result.update(
                compiled=bool(extension_builds),
                extension_used=None,
                correct=None,
                verification_status="compile_failed",
            )
        else:
            result["verification_status"] = "candidate_error"
            if phase_events[-1]["owner"] != "candidate":
                result.update(correct=None, verification_status="infrastructure_error", failure_kind="reference_loading")
        print(json.dumps(result))
        raise SystemExit(0)

    device = torch.device("cuda")
    max_diff = 0.0
    correct = True
    try:
        for trial in range(trials):
            mark_phase(f"correctness_trial_{trial + 1}")
            reference_completed = False
            mark_phase("reference_setup", owner="reference")
            init_inputs = get_init_inputs()
            model = model_class(*init_inputs).to(device)
            model.eval()
            result["batchnorm_state_cases"].append(
                {"trial": trial, "modules": prepare_batchnorm_state(model, trial)}
            )
            inputs = [move_to_device(value, device) for value in get_inputs()]
            candidate_inputs = clone_inputs(inputs)
            calls_before = len(extension_calls)
            with torch.no_grad():
                mark_phase("reference_forward", owner="reference")
                reference_output = model(*inputs)
                torch.cuda.synchronize()
                reference_completed = True
                mark_phase("candidate_setup", owner="candidate")
                run_candidate = candidate_forward(submission, model, init_inputs, device)
                mark_phase("candidate_forward", owner="candidate")
                candidate_output = run_candidate(*candidate_inputs)
                mark_phase("candidate_synchronize", owner="candidate")
                torch.cuda.synchronize()
            if not extension_builds:
                raise ValueError(
                    "submission must build a CUDA extension with torch.utils.cpp_extension.load_inline"
                )
            if len(extension_calls) == calls_before:
                raise ValueError("candidate forward did not invoke the compiled CUDA extension")
            mark_phase("output_compare")
            correct, difference, error = compare_outputs(reference_output, candidate_output, atol, rtol)
            if correct:
                correct, _, error = compare_outputs(inputs, candidate_inputs, atol, rtol)
                if not correct:
                    error = "input mutation mismatch: " + error
            max_diff = max(max_diff, difference)
            result["error"] = error
            del reference_output, candidate_output, candidate_inputs, run_candidate
            if inputs:
                del inputs
            del model
            if not correct:
                break
    except Exception as error:
        if guard and isinstance(error, torch.OutOfMemoryError):
            defer_overlap("allocator_limit_or_oom")
        result["error"] = f"{type(error).__name__}: {error}"
        if phase_events[-1]["owner"] == "candidate":
            correct = False
            result["verification_status"] = "candidate_error"
        else:
            correct = None
            result.update(verification_status="infrastructure_error", failure_kind="reference_or_comparison")

    result["compiled"] = bool(extension_builds)
    result["extension_used"] = bool(correct and extension_calls)
    result["correct"] = correct
    result["max_diff"] = max_diff

    if correct and benchmark_enabled:
        try:
            reference_completed = False
            mark_phase("benchmark_setup", owner="reference")
            inputs = [move_to_device(value, device) for value in get_inputs()]
            init_inputs = get_init_inputs()
            model = model_class(*init_inputs).to(device).eval()
            result["benchmark_batchnorm_state"] = prepare_batchnorm_state(model, trials - 1)
            with torch.no_grad():
                model(*inputs)
                torch.cuda.synchronize()
                reference_completed = True
                mark_phase("candidate_setup", owner="candidate")
                run_candidate = candidate_forward(submission, model, init_inputs, device)
            with torch.no_grad():
                mark_phase("reference_benchmark", owner="reference")
                reference_ms = benchmark(lambda: model(*inputs))
                mark_phase("candidate_benchmark", owner="candidate")
                benchmark_active = True
                try:
                    candidate_ms = benchmark(lambda: run_candidate(*inputs))
                finally:
                    benchmark_active = False
            result["ref_ms"] = reference_ms
            result["kernel_ms"] = candidate_ms
            result["speedup"] = reference_ms / max(candidate_ms, 1e-9)
        except Exception as error:
            result["error"] = f"benchmark failed: {type(error).__name__}: {error}"
            if phase_events[-1]["owner"] == "candidate":
                result.update(correct=False, verification_status="candidate_error")
            else:
                result.update(correct=None, verification_status="infrastructure_error", failure_kind="reference_benchmark")

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
    allow_feedback_overlap: bool = False,
    compile_only: bool = False,
) -> dict[str, Any]:
    """Preflight interfaces, drain during CPU compilation, then independently verify."""
    if timeout_s <= 0:
        raise ValueError("Verification timeout must be positive")
    if type(n_trials) is not int or n_trials < 1:
        raise ValueError("Verification trials must be a positive integer")
    started = time.monotonic()
    deadline = started + timeout_s
    preflight = run_contract_preflight(
        reference_code, submission_code, python=python, cwd=cwd, timeout_s=min(15, timeout_s / 10)
    )
    if preflight.get("status") == "contract_failed":
        return {
            "compiled": False,
            "extension_used": None,
            "correct": None,
            "max_diff": None,
            "speedup": None,
            "ref_ms": None,
            "kernel_ms": None,
            "verification_status": "contract_failed",
            "error": "CPU contract preflight: " + preflight["error"],
            "preflight": preflight,
            "trials": n_trials,
            "device": "cpu_meta",
            "verification_protocol": VERIFICATION_PROTOCOL,
        }
    shared_gpu = bool(os.environ.get("KERNEL_INFERENCE_ADMIN_URL"))
    memory_limit = None
    if shared_gpu and allow_feedback_overlap and not benchmark:
        memory_limit, _reason = overlap_budget(reference_code, submission_code)
    drain_early = shared_gpu and not benchmark and memory_limit is None and not compile_only
    try:
        with inference_drain(max(0, deadline - time.monotonic())) if drain_early else nullcontext():
            result = verify_cuda_execution(
                reference_code,
                submission_code,
                n_trials=n_trials,
                atol=atol,
                rtol=rtol,
                timeout_s=max(0, deadline - time.monotonic()),
                python=python,
                cwd=cwd,
                benchmark=benchmark,
                collect_phase_events=collect_phase_events,
                allow_feedback_overlap=allow_feedback_overlap,
                drain_intent_held=drain_early,
                compile_only=compile_only,
            )
    except TimeoutError as error:
        result = {
            "compiled": None,
            "extension_used": None,
            "correct": None,
            "verification_status": "timeout",
            "timeout_phase": "inference_drain",
            "error": str(error),
        }
    result["preflight"] = preflight
    result["verification_protocol"] = VERIFICATION_PROTOCOL
    result.setdefault("timings", {}).update(
        preflight_elapsed_seconds=preflight["elapsed_seconds"],
        total_elapsed_seconds=time.monotonic() - started,
    )
    result.setdefault("scheduling", {})["drain_started_before_compile"] = drain_early
    return result


def run_contract_preflight(
    reference_code: str,
    submission_code: str,
    *,
    python: str | None,
    cwd: str | None,
    timeout_s: float,
) -> dict[str, Any]:
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        reference = directory / "reference.py"
        submission = directory / "submission.py"
        reference.write_text(reference_code)
        submission.write_text(submission_code)
        try:
            process = _run_verifier_process(
                [
                    python or os.environ.get("KERNEL_VERIFY_PYTHON") or sys.executable,
                    "-m",
                    "kernel_rlm.contract_preflight",
                    str(reference),
                    str(submission),
                ],
                timeout_s=timeout_s,
                cwd=cwd,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            )
            result = json.loads(process.stdout.strip().splitlines()[-1])
            if process.returncode or result.get("status") not in {
                "contract_failed",
                "interface_passed",
                "unsupported",
            }:
                raise ValueError("Invalid preflight response")
        except (subprocess.TimeoutExpired, IndexError, json.JSONDecodeError, ValueError) as error:
            result = {"status": "unsupported", "error": f"{type(error).__name__}: {error}"}
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def verify_cuda_execution(
    reference_code: str,
    submission_code: str,
    *,
    n_trials: int,
    atol: float,
    rtol: float,
    timeout_s: float,
    python: str | None,
    cwd: str | None,
    benchmark: bool,
    collect_phase_events: bool,
    allow_feedback_overlap: bool,
    drain_intent_held: bool,
    compile_only: bool = False,
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
    verification_deadline = verification_started + timeout_s
    shared_gpu = bool(os.environ.get("KERNEL_INFERENCE_ADMIN_URL"))
    memory_limit = None
    reason = "final_benchmark" if benchmark else "overlap_not_requested"
    if allow_feedback_overlap and not benchmark and shared_gpu:
        memory_limit, reason = overlap_budget(reference_code, submission_code)
    scheduling = {"mode": "not_started", "reason": reason}
    failure["scheduling"] = scheduling

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
        extension_root = submission_extension_root(submission_code)
        command_prefix = [
            python,
            str(runner_path),
            str(reference_path),
            str(submission_path),
            str(n_trials),
            repr(float(atol)),
            repr(float(rtol)),
        ]
        compile_budget = min(90.0, max(0.0, timeout_s * 0.75))
        compiler_lock = submission_compile_lock(submission_code)
        compile_queue_started = time.monotonic()
        compile_command = [
            *command_prefix,
            "0",
            str(compile_status_path),
            "compile",
        ]
        try:
            with compile_slot(compile_budget, path=compiler_lock) as compile_remaining:
                timings["compile_queue_wait_seconds"] = time.monotonic() - compile_queue_started
                compile_started = time.monotonic()
                compile_environment = {
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "",
                    "KERNEL_RLM_EXTENSION_BUILD_ROOT": str(extension_root),
                }
                try:
                    compile_process = _run_verifier_process(
                        compile_command,
                        timeout_s=compile_remaining,
                        cwd=cwd,
                        env=compile_environment,
                    )
                except (subprocess.TimeoutExpired, TimeoutError):
                    shutil.rmtree(extension_root, ignore_errors=True)
                    raise
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
        if compile_only:
            compile_result["verification_status"] = "compile_completed"
            compile_result.update(correct=None, extension_used=None, timings=timings)
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
            for overlap in (True, False) if memory_limit else (False,):
                phase = "verifier_slot_queue"
                status_path.unlink(missing_ok=True)
                remaining_total = timeout_s - (time.monotonic() - verification_started)
                slot = feedback_slot if overlap else verifier_slot
                queue_started = time.monotonic()
                slot_arguments = {"intent_held": True} if drain_intent_held and not overlap else {}
                with slot(remaining_total, **slot_arguments):
                    timings["verifier_queue_wait_seconds"] = (
                        timings.get("verifier_queue_wait_seconds", 0.0)
                        + time.monotonic()
                        - queue_started
                    )
                    phase = "compile_cache_queue"
                    cache_queue_started = time.monotonic()
                    with compile_slot(
                        max(0, verification_deadline - time.monotonic()), path=compiler_lock
                    ):
                        remaining = max(0, verification_deadline - time.monotonic())
                        if remaining <= 0:
                            raise TimeoutError(
                                "Verification budget exhausted while acquiring locks"
                            )
                        timings["compile_cache_queue_wait_seconds"] = (
                            timings.get("compile_cache_queue_wait_seconds", 0.0)
                            + time.monotonic()
                            - cache_queue_started
                        )
                        process = execute_verification(
                            command,
                            status_path,
                            remaining,
                            overlap,
                            memory_limit,
                            extension_root,
                            shared_gpu,
                            cwd,
                            timings,
                            scheduling,
                        )
                phase = "result_parse"
                if overlap and process.returncode == 0:
                    try:
                        guard_result = json.loads(process.stdout.strip().splitlines()[-1])
                    except (IndexError, json.JSONDecodeError):
                        break
                    if guard_result.get("overlap_deferred"):
                        scheduling["fallback_reason"] = guard_result["overlap_deferred"]
                        scheduling["guard_attempt"] = guard_result
                        if not collect_phase_events:
                            guard_result.pop("phase_events", None)
                        continue
                break
        except (subprocess.TimeoutExpired, TimeoutError) as error:
            if phase == "verifier_slot_queue":
                timings["verifier_queue_wait_seconds"] = (
                    timings.get("verifier_queue_wait_seconds", 0.0)
                    + time.monotonic()
                    - queue_started
                )
            elif phase == "compile_cache_queue" and not status_path.exists():
                timings["compile_cache_queue_wait_seconds"] = (
                    timings.get("compile_cache_queue_wait_seconds", 0.0)
                    + time.monotonic()
                    - cache_queue_started
                )
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
            phase_details = _read_phase(status_path)
            health = getattr(process, "candidate_device_health", None)
            attributable = candidate_fault_boundary(process.returncode, phase_details)
            if attributable and health and health.get("status") == "healthy":
                signal_name = signal.Signals(-process.returncode).name
                failure.update(
                    compiled=True,
                    extension_used=None,
                    correct=False,
                    verification_status="candidate_crash",
                    failure_kind="candidate_process_signal",
                    process_returncode=process.returncode,
                    process_signal=signal_name,
                    device_health=health,
                    verifier_phase=phase_details["phase"],
                    verifier_phase_elapsed_seconds=phase_details.get("elapsed_seconds"),
                    timings=timings,
                    error=f"Candidate crashed with {signal_name} during {phase_details['phase']}; "
                    "reference execution completed and an independent fresh-process CUDA check passed. "
                    "Check invalid host/device pointer access or out-of-bounds memory use. "
                    f"{(process.stderr or '')[-1000:]}",
                )
                return failure
            failure.update(
                compiled=True,
                extension_used=None,
                correct=None,
                verification_status="runner_crash",
                process_returncode=process.returncode,
            )
            if health is not None:
                failure["device_health"] = health
            failure["error"] = (
                f"verifier crashed (rc={process.returncode}): {(process.stderr or '')[-1500:]}"
            )
            failure["timings"] = timings
            failure["verifier_phase"] = phase_details.get("phase")
            failure["verifier_phase_elapsed_seconds"] = phase_details.get("elapsed_seconds")
            return failure
        try:
            result = json.loads(process.stdout.strip().splitlines()[-1])
            if not collect_phase_events:
                result.pop("phase_events", None)
            result["timings"] = {**timings, **(result.get("timings") or {})}
            result["scheduling"] = scheduling
            result.setdefault("verification_status", "completed")
            if "OutOfMemoryError:" in str(result.get("error", "")):
                result.update(
                    verification_status="infrastructure_error",
                    failure_kind="verifier_oom",
                    correct=None,
                    max_diff=None,
                )
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


def execute_verification(
    command: list[str],
    status_path: Path,
    remaining: float,
    overlap: bool,
    memory_limit: int | None,
    extension_root: Path,
    shared_gpu: bool,
    cwd: str | None,
    timings: dict[str, float],
    scheduling: dict[str, Any],
) -> subprocess.CompletedProcess:
    scheduling["mode"] = "overlap" if overlap else ("exclusive" if shared_gpu else "dedicated")
    status_path.write_text(json.dumps({"phase": "pause_inference"}))
    pause_started = time.monotonic()
    try:
        access = nullcontext(remaining) if overlap else paused_inference(remaining)
        with access as execution_budget:
            status_path.write_text(json.dumps({"phase": "runner_startup"}))
            runner_started = time.monotonic()
            run_command = command
            if overlap:
                guard = {"allocator_limit_bytes": memory_limit, "reserve_bytes": RESERVE_BYTES}
                run_command = [*command[:-2], json.dumps(guard), *command[-2:]]
            try:
                environment = {
                    **os.environ,
                    **({"CUDA_VISIBLE_DEVICES": "0"} if shared_gpu else {}),
                    "KERNEL_RLM_EXTENSION_BUILD_ROOT": str(extension_root),
                }
                process = _run_verifier_process(
                    run_command,
                    timeout_s=execution_budget,
                    cwd=cwd,
                    env=environment,
                )
                if candidate_fault_boundary(process.returncode, _read_phase(status_path)):
                    health_started = time.monotonic()
                    process.candidate_device_health = check_device_health(
                        run_command[0],
                        timeout_s=execution_budget - (health_started - runner_started),
                        cwd=cwd,
                        env=environment,
                    )
                    timings["post_crash_device_check_seconds"] = time.monotonic() - health_started
                return process
            finally:
                timings["runner_elapsed_seconds"] = (
                    timings.get("runner_elapsed_seconds", 0.0) + time.monotonic() - runner_started
                )
    finally:
        timings["inference_paused_seconds"] = timings.get("inference_paused_seconds", 0.0) + (
            time.monotonic() - pause_started if shared_gpu and not overlap else 0.0
        )


def _read_phase(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _terminate_process_group(process: subprocess.Popen, *, grace_s: float = 5.0) -> tuple[str, str]:
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
    except BaseException:
        _terminate_process_group(process)
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def summarize_verification_error(error: str, *, max_chars: int = 1400) -> str:
    """Keep actionable compiler diagnostics without flooding the model context."""
    lines = [line.strip() for line in error.splitlines() if line.strip()]
    timeout_line = next((line for line in lines if "verification timed out" in line.lower()), None)
    if timeout_line:
        return timeout_line.split("Command:", 1)[0].strip()[:max_chars]
    for index, line in enumerate(lines):
        if "incompatible function arguments" in line.lower():
            signature = []
            for detail in lines[index:]:
                if detail.startswith("Invoked with:"):
                    break
                signature.append(detail)
            return "\n".join(signature)[:max_chars]
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
