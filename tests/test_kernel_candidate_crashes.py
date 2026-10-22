"""Candidate fault attribution is separate from reference and infrastructure failure."""

import ast
import contextlib
import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def boundary(**changes):
    return {
        "phase": "candidate_extension",
        "owner": "candidate",
        "boundary_version": 1,
        "reference_completed": True,
        **changes,
    }


@pytest.mark.parametrize(
    "phase",
    [
        "candidate_setup",
        "candidate_forward",
        "candidate_extension",
        "candidate_synchronize",
        "candidate_benchmark",
    ],
)
def test_exact_candidate_boundaries_are_recognized(phase):
    from kernel_rlm.verifier import candidate_fault_boundary

    assert candidate_fault_boundary(-signal.SIGSEGV, boundary(phase=phase))


@pytest.mark.parametrize(
    "changes",
    [
        {"phase": "reference_forward"},
        {"phase": "correctness_trial_1"},
        {"phase": "output_compare"},
        {"owner": "reference"},
        {"reference_completed": False},
        {"boundary_version": 0},
    ],
)
def test_legacy_and_ambiguous_markers_never_relabel_an_old_crash(changes):
    from kernel_rlm.verifier import candidate_fault_boundary

    assert not candidate_fault_boundary(-signal.SIGSEGV, boundary(**changes))
    assert not candidate_fault_boundary(-signal.SIGSEGV, {})


@pytest.mark.parametrize(
    "returncode", [0, 1, 139, -signal.SIGKILL, -signal.SIGTERM, -signal.SIGABRT]
)
def test_unknown_signals_and_resource_kills_stay_infrastructure(returncode):
    from kernel_rlm.verifier import candidate_fault_boundary

    assert not candidate_fault_boundary(returncode, boundary())


@pytest.mark.parametrize(
    "health",
    [{"status": "healthy", "device": "fixture"}, {"status": "unknown", "error": "fixture OOM"}],
)
@pytest.mark.parametrize("shared", [False, True])
def test_crash_health_probe_stays_inside_the_gpu_lease(monkeypatch, health, shared, tmp_path):
    from kernel_rlm import verifier
    from kernel_rollout.contract import terminal_reward

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    if shared:
        monkeypatch.setenv("KERNEL_INFERENCE_ADMIN_URL", "http://127.0.0.1:1")
    else:
        monkeypatch.delenv("KERNEL_INFERENCE_ADMIN_URL", raising=False)
    monkeypatch.setattr(
        verifier,
        "run_contract_preflight",
        lambda *args, **kwargs: {"status": "unsupported", "elapsed_seconds": 0},
    )
    lease = []

    @contextlib.contextmanager
    def access(seconds):
        lease.append(True)
        try:
            yield seconds
        finally:
            lease.pop()

    monkeypatch.setattr(verifier, "verifier_slot", access)
    monkeypatch.setattr(verifier, "paused_inference", access)
    seen = []

    def run(command, *, timeout_s, cwd, env):
        seen.append((command, timeout_s))
        if command[-1] == "compile":
            return SimpleNamespace(returncode=0, stdout='{"compiled":true}', stderr="")
        assert lease and timeout_s > 0
        Path(command[-2]).write_text(json.dumps(boundary()))
        return SimpleNamespace(returncode=-signal.SIGSEGV, stdout="", stderr="fixture native crash")

    def healthy(python, *, timeout_s, cwd, env):
        assert len(lease) == 2
        assert 0 < timeout_s < seen[-1][1]
        assert env["CUDA_VISIBLE_DEVICES"] == ("0" if shared else "1")
        return health

    monkeypatch.setattr(verifier, "_run_verifier_process", run)
    monkeypatch.setattr(verifier, "check_device_health", healthy)
    result = verifier.verify_cuda("reference", "submission", timeout_s=5)
    assert not lease
    assert len(seen) == 2
    assert result["process_returncode"] == -signal.SIGSEGV
    assert result["device_health"] == health
    if health["status"] == "healthy":
        assert result["verification_status"] == "candidate_crash"
        assert result["correct"] is False
        assert "SIGSEGV" in result["error"]
        assert terminal_reward(result) == 0.1
    else:
        assert result["verification_status"] == "runner_crash"
        assert result["correct"] is None


@pytest.mark.parametrize(
    "output", ["", "not-json", "[]", '{"status":"healthy"}', '{"status":"healthy","device":""}']
)
def test_malformed_device_probe_is_not_accepted_as_healthy(monkeypatch, output):
    from kernel_rlm import verifier

    monkeypatch.setattr(
        verifier,
        "_run_verifier_process",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output, stderr=""),
    )
    assert (
        verifier.check_device_health(sys.executable, timeout_s=1, cwd=None, env={})["status"]
        == "unknown"
    )


def test_exhausted_budget_never_starts_another_cuda_probe(monkeypatch):
    from kernel_rlm import verifier

    monkeypatch.setattr(
        verifier, "_run_verifier_process", lambda *args, **kwargs: pytest.fail("No budget")
    )
    assert (
        verifier.check_device_health(sys.executable, timeout_s=0, cwd=None, env={})["status"]
        == "unknown"
    )


def test_actual_cpu_signal_is_contained_in_a_fresh_worker(tmp_path):
    from kernel_rlm.verifier import _run_verifier_process, candidate_fault_boundary

    phase = tmp_path / "phase.json"
    code = f"import json,os,resource,signal; from pathlib import Path; resource.setrlimit(resource.RLIMIT_CORE,(0,0)); Path({str(phase)!r}).write_text({json.dumps(boundary())!r}); os.kill(os.getpid(), signal.SIGSEGV)"
    result = _run_verifier_process([sys.executable, "-c", code], timeout_s=5, cwd=None, env=None)
    assert result.returncode == -signal.SIGSEGV
    assert candidate_fault_boundary(result.returncode, json.loads(phase.read_text()))
    assert not list(tmp_path.glob("core*"))


def test_no_phase_logging_is_added_inside_benchmark_timing_loop():
    from kernel_rlm.verifier import _RUNNER

    tree = ast.parse(_RUNNER)
    timing = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "benchmark"
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "mark_phase"
        for node in ast.walk(timing)
    )
    extension = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TrackedExtension"
    )
    guarded = [
        node
        for node in ast.walk(extension)
        if isinstance(node, ast.If) and "not benchmark_active" in ast.unparse(node.test)
    ]
    assert len(guarded) == 1


@pytest.mark.parametrize(
    "error", ["OutOfMemoryError: fixture", "benchmark failed: OutOfMemoryError: fixture"]
)
def test_candidate_oom_is_still_unknown_infrastructure(monkeypatch, error):
    from kernel_rlm import verifier

    monkeypatch.delenv("KERNEL_INFERENCE_ADMIN_URL", raising=False)
    monkeypatch.setattr(
        verifier,
        "run_contract_preflight",
        lambda *args, **kwargs: {"status": "unsupported", "elapsed_seconds": 0},
    )
    monkeypatch.setattr(verifier, "verifier_slot", contextlib.nullcontext)

    def run(command, **kwargs):
        result = {"compiled": True}
        if command[-1] == "verify":
            result.update(correct=False, verification_status="candidate_error", error=error)
        return subprocess.CompletedProcess(command, 0, json.dumps(result), "")

    monkeypatch.setattr(verifier, "_run_verifier_process", run)
    result = verifier.verify_cuda("reference", "submission")
    assert result["verification_status"] == "infrastructure_error"
    assert result["correct"] is None
