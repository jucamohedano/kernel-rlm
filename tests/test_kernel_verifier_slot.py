import asyncio
import fcntl
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def slot_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "training/environments/kernel_rlm"))
    from kernel_rlm import verifier, verifier_slot

    monkeypatch.setattr(
        verifier,
        "run_contract_preflight",
        lambda *args, **kwargs: {"status": "unsupported", "elapsed_seconds": 0.0},
    )

    return verifier_slot


def test_slot_excludes_other_processes_and_releases(slot_module, tmp_path):
    path = tmp_path / "slot"
    command = [
        sys.executable,
        "-c",
        "import fcntl, sys; "
        "handle = open(sys.argv[1], 'a'); "
        "fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)",
        str(path),
    ]
    with slot_module.verifier_slot(5, path=path) as remaining:
        assert 0 < remaining <= 5
        assert subprocess.run(command, capture_output=True).returncode != 0
    assert subprocess.run(command, capture_output=True).returncode == 0


def test_wait_has_bounded_budget(slot_module, tmp_path):
    path = tmp_path / "slot"
    with path.open("a") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX)
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="waiting"):
            with slot_module.verifier_slot(0.05, path=path):
                pytest.fail("Busy slot must not admit a verifier")
        assert time.monotonic() - started < 1


def test_exception_releases_slot(slot_module, tmp_path):
    path = tmp_path / "slot"
    with pytest.raises(ValueError):
        with slot_module.verifier_slot(1, path=path):
            raise ValueError("Verifier failed")
    with slot_module.verifier_slot(1, path=path):
        pass


@pytest.mark.parametrize("collect_phase_events", [False, True])
def test_verifier_uses_remaining_budget(slot_module, monkeypatch, tmp_path, collect_phase_events):
    from kernel_rlm import verifier

    monkeypatch.delenv("KERNEL_INFERENCE_ADMIN_URL", raising=False)
    original_slot = slot_module.verifier_slot
    monkeypatch.setattr(
        verifier, "verifier_slot", lambda timeout: original_slot(timeout, path=tmp_path / "slot")
    )

    def run(command, *, timeout_s, cwd, env):
        assert 0 < timeout_s <= 5
        if command[-1] == "compile":
            Path(command[-2]).write_text('{"phase":"compile_complete"}')
            return SimpleNamespace(
                returncode=0,
                stdout='{"compiled":true,"verification_status":"compile_completed"}',
                stderr="",
            )
        Path(command[-2]).write_text('{"phase":"complete"}')
        with pytest.raises(TimeoutError):
            with original_slot(0.01, path=tmp_path / "slot"):
                pytest.fail("Subprocess must retain the slot")
        return SimpleNamespace(
            returncode=0,
            stdout='{"compiled":true,"extension_used":true,"correct":true,"phase_events":[]}',
            stderr="",
        )

    monkeypatch.setattr(verifier, "_run_verifier_process", run)
    result = verifier.verify_cuda("", "", timeout_s=5, collect_phase_events=collect_phase_events)
    assert result["correct"]
    assert result["verifier_phase"] == "complete"
    assert "runner_elapsed_seconds" in result["timings"]
    assert result["timings"]["inference_paused_seconds"] == 0
    assert ("phase_events" in result) is collect_phase_events


def test_verifier_runner_is_syntactically_valid(slot_module):
    from kernel_rlm.verifier import _RUNNER

    compile(_RUNNER, "verifier-runner", "exec")


def test_verification_error_summary_preserves_compiler_diagnostic(slot_module):
    from kernel_rlm.verifier import summarize_verification_error

    error = "compiler command output\n" + "x" * 4000 + "\nfile.cu:9:2: error: expected a ';'\n"
    summary = summarize_verification_error(error)
    assert len(summary) < 1500
    assert "expected a ';'" in summary


def test_timeout_summary_drops_unhelpful_subprocess_command(slot_module):
    from kernel_rlm.verifier import summarize_verification_error

    summary = summarize_verification_error(
        "verification timed out within 120s (queue + execution): Command: " + "x" * 4000
    )
    assert summary == "verification timed out within 120s (queue + execution):"


def test_argument_error_summary_keeps_required_signature_without_tensor_dump(slot_module):
    from kernel_rlm.verifier import summarize_verification_error

    error = (
        "TypeError: launch(): incompatible function arguments. "
        "The following argument types are supported:\n"
        "    1. (arg0: torch.Tensor, arg1: collections.abc.Sequence[int], "
        "arg2: collections.abc.Sequence[int]) -> torch.Tensor\n\n"
        "Invoked with: tensor([" + "1.234, " * 2000 + "])"
    )
    summary = summarize_verification_error(error)
    assert "arg0: torch.Tensor" in summary
    assert "arg1: collections.abc.Sequence[int]" in summary
    assert "arg2: collections.abc.Sequence[int]" in summary
    assert "Invoked with:" not in summary
    assert "1.234" not in summary
    assert len(summary) <= 1400
    assert len(summarize_verification_error(error, max_chars=80)) <= 80


def test_feedback_bootstrap_compiles_with_raw_diagnostic_sidecar(slot_module, tmp_path):
    from kernel_rlm.feedback import build_local_feedback_bootstrap

    source = build_local_feedback_bootstrap(
        "reference", details_path=str(tmp_path / "errors.jsonl")
    )
    compile(source, "feedback-bootstrap", "exec")
    assert "_kernel_feedback_details_path" in source
    assert "_kernel_feedback_summarize" in source


def test_queue_timeout_never_starts_cuda(slot_module, monkeypatch, tmp_path):
    from kernel_rlm import verifier

    path = tmp_path / "slot"
    monkeypatch.setattr(
        verifier, "verifier_slot", lambda timeout: slot_module.verifier_slot(timeout, path=path)
    )

    def compile_only(command, **_kwargs):
        assert command[-1] == "compile"
        Path(command[-2]).write_text('{"phase":"compile_complete"}')
        return SimpleNamespace(
            returncode=0,
            stdout='{"compiled":true,"verification_status":"compile_completed"}',
            stderr="",
        )

    monkeypatch.setattr(verifier, "_run_verifier_process", compile_only)
    with slot_module.verifier_slot(1, path=path):
        result = verifier.verify_cuda("", "", timeout_s=1)
    assert not result["correct"]
    assert "queue + execution" in result["error"]
    assert result["timeout_phase"] == "verifier_slot_queue"
    assert result["timings"]["verifier_queue_wait_seconds"] > 0


@pytest.mark.parametrize("compile_seconds", [0.0, 54.0, 90.0])
def test_compilation_consumes_the_same_feedback_deadline_as_gpu_admission(
    slot_module, monkeypatch, compile_seconds
):
    from kernel_rlm import verifier

    clock = SimpleNamespace(now=0.0)
    observed = {"phases": [], "queue_budget": None}
    monkeypatch.delenv("KERNEL_INFERENCE_ADMIN_URL", raising=False)
    monkeypatch.setattr(verifier.time, "monotonic", lambda: clock.now)

    @contextmanager
    def compile_gate(timeout, **kwargs):
        yield timeout

    @contextmanager
    def busy_gpu(timeout):
        observed["queue_budget"] = timeout
        clock.now += timeout
        raise TimeoutError("Model readers still active; GPU execution never started")
        yield timeout

    def compile_only(command, **kwargs):
        observed["phases"].append(command[-1])
        assert command[-1] == "compile"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        clock.now += compile_seconds
        return SimpleNamespace(
            returncode=0,
            stdout='{"compiled":true,"verification_status":"compile_completed"}',
            stderr="",
        )

    monkeypatch.setattr(verifier, "compile_slot", compile_gate)
    monkeypatch.setattr(verifier, "verifier_slot", busy_gpu)
    monkeypatch.setattr(verifier, "_run_verifier_process", compile_only)
    result = verifier.verify_cuda("", "", timeout_s=120, benchmark=False)

    assert observed["phases"] == ["compile"]
    assert observed["queue_budget"] == 120 - compile_seconds
    assert result["timeout_phase"] == "verifier_slot_queue"
    assert result["verification_status"] == "timeout"
    assert result["correct"] is None
    assert clock.now == 120


def test_single_inflight_reader_can_outlast_feedback_queue_budget(slot_module, tmp_path):
    path = tmp_path / "slot"

    async def check():
        async with slot_module.inference_request_slot(1, path=path):
            with pytest.raises(TimeoutError, match="verifier GPU slot"):
                async with slot_module.exclusive_inference_slot(0.03, path=path):
                    pytest.fail("The active reader has not finished")
        async with slot_module.exclusive_inference_slot(1, path=path):
            pass

    asyncio.run(check())


def test_runner_timeout_reports_last_verifier_phase(slot_module, monkeypatch, tmp_path):
    from kernel_rlm import verifier

    monkeypatch.setattr(
        verifier,
        "verifier_slot",
        lambda timeout: slot_module.verifier_slot(timeout, path=tmp_path / "slot"),
    )

    def run(command, *, timeout_s, cwd, env):
        if command[-1] == "compile":
            Path(command[-2]).write_text('{"phase":"compile_complete"}')
            return SimpleNamespace(
                returncode=0,
                stdout='{"compiled":true,"verification_status":"compile_completed"}',
                stderr="",
            )
        Path(command[-2]).write_text('{"phase":"correctness_trial_1"}')
        raise subprocess.TimeoutExpired(command, timeout_s)

    monkeypatch.setattr(verifier, "_run_verifier_process", run)
    result = verifier.verify_cuda("", "", timeout_s=5)
    assert not result["correct"]
    assert result["timeout_phase"] == "correctness_trial_1"
    assert result["timings"]["runner_elapsed_seconds"] >= 0


def test_inference_request_waits_outside_vllm_while_verifier_holds_slot(slot_module, tmp_path):
    path = tmp_path / "slot"

    async def request():
        async with slot_module.inference_request_slot(1, path=path) as waited:
            return waited

    async def check():
        with slot_module.verifier_slot(1, path=path):
            pending = asyncio.create_task(request())
            await asyncio.sleep(0.06)
            assert not pending.done()
        return await asyncio.wait_for(pending, 1)

    assert asyncio.run(check()) >= 0.05
