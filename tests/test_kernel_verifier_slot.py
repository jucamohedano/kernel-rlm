import fcntl
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def slot_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "training/environments/kernel_rlm"))
    from kernel_rlm import verifier_slot

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


def test_verifier_uses_remaining_budget(slot_module, monkeypatch, tmp_path):
    from kernel_rlm import verifier

    original_slot = slot_module.verifier_slot
    monkeypatch.setattr(
        verifier, "verifier_slot", lambda timeout: original_slot(timeout, path=tmp_path / "slot")
    )

    def run(command, *, timeout_s, cwd, env):
        assert 0 < timeout_s <= 5
        Path(command[-1]).write_text('{"phase":"complete"}')
        with pytest.raises(TimeoutError):
            with original_slot(0.01, path=tmp_path / "slot"):
                pytest.fail("Subprocess must retain the slot")
        return SimpleNamespace(returncode=0, stdout='{"correct": true}', stderr="")

    monkeypatch.setattr(verifier, "_run_verifier_process", run)
    result = verifier.verify_cuda("", "", timeout_s=5)
    assert result["correct"]
    assert result["verifier_phase"] == "complete"
    assert "runner_elapsed_seconds" in result["timings"]


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
    monkeypatch.setattr(
        verifier, "_run_verifier_process", lambda *args, **kwargs: pytest.fail("CUDA started")
    )
    with slot_module.verifier_slot(1, path=path):
        result = verifier.verify_cuda("", "", timeout_s=0.01)
    assert not result["correct"]
    assert "queue + execution" in result["error"]
    assert result["timeout_phase"] == "verifier_slot_queue"
    assert result["timings"]["verifier_queue_wait_seconds"] > 0


def test_runner_timeout_reports_last_verifier_phase(slot_module, monkeypatch, tmp_path):
    from kernel_rlm import verifier

    monkeypatch.setattr(
        verifier,
        "verifier_slot",
        lambda timeout: slot_module.verifier_slot(timeout, path=tmp_path / "slot"),
    )

    def run(command, *, timeout_s, cwd, env):
        Path(command[-1]).write_text('{"phase":"extension_compile"}')
        raise subprocess.TimeoutExpired(command, timeout_s)

    monkeypatch.setattr(verifier, "_run_verifier_process", run)
    result = verifier.verify_cuda("", "", timeout_s=5)
    assert not result["correct"]
    assert result["timeout_phase"] == "extension_compile"
    assert result["timings"]["runner_elapsed_seconds"] >= 0
