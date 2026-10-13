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

    def run(command, **kwargs):
        assert 0 < kwargs["timeout"] <= 5
        with pytest.raises(TimeoutError):
            with original_slot(0.01, path=tmp_path / "slot"):
                pytest.fail("Subprocess must retain the slot")
        return SimpleNamespace(returncode=0, stdout='{"correct": true}', stderr="")

    monkeypatch.setattr(verifier.subprocess, "run", run)
    assert verifier.verify_cuda("", "", timeout_s=5)["correct"]


def test_queue_timeout_never_starts_cuda(slot_module, monkeypatch, tmp_path):
    from kernel_rlm import verifier

    path = tmp_path / "slot"
    monkeypatch.setattr(
        verifier, "verifier_slot", lambda timeout: slot_module.verifier_slot(timeout, path=path)
    )
    monkeypatch.setattr(
        verifier.subprocess, "run", lambda *args, **kwargs: pytest.fail("CUDA started")
    )
    with slot_module.verifier_slot(1, path=path):
        result = verifier.verify_cuda("", "", timeout_s=0.01)
    assert not result["correct"]
    assert "queue + execution" in result["error"]
