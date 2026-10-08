import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SOURCE = (
    Path(__file__).parents[1] / "training/environments/kernel_rlm/modal_app/inference_runtime.py"
)
spec = importlib.util.spec_from_file_location("inference_runtime", SOURCE)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def test_cache_is_stable_and_version_scoped(tmp_path):
    identity = {"torch": "2.11", "vllm": "0.26", "gpu": "sm80"}
    first = runtime.cache_environment(tmp_path, identity)
    assert first == runtime.cache_environment(tmp_path, dict(reversed(list(identity.items()))))
    assert first != runtime.cache_environment(tmp_path, {**identity, "vllm": "0.27"})
    assert Path(first["VLLM_CACHE_ROOT"]).parent.joinpath("identity.json").exists()
    assert "TRITON_CACHE_DIR" not in first


def test_readiness_retries_then_succeeds(monkeypatch):
    health = Mock(side_effect=[OSError("not ready"), None])
    monkeypatch.setattr(runtime, "check_health", health)
    monkeypatch.setattr(runtime.asyncio, "sleep", async_noop)
    elapsed = asyncio.run(runtime.wait_for_inference(Mock(poll=Mock(return_value=None))))
    assert elapsed >= 0
    assert health.call_count == 2


async def async_noop(seconds):
    pass


def test_exited_server_fails_without_polling_health(monkeypatch):
    health = Mock()
    monkeypatch.setattr(runtime, "check_health", health)
    with pytest.raises(RuntimeError, match="server exited"):
        asyncio.run(runtime.wait_for_inference(Mock(poll=Mock(return_value=1))))
    health.assert_not_called()


def test_startup_uses_elapsed_deadline(monkeypatch):
    clock = Mock(side_effect=[0, 599, 599, 600])
    monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic=clock))
    monkeypatch.setattr(runtime, "check_health", Mock(side_effect=OSError("not ready")))
    monkeypatch.setattr(runtime.asyncio, "sleep", async_noop)
    with pytest.raises(TimeoutError, match="600 seconds"):
        asyncio.run(runtime.wait_for_inference(Mock(poll=Mock(return_value=None))))


def test_budget_does_not_increase():
    assert runtime.STARTUP_TIMEOUT_S == 600
    assert runtime.PROBE_TIMEOUT_S == 1200
