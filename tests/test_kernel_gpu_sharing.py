import ast
import asyncio
import fcntl
from pathlib import Path

import pytest


@pytest.fixture
def sharing(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "training/environments/kernel_rlm"))
    from kernel_rlm import gpu_sharing

    monkeypatch.setenv("KERNEL_INFERENCE_ADMIN_URL", "http://127.0.0.1:8100")
    monkeypatch.setattr(gpu_sharing, "LOCK_PATH", tmp_path / "slot")
    return gpu_sharing


def test_resume_after_verifier_failure(sharing, monkeypatch):
    calls = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def post(request, timeout):
        calls.append(request.full_url)
        assert timeout == 10
        return Response()

    monkeypatch.setattr(sharing.urllib.request, "urlopen", post)
    with pytest.raises(ValueError):
        with sharing.paused_inference(30) as remaining:
            assert 0 < remaining <= 30
            raise ValueError("CUDA failure")
    assert calls == ["http://127.0.0.1:8100/pause", "http://127.0.0.1:8100/resume"]


def test_control_failure_does_not_run_verifier(sharing, monkeypatch):
    calls = []

    def post(request, timeout):
        calls.append(request.full_url)
        raise OSError("Control endpoint unavailable")

    monkeypatch.setattr(sharing.urllib.request, "urlopen", post)
    with pytest.raises(OSError):
        with sharing.paused_inference(30):
            pytest.fail("Unsafe verification started")
    assert len(calls) == 2


def test_update_waits_without_blocking_event_loop(sharing):
    asyncio.run(check_update_waits(sharing))


async def check_update_waits(sharing):
    called = []

    @sharing.exclusive_weight_update
    async def update():
        called.append(True)
        raise ValueError("Update failed")

    with sharing.LOCK_PATH.open("a") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX)
        task = asyncio.create_task(update())
        await asyncio.sleep(0.06)
        assert not called
        fcntl.flock(holder, fcntl.LOCK_UN)
        with pytest.raises(ValueError):
            await asyncio.wait_for(task, 1)
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert called == [True]


def test_cancel_waiter_releases_descriptor(sharing):
    asyncio.run(check_cancel_waiter(sharing))


async def check_cancel_waiter(sharing):
    @sharing.exclusive_weight_update
    async def update():
        pytest.fail("Update must not run")

    with sharing.LOCK_PATH.open("a") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX)
        task = asyncio.create_task(update())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def test_prime_patch_is_syntax_valid_and_idempotent(tmp_path):
    source_root = Path("/tmp/prime-rl-v090/src/prime_rl")
    if not source_root.exists():
        pytest.skip("Pinned Prime source not downloaded")
    clients = tmp_path / "clients.py"
    model = tmp_path / "model.py"
    clients.write_text((source_root / "orchestrator/clients.py").read_text())
    model.write_text((source_root / "trainer/model.py").read_text())
    launcher = (
        Path(__file__).parents[1] / "training/environments/kernel_rlm/modal_app/grpo_live_smoke.py"
    )
    function = next(
        node
        for node in ast.parse(launcher.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "patch_prime_laguna_config"
    )
    namespace = {"Path": lambda path: clients if path.endswith("clients.py") else model}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(launcher), "exec"), namespace)
    namespace[function.name]()
    first = clients.read_text()
    namespace[function.name]()
    assert first == clients.read_text()
    compile(first, "clients.py", "exec")
    assert first.count("@exclusive_weight_update") == 2
