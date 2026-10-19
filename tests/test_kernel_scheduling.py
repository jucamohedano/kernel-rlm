import asyncio
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).parents[1] / "training/environments/kernel_rlm"


@pytest.fixture(autouse=True)
def imports(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    from kernel_trl import live_curriculum, live_driver
    from kernel_trl.inference_lifecycle import WarmInference

    process = SimpleNamespace(pid=12345, returncode=None, wait=AsyncMock())
    launcher = AsyncMock(return_value=process)
    stopper = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", launcher)
    monkeypatch.setattr(live_driver, "stop_group", stopper)
    monkeypatch.setattr(live_curriculum, "server_command", lambda *args: ["mock-inference"])
    monkeypatch.setattr(live_curriculum, "inference_environment", lambda *args: {})
    monkeypatch.setattr(live_curriculum, "wait_for_server", AsyncMock())
    arguments = SimpleNamespace(output_dir=tmp_path, inference_python="mock-python")
    lease = WarmInference(
        arguments,
        {
            "budget_usd": 60,
            "displayed_pair_rate_usd_per_hour": 1.26,
            "goal_started_unix_seconds": time.time() - 60,
        },
    )
    lease.control = AsyncMock()
    return lease, launcher, stopper


def test_same_policy_sleep_wake_reuses_one_process(runtime, tmp_path):
    lease, launcher, stopper = runtime

    async def check():
        original = await lease.ensure(0, None, tmp_path)
        await lease.sleep()
        assert lease.sleeping
        assert await lease.ensure(0, None, tmp_path) is original
        assert not lease.sleeping
        assert launcher.await_count == 1
        assert lease.control.await_args_list[0].args == ("sleep",)
        assert lease.control.await_args_list[1].args == ("wake_up",)
        await lease.close()
        stopper.assert_awaited_once_with(original.pid)
        assert lease.process is None

    asyncio.run(check())


def test_policy_change_closes_old_process_before_starting_new_one(runtime, tmp_path):
    lease, launcher, stopper = runtime
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text('{"step":1}')

    async def check():
        await lease.ensure(0, None, tmp_path)
        await lease.ensure(1, checkpoint, tmp_path)
        assert launcher.await_count == 2
        assert stopper.await_count == 1
        await lease.close()

    asyncio.run(check())


def test_failed_sleep_closes_engine_instead_of_verifying_beside_it(runtime, tmp_path):
    lease, launcher, stopper = runtime

    async def check():
        await lease.ensure(0, None, tmp_path)
        lease.control.side_effect = RuntimeError("sleep failed")
        with pytest.raises(RuntimeError, match="sleep failed"):
            await lease.sleep()
        assert lease.process is None
        stopper.assert_awaited_once()

    asyncio.run(check())


def test_compile_only_never_takes_gpu_or_pauses_inference(monkeypatch):
    from kernel_rlm import verifier

    monkeypatch.setenv("KERNEL_INFERENCE_ADMIN_URL", "http://127.0.0.1:1")
    monkeypatch.setattr(
        verifier,
        "run_contract_preflight",
        lambda *args, **kwargs: {"status": "interface_passed", "elapsed_seconds": 0.0},
    )
    monkeypatch.setattr(verifier, "inference_drain", lambda *args: pytest.fail("drain"))
    monkeypatch.setattr(verifier, "verifier_slot", lambda *args: pytest.fail("GPU"))

    def compile(command, **kwargs):
        assert command[-1] == "compile"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        return SimpleNamespace(returncode=0, stdout='{"compiled":true}', stderr="")

    monkeypatch.setattr(verifier, "_run_verifier_process", compile)
    result = verifier.verify_cuda("reference", "candidate", compile_only=True, benchmark=False)
    assert result["verification_status"] == "compile_completed"
    assert result["correct"] is None
    assert result["extension_used"] is None


def test_precompiler_is_bounded_and_does_not_block_other_rollouts(tmp_path):
    from kernel_trl.precompile import FinalPrecompiler

    builder = FinalPrecompiler(concurrency=1)
    active, peak = [0], [0]

    async def prepare(directory):
        async with builder.slots:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            await asyncio.sleep(0.01)
            active[0] -= 1

    builder.prepare = prepare

    async def check():
        for index in range(8):
            builder.submit({}, {"submission": "source"}, tmp_path / str(index))
        await asyncio.sleep(0)
        assert not all(task.done() for task in builder.tasks)
        await builder.finish()
        assert peak == [1]
        await builder.close()

    asyncio.run(check())


def test_precompile_worker_refuses_visible_gpus(monkeypatch):
    import sys

    from kernel_trl import precompile

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(sys, "argv", ["precompile", "--episode", "/unused"])
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        precompile.main()


def test_cancelling_precompile_reaps_worker_and_compiler(tmp_path, monkeypatch):
    from kernel_trl.precompile import FinalPrecompiler

    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.setenv("WANDB_API_KEY", "fixture-secret-must-not-reach-builder")
    compiler_pid = tmp_path / "compiler.pid"
    child = f"import os,time; from pathlib import Path; Path({str(compiler_pid)!r}).write_text(str(os.getpid())); time.sleep(60)"
    worker = (
        "import os,signal,sys; "
        "from kernel_trl.precompile import cancelled; "
        "from kernel_rlm.verifier import _run_verifier_process; "
        "assert os.environ['CUDA_VISIBLE_DEVICES']==''; "
        "assert 'WANDB_API_KEY' not in os.environ; "
        "signal.signal(signal.SIGTERM,cancelled); "
        f"_run_verifier_process([sys.executable,'-c',{child!r}],timeout_s=60,cwd=None,env=None)"
    )
    create = asyncio.create_subprocess_exec

    async def launch(*args, **kwargs):
        return await create(sys.executable, "-c", worker, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    builder = FinalPrecompiler()

    async def check():
        builder.submit({}, {"submission": "source"}, tmp_path)
        try:
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if compiler_pid.exists():
                    break
                await asyncio.sleep(0.01)
            assert compiler_pid.exists()
        finally:
            await builder.close()
        with pytest.raises(ProcessLookupError):
            os.kill(int(compiler_pid.read_text()), 0)

    asyncio.run(check())
