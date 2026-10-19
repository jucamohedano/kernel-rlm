import asyncio
import fcntl
import importlib
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).parents[1] / "training/environments/kernel_rlm"


@pytest.fixture
def collector(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    module = importlib.import_module("kernel_trl.collect")
    from kernel_skyrl.contract import RootSample

    smoke = {
        "prompt_token_ids": [2],
        "choices": [
            {
                "token_ids": [3],
                "logprobs": {"content": [{"token": "token_id:3", "logprob": -1.0}]},
                "finish_reason": "stop",
            }
        ],
    }
    observed = {
        "active": 0,
        "peak": 0,
        "versions": [],
        "adapters": [],
        "generator_kwargs": None,
        "inference_http_timeout": None,
        "client_class": module.InferenceClient,
    }

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class Generator:
        def __init__(self, client, adapter, *args, **kwargs):
            observed["adapters"].append(adapter)
            observed["generator_kwargs"] = kwargs

        async def run_episode(self, task, directory, version):
            observed["active"] += 1
            observed["peak"] = max(observed["active"], observed["peak"])
            observed["versions"].append(version)
            index = int(directory.name.rsplit("-", 1)[-1])
            try:
                if observed.get("episode_hook"):
                    await observed["episode_hook"](index)
                else:
                    await asyncio.sleep(0)
            finally:
                observed["active"] -= 1
            return {
                "reward": observed["rewards"][index],
                "samples": [RootSample([2], [index + 3], [-1.0], "stop")],
                "verification": observed.get("verification", {}).get(index, {}),
            }

    def client_session(**kwargs):
        observed["inference_http_timeout"] = kwargs["timeout"].total
        return Session()

    monkeypatch.setattr(module.aiohttp, "ClientSession", client_session)
    monkeypatch.setattr(
        module,
        "InferenceClient",
        lambda *args: SimpleNamespace(chat_completion=AsyncMock(return_value=smoke)),
    )
    monkeypatch.setattr(
        module.web,
        "AppRunner",
        lambda *args, **kwargs: SimpleNamespace(setup=AsyncMock(), cleanup=AsyncMock()),
    )
    monkeypatch.setattr(module.web, "TCPSite", lambda *args: SimpleNamespace(start=AsyncMock()))
    monkeypatch.setattr(module, "KernelGenerator", Generator)
    monkeypatch.setattr(module, "load_tasks", lambda path: [{"name": "task-a"}])
    return module, observed


def arguments(tmp_path):
    return SimpleNamespace(
        output_dir=tmp_path / "collection",
        endpoint="http://127.0.0.1:8006",
        checkpoint=tmp_path / "model",
        dataset=tmp_path / "dataset",
        adapter_name="kernel-step-7",
        policy_version=7,
        rollouts=8,
        task_ids=["task-a"],
        model_timeout_s=600,
        inference_http_timeout_s=630,
        episode_timeout_s=1500,
        group_timeout_s=2400,
    )


def test_eight_rollouts_use_four_slots_and_one_policy(collector, tmp_path):
    module, observed = collector
    observed["rewards"] = [0.0, 0.3, 0.7, 1.0] * 2
    args = arguments(tmp_path)
    asyncio.run(module.collect(args))
    assert observed["peak"] == 4
    assert observed["versions"] == [7] * 8
    assert observed["adapters"] == ["kernel-step-7"]
    result = json.loads((args.output_dir / "training-group.json").read_text())
    assert len(result["rewards"]) == 8
    assert result["policy_version"] == 7
    assert observed["generator_kwargs"]["limits"]["model_timeout_s"] == 600
    assert observed["generator_kwargs"]["episode_timeout_s"] == 1500
    assert observed["inference_http_timeout"] == 630


def test_transport_does_not_readmit_a_root_when_verifier_is_waiting(
    collector, tmp_path, monkeypatch
):
    module, observed = collector
    from kernel_rlm import verifier_slot

    path = tmp_path / "admission"
    monkeypatch.setattr(verifier_slot, "LOCK_PATH", path)
    monkeypatch.setattr(module, "INFERENCE_ADMISSION_TIMEOUT_S", 0.1)
    client = observed["client_class"](None, "http://unused", tmp_path / "requests")
    client.request_completion = AsyncMock(return_value={"completed": True})

    async def verify():
        async with verifier_slot.exclusive_inference_slot(1, path=path):
            return "verified"

    async def check():
        async with verifier_slot.inference_request_slot(1, path=path):
            pending_verifier = asyncio.create_task(verify())
            await asyncio.sleep(0)
            with path.with_name(path.name + ".writer").open("a") as intent:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(intent, fcntl.LOCK_SH | fcntl.LOCK_NB)
            result = await client.chat_completion({"json": {}})
            assert result == {"completed": True}
            assert not pending_verifier.done()
        assert await pending_verifier == "verified"

    asyncio.run(check())
    client.request_completion.assert_awaited_once()


def test_startup_smoke_owns_its_inference_admission(collector, tmp_path, monkeypatch):
    module, observed = collector
    observed["rewards"] = [0.1, 0.8] * 4
    entries = []

    @asynccontextmanager
    async def slot(timeout):
        entries.append(timeout)
        yield 0.0

    monkeypatch.setattr(module, "inference_request_slot", slot)
    asyncio.run(module.collect(arguments(tmp_path)))
    assert entries == [module.INFERENCE_ADMISSION_TIMEOUT_S]


def test_collection_passes_larger_context_to_generator_and_training(collector, tmp_path):
    module, observed = collector
    observed["rewards"] = [0.1, 0.8] * 4
    settings = arguments(tmp_path)
    settings.context_tokens = 16384
    asyncio.run(module.collect(settings))
    assert observed["generator_kwargs"]["limits"]["max_context_tokens"] == 16384
    group = json.loads((settings.output_dir / "training-group.json").read_text())
    assert group["context_tokens"] == 16384


def test_infrastructure_failure_does_not_cancel_independent_siblings(collector, tmp_path):
    module, observed = collector
    observed["rewards"] = [0.1, 0.8] * 4
    finished = []

    async def episode_hook(index):
        if index == 0:
            raise RuntimeError("actual transport failure")
        await asyncio.sleep(0.01)
        finished.append(index)

    observed["episode_hook"] = episode_hook
    settings = arguments(tmp_path)
    with pytest.raises(RuntimeError, match="independent siblings preserved"):
        asyncio.run(module.collect(settings))
    assert sorted(finished) == list(range(1, 8))
    saved = json.loads((settings.output_dir / "collection-failures.json").read_text())
    assert saved["successful_episodes_preserved"] == 7
    assert saved["optimizer_update_allowed"] is False
    assert saved["failures"][0]["trajectory"] == 0


def test_deferred_generation_saves_all_trajectories_without_final_scoring(
    collector, tmp_path, monkeypatch
):
    module, observed = collector

    class Generator:
        def __init__(self, *args, **kwargs):
            assert kwargs["defer_scoring"] is True

        async def run_episode(self, task, directory, version):
            directory.mkdir(parents=True)
            (directory / "task.json").write_text(json.dumps(task))
            (directory / "submission.json").write_text(json.dumps({"submission": "source"}))
            (directory / "calls.jsonl").write_text("saved exact calls")
            return {"reward": None, "samples": [object()]}

    monkeypatch.setattr(module, "KernelGenerator", Generator)
    args = arguments(tmp_path)
    args.defer_scoring = True
    asyncio.run(module.collect(args))
    manifest = json.loads((args.output_dir / "generation-manifest.json").read_text())
    assert len(manifest["episodes"]) == 8
    assert not list(args.output_dir.glob("*/score.json"))
    assert not (args.output_dir / "training-group.json").exists()


def test_root_admission_waits_for_verifier_and_releases_its_slot(collector, tmp_path, monkeypatch):
    _module, observed = collector
    from kernel_rlm import verifier_slot

    slot = tmp_path / "gpu-slot"
    monkeypatch.setattr(verifier_slot, "LOCK_PATH", slot)
    original_client = observed["client_class"](None, "unused", tmp_path / "trace")
    request = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(original_client, "request_completion", request)

    async def root_request():
        async with verifier_slot.inference_request_slot(1, path=slot):
            return await original_client.chat_completion({"json": {}})

    async def check():
        with slot.open("a") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX)
            pending = asyncio.create_task(root_request())
            await asyncio.sleep(0.02)
            request.assert_not_called()
            fcntl.flock(holder, fcntl.LOCK_UN)
            assert await asyncio.wait_for(pending, 1) == {"ok": True}
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)

    asyncio.run(check())


def test_main_enables_shared_gpu_control_for_workers_and_final_scorer(
    collector, tmp_path, monkeypatch
):
    module, _observed = collector
    monkeypatch.setenv("KERNEL_INFERENCE_ADMIN_URL", "http://127.0.0.1:9999")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect",
            "--checkpoint",
            str(tmp_path),
            "--dataset",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    async def check(arguments):
        assert os.environ["KERNEL_INFERENCE_ADMIN_URL"] == "http://127.0.0.1:8016"

    monkeypatch.setattr(module, "collect", check)
    module.main()


def test_free_slot_starts_next_rollout_before_slowest_finishes(collector, tmp_path):
    module, observed = collector
    observed["rewards"] = [0.0, 0.3, 0.7, 1.0] * 2

    async def check():
        fifth_started = asyncio.Event()

        async def episode_hook(index):
            if index == 0:
                await fifth_started.wait()
            elif index == 4:
                fifth_started.set()
            await asyncio.sleep(0)

        observed["episode_hook"] = episode_hook
        await asyncio.wait_for(module.collect(arguments(tmp_path)), 1)
        assert fifth_started.is_set()

    asyncio.run(check())
    result = json.loads((tmp_path / "collection/training-group.json").read_text())
    assert result["rewards"] == observed["rewards"]
    assert observed["peak"] <= 4
    assert observed["active"] == 0


def test_group_deadline_cancels_active_and_queued_rollouts(collector, tmp_path):
    module, observed = collector
    observed["rewards"] = [0.0, 0.3, 0.7, 1.0] * 2

    async def check():
        never = asyncio.Event()

        async def episode_hook(index):
            if index == 0:
                await asyncio.sleep(0)
                raise RuntimeError("injected episode failure")
            await never.wait()

        observed["episode_hook"] = episode_hook
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(module.collect(arguments(tmp_path)), 1)
        assert observed["active"] == 0

    asyncio.run(check())


@pytest.mark.parametrize("rewards", [[0.1] * 8, [0.0, 0.1] * 4])
def test_format_only_rewards_are_saved_but_never_admitted(collector, tmp_path, rewards):
    module, observed = collector
    observed["rewards"] = rewards
    args = arguments(tmp_path)
    with pytest.raises(RuntimeError, match="No reward above 0.1"):
        asyncio.run(module.collect(args))
    assert not (args.output_dir / "training-group.json").exists()
    recorded = json.loads((args.output_dir / "groups.json").read_text())
    assert recorded[0]["reason"] == "weak_signal"
    assert recorded[0]["rewards"] == rewards


def test_uniform_correct_rewards_are_filtered_without_changing_them(collector, tmp_path):
    module, observed = collector
    observed["rewards"] = [0.8] * 8
    args = arguments(tmp_path)
    with pytest.raises(RuntimeError, match="identical terminal rewards"):
        asyncio.run(module.collect(args))
    recorded = json.loads((args.output_dir / "groups.json").read_text())
    assert recorded[0]["reason"] == "zero_variance"
    assert recorded[0]["rewards"] == [0.8] * 8
    assert not (args.output_dir / "training-group.json").exists()


def test_verifier_oom_is_not_treated_as_a_trainable_reward(collector, tmp_path):
    module, observed = collector
    observed["rewards"] = [0.1, 0.8] * 4
    observed["verification"] = {
        0: {"verification_status": "infrastructure_error", "failure_kind": "verifier_oom"}
    }
    args = arguments(tmp_path)
    with pytest.raises(RuntimeError, match="Verifier infrastructure failed"):
        asyncio.run(module.collect(args))
    recorded = json.loads((args.output_dir / "groups.json").read_text())[0]
    assert recorded["reason"] == "infrastructure_error"
    assert recorded["rewards"] == observed["rewards"]
    assert recorded["infrastructure_errors"][0]["trajectory"] == 0
    assert not (args.output_dir / "training-group.json").exists()


def test_trackio_records_real_rollout_rewards_without_hosting(collector, tmp_path, monkeypatch):
    module, observed = collector
    observed["rewards"] = [0.1, 0.8] * 4
    tracker = SimpleNamespace(
        init=AsyncMock(),
        log=lambda metrics: records.append(metrics),
        finish=lambda: finished.append(True),
    )
    settings = []
    records = []
    finished = []
    tracker.init = lambda **kwargs: settings.append(kwargs)
    monkeypatch.setitem(sys.modules, "trackio", tracker)
    args = arguments(tmp_path)
    args.output_dir = tmp_path / "run/groups/step-002-task/collection/rollouts"
    args.report_to = "trackio"
    asyncio.run(module.collect(args))
    assert settings[0]["space_id"] is None
    assert [row["rollout/reward"] for row in records] == observed["rewards"]
    assert records[-1]["rollout/completed"] == 8
    assert finished == [True]
