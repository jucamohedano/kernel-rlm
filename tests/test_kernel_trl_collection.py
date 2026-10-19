import asyncio
import importlib
import json
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


def test_episode_failure_cancels_active_and_queued_rollouts(collector, tmp_path):
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
        with pytest.raises(RuntimeError, match="injected episode failure"):
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
