from kernel_skyrl.contract import RootSample
from kernel_trl.turn_credit import training_rows


def test_training_rows_preserve_call_provenance_and_exact_token_ranges():
    episodes = []
    for index, reward in enumerate([0.0, 0.2, 0.4, 0.6]):
        episodes.append(
            {
                "reward": reward,
                "execution_run_id": "run-a",
                "group_id": "group-a",
                "attempt_id": "attempt-2",
                "episode_id": f"episode-{index}",
                "samples": [
                    RootSample(
                        [10, 11],
                        [20, 21],
                        [-0.1, -0.2],
                        "tool_calls",
                        {
                            "root_turn_id": f"turn-{index}",
                            "call_id": f"call-{index}",
                            "model_call_id": f"call-{index}",
                        },
                    )
                ],
            }
        )
        if index == 0:
            episodes[-1]["samples"].append(
                RootSample(
                    [10],
                    [22],
                    [-0.3],
                    "stop",
                    {"root_turn_id": "turn-extra", "call_id": "call-extra"},
                )
            )
    result = training_rows(episodes, policy_version=3)
    row = result["rows"][0]
    assert result["execution_run_id"] == "run-a"
    assert row["root_turn_id"] == "turn-0"
    assert row["call_id"] == "call-0"
    assert row["completion_token_start"] == 0
    assert row["completion_token_end"] == 2
    assert row["loss_mask"] == [1, 1]
    assert result["rows"][-1]["padding"] is True
    assert result["rows"][-1]["advantage"] == 0
    assert not any(result["rows"][-1]["effective_loss_mask"])


def test_historical_samples_keep_provenance_unknown():
    episodes = [
        {"reward": value, "samples": [RootSample([1], [2], [-0.1], "stop")]}
        for value in [0.0, 0.2, 0.4, 0.6]
    ]
    result = training_rows(episodes)
    assert result["rows"][0]["call_id"] is None
    assert result["execution_run_id"] is None


def test_bounded_child_call_id_exists_before_admission_and_survives_failure(monkeypatch):
    import asyncio
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from kernel_rlm.subcalls import BoundedSubcalls

    @asynccontextmanager
    async def fail_admission(*args, **kwargs):
        raise TimeoutError("admission timeout")
        yield 0

    monkeypatch.setattr("kernel_rlm.subcalls.inference_request_slot", fail_admission)
    records = []
    calls = BoundedSubcalls(
        SimpleNamespace(), "policy", lambda value: 1, records, admission_enabled=True
    )
    try:
        asyncio.run(
            calls.query("question", {"parent_turn_id": "turn-a", "parent_tool_call_id": "tool-a"})
        )
    except TimeoutError:
        pass
    else:
        raise AssertionError("admission failure should propagate")
    assert records[0]["call_id"]
    assert records[0]["parent_turn_id"] == "turn-a"
    assert records[0]["parent_tool_call_id"] == "tool-a"
    assert records[0]["status"] == "failed"


def test_proxy_batched_queued_calls_share_dispatch_snapshot(monkeypatch):
    import asyncio
    from pathlib import Path
    from types import SimpleNamespace

    repository = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(repository / "training/src"))
    from rlm_train.proxy import ClientHandle, SubLLMProxy

    async def exercise():
        proxy = SubLLMProxy()
        entered = asyncio.Event()
        release = asyncio.Event()
        observed = []

        async def query(prompt, state):
            observed.append((prompt, dict(state)))
            if prompt == "one":
                entered.set()
                await release.wait()
            return "ok"

        state = {"parent_turn_id": "turn-a", "parent_tool_call_id": "tool-a"}
        proxy.register(
            "rollout",
            ClientHandle(None, "model", fake_query=query, max_concurrent=1, state_ref=state),
        )
        request = SimpleNamespace(
            match_info={"rollout_id": "rollout"},
            json=lambda: asyncio.sleep(0, result={"prompts": ["one", "two"]}),
        )
        pending = asyncio.create_task(proxy._handle_batched(request))
        await entered.wait()
        state.update(parent_turn_id="turn-b", parent_tool_call_id="tool-b")
        release.set()
        await pending
        assert observed == [
            ("one", {"parent_turn_id": "turn-a", "parent_tool_call_id": "tool-a"}),
            ("two", {"parent_turn_id": "turn-a", "parent_tool_call_id": "tool-a"}),
        ]

    asyncio.run(exercise())
