import asyncio
import time
from types import SimpleNamespace

import pytest
from rlm_train.native import NativeLimits, NativeRuntime
from rlm_train.token_budget import (
    ContextBudgetPolicy,
    GeneratedTokenBudget,
    VllmPromptCounter,
)


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class FakeClient:
    def __init__(self, data):
        self.data = data
        self.calls = []
        self.base_url = "http://localhost/v1/"

    async def post(self, endpoint, *, json):
        self.calls.append((endpoint, json))
        return FakeResponse(self.data)


def test_vllm_prompt_counter_counts_rendered_payload_and_checks_model_limit():
    client = FakeClient({"count": 17, "max_model_len": 4096})
    counter = VllmPromptCounter(client, model_context=4096)
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function"}],
        "chat_template_kwargs": {"enable_thinking": True},
        "max_tokens": 10,
    }
    assert asyncio.run(counter(payload)) == 17
    assert client.calls == [
        (
            "http://localhost/tokenize",
            {
                **{k: payload[k] for k in ("model", "messages", "tools", "chat_template_kwargs")},
                "add_generation_prompt": True,
                "continue_final_message": False,
                "add_special_tokens": False,
            },
        )
    ]


def test_vllm_prompt_counter_accepts_token_array_and_rejects_context_mismatch():
    assert (
        asyncio.run(
            VllmPromptCounter(FakeClient({"tokens": [1, 2], "max_model_len": 10}), 10)(
                {"messages": []}
            )
        )
        == 2
    )
    with pytest.raises(ValueError, match="differs"):
        asyncio.run(VllmPromptCounter(FakeClient({"count": 2, "max_model_len": 11}), 10)({}))


def test_shared_budget_reserves_then_settles_actual_usage():
    budget = GeneratedTokenBudget(
        ContextBudgetPolicy(server_context_tokens=100, output_fraction=0.9)
    )
    first = budget.reserve("a", 20)
    assert first == 72
    assert budget.outstanding == 72
    actual, reserved = budget.settle("a", {"completion_tokens": 30})
    assert (actual, reserved) == (30, 72)
    assert budget.used == 30 and budget.outstanding == 0
    assert budget.reserve("b", 20) == 72


def test_missing_or_invalid_usage_retains_reservation_and_fails_closed():
    budget = GeneratedTokenBudget(ContextBudgetPolicy(server_context_tokens=100))
    reserved = budget.reserve("a", 20)
    assert budget.settle("a", None) == (None, reserved)
    assert budget.outstanding == reserved and budget.incomplete
    with pytest.raises(RuntimeError, match="incomplete"):
        budget.reserve("b", 20)


def test_policy_validates_and_applies_fractional_context_cap():
    with pytest.raises(ValueError):
        ContextBudgetPolicy(output_fraction=0)
    budget = GeneratedTokenBudget(
        ContextBudgetPolicy(server_context_tokens=11, output_fraction=0.9)
    )
    assert budget.reserve("a", 2) == 8


class ChatClient:
    def __init__(self, completion_tokens):
        self.completion_tokens = completion_tokens
        self.payloads = []

    async def post(self, endpoint, *, json):
        self.payloads.append(json)
        return FakeResponse(
            {
                "usage": {"prompt_tokens": 20, "completion_tokens": self.completion_tokens},
                "choices": [{"logprobs": {"content": [{"token": "token_id:1", "logprob": -0.1}]}}],
            }
        )


def test_runtime_uses_policy_cap_and_settles_usage_for_all_branches():
    async def prompt_counter(payload):
        assert payload["messages"] == [{"role": "user", "content": "p"}]
        return 20

    client = ChatClient(completion_tokens=30)
    runtime = NativeRuntime(
        client,
        "model",
        object(),
        limits=NativeLimits(),
        budget_policy=ContextBudgetPolicy(server_context_tokens=100),
        prompt_counter=prompt_counter,
    )
    runtime.deadline = time.monotonic() + 10
    session = SimpleNamespace(
        depth=1,
        parent="parent",
        id="child",
        origin={},
        current_action={},
        record={"events": []},
        requests=set(),
        accepting=True,
    )
    event = asyncio.run(
        runtime.request(session, [{"role": "user", "content": "p"}], "model", plain=True)
    )
    assert client.payloads[0]["max_tokens"] == 72
    assert client.payloads[0]["logprobs"] is True
    assert client.payloads[0]["top_logprobs"] == 0
    assert client.payloads[0]["return_tokens_as_token_ids"] is True
    assert event["budget"]["used_tokens"] == 30
    assert event["budget"]["outstanding_after"] == 0
    assert event["response"]["usage"]["completion_tokens"] == 30
    assert event["timing"]["queue_wait_s"] >= 0
    assert event["timing"]["http_elapsed_s"] >= 0
    assert event["timing"]["request_elapsed_s"] >= event["timing"]["http_elapsed_s"]


def test_parallel_reservations_share_spending_but_not_context():
    budget = GeneratedTokenBudget(
        ContextBudgetPolicy(server_context_tokens=100, max_generated_tokens=100)
    )
    assert budget.reserve("parent", 20) == 72
    assert budget.reserve("child", 20) == 28
    assert budget.outstanding == 100
    assert budget.settle("child", {"completion_tokens": 10}) == (10, 28)
    assert budget.settle("parent", {"completion_tokens": 20}) == (20, 72)
    assert budget.reserve("next", 20) == 70


def test_undispatched_failure_releases_and_cancelled_dispatch_closes_budget():
    budget = GeneratedTokenBudget(ContextBudgetPolicy(server_context_tokens=100))
    budget.reserve("not-sent", 20)
    budget.release_undispatched("not-sent")
    assert budget.outstanding == 0
    cap = budget.reserve("sent", 20)
    assert budget.retain_uncertain("sent") == cap
    with pytest.raises(RuntimeError, match="incomplete"):
        budget.reserve("next", 20)


@pytest.mark.asyncio
async def test_async_parent_and_child_reserve_disjoint_allowances():
    gate = asyncio.Event()
    both = asyncio.Event()
    calls = []

    class Client:
        async def post(self, endpoint, *, json):
            calls.append(json)
            if len(calls) == 2:
                both.set()
            await gate.wait()
            return FakeResponse(
                {"usage": {"prompt_tokens": 20, "completion_tokens": 10}, "choices": []}
            )

    async def count(payload):
        return 20

    runtime = NativeRuntime(
        Client(),
        "m",
        object(),
        budget_policy=ContextBudgetPolicy(
            server_context_tokens=100, max_generated_tokens=100, logprobs=False
        ),
        prompt_counter=count,
    )
    runtime.deadline = time.monotonic() + 5

    def session(identity, parent, depth):
        return SimpleNamespace(
            id=identity,
            parent=parent,
            depth=depth,
            origin={},
            current_action={},
            record={"events": []},
            requests=set(),
            accepting=True,
        )

    tasks = [
        asyncio.create_task(runtime.request(session("root", None, 0), [], "m", plain=False)),
        asyncio.create_task(runtime.request(session("child", "root", 1), [], "m", plain=True)),
    ]
    try:
        await asyncio.wait_for(both.wait(), 1)
        assert sorted(call["max_tokens"] for call in calls) == [28, 72]
        assert runtime.generated_budget.outstanding == 100
    finally:
        gate.set()
        await asyncio.gather(*tasks)
    assert runtime.generated_budget.used == 20
    assert runtime.generated_budget.outstanding == 0


@pytest.mark.asyncio
async def test_predispatch_journal_error_does_not_charge_or_send_request():
    async def count(payload):
        return 20

    client = ChatClient(30)
    runtime = NativeRuntime(
        client,
        "m",
        object(),
        budget_policy=ContextBudgetPolicy(server_context_tokens=100),
        prompt_counter=count,
    )
    runtime.deadline = time.monotonic() + 5
    session = SimpleNamespace(
        id="s",
        parent=None,
        depth=0,
        origin={},
        current_action={},
        record={"events": []},
        requests=set(),
        accepting=True,
    )

    def broken_log(*args, **kwargs):
        raise OSError("journal disk full")

    runtime.log = broken_log
    with pytest.raises(OSError, match="journal"):
        await runtime.request(session, [], "m", plain=False)
    assert runtime.generated_budget.outstanding == 0
    assert runtime.calls == 0 and not client.payloads


@pytest.mark.asyncio
async def test_child_waits_for_parent_unused_reservation_without_spending_inference_slot():
    parent_started = asyncio.Event()
    release_parent = asyncio.Event()
    dispatched = []

    class Client:
        async def post(self, endpoint, *, json):
            dispatched.append(json["messages"][0]["content"])
            if dispatched[-1] == "parent":
                parent_started.set()
                await release_parent.wait()
            return FakeResponse(
                {"usage": {"prompt_tokens": 20, "completion_tokens": 10}, "choices": []}
            )

    async def count(payload):
        return 20

    runtime = NativeRuntime(
        Client(),
        "m",
        object(),
        budget_policy=ContextBudgetPolicy(
            server_context_tokens=100, max_generated_tokens=50, logprobs=False
        ),
        prompt_counter=count,
    )
    runtime.deadline = time.monotonic() + 5

    def session(name):
        return SimpleNamespace(
            id=name,
            parent=None,
            depth=0,
            origin={},
            current_action={},
            record={"events": []},
            requests=set(),
            accepting=True,
        )

    parent = asyncio.create_task(
        runtime.request(session("p"), [{"role": "user", "content": "parent"}], "m", plain=False)
    )
    await parent_started.wait()
    child = asyncio.create_task(
        runtime.request(session("c"), [{"role": "user", "content": "child"}], "m", plain=True)
    )
    await asyncio.sleep(0)
    assert not child.done()
    assert dispatched == ["parent"]
    release_parent.set()
    await asyncio.gather(parent, child)
    assert dispatched == ["parent", "child"]
    assert runtime.generated_budget.used == 20


@pytest.mark.asyncio
async def test_reserved_waiter_wakes_and_fails_closed_after_owner_error():
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    class Client:
        async def post(self, endpoint, *, json):
            calls.append(json)
            started.set()
            await release.wait()
            raise OSError("transport failed after dispatch")

    async def count(payload):
        return 20

    runtime = NativeRuntime(
        Client(),
        "m",
        object(),
        budget_policy=ContextBudgetPolicy(
            server_context_tokens=100, max_generated_tokens=50, logprobs=False
        ),
        prompt_counter=count,
    )
    runtime.deadline = time.monotonic() + 3

    def s(name):
        return SimpleNamespace(
            id=name,
            parent=None,
            depth=0,
            origin={},
            current_action={},
            record={"events": []},
            requests=set(),
            accepting=True,
        )

    first = asyncio.create_task(runtime.request(s("first"), [], "m", plain=True))
    await started.wait()
    second = asyncio.create_task(runtime.request(s("second"), [], "m", plain=True))
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert isinstance(results[0], OSError)
    assert isinstance(results[1], RuntimeError)
    assert "incomplete" in str(results[1])
    assert len(calls) == 1
