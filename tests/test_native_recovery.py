"""CPU-only checkpoints: preserve state and budget; never replay failed attempts."""

from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import httpx
import pytest
from rlm_train.native import NativeLimits, NativeRuntime
from rlm_train.proxy import SubLLMProxy
from rlm_train.repl.subprocess import SubprocessReplBackend
from rlm_train.token_budget import ContextBudgetPolicy


class DataBackend:
    def __init__(self, **kwargs):
        self.state = {}
        self.executed = []
        self.stopped = False

    async def start(self, *args, **kwargs):
        pass

    async def load_context(self, context):
        self.context = context

    async def update_history(self, messages):
        pass

    async def execute(self, code):
        self.executed.append(code)
        if code == "initialize":
            self.state["value"] = 41
            return SimpleNamespace(stdout="initialized", stderr="", final_answer=None)
        assert code == "finish"
        assert self.state["value"] == 41
        return SimpleNamespace(stdout="", stderr="", final_answer="42")

    async def snapshot(self):
        assert not self.stopped, "capture must precede cleanup"
        return {"schema_version": 1, "variables": copy.deepcopy(self.state), "unsupported": []}

    async def restore(self, snapshot):
        self.state = copy.deepcopy(snapshot["variables"])

    async def stop(self):
        self.stopped = True


def tool(code):
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "reasoning_content": "complete reasoning",
                    "tool_calls": [
                        {
                            "id": code,
                            "type": "function",
                            "function": {
                                "name": "repl",
                                "arguments": json.dumps({"code": code}),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }


class Client:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.requests = []

    async def post(self, path, *, json):
        self.requests.append(copy.deepcopy(json))
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: outcome)


async def count(payload):
    return 2


def runtime(tmp_path, name, client, **kwargs):
    return NativeRuntime(
        client,
        "mock",
        SubLLMProxy(),
        backend_factory=DataBackend,
        limits=NativeLimits(max_turns=4, episode_timeout=5, request_timeout=2),
        checkpoint_path=tmp_path / f"{name}.json",
        event_path=tmp_path / f"{name}.jsonl",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_resume_restores_without_replay_and_keeps_failed_attempt_immutable(tmp_path):
    first = runtime(
        tmp_path, "failed", Client([tool("initialize"), httpx.ReadError("lost response")])
    )
    failed = await first.run("question", {"source": "source code"})
    saved_bytes = first.checkpoint_path.read_bytes()
    saved = json.loads(saved_bytes)
    assert failed["status"] == "failed"
    assert saved["recoverable"] is True
    assert saved["cleanup_confirmed"] is True
    assert saved["session"]["turns_spent"] == 2
    assert saved["counters"]["calls"] == 2
    assert saved["failure"]["transport_phase"] == "read"
    assert first.root.backend.executed == ["initialize"]
    assert first.root.backend.stopped

    second = runtime(tmp_path, "resumed", Client([tool("finish")]))
    completed = await second.resume(first.checkpoint_path, additional_seconds=5)
    assert completed["final_answer"] == "42"
    assert completed["resume_of"]["episode_id"] == first.episode_id
    assert completed["turns_spent"] == 3
    assert second.calls == 3
    assert second.root.backend.executed == ["finish"]
    assert first.checkpoint_path.read_bytes() == saved_bytes
    assert failed["status"] == "failed"
    assert second.client.requests[0]["messages"][2]["reasoning_content"] == "complete reasoning"


@pytest.mark.asyncio
async def test_unknown_usage_requires_explicit_full_reservation_charge(tmp_path):
    policy = ContextBudgetPolicy(
        server_context_tokens=100, max_generated_tokens=1000, logprobs=False
    )
    kwargs = {"budget_policy": policy, "prompt_counter": count}
    first = runtime(
        tmp_path,
        "failed",
        Client([tool("initialize"), httpx.ConnectError("unavailable")]),
        **kwargs,
    )
    await first.run("question", "source")
    checkpoint = json.loads(first.checkpoint_path.read_text())
    assert checkpoint["generated_budget"]["used"] == 3
    reserved = sum(checkpoint["generated_budget"]["reserved"].values())
    assert reserved == 88
    denied = runtime(tmp_path, "denied", Client([]), **kwargs)
    with pytest.raises(ValueError, match="charge_uncertain_reservation"):
        await denied.resume(first.checkpoint_path, additional_seconds=5)
    second = runtime(tmp_path, "resumed", Client([tool("finish")]), **kwargs)
    record = await second.resume(
        first.checkpoint_path, additional_seconds=5, charge_uncertain_reservation=True
    )
    assert record["final_answer"] == "42"
    assert second.generated_budget.used == 3 + reserved + 3
    assert record["resume_adjustments"]["conservative_uncertain_charge"] == reserved
    assert record["resume_adjustments"]["charge_is_actual_usage"] is False


@pytest.mark.asyncio
async def test_resume_rejects_config_changes_and_spent_turns_do_not_reset(tmp_path):
    first = runtime(tmp_path, "failed", Client([tool("initialize"), httpx.ReadError("lost")]))
    await first.run("question", "source")
    different = runtime(tmp_path, "different", Client([]), system_prompt="different")
    with pytest.raises(ValueError, match="configuration differs"):
        await different.resume(first.checkpoint_path, additional_seconds=5)
    checkpoint = json.loads(first.checkpoint_path.read_text())
    checkpoint["session"]["turns_spent"] = 4
    exhausted = runtime(tmp_path, "exhausted", Client([]))
    record = await exhausted.resume(checkpoint, additional_seconds=5)
    assert record["status"] == "failed"
    assert "turn limit" in record["error"]
    assert exhausted.calls == 2
    assert exhausted.client.requests == []


@pytest.mark.asyncio
async def test_unsupported_snapshot_and_repl_failure_are_not_recoverable(tmp_path):
    class Unsupported(DataBackend):
        async def snapshot(self):
            return {
                "schema_version": 1,
                "variables": {},
                "unsupported": [{"name": "f", "type": "function"}],
            }

    first = runtime(tmp_path, "unsupported", Client([httpx.ReadError("lost")]))
    first.backend_factory = Unsupported
    await first.run("question", "source")
    checkpoint = json.loads(first.checkpoint_path.read_text())
    assert not checkpoint["recoverable"]
    assert "unsupported state" in str(checkpoint["recovery_refusal"])

    class Broken(DataBackend):
        async def execute(self, code):
            raise TimeoutError("compile or REPL action timed out")

    second = runtime(tmp_path, "broken", Client([tool("initialize")]))
    second.backend_factory = Broken
    await second.run("question", "source")
    checkpoint = json.loads(second.checkpoint_path.read_text())
    assert not checkpoint["recoverable"]
    assert checkpoint["failure"]["stage"] == "repl_execute"


@pytest.mark.asyncio
@pytest.mark.parametrize("episode_timeout,request_timeout", [(5, 0.01), (0.01, 5)])
async def test_model_deadline_has_local_evidence_and_checkpoint(
    tmp_path, episode_timeout, request_timeout
):
    class Stalled(Client):
        async def post(self, path, *, json):
            await asyncio.Event().wait()

    first = runtime(tmp_path, "deadline", Stalled([]))
    first.limits = NativeLimits(
        max_turns=4, episode_timeout=episode_timeout, request_timeout=request_timeout
    )
    record = await first.run("question", "source")
    checkpoint = json.loads(first.checkpoint_path.read_text())
    assert checkpoint["recoverable"]
    assert checkpoint["failure"]["local_deadline"]
    assert checkpoint["failure"]["server_root_cause"] == "unknown"
    assert record["events"][0]["failure"]["last_progress"]["stage"] == "dispatch"


class Wire(httpx.AsyncByteStream):
    def __init__(self, chunks, failure=None):
        self.chunks = chunks
        self.failure = failure

    async def __aiter__(self):
        for raw in self.chunks:
            yield raw
        if self.failure is not None:
            raise self.failure


def sse(data):
    return ("data: " + json.dumps(data) + "\n\n").encode()


@pytest.mark.asyncio
async def test_streamed_runtime_keeps_canonical_response_and_durable_wire_chunks(tmp_path):
    def endpoint(request):
        payload = json.loads(request.content)
        code = "finish" if len(payload["messages"]) > 2 else "initialize"
        raw = tool(code)
        message = raw["choices"][0]["message"]
        message["tool_calls"][0]["index"] = 0
        wire = (
            sse(
                {
                    "id": code,
                    "choices": [
                        {
                            "index": 0,
                            "delta": message,
                            "finish_reason": "tool_calls",
                            "logprobs": {"content": [{"token": "x", "logprob": -0.1}]},
                        }
                    ],
                }
            )
            + sse({"choices": [], "usage": raw["usage"]})
            + b"data: [DONE]\n\n"
        )
        return httpx.Response(200, stream=Wire([wire[:10], wire[10:]]))

    policy = ContextBudgetPolicy(server_context_tokens=100, max_generated_tokens=1000)
    async with httpx.AsyncClient(
        base_url="https://example.test/v1/", transport=httpx.MockTransport(endpoint)
    ) as client:
        first = runtime(
            tmp_path,
            "stream",
            client,
            stream_responses=True,
            budget_policy=policy,
            prompt_counter=count,
        )
        record = await first.run("question", "source")
    assert record["final_answer"] == "42"
    event = record["events"][0]
    assert event["request"]["stream_options"] == {"include_usage": True}
    assert event["response"]["choices"][0]["message"]["reasoning_content"] == "complete reasoning"
    assert (
        event["response"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        == '{"code": "initialize"}'
    )
    assert event["selected_logprob_tokens"] == 1
    journal = [json.loads(line) for line in first.event_path.read_text().splitlines()]
    chunks = [e for e in journal if e["type"] == "stream_chunk"]
    assert len(chunks) == 4
    assert chunks[0]["call_id"] == event["call_id"]
    first_chunk = next(i for i, e in enumerate(journal) if e["type"] == "stream_chunk")
    first_finished = next(i for i, e in enumerate(journal) if e["type"] == "request_finished")
    assert first_chunk < first_finished


@pytest.mark.asyncio
async def test_stream_disconnect_retains_partial_evidence_without_executing_tools(tmp_path):
    raw = sse(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_content": "partial reasoning",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "partial",
                                "type": "function",
                                "function": {
                                    "name": "repl",
                                    "arguments": '{"code": "initialize',
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ]
        }
    )

    def endpoint(request):
        return httpx.Response(200, stream=Wire([raw], httpx.ReadError("peer disconnected")))

    async with httpx.AsyncClient(
        base_url="https://example.test/v1/", transport=httpx.MockTransport(endpoint)
    ) as client:
        first = runtime(tmp_path, "partial", client, stream_responses=True)
        record = await first.run("question", "source")
    assert record["status"] == "failed"
    assert first.root.backend.executed == []
    assert record["failure"]["transport_phase"] == "read"
    assert (
        record["failure"]["last_progress"]["last_delta"]["choices"][0]["delta"]["reasoning_content"]
        == "partial reasoning"
    )
    journal = [json.loads(line) for line in first.event_path.read_text().splitlines()]
    chunks = [e for e in journal if e["type"] == "stream_chunk"]
    assert len(chunks) == 1
    assert chunks[0]["bytes_base64"]
    assert json.loads(first.checkpoint_path.read_text())["recoverable"]


@pytest.mark.asyncio
async def test_resume_real_subprocess_keeps_variables_without_replaying_actions(
    tmp_path, monkeypatch
):
    from pathlib import Path

    source = str(Path(__file__).parents[1] / "training" / "src")
    monkeypatch.setenv("PYTHONPATH", source)
    marker = tmp_path / "executed.txt"
    initialize = f"value = 41\nopen({str(marker)!r}, 'a').write('x')"
    finish = "answer['content'] = str(value + 1); answer['ready'] = True"
    proxy = SubLLMProxy(socket_path=tmp_path / "proxy.sock")
    await proxy.start()
    limits = NativeLimits(max_turns=4, episode_timeout=5, request_timeout=2)
    try:
        first = NativeRuntime(
            Client([tool(initialize), httpx.ReadError("interrupted")]),
            "mock",
            proxy,
            limits=limits,
            backend_factory=SubprocessReplBackend,
            checkpoint_path=tmp_path / "failed.json",
        )
        record = await first.run("question", "source")
        assert record["checkpoint"]["recoverable"], record
        second = NativeRuntime(
            Client([tool(finish)]),
            "mock",
            proxy,
            limits=limits,
            checkpoint_path=tmp_path / "resumed.json",
        )
        completed = await second.resume(first.checkpoint_path, additional_seconds=5)
    finally:
        await proxy.stop()
    assert completed["final_answer"] == "42", completed
    assert marker.read_text() == "x"
    assert first.root.backend._proc is None
    assert second.root.backend._proc is None


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["repl_execute", "compile_feedback"])
@pytest.mark.parametrize("finish", ["deadline", "close"])
async def test_caught_plain_failure_does_not_make_later_action_recoverable(
    tmp_path, operation, finish
):
    entered = asyncio.Event()
    first = runtime(
        tmp_path, "action", Client([tool("initialize"), httpx.ReadError("plain helper lost")])
    )
    first.limits = NativeLimits(
        max_turns=4, episode_timeout=0.05 if finish == "deadline" else 5, request_timeout=1
    )

    async def feedback(owner, source):
        assert first.root.stage == "compile_feedback"
        entered.set()
        await asyncio.Event().wait()

    class ActiveAction(DataBackend):
        async def execute(self, code):
            try:
                await first.request(first.root, [], "mock", plain=True)
            except httpx.ReadError:
                pass
            if operation == "compile_feedback":
                await first.candidate_feedback(first.root.id, "source")
            else:
                entered.set()
                await asyncio.Event().wait()

    first.backend_factory = ActiveAction
    first.compile_candidate = feedback
    running = asyncio.create_task(first.run("question", "source"))
    if finish == "close":
        await asyncio.wait_for(entered.wait(), 2)
        await first.close()
    record = await running
    saved = json.loads(first.checkpoint_path.read_text())
    assert record["status"] == "failed"
    assert saved["failure"]["stage"] in {"repl_execute", "compile_feedback"}
    assert saved["recoverable"] is False
    assert saved["cleanup_confirmed"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("episode_timeout,request_timeout", [(5, 0.01), (0.01, 5)])
async def test_prompt_count_deadline_is_not_a_recoverable_generation_deadline(
    tmp_path, episode_timeout, request_timeout
):
    async def stalled_counter(payload):
        await asyncio.Event().wait()

    first = runtime(
        tmp_path,
        "tokenize",
        Client([]),
        budget_policy=ContextBudgetPolicy(
            server_context_tokens=100, max_generated_tokens=1000, logprobs=False
        ),
        prompt_counter=stalled_counter,
    )
    first.limits = NativeLimits(
        max_turns=4, episode_timeout=episode_timeout, request_timeout=request_timeout
    )
    await first.run("question", "source")
    saved = json.loads(first.checkpoint_path.read_text())
    assert saved["failure"]["stage"] == "prompt_count"
    assert saved["failure"]["local_deadline"] is True
    assert saved["recoverable"] is False
    assert first.client.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("destination", ["checkpoint_path", "event_path"])
@pytest.mark.parametrize("source", ["checkpoint", "journal", "copied_checkpoint"])
async def test_resume_rejects_all_source_artifact_collisions(tmp_path, destination, source):
    first = runtime(tmp_path, "failed", Client([httpx.ReadError("lost")]))
    await first.run("question", "source")
    copied = tmp_path / "copied-checkpoint.json"
    copied.write_bytes(first.checkpoint_path.read_bytes())
    sources = {
        "checkpoint": first.checkpoint_path,
        "journal": first.event_path,
        "copied_checkpoint": copied,
    }
    before = {path: path.read_bytes() for path in sources.values()}
    paths = {
        "checkpoint_path": tmp_path / "new-checkpoint.json",
        "event_path": tmp_path / "new-events.jsonl",
    }
    paths[destination] = sources[source]
    second = NativeRuntime(
        Client([]),
        "mock",
        SubLLMProxy(),
        backend_factory=DataBackend,
        limits=first.limits,
        **paths,
    )
    with pytest.raises(ValueError, match="source artifact"):
        await second.resume(copied, additional_seconds=5)
    assert second.root is None
    assert second.client.requests == []
    for path, content in before.items():
        assert path.read_bytes() == content
    assert json.loads(first.checkpoint_path.read_text())["recoverable"] is True


def test_checkpoint_and_event_paths_must_be_distinct(tmp_path):
    path = tmp_path / "attempt.json"
    with pytest.raises(ValueError, match="different files"):
        NativeRuntime(Client([]), "mock", SubLLMProxy(), checkpoint_path=path, event_path=path)
    assert not path.exists()
