"""CPU-only native compile feedback through real worker subprocesses and proxy."""

import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import ClientSession
from rlm_train.native import NativeLimits, NativeRuntime
from rlm_train.proxy import ClientHandle, SubLLMProxy
from rlm_train.repl.subprocess import SubprocessReplBackend


@pytest.fixture(autouse=True)
def worker_import_path(monkeypatch):
    src = str(Path(__file__).parents[1] / "training/src")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (src, os.getenv("PYTHONPATH")))))


class ScriptedClient:
    def __init__(self, script):
        self.script = script
        self.calls = []

    async def post(self, path, *, json):
        self.calls.append(copy.deepcopy(json))
        code = self.script(json)
        response = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": f"tool-{len(self.calls)}",
                                "function": {"name": "repl", "arguments": {"code": code}},
                            }
                        ],
                    },
                }
            ],
        }
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: response)


@pytest.mark.asyncio
async def test_root_and_child_repeat_feedback_without_ready_latch(tmp_path):
    owners = []
    root_turn = 0

    async def feedback(owner, source):
        owners.append((owner, source))
        return {"status": "compile_failed", "stderr": "example compiler diagnostic", "stdout": ""}

    def script(payload):
        nonlocal root_turn
        prompt = payload["messages"][1]["content"]
        if prompt == "child":
            return (
                "checks = [compile_candidate('child-source') for _ in range(3)]\n"
                "assert not answer['ready']\n"
                "answer['content'] = checks[-1]['stderr']; answer['ready'] = True"
            )
        root_turn += 1
        if root_turn == 1:
            return (
                "checks = [compile_candidate('root-source') for _ in range(3)]\n"
                "assert not answer['ready']\n"
                "compile_candidate = 'overwritten'"
            )
        return (
            "assert callable(compile_candidate) and not answer['ready']\n"
            "last = compile_candidate('root-source')\n"
            "child = rlm_query('child')\n"
            "answer['content'] = last['stderr'] + ':' + child; answer['ready'] = True"
        )

    proxy = SubLLMProxy(socket_path=tmp_path / "proxy.sock")
    await proxy.start()
    client = ScriptedClient(script)
    runtime = NativeRuntime(
        client,
        "mock",
        proxy,
        compile_candidate=feedback,
        limits=NativeLimits(max_turns=3, max_model_calls=3, episode_timeout=10),
        event_path=tmp_path / "events.jsonl",
    )
    try:
        record = await runtime.run("root", {})
    finally:
        await runtime.close()
        await proxy.stop()
    assert record["status"] == "completed", record
    assert record["final_answer"] == "example compiler diagnostic:example compiler diagnostic"
    printed = record["events"][0]["actions"][0]["stdout"].splitlines()
    assert len(printed) == 3
    assert all(json.loads(line)["stderr"] == "example compiler diagnostic" for line in printed)
    assert len(client.calls) == runtime.calls == 3
    child = record["child_sessions"][0]
    for trajectory, source, count in ((record, "root-source", 4), (child, "child-source", 3)):
        entries = trajectory["compile_feedback"]
        assert len(entries) == count
        assert [e["ordinal"] for e in entries] == list(range(1, count + 1))
        assert all(e["session_id"] == trajectory["session_id"] for e in entries)
        assert all(
            e["source_sha256"] == hashlib.sha256(source.encode()).hexdigest() for e in entries
        )
        assert all(e["response"]["stderr"] == "example compiler diagnostic" for e in entries)
    assert {owner for owner, source in owners} == {record["session_id"], child["session_id"]}
    events = [json.loads(line) for line in runtime.event_path.read_text().splitlines()]
    assert len([e for e in events if e["type"] == "compile_feedback"]) == 7
    assert all(s.backend._proc is None for s in runtime.sessions.values())
    with pytest.raises(RuntimeError, match="owner session"):
        await runtime.candidate_feedback(record["session_id"], "stale")
    with pytest.raises(RuntimeError, match="owner session"):
        await runtime.candidate_feedback("foreign-owner", "foreign")


@pytest.mark.asyncio
async def test_worker_unconfigured_validation_and_proxy_errors(tmp_path):
    proxy = SubLLMProxy(socket_path=tmp_path / "proxy.sock")
    await proxy.start()
    proxy.register("owner", ClientHandle(client=None, model="mock"))
    worker = SubprocessReplBackend()
    await worker.start(proxy.url, "owner")
    try:
        for code, error in [
            ("compile_candidate(123)", "TypeError: compile_candidate source must be a string"),
            ("compile_candidate('  ')", "ValueError: compile_candidate source must not be empty"),
            ("compile_candidate('valid')", "RuntimeError: compile_candidate is not configured"),
        ]:
            result = await worker.execute(code)
            assert error in result.stderr
            assert result.final_answer is None

        async def bad_feedback(source):
            raise ValueError("feedback setup failed")

        proxy.register(
            "owner", ClientHandle(client=None, model="mock", compile_candidate=bad_feedback)
        )
        result = await worker.execute("compile_candidate('valid')")
        assert "ValueError: feedback setup failed" in result.stderr

        async def wrong_shape(source):
            return "not a dict"

        proxy.register(
            "owner", ClientHandle(client=None, model="mock", compile_candidate=wrong_shape)
        )
        result = await worker.execute("compile_candidate('valid')")
        assert "TypeError: compile_candidate callback must return a dict" in result.stderr
        # Missing, nontext and empty source fields also fail at the host boundary.
        from aiohttp import UnixConnector

        async with ClientSession(
            connector=UnixConnector(path=str(tmp_path / "proxy.sock"))
        ) as http:
            for body, error in [
                ({}, "ValueError"),
                ({"source": 3}, "TypeError"),
                ({"source": ""}, "ValueError"),
            ]:
                async with http.post(
                    "http://localhost/rollout/owner/compile_candidate", json=body
                ) as response:
                    assert (await response.json())["error_type"] == error
        proxy.unregister("owner")
        result = await worker.execute("compile_candidate('valid')")
        assert "unknown rollout_id" in result.stderr
    finally:
        await worker.stop()
        await proxy.stop()


@pytest.mark.asyncio
async def test_turn_notice_precedes_request_and_last_turn_can_finalize(tmp_path):
    count = 0
    seen = []

    def notice(current, maximum):
        seen.append((current, maximum))
        return f"turn {current}/{maximum}: submit" if current >= 28 else None

    def script(payload):
        nonlocal count
        count += 1
        if count < 28:
            assert not any(
                m["content"].startswith("turn ")
                for m in payload["messages"]
                if m.get("role") == "user"
            )
        else:
            assert payload["messages"][-1] == {
                "role": "user",
                "content": f"turn {count}/32: submit",
            }
        if count == 32:
            return "assert history[-2]['content'] == 'turn 32/32: submit'; answer['content'] = 'final'; answer['ready'] = True"
        return "print('continue')"

    proxy = SubLLMProxy(socket_path=tmp_path / "proxy.sock")
    await proxy.start()
    runtime = NativeRuntime(
        ScriptedClient(script),
        "mock",
        proxy,
        turn_notice=notice,
        limits=NativeLimits(max_turns=32, episode_timeout=10),
        event_path=tmp_path / "events.jsonl",
    )
    try:
        record = await runtime.run("question", {})
    finally:
        await runtime.close()
        await proxy.stop()
    assert record["final_answer"] == "final", record
    assert seen == [(turn, 32) for turn in range(1, 33)]
    notices = [
        m
        for m in record["messages"]
        if m.get("role") == "user" and m["content"].startswith("turn ")
    ]
    assert len(notices) == 5
    events = [json.loads(line) for line in runtime.event_path.read_text().splitlines()]
    assert [e["turn"] for e in events if e["type"] == "turn_notice"] == list(range(28, 33))


@pytest.mark.asyncio
async def test_default_notice_keeps_old_protocol(tmp_path):
    proxy = SubLLMProxy(socket_path=tmp_path / "proxy.sock")
    await proxy.start()
    client = ScriptedClient(lambda payload: "answer['content'] = 'ok'; answer['ready'] = True")
    runtime = NativeRuntime(client, "mock", proxy)
    try:
        record = await runtime.run("question", {})
    finally:
        await runtime.close()
        await proxy.stop()
    assert record["final_answer"] == "ok"
    assert [m["role"] for m in client.calls[0]["messages"]] == ["system", "user"]
    assert "compile_candidate" not in client.calls[0]["messages"][0]["content"]
    assert "compile_feedback" not in record


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["close", "episode_timeout"])
async def test_close_cancels_and_joins_host_feedback(tmp_path, finish):
    entered, cancelled, joined = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def feedback(owner, source):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        finally:
            await asyncio.sleep(0)
            joined.set()

    proxy = SubLLMProxy(socket_path=tmp_path / "proxy.sock")
    await proxy.start()
    runtime = NativeRuntime(
        ScriptedClient(lambda payload: "compile_candidate('source')"),
        "mock",
        proxy,
        compile_candidate=feedback,
        limits=NativeLimits(episode_timeout=0.3 if finish == "episode_timeout" else 10),
    )
    run = asyncio.create_task(runtime.run("question", {}))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        # A registered callback is owner-bound; even a retained callback cannot run after close.
        callback = proxy._handles[runtime.root.id].compile_candidate
        if finish == "close":
            await runtime.close()
        record = await run
        assert cancelled.is_set() and joined.is_set()
        if finish == "episode_timeout":
            assert record["error_class"] == "budget"
        assert not runtime.root.requests
        assert runtime.root.backend._proc is None
        entry = record["compile_feedback"][0]
        assert (entry["status"], entry["error_type"]) in {
            ("cancelled", "CancelledError"),
            ("failed", "TimeoutError"),
        }
        with pytest.raises(RuntimeError, match="owner session"):
            await callback("stale source")
    finally:
        await runtime.close()
        await proxy.stop()


@pytest.mark.asyncio
async def test_feedback_obeys_existing_request_timeout(tmp_path):
    cancelled = asyncio.Event()

    async def feedback(owner, source):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    count = 0

    def script(payload):
        nonlocal count
        count += 1
        if count == 1:
            return "compile_candidate('source')"
        return "assert not answer['ready']; answer['content']='explicit'; answer['ready']=True"

    proxy = SubLLMProxy(socket_path=tmp_path / "proxy.sock")
    await proxy.start()
    runtime = NativeRuntime(
        ScriptedClient(script),
        "mock",
        proxy,
        compile_candidate=feedback,
        limits=NativeLimits(request_timeout=0.05, episode_timeout=5),
    )
    try:
        record = await runtime.run("question", {})
    finally:
        await runtime.close()
        await proxy.stop()
    assert cancelled.is_set()
    assert record["final_answer"] == "explicit", record
    assert record["compile_feedback"][0]["error_type"] == "TimeoutError"
    assert "TimeoutError" in record["events"][0]["actions"][0]["stderr"]
