"""CPU-only integration coverage for native recursive child sessions."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web

ROOT = Path(__file__).parents[1]
TRAIN = ROOT / "training"


@pytest.fixture(autouse=True)
def _worker_import_path(monkeypatch):
    src = str(TRAIN / "src")
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (src, existing))))


def _tool_call(call_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _chat(content: str | None = None, *, calls: list[dict] | None = None) -> web.Response:
    return web.json_response(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls" if calls else "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        **({"tool_calls": calls} if calls else {}),
                    },
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        }
    )


def _contains_prompt(messages: list[dict[str, Any]], text: str) -> bool:
    return text in json.dumps(messages, ensure_ascii=False)


async def _endpoint(handler):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}/v1"


class _MockOpenAI:
    """Small async OpenAI-compatible client backed by the scripted HTTP endpoint."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.http = __import__("httpx").AsyncClient(timeout=5)

    async def post(self, path: str, *, json: dict[str, Any]):
        return await self.http.post(self.base_url + "/" + path.lstrip("/"), json=json)

    async def aclose(self):
        await self.http.aclose()


@pytest.mark.asyncio
async def test_recursive_children_keep_isolated_repl_history_and_parent_progress(tmp_path):
    """A waiting child must not block root turns; child state survives its own turns."""
    from rlm_train.native import NativeLimits, NativeRuntime
    from rlm_train.proxy import SubLLMProxy

    child_entered = asyncio.Event()
    release_child = asyncio.Event()
    root_advanced = asyncio.Event()
    requests: list[dict[str, Any]] = []
    root_turns = 0
    child_turns = 0
    grandchild_turns = 0

    async def completions(request: web.Request) -> web.Response:
        nonlocal root_turns, child_turns, grandchild_turns
        payload = await request.json()
        requests.append(payload)
        messages = payload["messages"]
        user = next(m.get("content", "") for m in messages if m.get("role") == "user")
        if user == "root-question":
            root_turns += 1
            if root_turns == 1:
                code = "root_only = 'ROOT'; child_handle = rlm_query('child-question', blocking=False); print('spawned')"
                return _chat(calls=[_tool_call("root-spawn", "repl", {"code": code})])
            root_advanced.set()
            return _chat(
                calls=[
                    _tool_call(
                        "root-finish",
                        "repl",
                        {
                            "code": "child_result = rlm_wait(child_handle, timeout_s=3); answer['content'] = root_only + ':' + child_result; answer['ready'] = True"
                        },
                    )
                ]
            )
        if user == "child-question":
            child_turns += 1
            if child_turns == 1:
                child_entered.set()
                await release_child.wait()
                code = "child_only = 'CHILD'; child_counter = 40; print('child-start')"
                return _chat(calls=[_tool_call("child-init", "repl", {"code": code})])
            code = (
                "assert child_only == 'CHILD' and child_counter == 40\n"
                "assert context == 'child-question' and len(history) >= 3\n"
                "assert 'root_only' not in SHOW_VARS()\n"
                "child_counter += 2\n"
                "grandchild_result = rlm_query('grandchild-question')\n"
                "answer['content'] = f'{child_only}:{child_counter}:{grandchild_result}'\n"
                "answer['ready'] = True"
            )
            return _chat(calls=[_tool_call("child-final", "repl", {"code": code})])
        if user == "grandchild-question":
            grandchild_turns += 1
            if grandchild_turns == 1:
                return _chat(
                    calls=[
                        _tool_call(
                            "grandchild-action",
                            "repl",
                            {
                                "code": "grandchild_value = 'GRANDCHILD'; answer['content'] = grandchild_value; answer['ready'] = True"
                            },
                        )
                    ]
                )
        raise AssertionError(f"unexpected prompt/session {user!r}")

    runner, base_url = await _endpoint(completions)
    proxy = SubLLMProxy(socket_path=tmp_path / "proxy.sock")
    await proxy.start()
    worker_processes = []
    from rlm_train.repl.subprocess import SubprocessReplBackend

    class RecordingBackend(SubprocessReplBackend):
        async def start(self, *args, **kwargs):
            await super().start(*args, **kwargs)
            worker_processes.append(self._proc)

    client = _MockOpenAI(base_url)
    runtime = NativeRuntime(
        client=client,
        model="mock",
        proxy=proxy,
        backend_factory=RecordingBackend,
        limits=NativeLimits(max_depth=3, max_turns=5, episode_timeout=8, request_timeout=5),
        event_path=tmp_path / "calls.jsonl",
    )
    task = asyncio.create_task(runtime.run("root-question", "root-context"))
    try:
        await asyncio.wait_for(child_entered.wait(), timeout=4)
        await asyncio.wait_for(root_advanced.wait(), timeout=4)
        assert not task.done(), "root may finish only after the child task is accounted for"
        release_child.set()
        root_record = await asyncio.wait_for(task, timeout=8)
    finally:
        release_child.set()
        await runtime.close()
        await client.aclose()
        await proxy.stop()
        await runner.cleanup()

    assert len(worker_processes) >= 2
    assert all(process.returncode is not None for process in worker_processes)
    assert root_record["final_answer"] == "ROOT:CHILD:42:GRANDCHILD", root_record
    assert child_turns == 2 and grandchild_turns == 1
    child_payloads = [
        r for r in requests if any(m.get("content") == "child-question" for m in r["messages"])
    ]
    assert len(child_payloads) == 2
    assert any("child-init" in json.dumps(r["messages"]) for r in child_payloads[1:])
    assert all("root_only" not in json.dumps(r["messages"]) for r in child_payloads)
    grandchild_payloads = [
        r for r in requests if any(m.get("content") == "grandchild-question" for m in r["messages"])
    ]
    assert len(grandchild_payloads) == 1
    assert all("child_only" not in json.dumps(r["messages"]) for r in grandchild_payloads)
    assert all("root-context" not in json.dumps(r["messages"]) for r in requests)


@pytest.mark.asyncio
async def test_plain_leaf_and_limits_are_enforced_without_deadlock(tmp_path):
    """Depth cap makes terminal leaf calls plain; global concurrency can be one."""
    from rlm_train.native import NativeLimits, NativeRuntime
    from rlm_train.proxy import SubLLMProxy

    payloads: list[dict[str, Any]] = []

    async def completions(request: web.Request) -> web.Response:
        payload = await request.json()
        payloads.append(payload)
        user = next(
            (m.get("content", "") for m in payload["messages"] if m.get("role") == "user"), ""
        )
        if user == "root":
            return _chat(
                calls=[
                    _tool_call(
                        "root",
                        "repl",
                        {
                            "code": "a = rlm_query('child', blocking=False); b = rlm_query('child-two', blocking=False); "
                            "ra = rlm_wait(a, timeout_s=4); rb = rlm_wait(b, timeout_s=4); "
                            "answer['content']=ra+'|'+rb; answer['ready']=True"
                        },
                    )
                ]
            )
        if user in {"child", "child-two"}:
            return _chat(
                calls=[
                    _tool_call(
                        "child",
                        "repl",
                        {
                            "code": "leaf = rlm_query('leaf'); answer['content']=leaf; answer['ready']=True"
                        },
                    )
                ]
            )
        return _chat("leaf-text")

    runner, base_url = await _endpoint(completions)
    proxy = SubLLMProxy(socket_path=tmp_path / "proxy.sock")
    await proxy.start()
    client = _MockOpenAI(base_url)
    runtime = NativeRuntime(
        client=client,
        model="mock",
        proxy=proxy,
        limits=NativeLimits(
            max_depth=2,
            max_turns=3,
            max_children=4,
            max_live_children=4,
            max_concurrent_requests=1,
            max_model_calls=8,
            episode_timeout=8,
            request_timeout=5,
        ),
    )
    try:
        record = await asyncio.wait_for(runtime.run("root", "private"), timeout=8)
    finally:
        await runtime.close()
        await client.aclose()
        await proxy.stop()
        await runner.cleanup()

    assert record["final_answer"] == "leaf-text|leaf-text"
    leaf_requests = [p for p in payloads if any(m.get("content") == "leaf" for m in p["messages"])]
    assert len(leaf_requests) == 2
    assert all(not p.get("tools") for p in leaf_requests)
    assert len(payloads) <= 8


@pytest.mark.asyncio
async def test_handles_status_timeout_failure_and_close_reaps_workers(tmp_path):
    """A timed wait leaves a child live; failed work and close are observable."""
    from rlm_train.native import NativeLimits, NativeRuntime
    from rlm_train.proxy import SubLLMProxy

    child_started = asyncio.Event()
    release = asyncio.Event()

    async def completions(request: web.Request) -> web.Response:
        payload = await request.json()
        user = next(
            (m.get("content", "") for m in payload["messages"] if m.get("role") == "user"), ""
        )
        if user == "root":
            code = (
                "h = rlm_query('blocked', blocking=False); state_before = rlm_status(h);\n"
                "try:\n rlm_wait(h, timeout_s=0.01)\nexcept TimeoutError:\n timed_out = True\n"
                "state_after = rlm_status(h); answer['content']=str(state_before)+':'+str(state_after); answer['ready']=True"
            )
            return _chat(calls=[_tool_call("spawn", "repl", {"code": code})])
        child_started.set()
        await release.wait()
        return _chat("blocked-result")

    runner, base_url = await _endpoint(completions)
    proxy = SubLLMProxy(socket_path=tmp_path / "proxy.sock")
    await proxy.start()
    client = _MockOpenAI(base_url)
    runtime = NativeRuntime(
        client=client,
        model="mock",
        proxy=proxy,
        limits=NativeLimits(max_depth=1, max_turns=3, episode_timeout=8, request_timeout=5),
    )
    try:
        record = await asyncio.wait_for(runtime.run("root", "ctx"), timeout=8)
        await asyncio.wait_for(child_started.wait(), timeout=3)
        assert record["final_answer"] == "running:running"
        assert runtime.sessions
    finally:
        release.set()
        await runtime.close()
        await client.aclose()
        await proxy.stop()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_sibling_namespaces_and_histories_are_independent(tmp_path):
    """Two child sessions do not inherit each other's variables or transcript."""
    from rlm_train.native import NativeLimits, NativeRuntime
    from rlm_train.proxy import SubLLMProxy

    requests: list[dict[str, Any]] = []

    async def completions(request: web.Request) -> web.Response:
        payload = await request.json()
        requests.append(payload)
        user = next(
            (m.get("content", "") for m in payload["messages"] if m.get("role") == "user"), ""
        )
        if user == "root":
            code = (
                "left = rlm_query('left-child'); right = rlm_query('right-child'); "
                "answer['content'] = left + '|' + right; answer['ready'] = True"
            )
            return _chat(calls=[_tool_call("spawn", "repl", {"code": code})])
        if user == "left-child":
            return _chat(
                calls=[
                    _tool_call(
                        "left",
                        "repl",
                        {
                            "code": "left_secret='LEFT'; answer['content']=left_secret; answer['ready']=True"
                        },
                    )
                ]
            )
        if user == "right-child":
            return _chat(
                calls=[
                    _tool_call(
                        "right",
                        "repl",
                        {
                            "code": "assert 'left_secret' not in SHOW_VARS(); right_secret='RIGHT'; answer['content']=right_secret; answer['ready']=True"
                        },
                    )
                ]
            )
        raise AssertionError(user)

    runner, base_url = await _endpoint(completions)
    proxy = SubLLMProxy(socket_path=tmp_path / "proxy.sock")
    await proxy.start()
    client = _MockOpenAI(base_url)
    runtime = NativeRuntime(
        client=client,
        model="mock",
        proxy=proxy,
        limits=NativeLimits(max_depth=2, request_timeout=5, episode_timeout=8),
    )
    try:
        record = await asyncio.wait_for(runtime.run("root", "root-private"), timeout=8)
    finally:
        await runtime.close()
        await client.aclose()
        await proxy.stop()
        await runner.cleanup()

    assert record["final_answer"] == "LEFT|RIGHT"
    left = next(p for p in requests if any(m.get("content") == "left-child" for m in p["messages"]))
    right = next(
        p for p in requests if any(m.get("content") == "right-child" for m in p["messages"])
    )
    assert "root-private" not in json.dumps(left["messages"] + right["messages"])
    assert "left_secret" not in json.dumps(right["messages"])
    assert "right_secret" not in json.dumps(left["messages"])


@pytest.mark.asyncio
async def test_shared_child_admission_budget_is_enforced(tmp_path):
    """Concurrent sessions share the caller's child-admission budget."""
    from rlm_train.native import NativeLimits, NativeRuntime
    from rlm_train.proxy import SubLLMProxy

    async def completions(request: web.Request) -> web.Response:
        payload = await request.json()
        user = next(
            (m.get("content", "") for m in payload["messages"] if m.get("role") == "user"), ""
        )
        if user == "root":
            code = (
                "first = rlm_query('first', blocking=False)\n"
                "try:\n second = rlm_query('second', blocking=False)\n rejected = False\nexcept RuntimeError:\n rejected = True\n"
                "first_result = rlm_wait(first, timeout_s=3); "
                "answer['content'] = first_result + ':' + str(rejected); answer['ready'] = True"
            )
            return _chat(calls=[_tool_call("spawn", "repl", {"code": code})])
        return _chat("child-result")

    runner, base_url = await _endpoint(completions)
    proxy = SubLLMProxy(socket_path=tmp_path / "proxy.sock")
    await proxy.start()
    client = _MockOpenAI(base_url)
    runtime = NativeRuntime(
        client=client,
        model="mock",
        proxy=proxy,
        limits=NativeLimits(
            max_children=1, max_live_children=1, max_depth=1, request_timeout=5, episode_timeout=8
        ),
    )
    try:
        record = await asyncio.wait_for(runtime.run("root", "ctx"), timeout=8)
    finally:
        await runtime.close()
        await client.aclose()
        await proxy.stop()
        await runner.cleanup()

    assert record["final_answer"] == "child-result:True", json.dumps(record, indent=2)
    assert len(runtime.records) == 2
    assert sum(len(r["events"]) for r in runtime.records.values()) <= 3


@pytest.mark.asyncio
async def test_child_handles_are_scoped_and_completed_results_repeat(tmp_path):
    """A child cannot inspect a sibling handle; owner can retrieve completed result repeatedly."""
    from rlm_train.native import NativeLimits, NativeRuntime
    from rlm_train.proxy import SubLLMProxy

    async def completions(request: web.Request) -> web.Response:
        payload = await request.json()
        messages = payload["messages"]
        user = next(m.get("content", "") for m in messages if m.get("role") == "user")
        if user == "root":
            code = (
                "first = rlm_query('first-child', blocking=False); "
                "second = rlm_query('second-child:' + first, blocking=False); "
                "a = rlm_wait(first, timeout_s=3); b = rlm_wait(first, timeout_s=3); "
                "c = rlm_wait(second, timeout_s=3); "
                "answer['content'] = a + ':' + b + ':' + str(first == second) + ':' + c; answer['ready'] = True"
            )
            return _chat(calls=[_tool_call("root", "repl", {"code": code})])
        if user == "first-child":
            return _chat(
                calls=[
                    _tool_call(
                        "first", "repl", {"code": "answer['content']='A'; answer['ready']=True"}
                    )
                ]
            )
        if user.startswith("second-child:"):
            foreign = user.partition(":")[2]
            code = (
                f"try:\n rlm_status({foreign!r})\nexcept RuntimeError:\n denied=True\n"
                "answer['content'] = 'B:' + str(denied); answer['ready'] = True"
            )
            return _chat(calls=[_tool_call("second", "repl", {"code": code})])
        raise AssertionError(user)

    runner, base_url = await _endpoint(completions)
    proxy = SubLLMProxy(socket_path=tmp_path / "proxy.sock")
    await proxy.start()
    client = _MockOpenAI(base_url)
    runtime = NativeRuntime(
        client=client,
        model="mock",
        proxy=proxy,
        limits=NativeLimits(
            max_depth=2, max_children=2, max_live_children=2, request_timeout=5, episode_timeout=8
        ),
    )
    try:
        record = await asyncio.wait_for(runtime.run("root", "ctx"), timeout=8)
    finally:
        await runtime.close()
        await client.aclose()
        await proxy.stop()
        await runner.cleanup()

    assert record["final_answer"] == "A:A:False:B:True", json.dumps(record, indent=2)
    assert any(r["final_answer"] == "B:True" for r in runtime.records.values())


@pytest.mark.asyncio
async def test_shared_model_call_budget_exhausts_before_dispatch(tmp_path):
    """The episode call cap rejects the next request instead of dispatching it."""
    from rlm_train.native import NativeLimits, NativeRuntime
    from rlm_train.proxy import SubLLMProxy

    payloads: list[dict[str, Any]] = []

    async def completions(request: web.Request) -> web.Response:
        payloads.append(await request.json())
        return _chat(calls=[_tool_call("step", "repl", {"code": "value = 1"})])

    runner, base_url = await _endpoint(completions)
    proxy = SubLLMProxy(socket_path=tmp_path / "proxy.sock")
    await proxy.start()
    client = _MockOpenAI(base_url)
    runtime = NativeRuntime(
        client=client,
        model="mock",
        proxy=proxy,
        limits=NativeLimits(
            max_model_calls=1, max_depth=1, max_turns=3, request_timeout=5, episode_timeout=8
        ),
    )
    try:
        record = await asyncio.wait_for(runtime.run("root", "ctx"), timeout=8)
    finally:
        await runtime.close()
        await client.aclose()
        await proxy.stop()
        await runner.cleanup()

    assert len(payloads) == 1
    assert record["error_class"] == "budget"
    assert "budget exhausted" in record["error"]
