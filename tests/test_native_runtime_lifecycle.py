"""Native runtime limits and cancellation contracts without external inference."""

import asyncio
from types import SimpleNamespace

import pytest
from rlm_train.native import NativeCleanupError, NativeLimits, NativeRuntime
from rlm_train.proxy import SubLLMProxy


class ScriptedClient:
    def __init__(self, code):
        self.code = code
        self.calls = []

    async def post(self, path, *, json):
        self.calls.append(json)
        raw = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "tool",
                                "function": {"name": "repl", "arguments": {"code": self.code}},
                            }
                        ],
                    },
                }
            ]
        }
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: raw)


class BlockingStopBackend:
    def __init__(self, entered, release, **kwargs):
        self.entered, self.release = entered, release
        self.stopped = False

    async def start(self, *args, **kwargs):
        return None

    async def load_context(self, context):
        return None

    async def update_history(self, messages):
        return None

    async def execute(self, code):
        return SimpleNamespace(stdout="", stderr="", final_answer="ok")

    async def stop(self):
        self.entered.set()
        await self.release.wait()
        self.stopped = True


@pytest.mark.asyncio
async def test_repeated_cancellation_close_joins_cleanup():
    entered, release = asyncio.Event(), asyncio.Event()
    backends = []

    def factory(**kwargs):
        backend = BlockingStopBackend(entered, release, **kwargs)
        backends.append(backend)
        return backend

    runtime = NativeRuntime(
        ScriptedClient("finish"), "mock", SubLLMProxy(), backend_factory=factory
    )
    run = asyncio.create_task(runtime.run("question", "context"))
    await entered.wait()
    root = runtime.root
    root.task.cancel()
    await asyncio.gather(root.task, return_exceptions=True)
    closer = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)
    closer.cancel()
    await asyncio.gather(closer, return_exceptions=True)
    assert not backends[0].stopped
    assert not runtime.close_task.done()
    release.set()
    await runtime.close()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert backends[0].stopped
    assert root.cleanup_task.done()


@pytest.mark.asyncio
async def test_batch_admission_failure_dispatches_no_orphan_child():
    from rlm_train.native import _ChildService

    runtime = NativeRuntime(
        ScriptedClient("unused"), "mock", SubLLMProxy(), limits=NativeLimits(max_children=1)
    )
    runtime.started = __import__("time").monotonic()
    runtime.deadline = runtime.started + 10
    root = runtime.new_session(None, "root", "context", "mock")
    # Admission is synchronous until the service either awaits work or rolls back.
    with pytest.raises(RuntimeError, match="admission budget"):
        await _ChildService(runtime, root.id).query_batched(["first", "second"])
    children = [r for r in runtime.records.values() if r["parent_session_id"]]
    assert len(children) == 1
    assert children[0]["status"] == "failed"
    assert children[0]["events"] == []
    await runtime.close()


@pytest.mark.parametrize(
    "config",
    [
        {"max_depth": 1.5},
        {"max_concurrent_requests": 0},
        {"max_turns": True},
        {"episode_timeout": float("nan")},
        {"request_timeout": float("inf")},
    ],
)
def test_limits_reject_invalid_values(config):
    with pytest.raises(ValueError):
        NativeLimits(**config)


@pytest.mark.asyncio
async def test_cleanup_failure_blocks_success_and_allows_retry():
    class FailedStop(BlockingStopBackend):
        async def stop(self):
            if not self.release.is_set():
                raise RuntimeError("exit unconfirmed")
            self.stopped = True

    entered, release = asyncio.Event(), asyncio.Event()
    runtime = NativeRuntime(
        ScriptedClient("finish"),
        "mock",
        SubLLMProxy(),
        backend_factory=lambda **kw: FailedStop(entered, release, **kw),
    )
    with pytest.raises(NativeCleanupError):
        await runtime.run("question", "context")
    assert runtime.root.record["error_class"] == "cleanup"
    assert runtime.root.record["final_answer"] == "ok"
    release.set()
    await runtime.close()
    assert runtime.root.backend.stopped


@pytest.mark.asyncio
async def test_retry_cleanup_descendants_before_parent():
    class RetriableBackend(BlockingStopBackend):
        def __init__(self):
            self.attempts = 0
            self.stopped = False

        async def stop(self):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("transient exit check failure")
            self.stopped = True

    runtime = NativeRuntime(ScriptedClient("unused"), "mock", SubLLMProxy())
    root = runtime.new_session(None, "root", "context", "mock")
    child = runtime.new_session(root, "child", "child", "mock")
    for session in (root, child):
        session.task.cancel()
    await asyncio.gather(root.task, child.task, return_exceptions=True)
    child.backend = RetriableBackend()
    await runtime.cleanup(child)
    assert child.record.get("cleanup_error")
    await runtime.cleanup(root)
    assert root.record.get("cleanup_error")
    await runtime.close()
    assert child.backend.stopped
    assert not root.record.get("cleanup_error")
    assert not child.record.get("cleanup_error")


@pytest.mark.asyncio
async def test_model_call_budget_stops_before_second_dispatch():
    class ContinueBackend(BlockingStopBackend):
        async def execute(self, code):
            return SimpleNamespace(stdout="continue", stderr="", final_answer=None)

    entered, release = asyncio.Event(), asyncio.Event()
    release.set()
    client = ScriptedClient("continue")
    runtime = NativeRuntime(
        client,
        "mock",
        SubLLMProxy(),
        limits=NativeLimits(max_model_calls=1),
        backend_factory=lambda **kw: ContinueBackend(entered, release, **kw),
    )
    record = await runtime.run("question", "context")
    assert record["error_class"] == "budget"
    assert len(client.calls) == 1
    assert len(record["events"]) == 1
    assert record["events"][0]["request"]["max_tokens"] == 8192
    assert record["events"][0]["token_limits"] == {
        "output_tokens": {"limit": 8192, "source": "request.max_tokens"},
        "context_tokens": {"limit": None, "source": None},
    }
    assert runtime.root.backend.stopped
