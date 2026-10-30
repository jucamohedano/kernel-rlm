from __future__ import annotations

import pytest
from rlm_train.proxy import ClientHandle, SubLLMProxy
from rlm_train.worker import Worker


class RecursiveService:
    async def query(self, prompt, model=None, *, blocking=True):
        return f"{model}:{prompt}:{blocking}"

    async def query_batched(self, prompts, model=None):
        return [f"{model}:{prompt}" for prompt in prompts]

    async def status(self, handle):
        return f"status:{handle}"

    async def result(self, handle):
        return f"result:{handle}"

    async def wait(self, handle, timeout_s=30):
        return f"wait:{handle}:{timeout_s}"


@pytest.mark.asyncio
async def test_recursive_proxy_routes_and_missing_service():
    proxy = SubLLMProxy()
    await proxy.start()
    proxy.register("r", ClientHandle(client=None, model="default", recursive=RecursiveService()))
    try:
        async with __import__("aiohttp").ClientSession() as session:
            base = proxy.url + "/rollout/r/"
            calls = [
                (
                    "rlm_query",
                    {"prompt": "p", "model": "m", "blocking": False},
                    {"response": "m:p:False"},
                ),
                (
                    "rlm_query_batched",
                    {"prompts": ["a", "b"], "model": "m"},
                    {"responses": ["m:a", "m:b"]},
                ),
                ("rlm_status", {"handle": "h"}, {"response": "status:h"}),
                ("rlm_result", {"handle": "h"}, {"response": "result:h"}),
                ("rlm_wait", {"handle": "h", "timeout_s": 2}, {"response": "wait:h:2"}),
            ]
            for route, body, expected in calls:
                async with session.post(base + route, json=body) as response:
                    assert await response.json() == expected
        proxy.register("none", ClientHandle(client=None, model="default"))
        async with __import__("aiohttp").ClientSession() as session:
            async with session.post(
                proxy.url + "/rollout/none/rlm_query", json={"prompt": "p"}
            ) as response:
                assert (await response.json())["error_type"] == "RuntimeError"
    finally:
        await proxy.stop()


def test_worker_recursive_helpers_error_types_and_history_restore():
    worker = Worker("http://unused", "r")
    calls = []

    def post(route, payload):
        calls.append((route, payload))
        if route == "rlm_wait":
            return {"error": "timed out", "error_type": "TimeoutError"}
        return {"response": "ok"}

    worker._proxy_post = post
    assert worker._rlm_query("p", "m", blocking=False) == "ok"
    assert calls[-1] == ("rlm_query", {"prompt": "p", "model": "m", "blocking": False})
    with pytest.raises(TimeoutError, match="timed out"):
        worker._rlm_wait("h")
    worker.update_history([{"role": "user", "content": "x"}])
    worker.execute("history = 7; rlm_query = 8")
    assert isinstance(worker.locals["history"], tuple)
    assert worker.globals["rlm_query"] == worker._rlm_query
