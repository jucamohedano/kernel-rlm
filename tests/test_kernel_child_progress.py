import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import aiohttp
import pytest
from aiohttp import web


@pytest.fixture(autouse=True)
def packages(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "training/environments/kernel_rlm"))


def stream_events(tokens=3):
    entries = [{"token": f"token_id:{index + 10}", "logprob": -0.5} for index in range(tokens)]
    events = [
        {"prompt_token_ids": [1, 2], "choices": [{"index": 0, "delta": {"role": "assistant"}}]},
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning": "private thinking"},
                    "token_ids": [10],
                    "logprobs": {"content": entries[:1]},
                }
            ]
        },
    ]
    for start in range(1, tokens, 256):
        end = min(start + 256, tokens)
        events.append(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Check boundaries."},
                        "token_ids": list(range(10 + start, 10 + end)),
                        "logprobs": {"content": entries[start:end]},
                        "finish_reason": "stop" if end == tokens else None,
                    }
                ]
            }
        )
    return events + [
        {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": tokens}},
        "[DONE]",
    ]


def encoded_event(event):
    return ("data: " + (event if isinstance(event, str) else json.dumps(event)) + "\n\n").encode()


@asynccontextmanager
async def serve(app):
    runner = web.AppRunner(app, handler_cancellation=True)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    try:
        yield f"http://127.0.0.1:{runner.addresses[0][1]}"
    finally:
        await runner.cleanup()


def test_child_progress_reaches_status_before_completion_and_never_enters_loss(tmp_path):
    from kernel_rlm.subcalls import CHILD_SYSTEM
    from kernel_skyrl.bridge import CompletionBridge
    from kernel_trl.collect import InferenceClient

    async def check():
        release = asyncio.Event()
        emitted = asyncio.Event()
        requests = []

        async def tokenize(request):
            return web.json_response({"count": 2, "tokens": [1, 2]})

        async def completion(request):
            body = await request.json()
            requests.append(body)
            assert body["stream"] is True
            assert body["stream_options"] == {"include_usage": True}
            assert body["chat_template_kwargs"]["enable_thinking"] is True
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            for index, event in enumerate(stream_events(4096)):
                await response.write(encoded_event(event))
                if index == 1:
                    emitted.set()
                    await release.wait()
            return response

        app = web.Application()
        app.router.add_post("/tokenize", tokenize)
        app.router.add_post("/v1/chat/completions", completion)
        async with serve(app) as endpoint, aiohttp.ClientSession() as session:
            client = InferenceClient(session, endpoint, tmp_path / "requests.jsonl", 14336)
            bridge = CompletionBridge(client, "policy", 14336, 512, 16, tmp_path / "calls.jsonl")
            await bridge.start()
            headers = {"Authorization": f"Bearer {bridge.api_key}"}
            pending = None
            try:
                pending = asyncio.create_task(
                    session.post(
                        bridge.endpoint + "/chat/completions",
                        headers=headers,
                        json={
                            "messages": [{"role": "system", "content": CHILD_SYSTEM}],
                            "max_tokens": 8192,
                            "chat_template_kwargs": {"enable_thinking": True},
                        },
                    )
                )
                await asyncio.wait_for(emitted.wait(), 2)
                async with session.get(bridge.endpoint + "/child-progress") as response:
                    assert response.status == 401
                async with session.get(
                    bridge.endpoint + "/child-progress", headers=headers
                ) as response:
                    snapshot = await response.json()
                child = snapshot["children"][0]
                assert child["stage"] == "generating"
                assert child["completion_tokens"] == 1
                assert child["token_progress_available"] is True
                assert not pending.done()
                release.set()
                response = await asyncio.wait_for(pending, 2)
                assert response.status == 200
                result = await response.json()
                assert result["choices"][0]["message"]["content"].startswith("Check boundaries.")
                assert len(result["choices"][0]["token_ids"]) == 4096
                assert requests[0]["max_tokens"] == 4096
                assert not bridge.samples
                assert not bridge.failures
                assert bridge.child_calls == 1
                events = [
                    json.loads(line)
                    for line in (tmp_path / "child-progress.jsonl").read_text().splitlines()
                ]
                assert [event["stage"] for event in events[:3]] == [
                    "queued",
                    "admitted",
                    "generating",
                ]
                assert all(event["stage"] == "generating" for event in events[3:-2])
                assert [event["stage"] for event in events[-2:]] == ["generated", "completed"]
                assert events[-1]["completion_tokens"] == 4096
                assert "private thinking" not in json.dumps(events)
                assert "Check boundaries" not in json.dumps(events)
                assert bridge.api_key not in json.dumps(events)
            finally:
                release.set()
                if pending is not None and not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
                await bridge.stop()

    asyncio.run(check())


@pytest.mark.parametrize(
    "failure", ["disconnect", "usage", "alignment", "prompt", "cap", "tool", "logprob", "error"]
)
def test_incomplete_or_invalid_child_stream_cannot_be_a_completion(failure):
    from kernel_trl.child_stream import read_child_stream

    events = stream_events()
    if failure == "disconnect":
        events.pop()
    elif failure == "usage":
        events[-2]["usage"]["completion_tokens"] = 99
    elif failure == "alignment":
        events[2]["choices"][0]["logprobs"]["content"].pop()
    elif failure == "prompt":
        events[0]["prompt_token_ids"] = [9]
    elif failure == "tool":
        events[2]["choices"][0]["delta"]["tool_calls"] = [{"id": "forbidden"}]
    elif failure == "logprob":
        events[1]["choices"][0]["logprobs"]["content"][0]["token"] = "token_id:99"
    elif failure == "error":
        events[2] = {"error": {"message": "backend fault"}}

    async def lines():
        for event in events:
            yield encoded_event(event)

    with pytest.raises((ValueError, RuntimeError)):
        asyncio.run(
            read_child_stream(
                SimpleNamespace(content=lines()),
                [1, 2],
                2 if failure == "cap" else 4096,
                14336,
                lambda event: None,
            )
        )


def test_child_stream_cancellation_closes_http_response_and_keeps_partial_counts(tmp_path):
    from kernel_trl.collect import InferenceClient

    async def check():
        active, closed = asyncio.Event(), asyncio.Event()
        progress = []

        async def lines():
            yield encoded_event(stream_events()[0])
            yield encoded_event(stream_events()[1])
            active.set()
            await asyncio.Event().wait()

        class Response:
            status = 200
            content = lines()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                closed.set()

        class TokenResponse(Response):
            def raise_for_status(self):
                pass

            async def json(self):
                return {"count": 2, "tokens": [1, 2]}

        session = SimpleNamespace(post=Mock(side_effect=[TokenResponse(), Response()]))
        client = InferenceClient(session, "http://unused", tmp_path / "requests")
        pending = asyncio.create_task(
            client.chat_completion_with_progress(
                {"json": {"model": "policy", "max_tokens": 4096}}, progress.append
            )
        )
        await asyncio.wait_for(active.wait(), 1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert closed.is_set()
        assert progress[-1]["stage"] == "generating"
        assert progress[-1]["completion_tokens"] == 1
        assert not (tmp_path / "requests").exists()

    asyncio.run(check())
