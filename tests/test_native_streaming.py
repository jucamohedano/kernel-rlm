"""Deterministic SSE assembly and durable-before-parse callback contracts."""

import asyncio
import json

import httpx
import pytest
from rlm_train.streaming import (
    StreamInterruptedError,
    StreamProtocolError,
    stream_chat_completion,
)


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks, error=None):
        self.chunks = chunks
        self.error = error
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error

    async def aclose(self):
        self.closed = True


def event(data):
    return ("data: " + json.dumps(data, ensure_ascii=False) + "\r\n\r\n").encode()


def choice(delta, *, index=0, finish=None, logprobs=None):
    return {
        "choices": [{"index": index, "delta": delta, "finish_reason": finish, "logprobs": logprobs}]
    }


def client_for(stream, *, status=200, requests=None):
    def handle(request):
        if requests is not None:
            requests.append(request)
        return httpx.Response(status, stream=stream)

    return httpx.AsyncClient(
        base_url="https://example.test/v1/", transport=httpx.MockTransport(handle)
    )


@pytest.mark.asyncio
async def test_fragmented_stream_preserves_reasoning_tools_logprobs_and_usage():
    metadata = {
        "id": "chat-1",
        "object": "chat.completion.chunk",
        "created": 123,
        "model": "m",
        "system_fingerprint": "fp",
    }
    selected = {"token": "hé", "logprob": -0.1, "bytes": [104, 195, 169], "top_logprobs": []}
    first = choice(
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "think ",
            "tool_calls": [
                {
                    "index": 1,
                    "id": "call-b",
                    "type": "function",
                    "function": {"name": "re", "arguments": '{"code":'},
                },
                {
                    "index": 0,
                    "id": "call-a",
                    "type": "function",
                    "function": {"name": "repl", "arguments": '{"code":"'},
                },
            ],
        },
        logprobs={"content": [selected], "refusal": None},
    )
    second = choice(
        {
            "content": "hé",
            "reasoning_content": "more",
            "tool_calls": [
                {"index": 0, "function": {"name": None, "arguments": 'one"}'}},
                {"index": 1, "function": {"name": "pl", "arguments": '"two"}'}},
            ],
        },
        logprobs={"content": [{"token": "!", "logprob": -0.2}]},
    )
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 6,
        "total_tokens": 16,
        "completion_tokens_details": {"reasoning_tokens": 3},
    }
    raw = (
        b": keepalive\r\n\r\n"
        + event({**metadata, **first})
        + event({**metadata, **second})
        + event({**metadata, **choice({"content": "!"}, finish="tool_calls")})
        + event({**metadata, "choices": [], "usage": usage})
        + b"data: [DONE]\r\n\r\n"
    )
    # One-byte fragments split CRLF, UTF-8, JSON and the completion marker.
    chunks = [raw[index : index + 1] for index in range(len(raw))]
    stream = Chunks(chunks)
    captured, progress, requests = [], [], []

    def update(patch):
        # Parsed progress can only refer to bytes already persisted by on_chunk.
        if "last_delta" in patch:
            assert event(patch["last_delta"]) in b"".join(captured)
        progress.append(patch)

    payload = {"model": "m", "stream": False, "stream_options": {"custom": True}}
    async with client_for(stream, requests=requests) as client:
        result = await stream_chat_completion(
            client, payload, on_chunk=captured.append, on_progress=update
        )
    assert captured == chunks and stream.closed
    assert payload == {"model": "m", "stream": False, "stream_options": {"custom": True}}
    sent = json.loads(requests[0].content)
    assert sent == {
        "model": "m",
        "stream": True,
        "stream_options": {"custom": True, "include_usage": True},
    }
    assert requests[0].url.path == "/v1/chat/completions"
    assert requests[0].headers["accept-encoding"] == "identity"
    assert result == {
        **metadata,
        "object": "chat.completion",
        "usage": usage,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hé!",
                    "reasoning_content": "think more",
                    "tool_calls": [
                        {
                            "id": "call-a",
                            "type": "function",
                            "function": {"name": "repl", "arguments": '{"code":"one"}'},
                        },
                        {
                            "id": "call-b",
                            "type": "function",
                            "function": {"name": "repl", "arguments": '{"code":"two"}'},
                        },
                    ],
                },
                "finish_reason": "tool_calls",
                "logprobs": {
                    "content": [selected, {"token": "!", "logprob": -0.2}],
                    "refusal": None,
                },
            }
        ],
    }
    assert progress[0] == {"stage": "headers", "http_status": 200}
    assert progress[-1] == {"stage": "done", "done": True}
    assert any(patch.get("finish") == {0: "tool_calls"} for patch in progress)
    assert max(patch.get("bytes_received", 0) for patch in progress) == len(raw)


@pytest.mark.asyncio
async def test_multiple_choices_multiline_data_and_reasoning_field():
    first = b'data: {"choices": [\r\ndata: {"index": 1, "delta": {"reasoning": "why"}, "finish_reason": null}]}\r\n\r\n'
    raw = (
        first
        + event(choice({"content": "answer"}, finish="stop"))
        + event(choice({"content": "second"}, index=1, finish="length"))
        + b"data: [DONE]\n\n"
    )
    async with client_for(Chunks([raw])) as client:
        result = await stream_chat_completion(client, {}, on_chunk=lambda chunk: None)
    assert [item["index"] for item in result["choices"]] == [0, 1]
    assert result["choices"][1]["finish_reason"] == "length"
    assert result["choices"][1]["message"] == {
        "role": "assistant",
        "content": "second",
        "reasoning": "why",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        event(choice({"content": "partial"})),
        event(choice({}, finish="stop")),
        event(choice({"content": "partial"})) + b"data: [DONE]\n\n",
        event(choice({}, finish="stop")) + b"data: [DONE]",
    ],
)
async def test_incomplete_stream_is_interrupted(raw):
    captured = []
    async with client_for(Chunks([raw])) as client:
        with pytest.raises(StreamInterruptedError):
            await stream_chat_completion(client, {}, on_chunk=captured.append)
    assert captured == [raw]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        b"data: not-json\n\n",
        b"data: \xff\n\n",
        event([]),
        event({"choices": {}}),
        event({"error": {"message": "provider failure"}}),
        event({"choices": [{"index": 0, "delta": []}]}),
        event(choice({"content": ["not-text"]})),
        event(choice({"tool_calls": "invalid"})),
        event(choice({"tool_calls": [{"index": 0, "function": {"arguments": {}}}]})),
        event(choice({}, index=-1)),
        event(choice({}, finish=7)),
        event(choice({}, finish="stop")) + event(choice({"content": "too late"})),
        b"data: [DONE]\n\n",
    ],
)
async def test_malformed_stream_is_not_recoverable_and_evidence_precedes_error(raw):
    captured = []
    stream = Chunks([raw])
    async with client_for(stream) as client:
        with pytest.raises(StreamProtocolError) as caught:
            await stream_chat_completion(client, {}, on_chunk=captured.append)
    assert not isinstance(caught.value, StreamInterruptedError)
    assert captured == [raw] and stream.closed


@pytest.mark.asyncio
async def test_rejects_nonidentity_content_encoding_before_parsing():
    stream = Chunks([b"compressed bytes"])
    progress = []
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, stream=stream, headers={"Content-Encoding": "gzip"})
    )
    async with httpx.AsyncClient(
        base_url="https://example.test/v1/", transport=transport
    ) as client:
        with pytest.raises(StreamProtocolError, match="Content-Encoding: gzip"):
            await stream_chat_completion(
                client, {}, on_chunk=lambda chunk: None, on_progress=progress.append
            )
    assert stream.closed
    assert progress == [{"stage": "headers", "http_status": 200}]


@pytest.mark.asyncio
async def test_http_error_status_and_raw_body_are_captured_before_raise():
    captured, progress = [], []
    stream = Chunks([b'{"error":', b'"unavailable"}'])
    async with client_for(stream, status=503) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await stream_chat_completion(
                client, {}, on_chunk=captured.append, on_progress=progress.append
            )
    assert captured == stream.chunks
    assert progress[0] == {"stage": "headers", "http_status": 503}
    assert stream.closed


@pytest.mark.asyncio
async def test_transport_failure_preserves_partial_reasoning_and_raw_bytes():
    raw = event(choice({"reasoning_content": "partial thought"}))
    failure = httpx.ReadError("connection reset")
    captured, progress = [], []
    stream = Chunks([raw], error=failure)
    async with client_for(stream) as client:
        with pytest.raises(httpx.ReadError) as caught:
            await stream_chat_completion(
                client, {}, on_chunk=captured.append, on_progress=progress.append
            )
    assert caught.value is failure
    assert captured == [raw] and stream.closed
    assert progress[-1]["last_delta"] == choice({"reasoning_content": "partial thought"})


@pytest.mark.asyncio
async def test_callback_failure_stops_before_parsing():
    failure = OSError("journal full")
    progress = []

    def cannot_persist(chunk):
        raise failure

    stream = Chunks([b"data: invalid-json\n\n"])
    async with client_for(stream) as client:
        with pytest.raises(OSError) as caught:
            await stream_chat_completion(
                client, {}, on_chunk=cannot_persist, on_progress=progress.append
            )
    assert caught.value is failure and stream.closed
    assert progress == [{"stage": "headers", "http_status": 200}]


@pytest.mark.asyncio
async def test_cancellation_closes_response_after_preserving_partial_arguments():
    waiting = asyncio.Event()
    raw = event(
        choice({"tool_calls": [{"index": 0, "function": {"arguments": '{"code":"partial'}}]})
    )

    class BlockingStream(Chunks):
        async def __aiter__(self):
            yield raw
            waiting.set()
            await asyncio.Event().wait()

    stream = BlockingStream([])
    captured, progress = [], []
    async with client_for(stream) as client:
        task = asyncio.create_task(
            stream_chat_completion(
                client, {}, on_chunk=captured.append, on_progress=progress.append
            )
        )
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert captured == [raw] and stream.closed
    assert (
        progress[-1]["last_delta"]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
        == '{"code":"partial'
    )


@pytest.mark.asyncio
async def test_tool_arguments_are_not_parsed_even_when_truncated_by_length():
    raw = (
        event(
            choice(
                {
                    "tool_calls": [
                        {"index": 0, "function": {"name": "repl", "arguments": '{"code":'}}
                    ]
                },
                finish="length",
            )
        )
        + b"data: [DONE]\n\n"
    )
    async with client_for(Chunks([raw])) as client:
        result = await stream_chat_completion(client, {}, on_chunk=lambda chunk: None)
    assert result["choices"][0]["finish_reason"] == "length"
    assert result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"code":'
