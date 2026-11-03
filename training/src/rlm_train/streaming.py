"""Capture raw SSE bytes before assembling complete OpenAI chat responses."""

from __future__ import annotations

import codecs
import copy
import json
from collections.abc import Callable
from typing import Any

import httpx


class StreamProtocolError(RuntimeError):
    """The provider sent an invalid chat completion stream."""


class StreamInterruptedError(StreamProtocolError):
    """The stream ended before a complete response was received."""


class SSEDecoder:
    def __init__(self) -> None:
        self.decoder = codecs.getincrementaldecoder("utf-8-sig")()
        self.buffer = ""
        self.data: list[str] = []

    def feed(self, chunk: bytes) -> list[str]:
        try:
            self.buffer += self.decoder.decode(chunk)
        except UnicodeDecodeError as error:
            raise StreamProtocolError("invalid UTF-8 in SSE stream") from error
        lines = self.buffer.split("\n")
        self.buffer = lines.pop()
        events = []
        for line in lines:
            line = line.removesuffix("\r")
            if not line:
                if self.data:
                    events.append("\n".join(self.data))
                    self.data.clear()
                continue
            field, _, value = line.partition(":")
            if field == "data":
                self.data.append(value.removeprefix(" "))
        return events


def merge_delta(target: dict, delta: dict) -> None:
    """Append streamed text and lists without interpreting generated arguments."""
    for key, value in delta.items():
        if value is None:
            target.setdefault(key, None)
        elif key not in target or target[key] is None:
            target[key] = copy.deepcopy(value)
        elif isinstance(value, str) and isinstance(target[key], str):
            target[key] += value
        elif isinstance(value, list) and isinstance(target[key], list):
            target[key].extend(copy.deepcopy(value))
        elif isinstance(value, dict) and isinstance(target[key], dict):
            merge_delta(target[key], value)
        elif target[key] != value:
            raise StreamProtocolError(f"conflicting delta field: {key}")


def stream_index(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise StreamProtocolError(f"{field} must be a nonnegative integer")
    return value


class ChatCompletionAccumulator:
    def __init__(self) -> None:
        self.response: dict[str, Any] = {"object": "chat.completion"}
        self.choices: dict[int, dict] = {}
        self.tools: dict[int, dict[int, dict]] = {}

    def add(self, chunk: dict) -> dict[int, str]:
        if "error" in chunk:
            raise StreamProtocolError(f"provider stream error: {chunk['error']}")
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            raise StreamProtocolError("stream chunk must contain a choices array")
        for key, value in chunk.items():
            if key not in {"object", "choices", "usage"}:
                self.response[key] = copy.deepcopy(value)
        if chunk.get("usage") is not None:
            if not isinstance(chunk["usage"], dict):
                raise StreamProtocolError("stream usage must be an object")
            self.response["usage"] = copy.deepcopy(chunk["usage"])
        finished = {}
        for incoming in choices:
            if not isinstance(incoming, dict):
                raise StreamProtocolError("stream choice must be an object")
            index = stream_index(incoming.get("index"), "choice index")
            delta = incoming.get("delta")
            if not isinstance(delta, dict):
                raise StreamProtocolError("stream choice delta must be an object")
            choice = self.choices.setdefault(
                index,
                {
                    "index": index,
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": None,
                    "logprobs": None,
                },
            )
            if choice["finish_reason"] is not None and delta:
                raise StreamProtocolError("received a delta after choice finished")
            self.add_message(index, choice["message"], delta)
            logprobs = incoming.get("logprobs")
            if logprobs is not None:
                if not isinstance(logprobs, dict):
                    raise StreamProtocolError("stream logprobs must be an object")
                if choice["logprobs"] is None:
                    choice["logprobs"] = {}
                merge_delta(choice["logprobs"], logprobs)
            finish = incoming.get("finish_reason")
            if finish is not None:
                if not isinstance(finish, str) or not finish:
                    raise StreamProtocolError("finish_reason must be a nonempty string")
                if choice["finish_reason"] not in (None, finish):
                    raise StreamProtocolError("conflicting choice finish reasons")
                choice["finish_reason"] = finish
                finished[index] = finish
            for key, value in incoming.items():
                if key not in {"index", "delta", "finish_reason", "logprobs"}:
                    choice[key] = copy.deepcopy(value)
        return finished

    def add_message(self, index: int, message: dict, delta: dict) -> None:
        delta = delta.copy()
        if "role" in delta:
            role = delta.pop("role")
            if role is not None:
                if not isinstance(role, str) or role != message["role"]:
                    raise StreamProtocolError("stream message role must be assistant")
        for key in ("content", "reasoning", "reasoning_content", "refusal"):
            if delta.get(key) is not None and not isinstance(delta[key], str):
                raise StreamProtocolError(f"stream {key} must be text or null")
        if "tool_calls" in delta:
            calls = delta.pop("tool_calls")
            if calls is not None:
                if not isinstance(calls, list):
                    raise StreamProtocolError("stream tool_calls must be an array")
                tools = self.tools.setdefault(index, {})
                for call in calls:
                    if not isinstance(call, dict):
                        raise StreamProtocolError("stream tool call must be an object")
                    tool_index = stream_index(call.get("index"), "tool index")
                    tool = tools.setdefault(tool_index, {})
                    fragment = {key: value for key, value in call.items() if key != "index"}
                    for key in ("id", "type"):
                        if key in fragment:
                            value = fragment.pop(key)
                            if value is not None:
                                if not isinstance(value, str):
                                    raise StreamProtocolError(f"tool {key} must be text")
                                if value:
                                    if tool.get(key) not in (None, value):
                                        raise StreamProtocolError(f"conflicting tool {key}")
                                    tool[key] = value
                    if "function" in fragment:
                        function = fragment["function"]
                        if not isinstance(function, dict):
                            raise StreamProtocolError("tool function must be an object")
                        for key in ("name", "arguments"):
                            if function.get(key) is not None and not isinstance(function[key], str):
                                raise StreamProtocolError(f"tool function {key} must be text")
                    merge_delta(tool, fragment)
        merge_delta(message, delta)

    def complete(self) -> dict:
        if not self.choices:
            raise StreamProtocolError("stream completed without any choices")
        if any(choice["finish_reason"] is None for choice in self.choices.values()):
            raise StreamInterruptedError("stream completed with an unfinished choice")
        for index, tools in self.tools.items():
            self.choices[index]["message"]["tool_calls"] = [tools[key] for key in sorted(tools)]
        self.response["choices"] = [self.choices[key] for key in sorted(self.choices)]
        return self.response


async def stream_chat_completion(
    client: httpx.AsyncClient,
    payload: dict,
    *,
    on_chunk: Callable[[bytes], None],
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Return a complete response; synchronous callbacks own durable evidence writes.

    ``on_chunk`` runs before decoding every raw body chunk, including HTTP errors.
    Callback and transport errors propagate unchanged. Missing completion markers
    raise ``StreamInterruptedError``; malformed SSE data raises ``StreamProtocolError``.
    ``on_progress`` receives patches suitable for ``dict.update``.
    """
    request_payload = {
        **payload,
        "stream": True,
        "stream_options": {**(payload.get("stream_options") or {}), "include_usage": True},
    }
    decoder = SSEDecoder()
    accumulator = ChatCompletionAccumulator()
    chunks_received = 0
    bytes_received = 0

    def progress(**update: Any) -> None:
        if on_progress is not None:
            on_progress(update)

    async with client.stream(
        "POST", "chat/completions", json=request_payload, headers={"Accept-Encoding": "identity"}
    ) as response:
        progress(stage="headers", http_status=response.status_code)
        encoding = response.headers.get("content-encoding", "identity").strip().lower()
        if encoding not in {"", "identity"}:
            raise StreamProtocolError(f"unsupported stream Content-Encoding: {encoding}")
        async for chunk in response.aiter_raw():
            on_chunk(chunk)
            chunks_received += 1
            bytes_received += len(chunk)
            progress(
                stage="receiving", chunks_received=chunks_received, bytes_received=bytes_received
            )
            if not response.is_success:
                continue
            for event in decoder.feed(chunk):
                if event == "[DONE]":
                    result = accumulator.complete()
                    progress(stage="done", done=True)
                    return result
                try:
                    data = json.loads(event)
                except ValueError as error:
                    raise StreamProtocolError("invalid JSON in SSE data") from error
                if not isinstance(data, dict):
                    raise StreamProtocolError("SSE data must be a JSON object")
                finished = accumulator.add(data)
                progress(stage="delta", last_delta=copy.deepcopy(data), finish=finished)
        response.raise_for_status()
    raise StreamInterruptedError("stream ended before [DONE]")
