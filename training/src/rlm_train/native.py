"""Native tool-calling sessions with episode-owned asynchronous recursive children."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rlm_train.proxy import ClientHandle, SubLLMProxy
from rlm_train.repl.base import ReplBackend
from rlm_train.repl.subprocess import SubprocessReplBackend

TOOL = {
    "type": "function",
    "function": {
        "name": "repl",
        "description": "Execute Python in this session's persistent REPL.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}
SYSTEM_PROMPT = (
    "Answer the question using the persistent Python REPL. Inspect the supplied "
    "read-only context variable; do not paste or execute the whole context as code. "
    "Your only external tool is repl(code). Variables persist across tool calls. "
    "history is a snapshot of this session's model/tool messages. Inside Python, "
    "llm_query(prompt, model=None) makes a plain completion with no tools, and "
    "llm_query_batched(prompts, model=None) returns ordered plain answers. "
    "rlm_query(prompt, model=None) runs a recursive child with its own context, "
    "variables and history. Pass all information it needs in its prompt; it does "
    "not inherit your variables. rlm_query(prompt, blocking=False) returns a handle "
    "so you can keep working. rlm_status(handle), rlm_result(handle), and "
    "rlm_wait(handle, timeout_s=30) inspect or collect your direct children. A wait "
    "timeout does not cancel work. rlm_query_batched(prompts) returns ordered child "
    "answers. Child admission and requests are bounded; handle errors rather than "
    "retrying exhausted budgets. At the depth cutoff a child is a plain completion "
    "without a REPL. Finalization stops outstanding descendants. Finish by setting "
    "answer['content'] to your answer string and answer['ready'] = True."
)


class NativeBudgetError(RuntimeError):
    """An episode or generation limit prevented completion."""


class NativeProtocolError(RuntimeError):
    """The model did not follow the native REPL protocol."""


class NativeCleanupError(RuntimeError):
    """A managed worker's exit could not be confirmed."""


@dataclass(frozen=True)
class NativeLimits:
    max_depth: int = 2
    max_turns: int = 32
    max_tokens: int = 8192
    child_max_tokens: int = 4096
    max_children: int = 32
    max_live_children: int = 8
    max_concurrent_requests: int = 2
    max_model_calls: int = 256
    request_timeout: float = 600
    episode_timeout: float = 1800
    max_prompt_chars: int = 65536

    def __post_init__(self):
        for name in (
            "max_depth",
            "max_turns",
            "max_tokens",
            "child_max_tokens",
            "max_live_children",
            "max_concurrent_requests",
            "max_model_calls",
            "max_prompt_chars",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.max_children) is not int or self.max_children < 0:
            raise ValueError("max_children must be a nonnegative integer")
        for name in ("request_timeout", "episode_timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass
class _Session:
    id: str
    parent: str | None
    depth: int
    model: str
    prompt: str
    context: Any
    record: dict[str, Any]
    task: asyncio.Task = field(init=False)
    cleanup_task: asyncio.Task | None = None
    backend: ReplBackend | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    requests: set[asyncio.Task] = field(default_factory=set)
    origin: dict[str, Any] = field(default_factory=dict)
    current_action: dict[str, Any] = field(default_factory=dict)
    accepting: bool = True


class _PlainClient:
    def __init__(self, runtime: NativeRuntime, session: _Session):
        self.runtime, self.session = runtime, session

    async def get_response(self, *, prompt, model, tools, sampling_args, state):
        del tools, sampling_args, state
        messages = [
            copy.deepcopy(m) if isinstance(m, dict) else m.model_dump(exclude_none=True)
            for m in prompt
        ]
        if len(json.dumps(messages, ensure_ascii=False)) > self.runtime.limits.max_prompt_chars:
            raise NativeBudgetError("plain prompt exceeds max_prompt_chars")
        event = await self.runtime.request(self.session, messages, model, plain=True)
        content = self.runtime.plain_answer(event["response"])
        usage = event["response"].get("usage") or {}
        return SimpleNamespace(
            message=SimpleNamespace(content=content),
            usage=SimpleNamespace(
                **{k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
            ),
        )


class _ChildService:
    def __init__(self, runtime: NativeRuntime, owner: str):
        self.runtime, self.owner = runtime, owner

    async def query(self, prompt: str, model: str | None = None, *, blocking: bool = True):
        if type(blocking) is not bool:
            raise ValueError("blocking must be a boolean")
        child = self.runtime.spawn(self.owner, prompt, model)
        if not blocking:
            return child.id
        await asyncio.shield(child.task)
        return self.runtime.child_result(child)

    async def query_batched(self, prompts: list[str], model: str | None = None):
        if not isinstance(prompts, list) or any(not isinstance(p, str) for p in prompts):
            raise ValueError("prompts must be a list of strings")
        children = []
        try:
            for prompt in prompts:
                children.append(self.runtime.spawn(self.owner, prompt, model))
        except Exception:
            for child in children:
                child.accepting = False
                child.task.cancel()
            await asyncio.gather(*(c.task for c in children), return_exceptions=True)
            for child in children:
                child.record.update(
                    status="failed", error_class="cancelled", error="batch admission failed"
                )
                self.runtime.log(
                    "session_terminal", child, status="failed", error="batch admission failed"
                )
            raise
        await asyncio.gather(*(asyncio.shield(c.task) for c in children))
        return [self.runtime.child_result(c) for c in children]

    async def status(self, handle: str):
        child = self.runtime.authorize(self.owner, handle)
        return child.record["status"]

    async def result(self, handle: str):
        child = self.runtime.authorize(self.owner, handle)
        if not child.task.done():
            raise RuntimeError("child has not finished")
        return self.runtime.child_result(child)

    async def wait(self, handle: str, timeout_s: float = 30):
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("timeout_s must be finite and nonnegative")
        child = self.runtime.authorize(self.owner, handle)
        # Shielding is essential: a wait timeout is not a cancellation request.
        await asyncio.wait_for(asyncio.shield(child.task), timeout=timeout_s)
        return self.runtime.child_result(child)


class NativeRuntime:
    """One episode's native sessions. The caller owns the HTTP client and proxy.

    ``records`` maps session IDs to serializable trajectories. Each ``events`` list
    contains one record per model request; the append-only journal also records
    admission, request-start, response, action and confirmed cleanup transitions.
    """

    def __init__(
        self,
        client,
        model: str,
        proxy: SubLLMProxy,
        *,
        limits: NativeLimits | None = None,
        backend_factory: Callable[..., ReplBackend] = SubprocessReplBackend,
        enable_thinking: bool = True,
        event_path: Path | None = None,
        episode_id: str | None = None,
    ):
        self.client, self.model, self.proxy = client, model, proxy
        self.limits = limits or NativeLimits()
        self.backend_factory, self.enable_thinking = backend_factory, enable_thinking
        self.event_path = Path(event_path) if event_path is not None else None
        self.episode_id = episode_id or uuid.uuid4().hex
        self.records: dict[str, dict[str, Any]] = {}
        self.sessions: dict[str, _Session] = {}
        self.semaphore = asyncio.Semaphore(self.limits.max_concurrent_requests)
        self.closed = False
        self.root: _Session | None = None
        self.calls = 0
        self.children = 0
        self.started = 0.0
        self.deadline = 0.0
        self.close_task: asyncio.Task | None = None

    async def run(self, prompt: str, context: Any) -> dict[str, Any]:
        if self.closed or self.root is not None:
            raise RuntimeError("runtime is single-use or already closed")
        self.started = time.monotonic()
        self.deadline = self.started + self.limits.episode_timeout
        self.root = self.new_session(None, prompt, context, self.model)
        try:
            await asyncio.wait_for(asyncio.shield(self.root.task), self.limits.episode_timeout)
        except TimeoutError:
            self.root.record.update(error_class="budget", error="total episode timeout")
        finally:
            try:
                await self.close()
            finally:
                self.root.record["elapsed_s"] = time.monotonic() - self.started
                self.root.record["child_sessions"] = [
                    r for sid, r in self.records.items() if sid != self.root.id
                ]
        return self.root.record

    async def close(self):
        if self.close_task is None or (self.close_task.done() and self.close_task.exception()):
            self.close_task = asyncio.create_task(self.finish_close())
        await asyncio.shield(self.close_task)

    async def finish_close(self):
        self.closed = True
        for session in self.sessions.values():
            session.accepting = False
            if not session.task.done() and session.cleanup_task is None:
                session.task.cancel()
        await asyncio.gather(*(s.task for s in self.sessions.values()), return_exceptions=True)
        await asyncio.gather(
            *(s.cleanup_task for s in self.sessions.values() if s.cleanup_task is not None),
            return_exceptions=True,
        )
        # A failed stop retains its process reference; allow cleanup to be retried.
        for session in sorted(self.sessions.values(), key=lambda s: s.depth, reverse=True):
            if session.record.get("cleanup_error"):
                await self.cleanup(session)
        failures = [s.id for s in self.sessions.values() if s.record.get("cleanup_error")]
        if failures:
            raise NativeCleanupError(f"worker exit not confirmed for sessions {failures}")

    def new_session(self, parent: _Session | None, prompt: str, context: Any, model: str):
        sid = uuid.uuid4().hex
        origin = copy.deepcopy(parent.current_action) if parent else {}
        record: dict[str, Any] = {
            "session_id": sid,
            "episode_id": self.episode_id,
            "parent_session_id": parent.id if parent else None,
            "depth": parent.depth + 1 if parent else 0,
            "model": model,
            "prompt": prompt,
            "spawn_origin": origin,
            "events": [],
            "messages": [],
            "status": "running",
            "final_answer": None,
            "error_class": None,
            "error": None,
        }
        session = _Session(
            sid,
            record["parent_session_id"],
            record["depth"],
            model,
            prompt,
            context,
            record,
            origin=origin,
        )
        self.sessions[sid], self.records[sid] = session, record
        self.log("session_admitted", session)
        session.task = asyncio.create_task(self.drive(session))
        return session

    def spawn(self, owner: str, prompt: str, model: str | None):
        parent = self.sessions.get(owner)
        if self.closed or parent is None or not parent.accepting:
            raise RuntimeError("parent session is finalizing or closed")
        if not isinstance(prompt, str) or len(prompt) > self.limits.max_prompt_chars:
            raise ValueError("child prompt must be text within max_prompt_chars")
        if model is not None and (not isinstance(model, str) or not model):
            raise ValueError("model must be a nonempty string")
        if self.children >= self.limits.max_children:
            raise NativeBudgetError("episode child admission budget exhausted")
        live = sum(s.parent is not None and not s.task.done() for s in self.sessions.values())
        if live >= self.limits.max_live_children:
            raise NativeBudgetError("live child session capacity exhausted")
        if parent.depth >= self.limits.max_depth:
            raise NativeBudgetError("maximum recursion depth exceeded")
        self.children += 1
        return self.new_session(parent, prompt, prompt, model or parent.model)

    def authorize(self, owner: str, handle: str):
        caller = self.sessions.get(owner)
        if self.closed or caller is None or not caller.accepting:
            raise RuntimeError("caller session is finalizing or closed")
        child = self.sessions.get(handle)
        if child is None or child.parent != owner:
            raise RuntimeError("unknown, stale or foreign child handle")
        return child

    @staticmethod
    def child_result(session: _Session):
        if session.record["status"] != "completed":
            raise RuntimeError(session.record.get("error") or "child did not complete")
        return session.record["final_answer"]

    async def drive(self, session: _Session):
        record = session.record
        try:
            if session.depth >= self.limits.max_depth:
                session.messages = [{"role": "user", "content": session.prompt}]
                event = await self.request(session, session.messages, session.model, plain=True)
                record["final_answer"] = self.plain_answer(event["response"])
                session.messages.append(copy.deepcopy(event["response"]["choices"][0]["message"]))
            else:
                await self.tool_loop(session)
        except asyncio.CancelledError:
            if record["error"] is None:
                record.update(error_class="cancelled", error="session cancelled")
        except (NativeBudgetError, TimeoutError) as error:
            record.update(error_class="budget", error=str(error) or "model request timed out")
        except NativeProtocolError as error:
            record.update(error_class="protocol", error=str(error))
        except Exception as error:
            record.update(error_class="infrastructure", error=repr(error))
        finally:
            session.accepting = False
            record["status"] = "stopping"
            session.cleanup_task = asyncio.create_task(self.cleanup(session))
            await asyncio.shield(session.cleanup_task)
            record["messages"] = copy.deepcopy(session.messages)
            record["status"] = "failed" if record["error"] else "completed"
            self.log(
                "session_terminal",
                session,
                status=record["status"],
                final_answer=record["final_answer"],
                error=record["error"],
            )

    async def cleanup(self, session: _Session):
        session.accepting = False
        children = [s for s in self.sessions.values() if s.parent == session.id]
        for child in children:
            child.accepting = False
            if not child.task.done() and child.cleanup_task is None:
                child.task.cancel()
        await asyncio.gather(*(c.task for c in children), return_exceptions=True)
        requests = list(session.requests)
        for request in requests:
            request.cancel()
        await asyncio.gather(*requests, return_exceptions=True)
        failures = [c.id for c in children if c.record.get("cleanup_error")]
        if session.backend is not None:
            try:
                await session.backend.stop()
            except Exception as error:
                failures.append(repr(error))
        if failures:
            session.record.update(
                cleanup_error=str(failures),
                error_class="cleanup",
                error=f"worker cleanup failed: {failures}",
            )
            self.log("cleanup_failed", session, error=session.record["cleanup_error"])
        else:
            session.record.pop("cleanup_error", None)
            self.proxy.unregister(session.id)
            self.log("cleanup_confirmed", session)

    async def tool_loop(self, session: _Session):
        session.backend = self.backend_factory(request_timeout=self.limits.episode_timeout)
        self.proxy.register(
            session.id,
            ClientHandle(
                client=_PlainClient(self, session),
                model=session.model,
                max_concurrent=self.limits.max_concurrent_requests,
                recursive=_ChildService(self, session.id),
            ),
        )
        await session.backend.start(self.proxy.url, session.id, depth=session.depth)
        await session.backend.load_context(session.context)
        session.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": session.prompt},
        ]
        for turn in range(self.limits.max_turns):
            event = await self.request(session, session.messages, session.model, plain=False)
            event["turn"] = turn
            choice = event["response"]["choices"][0]
            message = choice["message"]
            session.messages.append(copy.deepcopy(message))
            if choice.get("finish_reason") == "length":
                raise NativeBudgetError("model response truncated")
            calls = message.get("tool_calls") or []
            if not calls:
                raise NativeProtocolError("model ended without explicit REPL answer")
            for index, call in enumerate(calls):
                function = call.get("function", {})
                arguments = function.get("arguments", {})
                try:
                    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                except ValueError as error:
                    raise NativeProtocolError("invalid tool arguments") from error
                if (
                    function.get("name") != "repl"
                    or not isinstance(arguments, dict)
                    or set(arguments) != {"code"}
                    or not isinstance(arguments["code"], str)
                ):
                    raise NativeProtocolError("expected repl with exactly one string code argument")
                session.current_action = {
                    "call_id": event["call_id"],
                    "turn": turn,
                    "action_index": index,
                    "tool_call_id": call.get("id"),
                }
                await session.backend.update_history(session.messages)
                result = await session.backend.execute(arguments["code"])
                action = {
                    "code": arguments["code"],
                    "tool_call_id": call.get("id"),
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "final_answer": result.final_answer,
                }
                event.setdefault("actions", []).append(action)
                self.log("action_executed", session, call_id=event["call_id"], action=action)
                session.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": (
                            result.stdout + ("\n" + result.stderr if result.stderr else "")
                        )[:12000],
                    }
                )
                if result.final_answer is not None:
                    session.record["final_answer"] = result.final_answer
                    return
        raise NativeBudgetError("turn limit reached without explicit final answer")

    @staticmethod
    def plain_answer(raw: dict):
        choice = raw["choices"][0]
        message = choice["message"]
        if choice.get("finish_reason") == "length":
            raise NativeBudgetError("plain completion truncated")
        if choice.get("finish_reason") != "stop" or message.get("tool_calls"):
            raise NativeProtocolError("plain completion returned tools or did not stop")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise NativeProtocolError("plain completion returned no final text")
        return content

    async def request(self, session: _Session, messages: list, model: str, *, plain: bool):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise NativeBudgetError("episode deadline exceeded")
        # One request permit, never held across Python execution or child waits.
        async with asyncio.timeout(min(self.limits.request_timeout, remaining)):
            async with self.semaphore:
                if self.closed or not session.accepting:
                    raise RuntimeError("session is finalizing or closed")
                if self.calls >= self.limits.max_model_calls:
                    raise NativeBudgetError("episode model-call budget exhausted")
                self.calls += 1
                cap = (
                    self.limits.child_max_tokens
                    if plain or session.depth
                    else self.limits.max_tokens
                )
                payload = {
                    "model": model,
                    "messages": copy.deepcopy(messages),
                    "temperature": 0,
                    "top_p": 1,
                    "max_tokens": cap,
                    "logprobs": False,
                    "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
                }
                if not plain:
                    payload.update(tools=[copy.deepcopy(TOOL)], tool_choice="auto")
                event = {
                    "call_id": uuid.uuid4().hex,
                    "branch": "plain" if plain else ("child" if session.parent else "root"),
                    "episode_id": self.episode_id,
                    "session_id": session.id,
                    "parent_session_id": session.parent,
                    "depth": session.depth,
                    "spawn_origin": copy.deepcopy(session.origin),
                    "caller_action": copy.deepcopy(session.current_action),
                    "request": payload,
                }
                session.record["events"].append(event)
                self.log("request_started", session, call=event)
                # Track the actual request, not its owning session/broker handler task.
                request = asyncio.create_task(self.client.post("chat/completions", json=payload))
                session.requests.add(request)
                try:
                    response = await request
                    response.raise_for_status()
                    event["response"] = response.json()
                    self.log("request_finished", session, call=event)
                    return event
                except BaseException as error:
                    event["error"] = repr(error)
                    self.log("request_failed", session, call=event)
                    raise
                finally:
                    session.requests.discard(request)

    def log(self, kind: str, session: _Session, **fields):
        if self.event_path is None:
            return
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "type": kind,
            "episode_id": self.episode_id,
            "session_id": session.id,
            "parent_session_id": session.parent,
            "depth": session.depth,
            "at_utc_seconds": time.time(),
            **fields,
        }
        with self.event_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
