"""Native tool-calling sessions with episode-owned asynchronous recursive children."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import math
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

from rlm_train.checkpoint import SCHEMA_VERSION, read_checkpoint, write_checkpoint
from rlm_train.proxy import ClientHandle, SubLLMProxy
from rlm_train.repl.base import ReplBackend
from rlm_train.repl.subprocess import SubprocessReplBackend
from rlm_train.streaming import StreamInterruptedError, stream_chat_completion
from rlm_train.token_budget import (
    BudgetReservationPending,
    ContextBudgetPolicy,
    GeneratedTokenBudget,
    VllmPromptCounter,
)

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
    max_turns: int | None = 32
    max_tokens: int = 8192
    child_max_tokens: int = 4096
    max_children: int | None = 32
    max_live_children: int = 8
    max_concurrent_requests: int = 2
    max_model_calls: int | None = 256
    request_timeout: float = 600
    episode_timeout: float = 1800
    max_prompt_chars: int | None = 65536

    def __post_init__(self):
        for name in (
            "max_depth",
            "max_tokens",
            "child_max_tokens",
            "max_live_children",
            "max_concurrent_requests",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_turns", "max_model_calls", "max_children", "max_prompt_chars"):
            value = getattr(self, name)
            minimum = 0 if name == "max_children" else 1
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(f"{name} must be an integer >= {minimum}, or None")
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
    turns_spent: int = 0
    failure: dict[str, Any] | None = None
    restored: dict[str, Any] | None = None
    stage: str = "admission"


class _PlainClient:
    def __init__(self, runtime: NativeRuntime, session: _Session):
        self.runtime, self.session = runtime, session

    async def get_response(self, *, prompt, model, tools, sampling_args, state):
        del tools, sampling_args, state
        messages = [
            copy.deepcopy(m) if isinstance(m, dict) else m.model_dump(exclude_none=True)
            for m in prompt
        ]
        if (
            self.runtime.limits.max_prompt_chars is not None
            and len(json.dumps(messages, ensure_ascii=False)) > self.runtime.limits.max_prompt_chars
        ):
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
        budget_policy: ContextBudgetPolicy | None = None,
        prompt_counter: Callable[[dict[str, Any]], Any] | None = None,
        bootstrap_code: str = "",
        system_prompt: str = SYSTEM_PROMPT,
        compile_candidate: Callable[[str, str], Awaitable[dict[str, Any]]] | None = None,
        turn_notice: Callable[[int, int | None], str | None] | None = None,
        checkpoint_path: Path | None = None,
        stream_responses: bool = False,
        child_repl_at_limit: bool = False,
    ):
        self.client, self.model, self.proxy = client, model, proxy
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.stream_responses = stream_responses
        self.child_repl_at_limit = child_repl_at_limit
        self.resume_of: dict[str, Any] | None = None
        self.resume_adjustments: dict[str, Any] = {}
        self.prior_elapsed_s = 0.0
        if (
            isinstance(prompt_counter, VllmPromptCounter)
            and budget_policy is not None
            and prompt_counter.model_context != budget_policy.server_context_tokens
        ):
            raise ValueError("prompt_counter model_context must match server_context_tokens")
        if not isinstance(bootstrap_code, str) or not isinstance(system_prompt, str):
            raise TypeError("bootstrap_code and system_prompt must be strings")
        self.bootstrap_code, self.system_prompt = bootstrap_code, system_prompt
        self.compile_candidate = compile_candidate
        self.turn_notice = turn_notice
        if budget_policy is not None and prompt_counter is None:
            raise ValueError("budget_policy requires an async prompt_counter")
        self.budget_policy = budget_policy
        self.prompt_counter = prompt_counter
        self.generated_budget = GeneratedTokenBudget(budget_policy) if budget_policy else None
        self.limits = limits or NativeLimits()
        self.backend_factory, self.enable_thinking = backend_factory, enable_thinking
        self.event_path = Path(event_path) if event_path is not None else None
        if self.stream_responses and self.event_path is None:
            raise ValueError("stream_responses requires event_path for durable raw evidence")
        if (
            self.checkpoint_path is not None
            and self.event_path is not None
            and self.checkpoint_path.resolve() == self.event_path.resolve()
        ):
            raise ValueError("checkpoint_path and event_path must be different files")
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

    def checkpoint_config(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "limits": asdict(self.limits),
            "budget_policy": asdict(self.budget_policy) if self.budget_policy else None,
            "bootstrap_code": self.bootstrap_code,
            "system_prompt": self.system_prompt,
            "enable_thinking": self.enable_thinking,
            "stream_responses": self.stream_responses,
            "child_repl_at_limit": self.child_repl_at_limit,
        }

    async def resume(
        self,
        checkpoint: str | Path | dict[str, Any],
        *,
        additional_seconds: float,
        charge_uncertain_reservation: bool = False,
    ) -> dict[str, Any]:
        """Retry an interrupted root request from its last complete REPL state.

        The old attempt remains immutable. This is not a continuation of provider
        generation state. Unknown usage requires an explicit full-reservation charge.
        """
        if self.closed or self.root is not None:
            raise RuntimeError("runtime is single-use or already closed")
        if (
            isinstance(additional_seconds, bool)
            or not math.isfinite(additional_seconds)
            or additional_seconds <= 0
        ):
            raise ValueError("additional_seconds must be finite and positive")
        data = copy.deepcopy(read_checkpoint(checkpoint))
        if not data.get("recoverable"):
            raise ValueError(f"checkpoint is not recoverable: {data.get('recovery_refusal')}")
        if not data.get("cleanup_confirmed"):
            raise ValueError("checkpoint worker cleanup is not confirmed")
        if data["config"] != self.checkpoint_config():
            raise ValueError(
                "checkpoint model/system/bootstrap/limits/budget configuration differs"
            )
        if self.episode_id == data["episode_id"]:
            raise ValueError("resume requires a new episode_id")
        sources = {
            Path(source).resolve()
            for source in (data.get("checkpoint_path"), data.get("event_path"))
            if source is not None
        }
        if not isinstance(checkpoint, dict):
            sources.add(Path(checkpoint).resolve())
        for destination in (self.checkpoint_path, self.event_path):
            if destination is not None and destination.resolve() in sources:
                raise ValueError("resume output must not overwrite or append to a source artifact")
        snapshot = data["session"]["snapshot"]
        if snapshot.get("unsupported"):
            raise ValueError("checkpoint REPL state contains unsupported values")
        saved_budget = data.get("generated_budget")
        uncertain = sum(saved_budget["reserved"].values()) if saved_budget else 0
        if uncertain and not charge_uncertain_reservation:
            raise ValueError("unknown usage requires charge_uncertain_reservation=True")
        self.calls, self.children = data["counters"]["calls"], data["counters"]["children"]
        self.prior_elapsed_s = data["counters"]["elapsed_seconds"]
        if saved_budget:
            assert self.generated_budget is not None
            self.generated_budget.used = saved_budget["used"] + uncertain
            # The explicit conservative charge spends the full uncertain allowance;
            # it does not assert that the provider actually generated those tokens.
            if saved_budget["incomplete"] and not uncertain:
                raise ValueError("incomplete token accounting without a chargeable reservation")
        self.resume_of = {
            "episode_id": data["episode_id"],
            "session_id": data["session"]["id"],
            "checkpoint": str(checkpoint) if not isinstance(checkpoint, dict) else None,
        }
        self.resume_adjustments = {
            "additional_seconds": additional_seconds,
            "conservative_uncertain_charge": uncertain,
            "uncertain_reservations": saved_budget["reserved"] if saved_budget else {},
            "charge_is_actual_usage": False,
        }
        self.started = time.monotonic()
        self.deadline = self.started + additional_seconds
        saved = data["session"]
        self.root = self.new_session(None, saved["prompt"], saved["context"], self.model)
        self.root.restored = saved
        self.root.turns_spent = saved["turns_spent"]
        self.log(
            "resume_started",
            self.root,
            resume_of=self.resume_of,
            adjustments=self.resume_adjustments,
            calls_spent=self.calls,
            turns_spent=self.root.turns_spent,
        )
        return await self.finish_run(additional_seconds)

    async def capture_checkpoint(self, session: _Session, *, failure: bool = False) -> None:
        if self.checkpoint_path is None or session.parent is not None:
            return
        refusal = []
        snapshot = None
        if session.backend is None:
            refusal.append("root REPL has no completed state")
        else:
            try:
                snapshot = await session.backend.snapshot()
            except Exception as error:
                refusal.append(f"REPL snapshot unavailable: {type(error).__name__}: {error}")
        if snapshot is None and not refusal:
            refusal.append("REPL backend returned no snapshot")
        if snapshot is not None and snapshot.get("unsupported"):
            refusal.append("REPL snapshot contains unsupported state")
        if self.children:
            refusal.append(
                "root-only recovery does not restore descendant sessions or child handles"
            )
        if not failure:
            refusal.append("only a terminal recoverable model transport failure may be resumed")
        elif not session.failure or not session.failure.get("recoverable"):
            refusal.append("failure is not a recoverable model transport failure or model deadline")
        if session.record.get("cleanup_error"):
            refusal.append("worker cleanup was not confirmed")
        budget = self.generated_budget
        data = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "checkpoint_path": str(self.checkpoint_path.resolve()),
            "event_path": str(self.event_path.resolve()) if self.event_path else None,
            "saved_at_utc_seconds": time.time(),
            "reason": "failure" if failure else "completed_action",
            "recoverable": not refusal,
            "cleanup_confirmed": False,
            "recovery_refusal": refusal,
            "failure": copy.deepcopy(session.failure),
            "config": self.checkpoint_config(),
            "resume_of": self.resume_of,
            "resume_adjustments": self.resume_adjustments,
            "counters": {
                "calls": self.calls,
                "children": self.children,
                "elapsed_seconds": self.prior_elapsed_s + time.monotonic() - self.started,
            },
            "generated_budget": {
                "used": budget.used,
                "reserved": dict(budget.reserved),
                "incomplete": budget.incomplete,
            }
            if budget
            else None,
            "session": {
                "id": session.id,
                "depth": session.depth,
                "prompt": session.prompt,
                "context": session.context,
                "messages": copy.deepcopy(session.messages),
                "turns_spent": session.turns_spent,
                "current_action": copy.deepcopy(session.current_action),
                "snapshot": snapshot,
            },
        }
        write_checkpoint(self.checkpoint_path, data)
        session.record["checkpoint"] = {
            "path": str(self.checkpoint_path),
            "recoverable": data["recoverable"],
            "recovery_refusal": refusal,
        }
        self.log("checkpoint_saved", session, checkpoint=session.record["checkpoint"])

    @staticmethod
    def describe_failure(error: BaseException, stage: str, *, local_deadline: bool = False):
        chain = []
        current: BaseException | None = error
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            chain.append({"type": type(current).__name__, "message": str(current)})
            current = current.__cause__ or current.__context__
        status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
        transport = isinstance(error, (httpx.TransportError, StreamInterruptedError))
        retryable_status = status is not None and (status in (408, 429) or status >= 500)
        return {
            "stage": stage,
            "local_deadline": local_deadline,
            "transport_phase": "connect"
            if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout))
            else "read"
            if isinstance(error, (httpx.ReadError, httpx.ReadTimeout))
            else None,
            "http_status": status,
            "exception_chain": chain,
            "recoverable": stage == "model_request"
            and (local_deadline or transport or retryable_status),
            "server_root_cause": "unknown",
        }

    async def run(self, prompt: str, context: Any) -> dict[str, Any]:
        if self.closed or self.root is not None:
            raise RuntimeError("runtime is single-use or already closed")
        self.started = time.monotonic()
        self.deadline = self.started + self.limits.episode_timeout
        self.root = self.new_session(None, prompt, context, self.model)
        return await self.finish_run(self.limits.episode_timeout)

    async def finish_run(self, timeout: float) -> dict[str, Any]:
        assert self.root is not None
        try:
            await asyncio.wait_for(asyncio.shield(self.root.task), timeout)
        except TimeoutError as error:
            self.root.record.update(error_class="budget", error="total episode timeout")
            self.root.failure = self.describe_failure(error, self.root.stage, local_deadline=True)
            if self.root.record["events"] and "response" not in self.root.record["events"][-1]:
                self.root.failure["last_progress"] = copy.deepcopy(
                    self.root.record["events"][-1].get("progress")
                )
        finally:
            try:
                await self.close()
            finally:
                self.root.record["elapsed_s"] = time.monotonic() - self.started
                self.root.record["cumulative_elapsed_s"] = (
                    self.prior_elapsed_s + self.root.record["elapsed_s"]
                )
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
            "resume_of": copy.deepcopy(self.resume_of) if parent is None else None,
            "resume_adjustments": copy.deepcopy(self.resume_adjustments) if parent is None else {},
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
        if not isinstance(prompt, str) or (
            self.limits.max_prompt_chars is not None and len(prompt) > self.limits.max_prompt_chars
        ):
            raise ValueError("child prompt must be text within max_prompt_chars")
        if model is not None and (not isinstance(model, str) or not model):
            raise ValueError("model must be a nonempty string")
        if self.limits.max_children is not None and self.children >= self.limits.max_children:
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
            if session.depth >= self.limits.max_depth and not self.child_repl_at_limit:
                session.messages = [{"role": "user", "content": session.prompt}]
                event = await self.request(session, session.messages, session.model, plain=True)
                record["final_answer"] = self.plain_answer(event["response"])
                session.messages.append(copy.deepcopy(event["response"]["choices"][0]["message"]))
            else:
                await self.tool_loop(session)
        except asyncio.CancelledError as error:
            if record["error"] is None:
                record.update(error_class="cancelled", error="session cancelled")
            if session.failure is None or session.failure["stage"] != session.stage:
                session.failure = self.describe_failure(error, session.stage)
            if (
                session.failure.get("local_deadline")
                and session.stage == "model_request"
                and record["events"]
                and "response" not in record["events"][-1]
            ):
                record["events"][-1]["failure"] = copy.deepcopy(session.failure)
        except (NativeBudgetError, TimeoutError) as error:
            record.update(error_class="budget", error=str(error) or "model request timed out")
            if session.failure is None or session.failure["stage"] != session.stage:
                session.failure = self.describe_failure(error, session.stage)
        except NativeProtocolError as error:
            record.update(error_class="protocol", error=str(error))
            session.failure = self.describe_failure(error, session.stage)
        except Exception as error:
            record.update(error_class="infrastructure", error=repr(error))
            if session.failure is None or session.failure["stage"] != session.stage:
                session.failure = self.describe_failure(error, session.stage)
        finally:
            session.accepting = False
            record["status"] = "stopping"
            record["turns_spent"] = session.turns_spent
            if record["error"]:
                record["failure"] = copy.deepcopy(session.failure)
            session.cleanup_task = asyncio.create_task(self.checkpoint_and_cleanup(session))
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

    async def checkpoint_and_cleanup(self, session: _Session) -> None:
        try:
            await self.capture_checkpoint(session, failure=bool(session.record["error"]))
        except Exception as error:
            session.record["checkpoint_error"] = repr(error)
            if not session.record["error"]:
                session.record.update(error_class="checkpoint", error=repr(error))
        finally:
            await self.cleanup(session)

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
            session.record["cleanup_error"] = str(failures)
            if not session.record["error"]:
                session.record.update(
                    error_class="cleanup",
                    error=f"worker cleanup failed: {failures}",
                )
            self.checkpoint_cleanup_state(session, confirmed=False)
            self.log("cleanup_failed", session, error=session.record["cleanup_error"])
        else:
            session.record.pop("cleanup_error", None)
            self.checkpoint_cleanup_state(session, confirmed=True)
            self.proxy.unregister(session.id)
            self.log("cleanup_confirmed", session)

    def checkpoint_cleanup_state(self, session: _Session, *, confirmed: bool) -> None:
        if (
            self.checkpoint_path is None
            or session.parent is not None
            or not session.record.get("checkpoint")
        ):
            return
        try:
            saved = read_checkpoint(self.checkpoint_path)
            saved["cleanup_confirmed"] = confirmed
            reason = "worker cleanup was not confirmed"
            saved["recovery_refusal"] = [
                item for item in saved["recovery_refusal"] if item != reason
            ]
            if not confirmed:
                saved["recovery_refusal"].append(reason)
            saved["recoverable"] = not saved["recovery_refusal"]
            write_checkpoint(self.checkpoint_path, saved)
            session.record["checkpoint"].update(
                recoverable=saved["recoverable"], recovery_refusal=saved["recovery_refusal"]
            )
        except Exception as error:
            session.record["checkpoint_error"] = repr(error)
            if not session.record["error"]:
                session.record.update(error_class="checkpoint", error=repr(error))

    async def candidate_feedback(self, owner: str, source: str) -> dict[str, Any]:
        session = self.sessions.get(owner)
        if self.closed or session is None or not session.accepting or session.task.done():
            raise RuntimeError("compile_candidate owner session is finalizing or closed")
        if self.compile_candidate is None:
            raise RuntimeError("compile_candidate is not configured")
        if not isinstance(source, str):
            raise TypeError("compile_candidate source must be a string")
        if not source.strip():
            raise ValueError("compile_candidate source must not be empty")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise NativeBudgetError("episode deadline exceeded")
        entries = session.record.setdefault("compile_feedback", [])
        entry = {
            "session_id": owner,
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "ordinal": len(entries) + 1,
            "status": "running",
        }
        entries.append(entry)
        started = time.monotonic()
        previous_stage = session.stage
        session.stage = "compile_feedback"
        session.failure = None
        request = asyncio.create_task(self.compile_candidate(owner, source))
        session.requests.add(request)
        try:
            async with asyncio.timeout(min(self.limits.request_timeout, remaining)):
                response = await request
            if not isinstance(response, dict):
                raise TypeError("compile_candidate callback must return a dict")
            entry.update(status="completed", response=copy.deepcopy(response))
            return response
        except BaseException as error:
            entry.update(
                status="cancelled" if isinstance(error, asyncio.CancelledError) else "failed",
                error=str(error),
                error_type=type(error).__name__,
            )
            if session.accepting:
                session.failure = self.describe_failure(error, "compile_feedback")
            raise
        finally:
            if session.accepting:
                session.stage = previous_stage
            session.requests.discard(request)
            entry["duration_s"] = time.monotonic() - started
            # The trajectory retains actual diagnostics; the journal references it
            # without duplicating potentially large compiler output.
            self.log(
                "compile_feedback",
                session,
                **{
                    key: value
                    for key, value in entry.items()
                    if key not in {"session_id", "response"}
                },
            )

    async def tool_loop(self, session: _Session):
        session.backend = self.backend_factory(request_timeout=self.limits.episode_timeout)
        self.proxy.register(
            session.id,
            ClientHandle(
                client=_PlainClient(self, session),
                model=session.model,
                max_concurrent=self.limits.max_concurrent_requests,
                recursive=_ChildService(self, session.id),
                compile_candidate=(lambda source: self.candidate_feedback(session.id, source))
                if self.compile_candidate is not None
                else None,
            ),
        )
        session.stage = "repl_start"
        await session.backend.start(self.proxy.url, session.id, depth=session.depth)
        await session.backend.load_context(session.context)
        if self.bootstrap_code:
            await session.backend.bootstrap(self.bootstrap_code)
        if session.restored is not None:
            await session.backend.restore(session.restored["snapshot"])
            session.messages = copy.deepcopy(session.restored["messages"])
            session.current_action = copy.deepcopy(session.restored["current_action"])
        else:
            session.messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": session.prompt},
            ]
        await self.capture_checkpoint(session)
        while self.limits.max_turns is None or session.turns_spent < self.limits.max_turns:
            turn = session.turns_spent
            if self.turn_notice is not None:
                notice = self.turn_notice(turn + 1, self.limits.max_turns)
                if notice is not None:
                    if not isinstance(notice, str):
                        raise TypeError("turn_notice must return a string or None")
                    session.messages.append({"role": "user", "content": notice})
                    self.log("turn_notice", session, turn=turn + 1, notice=notice)
            session.turns_spent += 1
            session.stage = "model_request"
            event = await self.request(session, session.messages, session.model, plain=False)
            session.stage = "response_validation"
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
                repl_started = time.monotonic()
                session.stage = "repl_execute"
                session.failure = None
                result = await session.backend.execute(arguments["code"])
                session.stage = "action_complete"
                session.failure = None
                repl_elapsed = time.monotonic() - repl_started
                action = {
                    "code": arguments["code"],
                    "tool_call_id": call.get("id"),
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "final_answer": result.final_answer,
                    "elapsed_seconds": repl_elapsed,
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
                await self.capture_checkpoint(session)
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
        started = time.monotonic()
        events_before = len(session.record["events"])
        request_state = {"stage": "model_request"}
        session.failure = None
        try:
            async with asyncio.timeout(min(self.limits.request_timeout, self.deadline - started)):
                while True:
                    try:
                        return await self.request_attempt(
                            session,
                            messages,
                            model,
                            plain=plain,
                            budget_wait_started=started,
                            request_state=request_state,
                        )
                    except BudgetReservationPending:
                        # Do not hold an inference slot while another call settles.
                        assert self.generated_budget is not None
                        await self.generated_budget.changed.wait()
        except TimeoutError as error:
            session.failure = self.describe_failure(
                error, request_state["stage"], local_deadline=True
            )
            if len(session.record["events"]) > events_before:
                event = session.record["events"][-1]
                session.failure["last_progress"] = copy.deepcopy(event.get("progress"))
                event["failure"] = copy.deepcopy(session.failure)
                self.log(
                    "request_deadline", session, call_id=event["call_id"], failure=session.failure
                )
            raise

    async def request_attempt(
        self,
        session: _Session,
        messages: list,
        model: str,
        *,
        plain: bool,
        budget_wait_started: float,
        request_state: dict[str, str],
    ):
        started = time.monotonic()
        budget = self.generated_budget
        remaining = self.deadline - started
        if remaining <= 0:
            raise NativeBudgetError("episode deadline exceeded")
        async with asyncio.timeout(min(self.limits.request_timeout, remaining)):
            async with self.semaphore:
                queued = time.monotonic() - started
                if self.closed or not session.accepting:
                    raise RuntimeError("session is finalizing or closed")
                payload = {
                    "model": model,
                    "messages": copy.deepcopy(messages),
                    "temperature": 0,
                    "top_p": 1,
                    "max_tokens": self.limits.child_max_tokens
                    if plain or session.depth
                    else self.limits.max_tokens,
                    "logprobs": bool(self.budget_policy and self.budget_policy.logprobs),
                    "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
                }
                if payload["logprobs"]:
                    payload.update(top_logprobs=0, return_tokens_as_token_ids=True)
                if not plain:
                    payload.update(tools=[copy.deepcopy(TOOL)], tool_choice="auto")
                if self.stream_responses:
                    payload.update(stream=True, stream_options={"include_usage": True})
                prompt_tokens = None
                count_started = time.monotonic()
                if self.budget_policy:
                    assert self.prompt_counter is not None
                    request_state["stage"] = "prompt_count"
                    if not plain:
                        session.stage = "prompt_count"
                    try:
                        prompt_tokens = await self.prompt_counter(payload)
                    except BaseException as error:
                        session.failure = self.describe_failure(error, "prompt_count")
                        self.log("prompt_count_failed", session, failure=session.failure)
                        raise
                    if type(prompt_tokens) is not int or prompt_tokens < 0:
                        raise NativeProtocolError("prompt counter returned no valid token count")
                    request_state["stage"] = "model_request"
                    if not plain:
                        session.stage = "model_request"
                count_elapsed = time.monotonic() - count_started
                # Tokenization yielded: recheck admission before reserving or dispatching.
                if self.closed or not session.accepting:
                    raise RuntimeError("session is finalizing or closed")
                if (
                    self.limits.max_model_calls is not None
                    and self.calls >= self.limits.max_model_calls
                ):
                    raise NativeBudgetError("episode model-call budget exhausted")
                call_id = uuid.uuid4().hex
                reservation = None
                if budget:
                    assert prompt_tokens is not None
                    try:
                        reservation = budget.reserve(call_id, prompt_tokens)
                    except BudgetReservationPending:
                        raise
                    except RuntimeError as error:
                        raise NativeBudgetError(str(error)) from error
                    payload["max_tokens"] = reservation
                event: dict[str, Any] = {
                    "call_id": call_id,
                    "branch": "plain" if plain else ("child" if session.parent else "root"),
                    "episode_id": self.episode_id,
                    "session_id": session.id,
                    "parent_session_id": session.parent,
                    "depth": session.depth,
                    "spawn_origin": copy.deepcopy(session.origin),
                    "caller_action": copy.deepcopy(session.current_action),
                    "request": payload,
                    "token_limits": {
                        "output_tokens": {
                            "limit": payload["max_tokens"],
                            "source": "request.max_tokens",
                        },
                        "context_tokens": {
                            "limit": self.budget_policy.server_context_tokens
                            if self.budget_policy
                            else None,
                            "source": "tokenize.max_model_len" if self.budget_policy else None,
                        },
                    },
                    "rendered_prompt_tokens": prompt_tokens,
                    "timing": {
                        "queue_wait_s": queued,
                        "tokenize_elapsed_s": count_elapsed,
                        "budget_wait_s": started - budget_wait_started,
                    },
                }
                if budget:
                    assert reservation is not None
                    event["budget"] = {
                        "used_before": budget.used,
                        "outstanding_before": budget.outstanding - reservation,
                        "reserved_tokens": reservation,
                        "generated_limit": budget.policy.max_generated_tokens,
                    }
                try:
                    self.log("request_started", session, call=event)
                except BaseException:
                    if reservation is not None:
                        assert budget is not None
                        budget.release_undispatched(call_id)
                    raise
                session.record["events"].append(event)
                self.calls += 1
                dispatched = time.monotonic()
                request = asyncio.create_task(self.fetch_response(session, event, payload))
                session.requests.add(request)
                settled = False
                try:
                    event["response"] = await request
                    event["timing"]["http_elapsed_s"] = time.monotonic() - dispatched
                    usage = event["response"].get("usage")
                    if reservation is not None:
                        assert budget is not None
                        actual, _ = budget.settle(call_id, usage)
                        settled = True
                        event["budget"].update(
                            used_tokens=actual,
                            used_after=budget.used,
                            outstanding_after=budget.outstanding,
                            incomplete=budget.incomplete,
                        )
                        if actual is None:
                            raise NativeBudgetError(
                                "reported generation usage is missing or invalid; budget closed"
                            )
                        observed = usage.get("prompt_tokens")
                        event["prompt_count_matches"] = observed == prompt_tokens
                        if observed != prompt_tokens:
                            budget.incomplete = True
                            raise NativeProtocolError(
                                "rendered prompt count differs from reported prompt usage"
                            )
                    if payload["logprobs"]:
                        choices = event["response"].get("choices") or []
                        logprobs = choices[0].get("logprobs") if choices else None
                        selected = logprobs.get("content") if isinstance(logprobs, dict) else None
                        event["selected_logprob_tokens"] = (
                            len(selected) if isinstance(selected, list) else None
                        )
                        if not isinstance(selected, list) or not selected:
                            raise NativeProtocolError(
                                "selected token log probabilities were not returned"
                            )
                        event["logprob_coverage"] = {
                            "returned_tokens": len(selected),
                            "completion_tokens": usage.get("completion_tokens")
                            if isinstance(usage, dict)
                            else None,
                            "scope": "provider selected-token entries; reasoning coverage is not inferred",
                        }
                    event["timing"]["request_elapsed_s"] = time.monotonic() - started
                    self.log("request_finished", session, call=event)
                    return event
                except BaseException as error:
                    if reservation is not None and not settled:
                        assert budget is not None
                        budget.retain_uncertain(call_id)
                        event["budget"].update(
                            used_tokens=None, retained_uncertain=reservation, incomplete=True
                        )
                    event["timing"].update(
                        http_elapsed_s=time.monotonic() - dispatched,
                        request_elapsed_s=time.monotonic() - started,
                    )
                    event["error"] = repr(error)
                    failure = self.describe_failure(error, "model_request")
                    failure["last_progress"] = copy.deepcopy(event.get("progress"))
                    event["failure"] = failure
                    if not isinstance(error, asyncio.CancelledError):
                        session.failure = copy.deepcopy(failure)
                    self.log("request_failed", session, call=event)
                    raise
                finally:
                    session.requests.discard(request)

    async def fetch_response(self, session: _Session, event: dict, payload: dict) -> dict:
        def progress(update: dict) -> None:
            event.setdefault("progress", {}).update(
                **update, at_utc_seconds=time.time(), elapsed_s=time.monotonic() - started
            )
            if "last_delta" in update:
                self.log(
                    "stream_delta", session, call_id=event["call_id"], delta=update["last_delta"]
                )

        started = time.monotonic()
        progress({"stage": "dispatch"})
        if self.stream_responses:
            ordinal = 0

            def chunk(raw: bytes) -> None:
                nonlocal ordinal
                ordinal += 1
                self.log(
                    "stream_chunk",
                    session,
                    call_id=event["call_id"],
                    ordinal=ordinal,
                    bytes_base64=base64.b64encode(raw).decode("ascii"),
                )

            return await stream_chat_completion(
                self.client, payload, on_chunk=chunk, on_progress=progress
            )
        response = await self.client.post("chat/completions", json=payload)
        progress({"stage": "headers", "http_status": getattr(response, "status_code", None)})
        response.raise_for_status()
        raw = response.json()
        progress({"stage": "complete"})
        return raw

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
            stream.flush()
            os.fsync(stream.fileno())
