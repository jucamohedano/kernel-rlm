"""Modal DeepSeek OpenAI-compatible collection transport.

Callers own HTTP client/auth headers and protected NativeRuntime/REPL
configuration. The endpoint is a default base URL.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from rlm_train.native import NativeRuntime
from cute_rlm.trajectory_journal import TrajectoryJournal

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
BASE_URL = "https://jucamohedano--ep-deepseek-v4-1-flash-1017-server.us-west.modal.direct/v1"


class ModalProviderStop(RuntimeError):
    """Provider response requires a human decision; no request is retried."""


class ModalAuthenticationStop(ModalProviderStop):
    pass


class ModalRatePause(ModalProviderStop):
    pass


class ModalAmbiguousStop(ModalProviderStop):
    pass


class ModalCollectionRuntime(NativeRuntime):
    """Native runtime adapter retaining raw OpenAI responses and evidence."""

    def __init__(self, *args, journal: TrajectoryJournal, **kwargs):
        if len(args) >= 2:
            args = (args[0], MODEL, *args[2:])
            kwargs.pop("model", None)
        else:
            kwargs["model"] = MODEL
        super().__init__(*args, **kwargs)
        self.modal_journal = journal
        self.task_id = self.episode_id

    def log(self, kind: str, session: Any, **fields):
        if kind == "action_executed":
            action = fields.get("action")
            if isinstance(action, dict):
                self.modal_journal.tool_result(call_id=fields["call_id"], tool_result=action)
        return super().log(kind, session, **fields)

    async def fetch_response(self, session: Any, event: dict, payload: dict) -> dict:
        request = dict(payload)
        removed_cap = request.pop("max_tokens", None)
        request.pop("chat_template_kwargs", None)
        request["model"] = MODEL
        request["reasoning_effort"] = "high"
        request["temperature"] = 0.3
        request["top_p"] = 0.9
        call_id = event["call_id"]
        self.modal_journal.call_intent(
            task_id=str(self.task_id), branch=event["branch"], call_id=call_id,
            intent={"provider": "Modal OpenAI-compatible endpoint", "request": request,
                    "framework_max_tokens_removed": removed_cap,
                    "provider_default_output_limit": "not specified by this transport"},
        )
        response = await self.client.post("chat/completions", json=request,
                                          headers={"X-RLM-Call-Id": call_id})
        if response.status_code >= 400:
            try:
                body = response.json()
            except Exception:
                body = {"unparsed_body": response.text}
            self.modal_journal.raw_response(call_id=call_id, raw_response={
                "http_status": response.status_code, "body": body,
            })
            self.modal_journal.call_finished(call_id=call_id, outcome="http_error")
            details = f"Modal provider HTTP {response.status_code}; evidence retained; no retry"
            if response.status_code in (401, 403):
                raise ModalAuthenticationStop(details)
            if response.status_code == 429:
                raise ModalRatePause(details)
            raise ModalAmbiguousStop(details)
        raw = response.json()
        self.modal_journal.raw_response(call_id=call_id, raw_response=raw)
        self.modal_journal.call_finished(call_id=call_id, outcome="response_received")
        return raw


class ModalRunQueue:
    """Separate append-only Modal manifest; interrupted dispatches stay uncertain."""
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.items: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row = json.loads(line)
                if row["event"] == "task":
                    self.items[row["task_id"]] = row["task"]
                elif row["event"] == "running":
                    self.items[row["task_id"]]["status"] = "interrupted_uncertain"
                else:
                    self.items[row["task_id"]].update(status=row["event"], result=row.get("result"))

    def append(self, event: str, **fields):
        encoded = json.dumps({"event": event, **fields}, ensure_ascii=False, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n"); stream.flush(); os.fsync(stream.fileno())

    def enqueue(self, task: dict) -> str:
        task_id = str(task.get("task_id") or uuid.uuid4().hex)
        if task_id in self.items:
            raise ValueError(f"duplicate task id: {task_id}")
        stored = {**task, "task_id": task_id, "status": "queued"}
        self.items[task_id] = stored
        self.append("task", task_id=task_id, task=stored)
        return task_id

    def transition(self, task_id: str, state: str, result: Any = None):
        self.items[task_id].update(status=state, result=result)
        self.append(state, task_id=task_id, result=result)


def modal_manifest(*, task_ids: list[str], model: str = MODEL) -> dict[str, Any]:
    """Provider-specific identity for a separate Modal run."""
    return {"provider": "modal", "model": model, "base_url": BASE_URL,
            "task_ids": list(task_ids), "auth": "caller-owned; credentials are not recorded",
            "provider_output_limit": "provider default; transport sends no max_tokens"}
