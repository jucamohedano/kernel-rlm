"""Durable evidence journal for CuTe RLM collection.

This records what happened; it is not a transport, scheduler, or REPL serializer.
An intent without a recorded outcome is ambiguous and must never be replayed
automatically. Use one writer instance per trajectory/process; concurrent writers are
not coordinated. On reopen the append-only journal is authoritative; the progress
snapshot is rebuilt from it, so a stale/missing snapshot is recoverable.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from rlm_train.checkpoint import SCHEMA_VERSION, read_checkpoint, write_checkpoint


MINIMUM_NATIVE_RESUME_REQUIREMENTS = (
    "Native checkpoint resume is separate from this journal and requires a recoverable "
    "root-only checkpoint, confirmed worker cleanup, matching model/system/bootstrap/"
    "limits/budget configuration, a new episode id, and a terminal recoverable model "
    "transport failure or model deadline. Native recovery does not restore descendants "
    "or child handles; uncertain token usage requires an explicit conservative charge."
)

RESUME_CONTRACT = (
    "The journal preserves JSON evidence, not a live Python interpreter or full-tree "
    "REPL state. It cannot prove exact resumption across child sessions, provider "
    "requests, or live/data objects. A call intent with no recorded response is "
    "uncertain: inspect provider-side evidence and choose manually; never resend or "
    "replay a tool automatically. Persisted raw evidence remains useful even when "
    "execution cannot be resumed. " + MINIMUM_NATIVE_RESUME_REQUIREMENTS
)


class TrajectoryJournal:
    """Append-only JSONL evidence plus an atomically replaced progress snapshot."""

    def __init__(self, journal_path: str | Path, progress_path: str | Path):
        self.journal_path = Path(journal_path)
        self.progress_path = Path(progress_path)
        if self.journal_path.resolve() == self.progress_path.resolve():
            raise ValueError("journal and progress snapshot must be different files")
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self._sequence = 0
        self._calls: dict[str, dict[str, Any]] = {}
        self._quota_paused = False
        self._restore_index()
        self._write_progress()

    def _restore_index(self) -> None:
        if not self.journal_path.exists():
            return
        with self.journal_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.endswith("\n"):
                    raise ValueError(f"incomplete journal record at line {line_number}")
                record = json.loads(line)
                expected_sequence = self._sequence + 1
                if record.get("sequence") != expected_sequence:
                    raise ValueError(
                        f"non-monotonic journal sequence at line {line_number}: "
                        f"expected {expected_sequence}"
                    )
                self._sequence = expected_sequence
                call_id = record.get("call_id")
                if call_id is not None:
                    state = self._calls.setdefault(call_id, {"call_id": call_id})
                    state[record["event"]] = record
                if record["event"] == "quota_paused":
                    self._quota_paused = True
                elif record["event"] == "quota_resumed":
                    self._quota_paused = False

    def append(self, event_type: str, **fields: Any) -> dict[str, Any]:
        if not event_type or not isinstance(event_type, str):
            raise ValueError("event must be a non-empty string")
        reserved = {"sequence", "recorded_at_utc_seconds", "event"}
        conflicts = reserved.intersection(fields)
        if conflicts:
            raise ValueError(f"reserved journal field(s): {', '.join(sorted(conflicts))}")
        record = {
            "sequence": self._sequence + 1,
            "recorded_at_utc_seconds": time.time(),
            "event": event_type,
            **fields,
        }
        # Serialize before opening the journal, so invalid JSON cannot leave a partial row.
        encoded = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        with self.journal_path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._sequence += 1
        call_id = record.get("call_id")
        if call_id is not None:
            self._calls.setdefault(call_id, {"call_id": call_id})[event_type] = record
        if event_type == "quota_paused":
            self._quota_paused = True
        elif event_type == "quota_resumed":
            self._quota_paused = False
        self._write_progress()
        return record

    def call_intent(self, *, task_id: str, branch: str, call_id: str, intent: Any) -> dict[str, Any]:
        if call_id in self._calls:
            raise ValueError(f"call_id already exists: {call_id}")
        return self.append("call_intent", task_id=task_id, branch=branch, call_id=call_id, intent=intent)

    def raw_response(self, *, call_id: str, raw_response: Any) -> dict[str, Any]:
        self._require_call(call_id)
        return self.append("raw_response", call_id=call_id, raw_response=raw_response)

    def tool_result(self, *, call_id: str, tool_result: Any) -> dict[str, Any]:
        self._require_call(call_id)
        return self.append("tool_result", call_id=call_id, tool_result=tool_result)

    def call_finished(self, *, call_id: str, outcome: str) -> dict[str, Any]:
        self._require_call(call_id)
        return self.append("call_finished", call_id=call_id, outcome=outcome)

    def pause_for_quota(self, *, reason: str, details: Any = None) -> dict[str, Any]:
        return self.append("quota_paused", reason=reason, details=details)

    def resume_after_quota(self, *, note: str | None = None) -> dict[str, Any]:
        return self.append("quota_resumed", note=note)

    def _require_call(self, call_id: str) -> None:
        if call_id not in self._calls:
            raise ValueError(f"no recorded call intent: {call_id}")

    def uncertain_calls(self) -> list[dict[str, Any]]:
        """Return intents lacking a response or explicit terminal record; never retry them."""
        result = []
        for call_id, state in self._calls.items():
            if "call_finished" not in state:
                result.append({
                    "call_id": call_id,
                    "intent": state.get("call_intent", {}).get("intent"),
                    "has_raw_response": "raw_response" in state,
                    "has_tool_result": "tool_result" in state,
                })
        return result

    def progress(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "cute_rlm_progress",
            "last_sequence": self._sequence,
            "quota_paused": self._quota_paused,
            "uncertain_calls": self.uncertain_calls(),
            "journal_path": str(self.journal_path.resolve()),
            "resume_contract": RESUME_CONTRACT,
        }

    def _write_progress(self) -> None:
        write_checkpoint(self.progress_path, self.progress())


def read_progress(path: str | Path) -> dict[str, Any]:
    """Read the durable snapshot using the native checkpoint JSON/schema reader."""
    data = read_checkpoint(path)
    if data.get("kind") != "cute_rlm_progress":
        raise ValueError("not a CuTe RLM Trajectory progress snapshot")
    return data
