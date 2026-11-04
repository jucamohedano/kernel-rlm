"""Streaming, normalized Parquet export for native CuTe RLM trajectories.

The archive is evidence, not runtime recovery state. A session stores its initial
prompt once; turns store only each response and the corresponding actions/events.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "cute-trajectory-v1"
TARGET_SHARD_BYTES = 64 * 1024 * 1024
_CONFIGS = ("tasks", "sessions", "turns")
_SECRET_KEY = re.compile(r"(authorization|api[_-]?key|access[_-]?token|secret|password|credential|cookie)", re.I)
_SECRET_VALUE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*|\b(?:wk|ws)-[A-Za-z0-9_-]{8,}\b")
_REDACTION_COUNT = 0


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode()


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _clean(value: Any, key: str = "") -> Any:
    """Remove credential fields and known wire-secret strings before persistence."""
    global _REDACTION_COUNT
    if _SECRET_KEY.search(key):
        _REDACTION_COUNT += 1
        return None
    if isinstance(value, Mapping):
        cleaned = {}
        for item_key, item_value in value.items():
            if _SECRET_KEY.search(str(item_key)):
                _REDACTION_COUNT += 1
            else:
                cleaned[str(item_key)] = _clean(item_value, str(item_key))
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, str):
        value, count = _SECRET_VALUE.subn("[REDACTED]", value)
        _REDACTION_COUNT += count
    return value


def _json(value: Any) -> str | None:
    return None if value is None else json.dumps(_clean(value), ensure_ascii=False, separators=(",", ":"), default=str)


def _task_row(root: Mapping[str, Any], source_path: Path) -> dict[str, Any]:
    task = root.get("task") or root.get("task_record") or {}
    if not isinstance(task, Mapping):
        task = {}
    code = task.get("reference_code", root.get("reference_code"))
    ctx = root.get("context")
    if code is None and isinstance(ctx, Mapping):
        code = ctx.get("cute_dsl_task")
    task_id = str(task.get("task_id") or task.get("id") or root.get("task_id") or root.get("episode_id") or source_path.parent.name)
    return {
        "task_id": task_id,
        "task_hash": _hash({"code": code, "description": task.get("description"), "source": task.get("source")}),
        "reference_code": _clean(code) if isinstance(code, str) else None,
        "description": _clean(task.get("description")) if isinstance(task.get("description"), str) else None,
        "source_json": _json(task.get("source") or (ctx.get("source") if isinstance(ctx, Mapping) else None)),
    }


def _sessions(root: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = [root]
    children = root.get("child_sessions") or []
    if isinstance(children, list):
        rows.extend(row for row in children if isinstance(row, Mapping))
    elif isinstance(children, Mapping):
        rows.extend(row for row in children.values() if isinstance(row, Mapping))
    return rows


def _message_delta(previous: list[Any], current: Any) -> tuple[int, list[Any] | None]:
    if not isinstance(current, list):
        return 0, None
    shared = 0
    while shared < min(len(previous), len(current)) and previous[shared] == current[shared]:
        shared += 1
    return shared, current[shared:]


def _turn_rows(session: Mapping[str, Any], task_id: str, session_id: str) -> Iterator[dict[str, Any]]:
    events = session.get("events") or []
    previous_request: list[Any] = []
    if isinstance(events, list):
        for ordinal, event in enumerate(events):
            if not isinstance(event, Mapping):
                continue
            call = event.get("call") if isinstance(event.get("call"), Mapping) else event
            request = call.get("request") if isinstance(call.get("request"), Mapping) else {}
            shared_prefix, request_suffix = _message_delta(previous_request, request.get("messages"))
            if isinstance(request.get("messages"), list):
                previous_request = request["messages"]
            response = call.get("response")
            choices = response.get("choices", []) if isinstance(response, Mapping) else []
            message = choices[0].get("message") if choices and isinstance(choices[0], Mapping) else None
            if not isinstance(message, Mapping):
                # Preserve failed/incomplete request evidence without synthesizing a response.
                yield {
                    "task_id": task_id, "session_id": session_id,
                    "turn_ordinal": ordinal, "call_id": call.get("call_id"),
                    "request_prefix_shared_count": shared_prefix, "request_messages_suffix_json": _json(request_suffix),
                    "assistant_message_json": None,
                    "assistant_content": None, "reasoning_content": None,
                    "tool_calls": None, "tool_observations_json": None,
                    "usage_json": _json(response.get("usage") if isinstance(response, Mapping) else None),
                    "prompt_tokens": None, "completion_tokens": None,
                    "finish_reason": None, "logprobs_json": None, "token_ids": None, "token_mask": None,
                    "failure_json": _json(event.get("failure") or event.get("error")),
                    "request_metadata_json": _json({k: call.get(k) for k in ("branch", "rendered_prompt_tokens", "timing", "token_limits") if k in call}),
                    "sft_admitted": False,
                }
                continue
            raw_calls = message.get("tool_calls") or []
            tool_calls = []
            for tc in raw_calls:
                if not isinstance(tc, Mapping):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), Mapping) else {}
                tool_calls.append({"id": tc.get("id"), "type": tc.get("type"), "name": fn.get("name"), "arguments": fn.get("arguments") if isinstance(fn.get("arguments"), str) else _json(fn.get("arguments"))})
            usage = response.get("usage") if isinstance(response, Mapping) else None
            yield {
                "task_id": task_id, "session_id": session_id,
                "turn_ordinal": ordinal, "call_id": call.get("call_id"),
                "request_prefix_shared_count": shared_prefix, "request_messages_suffix_json": _json(request_suffix),
                "assistant_message_json": _json(message),
                "assistant_content": _clean(message.get("content")) if isinstance(message.get("content"), str) else None,
                "reasoning_content": _clean(message.get("reasoning_content") or message.get("reasoning")) if isinstance(message.get("reasoning_content") or message.get("reasoning"), str) else None,
                "tool_calls": tool_calls or None,
                "tool_observations_json": _json(call.get("actions") or event.get("actions") or event.get("action")),
                "usage_json": _json(usage),
                "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, Mapping) and type(usage.get("prompt_tokens")) is int else None,
                "completion_tokens": usage.get("completion_tokens") if isinstance(usage, Mapping) and type(usage.get("completion_tokens")) is int else None,
                "finish_reason": choices[0].get("finish_reason") if choices and isinstance(choices[0], Mapping) else None,
                "logprobs_json": _json(choices[0].get("logprobs")) if choices and isinstance(choices[0], Mapping) else None,
                "token_ids": message.get("token_ids") if isinstance(message.get("token_ids"), list) and all(type(item) is int for item in message["token_ids"]) else None,
                "token_mask": message.get("token_mask") if isinstance(message.get("token_mask"), list) and all(type(item) is int for item in message["token_mask"]) else None,
                "failure_json": _json(event.get("failure") or event.get("error")),
                "request_metadata_json": _json({k: call.get(k) for k in ("branch", "rendered_prompt_tokens", "timing", "token_limits") if k in call}),
                "sft_admitted": False,
            }
    # Legacy messages-only trajectories have no reliable request/response pairing.
    # Keep their initial prompt on the session; do not fabricate turn training rows.


def _load(path: Path) -> Iterator[tuple[dict[str, Any], Path]]:
    if path.is_dir():
        sources = sorted(path.rglob("trajectory.json"))
        for file in sources:
            with file.open(encoding="utf-8") as stream:
                root = json.load(stream)
            task_file = file.parent / "task.json"
            if task_file.exists():
                with task_file.open(encoding="utf-8") as stream:
                    root["task"] = json.load(stream)
            yield root, file
    elif path.suffix.lower() in (".jsonl", ".ndjson"):
        raise ValueError("JSONL event export is unsupported until request/response/action records are losslessly joined; provide native trajectory.json")
    else:
        with path.open(encoding="utf-8") as stream:
            yield json.load(stream), path


def _schema(pa, config: str):
    if config == "tasks":
        fields = [("task_id", pa.string()), ("task_hash", pa.string()), ("reference_code", pa.large_string()), ("description", pa.large_string()), ("source_json", pa.large_string())]
    elif config == "sessions":
        fields = [("task_id", pa.string()), ("episode_id", pa.string()), ("session_id", pa.string()), ("parent_session_id", pa.string()), ("depth", pa.int32()), ("model", pa.string()), ("prompt", pa.large_string()), ("spawn_origin_json", pa.large_string()), ("status", pa.string()), ("reward", pa.float64()), ("verification_json", pa.large_string()), ("error_class", pa.string()), ("error", pa.large_string()), ("final_answer", pa.large_string()), ("provider_json", pa.large_string()), ("prompt_world_verifier_json", pa.large_string()), ("artifact_refs_json", pa.large_string())]
    else:
        fields = [("task_id", pa.string()), ("session_id", pa.string()), ("turn_ordinal", pa.int32()), ("call_id", pa.string()), ("request_prefix_shared_count", pa.int32()), ("request_messages_suffix_json", pa.large_string()), ("assistant_message_json", pa.large_string()), ("assistant_content", pa.large_string()), ("reasoning_content", pa.large_string()), ("tool_calls", pa.list_(pa.struct([("id", pa.string()), ("type", pa.string()), ("name", pa.string()), ("arguments", pa.large_string())]))), ("tool_observations_json", pa.large_string()), ("usage_json", pa.large_string()), ("prompt_tokens", pa.int64()), ("completion_tokens", pa.int64()), ("finish_reason", pa.string()), ("logprobs_json", pa.large_string()), ("token_ids", pa.list_(pa.int64())), ("token_mask", pa.list_(pa.int8())), ("failure_json", pa.large_string()), ("request_metadata_json", pa.large_string()), ("sft_admitted", pa.bool_())]
    return pa.schema(fields, metadata={b"schema_version": SCHEMA_VERSION.encode(), b"config": config.encode()})


def export_trajectories(sources: Iterable[str | Path], output: str | Path, *, shard_target_bytes: int = TARGET_SHARD_BYTES) -> dict[str, Any]:
    """Write immutable compressed shards and an atomic, checksum-verified manifest."""
    global _REDACTION_COUNT
    _REDACTION_COUNT = 0
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(output)
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    if shard_target_bytes <= 0:
        raise ValueError("shard_target_bytes must be positive")
    out.mkdir(parents=True)
    for config in _CONFIGS:
        (out / config).mkdir()
    schemas = {name: _schema(pa, name) for name in _CONFIGS}
    manifest: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "format": "parquet", "compression": "zstd", "configs": {name: [] for name in _CONFIGS}, "counts": {name: 0 for name in _CONFIGS}}
    buffers: dict[str, list[dict[str, Any]]] = {name: [] for name in _CONFIGS}
    buffer_sizes = {name: 0 for name in _CONFIGS}
    task_hashes: dict[str, str] = {}

    def flush(config: str) -> None:
        rows = buffers[config]
        if not rows:
            return
        index = len(manifest["configs"][config])
        relative = f"{config}/part-{index:05d}.parquet"
        destination = out / relative
        table = pa.Table.from_pylist(rows, schema=schemas[config])
        fd, temporary = tempfile.mkstemp(prefix=".shard-", suffix=".tmp", dir=destination.parent)
        os.close(fd)
        try:
            pq.write_table(table, temporary, compression="zstd")
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        digest_state = hashlib.sha256()
        with destination.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest_state.update(block)
        digest = digest_state.hexdigest()
        manifest["configs"][config].append({"path": relative, "rows": len(rows), "bytes": destination.stat().st_size, "sha256": digest})
        manifest["counts"][config] += len(rows)
        buffers[config] = []
        buffer_sizes[config] = 0

    for source in sources:
        source = Path(source)
        for root, source_path in _load(source):
            task = _task_row(root, source_path)
            if task["task_hash"] not in task_hashes:
                task_hashes[task["task_hash"]] = task["task_id"]
                buffers["tasks"].append(task)
                buffer_sizes["tasks"] += len(_canonical(task))
            task_id = task_hashes[task["task_hash"]]
            sessions = _sessions(root)
            for session in sessions:
                sid = str(session.get("session_id") or root.get("episode_id") or _hash(session)[:24])
                status = str(session.get("status") or "unknown")
                row = {
                    "task_id": task_id, "episode_id": str(session.get("episode_id") or root.get("episode_id") or ""),
                    "session_id": sid, "parent_session_id": session.get("parent_session_id"),
                    "depth": session.get("depth") if type(session.get("depth")) is int else None,
                    "model": _clean(session.get("model")) if isinstance(session.get("model"), str) else None, "prompt": _clean(session.get("prompt")) if isinstance(session.get("prompt"), str) else None,
                    "spawn_origin_json": _json(session.get("spawn_origin")), "status": status,
                    "error_class": _clean(session.get("error_class")) if isinstance(session.get("error_class"), str) else None, "error": _clean(session.get("error")) if isinstance(session.get("error"), str) else None,
                    "reward": next((float(v) for v in (session.get("reward"), session.get("scalar_reward")) if isinstance(v, (int, float)) and not isinstance(v, bool)), None),
                    "verification_json": _json(session.get("verification") or session.get("verifier_result") or session.get("verifier")),
                    "final_answer": _clean(session.get("final_answer")) if isinstance(session.get("final_answer"), str) else None,
                    "provider_json": _json(root.get("provider") or root.get("provider_metadata")),
                    "prompt_world_verifier_json": _json({k: root.get(k) for k in ("prompt_hash", "world", "verifier", "verifier_identity", "system_prompt_hash") if k in root}),
                    "artifact_refs_json": _json(root.get("artifact_refs") or root.get("artifacts")),
                }
                buffers["sessions"].append(row)
                buffer_sizes["sessions"] += len(_canonical(row))
                for turn in _turn_rows(session, task_id, sid):
                    buffers["turns"].append(turn)
                    buffer_sizes["turns"] += len(_canonical(turn))
                    if buffer_sizes["turns"] >= shard_target_bytes:
                        flush("turns")
                for config in ("tasks", "sessions"):
                    if buffer_sizes[config] >= shard_target_bytes:
                        flush(config)
    for config in _CONFIGS:
        flush(config)
    card = """---
language:
- en
license: other
task_categories:
- text-generation
configs:
- config_name: tasks
  data_files: tasks/*.parquet
- config_name: sessions
  data_files: sessions/*.parquet
- config_name: turns
  data_files: turns/*.parquet
---

# CuTe RLM trajectory archive

Versioned normalized Parquet evidence, compressed with Zstandard. `tasks`, `sessions`, and `turns` are separate Hugging Face Datasets configs. Session prompts are stored once; each turn stores an exact request-message prefix length and suffix, so the request can be reconstructed even when rewritten. Failed and incomplete attempts are retained. `sft_admitted` is always false: success alone does not establish training eligibility. This archive is not a runtime checkpoint. Credentials and known wire-secret patterns are redacted; see the manifest for the number of redactions.
"""
    (out / "README.md").write_text(card, encoding="utf-8")
    with (out / "README.md").open("rb") as stream:
        os.fsync(stream.fileno())
    manifest["credential_redactions"] = _REDACTION_COUNT
    manifest_path = out / "manifest.json"
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode()
    fd, temporary = tempfile.mkstemp(prefix=".manifest-", dir=out)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
        directory_fd = os.open(out, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    card = """---
language:
- en
license: other
task_categories:
- text-generation
configs:
- config_name: tasks
  data_files: tasks/*.parquet
- config_name: sessions
  data_files: sessions/*.parquet
- config_name: turns
  data_files: turns/*.parquet
---

# CuTe RLM trajectory archive

Versioned normalized Parquet evidence, compressed with Zstandard. `tasks`, `sessions`, and `turns` are separate Hugging Face Datasets configs. Session prompts are stored once; turn rows store response/action deltas. Failed and incomplete attempts are retained. `sft_admitted` is always false: successful status alone does not establish training eligibility. Token IDs/masks are absent unless captured by source evidence (not synthesized). This archive is not a runtime checkpoint. See `manifest.json` for shard SHA-256 values. Credentials are removed during export.
"""
    (out / "README.md").write_text(card, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export native RLM trajectories to normalized Hugging Face Parquet configs")
    parser.add_argument("--input", action="append", required=True, help="native trajectory.json or directory containing trajectory.json files (repeatable); event JSONL is rejected until losslessly joinable")
    parser.add_argument("--output", required=True, help="new output directory; existing paths are never overwritten")
    args = parser.parse_args()
    result = export_trajectories(args.input, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
