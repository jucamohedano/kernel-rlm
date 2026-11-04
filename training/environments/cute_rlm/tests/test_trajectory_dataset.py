import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from cute_rlm.trajectory_dataset import SCHEMA_VERSION, export_trajectories


def _event(call_id, content, *, usage=None, tools=None, reasoning=None):
    return {
        "call_id": call_id,
        "branch": "root",
        "request": {"messages": [{"role": "system", "content": "large shared prefix"}]},
        "response": {"choices": [{"finish_reason": "tool_calls" if tools else "stop", "message": {
            "role": "assistant", "content": content,
            "reasoning_content": reasoning,
            "tool_calls": tools or [],
        }}], "usage": usage},
    }


def _trajectory(path, *, status="failed_no_sft", usage=None, reward=None):
    root = {
        "episode_id": "ep1", "task_id": "task1", "model": "mock-model",
        "prompt": "initial user context", "status": status,
        "error_class": "provider", "error": "stopped",
        "reference_code": "def kernel(): pass", "reward": reward,
        "events": [_event("c1", "thinking", usage=usage, reasoning="actual reasoning"),
                   _event("c2", None, tools=[{"id": "tc1", "type": "function", "function": {"name": "repl", "arguments": "{\"code\":\"print(1)\"}"}}])],
        "child_sessions": [{"session_id": "child1", "episode_id": "ep1", "parent_session_id": "root1", "depth": 1,
                            "model": "other", "prompt": "child prompt", "spawn_origin": {"call_id": "tc1"},
                            "status": "completed", "events": [_event("cc", "child answer", usage=None)]}],
    }
    root["session_id"] = "root1"
    path.write_text(json.dumps(root), encoding="utf-8")
    return root


def _export(tmp_path, root):
    source = tmp_path / "trajectory.json"
    source.write_text(json.dumps(root), encoding="utf-8")
    out = tmp_path / "archive"
    manifest = export_trajectories([source], out)
    return out, manifest


def test_roundtrip_preserves_tool_calls_reasoning_and_child_links(tmp_path):
    root = {
        "episode_id": "ep1", "session_id": "root1", "task_id": "task1", "model": "m", "prompt": "p", "status": "failed",
        "reference_code": "code", "events": [_event("c", "answer", tools=[{"id": "x", "type": "function", "function": {"name": "repl", "arguments": "{}"}}])],
        "child_sessions": [{"session_id": "ch", "parent_session_id": "root1", "depth": 1, "status": "completed", "prompt": "cp", "events": []}],
    }
    out, manifest = _export(tmp_path, root)
    sessions = pq.read_table(next((out / "sessions").glob("*.parquet"))).to_pylist()
    turns = pq.read_table(next((out / "turns").glob("*.parquet"))).to_pylist()
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert {row["session_id"] for row in sessions} == {"root1", "ch"}
    assert next(row for row in sessions if row["session_id"] == "ch")["parent_session_id"] == "root1"
    assert turns[0]["tool_calls"][0]["name"] == "repl"
    assert turns[0]["sft_admitted"] is False


def test_failed_incomplete_turn_kept_nullable_usage_reward_and_not_admitted(tmp_path):
    root = {"episode_id": "ep", "session_id": "s", "task_id": "t", "status": "quota_paused", "reward": None,
            "events": [{"call_id": "unknown", "error": "quota"}]}
    out, _ = _export(tmp_path, root)
    row = pq.read_table(next((out / "turns").glob("*.parquet"))).to_pylist()[0]
    assert row["prompt_tokens"] is None and row["completion_tokens"] is None
    assert row["assistant_content"] is None and row["sft_admitted"] is False
    session = pq.read_table(next((out / "sessions").glob("*.parquet"))).to_pylist()[0]
    assert session["status"] == "quota_paused"


def test_turn_storage_does_not_copy_request_history(tmp_path):
    root = {"episode_id": "ep", "session_id": "s", "task_id": "t", "status": "completed", "prompt": "P" * 10000,
            "events": [_event(f"c{i}", "x") for i in range(3)]}
    out, manifest = _export(tmp_path, root)
    turns = pq.read_table(next((out / "turns").glob("*.parquet"))).to_pylist()
    assert len(turns) == 3
    assert all("messages" not in row for row in turns)
    session = pq.read_table(next((out / "sessions").glob("*.parquet"))).to_pylist()[0]
    assert session["prompt"] == "P" * 10000
    assert sum(f["rows"] for f in manifest["configs"]["turns"]) == 3


def test_manifest_checksums_and_atomic_no_overwrite(tmp_path):
    root = {"episode_id": "ep", "session_id": "s", "task_id": "t", "status": "completed", "events": []}
    out, manifest = _export(tmp_path, root)
    assert (out / "manifest.json").exists() and (out / "README.md").exists()
    for entries in manifest["configs"].values():
        for entry in entries:
            assert hashlib.sha256((out / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    with pytest.raises(FileExistsError):
        export_trajectories([], out)


def test_credential_fields_removed_but_payload_kept(tmp_path):
    root = {"episode_id": "ep", "session_id": "s", "task_id": "t", "status": "completed", "provider": {"api_key": "sk-secret", "name": "mock"}, "events": []}
    out, _ = _export(tmp_path, root)
    row = pq.read_table(next((out / "sessions").glob("*.parquet"))).to_pylist()[0]
    assert "sk-secret" not in row["provider_json"]
    assert "mock" in row["provider_json"]


def test_jsonl_refused_until_events_can_be_joined_losslessly(tmp_path):
    events = tmp_path / "history.jsonl"
    events.write_text(json.dumps({"type": "request_started"}) + "\n")
    with pytest.raises(ValueError, match="losslessly joined"):
        export_trajectories([events], tmp_path / "out")


def test_exact_request_prefix_delta_and_reasoning_alias_reward_verification(tmp_path):
    first = _event("c1", "a", reasoning="r")
    first["request"] = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "one"}]}
    first["response"]["choices"][0]["message"].pop("reasoning_content")
    first["response"]["choices"][0]["message"]["reasoning"] = "wire reasoning"
    second = _event("c2", "b")
    second["request"] = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "rewritten"}]}
    root = {"episode_id": "ep", "session_id": "s", "task_id": "t", "status": "failed",
            "reward": 0.25, "verification": {"passed": False, "detail": "compile failed"}, "events": [first, second]}
    out, _ = _export(tmp_path, root)
    turns = pq.read_table(next((out / "turns").glob("*.parquet"))).to_pylist()
    assert turns[0]["request_prefix_shared_count"] == 0
    assert turns[1]["request_prefix_shared_count"] == 1
    suffix = json.loads(turns[1]["request_messages_suffix_json"])
    assert suffix == [{"role": "user", "content": "rewritten"}]
    assert json.loads(turns[0]["assistant_message_json"])["reasoning"] == "wire reasoning"
    assert turns[0]["reasoning_content"] == "wire reasoning"
    session = pq.read_table(next((out / "sessions").glob("*.parquet"))).to_pylist()[0]
    assert session["reward"] == 0.25
    assert json.loads(session["verification_json"])["passed"] is False


def test_known_wire_secrets_are_redacted_and_counted(tmp_path):
    root = {"episode_id": "ep", "session_id": "s", "task_id": "t", "status": "completed",
            "events": [_event("c", "wk-secretvalue123")], "provider": {"name": "Bearer abc.def.ghi"}}
    out, manifest = _export(tmp_path, root)
    assert manifest["credential_redactions"] >= 2
    turn = pq.read_table(next((out / "turns").glob("*.parquet"))).to_pylist()[0]
    assert "wk-secretvalue123" not in turn["assistant_message_json"]
    session = pq.read_table(next((out / "sessions").glob("*.parquet"))).to_pylist()[0]
    assert "abc.def.ghi" not in session["provider_json"]
