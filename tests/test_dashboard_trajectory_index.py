from __future__ import annotations

import json
from pathlib import Path

from dashboard import data


def test_index_keeps_distinct_rollouts_and_export_dedup(tmp_path: Path) -> None:
    run = tmp_path / "20261020T192919Z-run-a"
    run.mkdir()
    traces = [
        {
            "id": "raw-a",
            "task": {"data": {"name": "task0025"}},
            "info": {
                "kernel_submission": "code",
                "kernel_termination": "submitted",
                "kernel_verify": {"correct": False},
                "repl_transcript": [{"role": "assistant", "tool_calls": [{"id": "c"}]}],
            },
        },
        {
            "id": "raw-b",
            "task": {"data": {"name": "task0025"}},
            "info": {
                "kernel_submission": "code",
                "kernel_termination": "submitted",
                "kernel_verify": {"correct": True},
            },
        },
    ]
    line = json.dumps({"traces": traces}) + "\n"
    (run / "traces.jsonl").write_text(line)
    (run / "effective").mkdir()
    (run / "effective/traces.jsonl").write_text(line)
    (run / "report.json").write_text(
        json.dumps({"run_id": "20261020T192919Z-run-a", "title": "Run", "status": "complete"})
    )
    index = data.trajectory_index(tmp_path)
    assert len(index["trajectories"]) == 2
    assert [t["id"] for t in index["trajectories"]] == ["raw-a", "raw-b"]
    assert [t["attempt"] for t in index["trajectories"]] == [1, 2]
    assert [t["completion"] for t in index["trajectories"]] == ["completed", "completed"]
    assert index["trajectories"][0]["outcome"]["correct"] is False
    assert index["trajectories"][0]["metrics"]["tool_calls"] == 1
    assert index["trajectories"][0]["metrics"]["steps"] == 1
    assert "raw" not in index["trajectories"][0] and "events" not in index["trajectories"][0]


def test_detail_keeps_existing_id_and_task(tmp_path: Path) -> None:
    run = {"id": "r"}
    trace = {"id": "trace-id", "task": {"data": {"name": "task"}}, "info": {}}
    detail = data.trajectory_detail(run, trace, 0)
    assert detail["id"] == "trace-id"
    assert detail["task"] == "task"
    assert detail["run"]["id"] == "r"


def test_catalog_adds_explicit_task_title(tmp_path: Path) -> None:
    catalog = tmp_path / "tasks.jsonl"
    catalog.write_text(
        json.dumps(
            {"_task_id": "task0025", "ops": '["torch.max", "nn.ReLU"]', "data_source": "torch#1"}
        )
        + "\n"
    )
    run = {"id": "r", "report": {"model_id": "model-X"}}
    trace = {"id": "t", "task": {"data": {"name": "task0025"}}, "info": {}}
    metadata = data.trajectory_metadata(run, trace, 0, task_catalog_path=catalog)
    assert metadata["title"] == "max · ReLU"
    assert metadata["task_title_source"] == "catalog.ops"
    assert metadata["model"] == "model-X"
    assert metadata["benchmark"] is None
    assert metadata["completion"] == "unknown"


def test_root_usage_metrics_do_not_guess_cache_or_include_child_calls():
    trace = {
        "nodes": [{"message": {"role": "assistant"}}, {"message": {"role": "user"}}],
        "calls": [
            {
                "node": 0,
                "usage": {"prompt_tokens": 120, "prompt_tokens_details": {"cached_tokens": 30}},
            },
            {"node": 1, "usage": {"prompt_tokens": 9000}},
        ],
    }
    metrics = data.trajectory_metrics(trace)
    assert metrics["peak_context"] == 120
    assert metrics["prompt_cache"] == 25
    assert metrics["commands_failed"] is None
    trace["calls"][0]["usage"]["prompt_tokens_details"] = None
    assert data.trajectory_metrics(trace)["prompt_cache"] is None


def test_index_missing_ids_are_distinct_across_tasks(tmp_path):
    run = tmp_path / "20261020T192919Z-run"
    run.mkdir()
    (run / "traces.jsonl").write_text(
        json.dumps(
            {
                "traces": [
                    {"task": {"data": {"name": "task-a"}}},
                    {"task": {"data": {"name": "task-b"}}},
                ]
            }
        )
    )
    records = data.trajectory_index(tmp_path)["trajectories"]
    assert len({record["id"] for record in records}) == 2
    assert [record["attempt"] for record in records] == [1, 1]
