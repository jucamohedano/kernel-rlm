import importlib
import json
from pathlib import Path


def artifacts_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "training/environments/kernel_rlm"))
    return importlib.import_module("kernel_trl.artifacts")


def make_episode(path, identity):
    path.mkdir(parents=True)
    (path / "task.json").write_text(json.dumps({"name": "same-task", **identity}))
    (path / "submission.json").write_text(
        json.dumps({"messages": [{"role": "assistant", "content": "ok"}], **identity})
    )
    (path / "score.json").write_text(json.dumps({"reward": 1, **identity}))
    (path / "calls.jsonl").write_text(
        json.dumps(
            {
                "branch": "root",
                "response": {
                    "choices": [{"message": {"role": "assistant"}}],
                    "usage": {"prompt_tokens": 3},
                },
                "elapsed_seconds": 0.4,
                "execution_run_id": "launch",
                "root_turn_id": "root-1",
                "call_id": "call-1",
                "parent_tool_call_id": "tool-1",
                **identity,
            }
        )
        + "\n"
    )


def test_effective_uses_admitted_exact_provenance_and_preserves_raw_call(tmp_path, monkeypatch):
    artifacts = artifacts_module(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    (source / "report.json").write_text(
        json.dumps(
            {
                "run_id": "launch",
                "groups": [
                    {
                        "task": "same-task",
                        "admitted": True,
                        "group_id": "g1",
                        "episodes": [{"episode_id": "e1", "attempt_id": "a1"}],
                    }
                ],
            }
        )
    )
    (source / "inputs.json").write_text("{}")
    rollouts = source / "rollouts"
    make_episode(
        rollouts / "group-0-trajectory-0",
        {"group_id": "g1", "episode_id": "e1", "attempt_id": "a1"},
    )
    make_episode(
        rollouts / "group-0-trajectory-1",
        {"group_id": "g2", "episode_id": "e2", "attempt_id": "a1"},
    )
    destination = tmp_path / "normalized"
    artifacts.import_run(source, destination)
    all_traces = json.loads((destination / "traces.jsonl").read_text())["traces"]
    effective = json.loads((destination / "effective/traces.jsonl").read_text())["traces"]
    assert len(all_traces) == 2
    assert [trace["group_id"] for trace in effective] == ["g1"]
    assert effective[0]["execution_run_id"] == "launch"
    assert effective[0]["info"]["kernel_raw_call_records"][0]["root_turn_id"] == "root-1"
    assert effective[0]["calls"][0]["call_id"] == "call-1"
    assert effective[0]["calls"][0]["parent_tool_call_id"] == "tool-1"
    assert effective[0]["calls"][0]["usage"]["prompt_tokens"] == 3


def test_call_ids_stay_on_calls_and_failed_roots_are_counted(tmp_path, monkeypatch):
    artifacts = artifacts_module(monkeypatch)
    episode = tmp_path / "group-0-trajectory-0"
    make_episode(episode, {})
    rows = [
        {
            "branch": "root",
            "call_id": "ok-call",
            "root_turn_id": "ok-turn",
            "response": {
                "choices": [{"message": {"role": "assistant"}}],
                "usage": {"completion_tokens": 2},
            },
            "elapsed_seconds": 0.5,
        },
        {
            "branch": "root",
            "call_id": "failed-call",
            "root_turn_id": "failed-turn",
            "status": "failed",
            "error": "transport",
        },
    ]
    (episode / "calls.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    trace = artifacts.episode_trace(episode, "run")
    assert "root_turn_id" not in trace
    assert trace["calls"][0]["call_id"] == "ok-call"
    assert trace["calls"][1]["call_id"] == "failed-call"
    assert trace["calls"][1]["node"] is None
    assert [row["call_id"] for row in trace["info"]["kernel_raw_call_records"]] == [
        "ok-call",
        "failed-call",
    ]
    assert "request" not in trace["info"]["kernel_raw_call_records"][0]


def test_explicit_assistant_message_index_beats_historical_position(tmp_path, monkeypatch):
    artifacts = artifacts_module(monkeypatch)
    episode = tmp_path / "group-0-trajectory-0"
    make_episode(episode, {})
    submission = {
        "messages": [
            {"role": "assistant", "content": "application message"},
            {"role": "assistant", "content": "inference message"},
        ]
    }
    (episode / "submission.json").write_text(json.dumps(submission))
    rows = [
        {
            "branch": "root",
            "assistant_message_index": 1,
            "call_id": "mapped-call",
            "response": {"choices": [{"message": {"role": "assistant", "content": "api"}}]},
        },
    ]
    (episode / "calls.jsonl").write_text(json.dumps(rows[0]) + "\n")
    trace = artifacts.episode_trace(episode, "run")
    assert trace["calls"][0]["node"] == 1
    assert trace["nodes"][0]["message"]["content"] == "application message"
    assert trace["nodes"][1]["message"]["content"] == "api"


def test_historical_without_identity_not_attributed_and_unknown_defaults_omitted(
    tmp_path, monkeypatch
):
    artifacts = artifacts_module(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    (source / "report.json").write_text(
        json.dumps({"run_id": "old", "groups": [{"task": "sample", "admitted": True}]})
    )
    (source / "inputs.json").write_text("{}")
    make_episode(source / "rollouts/group-0-trajectory-0", {})
    destination = tmp_path / "normalized"
    artifacts.import_run(source, destination)
    assert not (destination / "effective").exists()
    manifest = json.loads((destination / "provenance.json").read_text())
    assert "model_revision" not in manifest["pins"]
    assert "model_id" not in json.loads((destination / "report.json").read_text())
    assert "context" not in manifest["config"]
    assert "learner_world_size" not in manifest["config"]


def test_historical_effective_uses_exact_trained_trace_directory(tmp_path, monkeypatch):
    artifacts = artifacts_module(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    admitted_path = "groups/step-002-same-task-attempt-01/collection/rollouts"
    rejected_path = "groups/step-003-same-task-attempt-02/collection/rollouts"
    (source / "report.json").write_text(
        json.dumps(
            {
                "run_id": "launch",
                "groups": [
                    {"task": "same-task", "admitted": True},
                    {"task": "same-task", "admitted": False},
                ],
                "results": [
                    {
                        "task_id": "same-task",
                        "status": "trained",
                        "trace_directory": "/remote/run/" + admitted_path,
                    },
                    {
                        "task_id": "same-task",
                        "status": "needs_diagnosis",
                        "trace_directory": "/remote/run/" + rejected_path,
                    },
                ],
            }
        )
    )
    (source / "inputs.json").write_text("{}")
    make_episode(source / admitted_path / "group-0-trajectory-0", {})
    make_episode(source / rejected_path / "group-0-trajectory-0", {})
    destination = tmp_path / "normalized"
    artifacts.import_run(source, destination)
    traces = json.loads((destination / "effective/traces.jsonl").read_text())["traces"]
    assert len(traces) == 1
    assert traces[0]["id"].endswith(admitted_path + "/group-0-trajectory-0")
    assert "group_id" not in traces[0]


def test_trace_directory_evidence_requires_safe_exact_group_path(tmp_path, monkeypatch):
    artifacts = artifacts_module(monkeypatch)
    directory = {("groups", "step-002-task-attempt-01", "collection", "rollouts")}
    assert artifacts.trace_matches_directory(
        "/remote/run/groups/step-002-task-attempt-01/collection/rollouts/group-0-trajectory-0",
        directory,
    )
    assert not artifacts.trace_matches_directory(
        "groups/step-002-task-attempt-010/collection/rollouts/group-0-trajectory-0",
        directory,
    )
    assert not artifacts.trace_matches_directory(
        "groups/../groups/step-002-task-attempt-01/collection/rollouts/group-0-trajectory-0",
        directory,
    )


def test_full_update_scalars_and_trackio_refs_are_retained(tmp_path, monkeypatch):
    artifacts = artifacts_module(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    (source / "report.json").write_text(
        json.dumps(
            {
                "run_id": "launch",
                "optimizer_steps": 1,
                "results": [
                    {
                        "optimizer_step": 1,
                        "execution_run_id": "launch",
                        "producer": "rlm_update",
                        "group_id": "g1",
                        "attempt_id": 2,
                        "trackio_run_name": "launch-rlm_update",
                        "update_reports": [
                            {
                                "producer": "rlm_update",
                                "trackio_run_name": "run-leaf",
                                "metrics": {
                                    "train_loss": 0.25,
                                    "entropy": 0.7,
                                    "grad_norm": 1.5,
                                    "learning_rate": 0.0001,
                                    "extra_metric": 6,
                                },
                            }
                        ],
                        "metrics": {
                            "train_loss": 0.25,
                            "entropy": 0.7,
                            "grad_norm": 1.5,
                            "learning_rate": 0.0001,
                            "extra_metric": 6,
                        },
                    }
                ],
            }
        )
    )
    (source / "inputs.json").write_text("{}")
    destination = tmp_path / "normalized"
    artifacts.import_run(source, destination)
    metric = json.loads((destination / "metrics.jsonl").read_text().splitlines()[0])
    assert metric["loss/mean"] == 0.25
    assert metric["entropy"] == 0.7 and metric["grad_norm"] == 1.5
    assert metric["learning_rate"] == 0.0001 and metric["extra_metric"] == 6
    assert metric["execution_run_id"] == "launch"
    assert metric["trackio_run_name"] == "run-leaf"
