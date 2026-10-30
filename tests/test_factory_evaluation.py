import json

import pytest

from training.environments.kernel_rlm.kernel_factory import (
    finalize_evaluation,
    prepare_evaluation,
    start_evaluation,
)


def test_evaluation_lifecycle_captures_sources_and_artifacts(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "tasks.jsonl").write_text('{"_task_id":"a","prompt":"x"}\n')
    (project / "runner.py").write_text("print('candidate')\n")
    (project / "protocol.md").write_text("Answer contract\n")
    import subprocess

    subprocess.run(["git", "init", "-q", str(project)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "add",
            ".",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    output = tmp_path / "runs"
    manifest = prepare_evaluation(
        project,
        "eval-1",
        "run-1",
        output,
        {"id": "org/model@rev"},
        {"id": "tasks", "revision": "abc"},
        {"temperature": 0},
        "tasks.jsonl",
        ["runner.py"],
        ["protocol.md"],
        {"python": "3.12"},
        {"answer": "json"},
    )
    start_evaluation(manifest, output)
    directory = output / "run-1"
    (directory / "responses.jsonl").write_text('{"response":"ok"}\n')
    result = finalize_evaluation(
        manifest,
        output,
        {"status": "passed"},
        ["responses.jsonl"],
        {"destination": "backup://verified", "verified": True},
    )
    assert result["artifacts"][0]["bytes"] > 0
    events = [json.loads(row) for row in (directory / "events.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["prepared", "started", "finalized"]
    provenance = json.loads((directory / "provenance.json").read_text())
    assert provenance["model"]["id"] == "org/model@rev"
    assert (directory / "worktree.diff").is_file()


def test_evaluation_rejects_changed_snapshot_and_missing_backup(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "tasks.jsonl").write_text("{}\n")
    import subprocess

    subprocess.run(["git", "init", "-q", str(project)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "add",
            ".",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    manifest = prepare_evaluation(
        project,
        "eval",
        "run",
        tmp_path / "runs",
        {"id": "model"},
        {"id": "data"},
        {},
        "tasks.jsonl",
        [],
    )
    snapshot = tmp_path / "runs/run/inputs/tasks.jsonl"
    snapshot.write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        start_evaluation(manifest, tmp_path / "runs")
    with pytest.raises(ValueError, match="verified backup"):
        finalize_evaluation(manifest, tmp_path / "runs", {}, [], {})
