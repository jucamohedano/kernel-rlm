"""CPU-only coverage and durable revalidation tests for the RTX continuation."""

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from kernel_trl import rtx_curriculum as module


def test_commands_keep_measured_split_card_protocol():
    arguments = SimpleNamespace(
        python="python",
        checkpoint=Path("base"),
        dataset=Path("tasks"),
        cuda_docs=Path("docs"),
        cuda_docs_sha256="hash",
        run_id="campaign-123",
        child_prompt_tokens=4999,
        child_reply_tokens=8193,
        child_total_tokens=12289,
    )
    command = module.collector_command(arguments, Path("out"), "task-a", 19, "group-a", "attempt-2")
    for name, value in (
        ("--execution-run-id", "campaign-123"),
        ("--group-id", "group-a"),
        ("--attempt-id", "attempt-2"),
        ("--rollouts", "8"),
        ("--context-tokens", "16384"),
        ("--max-turns", "10"),
        ("--child-prompt-tokens", "4999"),
        ("--child-reply-tokens", "8193"),
        ("--child-total-tokens", "12289"),
        ("--episode-timeout-s", "2400"),
        ("--group-timeout-s", "4800"),
        ("--report-to", "trackio"),
    ):
        assert command[command.index(name) + 1] == value
    assert "--dedicated-inference" in command and "--child-tools" in command
    with pytest.raises(ValueError, match="audit"):
        module.learner_command(arguments, Path("update"), Path("state"), Path("group"), "update")
    update = module.learner_command(
        arguments, Path("update"), Path("state"), Path("group"), "update", Path("audit")
    )
    assert "--nproc-per-node=1" in update
    assert update[update.index("--learner-world-size") + 1] == "1"
    readback = module.learner_command(
        arguments, Path("readback"), Path("state"), Path("group"), "readback"
    )
    assert "--readback-only" in readback and "--group" not in readback


@pytest.fixture
def pending(monkeypatch, tmp_path):
    directory = tmp_path / "rollouts"
    directory.mkdir()
    module.write_json(
        directory / "generation-manifest.json", {"task": "pending", "policy_version": 19}
    )
    episodes = []
    scores = {}
    for index in range(8):
        episode = directory / f"group-0-trajectory-{index}"
        episode.mkdir()
        module.write_json(
            episode / "score.json",
            {
                "reward": 0.1,
                "policy_version": 19,
                "verification": {"verification_status": "runner_crash"},
            },
        )
        scores[str(index)] = {"sha256": module.digest(episode / "score.json"), "attempts": 2}
        episodes.append(
            {
                "directory": episode,
                "task": {"name": "pending"},
                "record": {"submission": "same source"},
                "samples": ["same exact tokens"],
            }
        )
    module.write_json(
        directory / "scoring-state.json",
        {
            "generation_sha256": module.digest(directory / "generation-manifest.json"),
            "scores": scores,
        },
    )
    monkeypatch.setattr(module, "load_episodes", lambda *args: episodes)
    monkeypatch.setattr(module, "phase_deadline", lambda *args, **kwargs: 999)
    tracker = SimpleNamespace(log=lambda values: None)
    return directory, episodes, tracker


def test_migration_adds_one_explicit_attempt_without_resetting_history(pending, monkeypatch):
    directory, episodes, tracker = pending
    original = (directory / "scoring-state.json").read_bytes()
    scorer = AsyncMock(
        return_value=(0.1, {"verification_status": "candidate_crash", "correct": False})
    )
    monkeypatch.setattr(module, "score_submission", scorer)
    asyncio.run(module.revalidate_pending(directory, 19, {}, tracker))
    assert scorer.await_count == 8
    assert (directory / "scoring-state-before-migration.json").read_bytes() == original
    state = module.read_json(directory / "scoring-state.json")
    assert all(entry["attempts"] == 3 for entry in state["scores"].values())
    assert all(
        (episode["directory"] / "score-before-migration.json").exists() for episode in episodes
    )
    asyncio.run(module.revalidate_pending(directory, 19, {}, tracker))
    assert scorer.await_count == 8


def test_interrupted_migration_does_not_silently_retry(pending, monkeypatch):
    directory, _episodes, tracker = pending
    scorer = AsyncMock(side_effect=asyncio.CancelledError())
    monkeypatch.setattr(module, "score_submission", scorer)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(module.revalidate_pending(directory, 19, {}, tracker))
    with pytest.raises(ValueError, match="Interrupted"):
        asyncio.run(module.revalidate_pending(directory, 19, {}, tracker))
    assert scorer.await_count == 1


def test_true_migration_infrastructure_error_stops_before_siblings(pending, monkeypatch):
    directory, _episodes, tracker = pending
    scorer = AsyncMock(
        return_value=(0.1, {"verification_status": "infrastructure_error", "correct": None})
    )
    monkeypatch.setattr(module, "score_submission", scorer)
    with pytest.raises(ValueError, match="infrastructure"):
        asyncio.run(module.revalidate_pending(directory, 19, {}, tracker))
    assert scorer.await_count == 1


def test_migration_refuses_changed_original_scores(pending, monkeypatch):
    directory, episodes, tracker = pending
    (episodes[0]["directory"] / "score.json").write_text("tampered")
    scorer = AsyncMock()
    monkeypatch.setattr(module, "score_submission", scorer)
    with pytest.raises(ValueError, match="Original"):
        asyncio.run(module.revalidate_pending(directory, 19, {}, tracker))
    scorer.assert_not_called()


@pytest.fixture
def migration(monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    order = [f"task-{index}" for index in range(50)]
    dataset = tmp_path / "tasks"
    dataset.write_text("same full dataset")
    tasks = [{"name": name, "context": name} for name in order]
    monkeypatch.setattr(module, "load_tasks", lambda path: tasks)
    inputs = {
        "model_revision": module.MODEL_REVISION,
        "dataset_sha256": module.digest(dataset),
        "source_commit": "original",
    }
    state = {
        "run_id": "old",
        "status": "paused",
        "policy_version": 19,
        "completed_task_ids": order[:19],
        "pending_task_id": order[19],
        "task_order": order,
        "results": [
            {
                "status": "trained",
                "task_id": task,
                "optimizer_step": index + 1,
                "policy_version": index,
            }
            for index, task in enumerate(order[:19])
        ],
        "manifest_sha256": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
    }
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    monkeypatch.setattr(
        module,
        "checkpoint_metadata",
        lambda directory, version, ranks: {"sha256": {"adapter_model.safetensors": "same"}},
    )
    module.write_json(bundle / "parent-state.json", state)
    module.write_json(bundle / "parent-inputs.json", inputs)
    module.write_json(
        bundle / "source-checkpoint-metadata.json",
        {"step": 19, "world_size": 2, "sha256": {"adapter_model.safetensors": "same"}},
    )
    module.write_json(
        bundle / "native-readback.json",
        {
            "status": "passed",
            "goal_policy_version": 19,
            "optimizer_updates": 0,
            "native_resume": {
                "adapter_shards_bitwise_equal": 320,
                "momentum_shards_restored": 320,
                "full_rng_and_scheduler_restored": True,
            },
        },
    )
    module.write_json(
        bundle / "native-readback-launch.json",
        {"command": ["python", "--resume-from", str(checkpoint)]},
    )
    module.write_json(
        bundle / "pending/collection/rollouts/generation-manifest.json",
        {"task": order[19], "policy_version": 19, "episodes": [None] * 8},
    )
    monkeypatch.setattr(module, "load_episodes", lambda *args: [{"task": tasks[19]}] * 8)
    return bundle, dataset, checkpoint, order


@pytest.mark.parametrize(
    "fault", [None, "coverage", "policy", "dataset", "checkpoint", "restore", "partial"]
)
def test_migration_validates_complete_history_and_exact_source(migration, fault):
    bundle, dataset, checkpoint, order = migration
    if fault == "coverage":
        state = module.read_json(bundle / "parent-state.json")
        state["completed_task_ids"].pop()
        module.write_json(bundle / "parent-state.json", state)
    elif fault == "policy":
        state = module.read_json(bundle / "parent-state.json")
        state["results"][1]["policy_version"] = 999
        module.write_json(bundle / "parent-state.json", state)
    elif fault == "dataset":
        dataset.write_text("different")
    elif fault == "checkpoint":
        original = module.read_json(bundle / "source-checkpoint-metadata.json")
        original["sha256"]["adapter_model.safetensors"] = "different"
        module.write_json(bundle / "source-checkpoint-metadata.json", original)
    elif fault == "restore":
        native = module.read_json(bundle / "native-readback.json")
        native["native_resume"]["momentum_shards_restored"] = 1
        module.write_json(bundle / "native-readback.json", native)
    elif fault == "partial":
        pending = module.read_json(bundle / "pending/collection/rollouts/generation-manifest.json")
        pending["episodes"].pop()
        module.write_json(bundle / "pending/collection/rollouts/generation-manifest.json", pending)
    files = {
        str(path.relative_to(bundle)): module.digest(path)
        for path in bundle.rglob("*")
        if path.is_file()
    }
    module.write_json(bundle / "manifest.json", {"files": files})
    if fault:
        with pytest.raises(ValueError):
            module.migration_state(bundle, dataset, checkpoint)
    else:
        migrated = module.migration_state(bundle, dataset, checkpoint)
        assert migrated["completed_task_ids"] == order[:19]
        assert migrated["policy_version"] == 19
        assert migrated["task_order"] == order


def test_coordinator_tracking_initializes_lineage_without_replaying_history():
    calls = []
    tracker = SimpleNamespace(
        init=lambda **kwargs: calls.append(("init", kwargs)),
        log=lambda *args, **kwargs: calls.append(("log", args)),
    )
    arguments = SimpleNamespace(
        run_id="campaign",
        source_commit="abc",
        history_metrics=Path("history.jsonl"),
        checkpoint_state=Path("parent-checkpoint"),
    )
    state = {
        "continuation_parent": {"policy_version": 19, "run_id": "parent", "state_sha256": "sha"}
    }

    module.initialize_coordinator_tracking(tracker, arguments, state, "history-sha")

    assert [kind for kind, *_ in calls] == ["init"]
    config = calls[0][1]["config"]
    assert config["run_kind"] == "coordinator"
    assert config["execution_run_id"] == "campaign"
    assert config["history_source_sha256"] == "history-sha"
    assert config["parent_execution_run_id"] == "parent"
