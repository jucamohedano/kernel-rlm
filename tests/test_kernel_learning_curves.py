import sqlite3
from copy import deepcopy

import pytest
from kernel_trl.learning_curves import accepted_results, native_metrics, published_steps, summarize


@pytest.fixture
def group():
    episodes = [
        {"reward": 0.8, "verification": {"correct": True, "speedup": 2.0}, "submitted": True},
        {"reward": 0.7, "verification": {"correct": True, "speedup": 0.5}, "submitted": True},
    ] + [
        {"reward": 0.1, "verification": {"compiled": False, "correct": None}, "submitted": True}
        for _ in range(6)
    ]
    result = {
        "task_id": "task-one",
        "status": "trained",
        "optimizer_step": 1,
        "policy_version": 0,
        "rewards": [episode["reward"] for episode in episodes],
        "metrics": {"train_loss": -0.02},
    }
    return result, episodes


def test_curves_use_actual_verification_not_submission_or_loss(group):
    result, episodes = group
    metrics = summarize(result, episodes)
    assert metrics["outcomes/correct_fraction"] == 0.25
    assert metrics["outcomes/faster_correct_fraction"] == 0.125
    assert metrics["outcomes/submitted_fraction"] == 1
    assert metrics["outcomes/median_speedup_correct"] == 1.25
    assert metrics["train/loss"] == -0.02
    assert not any("kl" in key for key in metrics)
    assert "train/entropy" not in metrics


def test_no_correct_speed_is_missing_not_zero(group):
    result, episodes = group
    for episode in episodes:
        episode["verification"]["correct"] = False
    assert "outcomes/median_speedup_correct" not in summarize(result, episodes)


@pytest.mark.parametrize("change", ["partial", "scores", "nan"])
def test_invalid_evidence_does_not_publish(group, change):
    result, episodes = group
    if change == "partial":
        episodes.pop()
    elif change == "scores":
        episodes[0]["reward"] = 0.2
    else:
        result["metrics"]["train_loss"] = float("nan")
    with pytest.raises(ValueError):
        summarize(result, episodes)


def test_weak_and_incomplete_groups_are_not_optimizer_updates(group):
    result, _ = group
    weak = {"status": "weak_signal", "optimizer_step": None}
    incomplete = {"status": "incomplete_preserved"}
    state = {"policy_version": 1, "results": [weak, result, incomplete]}
    assert accepted_results(state) == [result]
    state["results"].append(deepcopy(result))
    with pytest.raises(ValueError):
        accepted_results(state)


def test_native_metrics_do_not_mix_pilot_branches_or_invent_kl(group):
    result, _ = group
    result["trace_directory"] = "/run/goal/groups/one/collection/rollouts"
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE metrics(run_name TEXT,step INTEGER,metrics TEXT)")
    connection.executemany(
        "INSERT INTO metrics VALUES(?,?,?)",
        [
            ("pilot-step-1", 1, '{"train/entropy":99}'),
            ("goal-step-1", 1, '{"train/entropy":0.3,"train/kl":1}'),
            ("curves", 1, '{"curves/optimizer_step":1}'),
            ("curves", 2, '{"unrelated":2}'),
        ],
    )
    assert native_metrics(connection, result) == {"train/entropy": 0.3}
    assert published_steps(connection, "curves") == {1}
