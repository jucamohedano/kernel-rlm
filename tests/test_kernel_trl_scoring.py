import asyncio
import importlib
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def saved(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "training/environments/kernel_rlm"))
    module = importlib.import_module("kernel_trl.scoring")
    directory = tmp_path / "saved"
    directory.mkdir()
    task = {"name": "task-a", "context": "reference code"}
    for index, reward in enumerate([0.75, 0.8, 0.1, 0.1]):
        episode = directory / f"group-0-trajectory-{index}"
        episode.mkdir()
        (episode / "task.json").write_text(json.dumps(task))
        (episode / "submission.json").write_text(json.dumps({"submission": f"source-{index}"}))
        response = {
            "prompt_token_ids": [1],
            "choices": [
                {
                    "token_ids": [2],
                    "logprobs": {"content": [{"token": "token_id:2", "logprob": -1.0}]},
                    "finish_reason": "tool_calls",
                }
            ],
        }
        calls = [
            {"policy_version": 1, "branch": "root", "response": response},
            {"policy_version": 1, "branch": "child", "response": {"unused_child_tokens": [999]}},
        ]
        (episode / "calls.jsonl").write_text("\n".join(json.dumps(call) for call in calls))
        score = {
            "reward": reward,
            "policy_version": 1,
            "verification": {
                "verification_status": "timeout"
                if index == 3
                else "compile_failed"
                if index == 2
                else "completed"
            },
        }
        (episode / "score.json").write_text(json.dumps(score))
    (directory / "groups.json").write_text(
        json.dumps([{"task": "task-a", "policy_version": 1, "admitted": False}])
    )
    module.capture_generation(directory, "task-a", 1, 4)
    return module, directory, task


def test_rechecks_only_failed_verification_and_never_resamples(saved):
    module, directory, _task = saved
    scorer = AsyncMock(return_value=(0.85, {"verification_status": "completed", "correct": True}))
    result = asyncio.run(module.score_generated(directory, 1, "task-a", scorer))
    assert result["admitted"] is True
    assert result["reverified_trajectories"] == [3]
    assert scorer.await_count == 1
    group = json.loads((directory / "training-group.json").read_text())
    assert group["rewards"] == [0.75, 0.8, 0.1, 0.85]
    assert group["real_rows"] == 4
    assert all(row["completion_ids"] == [2] for row in group["rows"])
    original = json.loads((directory / "group-0-trajectory-3/score-original.json").read_text())
    assert original["verification"]["verification_status"] == "timeout"
    asyncio.run(module.score_generated(directory, 1, "task-a", scorer))
    assert scorer.await_count == 1


def test_changed_sampled_tokens_fail_before_verification(saved):
    module, directory, _task = saved
    (directory / "group-0-trajectory-0/calls.jsonl").write_text("tampered")
    scorer = AsyncMock()
    with pytest.raises(ValueError, match="checksum"):
        asyncio.run(module.score_generated(directory, 1, "task-a", scorer))
    scorer.assert_not_called()


def test_wrong_policy_is_not_reused(saved):
    module, directory, _task = saved
    with pytest.raises(ValueError, match="task/policy"):
        asyncio.run(module.score_generated(directory, 2, "task-a"))


def test_persistent_infrastructure_failure_has_bounded_retries(saved):
    module, directory, _task = saved
    scorer = AsyncMock(return_value=(0.1, {"verification_status": "infrastructure_error"}))
    result = asyncio.run(module.score_generated(directory, 1, "task-a", scorer))
    assert result["reason"] == "infrastructure_error"
    assert scorer.await_count == 2
    assert not (directory / "training-group.json").exists()
    asyncio.run(module.score_generated(directory, 1, "task-a", scorer))
    assert scorer.await_count == 2


def test_recovery_preserves_original_run_and_checks_task_context(saved, tmp_path):
    module, source, task = saved
    original = (source / "group-0-trajectory-3/score.json").read_bytes()
    destination = tmp_path / "recovered"
    module.import_saved_group(source, destination, task, 1, 4)
    scorer = AsyncMock(return_value=(0.85, {"verification_status": "completed"}))
    asyncio.run(module.score_generated(destination, 1, "task-a", scorer))
    assert (source / "group-0-trajectory-3/score.json").read_bytes() == original
    assert (destination / "recovery-source.json").exists()
    with pytest.raises(ValueError, match="context"):
        module.import_saved_group(source, tmp_path / "bad", {**task, "context": "changed"}, 1, 4)


def test_successful_scores_survive_interrupted_retry(saved):
    module, directory, _task = saved
    interrupted = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(module.score_generated(directory, 1, "task-a", interrupted))
    assert len(json.loads((directory / "scoring-state.json").read_text())["scores"]) == 3
    repaired = AsyncMock(return_value=(0.85, {"verification_status": "completed"}))
    assert asyncio.run(module.score_generated(directory, 1, "task-a", repaired))["admitted"]
    assert repaired.await_count == 1
