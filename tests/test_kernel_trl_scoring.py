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


@pytest.mark.parametrize("matching", [True, False])
def test_terminal_capacity_outcome_is_not_a_fabricated_token_sample(saved, matching):
    module, directory, _task = saved
    episode = directory / "group-0-trajectory-0"
    details = {"prompt_tokens": 12200, "context_tokens": 12288, "requested_tokens": 4096}
    record = {
        "submission": "",
        "termination_reason": "context_exhausted",
        "context_exhaustion": details,
    }
    (episode / "submission.json").write_text(json.dumps(record))
    path = episode / "calls.jsonl"
    calls = [json.loads(line) for line in path.read_text().splitlines()]
    calls.append(
        {
            "policy_version": 1,
            "branch": "root",
            "outcome": "context_exhausted",
            "context_exhaustion": details if matching else {},
        }
    )
    path.write_text("\n".join(json.dumps(call) for call in calls))
    (directory / "generation-manifest.json").unlink()
    manifest = module.capture_generation(directory, "task-a", 1, 4)
    if matching:
        episodes = module.load_episodes(directory, manifest)
        assert len(episodes) == 4
        assert len(episodes[0]["samples"]) == 1
        assert not episodes[0]["record"]["submission"]
    else:
        with pytest.raises(ValueError, match="matching terminal root"):
            module.load_episodes(directory, manifest)


def test_long_context_survives_durable_scoring(saved):
    module, directory, _task = saved
    path = directory / "group-0-trajectory-0/calls.jsonl"
    calls = [json.loads(line) for line in path.read_text().splitlines()]
    calls[0]["response"]["prompt_token_ids"] = [2] * 13000
    path.write_text("\n".join(json.dumps(call) for call in calls))
    (directory / "generation-manifest.json").unlink()
    module.capture_generation(directory, "task-a", 1, 4, context_tokens=16384)
    scorer = AsyncMock(return_value=(0.85, {"verification_status": "completed", "correct": True}))
    result = asyncio.run(module.score_generated(directory, 1, "task-a", scorer))
    assert result["admitted"] is True
    group = json.loads((directory / "training-group.json").read_text())
    assert group["context_tokens"] == 16384
    assert len(group["rows"][0]["prompt_ids"]) == 13000


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


@pytest.mark.parametrize(
    "status",
    ["infrastructure_error", "runner_crash", "compile_runner_crash", "invalid_compile_output"],
)
def test_persistent_infrastructure_failure_has_bounded_retries(saved, status):
    module, directory, _task = saved
    scorer = AsyncMock(return_value=(0.1, {"verification_status": status}))
    result = asyncio.run(module.score_generated(directory, 1, "task-a", scorer))
    assert result["reason"] == "infrastructure_error"
    assert scorer.await_count == 2
    assert not (directory / "training-group.json").exists()
    asyncio.run(module.score_generated(directory, 1, "task-a", scorer))
    assert scorer.await_count == 2


def test_attributed_candidate_crash_is_not_retried_as_infrastructure(saved):
    module, directory, _task = saved
    scorer = AsyncMock(
        return_value=(0.1, {"verification_status": "candidate_crash", "correct": False})
    )
    result = asyncio.run(module.score_generated(directory, 1, "task-a", scorer))
    assert result["admitted"]
    assert result["infrastructure_errors"] == []
    assert scorer.await_count == 1
    asyncio.run(module.score_generated(directory, 1, "task-a", scorer))
    assert scorer.await_count == 1


def test_all_failed_candidates_remain_a_weak_complete_group_without_updates(saved):
    module, directory, _task = saved
    for index in range(4):
        path = directory / f"group-0-trajectory-{index}/score.json"
        saved_score = json.loads(path.read_text())
        saved_score.update(
            reward=0.1, verification={"verification_status": "candidate_crash", "correct": False}
        )
        path.write_text(json.dumps(saved_score))
    scorer = AsyncMock()
    result = asyncio.run(module.score_generated(directory, 1, "task-a", scorer))
    assert result["reason"] == "weak_signal"
    assert result["infrastructure_errors"] == []
    assert not (directory / "training-group.json").exists()
    scorer.assert_not_called()


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
