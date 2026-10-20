import hashlib
import json
from types import SimpleNamespace

import pytest
from kernel_trl import reward_repair as module
from kernel_trl.scoring import capture_generation


@pytest.fixture
def branch(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    tasks = [{"name": f"task-{index}", "context": "reference"} for index in range(50)]
    dataset = tmp_path / "dataset"
    dataset.write_text("unchanged dataset")
    monkeypatch.setattr(module, "load_tasks", lambda path: tasks)
    monkeypatch.setattr(module, "validate_submission_source", lambda source: (True, "ok"))
    monkeypatch.setattr(module, "source_hashes", lambda path: {"new": "source"})
    inputs = {
        "model_revision": module.MODEL_REVISION,
        "dataset_sha256": module.digest(dataset),
        "source_commit": "old",
        "run_id": "parent",
    }
    trained = []
    for step in range(1, 4):
        group = parent / "groups" / f"step-{step}"
        trained.append(
            {
                "status": "trained",
                "task_id": tasks[step - 1]["name"],
                "optimizer_step": step,
                "policy_version": step - 1,
                "checkpoint": str(group / "update" / f"checkpoint-step-{step}"),
                "update_directory": str(group / "update"),
                "trace_directory": str(group / "collection/rollouts"),
            }
        )
    checkpoint = module.Path(trained[1]["checkpoint"])
    checkpoint.mkdir(parents=True)
    for name in ["adapter_config.json", "adapter_model.safetensors", "state-rank-0.pt"]:
        (checkpoint / name).write_text(name)
    module.write_json(
        checkpoint / "metadata.json",
        {
            "step": 2,
            "world_size": 1,
            "frozen_weights_saved": False,
            "adapter_tensors": 320,
            "sha256": {path.name: module.digest(path) for path in checkpoint.iterdir()},
        },
    )
    native = module.Path(trained[1]["update_directory"]).parent / "readback/report-rank-0.json"
    native.parent.mkdir()
    module.write_json(
        native,
        {
            "status": "readback",
            "global_step": 2,
            "optimizer_steps": 0,
            "resume": {
                "step": 2,
                "adapter_shards_bitwise_equal": 320,
                "momentum_shards_restored": 320,
                "muon_step_restored": True,
                "full_rng_and_scheduler_restored": True,
            },
        },
    )

    def generated(directory, task, policy):
        directory.mkdir(parents=True)
        ledger = {"scores": {}}
        for index in range(8):
            episode = directory / f"group-0-trajectory-{index}"
            episode.mkdir()
            module.write_json(episode / "task.json", task)
            module.write_json(episode / "submission.json", {"submission": "saved source"})
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
            (episode / "calls.jsonl").write_text(
                json.dumps({"policy_version": policy, "branch": "root", "response": response})
                + "\n"
            )
            module.write_json(episode / "score.json", {"reward": 1.0, "policy_version": policy})
            ledger["scores"][str(index)] = {
                "attempts": 1,
                "sha256": module.digest(episode / "score.json"),
            }
        capture_generation(directory, task["name"], policy, 8, 16384)
        ledger["generation_sha256"] = module.digest(directory / "generation-manifest.json")
        module.write_json(directory / "scoring-state.json", ledger)
        module.write_json(directory / "training-group.json", {"old_reward": 1})

    rollouts = module.Path(trained[2]["trace_directory"])
    generated(rollouts, tasks[2], 2)
    failed = parent / "groups/failed"
    generated(failed / "collection/rollouts", tasks[3], 3)
    state = {
        "status": "paused",
        "run_id": "parent",
        "policy_version": 3,
        "task_order": [task["name"] for task in tasks],
        "completed_task_ids": [row["task_id"] for row in trained],
        "results": trained,
        "deferred_task_ids": [],
        "deferred_groups": [],
        "attempts": {"task-2": 1, "task-3": 1},
        "pending_task_id": "task-3",
        "pending_directory": str(failed),
        "failure": {"error": "timeout"},
        "manifest_sha256": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
    }
    module.write_json(parent / "state.json", state)
    module.write_json(parent / "inputs.json", inputs)
    validation = tmp_path / "validation"
    validation.mkdir()
    module.write_json(
        validation / "report.json",
        {
            "status": "completed",
            "verification_protocol": module.VERIFICATION_PROTOCOL,
            "sampled_policy": 2,
            "source_commit": "validated",
            "generation_sha256": module.digest(rollouts / "generation-manifest.json"),
            "original_scores_unchanged": True,
            "original_generation_and_ledger_unchanged": True,
            "results": [
                {
                    "trajectory": index,
                    "original_reward": 1.0,
                    "revalidated_reward": 0.1,
                    "original_score_sha256": module.digest(
                        rollouts / f"group-0-trajectory-{index}/score.json"
                    ),
                    "verification": {
                        "verification_protocol": module.VERIFICATION_PROTOCOL,
                        "verification_status": "completed",
                        "correct": False,
                    },
                }
                for index in range(8)
            ],
        },
    )
    source = tmp_path / "source"
    validated_source = tmp_path / "validated-source"
    for root in [source, validated_source]:
        for name in [
            "kernel_rlm/helper.py",
            "kernel_skyrl/generator.py",
            "kernel_skyrl/contract.py",
        ]:
            path = root / name
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_text("same verifier")
    return SimpleNamespace(
        parent_run=parent,
        output_dir=tmp_path / "new",
        dataset=dataset,
        base_step=2,
        defer_task="task-3",
        defer_reason="User explicitly approved timeout deferral",
        revalidation_directory=validation,
        revalidation_source_root=validated_source,
        source_root=source,
        run_id="new",
        source_commit="new-source",
    )


def test_branch_preserves_parent_tokens_attempts_and_explicit_deferral(branch):
    before = {
        str(path): path.read_bytes() for path in branch.parent_run.rglob("*") if path.is_file()
    }
    state = module.prepare_branch(branch)
    assert state["policy_version"] == 2
    assert state["completed_task_ids"] == ["task-0", "task-1"]
    assert state["deferred_task_ids"] == ["task-3"]
    assert state["attempts"] == {"task-2": 1, "task-3": 1}
    pending = module.Path(state["pending_directory"]) / "collection/rollouts"
    ledger = module.read_json(pending / "scoring-state.json")
    assert all(
        row["attempts"] == 2 and row["original_attempts"] == 1 for row in ledger["scores"].values()
    )
    assert module.read_json(pending / "group-0-trajectory-0/score.json")["reward"] == 0.1
    assert (
        module.read_json(pending / "group-0-trajectory-0/score-before-revalidation.json")["reward"]
        == 1
    )
    assert not (pending / "training-group.json").exists()
    assert (pending / "training-group-before-revalidation.json").exists()
    assert before == {
        str(path): path.read_bytes() for path in branch.parent_run.rglob("*") if path.is_file()
    }
    with pytest.raises(ValueError, match="new directory"):
        module.prepare_branch(branch)


@pytest.mark.parametrize("change", ["protocol", "reward", "verifier", "checkpoint", "deferral"])
def test_branch_rejects_incompatible_evidence_before_writing(branch, change):
    path = branch.revalidation_directory / "report.json"
    data = module.read_json(path)
    if change == "protocol":
        data["verification_protocol"] = "old"
    elif change == "reward":
        data["results"][0]["revalidated_reward"] = 1.0
    elif change == "verifier":
        (branch.source_root / "kernel_rlm/helper.py").write_text("different")
    elif change == "checkpoint":
        parent = module.read_json(branch.parent_run / "state.json")
        (module.Path(parent["results"][1]["checkpoint"]) / "state-rank-0.pt").write_text("changed")
    else:
        branch.defer_task = "task-4"
    module.write_json(path, data)
    with pytest.raises(ValueError):
        module.prepare_branch(branch)
    assert not branch.output_dir.exists()
