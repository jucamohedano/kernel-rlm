import copy
import hashlib
import json
import shutil

import pytest
from kernel_trl import loading_repair as module

from tests import test_kernel_reward_repair as fixtures


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    branch = fixtures.branch.__wrapped__(tmp_path, monkeypatch)
    from kernel_trl import reward_repair

    monkeypatch.setattr(module, "load_tasks", reward_repair.load_tasks)
    monkeypatch.setattr(module, "source_hashes", lambda path: {"new": "source"})
    monkeypatch.setattr(
        module, "docs_manifest", lambda *args: {"target_profile": "sm120", "sha256": "docs"}
    )
    state_path = branch.parent_run / "state.json"
    state = module.read_json(state_path)
    state.update(
        phase="final_scoring",
        failure={"error": "Persistent final verification infrastructure failure"},
    )
    previous = module.Path(state["results"][1]["checkpoint"])
    checkpoint = module.Path(state["results"][2]["checkpoint"])
    shutil.copytree(previous, checkpoint)
    metadata = module.read_json(checkpoint / "metadata.json")
    metadata["step"] = 3
    module.write_json(checkpoint / "metadata.json", metadata)
    native = module.read_json(previous.parent.parent / "readback/report-rank-0.json")
    native.update(global_step=3)
    native["resume"]["step"] = 3
    (checkpoint.parent.parent / "readback").mkdir()
    module.write_json(checkpoint.parent.parent / "readback/report-rank-0.json", native)
    state["checkpoint"] = str(checkpoint)
    for row in state["results"]:
        row["metrics"] = {"train_loss": 0.1}
    module.write_json(state_path, state)
    directory = module.Path(state["pending_directory"]) / "collection/rollouts"
    ledger = module.read_json(directory / "scoring-state.json")
    for index in range(8):
        path = directory / f"group-0-trajectory-{index}/score.json"
        score = {
            "reward": 0.1,
            "policy_version": 3,
            "root_calls": 1,
            "verification": {
                "verification_status": "candidate_error",
                "verification_protocol": module.VERIFICATION_PROTOCOL,
                "correct": False,
                "speedup": None,
            },
        }
        if index == 3:
            score["verification"].update(
                verification_status="infrastructure_error",
                failure_kind="reference_loading",
                error="NameError: name 'source' is not defined",
            )
            ledger["scores"][str(index)]["attempts"] = 2
        module.write_json(path, score)
        ledger["scores"][str(index)]["sha256"] = module.digest(path)
    module.write_json(directory / "scoring-state.json", ledger)
    original = module.read_json(directory / "group-0-trajectory-3/score.json")
    corrected = copy.deepcopy(original)
    corrected["verification"].pop("failure_kind")
    corrected["verification"]["verification_status"] = "candidate_error"
    branch.cuda_docs = branch.parent_run / "docs.json"
    branch.cuda_docs_sha256 = "docs"
    branch.revalidation = branch.parent_run / "revalidation.json"
    module.write_json(
        branch.revalidation,
        {
            "trajectory": 3,
            "original_score_sha256": module.digest(directory / "group-0-trajectory-3/score.json"),
            "generation_sha256": module.digest(directory / "generation-manifest.json"),
            "submission_sha256": hashlib.sha256(b"saved source").hexdigest(),
            "source_commit": branch.source_commit,
            "source_sha256": {"new": "source"},
            "score": corrected,
        },
    )
    (branch.parent_run / "provenance").mkdir()
    module.write_json(
        branch.parent_run / "provenance/verified-curve-history.json",
        {
            "updates": [
                {"optimizer_step": step, "task_id": f"task-{step - 1}", "episodes": [{}] * 8}
                for step in range(1, 4)
            ]
        },
    )
    return branch


def test_loading_continuation_preserves_samples_scores_and_parent(prepared):
    parent_hash = module.digest(prepared.parent_run / "state.json")
    previous = module.read_json(prepared.parent_run / "state.json")
    original = module.Path(previous["pending_directory"]) / "collection/rollouts"
    state = module.prepare(prepared)
    pending = module.Path(state["pending_directory"]) / "collection/rollouts"
    assert state["policy_version"] == 3
    assert state["completed_task_ids"] == previous["completed_task_ids"]
    assert state["deferred_task_ids"] == previous["deferred_task_ids"]
    assert module.digest(prepared.parent_run / "state.json") == parent_hash
    assert module.digest(original / "generation-manifest.json") == module.digest(
        pending / "generation-manifest.json"
    )
    for index in range(8):
        for name in ("task.json", "submission.json", "calls.jsonl"):
            assert module.digest(original / f"group-0-trajectory-{index}" / name) == module.digest(
                pending / f"group-0-trajectory-{index}" / name
            )
        if index != 3:
            assert module.digest(
                original / f"group-0-trajectory-{index}/score.json"
            ) == module.digest(pending / f"group-0-trajectory-{index}/score.json")
    ledger = module.read_json(pending / "scoring-state.json")
    assert ledger["scores"]["3"]["attempts"] == 3
    assert ledger["scores"]["3"]["original_attempts"] == 2
    inputs = module.read_json(prepared.output_dir / "inputs.json")
    assert inputs["cuda_docs"]["target_profile"] == "sm120"
    assert inputs["pending_generation_provenance"]["resampled"] is False
    assert (
        state["manifest_sha256"]
        == hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    )


@pytest.mark.parametrize(
    "change", ["score", "sample", "dataset", "attempts", "reference_error", "history", "active"]
)
def test_loading_continuation_rejects_changed_evidence(prepared, change, monkeypatch):
    state = module.read_json(prepared.parent_run / "state.json")
    directory = module.Path(state["pending_directory"]) / "collection/rollouts"
    if change == "score":
        (directory / "group-0-trajectory-0/score.json").write_text("{}")
    elif change == "sample":
        (directory / "group-0-trajectory-0/calls.jsonl").write_text("changed")
    elif change == "dataset":
        prepared.dataset.write_text("changed")
    elif change == "attempts":
        ledger = module.read_json(directory / "scoring-state.json")
        ledger["scores"]["3"]["attempts"] = 0
        module.write_json(directory / "scoring-state.json", ledger)
    elif change == "reference_error":
        receipt = module.read_json(prepared.revalidation)
        receipt["score"]["verification"]["error"] = "Reference failed"
        module.write_json(prepared.revalidation, receipt)
    elif change == "history":
        state["completed_task_ids"].reverse()
        module.write_json(prepared.parent_run / "state.json", state)
    else:
        state["status"] = "running"
        module.write_json(prepared.parent_run / "state.json", state)
    with pytest.raises(ValueError):
        module.prepare(prepared)
    assert not prepared.output_dir.exists()
