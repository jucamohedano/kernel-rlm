"""Preserve a paused, fully generated group across an explicit loading-owner repair."""

import argparse
import copy
import fcntl
import hashlib
import json
import shutil
import time
from pathlib import Path

from kernel_rlm.cuda_docs import docs_manifest
from kernel_rlm.verification_state import VERIFICATION_PROTOCOL
from kernel_rollout.contract import terminal_reward
from kernel_rollout.tasks import load_tasks

from kernel_trl.checkpoint_state import checkpoint_metadata
from kernel_trl.live_curriculum import MODEL_REVISION, ensure_gpus_idle, source_hashes
from kernel_trl.scoring import INFRASTRUCTURE_STATUSES, digest, load_episodes, read_json, write_json


def validate_correction(original, corrected):
    before = original["verification"]
    after = corrected["verification"]
    if (
        before.get("verification_status") != "infrastructure_error"
        or before.get("failure_kind") != "reference_loading"
        or before.get("error") != "NameError: name 'source' is not defined"
        or after.get("verification_status") != "candidate_error"
        or after.get("error") != before["error"]
        or after.get("verification_protocol") != VERIFICATION_PROTOCOL
        or after.get("correct") is not False
        or after.get("speedup") is not None
        or corrected["reward"] != 0.1
        or terminal_reward(after, 0.1) != 0.1
        or corrected["policy_version"] != original["policy_version"]
        or corrected["root_calls"] != original["root_calls"]
    ):
        raise ValueError("Repair is not the diagnosed candidate loading NameError")


def prepare(arguments):
    parent = arguments.parent_run.resolve()
    output = arguments.output_dir.resolve()
    if output.exists() or output.is_relative_to(parent):
        raise ValueError("Continuation must preserve its parent in a new directory")
    previous = read_json(parent / "state.json")
    inputs = read_json(parent / "inputs.json")
    trained = [row for row in previous["results"] if row.get("status") == "trained"]
    version = previous["policy_version"]
    tasks = load_tasks(arguments.dataset)
    order = [task["name"] for task in tasks]
    if (
        previous["status"] != "paused"
        or previous.get("phase") != "final_scoring"
        or previous.get("failure", {}).get("error")
        != "Persistent final verification infrastructure failure"
        or inputs["model_revision"] != MODEL_REVISION
        or inputs["dataset_sha256"] != digest(arguments.dataset)
        or previous["manifest_sha256"]
        != hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        or order != previous["task_order"]
        or len(order) != 50
        or len(set(order)) != 50
        or [row["optimizer_step"] for row in trained] != list(range(1, version + 1))
        or [row["policy_version"] for row in trained] != list(range(version))
        or [row["task_id"] for row in trained] != previous["completed_task_ids"]
        or trained[-1]["checkpoint"] != previous["checkpoint"]
    ):
        raise ValueError("Paused state, history or unchanged dataset differs")
    metadata = checkpoint_metadata(Path(previous["checkpoint"]), version, 1)
    native = read_json(Path(trained[-1]["update_directory"]).parent / "readback/report-rank-0.json")
    resume = native["resume"]
    if (
        metadata["adapter_tensors"] != 320
        or len(metadata["sha256"]) != 3
        or native["status"] != "readback"
        or native["global_step"] != version
        or native["optimizer_steps"] != 0
        or resume["step"] != version
        or resume["adapter_shards_bitwise_equal"] != 320
        or resume["momentum_shards_restored"] != 320
        or not resume["full_rng_and_scheduler_restored"]
        or not resume["muon_step_restored"]
    ):
        raise ValueError("Checkpoint lacks native restore proof")
    docs = {
        "path": str(arguments.cuda_docs),
        **docs_manifest(arguments.cuda_docs, arguments.cuda_docs_sha256),
    }
    if docs.get("target_profile") != "sm120":
        raise ValueError("Continuation requires hardware-scoped docs")
    pending = Path(previous["pending_directory"])
    directory = pending / "collection/rollouts"
    generation = read_json(directory / "generation-manifest.json")
    episodes = load_episodes(directory, generation)
    ledger = read_json(directory / "scoring-state.json")
    if (
        len(episodes) != 8
        or generation["policy_version"] != version
        or generation["task"] != previous["pending_task_id"]
        or ledger["generation_sha256"] != digest(directory / "generation-manifest.json")
        or set(ledger["scores"]) != {str(index) for index in range(8)}
        or any(
            episode["task"] != next(task for task in tasks if task["name"] == generation["task"])
            for episode in episodes
        )
    ):
        raise ValueError("Original pending group differs")
    scores = [read_json(episode["directory"] / "score.json") for episode in episodes]
    for index, episode in enumerate(episodes):
        if digest(episode["directory"] / "score.json") != ledger["scores"][str(index)]["sha256"]:
            raise ValueError("Original durable score differs")
    errors = [
        index
        for index, score in enumerate(scores)
        if score["verification"].get("verification_status") in INFRASTRUCTURE_STATUSES
    ]
    receipt = read_json(arguments.revalidation)
    if (
        errors != [3]
        or ledger["scores"]["3"]["attempts"] != 2
        or receipt["trajectory"] != 3
        or receipt["original_score_sha256"] != digest(episodes[3]["directory"] / "score.json")
        or receipt["generation_sha256"] != digest(directory / "generation-manifest.json")
        or receipt["submission_sha256"]
        != hashlib.sha256(episodes[3]["record"]["submission"].encode()).hexdigest()
        or receipt["source_commit"] != arguments.source_commit
        or receipt["source_sha256"] != source_hashes(arguments.source_root)
    ):
        raise ValueError("Exact one-member repair evidence differs")
    corrected = receipt["score"]
    validate_correction(scores[3], corrected)
    destination = output / "groups" / pending.name
    shutil.copytree(pending / "collection", destination / "collection")
    target = destination / "collection/rollouts"
    shutil.copy2(target / "scoring-state.json", target / "scoring-state-before-loading-repair.json")
    score_path = target / "group-0-trajectory-3/score.json"
    shutil.copy2(score_path, score_path.with_name("score-before-loading-repair.json"))
    write_json(score_path, corrected)
    ledger["scores"]["3"] = {
        "sha256": digest(score_path),
        "attempts": 3,
        "original_attempts": 2,
        "explicit_loading_repair_revalidations": 1,
    }
    write_json(target / "scoring-state.json", ledger)
    state = copy.deepcopy(previous)
    state.update(
        status="ready",
        run_id=arguments.run_id,
        pending_directory=str(destination),
        history_logged=False,
        started_unix_seconds=time.time(),
    )
    state.pop("failure", None)
    state["continuation_parent"] = {
        "run_id": previous["run_id"],
        "policy_version": version,
        "state_sha256": digest(parent / "state.json"),
        "source_commit": inputs["source_commit"],
        "reason": "Diagnosed candidate loading ownership repair plus approved hardware docs correction",
    }
    provenance = output / "provenance"
    provenance.mkdir()
    for source, name in (
        (parent / "state.json", "parent-state.json"),
        (parent / "inputs.json", "parent-inputs.json"),
        (arguments.revalidation, "loading-revalidation.json"),
    ):
        shutil.copy2(source, provenance / name)
    rows = [
        {"step": row["optimizer_step"], "loss/mean": row["metrics"]["train_loss"]}
        for row in trained
    ]
    (provenance / "history-metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    history = read_json(parent / "provenance/verified-curve-history.json")
    by_step = {row["optimizer_step"]: row for row in history["updates"]}
    for row in trained:
        if row["optimizer_step"] not in by_step:
            saved = []
            for episode_path in sorted(Path(row["trace_directory"]).glob("group-0-trajectory-*")):
                score = read_json(episode_path / "score.json")
                record = read_json(episode_path / "submission.json")
                saved.append(
                    {
                        "reward": score["reward"],
                        "verification": score["verification"],
                        "submitted": bool(record.get("submission")),
                    }
                )
            if len(saved) != 8:
                raise ValueError("Inherited curve history is incomplete")
            by_step[row["optimizer_step"]] = {
                "optimizer_step": row["optimizer_step"],
                "task_id": row["task_id"],
                "episodes": saved,
            }
    history["updates"] = [by_step[step] for step in range(1, version + 1)]
    write_json(provenance / "verified-curve-history.json", history)
    updated_inputs = copy.deepcopy(inputs)
    updated_inputs.update(
        run_id=arguments.run_id,
        source_commit=arguments.source_commit,
        source_sha256=source_hashes(arguments.source_root),
        cuda_docs=docs,
        continuation_parent=state["continuation_parent"],
    )
    updated_inputs["pending_generation_provenance"] = {
        "source_commit": inputs["source_commit"],
        "cuda_docs": inputs.get("cuda_docs"),
        "generation_sha256": ledger["generation_sha256"],
        "sampled_policy_version": version,
        "resampled": False,
        "successful_scores_reused": 7,
        "explicit_loading_revalidations": 1,
    }
    state["manifest_sha256"] = hashlib.sha256(
        json.dumps(updated_inputs, sort_keys=True).encode()
    ).hexdigest()
    write_json(output / "inputs.json", updated_inputs)
    write_json(output / "state.json", state)
    write_json(
        provenance / "loading-repair.json",
        {
            "parent_state_sha256": digest(parent / "state.json"),
            "parent_inputs_sha256": digest(parent / "inputs.json"),
            "native_metadata_sha256": digest(Path(previous["checkpoint"]) / "metadata.json"),
            "revalidation_sha256": digest(arguments.revalidation),
            "old_scores_preserved": True,
            "new_rollouts": 0,
            "docs_active_for_fresh_groups_only": True,
        },
    )
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-run", "output-dir", "dataset", "source-root", "cuda-docs", "revalidation"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-commit", "run-id", "cuda-docs-sha256"):
        parser.add_argument("--" + name, required=True)
    arguments = parser.parse_args()
    with (arguments.output_dir.parent / "active-driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_gpus_idle()
        state = prepare(arguments)
    print(json.dumps({key: state[key] for key in ("run_id", "policy_version", "pending_task_id")}))


if __name__ == "__main__":
    main()
