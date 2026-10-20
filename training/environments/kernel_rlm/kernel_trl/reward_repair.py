"""Prepare an explicit checkpoint branch without changing its preserved parent."""

import argparse
import copy
import fcntl
import hashlib
import json
import shutil
import time
from pathlib import Path

from kernel_rlm.submission_contract import validate_submission_source
from kernel_rlm.verification_state import VERIFICATION_PROTOCOL
from kernel_skyrl.contract import terminal_reward
from kernel_skyrl.preflight import load_tasks

from kernel_trl.checkpoint_state import checkpoint_metadata
from kernel_trl.live_curriculum import MODEL_REVISION, ensure_gpus_idle, source_hashes
from kernel_trl.scoring import INFRASTRUCTURE_STATUSES, digest, load_episodes, read_json, write_json


def prepare_branch(arguments) -> dict:
    parent = arguments.parent_run.resolve()
    output = arguments.output_dir.resolve()
    if output.exists() or output.is_relative_to(parent):
        raise ValueError("Branch must be a new directory outside the preserved parent")
    previous = read_json(parent / "state.json")
    original_inputs = read_json(parent / "inputs.json")
    version = arguments.base_step
    if type(version) is not int or version < 1:
        raise ValueError("Branch checkpoint must be a positive step")
    trained = [row for row in previous["results"] if row.get("status") == "trained"]
    order = [task["name"] for task in load_tasks(arguments.dataset)]
    if (
        previous["status"] != "paused"
        or previous["policy_version"] != version + 1
        or original_inputs["model_revision"] != MODEL_REVISION
        or original_inputs["dataset_sha256"] != digest(arguments.dataset)
        or previous["manifest_sha256"]
        != hashlib.sha256(json.dumps(original_inputs, sort_keys=True).encode()).hexdigest()
        or previous["task_order"] != order
        or len(order) != 50
        or len(set(order)) != 50
        or [row["optimizer_step"] for row in trained] != list(range(1, version + 2))
        or [row["policy_version"] for row in trained] != list(range(version + 1))
        or [row["task_id"] for row in trained] != previous["completed_task_ids"]
        or previous["pending_task_id"] != arguments.defer_task
        or not arguments.defer_reason.strip()
    ):
        raise ValueError("Branch changes parent history, dataset, or operator deferral")
    checkpoint = Path(trained[version - 1]["checkpoint"])
    metadata = checkpoint_metadata(checkpoint, version, 1)
    native_path = (
        Path(trained[version - 1]["update_directory"]).parent / "readback/report-rank-0.json"
    )
    native = read_json(native_path)
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
        raise ValueError("Base checkpoint lacks complete native restore proof")
    repaired = trained[-1]
    original_rollouts = Path(repaired["trace_directory"])
    generation = read_json(original_rollouts / "generation-manifest.json")
    episodes = load_episodes(original_rollouts, generation)
    task = next(
        task for task in load_tasks(arguments.dataset) if task["name"] == generation["task"]
    )
    if any(episode["task"] != task for episode in episodes):
        raise ValueError("Saved reference task differs from the unchanged dataset")
    revalidation_path = arguments.revalidation_directory / "report.json"
    revalidation = read_json(revalidation_path)
    results = revalidation["results"]
    if (
        generation["policy_version"] != version
        or generation["task"] != repaired["task_id"]
        or len(episodes) != 8
        or revalidation["status"] != "completed"
        or revalidation["verification_protocol"] != VERIFICATION_PROTOCOL
        or revalidation["sampled_policy"] != version
        or revalidation["generation_sha256"]
        != digest(original_rollouts / "generation-manifest.json")
        or not revalidation["original_scores_unchanged"]
        or not revalidation["original_generation_and_ledger_unchanged"]
        or [row["trajectory"] for row in results] != list(range(8))
    ):
        raise ValueError("Revalidation does not match the original sampled group")
    runtime_hashes = {}
    paths = list((arguments.source_root / "kernel_rlm").rglob("*.py")) + [
        arguments.source_root / "kernel_skyrl/generator.py",
        arguments.source_root / "kernel_skyrl/contract.py",
    ]
    for path in paths:
        relative = path.relative_to(arguments.source_root)
        expected = digest(arguments.revalidation_source_root / relative)
        if digest(path) != expected:
            raise ValueError("Revalidation verifier code differs from branch verifier")
        runtime_hashes[str(relative)] = expected
    ledger = read_json(original_rollouts / "scoring-state.json")
    if ledger["generation_sha256"] != digest(original_rollouts / "generation-manifest.json"):
        raise ValueError("Original scoring ledger differs from sampled group")
    for index, (episode, result) in enumerate(zip(episodes, results, strict=True)):
        path = episode["directory"] / "score.json"
        verification = result["verification"]
        submitted = episode["record"].get("submission", "")
        valid = submitted and validate_submission_source(submitted)[0]
        expected_reward = (
            terminal_reward(verification, episode["task"].get("speed_reward_weight", 0.1))
            if valid
            else 0.0
        )
        if (
            digest(path) != ledger["scores"][str(index)]["sha256"]
            or digest(path) != result["original_score_sha256"]
            or read_json(path)["reward"] != result["original_reward"]
            or verification.get("verification_protocol") != VERIFICATION_PROTOCOL
            or verification.get("verification_status") in INFRASTRUCTURE_STATUSES
            or result["revalidated_reward"] != expected_reward
        ):
            raise ValueError("Revalidation changed samples, reward rules, or score provenance")
    failed_rollouts = Path(previous["pending_directory"]) / "collection/rollouts"
    failed_generation = read_json(failed_rollouts / "generation-manifest.json")
    load_episodes(failed_rollouts, failed_generation)
    if (
        failed_generation["task"] != arguments.defer_task
        or failed_generation["policy_version"] != previous["policy_version"]
    ):
        raise ValueError("Deferred group differs from operator-selected task")
    state = copy.deepcopy(previous)
    state.update(
        status="ready",
        policy_version=version,
        checkpoint=str(checkpoint),
        completed_task_ids=previous["completed_task_ids"][:version],
        results=[
            row
            for row in previous["results"]
            if not row.get("optimizer_step") or row["optimizer_step"] <= version
        ],
        pending_task_id=repaired["task_id"],
        history_logged=False,
        run_id=arguments.run_id,
        started_unix_seconds=time.time(),
    )
    for key in ("failure", "budget_gate", "migration_pending"):
        state.pop(key, None)
    state["continuation_parent"] = {
        "run_id": previous["run_id"],
        "policy_version": version,
        "original_parent_policy_version": previous["policy_version"],
        "state_sha256": digest(parent / "state.json"),
        "source_commit": original_inputs["source_commit"],
        "reason": "Operator-approved branch before reward-shortcut update",
    }
    if arguments.defer_task not in state["deferred_task_ids"]:
        state["deferred_task_ids"].append(arguments.defer_task)
    state.setdefault("deferred_groups", []).append(
        {
            "task_id": arguments.defer_task,
            "policy_version": previous["policy_version"],
            "reason": "operator_deferred_infrastructure",
            "operator_reason": arguments.defer_reason,
            "trace_directory": str(failed_rollouts),
            "generation_sha256": digest(failed_rollouts / "generation-manifest.json"),
            "scoring_state_sha256": digest(failed_rollouts / "scoring-state.json"),
        }
    )
    receipt = {
        "parent_state_sha256": digest(parent / "state.json"),
        "parent_inputs_sha256": digest(parent / "inputs.json"),
        "base_checkpoint": str(checkpoint),
        "base_metadata_sha256": digest(checkpoint / "metadata.json"),
        "revalidation_report_sha256": digest(revalidation_path),
        "revalidation_source_commit": revalidation["source_commit"],
        "verification_runtime_hashes": runtime_hashes,
        "preserved_old_descendant": repaired["checkpoint"],
        "operator_deferred_task": arguments.defer_task,
        "new_rollouts": 0,
    }
    pending = output / "groups" / f"step-{version + 1:03d}-{repaired['task_id']}-revalidated"
    shutil.copytree(original_rollouts.parent, pending / "collection")
    destination = pending / "collection/rollouts"
    for name in ("training-group.json", "groups.json"):
        path = destination / name
        if path.exists():
            path.replace(path.with_name(path.stem + "-before-revalidation.json"))
    shutil.copy2(
        original_rollouts / "scoring-state.json",
        destination / "scoring-state-before-revalidation.json",
    )
    updated = copy.deepcopy(ledger)
    for index, (episode, result) in enumerate(zip(episodes, results, strict=True)):
        path = destination / episode["directory"].name / "score.json"
        shutil.copy2(path, path.with_name("score-before-revalidation.json"))
        write_json(
            path,
            {
                "reward": result["revalidated_reward"],
                "verification": result["verification"],
                "policy_version": version,
                "root_calls": len(episode["samples"]),
            },
        )
        original_attempts = ledger["scores"][str(index)]["attempts"]
        updated["scores"][str(index)] = {
            "sha256": digest(path),
            "attempts": original_attempts + 1,
            "original_attempts": original_attempts,
            "explicit_protocol_revalidations": 1,
        }
    write_json(destination / "scoring-state.json", updated)
    state["pending_directory"] = str(pending)
    provenance = output / "provenance"
    provenance.mkdir()
    for source, name in (
        (parent / "state.json", "parent-state.json"),
        (parent / "inputs.json", "parent-inputs.json"),
        (revalidation_path, "revalidation-report.json"),
        (native_path, "base-native-readback.json"),
    ):
        shutil.copy2(source, provenance / name)
    write_json(provenance / "reward-repair.json", receipt)
    inputs = copy.deepcopy(original_inputs)
    inputs.update(
        run_id=arguments.run_id,
        experiment_id="trl-l1-verifier-v2-repair",
        source_commit=arguments.source_commit,
        source_sha256=source_hashes(arguments.source_root),
        verification_protocol=VERIFICATION_PROTOCOL,
        continuation_parent=state["continuation_parent"],
        reward_repair=receipt,
        source_pending_protocol_retained=False,
    )
    state["manifest_sha256"] = hashlib.sha256(
        json.dumps(inputs, sort_keys=True).encode()
    ).hexdigest()
    write_json(output / "inputs.json", inputs)
    write_json(output / "state.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "parent-run",
        "output-dir",
        "dataset",
        "source-root",
        "revalidation-directory",
        "revalidation-source-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-commit", "run-id", "defer-task", "defer-reason"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--base-step", type=int, required=True)
    arguments = parser.parse_args()
    with (arguments.output_dir.parent / "active-driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_gpus_idle()
        state = prepare_branch(arguments)
    print(
        json.dumps(
            {
                key: state[key]
                for key in ("run_id", "policy_version", "pending_task_id", "deferred_task_ids")
            }
        )
    )


if __name__ == "__main__":
    main()
