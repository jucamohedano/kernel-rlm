"""Prepare a new immutable run from a clean checkpoint-only budget pause."""

import argparse
import copy
import fcntl
import hashlib
import json
import shutil
import time
from pathlib import Path

from kernel_rlm.cuda_docs import docs_manifest
from kernel_skyrl.preflight import load_tasks

from kernel_trl.checkpoint_state import checkpoint_metadata
from kernel_trl.live_curriculum import MODEL_REVISION, ensure_gpus_idle, source_hashes
from kernel_trl.scoring import digest, read_json, write_json


def prepare(arguments):
    parent = arguments.parent_run.resolve()
    output = arguments.output_dir.resolve()
    if output.exists() or output.is_relative_to(parent):
        raise ValueError("Continuation requires a new directory outside its preserved parent")
    state = read_json(parent / "state.json")
    inputs = read_json(parent / "inputs.json")
    version = state["policy_version"]
    trained = [row for row in state["results"] if row.get("status") == "trained"]
    order = [task["name"] for task in load_tasks(arguments.dataset)]
    if (
        state["status"] != "paused"
        or state.get("phase") != "budget_gate"
        or not state.get("budget_gate")
        or state.get("pending_task_id")
        or state.get("pending_directory")
        or state.get("failure")
        or inputs["model_revision"] != MODEL_REVISION
        or inputs["dataset_sha256"] != digest(arguments.dataset)
        or state["manifest_sha256"]
        != hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        or state["task_order"] != order
        or len(order) != 50
        or len(set(order)) != 50
        or [row["optimizer_step"] for row in trained] != list(range(1, version + 1))
        or [row["policy_version"] for row in trained] != list(range(version))
        or [row["task_id"] for row in trained] != state["completed_task_ids"]
        or trained[-1]["checkpoint"] != state["checkpoint"]
    ):
        raise ValueError("Not a clean immutable checkpoint-only budget pause")
    previous_hashes = source_hashes(arguments.parent_source_root)
    if previous_hashes != inputs["source_sha256"]:
        raise ValueError("Parent immutable runtime differs")
    current_hashes = source_hashes(arguments.source_root)
    changed = {
        name
        for name in previous_hashes.keys() | current_hashes.keys()
        if previous_hashes.get(name) != current_hashes.get(name)
    }
    allowed = {
        "kernel_rlm/child_tools.py",
        "kernel_rlm_v1/program.py",
        "kernel_trl/checkpoint_continuation.py",
    }
    if changed - allowed:
        raise ValueError("Continuation changes more than the approved child guidance")
    docs = {
        "path": str(arguments.cuda_docs),
        **docs_manifest(arguments.cuda_docs, arguments.cuda_docs_sha256),
    }
    if docs != inputs.get("cuda_docs") or docs.get("target_profile") != "sm120":
        raise ValueError("Continuation must retain the pinned hardware documentation")
    checkpoint = Path(state["checkpoint"])
    metadata = checkpoint_metadata(checkpoint, version, 1)
    native_path = Path(trained[-1]["update_directory"]).parent / "readback/report-rank-0.json"
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
        raise ValueError("Checkpoint lacks full native restore proof")
    history = read_json(parent / "provenance/verified-curve-history.json")
    by_step = {row["optimizer_step"]: row for row in history["updates"]}
    for row in trained:
        if row["optimizer_step"] not in by_step:
            episodes = []
            for directory in sorted(Path(row["trace_directory"]).glob("group-0-trajectory-*")):
                score = read_json(directory / "score.json")
                record = read_json(directory / "submission.json")
                episodes.append(
                    {
                        "reward": score["reward"],
                        "verification": score["verification"],
                        "submitted": bool(record.get("submission")),
                    }
                )
            if len(episodes) != 8 or sorted(item["reward"] for item in episodes) != sorted(
                row["rewards"]
            ):
                raise ValueError("Accepted curve evidence differs")
            by_step[row["optimizer_step"]] = {
                "optimizer_step": row["optimizer_step"],
                "task_id": row["task_id"],
                "episodes": episodes,
            }
        if by_step[row["optimizer_step"]]["task_id"] != row["task_id"]:
            raise ValueError("Inherited task history differs")
    history["updates"] = [by_step[step] for step in range(1, version + 1)]
    provenance = output / "provenance"
    provenance.mkdir(parents=True)
    for source, name in (
        (parent / "state.json", "parent-state.json"),
        (parent / "inputs.json", "parent-inputs.json"),
        (native_path, "base-native-readback.json"),
    ):
        shutil.copy2(source, provenance / name)
    write_json(provenance / "verified-curve-history.json", history)
    rows = [
        {"step": row["optimizer_step"], "loss/mean": row["metrics"]["train_loss"]}
        for row in trained
    ]
    (provenance / "history-metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    updated = copy.deepcopy(state)
    updated.update(
        status="ready",
        phase="ready",
        run_id=arguments.run_id,
        started_unix_seconds=time.time(),
        history_logged=False,
    )
    updated.pop("budget_gate", None)
    updated["continuation_parent"] = {
        "run_id": state["run_id"],
        "policy_version": version,
        "state_sha256": digest(parent / "state.json"),
        "source_commit": inputs["source_commit"],
        "reason": "User-approved65USDtotal final pass and next-run child-guidance correction",
    }
    new_inputs = copy.deepcopy(inputs)
    new_inputs.update(
        run_id=arguments.run_id,
        source_commit=arguments.source_commit,
        source_sha256=current_hashes,
        continuation_parent=updated["continuation_parent"],
    )
    updated["manifest_sha256"] = hashlib.sha256(
        json.dumps(new_inputs, sort_keys=True).encode()
    ).hexdigest()
    write_json(output / "inputs.json", new_inputs)
    write_json(output / "state.json", updated)
    write_json(
        provenance / "checkpoint-continuation.json",
        {
            "parent_state_sha256": digest(parent / "state.json"),
            "parent_inputs_sha256": digest(parent / "inputs.json"),
            "checkpoint_metadata_sha256": digest(checkpoint / "metadata.json"),
            "changed_runtime_files": sorted(changed),
            "all_prior_samples_scores_attempts_preserved": True,
            "new_rollouts": 0,
            "no_pending_group_discarded": True,
        },
    )
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "parent-run",
        "output-dir",
        "dataset",
        "parent-source-root",
        "source-root",
        "cuda-docs",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-commit", "run-id", "cuda-docs-sha256"):
        parser.add_argument("--" + name, required=True)
    arguments = parser.parse_args()
    with (arguments.output_dir.parent / "active-driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_gpus_idle()
        state = prepare(arguments)
    print(json.dumps({key: state[key] for key in ("run_id", "policy_version", "checkpoint")}))


if __name__ == "__main__":
    main()
