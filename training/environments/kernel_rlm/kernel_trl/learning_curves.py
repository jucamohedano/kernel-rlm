"""CPU-only, evidence-derived plotting; never controls or modifies training."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import sqlite3
import statistics
import time
from pathlib import Path

NATIVE_KEYS = (
    "train/entropy",
    "train/grad_norm",
    "train/learning_rate",
    "train/clip_ratio/region_mean",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def accepted_results(state: dict) -> list[dict]:
    results = sorted(
        (item for item in state["results"] if item.get("status") == "trained"),
        key=lambda item: item["optimizer_step"],
    )
    if [item["optimizer_step"] for item in results] != list(range(1, state["policy_version"] + 1)):
        raise ValueError("Accepted update history is incomplete or duplicated")
    return results


def summarize(result: dict, episodes: list[dict]) -> dict:
    rewards = result["rewards"]
    if len(episodes) != 8 or len(rewards) != 8:
        raise ValueError("Plotting requires the complete eight-member group")
    observed = sorted(item["reward"] for item in episodes)
    if any(not math.isfinite(value) for value in observed + rewards):
        raise ValueError("Rewards must be finite")
    if any(
        abs(actual - expected) > 1e-10
        for actual, expected in zip(observed, sorted(rewards), strict=True)
    ):
        raise ValueError("Trajectory scores differ from the accepted update")
    correct = [item for item in episodes if item["verification"].get("correct") is True]
    speeds = [
        item["verification"]["speedup"]
        for item in correct
        if isinstance(item["verification"].get("speedup"), int | float)
        and math.isfinite(item["verification"]["speedup"])
        and item["verification"]["speedup"] > 0
    ]
    metrics = {
        "curves/optimizer_step": result["optimizer_step"],
        "curves/sampled_policy_version": result["policy_version"],
        "train/loss": result["metrics"]["train_loss"],
        "outcomes/correct_fraction": len(correct) / 8,
        "outcomes/mean_reward": statistics.mean(rewards),
        "outcomes/reward_std": statistics.pstdev(rewards),
        "outcomes/submitted_fraction": sum(item["submitted"] for item in episodes) / 8,
        "outcomes/faster_correct_fraction": sum(speed > 1 for speed in speeds) / 8,
    }
    if speeds:
        metrics["outcomes/median_speedup_correct"] = statistics.median(speeds)
    audit = result.get("probability_audit", {})
    for source, target in (
        ("mean_absolute_log_ratio", "audit/mean_abs_log_ratio"),
        ("outside_ratio_0_8_to_1_2_fraction", "audit/outside_ratio_fraction"),
    ):
        if source in audit:
            metrics[target] = audit[source]
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("Curve metrics must be finite")
    return metrics


def native_metrics(connection: sqlite3.Connection, result: dict) -> dict:
    directory = result.get("trace_directory", result.get("source_run", ""))
    prefix = Path(directory.split("/groups/")[0]).name
    name = f"{prefix}-step-{result['optimizer_step']}"
    metrics = {}
    for (payload,) in connection.execute(
        "SELECT metrics FROM metrics WHERE run_name=? ORDER BY step", (name,)
    ):
        row = json.loads(payload)
        for key in NATIVE_KEYS:
            if key in row and isinstance(row[key], int | float) and math.isfinite(row[key]):
                metrics[key] = row[key]
    return metrics


def live_episodes(result: dict, run_directory: Path) -> list[dict]:
    directory = Path(result["trace_directory"]).resolve()
    if not directory.is_relative_to(run_directory.resolve()):
        raise ValueError("New update traces must belong to the monitored run")
    episodes = []
    for path in sorted(directory.glob("group-0-trajectory-*")):
        score = read_json(path / "score.json")
        submission = read_json(path / "submission.json")
        episodes.append(
            {
                "reward": score["reward"],
                "verification": score["verification"],
                "submitted": bool(submission.get("submission")),
            }
        )
    return episodes


def published_steps(connection: sqlite3.Connection, name: str) -> set[int]:
    return {
        row["curves/optimizer_step"]
        for (payload,) in connection.execute(
            "SELECT metrics FROM metrics WHERE run_name=?", (name,)
        )
        if "curves/optimizer_step" in (row := json.loads(payload))
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--history-sha256", required=True)
    parser.add_argument("--trackio-dir", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--name", default="laguna-l1-learning-curves")
    parser.add_argument("--poll-seconds", type=float, default=30)
    arguments = parser.parse_args()
    if arguments.poll_seconds < 5 or not math.isfinite(arguments.poll_seconds):
        raise ValueError("CPU observer poll interval must be at least five seconds")
    history_bytes = arguments.history.read_bytes()
    if hashlib.sha256(history_bytes).hexdigest() != arguments.history_sha256:
        raise ValueError("Historical trajectory evidence checksum differs")
    history = {item["optimizer_step"]: item for item in json.loads(history_bytes)["updates"]}
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    lock = (arguments.output_directory / "observer.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ["TRACKIO_DIR"] = str(arguments.trackio_dir.resolve())
    import trackio

    connection = sqlite3.connect(
        f"file:{arguments.trackio_dir.resolve()}/laguna-kernel-rlm.db?mode=ro", uri=True
    )
    completed = published_steps(connection, arguments.name)
    trackio.init(
        project="laguna-kernel-rlm",
        name=arguments.name,
        resume="allow",
        space_id=None,
        auto_log_gpu=False,
        auto_log_cpu=False,
        embed=False,
        config={
            "source_run": arguments.run_directory.name,
            "read_only_training_observer": True,
            "history_sha256": arguments.history_sha256,
            "scope": "Accepted training groups; changing tasks, not fixed held-out evaluation",
            "hardware_boundary": "Steps1-19 L40S; step20 onward RTX; speed ratios not hardware-paired",
            "reference_kl_measured": False,
        },
    )
    try:
        while True:
            state = read_json(arguments.run_directory / "state.json")
            for result in accepted_results(state):
                step = result["optimizer_step"]
                if step in completed:
                    continue
                if step in history:
                    original = history[step]
                    if original["task_id"] != result["task_id"]:
                        raise ValueError("Historical task differs from accepted policy history")
                    episodes = original["episodes"]
                else:
                    episodes = live_episodes(result, arguments.run_directory)
                metrics = summarize(result, episodes)
                metrics.update(native_metrics(connection, result))
                metrics["curves/inherited"] = int(step in history)
                metrics["curves/rtx_hardware"] = int(step > 19)
                trackio.log(metrics, step=step)
                with (arguments.output_directory / "evidence.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps({"task_id": result["task_id"], "metrics": metrics}) + "\n"
                    )
                completed.add(step)
                print(json.dumps({"published_step": step, "metrics": metrics}), flush=True)
            if state["status"] in {"paused", "completed", "failed"}:
                break
            time.sleep(arguments.poll_seconds)
    finally:
        trackio.finish()
        connection.close()
        lock.close()


if __name__ == "__main__":
    main()
