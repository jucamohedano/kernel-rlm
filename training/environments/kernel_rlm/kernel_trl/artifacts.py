"""Import saved TRL RLM artifacts into the unchanged local trajectory viewer."""

import argparse
import ast
import csv
import hashlib
import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else {}


def episode_trace(directory: Path, run_id: str, trace_id: str | None = None) -> dict:
    record = read_json(directory / "submission.json")
    score = read_json(directory / "score.json")
    task = read_json(directory / "task.json")
    rows = (
        [json.loads(line) for line in (directory / "calls.jsonl").read_text().splitlines()]
        if (directory / "calls.jsonl").exists()
        else []
    )
    roots = [row for row in rows if row.get("branch") == "root" and "response" in row]
    nodes = []
    calls = []
    root_index = 0
    messages = record.get("messages", [])
    for message in messages:
        original = message
        if message.get("role") == "assistant" and root_index < len(roots):
            root = roots[root_index]
            choice = root["response"]["choices"][0]
            original = dict(choice["message"])
            original["reasoning_content"] = original.get(
                "reasoning_content", original.get("reasoning", "")
            )
            calls.append(
                {
                    "node": len(nodes),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": root["response"].get("usage", {}),
                    "elapsed_seconds": root["elapsed_seconds"],
                }
            )
            root_index += 1
        nodes.append({"message": original, "parent": len(nodes) - 1 if nodes else None})
    children = [row for row in rows if row.get("branch") == "child"]
    for child in children:
        for message in child.get("request", {}).get("messages", [])[:1]:
            if message.get("role") == "system":
                nodes.append({"message": message, "parent": None})
    feedback = []
    for path in sorted(directory.glob("feedback-*.jsonl")):
        feedback.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {
        "id": run_id + ":" + (trace_id or directory.name),
        "task": {"data": {"name": task.get("name", directory.name)}},
        "nodes": nodes,
        "calls": calls,
        "rewards": {"kernel_reward": {"score": score.get("reward")}},
        "info": {
            "kernel_reward": score.get("reward"),
            "kernel_submission": record.get("submission", ""),
            "kernel_termination": record.get("termination_reason", "unknown"),
            "kernel_verify": score.get("verification", {}),
            "kernel_model_calls": calls,
            "kernel_subcalls": children,
            "repl_transcript": messages,
            "feedback_results": feedback,
            "program_log": (directory / "program.log").read_text()
            if (directory / "program.log").exists()
            else "",
            "transport_failure": read_json(directory / "failure.json"),
        },
    }


def import_run(source: Path, destination: Path, source_commit: str | None = None):
    destination.mkdir(parents=True, exist_ok=False)
    report = read_json(source / "report.json")
    run_id = report["run_id"]
    groups_path = source / "rollouts/groups.json"
    if groups_path.exists():
        groups = read_json(groups_path)
        if not groups:
            groups = report.get("groups", [])
    else:
        groups = report.get("groups", [])
    if not isinstance(groups, list):
        groups = []
    if not groups and report.get("results"):
        groups = [
            {
                "task": result.get("task_id"),
                "rewards": result.get("rewards", result.get("group_record", {}).get("rewards", [])),
                "admitted": result.get("status") == "trained",
                "policy_version": result.get("policy_version"),
                "reason": result.get("status"),
            }
            for result in report["results"]
        ]
    report.update(
        title="TRL L1 · exact-turn GRPO · TP2 inference",
        model_id="poolside/Laguna-XS.2",
        model_revision="a397bde04501591d0ac3d74a5d9d0a4ec298ffb8",
        gpu_topology="Two shared L40S; vLLM and TRL run in separate phases",
        groups=groups,
    )
    if isinstance(report["groups"], dict):
        report["groups"] = []
    write_json(destination / "report.json", report)
    episodes = sorted(
        {
            path.parent
            for path in source.rglob("submission.json")
            if path.parent.name.startswith("group-") and "trajectory-" in path.parent.name
        }
    )
    traces = [
        episode_trace(directory, run_id, str(directory.relative_to(source)))
        for directory in episodes
    ]
    (destination / "traces.jsonl").write_text(json.dumps({"traces": traces}) + "\n")
    admitted_tasks = {group.get("task") for group in report["groups"] if group.get("admitted")}
    effective_traces = [
        trace
        for trace in traces
        if trace.get("task", {}).get("data", {}).get("name") in admitted_tasks
    ]
    if effective_traces:
        effective = destination / "effective"
        effective.mkdir()
        (effective / "traces.jsonl").write_text(json.dumps({"traces": effective_traces}) + "\n")
    metrics_by_step = {}
    for result in report.get("results", []):
        step = result.get("optimizer_step")
        values = result.get("metrics", {})
        if not values and result.get("update_reports"):
            values = result["update_reports"][0].get("metrics", {})
        if step is not None and isinstance(values.get("train_loss"), int | float):
            metrics_by_step[step] = {
                "producer": "trainer",
                "step": step,
                "loss/mean": float(values["train_loss"]),
                "source": "native TRL report",
                **{
                    key: float(values[key])
                    for key in (
                        "train_runtime",
                        "train_steps_per_second",
                        "train_samples_per_second",
                    )
                    if isinstance(values.get(key), int | float)
                },
            }
    for log_path in sorted(source.rglob("*.log")):
        if log_path.name != "training.log" and not log_path.name.startswith("update"):
            continue
        match = re.search(r"step-(\d+)", str(log_path.parent))
        step = int(match.group(1)) if match else 1
        for line in log_path.read_text(errors="replace").splitlines():
            if not line.startswith("{'loss':"):
                continue
            value = ast.literal_eval(line)
            entry = metrics_by_step.setdefault(
                step,
                {
                    "producer": "trainer",
                    "step": step,
                    "loss/mean": float(value["loss"]),
                },
            )
            entry.update(
                {
                    key: float(item)
                    for key, item in value.items()
                    if key != "loss"
                    and (
                        key in {"entropy", "grad_norm", "learning_rate"}
                        or key.startswith("clip_ratio/")
                    )
                }
            )
            entry.setdefault("source", "native TRL report with trainer log metrics")
    metrics = [metrics_by_step[step] for step in sorted(metrics_by_step)]
    if report.get("passed") and not metrics:
        metrics.append(
            {
                "producer": "trainer",
                "step": report.get("optimizer_steps", 1),
                "loss/mean": report["reports"][0]["metrics"]["train_loss"],
            }
        )
    (destination / "metrics.jsonl").write_text(
        "".join(json.dumps(value) + "\n" for value in metrics)
    )
    samples = []
    for path in sorted(source.rglob("*telemetry.csv")):
        for row in list(csv.reader(path.open()))[1:]:
            if len(row) != 5:
                continue
            raw_timestamp = row[1].strip()
            try:
                parsed_timestamp = datetime.strptime(raw_timestamp, "%Y/%m/%d %H:%M:%S.%f")
            except ValueError:
                parsed_timestamp = datetime.strptime(raw_timestamp, "%Y/%m/%d %H:%M:%S")
            timestamp = parsed_timestamp.replace(tzinfo=UTC).isoformat()
            samples.append(
                {
                    "timestamp_utc": timestamp,
                    "gpus": [
                        {
                            "index": int(row[0]),
                            "memory_used_mib": float(row[2].split()[0]),
                            "memory_total_mib": 46068,
                            "gpu_utilization_percent": float(row[3].split()[0]),
                        }
                    ],
                }
            )
    samples.sort(key=lambda item: item["timestamp_utc"])
    (destination / "gpu_stats.jsonl").write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples)
    )
    captured = read_json(source / "inputs.json")
    inputs = []
    repository = source / "repo"
    environment = repository / "training/environments/kernel_rlm"
    for relative, digest in captured.get("source_sha256", {}).items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Captured input path escapes the snapshot")
        if relative_path.parts[0] == "rlm" or relative_path.parts[:3] == (
            "training",
            "src",
            "rlm_train",
        ):
            path = repository / relative_path
            commit_path = str(relative_path)
        else:
            path = environment / relative_path
            commit_path = f"training/environments/kernel_rlm/{relative_path}"
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Saved execution source changed: {relative}")
        if source_commit is not None:
            original = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(Path(__file__).resolve().parents[4]),
                    "show",
                    f"{source_commit}:{commit_path}",
                ]
            )
            if hashlib.sha256(original).hexdigest() != digest:
                raise ValueError("Claimed commit does not match captured source")
        target = destination / "inputs" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        inputs.append(
            {
                "path": str(target.relative_to(destination)),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "role": "executed source",
            }
        )
    data = source / "inputs/lvl1_seed42.jsonl"
    if not data.exists():
        data = source / "lvl1_seed42.jsonl"
    if data.exists():
        dataset_sha256 = hashlib.sha256(data.read_bytes()).hexdigest()
        if captured.get("dataset_sha256") and dataset_sha256 != captured["dataset_sha256"]:
            raise ValueError("Saved dataset does not match the captured SHA-256")
        target = destination / "inputs/lvl1_seed42.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data, target)
        inputs.append(
            {
                "path": "inputs/lvl1_seed42.jsonl",
                "sha256": dataset_sha256,
                "bytes": data.stat().st_size,
                "role": "dataset",
            }
        )
    events_path = source / "events.jsonl"
    events = (
        [json.loads(line) for line in events_path.read_text().splitlines()]
        if events_path.exists()
        else []
    )
    completed = next((row for row in events if row["phase"] == "training_completed"), None)
    if completed is not None:
        report["elapsed_seconds"] = (
            datetime.fromisoformat(completed["utc"]) - datetime.fromisoformat(events[0]["utc"])
        ).total_seconds()
        write_json(destination / "report.json", report)
    if (source / "training-runtime.json").exists():
        write_json(
            destination / "runtime.json",
            {
                "trainer": read_json(source / "training-runtime.json"),
                "inference": read_json(source / "inference-runtime.json"),
            },
        )
    (destination / "events.jsonl").write_text(
        "".join(json.dumps({**row, "event": row["phase"]}) + "\n" for row in events)
    )
    manifest = {
        "run_id": run_id,
        "experiment_id": captured.get("experiment_id", report.get("experiment_id", "trl-rlm-l40s")),
        "capture_kind": "source hashes captured before rollout; normalized offline after execution",
        "created_at_utc": events[0]["utc"] if events else report.get("started_at_utc"),
        "inputs": inputs,
        "pins": {
            "model_revision": report["model_revision"],
            "trl": "1.9.2",
            "vllm": "0.25.1",
            "source_equivalent_rlm_commit": source_commit,
        },
        "datasets": {"sha256": captured.get("dataset_sha256"), "upstream_revision": None},
        "config": {
            "group_size": len(report["groups"][0].get("rewards", []))
            if report.get("groups")
            else None,
            "root_tokens": 4096,
            "context": captured.get("context_tokens", 12288),
            "max_turns": captured.get("max_turns", 8),
            "enable_thinking": captured.get("enable_thinking"),
            "cuda_docs": captured.get("cuda_docs"),
            "children": 2,
            "max_optimizer_steps": report.get("optimizer_steps", 0),
        },
        "parent": None,
        "artifacts": {"weights": "Remain on CloudRift; not downloaded"},
    }
    write_json(destination / "provenance.json", manifest)
    return {
        "run_id": run_id,
        "trajectories": len(traces),
        "metrics": len(metrics),
        "destination": str(destination),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    print(json.dumps(import_run(args.source, args.destination, args.source_commit)))


if __name__ == "__main__":
    main()
