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


def compact_call_record(row: dict) -> dict:
    fields = {
        "branch",
        "execution_run_id",
        "group_id",
        "attempt_id",
        "episode_id",
        "policy_version",
        "root_turn_id",
        "call_id",
        "model_call_id",
        "parent_turn_id",
        "parent_tool_call_id",
        "producer",
        "policy_ref",
        "policy_revision",
        "usage",
        "elapsed_seconds",
        "started_at",
        "completed_at",
        "status",
        "error",
        "error_type",
        "failure",
        "request_id",
        "model",
        "provider",
        "finish_reason",
        "message_index",
        "assistant_message_index",
    }
    return {key: value for key, value in row.items() if key in fields}


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
    failed_roots = [row for row in rows if row.get("branch") == "root" and "response" not in row]
    call_fields = (
        "execution_run_id",
        "root_turn_id",
        "call_id",
        "model_call_id",
        "producer",
        "group_id",
        "episode_id",
        "attempt_id",
        "policy_version",
        "optimizer_step",
        "parent_turn_id",
        "parent_tool_call_id",
        "policy_ref",
        "policy_revision",
        "usage",
        "elapsed_seconds",
        "started_at",
        "completed_at",
        "status",
        "error",
        "error_type",
        "failure",
        "request_id",
    )
    calls = []
    nodes = []
    root_index = 0
    messages = record.get("messages", [])
    explicit_mapping = any(
        "message_index" in root or "assistant_message_index" in root for root in roots
    )
    roots_by_message = {
        root.get("assistant_message_index", root.get("message_index")): root
        for root in roots
        if root.get("assistant_message_index", root.get("message_index")) is not None
    }
    for message_index, message in enumerate(messages):
        original = message
        matched_root = None
        call_node = None
        if message.get("role") == "assistant":
            if explicit_mapping:
                matched_root = roots_by_message.get(message_index)
            elif root_index < len(roots):
                matched_root = roots[root_index]
        if matched_root is not None:
            choice = matched_root["response"]["choices"][0]
            original = dict(choice["message"])
            original["reasoning_content"] = original.get(
                "reasoning_content", original.get("reasoning", "")
            )
            call_node = len(nodes)
            calls.append(
                {
                    **{key: matched_root[key] for key in call_fields if key in matched_root},
                    "node": call_node,
                    "status": matched_root.get("status", "completed"),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": matched_root.get("usage", matched_root["response"].get("usage", {})),
                }
            )
            root_index += 1
        node = {"message": original, "parent": len(nodes) - 1 if nodes else None}
        if matched_root is not None:
            node.update({key: matched_root[key] for key in call_fields if key in matched_root})
        nodes.append(node)
    calls.extend(
        {
            **{key: root[key] for key in call_fields if key in root},
            "node": None,
            "status": root.get("status", "failed"),
        }
        for root in failed_roots
    )
    children = [row for row in rows if row.get("branch") == "child"]
    for child in children:
        for message in child.get("request", {}).get("messages", [])[:1]:
            if message.get("role") == "system":
                nodes.append({"message": message, "parent": None})
    feedback = []
    for path in sorted(directory.glob("feedback-*.jsonl")):
        feedback.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    episode_fields = {
        "execution_run_id",
        "group_id",
        "episode_id",
        "attempt_id",
        "optimizer_step",
        "policy_version",
        "producer",
        "policy_ref",
        "policy_revision",
    }
    provenance = {
        key: value
        for source in (record, task, score)
        for key, value in source.items()
        if key in episode_fields
    }
    for key in ("execution_run_id", "group_id", "episode_id", "attempt_id"):
        values = {row[key] for row in rows if key in row}
        if key not in provenance and len(values) == 1:
            provenance[key] = values.pop()
    return {
        "id": run_id + ":" + (trace_id or directory.name),
        **provenance,
        "task": {"data": {"name": task.get("name", directory.name)}},
        "nodes": nodes,
        "calls": calls,
        "rewards": {"kernel_reward": {"score": score.get("reward")}},
        "info": {
            **provenance,
            "kernel_reward": score.get("reward"),
            "kernel_submission": record.get("submission", ""),
            "kernel_termination": record.get("termination_reason", "unknown"),
            "kernel_verify": score.get("verification", {}),
            "kernel_model_calls": calls,
            "kernel_raw_call_records": [compact_call_record(row) for row in rows],
            "kernel_subcalls": children,
            "kernel_child_sessions": record.get("subcalls", []),
            "kernel_root_actions": record.get("root_actions", []),
            "kernel_child_progress": [
                json.loads(line)
                for line in (directory / "child-progress.jsonl").read_text().splitlines()
                if line.strip()
            ]
            if (directory / "child-progress.jsonl").exists()
            else [],
            "repl_transcript": messages,
            "feedback_results": feedback,
            "program_log": (directory / "program.log").read_text()
            if (directory / "program.log").exists()
            else "",
            "transport_failure": read_json(directory / "failure.json"),
        },
    }


def trace_identity(record: dict) -> tuple:
    info = record.get("info", {})
    return tuple(record.get(key, info.get(key)) for key in ("group_id", "episode_id", "attempt_id"))


def admitted_identities(groups: list) -> set[tuple]:
    identities = set()
    for group in groups:
        if not isinstance(group, dict) or not group.get("admitted"):
            continue
        group_id = group.get("group_id")
        episode_rows = group.get("episodes", [])
        if isinstance(episode_rows, list):
            for episode in episode_rows:
                if not isinstance(episode, dict):
                    continue
                episode_id = episode.get("episode_id")
                attempt_id = episode.get("attempt_id", group.get("attempt_id"))
                if group_id is not None and episode_id is not None and attempt_id is not None:
                    identities.add((group_id, episode_id, attempt_id))
        episode_ids = group.get("episode_ids", [])
        attempts = group.get("attempt_ids", [])
        if isinstance(episode_ids, list) and isinstance(attempts, list):
            for episode_id, attempt_id in zip(episode_ids, attempts, strict=False):
                if group_id is not None and episode_id is not None and attempt_id is not None:
                    identities.add((group_id, episode_id, attempt_id))
        identity = (group_id, group.get("episode_id"), group.get("attempt_id"))
        if all(value is not None for value in identity):
            identities.add(identity)
    return identities


def source_trace_directory(value: str | Path) -> tuple[str, ...] | None:
    raw_parts = str(value).replace("\\", "/").split("/")
    parts = tuple(part for part in raw_parts if part not in {"", "."})
    if ".." in parts:
        return None
    group_indexes = [index for index, part in enumerate(parts) if part == "groups"]
    candidates = [parts[index:] for index in group_indexes]
    matches = [
        candidate
        for candidate in candidates
        if len(candidate) >= 4 and candidate[-2:] == ("collection", "rollouts")
    ]
    return matches[0] if len(matches) == 1 else None


def admitted_trace_directories(results: list) -> set[tuple[str, ...]]:
    return {
        relative
        for result in results
        if isinstance(result, dict) and result.get("status") == "trained"
        if isinstance((path := result.get("trace_directory")), str)
        if (relative := source_trace_directory(path)) is not None
    }


def trace_matches_directory(trace_id: str, directories: set[tuple[str, ...]]) -> bool:
    raw_parts = str(trace_id).replace("\\", "/").split("/")
    parts = tuple(part for part in raw_parts if part not in {"", "."})
    if ".." in parts:
        return False
    group_indexes = [index for index, part in enumerate(parts) if part == "groups"]
    candidates = [parts[index:] for index in group_indexes]
    return any(
        candidate[: len(directory)] == directory
        and len(candidate) > len(directory)
        and candidate[len(directory)].startswith("group-")
        and "trajectory-" in candidate[len(directory)]
        for candidate in candidates
        for directory in directories
    )


def import_run(source: Path, destination: Path, source_commit: str | None = None):
    destination.mkdir(parents=True, exist_ok=False)
    report = read_json(source / "report.json")
    captured = read_json(source / "inputs.json")
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
    report.update(groups=groups)
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
    admitted = admitted_identities(report["groups"])
    admitted_directories = admitted_trace_directories(report.get("results", []))
    effective_traces = []
    for trace in traces:
        identity = trace_identity(trace)
        if any(value is not None for value in identity):
            if all(value is not None for value in identity) and identity in admitted:
                effective_traces.append(trace)
            continue
        if trace_matches_directory(trace["id"].split(":", 1)[-1], admitted_directories):
            effective_traces.append(trace)
    if effective_traces:
        effective = destination / "effective"
        effective.mkdir()
        (effective / "traces.jsonl").write_text(json.dumps({"traces": effective_traces}) + "\n")
    metrics_by_step = {}
    for result in report.get("results", []):
        step = result.get("optimizer_step")
        values = dict(result.get("metrics", {}))
        provenance_keys = {
            "execution_run_id",
            "producer",
            "group_id",
            "attempt_id",
            "policy_version",
            "trackio_run_id",
            "trackio_run_name",
            "trackio_space_id",
            "trackio_project",
            "trackio_url",
            "trackio_step",
            "trackio_config",
            "trackio_metadata",
        }
        provenance = {key: value for key, value in result.items() if key in provenance_keys}
        for update_report in result.get("update_reports", []):
            values.update(update_report.get("metrics", {}))
            provenance.update(
                {key: value for key, value in update_report.items() if key in provenance_keys}
            )
        if step is not None and isinstance(values.get("train_loss"), int | float):
            metrics_by_step[step] = {
                **provenance,
                "producer": result.get("producer", "trainer"),
                "step": step,
                "optimizer_step": result.get("optimizer_step", step),
                "loss/mean": float(values["train_loss"]),
                "source": "native TRL report",
                **{
                    key: float(value)
                    for key, value in values.items()
                    if isinstance(value, int | float) and not isinstance(value, bool)
                },
            }
    for log_path in sorted(source.rglob("*.log")):
        if log_path.name != "training.log" and not log_path.name.startswith("update"):
            continue
        match = re.search(r"step-(\d+)", str(log_path.parent))
        if match:
            step = int(match.group(1))
        elif len(metrics_by_step) == 1:
            step = next(iter(metrics_by_step))
        elif not metrics_by_step:
            step = 1
        else:
            continue
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
                            "memory_total_mib": captured["gpu_memory_total_mib_by_index"][
                                int(row[0])
                            ]
                            if "gpu_memory_total_mib_by_index" in captured
                            else None,
                            "gpu_utilization_percent": float(row[3].split()[0]),
                        }
                    ],
                }
            )
    samples.sort(key=lambda item: item["timestamp_utc"])
    (destination / "gpu_stats.jsonl").write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples)
    )
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
    if (source / "training-runtime.json").exists() or (source / "inference-runtime.json").exists():
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
        **{
            key: value
            for key, value in {**captured, **report}.items()
            if key
            in {
                "execution_run_id",
                "producer",
                "root_turn_id",
                "policy_ref",
                "policy_revision",
                "model_id",
                "model_revision",
                "gpu_topology",
                "hardware",
                "trl_version",
                "vllm_version",
                "trackio_run_id",
                "trackio_run_name",
                "trackio_space_id",
                "trackio_project",
                "trackio_url",
            }
        },
        "run_id": run_id,
        "experiment_id": captured.get("experiment_id", report.get("experiment_id")),
        "capture_kind": "source hashes captured before rollout; normalized offline after execution",
        "created_at_utc": events[0]["utc"] if events else report.get("started_at_utc"),
        "inputs": inputs,
        "pins": {
            key: value
            for key, value in {
                "model_revision": report.get("model_revision", captured.get("model_revision")),
                "trl": captured.get("trl_version", report.get("trl_version")),
                "vllm": captured.get("vllm_version", report.get("vllm_version")),
                "source_equivalent_rlm_commit": source_commit,
            }.items()
            if value is not None
        },
        "datasets": {
            "sha256": captured.get("dataset_sha256"),
            "upstream_revision": captured.get("dataset_revision"),
        },
        "config": {
            "group_size": len(report["groups"][0].get("rewards", []))
            if report.get("groups")
            else None,
            **{
                key: value
                for key, value in {
                    "root_tokens": captured.get("root_tokens"),
                    "context": captured.get("context_tokens"),
                    "max_turns": captured.get("max_turns"),
                    "enable_thinking": captured.get("enable_thinking"),
                    "cuda_docs": captured.get("cuda_docs"),
                    "children": captured.get("children"),
                    "child_prompt_tokens": captured.get("child_prompt_tokens"),
                    "child_reply_tokens": captured.get("child_reply_tokens"),
                    "child_tools": captured.get("child_tools"),
                    "learner_world_size": captured.get("learner_world_size"),
                    "inference_tensor_parallel_size": captured.get(
                        "inference_tensor_parallel_size"
                    ),
                    "max_optimizer_steps": report.get("optimizer_steps"),
                }.items()
                if value is not None
            },
        },
        "parent": captured.get("parent", report.get("parent")),
        "artifacts": captured.get("artifacts", report.get("artifacts", {})),
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
