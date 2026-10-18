"""Analyze saved rollout timings without loading a model or executing candidate code."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def source_hash(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def literal_text(node: ast.AST, values: dict[str, str]) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return literal_text(node.left, values) + literal_text(node.right, values)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "repr"
        and len(node.args) == 1
        and not node.keywords
    ):
        return repr(literal_text(node.args[0], values))
    raise ValueError("Only literal string construction is supported")


def feedback_sources(trace: dict) -> list[str | None]:
    values: dict[str, str] = {}
    sources = []
    for message in trace.get("info", {}).get("repl_transcript", []):
        for tool in message.get("tool_calls", []):
            try:
                code = json.loads(tool["function"]["arguments"])["code"]
                tree = ast.parse(code)
            except (KeyError, ValueError, SyntaxError):
                continue
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            try:
                                value = literal_text(node.value, values)
                            except (ValueError, KeyError):
                                values.pop(target.id, None)
                            else:
                                values[target.id] = value
                if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                    target = node.target.id
                    try:
                        if not isinstance(node.op, ast.Add):
                            raise ValueError("Unsupported augmented string operation")
                        values[target] += literal_text(node.value, values)
                    except (ValueError, KeyError):
                        values.pop(target, None)
                for call in ast.walk(node):
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                        if call.func.id == "cuda_feedback" and call.args:
                            try:
                                sources.append(literal_text(call.args[0], values))
                            except (ValueError, KeyError):
                                sources.append(None)
    return sources


def embedded_source_hashes(source: str) -> dict[str, str] | None:
    values: dict[str, str] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        value = literal_text(node.value, values)
                    except (ValueError, KeyError):
                        values.pop(target.id, None)
                    else:
                        values[target.id] = value
    if not all(key in values for key in ("CPP_SOURCE", "CUDA_SOURCE")):
        return None
    return {key: source_hash(values[key]) for key in ("CPP_SOURCE", "CUDA_SOURCE")}


def union_seconds(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    current_end = float("-inf")
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("Negative timing interval")
        total += max(0.0, end - max(start, current_end))
        current_end = max(current_end, end)
    return total


def distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "count": len(values),
        "p50_seconds": percentile(0.5),
        "p95_seconds": percentile(0.95),
        "max_seconds": ordered[-1],
    }


def read_traces(paths: list[Path]) -> dict[str, dict]:
    traces = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                for trace in row.get("traces", [row]):
                    traces[trace["id"]] = trace
    return traces


def profile_run(directory: Path) -> dict[str, Any]:
    paths = sorted(directory.glob("rollouts/step_*/train/all/traces.jsonl"))
    if not paths:
        raise ValueError("No saved all-trajectory exports found")
    traces = read_traces(paths)
    admitted = read_traces(sorted(directory.glob("rollouts/step_*/train/effective/traces.jsonl")))
    episodes = []
    segments = []
    stages: dict[str, list[float]] = defaultdict(list)
    model_intervals = []
    scoring_intervals = []
    for trace_id, trace in traces.items():
        info = trace.get("info", {})
        timing = trace.get("timing", {})
        agent = timing.get("agent", {})
        scoring = timing.get("scoring", {})
        durations = {
            "model_request": agent.get("model", {}).get("duration"),
            "repl_harness": agent.get("harness", {}).get("duration"),
            "scoring": scoring["end"] - scoring["start"] if "end" in scoring else None,
            "episode": scoring["end"] - timing["start"] if "end" in scoring else None,
        }
        for stage, duration in durations.items():
            if duration is not None:
                if duration < 0:
                    raise ValueError(f"Negative duration for {trace_id}: {stage}")
                stages[stage].append(duration)
        episode_intervals = []
        for call in trace.get("calls", []):
            period = call.get("time", {})
            if "start" in period and "end" in period:
                model_intervals.append((period["start"], period["end"]))
                episode_intervals.append((period["start"], period["end"]))
                segments.append({"episode": trace_id, "stage": "model_request", **period})
        if "start" in agent and "end" in agent:
            cursor = agent["start"]
            for start, end in sorted(episode_intervals):
                if start > cursor:
                    segments.append(
                        {
                            "episode": trace_id,
                            "stage": "harness_or_unattributed",
                            "start": cursor,
                            "end": min(start, agent["end"]),
                        }
                    )
                cursor = min(agent["end"], max(cursor, end))
            if cursor < agent["end"]:
                segments.append(
                    {
                        "episode": trace_id,
                        "stage": "harness_or_unattributed",
                        "start": cursor,
                        "end": agent["end"],
                    }
                )
        if "end" in scoring:
            scoring_intervals.append((scoring["start"], scoring["end"]))
            segments.append({"episode": trace_id, "stage": "scoring", **scoring})
        final_source = info.get("kernel_submission", "")
        recovered = feedback_sources(trace)
        final_embedded = embedded_source_hashes(final_source)
        feedback = []
        feedback_path = directory / "feedback" / Path(info.get("kernel_feedback_log", "")).name
        if feedback_path.is_file():
            feedback = [
                json.loads(line) for line in feedback_path.read_text().splitlines() if line.strip()
            ]
            for record in feedback:
                for key, value in record.get("timings", {}).items():
                    if isinstance(value, (int, float)):
                        stages["feedback_" + key].append(value)
        verification = info.get("kernel_verify", {})
        for key, value in verification.get("timings", {}).items():
            if isinstance(value, (int, float)):
                stages["final_" + key].append(value)
        episodes.append(
            {
                "id": trace_id,
                "task": trace.get("task", {}).get("data", {}).get("name"),
                "start": timing.get("start"),
                "end": scoring.get("end"),
                "durations_seconds": durations,
                "admitted": trace_id in admitted,
                "reward": info.get("kernel_reward"),
                "correct": verification.get("correct"),
                "final_source_sha256": source_hash(final_source) if final_source else None,
                "feedback_source_sha256": [
                    source_hash(source) if source else None for source in recovered
                ],
                "unchanged_final_after_feedback": final_source in recovered if recovered else None,
                "same_after_outer_whitespace_trim": any(
                    source.strip() == final_source.strip() for source in recovered if source
                )
                if recovered
                else None,
                "same_embedded_sources_as_feedback": any(
                    embedded_source_hashes(source) == final_embedded
                    for source in recovered
                    if source
                )
                if final_embedded
                else None,
                "feedback_records": feedback,
                "final_verifier_timings": verification.get("timings", {}),
            }
        )
    starts = [item["start"] for item in episodes if item["start"] is not None]
    ends = [item["end"] for item in episodes if item["end"] is not None]
    task_groups = defaultdict(list)
    for episode in episodes:
        task_groups[episode["task"]].append(episode)
    return {
        "kind": "diagnostic_not_experiment",
        "source_run": directory.name,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_files": [
            {"path": str(path), "sha256": source_hash(path.read_text())} for path in paths
        ],
        "episodes": episodes,
        "stages": {key: distribution(values) for key, values in stages.items()},
        "task_cohorts": [
            {
                "task": task,
                "episodes": len(items),
                "admitted": sum(item["admitted"] for item in items),
                "correct": sum(item["correct"] is True for item in items),
                "rewards": [item["reward"] for item in items],
            }
            for task, items in task_groups.items()
        ],
        "wall_clock": {
            "start": min(starts),
            "end": max(ends),
            "collection_span_seconds": max(ends) - min(starts),
            "model_request_union_seconds": union_seconds(model_intervals),
            "scoring_union_seconds": union_seconds(scoring_intervals),
        },
        "timeline": segments,
        "caveats": [
            "Per-episode durations overlap across episodes; do not sum them as wall time.",
            "Model request duration includes service/queue waits, not just GPU computation.",
            "Harness duration is not a compiler-only measurement.",
            "Compile elapsed includes interpreter/import/module loading.",
            "Initial batch collection is not steady-state throughput.",
            "Missing phases are unknown, not zero.",
            "Embedded-source matches ignore build flags/names; they are not proof of a cache hit.",
            "Feedback recovery is literal-only and may include a rejected feedback request.",
        ],
    }


def select_cases(directory: Path) -> list[dict]:
    traces = read_traces(sorted(directory.glob("rollouts/step_*/train/all/traces.jsonl")))
    cases = []
    hashes = set()
    for trace in traces.values():
        info = trace.get("info", {})
        source = info.get("kernel_submission", "")
        if info.get("kernel_verify", {}).get("correct") is not True or not source:
            continue
        digest = source_hash(source)
        if digest in hashes:
            continue
        ast.parse(source)
        hashes.add(digest)
        cases.append(
            {
                "name": trace["id"],
                "reference": trace["task"]["data"]["context"],
                "submission": source,
                "expected_correct": True,
                "source_sha256": digest,
            }
        )
        if len(cases) == 2:
            return cases
    raise ValueError("Two saved correct candidates are required")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = profile_run(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "episodes": len(report["episodes"]),
                "stages": report["stages"],
                "wall_clock": report["wall_clock"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
