"""Read-only adapters for the experiment registry and saved run evidence."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
WRITING_DOCS = {
    "README": "README.md",
    "first-campaign": "first-campaign.md",
    "protocol": "protocol.md",
    "literature-review": "literature-review.md",
    "artifact-publication": "artifact-publication.md",
    "experiment-tracking": "experiment-tracking.md",
}


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def registry() -> dict[str, Any]:
    value = read_json(REPO_ROOT / "training/environments/kernel_rlm/dashboard/experiments.json")
    if not isinstance(value.get("experiments"), list):
        return {"experiments": []}
    return value


def experiments() -> list[dict[str, Any]]:
    fields = (
        "id",
        "title",
        "status",
        "campaign",
        "category",
        "scope_note",
        "intent",
        "change",
        "conclusion",
        "decision",
        "limitations",
        "runs",
        "provenance",
    )
    return [
        {
            **{key: item.get(key) for key in fields},
            "follows_experiments": list(item.get("follows_experiments") or []),
        }
        for item in registry().get("experiments", [])
        if isinstance(item, dict)
    ]


def missing_export_reason(run: dict[str, Any]) -> str:
    report = run.get("report", {})
    if report.get("status") == "prepared":
        return "Preparation record only; no trajectory export was produced. Preparation does not imply a training launch."
    if report.get("status") == "failed":
        return "Failed execution; this local record contains diagnostics but no saved trajectory export."
    return "No supported trajectory export is available for this run."


def saved_traces(run: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Read unique exported episodes, retaining valid records beside damaged ones."""
    traces = {}
    warnings = []
    paths = sorted(run.get("rollouts", []), key=lambda path: ("effective" in path.parts, str(path)))
    for path in paths:
        try:
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        warnings.append(f"{path.name} line {line_number}: invalid JSON; skipped.")
                        continue
                    items = row.get("traces") if isinstance(row, dict) else None
                    if not isinstance(items, list):
                        warnings.append(
                            f"{path.name} line {line_number}: missing episode list; skipped."
                        )
                        continue
                    for index, trace in enumerate(items):
                        if not isinstance(trace, dict):
                            warnings.append(
                                f"{path.name} line {line_number}: invalid episode; skipped."
                            )
                            continue
                        identity = trace.get("id")
                        if not isinstance(identity, (str, int)):
                            identity = f"{path.relative_to(run['root'])}:{line_number}:{index}"
                        traces.setdefault(str(identity), trace)
        except (OSError, UnicodeError):
            warnings.append(f"Cannot read episode export {path.name}.")
    if not paths:
        warnings.append(missing_export_reason(run))
    return list(traces.values()), warnings


def discover_runs(root: Path) -> list[dict[str, Any]]:
    """Discover run folders without loading metric histories or trajectory bodies."""
    import os

    from training.environments.kernel_rlm.dashboard import server as reader

    runs: dict[str, dict[str, Any]] = {}
    for report_path in root.rglob("report.json"):
        report = read_json(report_path)
        run_id = str(report.get("run_id") or report_path.parent.name)
        run = runs.setdefault(run_id, {"id": run_id, "root": report_path.parent, "report": {}})
        run["report"] = report
    for manifest_path in root.rglob("provenance.json"):
        manifest = read_json(manifest_path)
        run_id = manifest.get("run_id")
        if not run_id:
            continue
        run = runs.setdefault(run_id, {"id": run_id, "root": manifest_path.parent, "report": {}})
        run["provenance"] = manifest
        if not run["report"]:
            events = []
            try:
                events = [
                    json.loads(line)
                    for line in (manifest_path.parent / "events.jsonl").read_text().splitlines()
                    if line.strip()
                ]
            except (OSError, json.JSONDecodeError):
                pass
            run["report"] = {
                "run_id": run_id,
                "status": events[-1].get("event", "prepared") if events else "prepared",
                "experiment_id": manifest.get("experiment_id"),
                "model_id": manifest.get("pins", {}).get("model_id"),
                "title": manifest.get("experiment", {}).get("title", "Registered experiment")
                + " · input capture",
            }
    for base, _, files in os.walk(root):
        names = set(files)
        directory = Path(base)
        if "trace.json" in names:
            run = runs.setdefault(
                directory.name, {"id": directory.name, "root": directory, "report": {}}
            )
            run.setdefault("traces", []).append(directory / "trace.json")
        if "episode.json" in names:
            run = runs.setdefault(
                directory.name, {"id": directory.name, "root": directory, "report": {}}
            )
            run.setdefault("episodes", []).append(directory / "episode.json")
        if "traces.jsonl" in names:
            run_id = next(
                (part for part in reversed(directory.parts) if part.startswith("20")),
                directory.name,
            )
            run = runs.setdefault(run_id, {"id": run_id, "root": directory, "report": {}})
            run.setdefault("rollouts", []).append(directory / "traces.jsonl")
    return sorted(
        runs.values(), key=lambda run: reader.run_timestamp({"id": run["id"]}), reverse=True
    )


def run_records(root: Path, *, count_trajectories: bool = True) -> list[dict[str, Any]]:
    from training.environments.kernel_rlm.dashboard import server as reader

    records = []
    for run in discover_runs(root):
        warnings = []
        has_export = bool(run.get("rollouts")) or bool(run.get("traces"))
        trajectory_count = None
        if run.get("rollouts") and count_trajectories:
            trace_ids = set()
            for export_path in run["rollouts"]:
                try:
                    with export_path.open(encoding="utf-8") as stream:
                        for line in stream:
                            try:
                                row = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if isinstance(row, dict) and isinstance(row.get("traces"), list):
                                for index, trace in enumerate(row["traces"]):
                                    if isinstance(trace, dict):
                                        identity = trace.get("id")
                                        trace_ids.add(
                                            str(identity)
                                            if isinstance(identity, (str, int))
                                            else f"{export_path}:{index}"
                                        )
                except (OSError, UnicodeError):
                    continue
            trajectory_count = len(trace_ids)
        elif run.get("traces"):
            trajectory_count = len(run["traces"])
        records.append(
            {
                "id": run["id"],
                "title": reader.run_title(run),
                "status": run.get("report", {}).get("status"),
                "experiment_id": run.get("report", {}).get("experiment_id"),
                "experiments": [
                    {
                        "id": experiment["id"],
                        "title": experiment.get("title"),
                        "category": experiment.get("category"),
                    }
                    for experiment in experiments()
                    if run["id"] in (experiment.get("runs") or [])
                ],
                "trajectory_count": trajectory_count,
                "elapsed_seconds": (run.get("report") or {}).get("elapsed_seconds"),
                "elapsed_scope": (run.get("report") or {}).get("elapsed_scope"),
                "evidence_available": has_export or bool(run.get("traces")),
                "warnings": warnings,
                "_reader": run,
            }
        )
    known = {record["id"] for record in records}
    for experiment in experiments():
        for run_id in experiment.get("runs", []):
            if run_id not in known:
                records.append(
                    {
                        "id": run_id,
                        "title": run_id,
                        "status": None,
                        "experiment_id": experiment["id"],
                        "experiments": [
                            {
                                "id": experiment["id"],
                                "title": experiment.get("title"),
                                "category": experiment.get("category"),
                            }
                        ],
                        "trajectory_count": None,
                        "evidence_available": False,
                        "warnings": [
                            "Linked run has no downloaded report or supported trajectory export; local evidence is unavailable."
                        ],
                    }
                )
    return records


def runs(root: Path) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in run.items() if key != "_reader"} for run in run_records(root)
    ]


def campaign(receipt_path: Path | None) -> tuple[dict[str, Any], list[str]]:
    if receipt_path is None:
        return {
            "status": None,
            "trained": None,
            "deferred": None,
            "checkpoint": None,
            "hardware": None,
            "source": None,
            "completion_status": None,
        }, ["Campaign receipt not configured; campaign summary unavailable."]
    receipt = read_json(receipt_path)
    if not receipt:
        return {
            "status": None,
            "trained": None,
            "deferred": None,
            "checkpoint": None,
            "hardware": None,
            "source": "Configured campaign receipt",
        }, ["Configured campaign receipt is missing or invalid."]
    deferred = receipt.get(
        "deferred", receipt.get("deferred_untrained", receipt.get("deferred_task_count"))
    )
    if isinstance(deferred, list):
        deferred = len(deferred)
    checkpoint = receipt.get(
        "checkpoint", receipt.get("checkpoint_id", receipt.get("accepted_policy"))
    )
    return {
        "status": receipt.get("status", receipt.get("training_status")),
        "trained": receipt.get(
            "trained", receipt.get("trained_tasks", receipt.get("completed_task_count"))
        ),
        "deferred": deferred,
        "checkpoint": checkpoint,
        "hardware": receipt.get("hardware"),
        "source": "Configured campaign receipt",
        "completion_status": receipt.get("training_status"),
    }, []


def home(root: Path, receipt_path: Path | None = None) -> dict[str, Any]:
    campaign_data, warnings = campaign(receipt_path)
    receipt = read_json(receipt_path) if receipt_path else {}
    run_status = None
    if receipt.get("run_id"):
        for run in run_records(root, count_trajectories=False):
            if run["id"] == receipt["run_id"]:
                run_status = run["status"]
                break
        campaign_data["run_status"] = run_status
        campaign_data["completion_status"] = receipt.get("training_status")
        if (
            run_status
            and campaign_data["completion_status"]
            and run_status != campaign_data["completion_status"]
        ):
            warnings.append("Run report status differs from the configured completion receipt.")
    return {"campaign": campaign_data, "experiments": experiments(), "warnings": warnings}


def assistant_response_text(content: Any, tool_calls: list[Any]) -> str:
    sections = [content] if isinstance(content, str) and content.strip() else []
    for call in tool_calls:
        function = call.get("function", {}) if isinstance(call, dict) else {}
        name = (
            function.get("name", call.get("name", "unknown tool"))
            if isinstance(call, dict)
            else "unknown tool"
        )
        arguments = (
            function.get("arguments", call.get("arguments")) if isinstance(call, dict) else None
        )
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                sections.append(f"{name} · emitted tool call, unparsed arguments:\n{arguments}")
                continue
        else:
            parsed = arguments
        code = parsed.get("code") if isinstance(parsed, dict) else None
        payload = code if isinstance(code, str) and code.strip() else json.dumps(parsed, indent=2)
        sections.append(f"{name} · emitted tool call, not proof of execution:\n{payload}")
    return "\n\n".join(sections)


def task_catalog(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load an explicitly selected JSONL task catalog keyed by its exact task ID."""
    if path is None:
        return {}
    return load_task_catalog(path, path.stat().st_mtime_ns)


@lru_cache(maxsize=2)
def load_task_catalog(path: Path, modified: int) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and isinstance(item.get("_task_id"), str):
                    ops = item.get("ops")
                    if isinstance(ops, str):
                        try:
                            ops = json.loads(ops)
                        except json.JSONDecodeError:
                            pass
                    catalog[item["_task_id"]] = {
                        "ops": ops,
                        "data_source": item.get("data_source"),
                        "source": item.get("code"),
                    }
    except (OSError, UnicodeError):
        return catalog
    return catalog


def usage_counts(records: list[dict[str, Any]], total_calls: int | None = -1) -> dict[str, Any]:
    result = {}
    for key, label in (
        ("prompt_tokens", "input_tokens"),
        ("completion_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        values = [
            record[key]
            for record in records
            if isinstance(record.get(key), int) and record[key] >= 0
        ]
        total = len(records) if total_calls is None else None if total_calls < 0 else total_calls
        result[label] = {
            "sum": sum(values) if values else 0 if total == 0 else None,
            "recorded": len(values),
            "total": total,
        }
    return result


def elapsed_counts(records: list[dict[str, Any]], total_calls: int | None) -> dict[str, Any]:
    values = [
        record["elapsed_seconds"]
        for record in records
        if isinstance(record.get("elapsed_seconds"), (int, float))
        and record["elapsed_seconds"] >= 0
    ]
    total = len(records) if total_calls is None else None if total_calls < 0 else total_calls
    return {
        "sum": sum(values) if values else 0 if total == 0 else None,
        "recorded": len(values),
        "total": total,
    }


def token_usage(trace: dict[str, Any]) -> dict[str, Any]:
    info = trace.get("info") if isinstance(trace.get("info"), dict) else {}
    root_calls = info.get("kernel_model_calls")
    if not isinstance(root_calls, list):
        root_calls = trace.get("calls")
    root_known = isinstance(root_calls, list)
    calls = root_calls if root_known else []
    roots = [
        call["usage"]
        for call in calls
        if isinstance(call, dict) and isinstance(call.get("usage"), dict)
    ]
    child_ledger = info.get("kernel_subcalls")
    child_known = isinstance(child_ledger, list)
    child_ledger = child_ledger if child_known else []
    children = []
    for child in child_ledger:
        if not isinstance(child, dict):
            continue
        response = child.get("response")
        usage = response.get("usage") if isinstance(response, dict) else None
        usage = usage if isinstance(usage, dict) else child.get("usage")
        if isinstance(usage, dict):
            children.append(usage)
    root = usage_counts(roots, len(calls) if root_known else -1)
    child = usage_counts(children, len(child_ledger) if child_known else -1)
    all_calls = roots + children
    total = len(calls) + len(child_ledger) if root_known and child_known else -1
    root["elapsed_seconds"] = elapsed_counts(calls, len(calls) if root_known else -1)
    child["elapsed_seconds"] = elapsed_counts(
        child_ledger, len(child_ledger) if child_known else -1
    )
    root_durations = [item for item in calls if isinstance(item, dict)]
    child_durations = [item for item in child_ledger if isinstance(item, dict)]
    return {
        "root": root,
        "children": child,
        "all": {
            **usage_counts(all_calls, total),
            "elapsed_seconds": elapsed_counts(root_durations + child_durations, total),
        },
    }


def trajectory_metrics(trace: dict[str, Any]) -> dict[str, Any]:
    info = trace.get("info") if isinstance(trace.get("info"), dict) else {}
    calls = info.get("kernel_model_calls") or trace.get("calls") or []
    nodes = trace.get("nodes") or []
    usage = []
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("usage"), dict):
            continue
        node = call.get("node")
        if isinstance(node, int) and 0 <= node < len(nodes):
            message = nodes[node].get("message", {}) if isinstance(nodes[node], dict) else {}
            if message.get("role") != "assistant":
                continue
        usage.append(call["usage"])
    transcript = info.get("repl_transcript")
    if not isinstance(transcript, list):
        transcript = [
            node["message"]
            for node in nodes
            if isinstance(node, dict) and isinstance(node.get("message"), dict)
        ] or None
    prompt_tokens = [
        item["prompt_tokens"] for item in usage if isinstance(item.get("prompt_tokens"), int)
    ]
    cached = [
        item.get("prompt_tokens_details", {}).get("cached_tokens")
        if isinstance(item.get("prompt_tokens_details"), dict)
        else None
        for item in usage
    ]
    cache_ratio = None
    if (
        usage
        and len(prompt_tokens) == len(usage)
        and all(isinstance(value, int) for value in cached)
        and sum(prompt_tokens)
    ):
        cache_ratio = 100 * sum(cached) / sum(prompt_tokens)
    failed = info.get("commands_failed")
    return {
        "token_usage": token_usage(trace),
        "steps": sum(
            item.get("role") == "assistant" for item in transcript if isinstance(item, dict)
        )
        if transcript is not None
        else None,
        "tool_calls": sum(
            len(item.get("tool_calls") or []) for item in transcript if isinstance(item, dict)
        )
        if transcript is not None
        else None,
        "peak_context": max(prompt_tokens) if prompt_tokens else None,
        "prompt_cache": cache_ratio,
        "commands_failed": failed if isinstance(failed, int) else None,
    }


def trajectory_model(run: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    """Prefer recorded model identity; declarations identify only a base model."""
    info = trace.get("info") or {}
    report = run.get("report") or run.get("_reader", {}).get("report") or {}
    for source, record in (("trace.info", info), ("trace", trace), ("report", report)):
        for field in ("model", "model_id"):
            if record.get(field):
                return {
                    "model": record[field],
                    "model_scope": "recorded",
                    "model_source": f"{source}.{field}",
                }
    declarations = [
        (experiment["id"], experiment["provenance"]["base_model"])
        for experiment in experiments()
        if run["id"] in (experiment.get("runs") or [])
        and (experiment.get("provenance") or {}).get("base_model")
    ]
    if len(declarations) == 1:
        experiment_id, base = declarations[0]
        return {
            "model": base["id"],
            "model_scope": "base",
            "model_revision": base.get("revision"),
            "model_source": "training/environments/kernel_rlm/dashboard/experiments.json"
            f"#experiments/{experiment_id}/provenance/base_model",
            "model_evidence": base.get("evidence"),
        }
    return {"model": None, "model_scope": None, "model_source": None}


def trajectory_metadata(
    run: dict[str, Any], trace: dict[str, Any], index: int, *, task_catalog_path: Path | None = None
) -> dict[str, Any]:
    task = trace.get("task")
    task_data = task.get("data") if isinstance(task, dict) else None
    task_data = task_data if isinstance(task_data, dict) else {}
    task_id = task_data.get("name")
    catalog_item = (
        task_catalog(task_catalog_path).get(task_id, {}) if isinstance(task_id, str) else {}
    )
    ops = catalog_item.get("ops")
    title = task_data.get("title")
    if not title and isinstance(ops, list) and ops:
        title = " · ".join(str(op).replace("nn.", "").replace("torch.", "") for op in ops)
    if not title:
        title = task_id
    info = trace.get("info") if isinstance(trace.get("info"), dict) else {}
    outcome = {
        "submitted": safe_trace_submitted(trace),
        "correct": (info.get("kernel_verify") or trace.get("verification") or {}).get("correct")
        if isinstance(info.get("kernel_verify", trace.get("verification", {})), dict)
        else None,
        "speedup": (info.get("kernel_verify") or trace.get("verification") or {}).get("speedup")
        if isinstance(info.get("kernel_verify", trace.get("verification", {})), dict)
        else None,
        "reward": safe_trace_reward(trace),
        "status": info.get("kernel_termination", trace.get("stop_condition")),
    }
    completion = (
        "completed"
        if outcome["submitted"] is True
        else "incomplete"
        if outcome["submitted"] is False
        else "unknown"
    )
    if outcome["status"] in ("submitted", "completed"):
        completion = "completed"
    started_at = trace.get("started_at") or info.get("started_at")
    datasets = sorted(
        {
            dataset
            for ex in experiments()
            if run["id"] in (ex.get("runs") or [])
            for dataset in (ex.get("provenance") or {}).get("dataset_ids", [])
        }
    )
    return {
        "id": str(trace.get("id", index)),
        "record_id": f"{run['id']}::{trace.get('id', index)}",
        "attempt": index + 1,
        "task": task_id,
        "task_id": task_id,
        "title": title,
        "task_title_source": "task.data.title"
        if task_data.get("title")
        else "catalog.ops"
        if catalog_item.get("ops")
        else "task.data.name",
        "benchmark": task_data.get("benchmark") or (" · ".join(datasets) if datasets else None),
        **trajectory_model(run, trace),
        "started_at": started_at,
        "completion": completion,
        "outcome": outcome,
        "metrics": {
            **trajectory_metrics(trace),
            "run_elapsed_seconds": (run.get("report") or {}).get("elapsed_seconds"),
            "run_elapsed_scope": (run.get("report") or {}).get("elapsed_scope"),
        },
        "experiments": run.get("experiments", []),
        "run": {"id": run["id"]},
    }


def trajectory_detail(
    run: dict[str, Any],
    trace: dict[str, Any],
    index: int,
    *,
    task_catalog_path: Path | None = None,
    attempt: int | None = None,
) -> dict[str, Any]:
    info = trace.get("info")
    info = info if isinstance(info, dict) else {}
    task = trace.get("task")
    task_data = task.get("data") if isinstance(task, dict) else None
    task_data = task_data if isinstance(task_data, dict) else {}
    verification = info.get("kernel_verify", trace.get("verification", {}))
    verification = verification if isinstance(verification, dict) else {}
    events = []
    transcript = info.get("repl_transcript")
    if isinstance(transcript, list) and transcript:
        calls = info.get("kernel_model_calls") or trace.get("calls", [])
        node_messages = [
            node.get("message", {}) if isinstance(node, dict) else {}
            for node in (trace.get("nodes") or [])
        ]
        root_calls = [
            call
            for call in calls
            if isinstance(call, dict)
            and isinstance(call.get("node"), int)
            and 0 <= call["node"] < len(node_messages)
            and node_messages[call["node"]].get("role") == "assistant"
        ]
        for event_index, item in enumerate(transcript):
            if isinstance(item, dict):
                event = {
                    "index": event_index,
                    "role": item.get("role"),
                    "content": item.get("content"),
                    "tool_calls": item.get("tool_calls"),
                    "source": "repl_transcript",
                }
                if item.get("role") == "assistant":
                    call = root_calls.pop(0) if root_calls else {}
                    original = node_messages[call["node"]] if call.get("node") is not None else {}
                    event.update(
                        {
                            "reasoning": original.get("reasoning_content"),
                            "usage": call.get("usage"),
                            "finish_reason": call.get("finish_reason"),
                            "assistant_response": assistant_response_text(
                                item.get("content"), event["tool_calls"] or []
                            ),
                        }
                    )
                    if not event["tool_calls"]:
                        event["tool_calls"] = original.get("tool_calls")
                events.append(event)
    else:
        nodes = trace.get("nodes")
        for event_index, node in enumerate(nodes if isinstance(nodes, list) else []):
            if isinstance(node, dict) and isinstance(node.get("message"), dict):
                message = node["message"]
                events.append(
                    {
                        "index": event_index,
                        "role": message.get("role"),
                        "content": message.get("content"),
                        "tool_calls": message.get("tool_calls"),
                        "reasoning": message.get("reasoning_content"),
                        "source": "nodes",
                    }
                )
    for source in ("kernel_child_progress", "kernel_subcalls", "kernel_child_sessions"):
        children = info.get(source, [])
        if children:
            events.append({"type": "child_activity", "source": source, "records": children})
    metadata = trajectory_metadata(run, trace, index, task_catalog_path=task_catalog_path)
    if attempt is not None:
        metadata["attempt"] = attempt
    return {
        **metadata,
        "task_source": task_data.get("source")
        or task_catalog(task_catalog_path).get(task_data.get("name"), {}).get("source"),
        "outcome": {
            "submitted": safe_trace_submitted(trace),
            "correct": verification.get("correct") if isinstance(verification, dict) else None,
            "speedup": verification.get("speedup") if isinstance(verification, dict) else None,
            "reward": safe_trace_reward(trace),
            "status": info.get("kernel_termination", trace.get("stop_condition")),
        },
        "run": {"id": run["id"]},
        "events": events,
        "verification": verification,
        "raw": trace,
    }


def safe_trace_submitted(trace: dict[str, Any]) -> bool | None:
    from training.environments.kernel_rlm.dashboard import server as reader

    info = trace.get("info")
    if not isinstance(info, dict) or not any(
        key in info for key in ("kernel_submission", "kernel_termination")
    ):
        return None
    try:
        return reader.trace_submitted(trace)
    except (AttributeError, TypeError):
        return None


def safe_trace_reward(trace: dict[str, Any]) -> float | None:
    from training.environments.kernel_rlm.dashboard import server as reader

    try:
        return reader.trace_reward(trace)
    except (AttributeError, TypeError):
        return None


def trajectory_index(root: Path, *, task_catalog_path: Path | None = None) -> dict[str, Any]:
    """Return lightweight trajectory summaries from supported local exports."""
    trajectories = []
    warnings = []
    for run in run_records(root, count_trajectories=False):
        reader_run = run.get("_reader")
        if not reader_run:
            continue
        traces, run_warnings = saved_traces(reader_run)
        warnings.extend(f"{run['id']}: {warning}" for warning in run_warnings)
        attempts: dict[str, int] = {}
        for index, trace in enumerate(traces):
            task = trace.get("task") if isinstance(trace, dict) else None
            task_data = task.get("data") if isinstance(task, dict) else None
            task_id = task_data.get("name") if isinstance(task_data, dict) else None
            key = str(task_id) if task_id is not None else ""
            attempts[key] = attempts.get(key, 0) + 1
            try:
                summary = trajectory_metadata(
                    run, trace, index, task_catalog_path=task_catalog_path
                )
                summary["attempt"] = attempts[key]
                trajectories.append(summary)
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                warnings.append(
                    f"{run['id']}: malformed trajectory {index} skipped ({type(error).__name__})."
                )
    return {"trajectories": trajectories, "warnings": warnings}
