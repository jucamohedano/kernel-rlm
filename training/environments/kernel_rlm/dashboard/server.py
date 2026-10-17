"""Serve a local, dependency-free dashboard for saved Kernel-RLM runs."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_metrics(path: Path) -> list[dict]:
    values = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return values
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def read_jsonl(path: Path) -> list[dict]:
    values = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return values
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def discover_runs(root: Path) -> list[dict]:
    runs: dict[str, dict] = {}
    for report_path in root.rglob("report.json"):
        report = read_json(report_path)
        run_id = str(report.get("run_id") or report_path.parent.name)
        run = runs.setdefault(run_id, {"id": run_id, "root": report_path.parent, "report": {}})
        run["report"] = report
    for metrics_path in root.rglob("metrics.jsonl"):
        run_id = metrics_path.parent.name
        run = runs.setdefault(run_id, {"id": run_id, "root": metrics_path.parent, "report": {}})
        run["metrics_path"] = metrics_path
        run["metrics"] = read_metrics(metrics_path)
    for trace_path in root.rglob("trace.json"):
        run_id = trace_path.parent.name
        run = runs.setdefault(run_id, {"id": run_id, "root": trace_path.parent, "report": {}})
        run.setdefault("traces", []).append(trace_path)
    for episode_path in root.rglob("episode.json"):
        run_id = episode_path.parent.name
        run = runs.setdefault(run_id, {"id": run_id, "root": episode_path.parent, "report": {}})
        run.setdefault("episodes", []).append(episode_path)
    for rollout_path in root.rglob("traces.jsonl"):
        run_id = next(
            (part for part in reversed(rollout_path.parts) if part.startswith("20")),
            rollout_path.parent.name,
        )
        run = runs.setdefault(run_id, {"id": run_id, "root": rollout_path.parent, "report": {}})
        run.setdefault("rollouts", []).append(rollout_path)
    return sorted(runs.values(), key=run_timestamp, reverse=True)


def run_timestamp(run: dict) -> str:
    match = re.search(r"\d{8}T\d{6}Z", run["id"])
    return match.group() if match else ""


def run_date(run: dict) -> str:
    timestamp = run_timestamp(run)
    if not timestamp:
        return "Date unavailable"
    return (
        datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ")
        .replace(tzinfo=UTC)
        .strftime("%d %b %Y · %H:%M UTC")
    )


def run_title(run: dict) -> str:
    labels = read_json(Path(__file__).with_name("run_labels.json"))
    report = run["report"]
    return str(
        report.get("title")
        or labels.get(run["id"])
        or report.get("scope")
        or run["root"].parent.name.replace("-", " ").replace("_", " ").title()
    )


def trace_reward(trace: dict) -> float | None:
    reward = trace.get("rewards", {}).get("kernel_reward", {}).get("score")
    if reward is None:
        reward = trace.get("info", {}).get("kernel_reward")
    return reward if isinstance(reward, (int, float)) and math.isfinite(reward) else None


def trace_submitted(trace: dict) -> bool:
    info = trace.get("info", {})
    return bool(info.get("kernel_submission")) or info.get("kernel_termination") == "submitted"


def model_calls(trace: dict) -> list[dict]:
    return trace.get("info", {}).get("kernel_model_calls", trace.get("calls", []))


def gpu_samples(run: dict) -> list[dict]:
    return read_jsonl(run["root"] / "gpu_stats.jsonl")


def run_summary(run: dict) -> dict:
    if "summary" in run:
        return run["summary"]
    traces = rollout_traces(run)
    effective_paths = [path for path in run.get("rollouts", []) if "effective" in path.parts]
    effective_ids = {
        trace.get("id")
        for path in effective_paths
        for row in read_jsonl(path)
        for trace in row.get("traces", [])
        if trace.get("id")
    }
    groups = run["report"].get("groups", [])
    groups = (
        [group for group in groups if isinstance(group, dict)] if isinstance(groups, list) else []
    )
    legacy = [read_json(path) for path in run.get("traces", [])]
    completed = (
        len(traces)
        if traces
        else (sum(len(group.get("rewards", [])) for group in groups) if groups else len(legacy))
    )
    correct = (
        sum(
            trace.get("info", {}).get("kernel_verify", {}).get("correct") is True
            for trace in traces
        )
        if traces
        else (
            sum(group.get("correct", 0) for group in groups)
            if groups
            else sum(value.get("verification", {}).get("correct") is True for value in legacy)
        )
    )
    verification_known = (
        any("kernel_verify" in trace.get("info", {}) for trace in traces)
        or any("correct" in group for group in groups)
        or any("correct" in value.get("verification", {}) for value in legacy)
    )
    rewards = [value for trace in traces if (value := trace_reward(trace)) is not None]
    if not traces:
        rewards = [
            value
            for group in groups
            for value in group.get("rewards", [])
            if isinstance(value, (int, float)) and math.isfinite(value)
        ]
    telemetry = gpu_samples(run)
    duration = run["report"].get("elapsed_seconds")
    duration_source = "Run report"
    if not isinstance(duration, (int, float)) and len(telemetry) > 1:
        duration = (
            datetime.fromisoformat(telemetry[-1]["timestamp_utc"])
            - datetime.fromisoformat(telemetry[0]["timestamp_utc"])
        ).total_seconds()
        duration_source = "GPU telemetry span; excludes queue time"
    calls = [call for trace in traces for call in model_calls(trace)]
    steps = [
        row["step"]
        for row in run.get("metrics", [])
        if row.get("producer") == "trainer" and "loss/mean" in row and "step" in row
    ]
    run["summary"] = {
        "traces": traces,
        "completed": completed,
        "correct": correct if verification_known else None,
        "submitted": sum(trace_submitted(trace) for trace in traces) if traces else None,
        "mean_reward": sum(rewards) / len(rewards) if rewards else None,
        "admitted": len(effective_ids) if effective_paths else None,
        "duration": duration,
        "duration_source": duration_source,
        "truncated_calls": sum(call.get("finish_reason") == "length" for call in calls),
        "model_calls": len(calls),
        "subcalls": sum(len(trace.get("info", {}).get("kernel_subcalls", [])) for trace in traces),
        "optimizer_step": max(steps) if steps else None,
    }
    return run["summary"]


def experiment_catalog() -> list[dict]:
    return read_json(Path(__file__).with_name("experiments.json")).get("experiments", [])


def execution_badge(run: dict) -> str:
    status = str(run["report"].get("status", "artifacts")).replace("_", " ")
    tone = (
        "success"
        if status in {"passed", "completed", "group passed", "trajectory passed"}
        else ("failure" if "failed" in status or status == "interrupted" else "neutral")
    )
    return f'<span class="pill {tone}">{text(status)}</span>'


def text(value: object) -> str:
    return html.escape(str(value))


def metric_value(metrics: list[dict], names: tuple[str, ...]) -> object:
    for row in reversed(metrics):
        for name in names:
            if name in row:
                return row[name]
    return "—"


def plot_svg(
    series: dict[str, list[tuple[float, float]]],
    x_label: str,
    y_label: str,
    source: str,
    bounds: tuple[float, float] | None = None,
) -> str:
    points = [point for values in series.values() for point in values]
    if not points:
        return '<p class="empty">No series recorded in downloaded evidence.</p>'
    x_min, x_max = min(point[0] for point in points), max(point[0] for point in points)
    y_min, y_max = bounds or (min(point[1] for point in points), max(point[1] for point in points))
    x_span, y_span = x_max - x_min or 1, y_max - y_min or 1
    shapes = []
    legend = []
    colors = ("#6250cc", "#26785c", "#af713d")
    for index, (name, values) in enumerate(series.items()):
        color = colors[index % len(colors)]
        coordinates = [
            (52 + (position - x_min) * 328 / x_span, 144 - (value - y_min) * 116 / y_span)
            for position, value in values
        ]
        if len(coordinates) > 1:
            shapes.append(
                f'<polyline points="{" ".join(f"{horizontal:.1f},{vertical:.1f}" for horizontal, vertical in coordinates)}" '
                f'fill="none" stroke="{color}" stroke-width="2"/>'
            )
        elif coordinates:
            shapes.append(
                f'<circle cx="{coordinates[0][0]}" cy="{coordinates[0][1]}" r="4" fill="{color}"/>'
            )
        legend.append(f'<span><i style="background:{color}"></i>{text(name)}</span>')
    if len(points) == 1:
        legend.append("<span>One saved point; no trend established</span>")
    return (
        f'<svg class="metric-plot" viewBox="0 0 400 190" role="img" aria-label="{text(y_label)} versus {text(x_label)}">'
        f"<title>{text(y_label)} versus {text(x_label)}</title>"
        '<path d="M52 28V144H380 M52 28H380 M52 86H380" class="plot-grid"/>'
        + "".join(shapes)
        + f'<text x="46" y="32" text-anchor="end">{y_max:.3g}</text>'
        f'<text x="46" y="148" text-anchor="end">{y_min:.3g}</text>'
        f'<text x="52" y="162">{x_min:.3g}</text><text x="380" y="162" text-anchor="end">{x_max:.3g}</text>'
        f'<text x="216" y="183" text-anchor="middle">{text(x_label)}</text>'
        f'<text x="12" y="86" text-anchor="middle" transform="rotate(-90 12 86)">{text(y_label)}</text></svg>'
        '<div class="plot-legend">' + "".join(legend) + "</div>"
        f'<p class="plot-source">{text(source)}</p>'
    )


def chart_svg(metrics: list[dict], key_names: tuple[str, ...], label: str) -> str:
    values = {}
    for index, row in enumerate(metrics):
        value = metric_value([row], key_names)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values[row.get("step", index)] = float(value)
    if len(values) == 1:
        step, value = next(iter(values.items()))
        return (
            f'<div class="single-measurement">{value:.4g}<small>Optimizer step {text(step)}</small></div>'
            '<p class="plot-source">One saved point; no trend established. Source: downloaded metrics.jsonl.</p>'
        )
    return plot_svg(
        {label: sorted(values.items())},
        "Optimizer step",
        label,
        "Source: downloaded metrics.jsonl; recorded steps only",
    )


def render_gpu_charts(run: dict) -> str:
    samples = gpu_samples(run)
    if not samples:
        return '<p class="muted">GPU telemetry not downloaded. No usage is inferred.</p>'
    start = datetime.fromisoformat(samples[0]["timestamp_utc"])
    figures = []
    for key, label, scale, bounds in (
        ("memory_used_mib", "GPU memory (GiB)", 1024, (0, 80)),
        ("gpu_utilization_percent", "GPU utilization (%)", 1, (0, 100)),
    ):
        series = {}
        for sample in samples:
            elapsed = (datetime.fromisoformat(sample["timestamp_utc"]) - start).total_seconds() / 60
            for gpu in sample.get("gpus", []):
                value = gpu.get(key)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    series.setdefault(f"GPU {gpu['index']}", []).append((elapsed, value / scale))
        if series:
            memory_bounds = (
                0,
                max(gpu.get("memory_total_mib", 81920) for gpu in samples[0]["gpus"]) / 1024,
            )
            figures.append(
                '<figure class="chart-card"><h3>'
                + label
                + "</h3>"
                + plot_svg(
                    series,
                    "Minutes since first sample",
                    label,
                    f"Source: gpu_stats.jsonl · {samples[0]['timestamp_utc']} to {samples[-1]['timestamp_utc']}",
                    memory_bounds if key == "memory_used_mib" else bounds,
                )
                + "</figure>"
            )
    return "".join(figures)


def layout(title: str, body: str, active: str = "experiments") -> str:
    assets = Path(__file__).parent
    styles = (assets / "viewer.css").read_text()
    script = (assets / "viewer.js").read_text()
    navigation = "".join(
        f'<a href="{url}" {"aria-current=page" if active == key else ""}>{label}</a>'
        for key, label, url in (
            ("overview", "Overview", "/"),
            ("experiments", "Experiments", "/?view=experiments"),
            ("lineage", "Lineage", "/?view=lineage"),
            ("runs", "Run attempts", "/?view=runs"),
            ("configs", "Configs & compatibility", "/?view=configs"),
            ("guide", "Learning report", "/guide"),
        )
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{text(title)}</title><style>{styles}</style></head>
<body><div class="app-frame"><aside class="app-nav"><a class="brand" href="/">Kernel<span>RLM</span><small>Research workspace</small></a>
<nav aria-label="Workspace">{navigation}</nav><div class="local-note"><span class="status-dot"></span> Local artifacts only<br>No GPU jobs or API calls</div></aside>
<main>{body}<footer>Saved evidence, not a live monitor. Missing measurements are not zero.
 · <a href="/guide">Project guide</a></footer></main></div><script>{script}</script></body></html>"""


def render_project_guide() -> str:
    path = Path(__file__).resolve().parents[5] / "PROJECT_TRACKER.html"
    return path.read_text(encoding="utf-8")


def render_guide_workspace() -> str:
    document = render_project_guide()
    article = document.split("<main>", 1)[1].split("</main>", 1)[0]
    return layout(
        "Kernel-RLM · Learning report", '<article class="guide">' + article + "</article>", "guide"
    )


def render_run_table(runs: list[dict]) -> str:
    rows = []
    for run in runs:
        report = run["report"]
        summary = run_summary(run)
        outcome = (
            f"{summary['correct']}/{summary['completed']} correct final submissions"
            if (summary["correct"] is not None)
            else "Outcome unavailable in downloaded evidence"
        )
        duration = summary["duration"]
        elapsed = (
            f"{duration / 60:.1f} min" if isinstance(duration, int | float) else "Not recorded"
        )
        admitted = summary["admitted"]
        rows.append(
            f'<tr data-run><td><a class="run-title" href="/?run={quote(run["id"])}">{text(run_title(run))}</a>'
            f"<small>{text(run_date(run))} · {text(run['root'].parent.name)}</small>"
            f'<small class="run-id">{text(run["id"])}</small></td>'
            f"<td>{execution_badge(run)}</td><td>{text(outcome)}"
            + (
                f"<small>{admitted} / {summary['completed']} admitted for training</small>"
                if admitted is not None
                else ""
            )
            + f"<small>{text(report.get('model_id', 'Model not recorded'))}</small></td>"
            f'<td title="{text(summary["duration_source"])}">{elapsed}</td>'
            f"<td>{summary['optimizer_step'] if summary['optimizer_step'] is not None else '—'}</td></tr>"
        )
    return (
        '<div class="table-toolbar"><label for="run-search">Find a run</label>'
        '<input id="run-search" type="search" placeholder="Search title, result, model or ID">'
        '<label for="run-status">Execution</label><select id="run-status"><option value="">All</option>'
        '<option value="success">Succeeded</option><option value="failure">Failed</option></select>'
        '<span id="run-count" aria-live="polite"></span></div>'
        '<div class="table-scroll"><table><caption>Downloaded run attempts · newest UTC timestamp first</caption>'
        "<thead><tr><th>Run / recorded date</th><th>Execution</th><th>Saved evidence</th><th>Duration</th><th>Optimizer step</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def render_experiment(experiment: dict, runs: list[dict]) -> str:
    attempts = [run for run in runs if run["id"] in experiment["runs"]]
    sections = "".join(
        f'<section class="research-section"><h2>{label}</h2><p>{text(experiment[key])}</p></section>'
        for key, label in (
            ("intent", "Intent"),
            ("change", "What changed"),
            ("conclusion", "Conclusion"),
            ("decision", "Next decision"),
        )
    )
    limitations = "".join(f"<li>{text(item)}</li>" for item in experiment["limitations"])
    evidence = ""
    evidence_file = experiment.get("evidence_file")
    if evidence_file:
        report = read_json(Path(__file__).resolve().parents[5] / evidence_file)
        cost = report.get("approximate_compute_cost", {}).get("estimated_total_range_usd")
        if cost:
            evidence += f'<p class="cost-note">Approximate compute: ${cost[0]:.2f}–${cost[1]:.2f}. Includes failed resume attempt; excludes storage and earlier experiments. Not an invoice.</p>'
        evidence += folded("Structured result evidence · " + evidence_file, report)
    commit = experiment.get("code_commit")
    return (
        f'<header><div><div class="eyebrow">Experiment · {text(experiment["id"])}</div>'
        f'<h1>{text(experiment["title"])}</h1></div><span class="pill {text(experiment["status"])}">'
        f"{text(experiment['status'])}</span></header>"
        '<div class="evidence-strip"><span>Execution ≠ validation ≠ model improvement</span>'
        "<span>Held-out quality improvement: not established</span></div>"
        + sections
        + "<details><summary>Limitations and untested behavior</summary><ul>"
        + limitations
        + "</ul></details>"
        + (
            f'<p class="muted">Code revision: <code>{text(commit)}</code>. Prepared change; no live result yet.</p>'
            if commit
            else ""
        )
        + (
            "<h2>Run attempts</h2>" + render_run_table(attempts)
            if attempts
            else '<p class="prepared-note">No run launched for this change. CPU checks are not model experiments.</p>'
        )
        + evidence
    )


def render_index(root: Path, experiment_id: str = "", view: str = "overview") -> str:
    runs = discover_runs(root)
    experiments = experiment_catalog()
    if view == "runs":
        return layout(
            "Run attempts",
            '<header><div><div class="eyebrow">Evidence library</div>'
            "<h1>Run attempts</h1><p>Failures remain visible. Correctness and speed are separate from execution status.</p></div></header>"
            + render_run_table(runs),
            "runs",
        )
    selected = next((item for item in experiments if item["id"] == experiment_id), None)
    if view == "experiments" and selected is None:
        selected = experiments[0] if experiments else None
    links = "".join(
        f'<a class="experiment-link {"selected" if selected == item else ""}" data-experiment '
        f'href="/?experiment={quote(item["id"])}"><span class="status-dot {text(item["status"])}"></span>'
        f"<span><strong>{text(item['title'])}</strong><small>{text(item['status'])} · {len(item['runs'])} attempts</small></span></a>"
        for item in experiments
    )
    if selected:
        content = render_experiment(selected, runs)
    else:
        content = (
            '<header><div><div class="eyebrow">Kernel generation · Laguna XS.2</div>'
            '<h1>Research overview</h1><p class="lede">What works, what remains untested, and what to run next.</p></div></header>'
            '<section class="hero-result"><span class="pill success">Latest verified milestone</span>'
            "<h2>GRPO resumed. Step 2 is durable.</h2><p>Three of four admitted cummax kernels passed. "
            "The compact checkpoint passed CPU read-back. These are workflow checks, not a held-out learning gain.</p>"
            '<a href="/?experiment=checkpoint-resume">Read the evidence →</a></section>'
            '<div class="decision-grid"><section><h2>Prepared, not tested live</h2><p>4,096-token normal replies and clearer optional subcall guidance. '
            "Context, recovery limits and hardware are unchanged.</p></section>"
            "<section><h2>Next validation</h2><p>Restart from the compact checkpoint before a longer run. "
            "Measure end-to-end throughput and evaluate held-out tasks before claiming improvement.</p></section></div>"
            "<h2>Downloaded run attempts</h2>" + render_run_table(runs)
        )
    return layout(
        "Kernel-RLM Research",
        '<div class="experiment-workspace"><aside class="experiment-list">'
        '<h2>Experiments</h2><label class="sr-only" for="experiment-search">Search experiments</label>'
        '<input id="experiment-search" type="search" placeholder="Search experiments…">'
        + links
        + '<p class="muted">An experiment is a question. A run is one attempt. A trajectory is one agent episode.</p>'
        '</aside><div class="experiment-detail">' + content + "</div></div>",
        "experiments" if selected else "overview",
    )


def render_lineage(root: Path) -> str:
    runs = discover_runs(root)
    by_id = {run["id"]: run for run in runs}
    edges = []
    for run in runs:
        parent = run["report"].get("resumed_from")
        if parent:
            action = (
                "Resumed from"
                if run["report"].get("status") == "passed"
                else "Resume requested from"
            )
            edges.append(
                {
                    "from": parent,
                    "to": run["id"],
                    "kind": "checkpoint",
                    "label": f"{action} step {run['report'].get('resume_step', 'unknown')} checkpoint",
                }
            )
    for experiment in experiment_catalog():
        edges.extend(
            edge
            for edge in experiment.get("relationships", [])
            if edge["from"] in by_id and edge["to"] in by_id
        )
    connections = []
    for edge in edges:
        parent = by_id.get(edge["from"])
        child = by_id.get(edge["to"])
        connections.append(
            '<article class="lineage-edge">'
            f'<a class="lineage-node" href="/?run={quote(edge["from"])}"><strong>{text(run_title(parent) if parent else edge["from"])}</strong>'
            f"<small>{text(edge['from'])}</small></a>"
            f'<div class="lineage-connector {text(edge["kind"])}"><span>{text(edge["label"])}</span><b aria-hidden="true">→</b></div>'
            f'<a class="lineage-node" href="/?run={quote(edge["to"])}"><strong>{text(run_title(child) if child else edge["to"])}</strong>'
            f"<small>{text(edge['to'])}</small></a></article>"
        )
    return layout(
        "Experiment lineage",
        '<header><div><div class="eyebrow">Recorded dependencies</div><h1>Lineage</h1>'
        "<p>Solid links restore checkpoint state. Dashed links describe a retry, not inherited trained weights.</p></div></header>"
        '<div class="lineage-board">' + "".join(connections) + "</div>"
        '<p class="muted">Only explicit report or experiment relationships are shown. Chronological order does not create a dependency. '
        "A repeated source node represents the same run, not a duplicated checkpoint.</p>",
        "lineage",
    )


def render_configs(root: Path) -> str:
    rows = []
    for run in discover_runs(root):
        report = run["report"]
        if not any(report.get(key) for key in ("prime_revision", "model_revision")):
            continue
        rows.append(
            f'<tr><td><a href="/?run={quote(run["id"])}">{text(run_title(run))}</a><small>{text(run_date(run))}</small></td>'
            f"<td><code>{text(report.get('model_revision', 'Not recorded'))}</code></td>"
            f"<td><code>{text(report.get('prime_revision', 'Not recorded'))}</code></td>"
            f"<td>{execution_badge(run)}<small>Recorded run outcome only</small></td></tr>"
        )
    import tomllib

    configs = {}
    for name in ("grpo_pilot.toml", "grpo_level1.toml"):
        config = tomllib.loads((Path(__file__).parent.parent / "configs" / name).read_text())
        harness = config["orchestrator"]["train"]["source"][0]["env"]["agent"]["harness"]
        configs[name] = {
            "normal_reply_tokens": harness["max_tokens"],
            "context_tokens": config["seq_len"],
            "max_turns": harness["max_turns"],
            "max_subcalls": harness["max_subcalls"],
            "group_size": config["orchestrator"]["group_size"],
            "lora_rank": config["trainer"]["model"]["lora"]["rank"],
        }
    return layout(
        "Configs and compatibility",
        '<header><div><div class="eyebrow">Provenance</div><h1>Configs & compatibility</h1>'
        "<p>Historical revisions come from each saved report. Current files are shown separately; they do not describe older runs.</p></div></header>"
        + folded("Current training configs · prepared settings, not historical results", configs)
        + '<div class="table-scroll"><table><caption>Recorded component combinations; not an independent compatibility certification</caption>'
        "<thead><tr><th>Run</th><th>Model revision</th><th>Prime revision</th><th>Execution</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>",
        "configs",
    )


def render_trace(path: Path) -> str:
    value = read_json(path)
    output = value.get("output", {}) if isinstance(value, dict) else {}
    verification = value.get("verification", {}) if isinstance(value, dict) else {}
    feedback = value.get("feedback_results", []) if isinstance(value, dict) else []
    transcript = output.get("rlm_history", []) if isinstance(output, dict) else []
    items = [f'<div class="card"><b>{text(path.name)}</b>']
    items.append(
        f'<div class="muted">reward={text(output.get("reward", value.get("reward", "—")))} · turns={text(output.get("num_turns", "—"))}</div>'
    )
    if verification:
        items.append(
            f"<p>compiled={text(verification.get('compiled'))} · correct={text(verification.get('correct'))} · max_diff={text(verification.get('max_diff'))} · speedup={text(verification.get('speedup'))}</p>"
        )
    if feedback:
        items.append(
            "<h3>CUDA feedback</h3><pre>" + text(json.dumps(feedback, indent=2)) + "</pre>"
        )
    if transcript:
        items.append("<h3>Trajectory</h3>")
        for index, item in enumerate(transcript, 1):
            items.append(
                f"<details><summary>Step {index}</summary><pre>{text(json.dumps(item, indent=2))}</pre></details>"
            )
    items.append(
        "<details><summary>Raw trace</summary><pre>"
        + text(json.dumps(value, indent=2))
        + "</pre></details></div>"
    )
    return "".join(items)


def rollout_traces(run: dict) -> list[dict]:
    traces = {}
    paths = sorted(run.get("rollouts", []), key=lambda path: ("effective" in path.parts, str(path)))
    for path in paths:
        for row_index, row in enumerate(read_jsonl(path)):
            for trace_index, trace in enumerate(row.get("traces", [])):
                identity = trace.get("id") or (str(path), row_index, trace_index)
                traces.setdefault(identity, trace)
    return list(traces.values())


def folded(label: str, value: object) -> str:
    content = value if isinstance(value, str) else json.dumps(value, indent=2)
    return f"<details><summary>{text(label)}</summary><pre>{text(content)}</pre></details>"


def render_timeline(trace: dict) -> str:
    info = trace.get("info", {})
    calls = iter(trace.get("calls", []))
    if info.get("kernel_subcalls"):
        calls = iter(info.get("kernel_model_calls", []))
    events = []
    transcript = info.get("repl_transcript", [])
    if not transcript:
        sampled_nodes = {call["node"] for call in trace.get("calls", [])}
        transcript = [
            node["message"]
            for index, node in enumerate(trace.get("nodes", []))
            if node["message"].get("role") != "assistant" or index in sampled_nodes
        ]
    for index, message in enumerate(transcript, 1):
        role = message.get("role", "unknown")
        content = message.get("content", "")
        label = f"{index:02d} · {role.upper()}"
        if role == "assistant":
            call = next(calls, {})
            node_index = call.get("node")
            original = trace["nodes"][node_index]["message"] if node_index is not None else {}
            usage = call.get("usage", {})
            label += f" · {call.get('finish_reason', 'unknown')} · {usage.get('completion_tokens', '—')} output tokens"
            body = ""
            if original.get("reasoning_content"):
                body += folded("Reasoning", original["reasoning_content"])
            preview = content.split("```")[0].strip()[:240]
            if preview:
                body += f'<div class="card message">{text(preview)}</div>'
            body += folded("Assistant response · full text and code", content)
            if original.get("tool_calls"):
                body += folded(
                    "Structured tool calls · emitted, not proof of execution",
                    original["tool_calls"],
                )
        else:
            label_name = "System instructions" if role == "system" else "Prompt / REPL observation"
            body = folded(label_name, content)
        events.append(
            f'<article class="event"><div class="event-label">{text(label)}</div>{body}</article>'
        )
    if info.get("kernel_subcalls"):
        child_systems = [
            node["message"]
            for index, node in enumerate(trace.get("nodes", []))
            if index > 0 and node.get("parent") is None and node["message"].get("role") == "system"
        ]
        if child_systems:
            events.append(folded("Child system instructions", child_systems))
        events.append(
            folded("Child calls · full prompts, replies, usage and errors", info["kernel_subcalls"])
        )
    return '<div class="timeline">' + "".join(events) + "</div>"


def render_run(root: Path, run_id: str, trajectory: int = 0) -> str:
    run = next(
        (
            candidate
            for candidate in discover_runs(root)
            if candidate["id"] == run_id or candidate["id"] == run_id.rsplit("/", 1)[-1]
        ),
        None,
    )
    if run is None:
        return layout("Not found", '<h1>Run not found</h1><a href="/">Back</a>')
    report = run["report"]
    metrics = run.get("metrics", [])
    normalized = run_summary(run)
    traces = normalized["traces"]
    submitted = normalized["submitted"]
    rewards = [trace_reward(trace) for trace in traces]
    numeric_rewards = [reward for reward in rewards if isinstance(reward, int | float)]
    mean_reward = f"{sum(numeric_rewards) / len(numeric_rewards):.3f}" if numeric_rewards else "—"
    trajectory = max(0, min(trajectory, len(traces) - 1))
    selected = traces[trajectory] if traces else {}
    tokens = {
        key: sum(
            (call.get("usage") or {}).get(key, 0) for trace in traces for call in model_calls(trace)
        )
        for key in ("prompt_tokens", "completion_tokens")
    }
    summary = (
        '<aside class="sidebar"><a href="/?view=runs">← All run attempts</a><div class="card">'
        f"<h2>{text(report.get('model_id', 'Kernel-RLM'))}</h2>"
        f"{execution_badge(run)}"
        '<hr><div class="eyebrow">Mean saved kernel reward</div>'
        f'<div class="value">{mean_reward}</div><dl>'
        f"<dt>TRAJECTORIES</dt><dd>{len(traces)}</dd>"
        f"<dt>NONEMPTY SUBMISSIONS</dt><dd>{submitted} / {len(traces)}</dd>"
        f"<dt>ADMITTED FOR TRAINING</dt><dd>{normalized['admitted'] if normalized['admitted'] is not None else 'Not recorded'}</dd>"
        f"<dt>OPTIMIZER STEP</dt><dd>{normalized['optimizer_step'] if normalized['optimizer_step'] is not None else 'Not recorded'}</dd>"
        f"<dt>INPUT / OUTPUT TOKENS</dt><dd>{tokens['prompt_tokens']:,} / {tokens['completion_tokens']:,}</dd>"
        f"<dt>HARDWARE</dt><dd>{text(report.get('gpu_topology', 'Not recorded'))}</dd>"
        f"<dt>RUN ID</dt><dd>{text(run_id)}</dd></dl>"
        '<p class="muted">Counts describe downloaded artifacts, not necessarily the whole remote run.</p></div></aside>'
    )
    options = "".join(
        f'<option value="{index}" {"selected" if index == trajectory else ""}>'
        f"{index + 1} / {len(traces)} · {text(trace.get('task', {}).get('data', {}).get('name', 'unknown'))}"
        f" · reward {text(rewards[index])}</option>"
        for index, trace in enumerate(traces)
    )
    controls = (
        (
            '<div class="toolbar"><label for="trajectory">Trajectory</label>'
            f'<select id="trajectory">{options}</select>'
            '<label><input id="expand" type="checkbox"> Expand messages</label></div>'
        )
        if traces
        else ""
    )
    info = selected.get("info", {})
    has_submission = trace_submitted(selected)
    length_calls = sum(call.get("finish_reason") == "length" for call in model_calls(selected))
    evidence = (
        (
            f'<p><span class="badge">{"Submitted" if has_submission else "No final submission"}</span> '
            f'<span class="badge">{length_calls} length-limited calls</span></p>'
            '<p class="muted">A completed agent process does not imply a submitted or correct kernel.</p>'
        )
        if traces
        else ""
    )
    timeline = (
        render_timeline(selected)
        if traces
        else "".join(render_trace(path) for path in run.get("traces", []))
        or '<p class="empty">No trajectory messages downloaded.</p>'
    )
    verification = (
        (
            folded("Recorded reward", selected.get("rewards", {}))
            + folded("Final kernel submission", info.get("kernel_submission", ""))
            + folded(
                "Final verifier result · correctness and measured speed",
                info.get("kernel_verify", {}),
            )
            + folded(
                "Recovery and rejected actions",
                {
                    "recovery": info.get("kernel_recovery", []),
                    "rejected_actions": info.get("kernel_action_rejections", []),
                },
            )
            + folded("Recorded errors", selected.get("errors", []))
            + folded("Task and reference", selected.get("task", {}))
            + '<p class="muted">Compilation feedback appears in the REPL observations. An empty submission never reached final kernel scoring.</p>'
        )
        if traces
        else '<p class="empty">No Prime verification record downloaded. Inspect the saved eval trace.</p>'
    )
    artifacts = folded("Run report and configuration", report)
    artifacts += folded("Selected raw Prime trace", selected) if selected else ""
    for path in run.get("episodes", []):
        artifacts += folded(path.name, read_json(path))
    artifacts += folded(
        "Downloaded trace files", [str(path.relative_to(root)) for path in run.get("rollouts", [])]
    )
    center = (
        '<section><div class="tabs" role="tablist" aria-label="Run views">'
        '<button role="tab" data-tab="trace" aria-selected="true">Run trace</button>'
        '<button role="tab" data-tab="verification" aria-selected="false">Verification</button>'
        '<button role="tab" data-tab="artifacts" aria-selected="false">Artifacts</button></div>'
        + controls
        + evidence
        + f'<div class="panel" id="trace">{timeline}</div>'
        + f'<div class="panel" id="verification" hidden>{verification}</div>'
        + f'<div class="panel" id="artifacts" hidden>{artifacts}</div></section>'
    )
    curves = (
        '<aside class="metrics"><h2 class="eyebrow">Saved measurements</h2>'
        '<button id="expand-charts" type="button" aria-expanded="false">Expand charts</button>'
    )
    for label, keys in (
        (
            "Admitted reward",
            (
                "train/agg/effective/agent/reward/mean",
                "train/kernel-rlm-level1-pilot/effective/agent/reward/mean",
                "reward/mean",
                "reward",
                "train/reward",
            ),
        ),
        ("Loss", ("loss/mean", "loss", "train/loss", "policy_loss")),
        ("Entropy (nats)", ("entropy/all/mean", "entropy", "train/entropy")),
    ):
        if any(isinstance(metric_value([row], keys), (int, float)) for row in metrics):
            curves += (
                f'<figure class="chart-card"><h3>{label}</h3>'
                + chart_svg(metrics, keys, label)
                + "</figure>"
            )
    if traces:
        reward_bars = "".join(
            f'<div class="reward-row"><span title="{text(trace.get("task", {}).get("data", {}).get("name", "unknown"))}">T{index + 1}</span>'
            f'<meter min="0" max="1" value="{reward}">{reward:.3f}</meter><strong>{reward:.3f}</strong></div>'
            for index, trace in enumerate(traces)
            if (reward := trace_reward(trace)) is not None
        )
        curves += (
            '<figure class="chart-card"><h3>Reward by saved trajectory</h3>'
            + reward_bars
            + (
                '<p class="plot-source">Reward (0–1) · all unique saved trajectories, including filtered groups. '
                "Source: traces.jsonl. Each T number matches the selector.</p></figure>"
            )
        )
    curves += render_gpu_charts(run) + "</aside>"
    heading = (
        f'<header class="run-header"><div><div class="eyebrow">Run attempt · {text(run_date(run))}</div>'
        f'<h1>{text(run_title(run))}</h1><p class="muted">Execution status is not a held-out quality result.</p></div>'
        '<a class="button-link" href="/?view=lineage">View lineage</a></header>'
    )
    return layout(
        f"Kernel-RLM · {run_id}",
        heading + '<div class="workspace">' + summary + center + curves + "</div>",
    )


def make_handler(root: Path):
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            request = urlparse(self.path)
            project_root = Path(__file__).resolve().parents[5]
            static_pages = {
                "/guide": ("PROJECT_TRACKER.html", "text/html"),
                "/PROJECT_TRACKER_ARCHIVE.html": ("PROJECT_TRACKER_ARCHIVE.html", "text/html"),
                "/notes/datasets.md": ("notes/datasets.md", "text/markdown"),
                "/notes/reward-design.md": ("notes/reward-design.md", "text/markdown"),
                "/rlm/training/environments/kernel_rlm/README.md": (
                    "rlm/training/environments/kernel_rlm/README.md",
                    "text/markdown",
                ),
            }
            if request.path in static_pages:
                relative_path, media_type = static_pages[request.path]
                try:
                    document = (
                        render_guide_workspace()
                        if request.path == "/guide"
                        else (project_root / relative_path).read_text(encoding="utf-8")
                    )
                except OSError:
                    self.send_error(404, "Project document is unavailable")
                    return
                payload = document.encode()
                self.send_response(200)
                self.send_header("Content-Type", f"{media_type}; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            query = parse_qs(request.query)
            run_id = unquote(query.get("run", [""])[0])
            raw_trajectory = query.get("trajectory", ["0"])[0]
            trajectory = (
                int(raw_trajectory)
                if raw_trajectory.isdecimal() and len(raw_trajectory) < 10
                else 0
            )
            view = query.get("view", ["overview"])[0]
            experiment = query.get("experiment", [""])[0]
            if run_id:
                document = render_run(root, run_id, trajectory)
            elif view == "lineage":
                document = render_lineage(root)
            elif view == "configs":
                document = render_configs(root)
            else:
                document = render_index(root, experiment, view)
            payload = document.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    return DashboardHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("logs"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7789)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(root))
    print(f"Kernel-RLM dashboard: http://{args.host}:{args.port}/ (root={root})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
