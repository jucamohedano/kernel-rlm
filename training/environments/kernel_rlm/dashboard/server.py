"""Serve a local, dependency-free dashboard for saved Kernel-RLM runs."""

from __future__ import annotations

import argparse
import html
import json
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
        run_id = str(report.get("run_id") or report_path.parent.relative_to(root))
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
    return sorted(runs.values(), key=lambda run: run["id"], reverse=True)


def text(value: object) -> str:
    return html.escape(str(value))


def metric_value(metrics: list[dict], names: tuple[str, ...]) -> object:
    for row in reversed(metrics):
        for name in names:
            if name in row:
                return row[name]
    return "—"


def chart_svg(metrics: list[dict], key_names: tuple[str, ...], color: str) -> str:
    points: list[float] = []
    for row in metrics:
        value = metric_value([row], key_names)
        if isinstance(value, int | float):
            points.append(float(value))
    if len(points) < 2:
        return '<div class="empty">No series recorded yet.</div>'
    low, high = min(points), max(points)
    span = high - low or 1.0
    coords = []
    for index, value in enumerate(points):
        x = 8 + index * 384 / max(len(points) - 1, 1)
        y = 112 - (value - low) * 96 / span
        coords.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg viewBox="0 0 400 120" role="img" aria-label="metric chart">'
        f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="3" />'
        f'<text x="8" y="12">max {high:.3g}</text><text x="8" y="118">min {low:.3g}</text></svg>'
        '<div class="muted">X: saved sample order · Y: metric value</div>'
    )


def layout(title: str, body: str) -> str:
    assets = Path(__file__).parent
    styles = (assets / "viewer.css").read_text()
    script = (assets / "viewer.js").read_text()
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{text(title)}</title><style>{styles}</style></head>
<body><nav><strong>Kernel<span style="color:var(--accent)">RLM</span> / experiments</strong>
<a href="/">All runs</a></nav><main>{body}
<footer>Local saved artifacts · No GPU jobs · Layout inspired by
<a href="https://posttrainbench.com/traces/">PostTrainBench</a></footer>
</main><script>{script}</script></body></html>"""


def render_index(root: Path) -> str:
    runs = discover_runs(root)
    cards = []
    for run in runs:
        report = run["report"]
        metrics = run.get("metrics", [])
        status = report.get("status", "artifacts")
        cards.append(
            f'<div class="card run"><div><h3><a href="/?run={quote(run["id"])}">{text(run["id"])}</a></h3>'
            f'<div class="muted">{text(report.get("model_id", "Kernel-RLM"))}</div></div>'
            f'<span class="pill">{text(status)} · {len(metrics)} metric rows</span></div>'
        )
    body = (
        '<header><div><h1>Kernel-RLM Experiments</h1><div class="muted">Saved Modal artifacts · CPU-only viewer</div></div>'
        f'<span class="pill">{len(runs)} runs</span></header>'
        "<h2>Runs</h2>"
        + (
            "".join(cards)
            or '<div class="card empty">No reports found. Download a Modal run directory into this root.</div>'
        )
    )
    return layout("Kernel-RLM Experiments", body)


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
    return [
        trace
        for path in sorted(run.get("rollouts", []))
        for row in read_jsonl(path)
        for trace in row.get("traces", [])
    ]


def folded(label: str, value: object) -> str:
    content = value if isinstance(value, str) else json.dumps(value, indent=2)
    return f"<details><summary>{text(label)}</summary><pre>{text(content)}</pre></details>"


def render_timeline(trace: dict) -> str:
    info = trace.get("info", {})
    calls = iter(trace.get("calls", []))
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
    return '<div class="timeline">' + "".join(events) + "</div>"


def render_run(root: Path, run_id: str, trajectory: int = 0) -> str:
    run = next((candidate for candidate in discover_runs(root) if candidate["id"] == run_id), None)
    if run is None:
        return layout("Not found", '<h1>Run not found</h1><a href="/">Back</a>')
    report = run["report"]
    metrics = run.get("metrics", [])
    traces = rollout_traces(run)
    submitted = sum(bool(trace.get("info", {}).get("kernel_submission")) for trace in traces)
    rewards = [trace.get("rewards", {}).get("kernel_reward", {}).get("score") for trace in traces]
    numeric_rewards = [reward for reward in rewards if isinstance(reward, int | float)]
    mean_reward = f"{sum(numeric_rewards) / len(numeric_rewards):.3f}" if numeric_rewards else "—"
    trajectory = max(0, min(trajectory, len(traces) - 1))
    selected = traces[trajectory] if traces else {}
    tokens = {
        key: sum(
            call.get("usage", {}).get(key, 0) for trace in traces for call in trace.get("calls", [])
        )
        for key in ("prompt_tokens", "completion_tokens")
    }
    summary = (
        '<aside class="sidebar"><a href="/">← Back to all runs</a><div class="card">'
        f"<h2>{text(report.get('model_id', 'Kernel-RLM'))}</h2>"
        f'<span class="badge">{text(report.get("status", "saved artifacts"))}</span>'
        '<hr><div class="eyebrow">Mean saved kernel reward</div>'
        f'<div class="value">{mean_reward}</div><dl>'
        f"<dt>TRAJECTORIES</dt><dd>{len(traces)}</dd>"
        f"<dt>NONEMPTY SUBMISSIONS</dt><dd>{submitted} / {len(traces)}</dd>"
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
    has_submission = bool(info.get("kernel_submission"))
    length_calls = sum(call.get("finish_reason") == "length" for call in selected.get("calls", []))
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
    curves = '<aside class="metrics"><h2 class="eyebrow">Saved metrics</h2>'
    for label, keys, color in (
        ("Reward", ("reward/mean", "reward", "train/reward"), "#ab694c"),
        ("Loss", ("loss", "train/loss", "policy_loss"), "#8c803d"),
        ("Entropy", ("entropy", "train/entropy"), "#6e8150"),
    ):
        curves += f'<div class="card"><h3>{label}</h3>' + chart_svg(metrics, keys, color) + "</div>"
    curves += (
        '<div class="card"><h3>GPU telemetry</h3><p class="muted">'
        "Not connected in this viewer. No utilization or memory values are inferred from missing data."
        "</p></div></aside>"
    )
    return layout(
        f"Kernel-RLM · {run_id}", '<div class="workspace">' + summary + center + curves + "</div>"
    )


def make_handler(root: Path):
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            query = parse_qs(urlparse(self.path).query)
            run_id = unquote(query.get("run", [""])[0])
            raw_trajectory = query.get("trajectory", ["0"])[0]
            trajectory = (
                int(raw_trajectory)
                if raw_trajectory.isdecimal() and len(raw_trajectory) < 10
                else 0
            )
            document = render_run(root, run_id, trajectory) if run_id else render_index(root)
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
