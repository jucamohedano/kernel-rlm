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
        f'<text x="8" y="118">{low:.3g}</text><text x="350" y="118">{high:.3g}</text></svg>'
    )


def layout(title: str, body: str) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{text(title)}</title><style>
:root {{ color-scheme: dark; --bg:#101512; --card:#18221b; --line:#2d4433; --muted:#9cad9e; --green:#71e39a; --amber:#f3c969; --red:#ff8b8b; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:#eaf2eb; font:14px/1.5 system-ui,sans-serif; }}
main {{ max-width:1280px; margin:0 auto; padding:28px; }} h1,h2,h3 {{ margin:0 0 10px; }} h1 {{ font-size:28px; }} h2 {{ font-size:18px; margin-top:26px; }}
a {{ color:var(--green); }} .muted,.empty {{ color:var(--muted); }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; }} .value {{ font-size:24px; font-weight:700; color:var(--green); }}
.run {{ display:flex; justify-content:space-between; gap:16px; align-items:center; }} .run + .run {{ margin-top:10px; }}
table {{ width:100%; border-collapse:collapse; }} td,th {{ text-align:left; padding:8px; border-bottom:1px solid var(--line); vertical-align:top; }} th {{ color:var(--muted); font-weight:500; }}
svg {{ width:100%; height:180px; background:#0c110d; border-radius:8px; }} svg text {{ fill:var(--muted); font-size:10px; }} pre {{ white-space:pre-wrap; overflow:auto; max-height:520px; background:#0c110d; padding:14px; border-radius:8px; }}
details {{ border:1px solid var(--line); border-radius:8px; padding:10px 12px; margin:8px 0; }} summary {{ cursor:pointer; color:var(--green); }} .pill {{ border:1px solid var(--line); border-radius:999px; padding:3px 8px; color:var(--muted); }}
header {{ display:flex; justify-content:space-between; align-items:baseline; gap:12px; margin-bottom:22px; }}
@media(max-width:650px) {{ main {{ padding:16px; }} .run {{ display:block; }} }}
</style></head><body><main>{body}</main></body></html>"""


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
        '<h2>Runs</h2>'
        + ("".join(cards) or '<div class="card empty">No reports found. Download a Modal run directory into this root.</div>')
    )
    return layout("Kernel-RLM Experiments", body)


def render_trace(path: Path) -> str:
    value = read_json(path)
    output = value.get("output", {}) if isinstance(value, dict) else {}
    verification = value.get("verification", {}) if isinstance(value, dict) else {}
    feedback = value.get("feedback_results", []) if isinstance(value, dict) else []
    transcript = output.get("rlm_history", []) if isinstance(output, dict) else []
    items = [f"<div class=\"card\"><b>{text(path.name)}</b>"]
    items.append(f"<div class=\"muted\">reward={text(output.get('reward', value.get('reward', '—')))} · turns={text(output.get('num_turns', '—'))}</div>")
    if verification:
        items.append(f"<p>compiled={text(verification.get('compiled'))} · correct={text(verification.get('correct'))} · max_diff={text(verification.get('max_diff'))} · speedup={text(verification.get('speedup'))}</p>")
    if feedback:
        items.append("<h3>CUDA feedback</h3><pre>" + text(json.dumps(feedback, indent=2)) + "</pre>")
    if transcript:
        items.append("<h3>Trajectory</h3>")
        for index, item in enumerate(transcript, 1):
            items.append(f"<details><summary>Step {index}</summary><pre>{text(json.dumps(item, indent=2))}</pre></details>")
    items.append("<details><summary>Raw trace</summary><pre>" + text(json.dumps(value, indent=2)) + "</pre></details></div>")
    return "".join(items)


def render_run(root: Path, run_id: str) -> str:
    run = next((candidate for candidate in discover_runs(root) if candidate["id"] == run_id), None)
    if run is None:
        return layout("Not found", '<h1>Run not found</h1><a href="/">Back</a>')
    report = run["report"]
    metrics = run.get("metrics", [])
    trace_cards = [render_trace(path) for path in run.get("traces", [])]
    episode_cards = [f'<details class="card"><summary>{text(path.name)}</summary><pre>{text(json.dumps(read_json(path), indent=2))}</pre></details>' for path in run.get("episodes", [])]
    body = (
        f'<header><div><h1>{text(run_id)}</h1><div class="muted">{text(report.get("model_id", "Kernel-RLM"))}</div></div><a href="/">← all runs</a></header>'
        '<div class="grid">'
        f'<div class="card"><div class="muted">status</div><div class="value">{text(report.get("status", "artifacts"))}</div></div>'
        f'<div class="card"><div class="muted">latest reward</div><div class="value">{text(metric_value(metrics, ("reward/mean", "reward", "train/reward")))}</div></div>'
        f'<div class="card"><div class="muted">metric rows</div><div class="value">{len(metrics)}</div></div>'
        f'<div class="card"><div class="muted">traces / episodes</div><div class="value">{len(trace_cards)} / {len(episode_cards)}</div></div></div>'
        '<h2>Learning curves</h2><div class="grid">'
        '<div class="card"><h3>Reward</h3>' + chart_svg(metrics, ("reward/mean", "reward", "train/reward"), "#71e39a") + '</div>'
        '<div class="card"><h3>Loss</h3>' + chart_svg(metrics, ("loss", "train/loss", "policy_loss"), "#f3c969") + '</div>'
        '<div class="card"><h3>Entropy</h3>' + chart_svg(metrics, ("entropy", "train/entropy"), "#8ecbff") + '</div></div>'
        '<h2>Run metadata</h2><div class="card"><pre>' + text(json.dumps(report, indent=2)) + '</pre></div>'
        '<h2>Traces</h2>' + ("".join(trace_cards) or '<div class="card empty">No rollout traces downloaded yet.</div>')
        + '<h2>Episodes</h2>' + ("".join(episode_cards) or '<div class="card empty">No episode records downloaded yet.</div>')
    )
    return layout(f"Kernel-RLM · {run_id}", body)


def make_handler(root: Path):
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            query = parse_qs(urlparse(self.path).query)
            run_id = unquote(query.get("run", [""])[0])
            document = render_run(root, run_id) if run_id else render_index(root)
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
