"""Serve a read-only local research dashboard."""

from __future__ import annotations

import argparse
import json
import mimetypes
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import data, writing


def make_handler(root: Path, receipt_path: Path | None = None, task_catalog: Path | None = None):
    root = root.resolve()
    write_token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def send_data(self, value, content_type="application/json; charset=utf-8"):
            body = value if isinstance(value, bytes) else json.dumps(value).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            request = urlparse(self.path)
            path = request.path
            selected_trajectory = parse_qs(request.query).get("trajectory", [None])[0]
            if path == "/api/home":
                return self.send_data(data.home(root, receipt_path))
            if path == "/api/trajectories":
                return self.send_data(data.trajectory_index(root, task_catalog_path=task_catalog))
            if path == "/api/experiments":
                return self.send_data({"experiments": data.experiments()})
            if path == "/api/runs":
                records = data.runs(root)
                receipt = data.read_json(receipt_path) if receipt_path else {}
                for record in records:
                    if record["id"] == receipt.get("run_id"):
                        record["driver_status"] = record["status"]
                        record["status"] = receipt.get("training_status") or record["status"]
                return self.send_data({"runs": records})
            if path == "/api/writing":
                return self.send_data({"documents": writing.list_documents()})
            if path.startswith("/api/writing/"):
                doc_id = unquote(path.removeprefix("/api/writing/"))
                try:
                    document = writing.read_document(doc_id)
                except (FileNotFoundError, ValueError):
                    return self.send_error(404)
                document["write_token"] = write_token
                return self.send_data(document)
            if path.startswith("/api/runs/"):
                run_id = unquote(path.removeprefix("/api/runs/"))
                from training.environments.kernel_rlm.dashboard import server as reader

                found = next(
                    (
                        record
                        for record in data.run_records(root, count_trajectories=False)
                        if record["id"] == run_id
                    ),
                    None,
                )
                if found is None:
                    return self.send_error(404)
                if not found.get("evidence_available"):
                    return self.send_data(
                        {
                            "run": {
                                "id": run_id,
                                "title": found.get("title", run_id),
                                "status": found.get("status"),
                                "model": None,
                                "hardware": None,
                                "verifier": None,
                                "source": run_id,
                                "evidence_available": False,
                            },
                            "trajectories": [],
                            "warnings": found.get("warnings")
                            or [
                                "No downloaded report or supported trajectory export; local evidence is unavailable."
                            ],
                        }
                    )
                run = {**found["_reader"], "experiments": found.get("experiments", [])}
                report = run.get("report", {})
                warnings = list(found.get("warnings", []))
                trajectories = []
                traces, trace_warnings = data.saved_traces(run)
                warnings.extend(trace_warnings)
                attempts = {}
                for index, trace in enumerate(traces):
                    if not isinstance(trace, dict):
                        warnings.append(f"Trajectory {index} is malformed and was skipped.")
                        continue
                    try:
                        detail = data.trajectory_detail(
                            run, trace, index, task_catalog_path=task_catalog
                        )
                        task_id = detail.get("task")
                        attempts[task_id] = attempts.get(task_id, 0) + 1
                        detail["attempt"] = attempts[task_id]
                        if selected_trajectory != detail["id"]:
                            for field in ("events", "raw", "verification"):
                                detail.pop(field, None)
                        trajectories.append(detail)
                    except (AttributeError, TypeError, ValueError) as error:
                        warnings.append(
                            f"Trajectory {index} could not be normalized: {type(error).__name__}."
                        )
                run_detail = {
                    "id": run_id,
                    "title": reader.run_title(run),
                    "status": report.get("status"),
                    "model": report.get("model_id") or report.get("model"),
                    "hardware": report.get("gpu_topology") or report.get("hardware"),
                    "elapsed_seconds": report.get("elapsed_seconds"),
                    "elapsed_scope": report.get("elapsed_scope"),
                    "verifier": None,
                    "source": run_id,
                    "evidence_available": found.get("evidence_available", False),
                }
                receipt = data.read_json(receipt_path) if receipt_path else {}
                if receipt.get("run_id") == run_id:
                    run_detail["driver_status"] = report.get("status")
                    run_detail["completion_status"] = receipt.get("training_status")
                    run_detail["status"] = receipt.get("training_status") or report.get("status")
                return self.send_data(
                    {"run": run_detail, "trajectories": trajectories, "warnings": warnings}
                )
            if path == "/archive/guide":
                legacy = data.REPO_ROOT / "training/environments/kernel_rlm/dashboard/guide.html"
                if not legacy.is_file():
                    return self.send_error(404)
                return self.send_data(legacy.read_bytes(), "text/html; charset=utf-8")
            files = {
                "/": "index.html",
                "/app.js": "app.js",
                "/styles.css": "styles.css",
                "/layout.js": "layout.js",
                "/fonts/InterVariable.woff2": "fonts/InterVariable.woff2",
                "/fonts/JetBrainsMono-Regular.woff2": "fonts/JetBrainsMono-Regular.woff2",
                "/fonts/JetBrainsMono-Medium.woff2": "fonts/JetBrainsMono-Medium.woff2",
                "/fonts/JetBrainsMono-Bold.woff2": "fonts/JetBrainsMono-Bold.woff2",
            }
            if path in files:
                file = Path(__file__).parent / files[path]
                if not file.is_file():
                    return self.send_error(404)
                content_type = (
                    "font/woff2"
                    if file.suffix == ".woff2"
                    else mimetypes.guess_type(file.name)[0] + "; charset=utf-8"
                )
                return self.send_data(file.read_bytes(), content_type)
            return self.send_error(404)

        def do_POST(self):
            path = urlparse(self.path).path
            if not path.startswith("/api/writing/"):
                return self.send_error(404)
            expected_origin = f"http://{self.headers.get('Host')}"
            if (
                self.headers.get("Origin") != expected_origin
                or self.headers.get("X-Write-Token") != write_token
            ):
                return self.send_error(403, "Same-origin write token required")
            if self.headers.get("Content-Type") != "application/json":
                return self.send_error(415)
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self.send_error(400)
            if not 0 < length <= 2 * writing.MAX_DOCUMENT_BYTES:
                return self.send_error(413)
            try:
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Expected JSON object")
                result = writing.save_document(
                    unquote(path.removeprefix("/api/writing/")),
                    payload.get("markdown"),
                    payload.get("version"),
                )
            except writing.ConflictError:
                return self.send_error(409, "File changed on disk")
            except FileNotFoundError:
                return self.send_error(404)
            except (ValueError, UnicodeError):
                return self.send_error(400)
            except OSError:
                return self.send_error(500, "Could not save local document")
            return self.send_data(result)

        def log_message(self, fmt, *args):
            return

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        required=True,
        help="Existing directory recursively searched for saved run artifacts",
    )
    parser.add_argument(
        "--campaign-receipt", type=Path, help="Optional explicit campaign completion receipt JSON"
    )
    parser.add_argument(
        "--task-catalog",
        type=Path,
        help="Optional task JSONL with exact _task_id and ops/source metadata",
    )
    parser.add_argument("--port", type=int, default=7790)
    args = parser.parse_args()
    root = args.evidence_root.resolve()
    if not root.is_dir():
        parser.error("evidence root must be an existing directory")
    receipt = args.campaign_receipt.resolve() if args.campaign_receipt else None
    catalog = args.task_catalog.resolve() if args.task_catalog else None
    if catalog is not None and not catalog.is_file():
        parser.error("task catalog must be an existing file")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(root, receipt, catalog))
    print(f"Dashboard: http://127.0.0.1:{args.port}/ (evidence={root})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
