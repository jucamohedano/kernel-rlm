"""Check Prime external-run logging on Modal CPU without launching training."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import modal

app = modal.App("kernel-rlm-prime-monitoring-check")
reports = modal.Volume.from_name("kernel-rlm-traces")


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    secrets=[
        modal.Secret.from_name("prime-intellect-secret"),
        modal.Secret.from_name("prime-intellect-team-id-secret"),
    ],
    volumes={"/reports": reports},
    cpu=0.25,
    memory=256,
    timeout=120,
    retries=0,
    max_containers=1,
)
def verify(run_id: str) -> dict[str, Any]:
    import http.client
    import os
    import re

    api_key = os.environ.get("PRIME_API_KEY", "")
    team_id = os.environ.get("PRIME_TEAM_ID", "")
    report: dict[str, Any] = {
        "status": "failed",
        "scope": "external-run registration, connectivity metric, finalization; no training",
        "gpu_allocated": False,
        "api_key_present": bool(api_key),
        "team_id_present": bool(team_id),
        "api_key_has_outer_whitespace": api_key != api_key.strip(),
    }
    destination = Path("/reports/prime-monitoring") / run_id / "report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)

    def request(
        method: str, path: str, payload: dict[str, Any] | None = None
    ) -> tuple[int, dict[str, Any]]:
        connection = http.client.HTTPSConnection("api.primeintellect.ai", timeout=20)
        try:
            connection.request(
                method,
                path,
                body=json.dumps(payload) if payload is not None else None,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            response = connection.getresponse()
            body = response.read()
            return response.status, json.loads(body) if body else {}
        finally:
            connection.close()

    try:
        if not api_key:
            report["reason"] = "PRIME_API_KEY missing from Modal Secret"
            return report
        status, _identity = request("GET", "/api/v1/user/whoami")
        report["identity_http_status"] = status
        if status != 200:
            report["reason"] = "Read-only identity authentication failed; registration skipped"
            return report
        payload = {
            "name": f"kernel-rlm-monitoring-preflight-{run_id}",
            "base_model": "poolside/Laguna-XS.2",
            "max_steps": 0,
            "run_config": {"purpose": "logging access check only", "training": False},
        }
        if team_id:
            payload["team_id"] = team_id
        status, body = request("POST", "/api/v1/rft/external-runs", payload)
        report["registration_http_status"] = status
        if status != 201:
            report["reason"] = "Platform rejected external-run registration"
            detail = body.get("detail", body.get("message", ""))
            if isinstance(detail, str):
                detail = detail.replace(api_key, "[REDACTED]")
                if team_id:
                    detail = detail.replace(team_id, "[TEAM]")
                report["platform_detail"] = re.sub(r"pit_[A-Za-z0-9_-]+", "[REDACTED]", detail)[
                    :300
                ]
            return report
        platform_id = body["run"]["id"]
        report["platform_run_id"] = platform_id
        destination.write_text(json.dumps(report, indent=2))
        reports.commit()
        metric_ok = False
        try:
            status, _body = request(
                "POST",
                "/api/v1/rft/metrics",
                {"run_id": platform_id, "metrics": {"preflight/connectivity": 1}},
            )
            report["metrics_http_status"] = status
            metric_ok = 200 <= status < 300
        finally:
            status, _body = request(
                "PUT",
                f"/api/v1/rft/external-runs/{platform_id}/status",
                {"status": "completed" if metric_ok else "failed"},
            )
            report["finalization_http_status"] = status
        if metric_ok and 200 <= status < 300:
            report["status"] = "passed"
    except Exception as error:
        report["error_type"] = type(error).__name__
    finally:
        destination.write_text(json.dumps(report, indent=2))
        reports.commit()
    return report


@app.local_entrypoint()
def main() -> None:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12]
    print(f"Report: kernel-rlm-traces:prime-monitoring/{run_id}/report.json", flush=True)
    report = verify.remote(run_id)
    destination = Path("logs/prime-monitoring") / run_id / "report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("Prime monitoring preflight failed; see sanitized report")
