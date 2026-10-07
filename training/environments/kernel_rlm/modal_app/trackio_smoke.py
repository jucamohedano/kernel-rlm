"""Private Trackio connectivity test on Modal CPU, with no training metrics."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import modal

SPACE_ID = "jucamohedano/laguna-kernel-tracking"
BUCKET_ID = "jucamohedano/laguna-kernel-tracking-data"
PROJECT = "laguna-kernel-rlm"
VERSION = "0.40.0"
app = modal.App("kernel-rlm-trackio-check")
reports = modal.Volume.from_name("kernel-rlm-traces")
secret = modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]} if modal.is_local() else {})


@app.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install(f"trackio=={VERSION}"),
    secrets=[secret],
    volumes={"/reports": reports},
    cpu=1,
    memory=2048,
    timeout=600,
    retries=0,
    max_containers=1,
)
def verify(run_id: str, hosted: bool = False) -> dict[str, Any]:
    os.environ["TRACKIO_DIR"] = "/reports/trackio-hosted" if hosted else "/reports/trackio-local"
    import trackio
    from gradio_client import Client
    from huggingface_hub import HfApi
    from huggingface_hub.errors import BucketNotFoundError, RepositoryNotFoundError

    report: dict[str, Any] = {
        "status": "running",
        "scope": "private dashboard connectivity; not training",
        "gpu_allocated": False,
        "space_id": SPACE_ID,
        "bucket_id": BUCKET_ID,
        "trackio_version": VERSION,
        "run_name": run_id,
        "hosted": hosted,
    }
    destination = Path("/reports/trackio-smoke") / run_id / "report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not hosted:
            from trackio.sqlite_storage import SQLiteStorage

            trackio.init(
                project=PROJECT,
                name=run_id,
                group="connectivity-tests",
                config={"purpose": "local persistence test on Modal CPU", "training": False},
                auto_log_gpu=False,
                auto_log_cpu=False,
                embed=False,
            )
            try:
                trackio.log({"preflight/connectivity": 1}, step=0)
            finally:
                trackio.finish()
            metric_names = SQLiteStorage.get_all_metrics_for_run(PROJECT, run_id)
            assert "preflight/connectivity" in metric_names
            report.update(
                status="passed",
                persisted_metrics=metric_names,
                storage="kernel-rlm-traces:trackio-local",
                dashboard_url=None,
            )
            return report
        api = HfApi()
        assert api.whoami()["name"] == SPACE_ID.split("/")[0]
        try:
            assert api.space_info(SPACE_ID).private, "Existing Space must be private"
        except RepositoryNotFoundError:
            pass
        try:
            assert api.bucket_info(BUCKET_ID).private, "Existing bucket must be private"
        except BucketNotFoundError:
            pass
        report["stage"] = "initialize-private-space"
        destination.write_text(json.dumps(report, indent=2))
        reports.commit()
        trackio.init(
            project=PROJECT,
            name=run_id,
            group="connectivity-tests",
            space_id=SPACE_ID,
            bucket_id=BUCKET_ID,
            private=True,
            config={"purpose": "connectivity test", "training": False},
            auto_log_gpu=False,
            auto_log_cpu=False,
            embed=False,
        )
        try:
            trackio.log({"preflight/connectivity": 1}, step=0)
        finally:
            trackio.finish()
        assert api.space_info(SPACE_ID).private
        assert api.bucket_info(BUCKET_ID).private
        client = Client(SPACE_ID, token=os.environ["HF_TOKEN"], verbose=False)
        metric_names = client.predict(project=PROJECT, run=run_id, api_name="/get_metrics_for_run")
        assert "preflight/connectivity" in metric_names, "Remote metric read-back failed"
        report.update(
            status="passed",
            private_space=True,
            private_bucket=True,
            remote_metrics=metric_names,
            dashboard_url=f"https://huggingface.co/spaces/{SPACE_ID}",
        )
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise RuntimeError(f"Trackio check failed: {type(error).__name__}") from None
    finally:
        destination.write_text(json.dumps(report, indent=2))
        reports.commit()
    return report


@app.local_entrypoint()
def main(hosted: bool = False) -> None:
    run_id = "connectivity-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    print(f"Report: kernel-rlm-traces:trackio-smoke/{run_id}/report.json", flush=True)
    report = verify.remote(run_id, hosted=hosted)
    destination = Path("logs/trackio-smoke") / run_id / "report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
