"""Export a completed metrics file from Modal storage without occupying a GPU."""

import os
from pathlib import Path

import modal

from .trackio_metrics import import_metrics

reports = modal.Volume.from_name("kernel-rlm-traces")
app = modal.App("kernel-rlm-trackio-export")


@app.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install("trackio==0.40.0"),
    volumes={"/reports": reports},
    cpu=1,
    memory=2048,
    timeout=180,
    max_containers=1,
    retries=0,
)
def export(relative_path: str, run_name: str, scope: str) -> dict:
    root = Path("/reports").resolve()
    source = (root / relative_path).resolve()
    if not source.is_relative_to(root):
        raise ValueError("Metrics must be inside the reports volume")
    reports.reload()
    os.environ["TRACKIO_DIR"] = "/reports/trackio-local"
    result = import_metrics(source, run_name, {"scope": scope})
    reports.commit()
    return result


@app.local_entrypoint()
def main(relative_path: str, run_name: str, scope: str = "prime-rl metrics snapshot") -> None:
    print(export.remote(relative_path, run_name, scope))
