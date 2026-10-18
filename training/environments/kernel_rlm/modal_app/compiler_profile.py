"""CPU-only compiler attribution and contention diagnostic; GPU invocation is absent."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import modal

from .grpo_live_smoke import live_image

app = modal.App("kernel-rlm-cpu-compiler-profile")
storage = modal.Volume.from_name("kernel-rlm-diagnostics", create_if_missing=True)


@app.function(
    image=live_image,
    cpu=(8, 8),
    memory=(16384, 32768),
    timeout=720,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/diagnostics": storage},
)
def cpu_profile(diagnostic_id: str, cases: list[dict]) -> dict:
    import torch
    from kernel_rlm.build_profile import cgroup_sample, compiler_environment, run_batch

    if not diagnostic_id.replace("-", "").isalnum() or len(cases) != 2:
        raise ValueError("A diagnostic ID and exactly two saved cases are required")
    devices = [str(path) for path in Path("/dev").glob("nvidia[0-9]*")]
    if devices or torch.cuda.device_count():
        raise RuntimeError("This diagnostic must run without GPU devices")
    directory = Path("/diagnostics") / diagnostic_id
    directory.mkdir(parents=True, exist_ok=False)
    environment, tools = compiler_environment(directory)
    report = {
        "kind": "maintenance_diagnostic_not_experiment",
        "diagnostic_id": diagnostic_id,
        "gpu_count": 0,
        "gpu_device_nodes": devices,
        "model_calls": 0,
        "optimizer_updates": 0,
        "torch_version": torch.__version__,
        "tools": tools,
        "batches": [],
        "resources": {
            "cpu_request": 8,
            "cpu_limit": 8,
            "memory_request_mib": 16384,
            "memory_limit_mib": 32768,
            "host_logical_cpus": os.cpu_count(),
            "initial_cgroup": cgroup_sample(),
        },
        "caveats": [
            "Two saved cummax candidates, not a representative kernel workload.",
            "Fresh build directories prevent artifact-cache hits; OS caches remain warm.",
            "Per-command wall times overlap; their sum is not batch elapsed time.",
            "Cgroup memory includes page cache, not just compiler heaps.",
            "No GPU execution, correctness, speedup or learning is tested.",
        ],
    }
    started = time.monotonic()
    deadline = started + 600
    try:
        for label, parallel, jobs, selected in (
            ("serial-j2", 1, 2, cases),
            ("parallel-j2", 2, 2, cases),
            ("parallel-j1", 2, 1, cases),
            ("serial-j2-recheck", 1, 2, cases[:1]),
        ):
            batch = run_batch(selected, directory / label, environment, parallel, jobs, deadline)
            report["batches"].append(batch)
            (directory / "report.json").write_text(json.dumps(report, indent=2))
            storage.commit()
            print(
                json.dumps(
                    {
                        "stage": label,
                        "seconds": batch["wall_seconds"],
                        "commands": [build["command_summary"] for build in batch["builds"]],
                    }
                ),
                flush=True,
            )
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (directory / "report.json").write_text(json.dumps(report, indent=2))
        storage.commit()
    return report


@app.local_entrypoint()
def main(source_run: str, output_dir: str) -> None:
    import hashlib
    import subprocess

    from kernel_rlm.diagnostics import select_cases

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    cases = select_cases(Path(source_run))
    diagnostic_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-cpu-" + uuid4().hex[:8]
    source_root = Path(__file__).resolve().parents[1]
    files = [
        Path(__file__),
        source_root / "kernel_rlm/build_profile.py",
        source_root / "kernel_rlm/verifier.py",
    ]
    metadata = {
        "kind": "maintenance_diagnostic_not_experiment",
        "diagnostic_id": diagnostic_id,
        "source_run": Path(source_run).name,
        "gpu_request": None,
        "rlm_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": {
            str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
        "cases": [{"id": case["name"], "sha256": case["source_sha256"]} for case in cases],
        "limits": {
            "function_seconds": 720,
            "work_seconds": 600,
            "retries": 0,
            "gpu_count": 0,
            "model_calls": 0,
            "optimizer_updates": 0,
        },
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
    report = cpu_profile.remote(diagnostic_id, cases)
    (directory / "report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "diagnostic_id": diagnostic_id,
                "status": report["status"],
                "elapsed_seconds": report["elapsed_seconds"],
                "gpu_count": 0,
            }
        )
    )
    if report["status"] != "passed":
        raise RuntimeError(report.get("error", "CPU diagnostic failed"))
