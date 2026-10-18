"""Bounded build-cache diagnostic: no model, training, or experiment registration."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import modal

from .grpo_live_smoke import live_image

app = modal.App("kernel-rlm-cache-diagnostic")
storage = modal.Volume.from_name("kernel-rlm-diagnostics", create_if_missing=True)


def configure_cache(diagnostic_id: str) -> Path:
    if not diagnostic_id.replace("-", "").isalnum():
        raise ValueError("Invalid diagnostic ID")
    directory = Path("/diagnostics") / diagnostic_id
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(directory / "cache")
    os.environ["MAX_JOBS"] = "2"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    os.environ.pop("KERNEL_INFERENCE_ADMIN_URL", None)
    return directory


def cache_files() -> dict:
    root = Path(os.environ["TORCH_EXTENSIONS_DIR"])
    return {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in root.rglob("*")
        if path.suffix in {".so", ".o"} and path.is_file()
    }


def compile_only(case: dict, label: str, directory: Path) -> dict:
    from kernel_rlm.verifier import _RUNNER, _run_verifier_process, submission_extension_root

    started = time.monotonic()
    before = cache_files()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "runner.py").write_text(_RUNNER)
        (root / "submission.py").write_text(case["submission"])
        process = _run_verifier_process(
            [
                sys.executable,
                str(root / "runner.py"),
                "unused_reference.py",
                str(root / "submission.py"),
                "1",
                "0.001",
                "0.0001",
                "0",
                str(root / "phase.json"),
                "compile",
            ],
            timeout_s=100,
            cwd=None,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "KERNEL_RLM_EXTENSION_BUILD_ROOT": str(
                    submission_extension_root(case["submission"])
                ),
            },
        )
        (directory / f"{label}-{case['name']}.log").write_text(process.stdout + process.stderr)
    if process.returncode:
        raise RuntimeError(process.stderr[-1800:])
    record = json.loads(process.stdout.strip().splitlines()[-1])
    after = cache_files()
    record.update(
        case=case["name"],
        stage=label,
        seconds=time.monotonic() - started,
        artifacts_before=before,
        artifacts_after=after,
        objects_unchanged=bool(before) and before == after,
    )
    if not record.get("compiled"):
        raise RuntimeError(record.get("error", "Compilation failed"))
    return record


def save_report(directory: Path, name: str, report: dict) -> None:
    (directory / name).write_text(json.dumps(report, indent=2))
    storage.commit()


@app.function(
    image=live_image,
    cpu=8,
    memory=16384,
    timeout=480,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/diagnostics": storage},
)
def cpu_replay(diagnostic_id: str, cases: list[dict]) -> dict:
    from importlib.metadata import version

    directory = configure_cache(diagnostic_id)
    report = {
        "kind": "diagnostic_not_experiment",
        "diagnostic_id": diagnostic_id,
        "model_calls": 0,
        "optimizer_updates": 0,
        "gpu_count": 0,
        "torch_version": version("torch"),
        "records": [],
    }
    started = time.monotonic()
    try:
        for case in cases:
            for label in ("cold", "warm"):
                record = compile_only(case, label, directory)
                report["records"].append(record)
                save_report(directory, "cpu.json", report)
                print(
                    json.dumps(
                        {
                            "stage": label,
                            "case": case["name"],
                            "seconds": record["seconds"],
                            "objects_unchanged": record["objects_unchanged"],
                        }
                    ),
                    flush=True,
                )
        modified = {**cases[0], "submission": cases[0]["submission"] + "\n"}
        report["records"].append(compile_only(modified, "wrapper-newline", directory))
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save_report(directory, "cpu.json", report)
    return report


@app.function(
    image=live_image,
    gpu="A100-80GB",
    cpu=8,
    memory=16384,
    timeout=180,
    startup_timeout=120,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/diagnostics": storage},
)
def gpu_replay(diagnostic_id: str, cases: list[dict]) -> dict:
    import torch
    from kernel_rlm.verifier import verify_cuda

    directory = configure_cache(diagnostic_id)
    report = {
        "kind": "diagnostic_not_experiment",
        "diagnostic_id": diagnostic_id,
        "gpu": torch.cuda.get_device_name(0),
        "model_calls": 0,
        "optimizer_updates": 0,
        "records": [],
    }
    started = time.monotonic()
    try:
        for case in cases:
            for benchmark in (False, True):
                remaining = 150 - (time.monotonic() - started)
                if remaining < 15:
                    raise TimeoutError("Diagnostic work budget exhausted")
                before = cache_files()
                stage_started = time.monotonic()
                result = verify_cuda(
                    case["reference"],
                    case["submission"],
                    n_trials=3 if benchmark else 1,
                    timeout_s=min(45, remaining),
                    benchmark=benchmark,
                    collect_phase_events=True,
                )
                record = {
                    "case": case["name"],
                    "stage": "final" if benchmark else "feedback",
                    "seconds": time.monotonic() - stage_started,
                    "result": result,
                    "objects_unchanged": before == cache_files(),
                }
                report["records"].append(record)
                save_report(directory, "gpu.json", report)
                print(
                    json.dumps(
                        {
                            "stage": record["stage"],
                            "case": case["name"],
                            "seconds": record["seconds"],
                            "correct": result.get("correct"),
                        }
                    ),
                    flush=True,
                )
                if result.get("correct") is not True:
                    raise RuntimeError("Saved correct candidate failed diagnostic verification")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save_report(directory, "gpu.json", report)
    return report


@app.local_entrypoint()
def main(source_run: str, output_dir: str, gpu: bool = False) -> None:
    from kernel_rlm.diagnostics import select_cases

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    cases = select_cases(Path(source_run))
    diagnostic_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    metadata = {
        "kind": "diagnostic_not_experiment",
        "diagnostic_id": diagnostic_id,
        "source_run": Path(source_run).name,
        "cases": [{"id": case["name"], "sha256": case["source_sha256"]} for case in cases],
        "limits": {
            "cpu_seconds": 480,
            "gpu_seconds": 180,
            "gpu_count": 1,
            "retries": 0,
            "model_calls": 0,
            "optimizer_updates": 0,
        },
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
    report = cpu_replay.remote(diagnostic_id, cases)
    (directory / "cpu.json").write_text(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise RuntimeError(report.get("error", "CPU diagnostic failed"))
    if gpu:
        report = gpu_replay.remote(diagnostic_id, cases)
        (directory / "gpu.json").write_text(json.dumps(report, indent=2))
        if report["status"] != "passed":
            raise RuntimeError(report.get("error", "GPU diagnostic failed"))
    print(
        json.dumps(
            {
                "diagnostic_id": diagnostic_id,
                "output": str(directory),
                "status": report["status"],
                "kind": "diagnostic_not_experiment",
            }
        )
    )
