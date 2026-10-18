"""Bounded existing-server rollout collection followed by one native TRL update."""

import argparse
import asyncio
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import aiohttp


def event(directory: Path, phase: str, **details):
    with (directory / "events.jsonl").open("a") as stream:
        stream.write(
            json.dumps({"phase": phase, "utc": datetime.now(UTC).isoformat(), **details})
            + "\n"
        )


async def stop_group(process_group: int):
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    await asyncio.sleep(5)
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass


async def execute(command: list[str], log: Path, timeout: float):
    with log.open("wb") as stream:
        process = await asyncio.create_subprocess_exec(
            *command, stdout=stream, stderr=asyncio.subprocess.STDOUT, start_new_session=True
        )
        try:
            status = await asyncio.wait_for(process.wait(), timeout)
            if status:
                raise RuntimeError(f"Process exited {status}; see {log.name}")
        finally:
            if process.returncode is None:
                await stop_group(process.pid)
                await process.wait()


async def run(arguments):
    directory = arguments.output_dir
    if (directory / "report.json").exists():
        raise ValueError("This run already has a terminal report")
    if os.getpgid(arguments.inference_process_group) != arguments.inference_process_group:
        raise ValueError("Expected the explicitly supplied existing server's process group")
    event(directory, "driver_started", run_id=arguments.run_id, deadline=arguments.deadline)
    source_root = Path(__file__).resolve().parents[1]
    input_hashes = {
        str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for package in ("kernel_trl", "kernel_skyrl", "kernel_rlm", "kernel_rlm_v1")
        for path in sorted((source_root / package).rglob("*.py"))
    }
    (directory / "inputs.json").write_text(
        json.dumps(
            {
                "run_id": arguments.run_id,
                "experiment_id": "trl-rlm-l40s",
                "backend": "trl",
                "provider": "cloudrift",
                "model_revision": "a397bde04501591d0ac3d74a5d9d0a4ec298ffb8",
                "dataset_sha256": hashlib.sha256(arguments.dataset.read_bytes()).hexdigest(),
                "source_sha256": input_hashes,
            },
            indent=2,
        )
    )
    def remaining():
        return arguments.deadline - time.time()
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            ready = False
            for _attempt in range(90):
                if remaining() < 600:
                    raise TimeoutError(
                        "Insufficient original time budget before inference readiness"
                    )
                try:
                    async with session.get("http://127.0.0.1:8006/health") as response:
                        ready = response.status == 200
                except aiohttp.ClientError:
                    pass
                if ready:
                    break
                await asyncio.sleep(2)
            if not ready:
                raise TimeoutError("Existing inference server did not become ready")
        event(directory, "inference_ready")
        await execute(
            [
                arguments.python,
                "-m",
                "kernel_trl.collect",
                "--checkpoint",
                str(arguments.checkpoint),
                "--dataset",
                str(arguments.dataset),
                "--output-dir",
                str(directory / "rollouts"),
            ],
            directory / "collect.log",
            min(900, remaining() - 420),
        )
        event(directory, "rollouts_admitted")
    finally:
        await stop_group(arguments.inference_process_group)
    event(directory, "inference_stopped")
    await asyncio.sleep(3)
    if remaining() < 180:
        raise TimeoutError("Insufficient original budget to start training")
    training = directory / "update"
    training.mkdir(exist_ok=False)
    telemetry = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=index,timestamp,memory.used,utilization.gpu,power.draw",
            "--format=csv",
            "-l",
            "1",
        ],
        stdout=(training / "telemetry.csv").open("wb"),
        start_new_session=True,
    )
    try:
        event(directory, "training_started")
        await execute(
            [
                arguments.python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                "--nproc_per_node=2",
                "--max_restarts=0",
                "-m",
                "kernel_trl.rlm_update",
                "--checkpoint",
                str(arguments.checkpoint),
                "--group",
                str(directory / "rollouts/training-group.json"),
                "--output-dir",
                str(training),
            ],
            directory / "training.log",
            min(600, remaining() - 15),
        )
    finally:
        await stop_group(telemetry.pid)
        telemetry.wait()
    reports = [json.loads((training / f"report-rank-{rank}.json").read_text()) for rank in range(2)]
    if any(report["global_step"] != 1 for report in reports):
        raise ValueError("No complete distributed update")
    event(directory, "training_completed", optimizer_step=1, rewards=reports[0]["rewards"])
    return {
        "run_id": arguments.run_id,
        "experiment_id": "trl-rlm-l40s",
        "backend": "trl",
        "status": "completed",
        "passed": True,
        "optimizer_steps": 1,
        "quality_improvement_claimed": False,
        "reports": reports,
        "checkpoint_weights_remain_on_vm": True,
        "checkpoint_path": str(training / "checkpoint-step-1"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--inference-process-group", type=int, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    arguments = parser.parse_args()
    try:
        result = asyncio.run(run(arguments))
    except Exception as error:
        result = {
            "run_id": arguments.run_id,
            "experiment_id": "trl-rlm-l40s",
            "backend": "trl",
            "status": "failed",
            "passed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "quality_improvement_claimed": False,
        }
        event(arguments.output_dir, "failed", **result)
        (arguments.output_dir / "report.json").write_text(json.dumps(result, indent=2))
        raise
    (arguments.output_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
