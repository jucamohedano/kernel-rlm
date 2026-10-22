"""Bounded fixed-policy timing diagnostic; never invokes an optimizer."""

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import signal
import statistics
import time
from pathlib import Path

import aiohttp
from kernel_rollout.spending import phase_deadline, spending_snapshot

from kernel_trl.checkpoint_state import checkpoint_metadata
from kernel_trl.inference_lifecycle import WarmInference
from kernel_trl.live_curriculum import (
    ensure_gpus_idle,
    source_hashes,
    start_telemetry,
    stop_telemetry,
    training_environment,
    write_json,
)
from kernel_trl.live_driver import execute
from kernel_trl.precompile import FinalPrecompiler, cancelled

ORDER = ("serial", "overlap", "overlap", "serial")
PROMPT = "Explain bounds checking and thread-to-element indexing for a CUDA vector operation, with concrete examples."


async def model_request(arguments, version: int, directory: Path, tokens: int) -> dict:
    body = {
        "model": f"kernel-policy-{version}",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": tokens,
        "min_tokens": tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    started = time.monotonic()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        async with session.post("http://127.0.0.1:8006/v1/chat/completions", json=body) as response:
            payload = await response.json()
            if response.status != 200:
                raise RuntimeError(f"Fixed-policy inference failed: {response.status}: {payload}")
    if payload.get("usage", {}).get("completion_tokens") != tokens:
        raise ValueError("Diagnostic did not receive its fixed completion-token workload")
    write_json(directory / "request.json", body)
    write_json(directory / "response.json", payload)
    return {"elapsed_seconds": time.monotonic() - started, "completion_tokens": tokens}


async def run(arguments):
    arguments.output_dir.mkdir(parents=True, exist_ok=False)
    lock = Path("/tmp/kernel-rlm-scheduling-probe.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ensure_gpus_idle()
    budget = json.loads(arguments.budget_file.read_text())
    deadline = phase_deadline(budget, time.time(), 1800)
    state = json.loads((arguments.parent_run / "state.json").read_text())
    if state["status"] != "paused" or state["policy_version"] != 8:
        raise ValueError("Diagnostic requires the preserved paused policy-8 checkpoint")
    checkpoint = Path(state["checkpoint"])
    metadata = checkpoint_metadata(checkpoint, 8)
    episode = (
        arguments.parent_run
        / "groups/step-008-lvl1-seed42-0035-attempt-01/collection/rollouts/group-0-trajectory-4"
    )
    task = json.loads((episode / "task.json").read_text())
    record = json.loads((episode / "submission.json").read_text())
    if not json.loads((episode / "score.json").read_text())["verification"]["correct"]:
        raise ValueError("The fixed candidate must have a preserved successful verification")
    os.environ.update(training_environment(arguments.python, arguments.source_root, "8.9"))
    os.environ.pop("KERNEL_INFERENCE_ADMIN_URL", None)
    os.environ["MAX_JOBS"] = "2"
    recipe = {
        "kind": "infrastructure timing diagnostic, not a training experiment",
        "source_commit": arguments.source_commit,
        "source_sha256": source_hashes(arguments.source_root),
        "policy_version": 8,
        "checkpoint_metadata": metadata,
        "parent_run": str(arguments.parent_run),
        "reference_sha256": hashlib.sha256(task["context"].encode()).hexdigest(),
        "candidate_sha256": hashlib.sha256(record["submission"].encode()).hexdigest(),
        "order": ORDER,
        "tokens_per_timed_request": arguments.tokens,
        "cold_build_cache_per_trial": True,
        "optimizer_updates": 0,
        "deadline_unix_seconds": deadline,
        "budget_at_start": spending_snapshot(budget, time.time()),
    }
    write_json(arguments.output_dir / "recipe.json", recipe)
    lease = WarmInference(arguments, budget)
    telemetry = None
    builder = None
    started = time.monotonic()
    report = {"status": "running", "recipe": recipe, "trials": [], "optimizer_updates": 0}
    write_json(arguments.output_dir / "report.json", report)
    try:
        telemetry = await start_telemetry(arguments.output_dir / "gpu-telemetry.csv")
        await lease.ensure(8, checkpoint, arguments.output_dir)
        cold = next(
            event["elapsed_seconds"] for event in lease.events if event["operation"] == "start"
        )
        for name in ("warmup", "warmup-after-wake"):
            directory = arguments.output_dir / name
            directory.mkdir()
            await model_request(arguments, 8, directory, 64)
            if name == "warmup":
                await lease.sleep()
                await lease.ensure(8, checkpoint, arguments.output_dir)
        for index, mode in enumerate(ORDER):
            if time.time() + 690 > deadline:
                report["status"] = "partial_time_gate"
                break
            phase_deadline(budget, time.time(), 660)
            directory = arguments.output_dir / f"trial-{index}-{mode}"
            directory.mkdir()
            write_json(directory / "task.json", task)
            write_json(directory / "submission.json", record)
            os.environ["TORCH_EXTENSIONS_DIR"] = str(directory / "cold-extension-cache")
            trial_started = time.monotonic()
            builder = FinalPrecompiler()
            if mode == "overlap":
                builder.submit(task, record, directory)
            request = await model_request(arguments, 8, directory, arguments.tokens)
            wait_started = time.monotonic()
            await builder.finish()
            build_tail = time.monotonic() - wait_started
            precompile = None
            if mode == "overlap":
                precompile = json.loads((directory / "precompile.json").read_text())
                if precompile["result"].get("verification_status") != "compile_completed":
                    raise ValueError("CPU precompilation did not complete successfully")
            await lease.sleep()
            verify_started = time.monotonic()
            await execute(
                [
                    arguments.python,
                    "-m",
                    "kernel_trl.scheduling_probe",
                    "--verify-episode",
                    str(directory),
                ],
                directory / "verify.log",
                min(270, deadline - time.time() - 30),
            )
            verification = json.loads((directory / "verify.json").read_text())
            if not verification.get("correct") or not verification.get("extension_used"):
                raise ValueError("Fixed candidate failed independent final CUDA verification")
            verify_elapsed = time.monotonic() - verify_started
            await lease.ensure(8, checkpoint, arguments.output_dir)
            trial = {
                "mode": mode,
                "elapsed_seconds": time.monotonic() - trial_started,
                "model_request": request,
                "precompile_tail_wait_seconds": build_tail,
                "precompile": precompile,
                "independent_verifier_seconds": verify_elapsed,
                "verification": verification,
                "server_pid": lease.process.pid,
            }
            report["trials"].append(trial)
            write_json(arguments.output_dir / "report.json", report)
            await builder.close()
            builder = None
        else:
            report["status"] = "completed"
        wake_times = [
            event["elapsed_seconds"] for event in lease.events if event["operation"] == "wake_up"
        ]
        report["startup"] = {
            "cold_seconds": cold,
            "wake_seconds": wake_times,
            "median_wake_seconds": statistics.median(wake_times),
        }
        serial = [
            trial["elapsed_seconds"] for trial in report["trials"] if trial["mode"] == "serial"
        ]
        overlap = [
            trial["elapsed_seconds"] for trial in report["trials"] if trial["mode"] == "overlap"
        ]
        if serial and overlap:
            report["pipeline_speedup"] = statistics.mean(serial) / statistics.mean(overlap)
            report["pipeline_time_reduction_fraction"] = 1 - statistics.mean(
                overlap
            ) / statistics.mean(serial)
        report["quality_improvement_claimed"] = False
        report["full_training_speedup_measured"] = False
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if builder is not None:
            await builder.close()
        await lease.close()
        if telemetry is not None:
            await stop_telemetry(*telemetry)
        report["elapsed_seconds"] = time.monotonic() - started
        report["budget_at_finish"] = spending_snapshot(budget, time.time())
        write_json(arguments.output_dir / "report.json", report)
        lock.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-episode", type=Path)
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--budget-file", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--inference-python")
    parser.add_argument("--python", default=os.sys.executable)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--context-tokens", type=int, default=14336)
    arguments = parser.parse_args()
    signal.signal(signal.SIGTERM, cancelled)
    if arguments.verify_episode is not None:
        from kernel_rlm.verifier import verify_cuda

        directory = arguments.verify_episode
        task = json.loads((directory / "task.json").read_text())
        record = json.loads((directory / "submission.json").read_text())
        result = verify_cuda(task["context"], record["submission"], timeout_s=240, benchmark=True)
        write_json(directory / "verify.json", result)
    else:
        asyncio.run(asyncio.wait_for(run(arguments), 1800))


if __name__ == "__main__":
    main()
