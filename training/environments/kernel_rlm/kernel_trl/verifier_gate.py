"""Replay a saved submission with live vLLM sleep/wake before further training."""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import aiohttp
from aiohttp import web
from kernel_rlm.verifier import verify_cuda
from kernel_rollout.spending import phase_deadline, spending_snapshot

from kernel_trl.collect import InferenceClient, inference_admin_app
from kernel_trl.live_curriculum import (
    ensure_gpus_idle,
    inference_environment,
    server_command,
    training_environment,
    wait_for_server,
)
from kernel_trl.live_driver import stop_group


def gpu_memory() -> list[int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    return [int(value.strip()) for value in output.splitlines()]


async def run(arguments):
    ensure_gpus_idle()
    arguments.output_dir.mkdir(parents=True, exist_ok=False)
    budget = json.loads(arguments.budget_file.read_text())
    deadline = phase_deadline(budget, time.time(), 900)
    os.environ.update(training_environment(arguments.python, arguments.source_root, "8.9"))
    os.environ["KERNEL_INFERENCE_ADMIN_URL"] = "http://127.0.0.1:8016"
    record = json.loads((arguments.episode / "submission.json").read_text())
    task = json.loads((arguments.episode / "task.json").read_text())
    report = {
        "status": "running",
        "episode": str(arguments.episode),
        "source_commit": arguments.source_commit,
        "submission_sha256": hashlib.sha256(record["submission"].encode()).hexdigest(),
        "historical_reward_unchanged": True,
        "optimizer_updates": 0,
    }
    runner = None
    server = None
    try:
        with (arguments.output_dir / "inference.log").open("wb") as log:
            server = await asyncio.create_subprocess_exec(
                *server_command(arguments, 1, arguments.resume_from),
                env=inference_environment(arguments.inference_python),
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=630)) as session:
            await wait_for_server(server, session, min(deadline, time.time() + 600))
            runner = web.AppRunner(inference_admin_app(session, "http://127.0.0.1:8006"))
            await runner.setup()
            await web.TCPSite(runner, "127.0.0.1", 8016).start()
            client = InferenceClient(
                session, "http://127.0.0.1:8006", arguments.output_dir / "requests.jsonl"
            )
            report["gpu_memory_before_mib"] = gpu_memory()
            result = await asyncio.to_thread(
                verify_cuda,
                task["context"],
                record["submission"],
                n_trials=1,
                timeout_s=min(240, deadline - time.time()),
                benchmark=True,
            )
            report["verification"] = result
            report["gpu_memory_after_wake_mib"] = gpu_memory()
            response = await client.chat_completion(
                {
                    "json": {
                        "model": "kernel-policy-1",
                        "messages": [
                            {"role": "user", "content": "Write one CUDA indexing expression."}
                        ],
                        "max_tokens": 32,
                        "temperature": 1.0,
                        "logprobs": True,
                        "return_token_ids": True,
                        "return_tokens_as_token_ids": True,
                        "chat_template_kwargs": {"enable_thinking": True},
                    }
                }
            )
            report["inference_after_wake_succeeded"] = bool(response.get("choices"))
            report["status"] = (
                "passed"
                if (
                    result.get("correct") is True
                    and result.get("scheduling", {}).get("mode") == "exclusive"
                    and result.get("timings", {}).get("inference_paused_seconds", 0) > 0
                    and report["inference_after_wake_succeeded"]
                )
                else "failed"
            )
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        if server is not None:
            await stop_group(server.pid)
            await server.wait()
        if runner is not None:
            await runner.cleanup()
        report["spending"] = spending_snapshot(budget, time.time())
        (arguments.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "checkpoint",
        "resume-from",
        "episode",
        "source-root",
        "budget-file",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--inference-python", required=True)
    parser.add_argument("--source-commit", required=True)
    arguments = parser.parse_args()
    result = asyncio.run(asyncio.wait_for(run(arguments), 900))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
