"""Prepare submitted CUDA builds on CPU without grading or taking GPU admission."""

import argparse
import asyncio
import hashlib
import json
import os
import signal
import sys
import time
from contextlib import suppress
from pathlib import Path


class FinalPrecompiler:
    def __init__(self, concurrency: int = 1, timeout_s: float = 120):
        if concurrency not in {1, 2} or timeout_s <= 0:
            raise ValueError("Use one or two CPU builders and a positive deadline")
        self.slots = asyncio.Semaphore(concurrency)
        self.timeout_s = timeout_s
        self.tasks = []

    def submit(self, task: dict, record: dict, directory: Path):
        if record.get("submission", "").strip():
            self.tasks.append(asyncio.create_task(self.prepare(directory)))

    async def prepare(self, directory: Path):
        process = None
        async with self.slots:
            environment = {
                key: value
                for key, value in os.environ.items()
                if not key.endswith(("_API_KEY", "_TOKEN")) and not key.startswith("PRIME_")
            }
            environment.update(CUDA_VISIBLE_DEVICES="", MAX_JOBS="2")
            try:
                with (directory / "precompile.log").open("wb") as stream:
                    process = await asyncio.create_subprocess_exec(
                        sys.executable,
                        "-m",
                        "kernel_trl.precompile",
                        "--episode",
                        str(directory),
                        "--timeout-s",
                        str(self.timeout_s),
                        stdout=stream,
                        stderr=asyncio.subprocess.STDOUT,
                        env=environment,
                        start_new_session=True,
                    )
                    await asyncio.wait_for(process.wait(), self.timeout_s + 15)
                    if process.returncode:
                        raise RuntimeError(f"CPU precompile process exited {process.returncode}")
            except Exception as error:
                (directory / "precompile-error.json").write_text(
                    json.dumps(
                        {
                            "error": f"{type(error).__name__}: {error}",
                            "not_a_reward_or_correctness_result": True,
                        }
                    )
                )
            finally:
                if process is not None and process.returncode is None:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGTERM)
                    try:
                        await asyncio.wait_for(process.wait(), 10)
                    except TimeoutError:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                        await process.wait()

    async def finish(self):
        await asyncio.gather(*self.tasks)

    async def close(self):
        for task in self.tasks:
            if not task.done() and not task.cancelling():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


def cancelled(signum, frame):
    raise SystemExit("CPU precompile cancelled")


def prepare_episode(directory: Path, timeout_s: float):
    from kernel_rlm.verifier import verify_cuda

    task = json.loads((directory / "task.json").read_text())
    record = json.loads((directory / "submission.json").read_text())
    started = time.monotonic()
    result = verify_cuda(
        task["context"],
        record["submission"],
        n_trials=1,
        timeout_s=timeout_s,
        benchmark=False,
        compile_only=True,
    )
    report = {
        "reference_sha256": hashlib.sha256(task["context"].encode()).hexdigest(),
        "submission_sha256": hashlib.sha256(record["submission"].encode()).hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "not_a_reward_or_correctness_result": True,
        "result": result,
    }
    (directory / "precompile.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=120)
    arguments = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU precompilation requires CUDA_VISIBLE_DEVICES empty")
    signal.signal(signal.SIGTERM, cancelled)
    prepare_episode(arguments.episode, arguments.timeout_s)


if __name__ == "__main__":
    main()
