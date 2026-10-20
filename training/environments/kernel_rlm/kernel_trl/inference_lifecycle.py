"""Reuse a fixed-policy inference process only between learner phases."""

import asyncio
import hashlib
import json
import time
from pathlib import Path

import aiohttp

from kernel_trl.spending import phase_deadline


async def wait_for_gpu_release(timeout_seconds: float = 30, poll_seconds: float = 0.5):
    from kernel_trl.live_curriculum import ensure_gpus_idle

    started = time.monotonic()
    while True:
        try:
            ensure_gpus_idle()
            return time.monotonic() - started
        except RuntimeError:
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise
            await asyncio.sleep(min(poll_seconds, remaining))


class WarmInference:
    def __init__(self, arguments, budget: dict):
        self.arguments = arguments
        self.budget = budget
        self.process = None
        self.identity = None
        self.sleeping = False
        self.stream = None
        self.events = []

    def record(self, operation: str, started: float, **extra):
        event = {
            "operation": operation,
            "elapsed_seconds": time.monotonic() - started,
            "pid": self.process.pid if self.process is not None else None,
            "identity": self.identity,
            **extra,
        }
        self.events.append(event)
        path = self.arguments.output_dir / "inference-lifecycle.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    async def ensure(self, version: int, checkpoint: Path | None, directory: Path):
        from kernel_trl.live_curriculum import (
            inference_environment,
            server_command,
            wait_for_server,
        )

        metadata = checkpoint / "metadata.json" if checkpoint else None
        identity = (
            version,
            str(checkpoint.resolve()) if checkpoint else None,
            hashlib.sha256(metadata.read_bytes()).hexdigest() if metadata else None,
        )
        if self.process is not None and (
            self.process.returncode is not None or self.identity != identity
        ):
            await self.close()
        if self.process is not None:
            if self.sleeping:
                await self.control("wake_up")
                self.sleeping = False
            self.record("reuse", time.monotonic())
            return self.process
        deadline = phase_deadline(self.budget, time.time(), 600)
        started = time.monotonic()
        self.stream = (directory / "inference.log").open("wb")
        self.identity = identity
        try:
            self.process = await asyncio.create_subprocess_exec(
                *server_command(self.arguments, version, checkpoint),
                stdout=self.stream,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
                env=inference_environment(self.arguments.inference_python),
            )
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                await wait_for_server(self.process, session, deadline)
            self.record("start", started)
            return self.process
        except BaseException:
            await self.close()
            raise

    async def control(self, route: str):
        phase_deadline(self.budget, time.time(), 120)
        started = time.monotonic()
        suffix = "/sleep?level=1" if route == "sleep" else "/wake_up"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
            async with session.post("http://127.0.0.1:8006" + suffix) as response:
                response.raise_for_status()
                await response.read()
        self.record(route, started)

    async def sleep(self):
        if self.process is not None and not self.sleeping:
            try:
                await self.control("sleep")
                self.sleeping = True
            except BaseException:
                await self.close()
                raise

    async def close(self):
        from kernel_trl.live_driver import stop_group

        started = time.monotonic()
        try:
            if self.process is not None:
                await stop_group(self.process.pid)
                await self.process.wait()
                released = await wait_for_gpu_release()
                self.record("stop", started, gpu_release_wait_seconds=released)
        finally:
            self.process = None
            self.identity = None
            self.sleeping = False
            if self.stream is not None:
                self.stream.close()
                self.stream = None
