"""CPU-testable cache and readiness helpers for bounded inference probes."""

import asyncio
import hashlib
import json
import time
import urllib.request
from pathlib import Path

STARTUP_TIMEOUT_S = 600
PROBE_TIMEOUT_S = 1200


def cache_environment(root: Path, identity: dict[str, str]) -> dict[str, str]:
    namespace = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    directory = root / namespace
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "identity.json").write_text(json.dumps(identity, indent=2))
    return {"VLLM_CACHE_ROOT": str(directory / "vllm")}


def check_health(timeout: float) -> None:
    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=timeout):
        pass


async def wait_for_inference(server, timeout_s: float = STARTUP_TIMEOUT_S) -> float:
    started = time.monotonic()
    deadline = started + timeout_s
    while True:
        if server.poll() is not None:
            raise RuntimeError("Inference server exited; inspect inference.log")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Inference startup exceeded {timeout_s:g} seconds")
        try:
            await asyncio.to_thread(check_health, min(2, remaining))
            return time.monotonic() - started
        except OSError:
            await asyncio.sleep(min(2, max(0, deadline - time.monotonic())))
