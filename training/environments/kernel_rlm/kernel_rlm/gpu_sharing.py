"""Coordinate verification and Prime weight updates in one container."""

import asyncio
import fcntl
import os
import time
import urllib.request
from contextlib import contextmanager
from functools import wraps

from kernel_rlm.verifier_slot import LOCK_PATH


def exclusive_weight_update(function):
    @wraps(function)
    async def wrapped(*args, **kwargs):
        if not os.environ.get("KERNEL_INFERENCE_ADMIN_URL"):
            return await function(*args, **kwargs)
        deadline = time.monotonic() + 240
        with LOCK_PATH.open("a") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Weight update timed out waiting for verifier") from None
                    await asyncio.sleep(0.05)
            try:
                return await function(*args, **kwargs)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    return wrapped


@contextmanager
def paused_inference(timeout_s: float):
    url = os.environ.get("KERNEL_INFERENCE_ADMIN_URL")
    if not url:
        yield timeout_s
        return
    if not url.startswith("http://127.0.0.1:"):
        raise ValueError("Shared-GPU admin endpoint must be local")
    deadline = time.monotonic() + timeout_s

    def post(route):
        request = urllib.request.Request(url.rstrip("/") + route, data=b"", method="POST")
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status != 200:
                raise RuntimeError(f"Inference control failed: {route}")

    try:
        post("/pause")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Inference pause exhausted verification budget")
        yield remaining
    finally:
        post("/resume")
