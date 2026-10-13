"""Bound verifier concurrency within one Linux container."""

import fcntl
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOCK_PATH = Path("/tmp/kernel-rlm-verifier.lock")


@contextmanager
def verifier_slot(timeout_s: float, *, path: Path = LOCK_PATH) -> Iterator[float]:
    deadline = time.monotonic() + timeout_s
    with path.open("a") as lock:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for the verifier GPU slot")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(min(0.05, remaining))
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for the verifier GPU slot")
            yield remaining
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
