"""Bound verifier concurrency within one Linux container."""

import asyncio
import fcntl
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

LOCK_PATH = Path("/tmp/kernel-rlm-verifier.lock")
COMPILE_LOCK_PATH = Path("/tmp/kernel-rlm-compile.lock")
INFERENCE_ADMISSION_TIMEOUT_S = 300.0


def _intent_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.writer")


def _acquire_sync(lock, operation: int, deadline: float, message: str) -> None:
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(message)
        try:
            fcntl.flock(lock, operation | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            time.sleep(min(0.05, remaining))


async def _acquire_async(lock, operation: int, deadline: float, message: str) -> None:
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(message)
        try:
            fcntl.flock(lock, operation | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            await asyncio.sleep(min(0.05, remaining))


@asynccontextmanager
async def inference_request_slot(
    timeout_s: float, *, path: Path | None = None, enabled: bool = True
) -> AsyncIterator[float]:
    """Admit concurrent model requests while giving verifier writers priority."""
    if not enabled:
        yield 0.0
        return
    path = path or LOCK_PATH
    deadline = time.monotonic() + timeout_s
    intent_acquired = False
    with _intent_path(path).open("a") as intent:
        try:
            await _acquire_async(
                intent,
                fcntl.LOCK_SH,
                deadline,
                "Timed out waiting to enter inference",
            )
            intent_acquired = True
            with path.open("a") as lock:
                await _acquire_async(
                    lock,
                    fcntl.LOCK_SH,
                    deadline,
                    "Timed out waiting to enter inference",
                )
                fcntl.flock(intent, fcntl.LOCK_UN)
                intent_acquired = False
                try:
                    yield time.monotonic() - (deadline - timeout_s)
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
        finally:
            if intent_acquired:
                fcntl.flock(intent, fcntl.LOCK_UN)


@asynccontextmanager
async def exclusive_inference_slot(
    timeout_s: float, *, path: Path | None = None
) -> AsyncIterator[float]:
    """Exclude new inference requests and wait for active requests to finish."""
    path = path or LOCK_PATH
    deadline = time.monotonic() + timeout_s
    intent_acquired = False
    with _intent_path(path).open("a") as intent:
        try:
            await _acquire_async(
                intent,
                fcntl.LOCK_EX,
                deadline,
                "Timed out waiting for the verifier GPU slot",
            )
            intent_acquired = True
            with path.open("a") as lock:
                await _acquire_async(
                    lock,
                    fcntl.LOCK_EX,
                    deadline,
                    "Timed out waiting for the verifier GPU slot",
                )
                try:
                    yield max(0.0, deadline - time.monotonic())
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
        finally:
            if intent_acquired:
                fcntl.flock(intent, fcntl.LOCK_UN)


@contextmanager
def verifier_slot(timeout_s: float, *, path: Path = LOCK_PATH) -> Iterator[float]:
    deadline = time.monotonic() + timeout_s
    intent_path = _intent_path(path)
    with intent_path.open("a") as intent:
        _acquire_sync(
            intent,
            fcntl.LOCK_EX,
            deadline,
            "Timed out waiting for the verifier GPU slot",
        )
        try:
            with path.open("a") as lock:
                _acquire_sync(
                    lock,
                    fcntl.LOCK_EX,
                    deadline,
                    "Timed out waiting for the verifier GPU slot",
                )
                try:
                    yield max(0.0, deadline - time.monotonic())
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
        finally:
            fcntl.flock(intent, fcntl.LOCK_UN)


@contextmanager
def feedback_slot(timeout_s: float, *, path: Path = LOCK_PATH) -> Iterator[float]:
    """Serialize feedback while sharing inference and excluding weight updates."""
    deadline = time.monotonic() + timeout_s
    mutex = path.with_name(f"{path.name}.feedback")
    with compile_slot(timeout_s, path=mutex):
        with _intent_path(path).open("a") as intent:
            _acquire_sync(intent, fcntl.LOCK_SH, deadline, "Feedback admission timed out")
            try:
                with path.open("a") as lock:
                    _acquire_sync(lock, fcntl.LOCK_SH, deadline, "Feedback admission timed out")
                    fcntl.flock(intent, fcntl.LOCK_UN)
                    try:
                        yield max(0.0, deadline - time.monotonic())
                    finally:
                        fcntl.flock(lock, fcntl.LOCK_UN)
            finally:
                fcntl.flock(intent, fcntl.LOCK_UN)


@contextmanager
def compile_slot(timeout_s: float, *, path: Path = COMPILE_LOCK_PATH) -> Iterator[float]:
    """Serialize CPU extension builds without blocking inference requests."""
    deadline = time.monotonic() + timeout_s
    with path.open("a") as lock:
        _acquire_sync(
            lock,
            fcntl.LOCK_EX,
            deadline,
            "Timed out waiting for the CUDA compiler slot",
        )
        try:
            yield max(0.0, deadline - time.monotonic())
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
