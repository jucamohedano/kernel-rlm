"""Bound verifier concurrency within one Linux container."""

import asyncio
import fcntl
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager, nullcontext
from pathlib import Path

LOCK_PATH = Path("/tmp/kernel-rlm-verifier.lock")
COMPILE_LOCK_PATH = Path("/tmp/kernel-rlm-compile.lock")
INFERENCE_ADMISSION_TIMEOUT_S = 300.0


def _intent_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.writer")


def reader_waiting_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.readers")


def try_writer_intent(intent, waiting) -> bool:
    try:
        fcntl.flock(waiting, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    try:
        try:
            fcntl.flock(intent, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False
    finally:
        fcntl.flock(waiting, fcntl.LOCK_UN)


def acquire_writer_sync(intent, path: Path, deadline: float, message: str) -> None:
    with reader_waiting_path(path).open("a") as waiting:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(message)
            if try_writer_intent(intent, waiting):
                return
            time.sleep(min(0.05, remaining))


async def acquire_writer_async(intent, path: Path, deadline: float, message: str) -> None:
    with reader_waiting_path(path).open("a") as waiting:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(message)
            if try_writer_intent(intent, waiting):
                return
            await asyncio.sleep(min(0.05, remaining))


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
    """Admit queued model requests between exclusive verifier phases."""
    if not enabled:
        yield 0.0
        return
    path = path or LOCK_PATH
    deadline = time.monotonic() + timeout_s
    intent_acquired = False
    with reader_waiting_path(path).open("a") as waiting, _intent_path(path).open("a") as intent:
        try:
            await _acquire_async(
                waiting, fcntl.LOCK_SH, deadline, "Timed out waiting to enter inference"
            )
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
                fcntl.flock(waiting, fcntl.LOCK_UN)
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
            await acquire_writer_async(
                intent,
                path,
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
def inference_drain(timeout_s: float, *, path: Path = LOCK_PATH) -> Iterator[float]:
    deadline = time.monotonic() + timeout_s
    with _intent_path(path).open("a") as intent:
        acquire_writer_sync(
            intent,
            path,
            deadline,
            "Timed out waiting for the verifier GPU slot",
        )
        try:
            yield max(0.0, deadline - time.monotonic())
        finally:
            fcntl.flock(intent, fcntl.LOCK_UN)


@contextmanager
def verifier_slot(
    timeout_s: float, *, path: Path = LOCK_PATH, intent_held: bool = False
) -> Iterator[float]:
    deadline = time.monotonic() + timeout_s
    with nullcontext() if intent_held else inference_drain(timeout_s, path=path):
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
