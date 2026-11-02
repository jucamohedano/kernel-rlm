"""Candidate-only compile/forward diagnostics through the isolated ICL worker.

The host creates only the public input contract. It never runs a reference,
loads output arrays, or imports candidate code. Saved worker artifacts remain
private; model feedback is not a numerical or CuTe-execution verdict.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace

from cute_rlm.icl.worker import input_contract
from cute_rlm.verifier import launch_phase

# Same isolated-phase deadline as the independent ICL final validator.
TIMEOUT_S = 600
CANDIDATE_ERRORS = frozenset(
    {"compile_error", "unsupported", "initialization_error", "runtime_error", "compile_timeout"}
)
DIAGNOSTIC_FIELDS = (
    "detail",
    "error_type",
    "traceback",
    "stdout",
    "stderr",
    "launcher_stderr",
    "returncode",
    "worker_returncode",
    "wall_s",
    "diagnostic_limit",
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


class ReplCompileFeedback:
    """Serialize one task's root/child GPU checks; never retry an attempt.

    Infrastructure and persistence failures propagate to stop the experiment.
    Candidate errors remain ordinary diagnostic responses to the model.
    """

    def __init__(self, task: dict, run_dir: Path, *, worker_command: list[str]):
        if (
            not isinstance(worker_command, list)
            or len(worker_command) != 5
            or worker_command[1:4] != ["-m", "cute_rlm.sandbox", "--spec"]
            or any(not isinstance(item, str) or not item for item in worker_command)
        ):
            raise ValueError("explicit cute_rlm.sandbox --spec worker command required")
        if len(task["cases"]) != 4:
            raise ValueError("compile feedback requires all four public task cases")
        self.task = copy.deepcopy(task)
        self.contract = input_contract(self.task)
        self.worker_command = worker_command.copy()
        self.directory = Path(run_dir).resolve() / "compile-feedback"
        self.directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.lock = asyncio.Lock()
        self.ordinal = 0
        self.fatal_error: Exception | None = None

    async def __call__(self, session_id, source) -> dict:
        async with self.lock:
            if self.fatal_error is not None:
                raise RuntimeError(
                    "compile feedback already failed; no retries"
                ) from self.fatal_error
            try:
                return await self.run_attempt(session_id, source)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # The runtime may turn a broker exception into a REPL error.
                # Keep a sticky failure for the experiment driver to inspect.
                self.fatal_error = error
                raise

    async def run_attempt(self, session_id, source) -> dict:
        self.ordinal += 1
        request_id = uuid.uuid4().hex
        directory = self.directory / f"{self.ordinal:06d}-{request_id}"
        directory.mkdir(mode=0o700)
        started = time.monotonic()
        invalid = None
        source_hash = None
        if not isinstance(source, str) or not source.strip():
            invalid = "source must be a nonempty Python source string"
        elif "\0" in source:
            invalid = "source must not contain NUL characters"
        else:
            try:
                source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
            except UnicodeEncodeError:
                invalid = "source must be valid UTF-8 text"
        if not isinstance(session_id, str) or not session_id.strip():
            invalid = "session_id must be a nonempty string"
        metadata = {
            "request_id": request_id,
            "session_id": session_id if isinstance(session_id, str) else repr(session_id),
            "ordinal": self.ordinal,
            "task_id": self.task["id"],
            "source_sha256": source_hash,
        }
        write_json(
            directory / "request.json",
            {**metadata, "source": source if isinstance(source, str) else repr(source)},
        )
        try:
            if invalid is not None:
                result = {"status": "invalid_source", "detail": invalid}
            else:
                (directory / "candidate.py").write_text(source)
                inputs = directory / "candidate-inputs"
                inputs.mkdir(mode=0o700)
                write_json(inputs / "test-input-contract.json", self.contract)
                request = {
                    **metadata,
                    "role": "candidate",
                    "worker_module": "cute_rlm.icl.worker",
                    "icl_task": self.task,
                    "source": source,
                    "compile_feedback": True,
                    "profile_compile": True,
                    "input_dir": str(inputs),
                    "output_dir": str(directory / "output"),
                }
                write_json(directory / "worker-request.json", request)
                try:
                    phase = await launch_phase(
                        self.worker_command, request, SimpleNamespace(timeout_s=TIMEOUT_S)
                    )
                except TimeoutError:
                    # launch_phase kills and reaps its process group before
                    # propagating the phase deadline, unlike infrastructure errors.
                    phase = {
                        "status": "compile_timeout",
                        "error_type": "TimeoutError",
                        "detail": f"candidate compile/forward exceeded {TIMEOUT_S} seconds",
                        "traceback": traceback.format_exc(),
                        "diagnostic_limit": "phase termination can lose partial in-process stdout/stderr",
                    }
                write_json(directory / "worker-response.json", phase)
                status = phase.get("status")
                if status == "executed":
                    if phase.get("source_sha256") != source_hash:
                        raise ValueError("worker source identity mismatch")
                    if phase.get("case_count") != len(self.contract["cases"]):
                        raise ValueError("worker case count mismatch")
                elif status not in CANDIDATE_ERRORS:
                    raise RuntimeError(f"isolated compile worker failed: {status!r}")
                # Keep full worker JSON/artifacts on disk, but expose only
                # diagnostics, which may contain candidate-printed values.
                # Never add host-loaded outputs, provenance, or numeric verdicts.
                result = {key: phase[key] for key in DIAGNOSTIC_FIELDS if key in phase}
                result.update(
                    status="compiled" if status == "executed" else status,
                    compile_forward=status == "executed",
                )
        except asyncio.CancelledError:
            write_json(
                directory / "response.json",
                {
                    **metadata,
                    "status": "cancelled",
                    "elapsed_s": time.monotonic() - started,
                    "correct": None,
                    "cute_verified": None,
                },
            )
            raise
        except Exception as error:
            write_json(
                directory / "response.json",
                {
                    **metadata,
                    "status": "infrastructure",
                    "detail": str(error),
                    "error_type": type(error).__name__,
                    "traceback": traceback.format_exc(),
                    "elapsed_s": time.monotonic() - started,
                    "correct": None,
                    "cute_verified": None,
                },
            )
            raise RuntimeError(
                "compile feedback infrastructure failure; see saved diagnostics"
            ) from error
        result.update(
            **metadata,
            elapsed_s=time.monotonic() - started,
            correct=None,
            cute_verified=None,
        )
        write_json(directory / "response.json", result)
        return result
