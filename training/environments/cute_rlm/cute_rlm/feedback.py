"""Task-scoped compile/one-forward feedback through the isolated GPU launcher.

This is a diagnostic service, not a correctness or CuTe provenance verifier.
No generated source is imported or executed by the host.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path

from cute_rlm.contracts import VerificationConfig
from cute_rlm.verifier import checked_output, configuration_hash, launch_phase


class CompileFeedback:
    def __init__(self, task, config, run_dir, *, worker_command, capture_compiler_artifacts=False):
        self.config = VerificationConfig(**config) if isinstance(config, dict) else config
        if not isinstance(self.config, VerificationConfig):
            raise TypeError("config must be VerificationConfig or its complete mapping")
        if not isinstance(worker_command, (list, tuple)) or not worker_command:
            raise ValueError("An explicit isolated worker command is required")
        source = task["reference_code"]
        self.reference_hash = hashlib.sha256(source.encode()).hexdigest()
        if self.reference_hash != task["source"]["sha256"]:
            raise ValueError("reference source hash mismatch")
        self.task = task
        self.worker_command = list(worker_command)
        self.capture_compiler_artifacts = capture_compiler_artifacts
        self.config_hash = configuration_hash(self.config)
        self.directory = Path(run_dir).resolve() / "compile-feedback"
        self.directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.lock = asyncio.Lock()
        self.ordinal = 0
        self.preparation = None
        self.input_directory = None

    def save(self, path, value):
        path.write_text(json.dumps(value, indent=2, allow_nan=False))

    async def prepare_once(self):
        """Called under the task lock; retain failures as well as successes."""
        if self.preparation is not None:
            return self.preparation
        request_id = uuid.uuid4().hex
        directory = self.directory / f"reference-{request_id}"
        directory.mkdir(mode=0o700)
        request = {
            "request_id": request_id,
            "task_id": self.task["task_id"],
            "role": "reference",
            "source": self.task["reference_code"],
            "source_sha256": self.reference_hash,
            "config": self.config.payload(),
            "config_sha256": self.config_hash,
            "prepare_only": True,
            "output_dir": str(directory / "output"),
        }
        self.save(directory / "request.json", request)
        started = time.monotonic()
        try:
            result = await launch_phase(self.worker_command, request, self.config)
            self.save(directory / "worker-response.json", result)
            if result.get("status") == "prepared":
                if result.get("source_sha256") != self.reference_hash:
                    raise ValueError("prepared reference source identity mismatch")
                # The candidate mount contains only initial state and example
                # arguments. Reference outputs and source are never mounted.
                inputs = directory / "candidate-inputs"
                inputs.mkdir(mode=0o700)
                for name in (
                    "initial_state.npz",
                    "init_args.npz",
                    *(f"inputs-{i}.npz" for i in range(self.config.correctness_runs)),
                ):
                    shutil.copyfile(checked_output(directory / "output" / name), inputs / name)
                if "training" not in result or "module_training" not in result:
                    raise ValueError("prepared reference omitted module modes")
                # Publish only a complete, validated bundle. A cancelled earlier
                # preparation keeps its own evidence and cannot poison retries.
                self.input_directory = inputs
        except asyncio.CancelledError:
            self.save(
                directory / "response.json",
                {
                    "status": "cancelled",
                    "elapsed_s": time.monotonic() - started,
                },
            )
            raise
        except Exception as error:
            result = {"status": "infrastructure", "detail": repr(error)}
        result["elapsed_s"] = time.monotonic() - started
        self.save(directory / "response.json", result)
        self.preparation = result
        return result

    async def __call__(self, session_id, source):
        # Reference preparation and GPU attempts share one task-level lock.
        # There is no feedback-call quota; existing runtime deadlines still apply.
        async with self.lock:
            self.ordinal += 1
            request_id = uuid.uuid4().hex
            directory = self.directory / f"{self.ordinal:06d}-{request_id}"
            directory.mkdir(mode=0o700)
            started = time.monotonic()
            source_hash = None
            invalid = None
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
                "source_sha256": source_hash,
                "config_sha256": self.config_hash,
            }
            self.save(
                directory / "request.json",
                {
                    **metadata,
                    "source": source if isinstance(source, str) else repr(source),
                },
            )
            try:
                if invalid is not None:
                    result = {"status": "invalid_source", "detail": invalid, "stage": "request"}
                else:
                    reference = await self.prepare_once()
                    if reference.get("status") != "prepared":
                        # Tracebacks, source excerpts, and even error text from
                        # trusted preparation can expose the hidden reference.
                        result = {
                            "status": "infrastructure",
                            "stage": "reference",
                            "detail": "Trusted reference preparation failed; diagnostics saved on host.",
                        }
                    else:
                        request = {
                            **metadata,
                            "task_id": self.task["task_id"],
                            "role": "candidate",
                            "source": source,
                            "candidate_sha256": source_hash,
                            "config": self.config.payload(),
                            "input_dir": str(self.input_directory),
                            "output_dir": str(directory / "output"),
                            "training": reference["training"],
                            "module_training": reference["module_training"],
                            "compile_feedback": True,
                        }
                        if self.capture_compiler_artifacts:
                            request["profile_compile"] = True
                        self.save(directory / "worker-request.json", request)
                        result = await launch_phase(self.worker_command, request, self.config)
                        self.save(directory / "worker-response.json", result)
                        result["stage"] = "candidate"
                        if (
                            result.get("status") == "compiled"
                            and result.get("source_sha256") != source_hash
                        ):
                            result = {
                                "status": "infrastructure",
                                "stage": "candidate",
                                "detail": "worker source identity mismatch",
                            }
            except asyncio.CancelledError:
                self.save(
                    directory / "response.json",
                    {
                        **metadata,
                        "status": "cancelled",
                        "elapsed_s": time.monotonic() - started,
                    },
                )
                raise
            except Exception as error:
                result = {"status": "infrastructure", "detail": repr(error)}
            result.update(metadata)
            result.update(
                {
                    "elapsed_s": time.monotonic() - started,
                    "correct": None,
                    "cute_verified": None,
                }
            )
            self.save(directory / "response.json", result)
            return result
