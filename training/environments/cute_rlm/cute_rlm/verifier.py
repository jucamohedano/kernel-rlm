"""Host-side verification orchestration; generated code never runs in this process."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import signal
import time
import uuid
from pathlib import Path

from cute_rlm.contracts import VerificationConfig


def checked_output(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError("verifier output must be a regular file, not a symbolic link")
    return path


def configuration_hash(config):
    return hashlib.sha256(json.dumps(config.payload(), sort_keys=True).encode()).hexdigest()


async def launch_phase(command, request, config):
    """The supplied launcher must enforce the documented filesystem/GPU boundary."""
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": "/nonexistent",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    }
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    started = time.monotonic()
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=environment,
        start_new_session=True,
    )
    try:
        async with asyncio.timeout(config.timeout_s):
            stdout, stderr = await process.communicate(json.dumps(request).encode())
    except BaseException:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await asyncio.wait_for(process.wait(), timeout=5)
        raise
    if process.returncode != 0:
        return {
            "status": "infrastructure",
            "detail": stderr.decode(errors="replace")[-4000:],
            "returncode": process.returncode,
            "wall_s": time.monotonic() - started,
        }
    try:
        result = json.loads(stdout)
        if not isinstance(result, dict):
            raise ValueError("result must be an object")
    except (ValueError, TypeError):
        return {"status": "infrastructure", "detail": "worker returned no valid result object"}
    result["wall_s"] = time.monotonic() - started
    return result


async def verify_submission(
    task, candidate_source, config, run_dir, *, worker_command, cute_observer=None
):
    """Compare separate reference/candidate executions and require external CuTe evidence.

    ``cute_observer(request_id, candidate_sha256, config_sha256)`` may retrieve a
    protected launch observation made by the configured GPU sandbox launcher. This
    callback is not candidate output. Without it, a numerical pass is reported as
    ``launch_unverified`` and no speedup is claimed. A plain subprocess is not a
    sandbox; GPU preflight must validate the supplied launcher's actual isolation.
    """
    from cute_rlm.outputs import compare_trees, load_tree

    config = VerificationConfig(**config) if isinstance(config, dict) else config
    if not isinstance(config, VerificationConfig):
        raise TypeError("config must be VerificationConfig or its complete mapping")
    if not isinstance(worker_command, (list, tuple)) or not worker_command:
        raise ValueError("An explicit isolated worker command is required")
    if not candidate_source.strip():
        raise ValueError("candidate source is empty")
    source_hash = hashlib.sha256(task["reference_code"].encode()).hexdigest()
    if source_hash != task["source"]["sha256"]:
        raise ValueError("reference source hash mismatch")
    directory = Path(run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    candidate_hash = hashlib.sha256(candidate_source.encode()).hexdigest()
    config_hash = configuration_hash(config)
    request_id = uuid.uuid4().hex
    common = {
        "request_id": request_id,
        "task_id": task["task_id"],
        "config": config.payload(),
        "config_sha256": config_hash,
        "candidate_sha256": candidate_hash,
    }
    reference_request = {
        **common,
        "role": "reference",
        "source": task["reference_code"],
        "source_sha256": source_hash,
        "output_dir": str(directory / "reference"),
    }
    (directory / "reference-request.json").write_text(json.dumps(reference_request))
    try:
        reference = await launch_phase(worker_command, reference_request, config)
        if reference.get("status") != "executed":
            return {
                "status": reference.get("status", "infrastructure"),
                "stage": "reference",
                "correct": None,
                "reference": reference,
            }
        if reference.get("source_sha256") != source_hash:
            return {
                "status": "infrastructure",
                "stage": "reference",
                "detail": "worker source identity mismatch",
            }
        # Candidate input mount contains no expected outputs or source oracle module.
        inputs = directory / "candidate-inputs"
        inputs.mkdir(mode=0o700)
        for name in (
            "initial_state.npz",
            "init_args.npz",
            *(f"inputs-{i}.npz" for i in range(config.correctness_runs)),
        ):
            shutil.copyfile(checked_output(directory / "reference" / name), inputs / name)
        candidate_request = {
            **common,
            "role": "candidate",
            "source": candidate_source,
            "source_sha256": candidate_hash,
            "input_dir": str(inputs),
            "training": reference["training"],
            "module_training": reference["module_training"],
            "output_dir": str(directory / "candidate"),
        }
        (directory / "candidate-request.json").write_text(json.dumps(candidate_request))
        candidate = await launch_phase(worker_command, candidate_request, config)
        if candidate.get("status") != "executed":
            return {
                "status": candidate.get("status", "infrastructure"),
                "stage": "candidate",
                "correct": False
                if candidate.get("status") in {"compile_error", "runtime_error"}
                else None,
                "reference": reference,
                "candidate": candidate,
            }
        if candidate.get("source_sha256") != candidate_hash:
            return {
                "status": "infrastructure",
                "stage": "candidate",
                "detail": "worker source identity mismatch",
            }
        comparisons = []
        for trial in range(config.correctness_runs):
            for kind in ("outputs", "mutated-inputs", "post-state"):
                result = compare_trees(
                    load_tree(checked_output(directory / "reference" / f"{kind}-{trial}.npz")),
                    load_tree(checked_output(directory / "candidate" / f"{kind}-{trial}.npz")),
                    atol=config.atol,
                    rtol=config.rtol,
                )
                comparisons.append({"trial": trial, "kind": kind, **result})
        correct = all(check["correct"] for check in comparisons)
        observation = None
        if correct and cute_observer is not None:
            observation = await cute_observer(request_id, candidate_hash, config_hash)
        verified = bool(
            isinstance(observation, dict)
            and observation.get("request_id") == request_id
            and observation.get("candidate_sha256") == candidate_hash
            and observation.get("config_sha256") == config_hash
            and observation.get("compiled") is True
            and observation.get("launched") is True
            and observation.get("evidence")
        )
        speedup = None
        if correct and verified:
            import math

            reference_ms, candidate_ms = reference["median_ms"], candidate["median_ms"]
            if not all(
                isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                for value in (reference_ms, candidate_ms)
            ):
                raise ValueError("worker timing samples must be finite and positive")
            speedup = reference_ms / candidate_ms
        result = {
            "status": "incorrect"
            if not correct
            else "correct"
            if verified
            else "launch_unverified",
            "correct": correct,
            "cute_verified": verified,
            "speedup": speedup,
            "comparisons": comparisons,
            "reference": reference,
            "candidate": candidate,
            "launch_observation": observation,
            "request_id": request_id,
            "candidate_sha256": candidate_hash,
            "config_sha256": config_hash,
        }
    except TimeoutError:
        result = {
            "status": "infrastructure",
            "detail": "verifier worker deadline exceeded",
            "correct": None,
        }
    except (OSError, ValueError, KeyError) as error:
        result = {"status": "infrastructure", "detail": repr(error), "correct": None}
    (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
