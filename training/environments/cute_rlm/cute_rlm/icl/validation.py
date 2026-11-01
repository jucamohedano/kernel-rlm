"""Offline correctness comparison, with no candidate code executed on the host."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import traceback
from pathlib import Path
from types import SimpleNamespace

from cute_rlm.icl.worker import input_contract
from cute_rlm.outputs import compare_trees, load_tree
from cute_rlm.verifier import launch_phase

TIMEOUT_S = 600


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _compile_observation(directory):
    path = directory / "compile-provenance.json"
    if not path.is_file() or path.is_symlink():
        return {"observed": False, "reason": "no compile provenance"}
    provenance = json.loads(path.read_text())
    artifacts = {}
    for artifact in provenance.get("artifacts", []):
        valid = False
        for kind in ("ptx", "cubin"):
            name = artifact.get(f"{kind}_path")
            if name is None:
                continue
            artifact_path = directory / name
            if (
                artifact_path.parent != directory / "compile-artifacts"
                or artifact_path.is_symlink()
                or not artifact_path.is_file()
            ):
                raise ValueError("invalid compile artifact path")
            digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if digest != artifact.get(f"{kind}_sha256"):
                raise ValueError("compile artifact hash mismatch")
            valid = True
        if valid:
            artifacts[artifact["id"]] = artifact
    completed = [
        invocation
        for invocation in provenance.get("invocations", [])
        if invocation.get("status") == "completed" and invocation.get("artifact_id") in artifacts
    ]
    return {
        "observed": bool(completed),
        "artifacts": list(artifacts.values()),
        "completed_invocations": completed,
        "provenance": provenance,
        "scope": "candidate-local observational evidence; not attestation or computation coverage",
    }


async def validate_candidate(task, source, directory, *, worker_command: list[str]):
    """Execute one reference and one candidate phase; never repair or retry a source.

    ``worker_command`` must be the explicit bubblewrap supervisor module command.
    Numerical agreement and observed compile invocation remain separate from review.
    """
    if (
        not isinstance(worker_command, list)
        or len(worker_command) != 5
        or worker_command[1:4] != ["-m", "cute_rlm.sandbox", "--spec"]
    ):
        raise ValueError("explicit cute_rlm.sandbox --spec worker command required")
    if not source.strip():
        raise ValueError("candidate source is empty")
    from cute_rlm.icl.tasks import reference_source

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "candidate.py").write_text(source)
    reference_code = reference_source(task)
    (directory / "reference.py").write_text(reference_code)
    result = {
        "status": "infrastructure",
        "numerical_correct": None,
        "compile_forward": False,
        "cute_execution_observed": False,
        "valid_solution": None,
        "source_review_required": True,
        "review_requirement": "check required computation follows a real CuTe path; dummy kernels do not qualify",
        "timing_performed": False,
        "evidence_scope": "local observational correctness, not a malicious-code attestation; trusted oracle module is available in runtime and source review is mandatory",
        "limitations": [
            "launcher timeout or external termination can lose partial in-process stdout/stderr",
            "input mutation and required-computation coverage require source review",
        ],
    }
    common = {"icl_task": task, "worker_module": "cute_rlm.icl.worker", "compile_feedback": True}
    config = SimpleNamespace(timeout_s=TIMEOUT_S)
    try:
        for role, code in (("reference", reference_code), ("candidate", source)):
            request = {
                **common,
                "role": role,
                "source": code,
                "source_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "output_dir": str(directory / role),
                "profile_compile": role == "candidate",
            }
            if role == "candidate":
                inputs = directory / "candidate-inputs"
                inputs.mkdir()
                # Never expose oracle outputs to the candidate mount.
                shutil.copyfile(
                    directory / "reference/test-input-contract.json",
                    inputs / "test-input-contract.json",
                )
                request["input_dir"] = str(inputs)
            _write(directory / f"{role}-request.json", request)
            phase = await launch_phase(worker_command, request, config)
            result[role] = phase
            _write(directory / f"{role}-result.json", phase)
            result["stage"] = role
            if phase.get("status") != "executed":
                result["status"] = phase.get("status", "infrastructure")
                return result
            if phase.get("source_sha256") != request["source_sha256"]:
                raise ValueError("worker source identity mismatch")
            if phase.get("case_count") != len(task["cases"]):
                raise ValueError("worker case count mismatch")
            if role == "reference":
                contract = json.loads(
                    (directory / "reference/test-input-contract.json").read_text()
                )
                if contract != input_contract(task):
                    raise ValueError("reference input contract mismatch")
        result["compile_forward"] = True
        comparisons = []
        for i in range(len(task["cases"])):
            comparisons.append(
                compare_trees(
                    load_tree(directory / f"reference/expected-{i}.npz"),
                    load_tree(directory / f"candidate/outputs-{i}.npz"),
                    atol=task["atol"],
                    rtol=task["rtol"],
                )
            )
        result["cases"] = comparisons
        result["numerical_correct"] = bool(comparisons) and all(c["correct"] for c in comparisons)
        observation = _compile_observation(directory / "candidate")
        result["compile_observation"] = observation
        result["cute_execution_observed"] = observation["observed"]
        result["status"] = (
            "review_required" if result["numerical_correct"] else "numerical_mismatch"
        )
        return result
    except asyncio.CancelledError:
        result.update(
            status="cancelled", detail="validation cancelled; launcher terminates phase process"
        )
        raise
    except Exception as error:
        result.update(
            status="infrastructure",
            detail=str(error),
            error_type=type(error).__name__,
            traceback=traceback.format_exc(),
        )
        return result
    finally:
        _write(directory / "result.json", result)
