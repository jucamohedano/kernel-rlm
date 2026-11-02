"""CPU-only binding checks for parent-collected REPL24 readiness and stop evidence.

These checks authenticate neither the collector nor the truth of its observations.
The operator must protect receipts/evidence from model writes. Hashes bind supplied
files, not a GPU attestation. Synthetic CPU fixtures are not readiness evidence.
No process, network, CUDA or runtime inspection is performed here.

Readiness receipts contain mode/run_id/plan_sha256/config_sha256,
sandbox_spec_sha256, server_identity, startup_evidence, weights_manifest and checks. Evidence paths
are relative to that receipt's directory. Each check JSON repeats those bindings,
the full server_identity, its check name and passed=true. Startup JSON instead has
observed_server_identity (the full identity without startup_evidence_sha256), so
its hash can be included in server_identity without a circular hash. The protected
weights_manifest records model_id/model_revision/model_path, complete=true, and a
files mapping of every model file's relative path to its observed SHA256, including
config.json, model.safetensors.index.json and all weight shards. download_metadata
is a pinned JSON evidence reference with model_id/model_revision and the same files
mapping established at download. weights_index references the actual safetensors
index JSON; every shard in its weight_map must be in files. The manifest's hash is
also part of server_identity. The collector must establish its metadata against
the pinned downloaded revision; this module checks the supplied JSON, not live
weights or remote metadata.

The runner must additionally compare sandbox_spec_sha256 with its actual launch
spec and await cute_rlm.server_identity.check_server_identity for live /models.
This module does not inspect the host or GPUs.
Stop receipts repeat the generation nonce, model and server identity. Stop JSON
records observed_at, every owned process identity with absent=true, port_closed,
and gpu_memory_bytes keyed by the exact generating GPU UUIDs. Observation times
must be timezone-aware and no earlier than GENERATION_FINISHED.json.finished_at.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from cute_rlm.icl.experiment import ExperimentConfig

MODE = "native-repl-cute-icl"
REQUIRED_CHECKS = (
    "server_context",
    "tokenize",
    "logprobs",
    "reasoning",
    "native_tool_protocol",
    "child_repl_tools",
    "cute_compile_forward_model_loaded",
    "namespace_isolation",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_hash(value: object, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Missing or invalid SHA256: {name}")


def read_evidence(root: Path, reference: dict, name: str) -> dict:
    """Read only a hash-bound JSON object below this supplied receipt directory."""
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        raise ValueError(f"{name} needs a pinned evidence file")
    relative = Path(reference["path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"{name} evidence escapes its directory or is not a file")
    data = path.read_bytes()
    if digest(data) != reference.get("sha256"):
        raise ValueError(f"{name} evidence SHA256 mismatch")
    observed = json.loads(data)
    if not isinstance(observed, dict):
        raise ValueError(f"{name} evidence must be a JSON object")
    return observed


def require_bindings(value: dict, expected: dict, name: str) -> None:
    if not isinstance(value, dict) or any(value.get(key) != item for key, item in expected.items()):
        raise ValueError(f"{name} does not bind this run, configuration or server identity")


def run_bindings(run: Path, config: ExperimentConfig) -> dict:
    config_bytes = (run / "config.json").read_bytes()
    if json.loads(config_bytes) != asdict(config):
        raise ValueError("Configuration differs from frozen config.json")
    if config.requested_precision != "bf16" or config.server_context_tokens != 262144:
        raise ValueError("Readiness requires the full BF16 262144-token deployment")
    return {
        "mode": MODE,
        "run_id": run.name,
        "plan_sha256": digest((run / "plan.json").read_bytes()),
        "config_sha256": digest(config_bytes),
    }


def startup_option(argv: list[str], option: str) -> str | None:
    values = []
    for index, item in enumerate(argv):
        if item == option:
            if index + 1 == len(argv) or argv[index + 1].startswith("--"):
                raise ValueError(f"Missing startup argument value: {option}")
            values.append(argv[index + 1])
        elif item.startswith(option + "="):
            values.append(item.split("=", 1)[1])
    if len(values) > 1:
        raise ValueError(f"Ambiguous startup argument: {option}")
    return values[0] if values else None


def check_identity(identity: dict, config: ExperimentConfig, base_url: str) -> None:
    """Validate recorded startup observations, not merely /models declarations."""
    endpoint = urlsplit(base_url)
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
        raise ValueError("Invalid serving base URL")
    port = endpoint.port or (443 if endpoint.scheme == "https" else 80)
    require_bindings(
        identity,
        {
            "served_model": config.model_id,
            "model_revision": config.model_revision,
            "precision": "bf16",
            "server_context_tokens": 262144,
            "tensor_parallel_size": 2,
            "port": port,
        },
        "Observed server identity",
    )
    if any(
        type(identity.get(name)) is not int
        for name in ("port", "server_context_tokens", "tensor_parallel_size")
    ):
        raise ValueError("Observed server identity context, TP and port must be integers")
    if identity.get("base_url", "").rstrip("/") != base_url.rstrip("/"):
        raise ValueError("Observed server identity belongs to a different endpoint")
    for name in ("model_path", "host_id"):
        if not isinstance(identity.get(name), str) or not identity[name].strip():
            raise ValueError(f"Observed server identity lacks {name}")
    for name in ("model_card_sha256", "startup_evidence_sha256", "weights_manifest_sha256"):
        require_hash(identity.get(name), name)
    if "quantization" not in identity or identity["quantization"] is not None:
        raise ValueError("Observed server must explicitly be unquantized BF16")
    gpus = identity.get("gpu_uuids")
    if (
        not isinstance(gpus, list)
        or len(gpus) != 2
        or any(
            not isinstance(gpu, str)
            or re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", gpu)
            is None
            for gpu in gpus
        )
        or len(set(gpus)) != 2
    ):
        raise ValueError("Observed server needs two distinct GPU UUIDs")
    processes = identity.get("owned_processes")
    if not isinstance(processes, list) or not processes:
        raise ValueError("Observed server needs all owned API and engine process identities")
    pids, roles = [], []
    for process in processes:
        if (
            not isinstance(process, dict)
            or type(process.get("pid")) is not int
            or process["pid"] <= 0
            or not isinstance(process.get("start_time"), str)
            or not process["start_time"].strip()
            or process.get("role") not in {"api", "engine", "worker"}
        ):
            raise ValueError("Owned process needs PID, start_time and API/engine role")
        pids.append(process["pid"])
        roles.append(process["role"])
    if (
        len(set(pids)) != len(pids)
        or roles.count("api") != 1
        or not {"engine", "worker"} & set(roles)
    ):
        raise ValueError("Owned process identities need one API owner and engine workers")
    argv = identity.get("startup_args")
    if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) for arg in argv):
        raise ValueError("Observed server lacks actual startup argv")
    model_arg = startup_option(argv, "--model")
    positional = [argv[index + 1] for index, item in enumerate(argv[:-1]) if item == "serve"]
    if model_arg is None:
        if positional != [identity["model_path"]]:
            raise ValueError("Observed startup model path differs from pinned serving identity")
    elif model_arg != identity["model_path"] or positional:
        raise ValueError("Observed startup model path differs from pinned serving identity")
    revision_arg = startup_option(argv, "--revision")
    if revision_arg is not None and revision_arg != config.model_revision:
        raise ValueError("Observed startup revision differs from pinned serving identity")
    options = {
        "--served-model-name": config.model_id,
        "--max-model-len": "262144",
        "--tensor-parallel-size": "2",
        "--port": str(port),
    }
    if any(startup_option(argv, key) != value for key, value in options.items()):
        raise ValueError("Observed startup arguments differ from pinned serving identity")
    if startup_option(argv, "--dtype") not in {"bf16", "bfloat16"}:
        raise ValueError("Observed startup precision must be BF16")
    if startup_option(argv, "--quantization") not in {None, "None", "none"}:
        raise ValueError("Quantized startup is not the pinned BF16 deployment")


def check_readiness(
    run: Path, plan: dict, config: ExperimentConfig, receipt_path: Path, base_url: str
) -> dict:
    """Validate this receipt's frozen bindings and supplied startup/check files."""
    run, receipt_path = run.resolve(), receipt_path.resolve(strict=True)
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    bindings = run_bindings(run, config)
    if json.loads((run / "plan.json").read_bytes()) != plan or plan.get("run_id") != run.name:
        raise ValueError("Readiness plan differs from this run's frozen plan")
    require_bindings(receipt, bindings, "Readiness receipt")
    require_hash(receipt.get("sandbox_spec_sha256"), "sandbox_spec_sha256")
    bindings["sandbox_spec_sha256"] = receipt["sandbox_spec_sha256"]
    identity = receipt.get("server_identity")
    check_identity(identity, config, base_url)
    weights = read_evidence(
        receipt_path.parent, receipt.get("weights_manifest"), "Weights manifest"
    )
    if receipt["weights_manifest"]["sha256"] != identity["weights_manifest_sha256"]:
        raise ValueError("Weights manifest differs from server identity hash")
    require_bindings(
        weights,
        {
            **bindings,
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "model_path": identity["model_path"],
            "complete": True,
        },
        "Weights manifest",
    )
    files = weights.get("files")
    if (
        weights["complete"] is not True
        or not isinstance(files, dict)
        or not {"config.json", "model.safetensors.index.json"} <= files.keys()
        or not any(name.endswith(".safetensors") for name in files)
    ):
        raise ValueError("Weights manifest needs the complete model config and weight inventory")
    for name, file_hash in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Weights manifest file path escapes the model directory")
        require_hash(file_hash, f"weights manifest {name}")
    metadata = read_evidence(
        receipt_path.parent, weights.get("download_metadata"), "Pinned download metadata"
    )
    require_bindings(
        metadata,
        {"model_id": config.model_id, "model_revision": config.model_revision, "files": files},
        "Pinned download metadata",
    )
    index = read_evidence(receipt_path.parent, weights.get("weights_index"), "Weights index")
    if weights["weights_index"]["sha256"] != files["model.safetensors.index.json"]:
        raise ValueError("Weights index differs from manifest hash")
    weight_map = index.get("weight_map")
    if (
        not isinstance(weight_map, dict)
        or not weight_map
        or any(not isinstance(name, str) for name in weight_map.values())
        or set(weight_map.values()) != {name for name in files if name.endswith(".safetensors")}
    ):
        raise ValueError("Weights manifest must include every indexed weight shard")
    startup = read_evidence(receipt_path.parent, receipt.get("startup_evidence"), "Startup")
    if receipt["startup_evidence"]["sha256"] != identity["startup_evidence_sha256"]:
        raise ValueError("Startup evidence differs from server identity hash")
    require_bindings(
        startup,
        {
            **bindings,
            "observed_server_identity": {
                key: value for key, value in identity.items() if key != "startup_evidence_sha256"
            },
        },
        "Startup observation",
    )
    checks = receipt.get("checks")
    if not isinstance(checks, dict) or not set(REQUIRED_CHECKS) <= checks.keys():
        raise ValueError("Readiness receipt lacks required passed checks")
    for name in REQUIRED_CHECKS:
        check = checks[name]
        if not isinstance(check, dict) or check.get("passed") is not True:
            raise ValueError(f"Readiness check did not pass: {name}")
        observed = read_evidence(receipt_path.parent, check.get("evidence"), name)
        require_bindings(
            observed,
            {**bindings, "server_identity": identity, "check": name, "passed": True},
            f"Readiness {name} observation",
        )
        if observed["passed"] is not True:
            raise ValueError(f"Readiness observation did not pass: {name}")
    return {**receipt, "receipt_path": str(receipt_path), "receipt_sha256": digest(receipt_bytes)}


def observation_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Observation timestamp must include a timezone")
    return parsed


def check_model_stopped(
    run: Path, config: ExperimentConfig, started: dict, stop_receipt_path: Path
) -> dict:
    """Bind shutdown observations to the exact generation server and finish time."""
    run, receipt_path = run.resolve(), stop_receipt_path.resolve(strict=True)
    bindings = run_bindings(run, config)
    require_bindings(
        started,
        {
            "run": str(run),
            "run_id": run.name,
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "config_sha256": bindings["config_sha256"],
        },
        "Generation marker",
    )
    identity = started.get("server_identity")
    check_identity(identity, config, started["base_url"])
    if not isinstance(started.get("nonce"), str) or not started["nonce"]:
        raise ValueError("Generation marker needs a nonce")
    stop_bytes = receipt_path.read_bytes()
    stop = json.loads(stop_bytes)
    stop_bindings = {
        **bindings,
        "generation_nonce": started["nonce"],
        "server_identity": identity,
    }
    require_bindings(
        stop,
        {
            **stop_bindings,
            "run": str(run),
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "model_stopped": True,
        },
        "Model-stop receipt",
    )
    if stop["model_stopped"] is not True:
        raise ValueError("Model-stop acknowledgement must be true")
    observed = read_evidence(receipt_path.parent, stop.get("evidence"), "Model-stop")
    require_bindings(observed, stop_bindings, "Model-stop observation")
    expected_absent = [{**process, "absent": True} for process in identity["owned_processes"]]
    if observed.get("pids_absent") != expected_absent or any(
        process.get("absent") is not True for process in observed.get("pids_absent", [])
    ):
        raise ValueError("Model-stop evidence must mark every exact owned PID/start_time absent")
    if observed.get("port_closed") is not True:
        raise ValueError("Model-stop evidence must observe that exact serving port closed")
    memory = observed.get("gpu_memory_bytes")
    if (
        not isinstance(memory, dict)
        or set(memory) != set(identity["gpu_uuids"])
        or any(type(value) is not int or value != 0 for value in memory.values())
    ):
        raise ValueError("Model-stop evidence needs zero memory on each exact generating GPU")
    finished = json.loads((run / "GENERATION_FINISHED.json").read_bytes())
    if observation_time(observed["observed_at"]) < observation_time(finished["finished_at"]):
        raise ValueError("Model-stop observation predates generation finish")
    return {**stop, "receipt_path": str(receipt_path), "receipt_sha256": digest(stop_bytes)}
