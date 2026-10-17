"""Small, offline experiment provenance and launch gates."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def checked_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Asset escapes project root: {relative}")
    if not path.is_file():
        raise ValueError(f"Required asset is missing: {relative}")
    return path


def load_registry(environment: Path) -> dict:
    registry = json.loads((environment / "dashboard/experiments.json").read_text())
    experiments = registry["experiments"]
    identities = [item["id"] for item in experiments]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate experiment IDs")
    datasets = registry["datasets"]
    for experiment in experiments:
        for field in ("id", "intent", "change", "decision", "provenance"):
            if not experiment.get(field):
                raise ValueError(f"Experiment lacks {field}: {experiment.get('id')}")
        provenance = experiment["provenance"]
        if not all(provenance.get(field) for field in ("code_paths", "notes", "dataset_ids")):
            raise ValueError(f"Incomplete experiment provenance: {experiment['id']}")
        for identity in provenance["dataset_ids"]:
            if identity not in datasets:
                raise ValueError(f"Unknown dataset: {identity}")
    return registry


def dataset_record(project: Path, definition: dict) -> dict:
    path = checked_path(project, definition["path"])
    content = path.read_bytes()
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    identities = [row["_task_id"] for row in rows]
    if not identities or len(identities) != len(set(identities)):
        raise ValueError("Dataset must contain unique nonempty task IDs")
    if any(not row.get("code") for row in rows):
        raise ValueError("Dataset contains a task without reference code")
    return {
        **definition,
        "sha256": digest(content),
        "bytes": len(content),
        "task_count": len(rows),
        "task_ids_sha256": digest(canonical(sorted(identities))),
        "identity_scope": "Exact materialized local bytes; upstream revision is not inferred",
    }


def validate_registry(project: Path, environment: Path) -> dict:
    registry = load_registry(environment)
    for experiment in registry["experiments"]:
        provenance = experiment["provenance"]
        for relative in provenance["code_paths"] + provenance["notes"]:
            checked_path(project, relative)
    datasets = {
        identity: dataset_record(project, definition)
        for identity, definition in registry["datasets"].items()
    }
    return {"status": "passed", "experiments": len(registry["experiments"]), "datasets": datasets}


def git_state(repository: Path) -> dict:
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain", "-z"])
    return {"commit": revision, "dirty": bool(status), "worktree_status_sha256": digest(status)}


def event(directory: Path, kind: str, details: dict | None = None) -> None:
    record = {"at_utc": datetime.now(UTC).isoformat(), "event": kind, "details": details or {}}
    with (directory / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def launcher_pins(environment: Path) -> dict:
    source = ast.parse((environment / "modal_app/attention_lora.py").read_text())
    names = {
        "MODEL_ID": "model_id",
        "REVISION": "model_revision",
        "PRIME_REVISION": "prime_revision",
    }
    return {
        names[target.id]: ast.literal_eval(node.value)
        for node in source.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in names
    }


def prepare_run(
    project: Path,
    environment: Path,
    experiment_id: str,
    run_id: str,
    pilot: bool = False,
    resume_run: str = "",
    resume_step: int = 1,
    pins: dict | None = None,
    allow_dirty: bool = False,
    dependency_repositories: dict[str, Path] | None = None,
) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", run_id):
        raise ValueError("Run ID must be a safe, nonempty identifier")
    if resume_step < 1:
        raise ValueError("Resume step must be positive")
    if resume_run and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", resume_run):
        raise ValueError("Parent run ID must be a safe identifier")
    pins = launcher_pins(environment) if pins is None else pins
    if not all(pins.get(key) for key in ("model_id", "model_revision", "prime_revision")):
        raise ValueError("Pinned model and Prime revisions are required")
    validation = validate_registry(project, environment)
    registry = load_registry(environment)
    experiment = next(
        (item for item in registry["experiments"] if item["id"] == experiment_id), None
    )
    if experiment is None:
        raise ValueError("Select a registered experiment ID before launch")
    repositories = {"project": git_state(project), "rlm": git_state(project / "rlm")}
    dependencies = {name: git_state(path) for name, path in (dependency_repositories or {}).items()}
    if "prime" in dependencies and dependencies["prime"]["commit"] != pins["prime_revision"]:
        raise ValueError("Prime source checkout does not match its pinned revision")
    if not allow_dirty and any(state["dirty"] for state in repositories.values()):
        raise ValueError("Commit both repositories before launch, or explicitly use --allow-dirty")
    if any(state["dirty"] for state in dependencies.values()):
        raise ValueError("Commit/pin dependency source changes before launch")
    config = environment / "configs" / ("grpo_pilot.toml" if pilot else "grpo_level1.toml")
    resolved = tomllib.loads(config.read_text())
    if resolved["model"]["name"] != pins["model_id"]:
        raise ValueError("Model pin and selected config disagree")
    sources = resolved["orchestrator"]["train"]["source"]
    if len(sources) != 1:
        raise ValueError("Factory currently supports one registered dataset per run")
    provenance = experiment["provenance"]
    datasets = {
        identity: validation["datasets"][identity] for identity in provenance["dataset_ids"]
    }
    dataset = next(
        (
            item
            for item in datasets.values()
            if item["remote_path"] == sources[0]["env"]["taskset"]["manifest_path"]
        ),
        None,
    )
    if dataset is None or len(datasets) != 1:
        raise ValueError("Training config must reference exactly the registered dataset")
    payload = {}
    code_roots = (
        (project / "rlm/rlm", "/root/rlm"),
        (project / "rlm/training/src/rlm_train", "/root/rlm_train"),
        (environment / "kernel_rlm", "/root/kernel_rlm"),
        (environment / "kernel_rlm_v1", "/root/kernel_rlm_v1"),
        (environment / "kernel_factory", "/root/kernel_factory"),
    )
    for local_root, remote_root in code_roots:
        for path in sorted(local_root.rglob("*.py")):
            if "__pycache__" not in path.parts:
                payload[path] = (
                    "code",
                    remote_root + "/" + path.relative_to(local_root).as_posix(),
                )
    for path in sorted((environment / "modal_app").rglob("*.py")):
        if "__pycache__" not in path.parts:
            payload[path] = ("launcher", None)
    payload[config] = ("config", "/root/configs/" + config.name)
    payload[checked_path(project, dataset["path"])] = ("dataset", dataset["remote_path"])
    for relative in provenance["notes"]:
        payload[checked_path(project, relative)] = ("notes", None)
    payload[environment / "dashboard/experiments.json"] = ("experiment_registry", None)
    contents = {
        path: checked_path(project, path.relative_to(project).as_posix()).read_bytes()
        for path in payload
    }
    inputs = [
        {
            "path": path.relative_to(project).as_posix(),
            "role": payload[path][0],
            "remote_path": payload[path][1],
            "sha256": digest(content),
            "bytes": len(content),
        }
        for path, content in contents.items()
    ]
    if repositories != {"project": git_state(project), "rlm": git_state(project / "rlm")}:
        raise ValueError("Repository changed during input capture; prepare again")
    manifest = {
        "schema_version": 1,
        "capture_kind": "before_launch",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "experiment_id": experiment_id,
        "experiment": experiment,
        "repositories": repositories,
        "dependency_repositories": dependencies,
        "dirty_launch_explicitly_allowed": allow_dirty,
        "pins": pins,
        "datasets": datasets,
        "config": {"path": config.relative_to(project).as_posix(), "values": resolved},
        "inputs": inputs,
        "input_bundle_sha256": digest(canonical(inputs)),
        "parent": {
            "run_id": resume_run,
            "checkpoint_step": resume_step,
            "uri": f"modal://kernel-rlm-traces/outputs/{resume_run}/checkpoints/step_{resume_step}",
        }
        if resume_run
        else None,
        "artifacts": {
            "report": f"modal://kernel-rlm-traces/live-grpo/{run_id}/report.json",
            "metrics": f"modal://kernel-rlm-traces/outputs/{run_id}/metrics.jsonl",
            "trajectories": f"modal://kernel-rlm-traces/outputs/{run_id}/rollouts",
            "checkpoints": f"modal://kernel-rlm-traces/outputs/{run_id}/checkpoints",
            "prompt_scope": "Exact prompt-building source captured; resolved per-call messages are in trajectories",
        },
        "scope": "No credentials/environment dump or checkpoint weights captured. Pin IDs are supplied by the launcher.",
    }
    manifest["manifest_sha256"] = digest(canonical(manifest))
    directory = environment / "logs/live-grpo" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    for path, content in contents.items():
        snapshot = directory / "inputs" / path.relative_to(project)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(content)
    (directory / "provenance.json").write_text(json.dumps(manifest, indent=2))
    event(directory, "prepared", {"manifest_sha256": manifest["manifest_sha256"]})
    return manifest


def launch_bundle(manifest: dict, environment: Path) -> dict:
    directory = environment / "logs/live-grpo" / manifest["run_id"] / "inputs"
    snapshots = {}
    for item in manifest["inputs"]:
        content = checked_path(directory, item["path"]).read_bytes()
        if digest(content) != item["sha256"]:
            raise ValueError(f"Prepared input changed: {item['path']}")
        snapshots[item["path"]] = content.decode("utf-8")
    return {**manifest, "input_snapshots": snapshots}


def start_run(manifest: dict | None, directory: Path, run_id: str) -> None:
    if not manifest or manifest.get("run_id") != run_id:
        raise ValueError("A matching factory manifest is required before running training")
    recorded = {key: value for key, value in manifest.items() if key != "input_snapshots"}
    unsigned = {key: value for key, value in recorded.items() if key != "manifest_sha256"}
    if digest(canonical(unsigned)) != manifest.get("manifest_sha256"):
        raise ValueError("Factory manifest integrity check failed")
    with (directory / "provenance.json").open("x") as stream:
        stream.write(json.dumps(recorded, indent=2))
    try:
        for item in manifest["inputs"]:
            relative = Path(item["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unexpected snapshot path")
            content = manifest["input_snapshots"][item["path"]].encode("utf-8")
            if digest(content) != item["sha256"]:
                raise ValueError(f"Snapshot integrity check failed: {item['path']}")
            snapshot = directory / "inputs" / relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(content)
            remote = item["remote_path"]
            if remote is not None:
                if not remote.startswith("/root/") or ".." in Path(remote).parts:
                    raise ValueError("Unexpected mounted input path")
                if digest(Path(remote).read_bytes()) != item["sha256"]:
                    raise ValueError(
                        f"Mounted input differs from prepared snapshot: {item['path']}"
                    )
        versions = {}
        for package in ("torch", "vllm", "verifiers", "renderers", "prime-rl", "modal"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        runtime = {
            "python": sys.version,
            "packages": versions,
            "input_validation": "passed",
            "validated_at_utc": datetime.now(UTC).isoformat(),
        }
        (directory / "runtime.json").write_text(json.dumps(runtime, indent=2))
        event(directory, "started", {"manifest_sha256": manifest["manifest_sha256"]})
    except Exception as error:
        event(directory, "input_validation_failed", {"error": str(error)})
        raise
