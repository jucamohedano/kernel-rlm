"""Run one-policy-version-at-a-time TRL GRPO over the materialized L1 set."""

import argparse
import asyncio
import copy
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import aiohttp
from kernel_skyrl.contract import has_signal
from kernel_skyrl.preflight import load_tasks

from kernel_trl.checkpoint_state import checkpoint_metadata
from kernel_trl.context import (
    DEFAULT_CONTEXT_TOKENS,
    LEGACY_CONTEXT_TOKENS,
    validate_context_tokens,
)
from kernel_trl.live_driver import execute, stop_group
from kernel_trl.probability_audit import require_probability_agreement
from kernel_trl.spending import phase_deadline, spending_snapshot

MODEL_REVISION = "a397bde04501591d0ac3d74a5d9d0a4ec298ffb8"
DEFAULT_CUDA_ARCH_LIST = "8.9"
GROUP_SIZE = 8
WORLD_SIZE = 2
INFLIGHT = 4
COLLECTION_LIMIT_SECONDS = 2400
PHASE_LIMIT_SECONDS = 600
MODEL_TIMEOUT_SECONDS = 600
INFERENCE_HTTP_TIMEOUT_SECONDS = 630
EPISODE_TIMEOUT_SECONDS = 1500


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def record_event(directory: Path, phase: str, **details) -> None:
    with (directory / "events.jsonl").open("a") as stream:
        stream.write(
            json.dumps({"phase": phase, "utc": datetime.now(UTC).isoformat(), **details}) + "\n"
        )


def source_hashes(source_root: Path) -> dict[str, str]:
    hashes = {}
    for package in ("kernel_trl", "kernel_skyrl", "kernel_rlm", "kernel_rlm_v1"):
        for path in sorted((source_root / package).rglob("*.py")):
            hashes[str(path.relative_to(source_root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    repository_root = source_root.parents[2]
    core_root = repository_root / "rlm"
    if not core_root.is_dir():
        raise ValueError("The source snapshot must include the RLM core package")
    for path in sorted(core_root.rglob("*.py")):
        relative = Path("rlm") / path.relative_to(core_root)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    training_package = repository_root / "training/src/rlm_train"
    if not training_package.is_dir():
        raise ValueError("The source snapshot must include training/src/rlm_train")
    for path in sorted(training_package.rglob("*.py")):
        relative = Path("training/src/rlm_train") / path.relative_to(training_package)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def python_path(source_root: Path, existing: str | None = None) -> str:
    repository_root = source_root.parents[2]
    if not (repository_root / "rlm").is_dir():
        raise ValueError("The source snapshot must include the RLM core package")
    if not (repository_root / "training/src/rlm_train").is_dir():
        raise ValueError("The source snapshot must include training/src/rlm_train")
    paths = [
        str(source_root),
        str(repository_root / "training/src"),
        str(repository_root),
    ]
    if existing:
        paths.append(existing)
    return os.pathsep.join(paths)


def training_environment(
    python: str,
    source_root: Path,
    cuda_arch_list: str,
    environment: dict[str, str] | None = None,
) -> dict[str, str]:
    result = runtime_environment(python, environment)
    result["PYTHONPATH"] = python_path(source_root, result.get("PYTHONPATH"))
    result["TORCH_CUDA_ARCH_LIST"] = cuda_arch_list
    return python_header_environment(python, result)


def curriculum_order(dataset: Path, first_task: str | None = None) -> list[str]:
    task_ids = [task["name"] for task in load_tasks(dataset)]
    if len(task_ids) != 50:
        raise ValueError(f"Expected the complete 50-task L1 curriculum; found {len(task_ids)}")
    if first_task is None:
        return task_ids
    if first_task not in task_ids:
        raise ValueError(f"Bootstrap task {first_task} is not in the L1 dataset")
    index = task_ids.index(first_task)
    return task_ids[index:] + task_ids[:index]


def server_command(arguments, policy_version: int, checkpoint: Path | None) -> list[str]:
    command = [
        arguments.inference_python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(arguments.checkpoint),
        "--host",
        "127.0.0.1",
        "--port",
        "8006",
        "--served-model-name",
        "kernel-base",
        "--tensor-parallel-size",
        "2",
        "--distributed-executor-backend",
        "mp",
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.85",
        "--max-model-len",
        str(validate_context_tokens(getattr(arguments, "context_tokens", LEGACY_CONTEXT_TOKENS))),
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "4096",
        "--enable-lora",
        "--max-loras",
        "1",
        "--max-lora-rank",
        "8",
        "--enforce-eager",
        "--enable-sleep-mode",
        "--disable-custom-all-reduce",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "poolside_v1",
        "--reasoning-parser",
        "poolside_v1",
        "--default-chat-template-kwargs",
        '{"enable_thinking":true}',
        "--generation-config",
        "vllm",
    ]
    if policy_version:
        if checkpoint is None:
            raise ValueError("A nonzero policy version requires its compact checkpoint")
        command.extend(["--lora-modules", f"kernel-policy-{policy_version}={checkpoint}"])
    return command


def runtime_environment(python: str, environment: dict[str, str] | None = None) -> dict[str, str]:
    result = os.environ.copy() if environment is None else environment.copy()
    python_executable = Path(python).absolute()
    executable_directory = str(python_executable.parent)
    result["PATH"] = os.pathsep.join([executable_directory, result.get("PATH", "")]).rstrip(
        os.pathsep
    )
    return result


def python_header_environment(python: str, environment: dict[str, str]) -> dict[str, str]:
    python_executable = Path(python).absolute()
    runtime_root = python_executable.parents[2]
    header_root = runtime_root / "python-headers/usr/include"
    include_directory = header_root / "python3.12"
    multiarch_directory = header_root / "x86_64-linux-gnu/python3.12"
    if (
        not (include_directory / "Python.h").is_file()
        or not (multiarch_directory / "pyconfig.h").is_file()
    ):
        raise FileNotFoundError(
            "Cached Python 3.12 headers are required for native extension builds"
        )
    cpath = [str(include_directory), str(header_root)]
    if environment.get("CPATH"):
        cpath.append(environment["CPATH"])
    environment["CPATH"] = os.pathsep.join(cpath)
    return environment


def inference_environment(
    inference_python: str, environment: dict[str, str] | None = None
) -> dict[str, str]:
    result = python_header_environment(
        inference_python, runtime_environment(inference_python, environment)
    )
    result["VLLM_SERVER_DEV_MODE"] = "1"
    return result


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def collection_record(output_directory: Path) -> dict | None:
    path = output_directory / "groups.json"
    if not path.exists():
        return None
    groups = json.loads(path.read_text())
    if len(groups) != 1:
        raise ValueError("Single-task collection must produce one group record")
    return groups[0]


def ensure_gpus_idle() -> None:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    )
    pids = [line.strip() for line in output.splitlines() if line.strip().isdigit()]
    if pids:
        raise RuntimeError(f"Refusing resume while GPU compute PIDs remain: {pids}")


def bootstrap_state(arguments, task_order: list[str]) -> dict:
    if getattr(arguments, "parent_run_dir", None) is not None:
        return continuation_state(arguments, task_order)
    if arguments.bootstrap_dir is None:
        return {
            "status": "ready",
            "policy_version": 0,
            "checkpoint": None,
            "completed_task_ids": [],
            "results": [],
            "task_order": task_order,
        }
    bootstrap = arguments.bootstrap_dir
    group_path = bootstrap / "input/training-group.json"
    if not group_path.exists():
        group_path = bootstrap / "rollouts/training-group.json"
    if not group_path.exists():
        group_path = bootstrap / "collection/rollouts/training-group.json"
    group = read_json(group_path)
    report = read_json(bootstrap / "update/report-rank-0.json")
    second_rank = read_json(bootstrap / "update/report-rank-1.json")
    version = report.get("global_step")
    task_id = group.get("task")
    if (
        report.get("status") != "passed"
        or second_rank.get("status") != "passed"
        or second_rank.get("global_step") != version
        or second_rank.get("sampled_policy_version") != report.get("sampled_policy_version")
        or not isinstance(version, int)
        or isinstance(version, bool)
        or version < 1
    ):
        raise ValueError("Bootstrap requires a completed, successful optimizer update")
    if (
        group.get("policy_version") != version - 1
        or report.get("sampled_policy_version") != version - 1
        or task_id not in task_order
    ):
        raise ValueError("Bootstrap group, policy version or curriculum task does not match")
    audit_path = bootstrap / "audit/report-rank-0.json"
    audit_rank1_path = bootstrap / "audit/report-rank-1.json"
    audit_report = read_json(audit_path)
    audit_rank1 = read_json(audit_rank1_path)
    if audit_report.get("status") != "audited" or audit_rank1.get("status") != "audited":
        raise ValueError("Bootstrap lacks a completed two-rank probability audit")
    audit_summary = audit_passed(audit_path, group)
    checkpoint = bootstrap / "update" / f"checkpoint-step-{version}"
    checkpoint_metadata(checkpoint, version)
    order = curriculum_order(arguments.dataset, task_id)
    return {
        "status": "ready",
        "policy_version": version,
        "checkpoint": str(checkpoint),
        "completed_task_ids": [task_id],
        "results": [
            {
                "task_id": task_id,
                "status": "trained",
                "policy_version": version - 1,
                "optimizer_step": version,
                "rewards": group["rewards"],
                "probability_audit": audit_summary,
                "metrics": report.get("metrics", {}),
                "update_summary": {
                    "updates": report.get("updates", []),
                    "native_loss": report.get("native_loss"),
                    "peak_allocated_bytes": report.get("peak_allocated_bytes"),
                    "peak_reserved_bytes": report.get("peak_reserved_bytes"),
                },
                "source_run": str(bootstrap),
                "quality_improvement_claimed": False,
            }
        ],
        "task_order": order,
    }


def continuation_state(arguments, task_order: list[str]) -> dict:
    if arguments.bootstrap_dir is not None or getattr(arguments, "replay_source", None):
        raise ValueError("A parent continuation cannot also select a bootstrap or replay group")
    parent = arguments.parent_run_dir.resolve()
    with (parent / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = parent / "state.json"
        state = read_json(state_path)
        manifest = read_json(parent / "inputs.json")
        if state.get("status") not in {"paused", "needs_diagnosis"}:
            raise ValueError("Parent must have stopped at a saved continuation boundary")
        if state.get("pending_directory") or state.get("pending_task_id"):
            raise ValueError("Preserve and diagnose unfinished parent samples before continuing")
        if (
            manifest.get("model_revision") != MODEL_REVISION
            or manifest.get("dataset_sha256")
            != hashlib.sha256(arguments.dataset.read_bytes()).hexdigest()
            or state.get("task_order") != task_order
            or len(task_order) != 50
            or len(set(task_order)) != 50
        ):
            raise ValueError("Parent model, dataset or full curriculum order changed")
        if (
            state.get("manifest_sha256")
            != hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        ):
            raise ValueError("Parent input manifest checksum changed")
        version = state.get("policy_version")
        completed = state.get("completed_task_ids", [])
        trained = [
            result for result in state.get("results", []) if result.get("status") == "trained"
        ]
        if (
            type(version) is not int
            or version < 1
            or len(completed) != version
            or len(set(completed)) != version
            or not set(completed).issubset(task_order)
            or [result.get("task_id") for result in trained] != completed
            or [result.get("optimizer_step") for result in trained] != list(range(1, version + 1))
            or any(result.get("policy_version") != index for index, result in enumerate(trained))
        ):
            raise ValueError("Parent coverage or optimizer history is inconsistent")
        checkpoint = Path(state["checkpoint"])
        if not checkpoint.resolve().is_relative_to(parent):
            raise ValueError("Parent checkpoint must belong to the selected run")
        selected = copy.copy(arguments)
        selected.parent_run_dir = None
        selected.bootstrap_dir = checkpoint.parent.parent
        validated = bootstrap_state(selected, task_order)
        if (
            validated["policy_version"] != version
            or validated["completed_task_ids"] != completed[-1:]
            or validated["results"][0]["rewards"] != trained[-1]["rewards"]
        ):
            raise ValueError("Parent checkpoint audit differs from the saved optimizer history")
        result = {
            "status": "ready",
            "policy_version": version,
            "checkpoint": str(checkpoint),
            "completed_task_ids": copy.deepcopy(completed),
            "results": copy.deepcopy(state["results"]),
            "task_order": list(task_order),
            "attempts": copy.deepcopy(state.get("attempts", {})),
            "deferred_task_ids": copy.deepcopy(state.get("deferred_task_ids", [])),
            "deferred_groups": copy.deepcopy(state.get("deferred_groups", [])),
            "preserved_incomplete_runs": copy.deepcopy(state.get("preserved_incomplete_runs", [])),
            "continuation_parent": {
                "run_id": state["run_id"],
                "directory": str(parent),
                "state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
                "input_manifest_sha256": state["manifest_sha256"],
                "source_commit": manifest["source_commit"],
                "policy_version": version,
                "failure_preserved": copy.deepcopy(state.get("failure")),
                "rejected_group_trained": False,
            },
        }
    return result


def preserve_incomplete_run(arguments, state: dict) -> None:
    selected = getattr(arguments, "preserved_incomplete_run", None)
    if selected is None:
        return
    directory = selected.resolve()
    with (directory / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = read_json(directory / "state.json")
        manifest = read_json(directory / "inputs.json")
        task = original.get("pending_task_id")
        if (
            original.get("status") != "failed"
            or original.get("policy_version") != state["policy_version"]
            or original.get("completed_task_ids") != state["completed_task_ids"]
            or original.get("task_order") != state["task_order"]
            or original.get("checkpoint") != state["checkpoint"]
            or task not in state["task_order"]
            or task in state["completed_task_ids"]
            or manifest.get("dataset_sha256")
            != hashlib.sha256(arguments.dataset.read_bytes()).hexdigest()
            or manifest.get("model_revision") != MODEL_REVISION
            or original.get("manifest_sha256")
            != hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
            or original.get("continuation_parent", {}).get("run_id")
            != state.get("continuation_parent", {}).get("run_id")
        ):
            raise ValueError("Incomplete run differs from the validated checkpoint/coverage parent")
        pending = Path(original["pending_directory"]).resolve()
        if not pending.is_relative_to(directory) or not pending.is_dir():
            raise ValueError("Incomplete samples must remain inside their original run")
        if (pending / "collection/rollouts/generation-manifest.json").exists():
            raise ValueError("A complete generated group must be scored, not discarded")
        saved = {
            "run_id": original["run_id"],
            "directory": str(directory),
            "state_sha256": hashlib.sha256((directory / "state.json").read_bytes()).hexdigest(),
            "input_manifest_sha256": original["manifest_sha256"],
            "source_commit": manifest["source_commit"],
            "task_id": task,
            "sampled_policy_version": original["policy_version"],
            "failure": original.get("failure"),
            "preserved_files_sha256": {
                str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(pending.rglob("*"))
                if path.is_file()
                and not path.is_symlink()
                and path.suffix in {".json", ".jsonl", ".log", ".csv"}
            },
            "fresh_generation_reason": "Explicit continuation after capacity/cancellation repair and context-limit change",
            "incomplete_samples_used_for_training": False,
        }
        state.setdefault("preserved_incomplete_runs", []).append(saved)
        state.setdefault("attempts", {})[task] = max(
            state.get("attempts", {}).get(task, 0), original.get("attempts", {}).get(task, 0)
        )
        state["results"].append(
            {
                "task_id": task,
                "status": "incomplete_preserved",
                "policy_version": state["policy_version"],
                "optimizer_step": None,
                "trace_directory": str(pending / "collection/rollouts"),
                "source_run": original["run_id"],
                "failure": original.get("failure"),
            }
        )


def defer_uninformative_group(state: dict, result: dict) -> bool:
    group = result.get("group_record", {})
    task_id = result.get("task_id")
    rewards = group.get("rewards", [])
    reason = group.get("reason")
    if (
        result.get("status") != "needs_diagnosis"
        or task_id not in state["task_order"]
        or task_id in state["completed_task_ids"]
        or result.get("optimizer_step") is not None
        or group.get("task") != task_id
        or group.get("policy_version") != result.get("policy_version")
        or group.get("admitted") is not False
        or group.get("infrastructure_errors") != []
        or not isinstance(rewards, list)
        or len(rewards) != GROUP_SIZE
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in rewards)
        or reason not in {"weak_signal", "zero_variance"}
    ):
        return False
    if (reason == "weak_signal" and max(rewards) > 0.1) or (
        reason == "zero_variance" and has_signal(rewards)
    ):
        return False
    deferred = state.setdefault("deferred_task_ids", [])
    if task_id not in deferred:
        deferred.append(task_id)
        state.setdefault("deferred_groups", []).append(
            {
                "task_id": task_id,
                "sampled_policy_version": result["policy_version"],
                "reason": reason,
                "rewards": list(rewards),
                "trace_directory": result["trace_directory"],
                "optimizer_update": False,
            }
        )
    return True


def audit_passed(report_path: Path, group: dict) -> dict:
    result = require_probability_agreement(report_path, group)
    return {
        "group_sha256": result["group_sha256"],
        "policy_version": result["policy_version"],
        "tokens": result["tokens"],
        "mean_absolute_log_ratio": result["mean_absolute_log_ratio"],
        "p95_absolute_log_ratio": result["p95_absolute_log_ratio"],
        "outside_ratio_0_8_to_1_2_fraction": result["outside_ratio_0_8_to_1_2_fraction"],
        "optimizer_steps": result["optimizer_steps"],
    }


def next_phase_directory(step_directory: Path, name: str) -> Path:
    candidate = step_directory / name
    if not candidate.exists():
        return candidate
    attempt = 2
    while (candidate := step_directory / f"{name}-attempt-{attempt:02d}").exists():
        attempt += 1
    return candidate


async def wait_for_server(server, session, deadline: float) -> None:
    while time.time() < deadline:
        if server.returncode is not None:
            raise RuntimeError("vLLM exited before readiness; inspect inference.log")
        try:
            async with session.get("http://127.0.0.1:8006/health") as response:
                if response.status == 200:
                    return
        except (TimeoutError, aiohttp.ClientError):
            pass
        await asyncio.sleep(2)
    raise TimeoutError("vLLM readiness exceeded its phase deadline")


async def start_telemetry(path: Path):
    stream = path.open("wb")
    process = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        "--query-gpu=index,timestamp,memory.used,utilization.gpu,power.draw",
        "--format=csv",
        "-l",
        "1",
        stdout=stream,
        start_new_session=True,
    )
    return process, stream


async def stop_telemetry(process, stream) -> None:
    await stop_group(process.pid)
    await process.wait()
    stream.close()


async def collect_task(
    arguments,
    budget: dict,
    step_directory: Path,
    task_id: str,
    version: int,
    checkpoint: Path | None,
    inference=None,
) -> dict:
    deadline = phase_deadline(budget, time.time(), 2400)
    telemetry, telemetry_stream = await start_telemetry(step_directory / "inference-telemetry.csv")
    server = None
    result = {}
    try:
        if inference is not None:
            running_server = await inference.ensure(version, checkpoint, step_directory)
        else:
            environment = inference_environment(arguments.inference_python)
            server = await asyncio.create_subprocess_exec(
                *server_command(arguments, version, checkpoint),
                stdout=(step_directory / "inference.log").open("wb"),
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
                env=environment,
            )
            running_server = server
        timeout = min(600, deadline - time.time())
        if timeout <= 0:
            raise TimeoutError("No budget remains for vLLM startup")
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            await wait_for_server(running_server, session, time.time() + timeout)
            async with session.get("http://127.0.0.1:8006/v1/models") as response:
                response.raise_for_status()
                models = await response.json()
        expected_model = "kernel-base" if version == 0 else f"kernel-policy-{version}"
        model_ids = [item.get("id") for item in models.get("data", [])]
        if expected_model not in model_ids:
            raise ValueError(f"Expected served LoRA model {expected_model}; got {model_ids}")
        (step_directory / "serving.json").write_text(
            json.dumps(
                {
                    "policy_version": version,
                    "served_model": expected_model,
                    "available_models": model_ids,
                    "adapter_path": str(checkpoint) if checkpoint else None,
                },
                indent=2,
            )
        )
        output_directory = step_directory / "collection/rollouts"
        adapter_name = expected_model
        command = [
            arguments.python,
            "-m",
            "kernel_trl.collect",
            "--checkpoint",
            str(arguments.checkpoint),
            "--dataset",
            str(arguments.dataset),
            "--output-dir",
            str(output_directory),
            "--rollouts",
            str(GROUP_SIZE),
            "--model-timeout-s",
            str(MODEL_TIMEOUT_SECONDS),
            "--inference-http-timeout-s",
            str(INFERENCE_HTTP_TIMEOUT_SECONDS),
            "--episode-timeout-s",
            str(EPISODE_TIMEOUT_SECONDS),
            "--group-timeout-s",
            str(COLLECTION_LIMIT_SECONDS),
            "--policy-version",
            str(version),
            "--adapter-name",
            adapter_name,
            "--task-ids",
            task_id,
            "--defer-scoring",
            "--context-tokens",
            str(getattr(arguments, "context_tokens", LEGACY_CONTEXT_TOKENS)),
            "--max-turns",
            str(getattr(arguments, "max_turns", 8)),
        ]
        if getattr(arguments, "cuda_docs", None):
            command.extend(
                [
                    "--cuda-docs",
                    str(arguments.cuda_docs),
                    "--cuda-docs-sha256",
                    arguments.cuda_docs_sha256,
                ]
            )
        if getattr(arguments, "precompile_finals", False):
            command.append("--precompile-finals")
        collect_timeout = min(COLLECTION_LIMIT_SECONDS, deadline - time.time())
        try:
            await execute(command, step_directory / "collect.log", collect_timeout)
        except Exception as error:
            result = collection_record(output_directory)
            if result is None or result.get("admitted"):
                raise
            result.update(
                status="needs_diagnosis",
                collection_error=f"{type(error).__name__}: {error}",
            )
            return result
        if (output_directory / "generation-manifest.json").exists():
            return {"status": "generated", "task": task_id, "policy_version": version}
        result = collection_record(output_directory)
        if result is None:
            raise ValueError("Collector exited without a group record")
        if not result.get("admitted"):
            result["status"] = "needs_diagnosis"
            return result
        result["status"] = "admitted"
        return result
    finally:
        if server is not None:
            await stop_group(server.pid)
            await server.wait()
        await stop_telemetry(telemetry, telemetry_stream)


async def run_torch_phase(
    arguments, budget: dict, command: list[str], log_path: Path, telemetry_path: Path, timeout: int
) -> None:
    deadline = phase_deadline(budget, time.time(), timeout)
    telemetry, telemetry_stream = await start_telemetry(telemetry_path)
    try:
        await execute(command, log_path, min(timeout, deadline - time.time()))
    finally:
        await stop_telemetry(telemetry, telemetry_stream)


def torchrun_command(
    arguments,
    group_path: Path,
    output_directory: Path,
    checkpoint: Path | None,
    version: int,
    audit_report: Path | None = None,
    audit_only: bool = False,
) -> list[str]:
    command = [
        arguments.python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        "--max_restarts=0",
        "-m",
        "kernel_trl.rlm_update",
        "--checkpoint",
        str(arguments.checkpoint),
        "--group",
        str(group_path),
        "--output-dir",
        str(output_directory),
    ]
    if checkpoint is not None:
        command.extend(["--resume-from", str(checkpoint)])
    if audit_only:
        command.append("--audit-only")
    elif audit_report is not None:
        command.extend(["--audit-report", str(audit_report)])
    else:
        raise ValueError("Training requires the exact-group probability audit report")
    return command


def read_update_reports(
    output_directory: Path, expected_step: int, sampled_policy_version: int
) -> dict:
    reports = [
        read_json(output_directory / f"report-rank-{rank}.json") for rank in range(WORLD_SIZE)
    ]
    for report in reports:
        if (
            report.get("status") != "passed"
            or report.get("global_step") != expected_step
            or report.get("sampled_policy_version") != sampled_policy_version
            or report.get("quality_improvement_claimed") is not False
        ):
            raise ValueError("Both ranks must report exactly one verified update")
    checkpoint = output_directory / f"checkpoint-step-{expected_step}"
    metadata = checkpoint_metadata(checkpoint, expected_step)
    return {
        "reports": reports,
        "checkpoint": str(checkpoint),
        "checkpoint_metadata": metadata,
    }


def successful_update_directory(
    step_directory: Path, expected_step: int, version: int
) -> tuple[Path, dict] | None:
    for directory in sorted(step_directory.glob("update*")):
        if not all((directory / f"report-rank-{rank}.json").exists() for rank in range(WORLD_SIZE)):
            continue
        result = read_update_reports(directory, expected_step, version)
        return directory, result
    return None


def record_training_result(
    arguments,
    state: dict,
    task_id: str,
    version: int,
    step_directory: Path,
    group: dict,
    audit_summary: dict,
    update_directory: Path,
    update: dict,
) -> None:
    result = {
        "task_id": task_id,
        "status": "trained",
        "policy_version": version,
        "optimizer_step": version + 1,
        "rewards": group["rewards"],
        "advantages": group["advantages"],
        "root_turns": group["real_rows"],
        "trainable_tokens": group["trainable_tokens"],
        "probability_audit": audit_summary,
        "update_reports": update["reports"],
        "metrics": update["reports"][0].get("metrics", {}),
        "checkpoint": update["checkpoint"],
        "checkpoint_bytes": sum(
            (Path(update["checkpoint"]) / name).stat().st_size
            for name in update["checkpoint_metadata"]["sha256"]
        ),
        "trace_directory": str(step_directory / "collection/rollouts"),
        "update_directory": str(update_directory),
        "quality_improvement_claimed": False,
    }
    state["results"].append(result)
    state["completed_task_ids"].append(task_id)
    state["policy_version"] = version + 1
    state["checkpoint"] = update["checkpoint"]
    state.pop("pending_task_id", None)
    state.pop("pending_directory", None)
    write_json(arguments.output_dir / "state.json", state)
    record_event(
        arguments.output_dir,
        "optimizer_update_completed",
        task_id=task_id,
        optimizer_step=version + 1,
        rewards=group["rewards"],
        checkpoint=update["checkpoint"],
    )


def initialize_run(arguments) -> dict:
    ordered = curriculum_order(arguments.dataset)
    state = bootstrap_state(arguments, ordered)
    preserve_incomplete_run(arguments, state)
    if getattr(arguments, "continue_weak_groups", False):
        for result in state["results"]:
            defer_uninformative_group(state, result)
    manifest = {
        "run_id": arguments.run_id,
        "experiment_id": getattr(arguments, "experiment_id", "trl-l1-grpo-baseline"),
        "provider": "cloudrift",
        "backend": "trl",
        "model_revision": MODEL_REVISION,
        "dataset_sha256": hashlib.sha256(arguments.dataset.read_bytes()).hexdigest(),
        "dataset_path": str(arguments.dataset),
        "source_sha256": source_hashes(arguments.source_root),
        "source_commit": arguments.source_commit,
        "cuda_arch_list": arguments.cuda_arch_list,
        "rollouts_per_task": GROUP_SIZE,
        "initial_concurrency": INFLIGHT,
        "context_tokens": validate_context_tokens(
            getattr(arguments, "context_tokens", LEGACY_CONTEXT_TOKENS)
        ),
        "normal_reply_tokens": 4096,
        "recovery_reply_tokens": 1024,
        "max_turns": getattr(arguments, "max_turns", 8),
        "enable_thinking": True,
        "cuda_docs": documentation_configuration(arguments),
        "model_request_timeout_seconds": MODEL_TIMEOUT_SECONDS,
        "inference_http_timeout_seconds": INFERENCE_HTTP_TIMEOUT_SECONDS,
        "episode_timeout_seconds": EPISODE_TIMEOUT_SECONDS,
        "group_collection_timeout_seconds": COLLECTION_LIMIT_SECONDS,
        "max_child_calls": 2,
        "one_update_per_admitted_task_group": True,
        "inference_training_overlap": False,
        "lora": {
            "rank": 8,
            "alpha": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "optimizer": {
            "name": "Dion Muon",
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "mu": 0.95,
            "adjust_lr": "rms_norm",
        },
        "trainer": {
            "loss": "TRL DAPO with externally calculated trajectory GRPO advantages",
            "kl_beta": 0.0,
            "router_auxiliary_loss": False,
            "updates_per_group": 1,
        },
        "speed_reward_weight": 0.1,
        "bootstrap_dir": str(arguments.bootstrap_dir) if arguments.bootstrap_dir else None,
        "continuation_parent": state.get("continuation_parent"),
        "preserved_incomplete_runs": state.get("preserved_incomplete_runs", []),
        "replay_source": str(arguments.replay_source)
        if getattr(arguments, "replay_source", None)
        else None,
        "final_scoring_after_inference_exit": not getattr(arguments, "reuse_inference", False),
        "reuse_inference": getattr(arguments, "reuse_inference", False),
        "precompile_finals": getattr(arguments, "precompile_finals", False),
        "weak_group_handling": "defer"
        if getattr(arguments, "continue_weak_groups", False)
        else "stop",
    }
    dataset_copy = arguments.output_dir / "inputs/lvl1_seed42.jsonl"
    dataset_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(arguments.dataset, dataset_copy)
    write_json(arguments.output_dir / "inputs.json", manifest)
    state.update(
        {
            "run_id": arguments.run_id,
            "target_task_count": 50,
            "manifest_sha256": hashlib.sha256(
                json.dumps(manifest, sort_keys=True).encode()
            ).hexdigest(),
            "started_unix_seconds": time.time(),
        }
    )
    write_json(arguments.output_dir / "state.json", state)
    return state


def load_state(arguments) -> dict:
    state = read_json(arguments.output_dir / "state.json")
    if state.get("status") not in {"ready", "paused", "running"}:
        raise ValueError(f"Cannot resume a run in state {state.get('status')}")
    ensure_gpus_idle()
    manifest = read_json(arguments.output_dir / "inputs.json")
    dataset_sha256 = hashlib.sha256(arguments.dataset.read_bytes()).hexdigest()
    if manifest.get("dataset_sha256") != dataset_sha256:
        raise ValueError("Dataset bytes changed since this run started")
    if manifest.get("source_commit") != arguments.source_commit:
        raise ValueError("Source commit changed since this run started")
    if manifest.get("experiment_id", "trl-l1-grpo-baseline") != getattr(
        arguments, "experiment_id", "trl-l1-grpo-baseline"
    ):
        raise ValueError("Resume cannot change immutable experiment identity")
    requested_handling = "defer" if getattr(arguments, "continue_weak_groups", False) else "stop"
    if manifest.get("context_tokens", LEGACY_CONTEXT_TOKENS) != getattr(
        arguments, "context_tokens", LEGACY_CONTEXT_TOKENS
    ):
        raise ValueError("Resume cannot change the immutable run context limit")
    if manifest.get("weak_group_handling", "stop") != requested_handling:
        raise ValueError("Weak-group scheduling changed since this run started")
    if manifest.get("cuda_arch_list") != arguments.cuda_arch_list:
        raise ValueError("CUDA architecture target changed since this run started")
    if manifest.get("max_turns", 8) != getattr(arguments, "max_turns", 8):
        raise ValueError("Resume cannot change immutable turn limits")
    if manifest.get("cuda_docs") != documentation_configuration(arguments):
        raise ValueError("Resume cannot change immutable documentation")
    for option in ("reuse_inference", "precompile_finals"):
        if manifest.get(option, False) != getattr(arguments, option, False):
            raise ValueError(f"Resume cannot change immutable scheduling option {option}")
    checkpoint = state.get("checkpoint")
    version = state.get("policy_version", 0)
    if version:
        if checkpoint is None:
            raise ValueError("Nonzero policy state has no compact checkpoint")
        checkpoint_metadata(Path(checkpoint), version)
    return state


def write_report(arguments, state: dict, budget: dict) -> None:
    groups = [
        {
            "task": result["task_id"],
            "rewards": result.get("rewards", result.get("group_record", {}).get("rewards", [])),
            "admitted": result.get("status") == "trained",
            "policy_version": result.get("policy_version"),
            "reason": result.get("status"),
        }
        for result in state["results"]
    ]
    report = {
        "run_id": arguments.run_id,
        "experiment_id": getattr(arguments, "experiment_id", "trl-l1-grpo-baseline"),
        "backend": "trl",
        "provider": "cloudrift",
        "status": state["status"],
        "target_task_count": state["target_task_count"],
        "completed_task_count": len(state["completed_task_ids"]),
        "completed_task_ids": state["completed_task_ids"],
        "deferred_task_ids": state.get("deferred_task_ids", []),
        "deferred_task_count": len(state.get("deferred_task_ids", [])),
        "deferred_groups": state.get("deferred_groups", []),
        "optimizer_steps": state["policy_version"],
        "current_policy_version": state["policy_version"],
        "context_tokens": getattr(arguments, "context_tokens", LEGACY_CONTEXT_TOKENS),
        "max_turns": getattr(arguments, "max_turns", 8),
        "enable_thinking": True,
        "cuda_docs": documentation_configuration(arguments),
        "current_checkpoint": state.get("checkpoint"),
        "resumed_from": state.get("continuation_parent", {}).get("run_id"),
        "preserved_incomplete_runs": state.get("preserved_incomplete_runs", []),
        "resume_step": state.get("continuation_parent", {}).get("policy_version"),
        "groups": groups,
        "results": state["results"],
        "failure": state.get("failure"),
        "quality_improvement_claimed": False,
        "elapsed_seconds": time.time() - state.get("started_unix_seconds", time.time()),
        "budget": spending_snapshot(budget, time.time()),
    }
    write_json(arguments.output_dir / "report.json", report)


async def train_task(arguments, budget: dict, state: dict, task_id: str, inference=None) -> bool:
    version = state["policy_version"]
    checkpoint = Path(state["checkpoint"]) if state.get("checkpoint") else None
    pending_directory = state.get("pending_directory")
    if state.get("pending_task_id") == task_id and pending_directory:
        candidate = Path(pending_directory)
        if candidate.exists() and any(
            (candidate / "collection/rollouts" / name).exists()
            for name in ("training-group.json", "generation-manifest.json")
        ):
            step_directory = candidate
            attempt = state.get("attempts", {}).get(task_id, 1)
        else:
            attempt = state.get("attempts", {}).get(task_id, 0) + 1
            step_directory = (
                arguments.output_dir
                / "groups"
                / f"step-{version + 1:03d}-{task_id}-attempt-{attempt:02d}"
            )
            step_directory.mkdir(parents=True, exist_ok=False)
    else:
        attempt = state.get("attempts", {}).get(task_id, 0) + 1
        step_directory = (
            arguments.output_dir
            / "groups"
            / f"step-{version + 1:03d}-{task_id}-attempt-{attempt:02d}"
        )
        step_directory.mkdir(parents=True, exist_ok=False)
    state["pending_task_id"] = task_id
    state["pending_directory"] = str(step_directory)
    state["status"] = "running"
    state.setdefault("attempts", {})[task_id] = attempt
    write_json(arguments.output_dir / "state.json", state)
    record_event(
        arguments.output_dir, "task_group_started", task_id=task_id, policy_version=version
    )

    try:
        group_path = step_directory / "collection/rollouts/training-group.json"
        if group_path.exists():
            group = read_json(group_path)
            collection = collection_record(step_directory / "collection/rollouts") or {}
            collection["status"] = (
                "admitted" if group.get("policy_version") == version else "needs_diagnosis"
            )
        else:
            output = step_directory / "collection/rollouts"
            replay = getattr(arguments, "replay_source", None)
            if replay and not state.get("replay_source_consumed"):
                from kernel_trl.scoring import import_saved_group

                task = next(
                    task for task in load_tasks(arguments.dataset) if task["name"] == task_id
                )
                import_saved_group(
                    replay, output, task, version, GROUP_SIZE, checkpoint, MODEL_REVISION
                )
                state["replay_source_consumed"] = True
                write_json(arguments.output_dir / "state.json", state)
                record_event(
                    arguments.output_dir,
                    "saved_group_imported",
                    task_id=task_id,
                    source=str(replay),
                )
            if (output / "generation-manifest.json").exists():
                collection = {"status": "generated"}
            else:
                collection = await collect_task(
                    arguments,
                    budget,
                    step_directory,
                    task_id,
                    version,
                    checkpoint,
                    **({"inference": inference} if inference is not None else {}),
                )
            if collection.get("status") == "generated":
                if inference is not None:
                    await inference.sleep()
                record_event(arguments.output_dir, "final_scoring_started", task_id=task_id)
                await run_torch_phase(
                    arguments,
                    budget,
                    [
                        arguments.python,
                        "-m",
                        "kernel_trl.scoring",
                        "--directory",
                        str(output),
                        "--policy-version",
                        str(version),
                        "--task-id",
                        task_id,
                    ],
                    step_directory / "scoring.log",
                    step_directory / "scoring-telemetry.csv",
                    GROUP_SIZE * 480 + 60,
                )
                collection = collection_record(output) or {}
                collection["status"] = (
                    "admitted" if collection.get("admitted") else "needs_diagnosis"
                )
        if collection.get("status") != "admitted" or not group_path.exists():
            result = {
                "task_id": task_id,
                "status": collection.get("status", "collection_failed"),
                "policy_version": version,
                "group_record": collection,
                "optimizer_step": None,
                "trace_directory": str(step_directory / "collection/rollouts"),
            }
            state["results"].append(result)
            state["failure"] = result
            state["status"] = "needs_diagnosis"
            if collection.get("reason") == "infrastructure_error":
                state["status"] = "paused"
            else:
                state.pop("pending_task_id", None)
                state.pop("pending_directory", None)
            write_json(arguments.output_dir / "state.json", state)
            record_event(
                arguments.output_dir,
                "task_group_rejected",
                task_id=task_id,
                reason=result["status"],
            )
            return False

        group = read_json(group_path)
        if group["policy_version"] != version or group.get("task") != task_id:
            raise ValueError("Collected group does not match the active policy and task")
        if inference is not None:
            await inference.close()
        audit_dir = None
        for candidate in sorted(step_directory.glob("audit*")):
            if all((candidate / f"report-rank-{rank}.json").exists() for rank in range(WORLD_SIZE)):
                reports = [
                    read_json(candidate / f"report-rank-{rank}.json") for rank in range(WORLD_SIZE)
                ]
                if all(report.get("status") == "audited" for report in reports):
                    audit_dir = candidate
                    break
        if audit_dir is None:
            audit_dir = next_phase_directory(step_directory, "audit")
            audit_command = torchrun_command(
                arguments, group_path, audit_dir, checkpoint, version, audit_only=True
            )
            await run_torch_phase(
                arguments,
                budget,
                audit_command,
                step_directory / f"{audit_dir.name}.log",
                step_directory / f"{audit_dir.name}-telemetry.csv",
                PHASE_LIMIT_SECONDS,
            )
        audit_path = audit_dir / "report-rank-0.json"
        audit_report = read_json(audit_path)
        if audit_report.get("status") != "audited":
            raise ValueError("Audit-only phase did not finish without an optimizer update")
        audit_summary = audit_passed(audit_path, group)
        write_json(step_directory / "audit-summary.json", audit_summary)

        completed = successful_update_directory(step_directory, version + 1, version)
        if completed is None:
            update_dir = next_phase_directory(step_directory, "update")
            update_command = torchrun_command(
                arguments, group_path, update_dir, checkpoint, version, audit_report=audit_path
            )
            await run_torch_phase(
                arguments,
                budget,
                update_command,
                step_directory / f"{update_dir.name}.log",
                step_directory / f"{update_dir.name}-telemetry.csv",
                PHASE_LIMIT_SECONDS,
            )
            update = read_update_reports(update_dir, version + 1, version)
        else:
            update_dir, update = completed
        record_training_result(
            arguments,
            state,
            task_id,
            version,
            step_directory,
            group,
            audit_summary,
            update_dir,
            update,
        )
        return True
    except Exception as error:
        state["status"] = "failed"
        state["failure"] = {
            "task_id": task_id,
            "policy_version": version,
            "type": type(error).__name__,
            "error": str(error),
            "preserved_directory": str(step_directory),
        }
        write_json(arguments.output_dir / "state.json", state)
        record_event(arguments.output_dir, "task_failed", **state["failure"])
        raise


async def run(arguments) -> dict:
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRACKIO_DIR", str(arguments.output_dir / "trackio"))
    os.environ.update(
        training_environment(
            arguments.python,
            arguments.source_root,
            arguments.cuda_arch_list,
        )
    )
    lock_path = arguments.output_dir / "driver.lock"
    lock_stream = lock_path.open("a")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Another curriculum driver holds the run lock") from error
    state = None
    budget = None
    inference = None
    try:
        if arguments.resume:
            state = load_state(arguments)
        else:
            if (arguments.output_dir / "state.json").exists():
                raise ValueError(
                    "Run already exists; use --resume rather than starting a duplicate"
                )
            state = initialize_run(arguments)
        budget = read_json(arguments.budget_file)
        if getattr(arguments, "reuse_inference", False):
            from kernel_trl.inference_lifecycle import WarmInference

            inference = WarmInference(arguments, budget)
        state["status"] = "running"
        write_json(arguments.output_dir / "state.json", state)
        write_report(arguments, state, budget)
        record_event(arguments.output_dir, "driver_started", policy_version=state["policy_version"])

        remaining = [
            task
            for task in state["task_order"]
            if task not in state["completed_task_ids"]
            and task not in state.get("deferred_task_ids", [])
        ]
        already_run = 0
        initial_version = state["policy_version"]
        for task_id in remaining:
            if arguments.max_groups is not None and already_run >= arguments.max_groups:
                break
            max_updates = getattr(arguments, "max_updates", None)
            if max_updates is not None and state["policy_version"] - initial_version >= max_updates:
                break
            trained = await train_task(
                arguments,
                budget,
                state,
                task_id,
                **({"inference": inference} if inference is not None else {}),
            )
            already_run += 1
            if not trained:
                if getattr(arguments, "continue_weak_groups", False) and defer_uninformative_group(
                    state, state.get("failure", {})
                ):
                    state["status"] = "running"
                    state.pop("failure", None)
                    write_json(arguments.output_dir / "state.json", state)
                    write_report(arguments, state, budget)
                    record_event(
                        arguments.output_dir,
                        "task_deferred_without_update",
                        **state["deferred_groups"][-1],
                    )
                    continue
                break
            write_report(arguments, state, budget)
        if state["status"] == "running":
            state["status"] = "completed" if len(state["completed_task_ids"]) == 50 else "paused"
        record_event(arguments.output_dir, "driver_stopped", status=state["status"])
        write_json(arguments.output_dir / "state.json", state)
        write_report(arguments, state, budget)
        return read_json(arguments.output_dir / "report.json")
    except Exception as error:
        if state is not None:
            state["status"] = "failed"
            state["failure"] = {"type": type(error).__name__, "error": str(error)}
            write_json(arguments.output_dir / "state.json", state)
            record_event(arguments.output_dir, "driver_failed", **state["failure"])
            if budget is not None:
                write_report(arguments, state, budget)
        raise
    finally:
        if inference is not None:
            await inference.close()


def documentation_configuration(arguments) -> dict | None:
    from kernel_rlm.cuda_docs import docs_manifest

    path = getattr(arguments, "cuda_docs", None)
    if path is None:
        return None
    expected = getattr(arguments, "cuda_docs_sha256", None)
    if not expected:
        raise ValueError("A pinned CUDA documentation checksum is required")
    return {"path": str(path), **docs_manifest(path, expected)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-file", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--cuda-arch-list", default=DEFAULT_CUDA_ARCH_LIST)
    parser.add_argument("--context-tokens", type=int, default=DEFAULT_CONTEXT_TOKENS)
    parser.add_argument("--max-turns", type=int, choices=(8, 10), default=8)
    parser.add_argument("--cuda-docs", type=Path)
    parser.add_argument("--cuda-docs-sha256")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiment-id", default="trl-l1-grpo-baseline")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--inference-python", required=True)
    parser.add_argument("--bootstrap-dir", type=Path)
    parser.add_argument("--parent-run-dir", type=Path)
    parser.add_argument("--preserved-incomplete-run", type=Path)
    parser.add_argument("--replay-source", type=Path)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--continue-weak-groups", action="store_true")
    parser.add_argument("--reuse-inference", action="store_true")
    parser.add_argument("--precompile-finals", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    validate_context_tokens(arguments.context_tokens)
    if arguments.max_updates is not None and arguments.max_updates < 1:
        raise ValueError("The optimizer-update cap must be positive")
    print(json.dumps(asyncio.run(run(arguments)), indent=2), flush=True)


if __name__ == "__main__":
    main()
