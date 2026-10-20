"""Checkpoint-preserving single-learner RTX curriculum with dedicated inference."""

import argparse
import asyncio
import copy
import fcntl
import hashlib
import json
import os
import shutil
import signal
import time
from pathlib import Path

import aiohttp
from kernel_skyrl.generator import score_submission
from kernel_skyrl.preflight import load_tasks

from kernel_trl.checkpoint_state import checkpoint_metadata
from kernel_trl.inference_lifecycle import wait_for_gpu_release
from kernel_trl.live_curriculum import (
    MODEL_REVISION,
    ensure_gpus_idle,
    inference_environment,
    read_json,
    record_training_result,
    server_command,
    source_hashes,
    training_environment,
    wait_for_server,
    write_json,
    write_report,
)
from kernel_trl.live_driver import stop_group
from kernel_trl.probability_audit import require_probability_agreement
from kernel_trl.scoring import INFRASTRUCTURE_STATUSES, digest, load_episodes, score_generated
from kernel_trl.spending import phase_deadline, spending_snapshot

GROUP_SECONDS = 12000


def migration_state(bundle: Path, dataset: Path, checkpoint: Path) -> dict:
    receipt = read_json(bundle / "manifest.json")
    for name, expected in receipt["files"].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle.resolve()) or digest(path) != expected:
            raise ValueError("Migration input checksum differs")
    previous = read_json(bundle / "parent-state.json")
    inputs = read_json(bundle / "parent-inputs.json")
    version = previous["policy_version"]
    completed = previous["completed_task_ids"]
    trained = [item for item in previous["results"] if item.get("status") == "trained"]
    order = [task["name"] for task in load_tasks(dataset)]
    if (
        inputs["model_revision"] != MODEL_REVISION
        or inputs["dataset_sha256"] != digest(dataset)
        or previous["manifest_sha256"]
        != hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        or previous["status"] != "paused"
        or type(version) is not int
        or version < 1
        or len(completed) != version
        or len(set(completed)) != version
        or [item["task_id"] for item in trained] != completed
        or [item["optimizer_step"] for item in trained] != list(range(1, version + 1))
        or any(item["policy_version"] != index for index, item in enumerate(trained))
        or order != previous["task_order"]
        or len(order) != 50
        or len(set(order)) != 50
    ):
        raise ValueError("Migration changes source model, history, dataset or coverage")
    metadata = checkpoint_metadata(checkpoint, version, 1)
    original = read_json(bundle / "source-checkpoint-metadata.json")
    if (
        original["step"] != version
        or original["world_size"] != 2
        or original["sha256"]["adapter_model.safetensors"]
        != metadata["sha256"]["adapter_model.safetensors"]
    ):
        raise ValueError("Migrated adapters differ from original sampled policy")
    native = read_json(bundle / "native-readback.json")
    launch = read_json(bundle / "native-readback-launch.json")["command"]
    if (
        native["status"] != "passed"
        or native["goal_policy_version"] != version
        or native["optimizer_updates"] != 0
        or native["native_resume"]["adapter_shards_bitwise_equal"] != 320
        or native["native_resume"]["momentum_shards_restored"] != 320
        or not native["native_resume"]["full_rng_and_scheduler_restored"]
        or Path(launch[launch.index("--resume-from") + 1]).resolve() != checkpoint.resolve()
    ):
        raise ValueError("Native one-rank restore proof does not match checkpoint")
    pending = bundle / "pending/collection/rollouts"
    generation = read_json(pending / "generation-manifest.json")
    if (
        generation["task"] != previous["pending_task_id"]
        or generation["policy_version"] != version
        or len(generation["episodes"]) != 8
    ):
        raise ValueError("Pending migration group differs from source state")
    episodes = load_episodes(pending, generation)
    task = next(item for item in load_tasks(dataset) if item["name"] == generation["task"])
    if any(item["task"] != task for item in episodes):
        raise ValueError("Pending task bytes changed during migration")
    state = copy.deepcopy(previous)
    state.update(status="ready", checkpoint=str(checkpoint), pending_directory=None)
    state.pop("failure", None)
    state["continuation_parent"] = {
        "run_id": previous["run_id"],
        "policy_version": version,
        "state_sha256": digest(bundle / "parent-state.json"),
        "source_commit": inputs["source_commit"],
        "failure_preserved": previous.get("failure"),
    }
    state["migration_bundle_sha256"] = digest(bundle / "manifest.json")
    return state


def collector_command(arguments, output: Path, task: str, version: int) -> list[str]:
    return [
        arguments.python,
        "-m",
        "kernel_trl.collect",
        "--checkpoint",
        str(arguments.checkpoint),
        "--dataset",
        str(arguments.dataset),
        "--output-dir",
        str(output),
        "--rollouts",
        "8",
        "--max-turns",
        "10",
        "--context-tokens",
        "16384",
        "--child-prompt-tokens",
        "4096",
        "--child-tools",
        "--policy-version",
        str(version),
        "--adapter-name",
        f"kernel-policy-{version}",
        "--model-timeout-s",
        "600",
        "--inference-http-timeout-s",
        "630",
        "--episode-timeout-s",
        "2400",
        "--group-timeout-s",
        "4800",
        "--task-ids",
        task,
        "--cuda-docs",
        str(arguments.cuda_docs),
        "--cuda-docs-sha256",
        arguments.cuda_docs_sha256,
        "--dedicated-inference",
        "--defer-scoring",
        "--precompile-finals",
        "--report-to",
        "trackio",
    ]


def learner_command(
    arguments, output: Path, checkpoint: Path, group: Path, mode: str, audit: Path | None = None
) -> list[str]:
    command = [
        arguments.python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=1",
        "--max-restarts=0",
        "-m",
        "kernel_trl.rlm_update",
        "--checkpoint",
        str(arguments.checkpoint),
        "--resume-from",
        str(checkpoint),
        "--output-dir",
        str(output),
        "--learner-world-size",
        "1",
        "--report-to",
        "trackio",
    ]
    if mode == "readback":
        command.append("--readback-only")
    else:
        command.extend(["--group", str(group)])
        if mode == "audit":
            command.append("--audit-only")
        elif mode == "update" and audit is not None:
            command.extend(["--audit-report", str(audit)])
        else:
            raise ValueError("An optimizer update requires its exact-group audit")
    return command


async def revalidate_pending(output: Path, version: int, budget: dict, tracker) -> None:
    manifest = read_json(output / "generation-manifest.json")
    episodes = load_episodes(output, manifest)
    ledger_path = output / "migration-revalidation.json"
    ledger = (
        read_json(ledger_path)
        if ledger_path.exists()
        else {
            "generation_sha256": digest(output / "generation-manifest.json"),
            "reason": "Explicit hardware migration and validated candidate-crash repair",
            "scores": {},
        }
    )
    if ledger["generation_sha256"] != digest(output / "generation-manifest.json"):
        raise ValueError("Migration scoring samples changed")
    prior = output / "scoring-state-before-migration.json"
    if not prior.exists():
        shutil.copy2(output / "scoring-state.json", prior)
    original = read_json(prior)
    current = read_json(output / "scoring-state.json")
    for index, episode in enumerate(episodes):
        key = str(index)
        score_path = episode["directory"] / "score.json"
        before = episode["directory"] / "score-before-migration.json"
        if not before.exists():
            shutil.copy2(score_path, before)
        if digest(before) != original["scores"][key]["sha256"]:
            raise ValueError("Original migration score changed")
        entry = ledger["scores"].get(key)
        if entry is not None:
            if entry["status"] != "completed" or digest(score_path) != entry["sha256"]:
                raise ValueError(
                    "Interrupted migration verification requires diagnosis, not automatic retry"
                )
            score = read_json(score_path)
        else:
            phase_deadline(budget, time.time(), 480)
            ledger["scores"][key] = {
                "status": "started",
                "previous_attempts": original["scores"][key]["attempts"],
                "original_score_sha256": digest(before),
            }
            write_json(ledger_path, ledger)
            reward, verification = await score_submission(episode["task"], episode["record"])
            score = {
                "reward": reward,
                "verification": verification,
                "policy_version": version,
                "root_calls": len(episode["samples"]),
                "migration_revalidation": True,
            }
            write_json(score_path, score)
            ledger["scores"][key].update(
                status="completed",
                sha256=digest(score_path),
                total_attempts=original["scores"][key]["attempts"] + 1,
            )
            write_json(ledger_path, ledger)
        current["scores"][key] = {
            "sha256": digest(score_path),
            "attempts": ledger["scores"][key]["total_attempts"],
        }
        write_json(output / "scoring-state.json", current)
        tracker.log(
            {
                "migration/completed_revalidations": index + 1,
                "rollout/reward": score["reward"],
                "policy/global_step": version,
            }
        )
        if score["verification"].get("verification_status") in INFRASTRUCTURE_STATUSES:
            raise ValueError(
                "Migration verifier infrastructure failed; original scores and new attempt retained"
            )


async def run(arguments):
    os.environ["TRACKIO_DIR"] = str(arguments.trackio_dir.resolve())
    import trackio

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    global_lock = (arguments.output_dir.parent / "active-driver.lock").open("a")
    fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock = (arguments.output_dir / "driver.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ensure_gpus_idle()
    budget = read_json(arguments.budget_file)
    os.environ.update(training_environment(arguments.python, arguments.source_root, "12.0"))
    os.environ.update(
        CUDA_VISIBLE_DEVICES="0", CUDA_HOME="/usr/local/cuda", OMP_NUM_THREADS="4", MAX_JOBS="2"
    )
    os.environ.pop("KERNEL_INFERENCE_ADMIN_URL", None)
    state_path = arguments.output_dir / "state.json"
    if arguments.resume:
        state = read_json(state_path)
        inputs = read_json(arguments.output_dir / "inputs.json")
        if (
            inputs["source_commit"] != arguments.source_commit
            or inputs["source_sha256"] != source_hashes(arguments.source_root)
            or inputs["dataset_sha256"] != digest(arguments.dataset)
            or state["manifest_sha256"]
            != hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        ):
            raise ValueError("Resume changed immutable source or dataset")
        checkpoint_metadata(Path(state["checkpoint"]), state["policy_version"], 1)
    else:
        if state_path.exists():
            raise ValueError("Existing run requires --resume")
        state = migration_state(
            arguments.migration_bundle, arguments.dataset, arguments.checkpoint_state
        )
        state.update(
            run_id=arguments.run_id, started_unix_seconds=time.time(), target_task_count=50
        )
        pending = (
            arguments.output_dir
            / "groups"
            / f"step-{state['policy_version'] + 1:03d}-{state['pending_task_id']}-migrated"
        )
        shutil.copytree(arguments.migration_bundle / "pending", pending)
        state["pending_directory"] = str(pending)
        state["migration_pending"] = True
        inputs = {
            "run_id": arguments.run_id,
            "experiment_id": arguments.experiment_id,
            "backend": "trl",
            "provider": "cloudrift",
            "model_revision": MODEL_REVISION,
            "source_commit": arguments.source_commit,
            "source_sha256": source_hashes(arguments.source_root),
            "dataset_sha256": digest(arguments.dataset),
            "context_tokens": 16384,
            "max_turns": 10,
            "enable_thinking": True,
            "rollouts_per_task": 8,
            "initial_concurrency": 4,
            "child_prompt_tokens": 4096,
            "child_reply_tokens": 4096,
            "child_tools": True,
            "max_child_calls": 2,
            "learner_world_size": 1,
            "inference_physical_gpu": 1,
            "learner_verifier_physical_gpu": 0,
            "cuda_arch_list": "12.0",
            "inference_training_overlap": False,
            "continuation_parent": state["continuation_parent"],
            "migration_bundle_sha256": state["migration_bundle_sha256"],
            "source_pending_protocol_retained": True,
        }
        state["manifest_sha256"] = hashlib.sha256(
            json.dumps(inputs, sort_keys=True).encode()
        ).hexdigest()
        write_json(arguments.output_dir / "inputs.json", inputs)
        write_json(state_path, state)
    trackio.init(
        project="laguna-kernel-rlm",
        name=arguments.run_id,
        resume="allow",
        space_id=None,
        auto_log_gpu=False,
        auto_log_cpu=False,
        embed=False,
        config={
            "goal_resume_step": 19,
            "source_commit": arguments.source_commit,
            "gpu_roles": "RTX GPU1 inference; GPU0 verifier or learner",
            "historical_metrics_are_inherited": True,
        },
    )
    if not state.get("history_logged"):
        for row in [
            json.loads(line)
            for line in arguments.history_metrics.read_text().splitlines()
            if line.strip()
        ]:
            if row["step"] <= state["continuation_parent"]["policy_version"]:
                trackio.log(
                    {
                        "train/loss": row["loss/mean"],
                        "policy/global_step": row["step"],
                        "history/inherited": 1,
                    }
                )
        state["history_logged"] = True
        write_json(state_path, state)
    server = None

    def save():
        write_json(state_path, state)
        write_report(arguments, state, budget)
        report = read_json(arguments.output_dir / "report.json")
        report.update(
            title="RTX L1 continuation from checkpoint19",
            gpu_topology="Two RTX PRO6000: GPU1 inference, GPU0 verifier/one-rank learner",
            new_optimizer_steps=state["policy_version"] - 19,
        )
        write_json(arguments.output_dir / "report.json", report)

    async def phase(command, directory, name, seconds):
        phase_deadline(budget, time.time(), seconds)
        state["phase"] = name
        save()
        with (directory / f"{name}.log").open("ab") as stream:
            process = await asyncio.create_subprocess_exec(
                *command, stdout=stream, stderr=asyncio.subprocess.STDOUT, start_new_session=True
            )
        waiter = asyncio.create_task(process.wait())
        deadline = time.monotonic() + seconds
        try:
            while process.returncode is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{name} phase exceeded its original budget")
                output = directory / "collection/rollouts"
                completed = list(output.glob("group-0-trajectory-*/submission.json"))
                trackio.log(
                    {
                        "rollout/completed_trajectories": len(completed),
                        "policy/global_step": state["policy_version"],
                        "budget/estimated_usd": spending_snapshot(budget, time.time())[
                            "estimated_spent_usd"
                        ],
                    }
                )
                try:
                    await asyncio.wait_for(
                        asyncio.shield(waiter), min(15, max(0.01, deadline - time.monotonic()))
                    )
                except TimeoutError:
                    pass
            if process.returncode:
                raise RuntimeError(f"{name} exited{process.returncode}; inspect preserved log")
        finally:
            if process.returncode is None:
                await stop_group(process.pid)
                await process.wait()
            await waiter

    try:
        state["status"] = "running"
        save()
        for task in state["task_order"]:
            if task in state["completed_task_ids"] or task in state.get("deferred_task_ids", []):
                continue
            try:
                phase_deadline(budget, time.time(), GROUP_SECONDS)
            except ValueError as error:
                state.update(status="paused", phase="budget_gate", budget_gate=str(error))
                break
            version = state["policy_version"]
            checkpoint = Path(state["checkpoint"])
            if state.get("pending_task_id") == task and state.get("pending_directory"):
                directory = Path(state["pending_directory"])
            else:
                attempt = state.setdefault("attempts", {}).get(task, 0) + 1
                directory = (
                    arguments.output_dir
                    / "groups"
                    / f"step-{version + 1:03d}-{task}-attempt-{attempt:02d}"
                )
                directory.mkdir(parents=True, exist_ok=False)
                state["attempts"][task] = attempt
                state.update(pending_task_id=task, pending_directory=str(directory))
                save()
            output = directory / "collection/rollouts"
            if state.get("migration_pending") and state.get("pending_task_id") == task:
                state["phase"] = "migration_revalidation"
                save()
                await revalidate_pending(output, version, budget, trackio)
            if not (output / "generation-manifest.json").exists():
                if output.exists():
                    raise ValueError("Incomplete collection must be diagnosed, not regenerated")
                command = server_command(arguments, version, checkpoint)
                command[command.index("--tensor-parallel-size") + 1] = "1"
                environment = inference_environment(arguments.inference_python)
                environment.update(
                    CUDA_VISIBLE_DEVICES="1",
                    CUDA_HOME="/usr/local/cuda",
                    OMP_NUM_THREADS="4",
                    PATH=str(Path(arguments.inference_python).parent)
                    + ":/usr/local/cuda/bin:"
                    + os.environ["PATH"],
                )
                phase_deadline(budget, time.time(), 600)
                with (directory / "inference.log").open("ab") as stream:
                    server = await asyncio.create_subprocess_exec(
                        *command,
                        env=environment,
                        stdout=stream,
                        stderr=asyncio.subprocess.STDOUT,
                        start_new_session=True,
                    )
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    await wait_for_server(server, session, time.time() + 600)
                    async with session.get("http://127.0.0.1:8006/v1/models") as response:
                        response.raise_for_status()
                        models = await response.json()
                    if f"kernel-policy-{version}" not in [item["id"] for item in models["data"]]:
                        raise ValueError("Inference did not load the active sampled adapter")
                await phase(
                    collector_command(arguments, output, task, version), directory, "collect", 4800
                )
                await stop_group(server.pid)
                await server.wait()
                server = None
                await wait_for_gpu_release()
            phase_deadline(budget, time.time(), 3900)
            state["phase"] = "final_scoring"
            save()
            collection = await asyncio.wait_for(
                score_generated(
                    output,
                    version,
                    task,
                    on_score=lambda index, score, reused, version=version: trackio.log(
                        {
                            "verification/completed_scores": index + 1,
                            "rollout/reward": score["reward"],
                            "policy/global_step": version,
                        }
                    ),
                ),
                timeout=3900,
            )
            if collection["infrastructure_errors"]:
                raise ValueError("Persistent final verification infrastructure failure")
            if not collection["admitted"]:
                state.setdefault("deferred_task_ids", []).append(task)
                state.setdefault("deferred_groups", []).append(
                    {
                        "task_id": task,
                        "policy_version": version,
                        "reason": collection["reason"],
                        "rewards": collection["rewards"],
                    }
                )
                state["results"].append(
                    {
                        "task_id": task,
                        "status": collection["reason"],
                        "policy_version": version,
                        "optimizer_step": None,
                        "group_record": collection,
                        "trace_directory": str(output),
                    }
                )
                state.pop("pending_task_id", None)
                state.pop("pending_directory", None)
                state.pop("migration_pending", None)
                save()
                continue
            group_path = output / "training-group.json"
            group = read_json(group_path)
            for name, seconds in (("audit", 900), ("update", 1200), ("readback", 600)):
                ensure_gpus_idle()
                phase_directory = directory / name
                report_path = phase_directory / "report-rank-0.json"
                if not report_path.exists():
                    if phase_directory.exists():
                        raise ValueError("Incomplete learner phase requires diagnosis before retry")
                    current = (
                        directory / f"update/checkpoint-step-{version + 1}"
                        if name == "readback"
                        else checkpoint
                    )
                    command = learner_command(
                        arguments,
                        phase_directory,
                        current,
                        group_path,
                        name,
                        directory / "audit/report-rank-0.json" if name == "update" else None,
                    )
                    await phase(command, directory, name, seconds)
                result = read_json(report_path)
                expected = (
                    "audited" if name == "audit" else "readback" if name == "readback" else "passed"
                )
                if result["status"] != expected:
                    raise ValueError("Learner phase did not finish successfully")
                if name == "audit":
                    require_probability_agreement(report_path, group)
            update = read_json(directory / "update/report-rank-0.json")
            if (
                update["global_step"] != version + 1
                or update["sampled_policy_version"] != version
                or update["local_optimizer_steps"] != 1
            ):
                raise ValueError("Optimizer step or sampled policy differs")
            readback = read_json(directory / "readback/report-rank-0.json")
            if (
                readback["global_step"] != version + 1
                or readback["optimizer_steps"] != 0
                or readback["resume"]["adapter_shards_bitwise_equal"] != 320
                or readback["resume"]["momentum_shards_restored"] != 320
                or not readback["resume"]["full_rng_and_scheduler_restored"]
            ):
                raise ValueError("New checkpoint native readback did not restore complete state")
            saved = directory / f"update/checkpoint-step-{version + 1}"
            metadata = checkpoint_metadata(saved, version + 1, 1)
            record_training_result(
                arguments,
                state,
                task,
                version,
                directory,
                group,
                read_json(directory / "audit/report-rank-0.json")["probability_audit"],
                directory / "update",
                {"reports": [update], "checkpoint": str(saved), "checkpoint_metadata": metadata},
            )
            state.pop("migration_pending", None)
            trackio.log(
                {
                    "train/loss": update["metrics"]["train_loss"],
                    "policy/global_step": version + 1,
                    "history/inherited": 0,
                    "rollout/correct_final_kernels": sum(
                        1
                        for entry in load_episodes(
                            output, read_json(output / "generation-manifest.json")
                        )
                        if read_json(entry["directory"] / "score.json")["verification"].get(
                            "correct"
                        )
                        is True
                    ),
                }
            )
            save()
        if state["status"] == "running":
            state["status"] = "completed" if len(state["completed_task_ids"]) == 50 else "paused"
        save()
    except BaseException as error:
        state.update(status="paused", failure={"type": type(error).__name__, "error": str(error)})
        save()
        raise
    finally:
        if server is not None:
            await stop_group(server.pid)
            await server.wait()
            await wait_for_gpu_release()
        trackio.finish()
        lock.close()
        global_lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "checkpoint",
        "checkpoint-state",
        "dataset",
        "output-dir",
        "budget-file",
        "source-root",
        "migration-bundle",
        "history-metrics",
        "cuda-docs",
        "trackio-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-commit", "run-id", "python", "inference-python", "cuda-docs-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--experiment-id", default="trl-l1-rtx-continuation")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    arguments.context_tokens = 16384
    arguments.max_turns = 10
    arguments.cuda_arch_list = "12.0"
    if digest(arguments.cuda_docs) != arguments.cuda_docs_sha256:
        raise ValueError("Pinned CUDA documentation changed")
    signal.signal(
        signal.SIGTERM, lambda signal_number, frame: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    asyncio.run(run(arguments))


if __name__ == "__main__":
    main()
