"""SkyRL GeneratorInterface-compatible adapter around the unchanged V1 REPL program."""

import asyncio
import dataclasses
import json
import os
import signal
import sys
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from kernel_skyrl.bridge import CompletionBridge
from kernel_skyrl.contract import has_signal, stepwise_output, terminal_reward

DEFAULT_LIMITS = {
    "max_turns": 8,
    "max_tokens": 4096,
    "max_context_tokens": 12288,
    "model_timeout_s": 150,
    "enable_thinking": True,
    "enable_subcall": True,
    "max_subcalls": 2,
    "subcall_probe": False,
}


async def score_submission(task: dict, record: dict) -> tuple[float, dict]:
    from kernel_rlm.submission_contract import validate_submission_source
    from kernel_rlm.verification_state import VERIFICATION_PROTOCOL
    from kernel_rlm.verifier import verify_cuda

    source = record["submission"]
    if not source.strip():
        return 0.0, {"error": "No final submission", "verification_protocol": VERIFICATION_PROTOCOL}
    valid, error = validate_submission_source(source)
    if not valid:
        return 0.0, {
            "error": f"static validation: {error}",
            "verification_protocol": VERIFICATION_PROTOCOL,
        }
    result = await asyncio.to_thread(verify_cuda, task["context"], source)
    return terminal_reward(result, task.get("speed_reward_weight", 0.1)), result


class KernelGenerator:
    def __init__(
        self,
        client: Any,
        adapter_name: str,
        tokenizer_path: str,
        output_dir: Path,
        replacement_tasks: list[dict] | None = None,
        limits: dict | None = None,
        max_sample_batches: int = 3,
        episode_timeout_s: float = 720,
        scorer=score_submission,
        program_command: list[str] | None = None,
        defer_scoring: bool = False,
        execution_run_id: str | None = None,
        group_id: str | None = None,
        attempt_id: str | None = None,
    ):
        if not 1 <= max_sample_batches <= 3 or episode_timeout_s <= 0:
            raise ValueError("Use a positive timeout and at most three sampled groups")
        self.client = client
        self.adapter_name = adapter_name
        self.tokenizer_path = tokenizer_path
        self.output_dir = Path(output_dir)
        self.replacement_tasks = replacement_tasks or []
        self.limits = {**DEFAULT_LIMITS, **(limits or {})}
        if not 1 <= self.limits["max_tokens"] <= 4096:
            raise ValueError("Root output cap must be between 1 and 4096 tokens")
        if not 1 <= self.limits["max_turns"] <= 10:
            raise ValueError("At most ten root turns are supported")
        if not 12288 <= self.limits["max_context_tokens"] <= 16384:
            raise ValueError("The L40S adapter supports 12288 to 16384 context tokens")
        if not 0 <= self.limits["max_subcalls"] <= 2:
            raise ValueError("At most two child calls are supported")
        if self.limits.get("child_tools") and (
            not self.limits.get("dedicated_inference")
            or not self.limits.get("enable_thinking")
            or self.limits.get("child_prompt_tokens") != 4096
            or episode_timeout_s < 2400
        ):
            raise ValueError(
                "Iterative children require dedicated inference, thinking and explicit budgets"
            )
        self.max_sample_batches = max_sample_batches
        self.episode_timeout_s = episode_timeout_s
        self.scorer = scorer
        self.defer_scoring = defer_scoring
        self.execution_run_id = execution_run_id
        self.group_id = group_id
        self.attempt_id = attempt_id
        self.program_command = program_command or [
            sys.executable,
            "-m",
            "kernel_rlm_v1.program",
            "task.json",
        ]

    async def run_episode(self, task: dict, directory: Path, policy_version: int) -> dict:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "task.json").write_text(json.dumps(task))
        episode_provenance = {
            "execution_run_id": self.execution_run_id,
            "group_id": f"{self.group_id}:{task['name']}",
            "attempt_id": self.attempt_id,
            "episode_id": directory.name,
        }
        bridge = CompletionBridge(
            self.client,
            self.adapter_name,
            self.limits["max_context_tokens"],
            self.limits["max_tokens"],
            policy_version,
            directory / "calls.jsonl",
            self.limits["model_timeout_s"],
            provenance=episode_provenance,
        )
        process = None
        try:
            await bridge.start()
            environment = dict(os.environ)
            for key in list(environment):
                if key.endswith(("_API_KEY", "_TOKEN")) or key.startswith("PRIME_"):
                    environment.pop(key)
            environment.update(
                KERNEL_ENDPOINT=bridge.endpoint,
                KERNEL_API_KEY=bridge.api_key,
                KERNEL_MODEL=self.tokenizer_path,
                KERNEL_LIMITS=json.dumps({**self.limits, "bridge_owns_admission": True}),
                KERNEL_FEEDBACK_LOG_DIR=str(directory.resolve()),
                KERNEL_PROVENANCE=json.dumps(
                    {
                        "execution_run_id": self.execution_run_id,
                        "group_id": f"{self.group_id}:{task['name']}",
                        "attempt_id": self.attempt_id,
                        "episode_id": directory.name,
                        "policy_version": policy_version,
                    }
                ),
            )
            with (directory / "program.log").open("wb") as log:
                process = await asyncio.create_subprocess_exec(
                    *self.program_command,
                    cwd=directory,
                    env=environment,
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
                await asyncio.wait_for(process.wait(), self.episode_timeout_s)
            if process.returncode or bridge.failures:
                raise RuntimeError(f"REPL transport/program failed; see {directory}")
            record = json.loads((directory / "submission.json").read_text())
            provenance = {**episode_provenance, "policy_version": policy_version}
            record.update(provenance)
            (directory / "submission.json").write_text(json.dumps(record))
            if self.defer_scoring:
                return {**provenance, "samples": bridge.samples, "reward": None, "record": record}
            reward, verification = await self.scorer(task, record)
            (directory / "score.json").write_text(
                json.dumps(
                    {
                        "reward": reward,
                        "verification": verification,
                        "root_calls": len(bridge.samples),
                        "child_calls": bridge.child_calls,
                        "policy_version": policy_version,
                    },
                    indent=2,
                )
            )
            return {
                **provenance,
                "samples": bridge.samples,
                "reward": reward,
                "record": record,
                "verification": verification,
            }
        except BaseException as error:
            (directory / "failure.json").write_text(
                json.dumps(
                    {
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "policy_version": policy_version,
                        "optimizer_update_allowed": False,
                    },
                    indent=2,
                )
            )
            raise
        finally:
            if process is not None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
            await bridge.stop()

    async def generate(self, input_batch: dict[str, Any]) -> dict[str, Any]:
        trajectory_ids = input_batch.get("trajectory_ids")
        tasks = input_batch.get("env_extras")
        metadata = input_batch.get("batch_metadata")
        if not trajectory_ids or not tasks or metadata is None:
            raise ValueError("SkyRL trajectory IDs, task data and policy version are required")
        if len(trajectory_ids) != 4 or len(tasks) != 4 or len(input_batch["prompts"]) != 4:
            raise ValueError("The L40S baseline uses one prompt with four trajectories")
        if len({item.instance_id for item in trajectory_ids}) != 1:
            raise ValueError("All four trajectories must belong to one GRPO prompt")
        if len({item.repetition_id for item in trajectory_ids}) != 4:
            raise ValueError("Trajectory repetition IDs must be distinct")
        if any(task != tasks[0] for task in tasks):
            raise ValueError("Each repetition must receive the same task")
        if metadata.training_phase != "train":
            raise ValueError("Evaluation needs a separate unfiltered evaluation adapter")
        batch_dir = self.output_dir / f"step-{metadata.global_step}-{uuid.uuid4().hex[:8]}"
        batch_dir.mkdir(parents=True)
        attempts = []
        for attempt in range(self.max_sample_batches):
            task = (
                tasks[0]
                if attempt == 0
                else self.replacement_tasks[
                    (metadata.global_step + attempt - 1) % len(self.replacement_tasks)
                ]
            )
            ids = (
                trajectory_ids
                if attempt == 0
                else [
                    dataclasses.replace(
                        item, instance_id=f"{item.instance_id}:replacement-{attempt}"
                    )
                    for item in trajectory_ids
                ]
            )
            jobs = [
                asyncio.create_task(
                    self.run_episode(
                        task,
                        batch_dir / f"group-{attempt}-trajectory-{index}",
                        metadata.global_step,
                    )
                )
                for index in range(4)
            ]
            try:
                episodes = await asyncio.gather(*jobs)
            finally:
                for job in jobs:
                    if not job.done():
                        job.cancel()
                await asyncio.gather(*jobs, return_exceptions=True)
            rewards = [episode["reward"] for episode in episodes]
            admitted = has_signal(rewards)
            attempts.append({"task": task["name"], "rewards": rewards, "admitted": admitted})
            (batch_dir / "groups.json").write_text(json.dumps(attempts, indent=2))
            if admitted:
                output = stepwise_output(episodes, ids)
                output["rollout_metrics"] = {
                    "kernel/groups_sampled": len(attempts),
                    "kernel/zero_variance_groups": len(attempts) - 1,
                }
                return output
            if not self.replacement_tasks:
                break
        raise RuntimeError(
            "No learning signal within the bounded sampling budget; no update allowed"
        )
