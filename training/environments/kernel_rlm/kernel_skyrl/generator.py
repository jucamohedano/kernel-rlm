"""SkyRL batch handling around the shared kernel episode runner."""

import asyncio
import dataclasses
import json
import uuid
from pathlib import Path
from typing import Any

from kernel_rollout.contract import has_signal
from kernel_rollout.episode import EpisodeRequest, EpisodeRunner
from kernel_rollout.scoring import score_submission

from kernel_skyrl.contract import stepwise_output


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
        self.output_dir = Path(output_dir)
        self.replacement_tasks = replacement_tasks or []
        self.max_sample_batches = max_sample_batches
        self.runner = EpisodeRunner(
            client,
            adapter_name,
            tokenizer_path,
            limits=limits,
            episode_timeout_s=episode_timeout_s,
            scorer=scorer,
            program_command=program_command,
            defer_scoring=defer_scoring,
            execution_run_id=execution_run_id,
            group_id=group_id,
            attempt_id=attempt_id,
        )

    async def run_episode(self, task: dict, directory: Path, policy_version: int) -> dict:
        return await self.runner.run(EpisodeRequest(task, directory, policy_version))

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
