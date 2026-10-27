"""Execute one unchanged REPL episode with injected inference and scoring."""

import asyncio
import json
import os
import signal
import sys
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NotRequired, TypedDict

from kernel_rollout.bridge import CompletionBridge
from kernel_rollout.contract import RootSample
from kernel_rollout.scoring import score_submission


@dataclass(frozen=True)
class EpisodeRequest:
    task: dict[str, Any]
    directory: Path
    policy_version: int


class EpisodeResult(TypedDict):
    execution_run_id: str | None
    group_id: str | None
    attempt_id: str | None
    episode_id: str
    policy_version: int
    samples: list[RootSample]
    reward: float | None
    record: dict[str, Any]
    verification: NotRequired[dict[str, Any]]


Scorer = Callable[[dict, dict], Awaitable[tuple[float, dict]]]


DEFAULT_LIMITS = {
    "max_turns": 8,
    "max_tokens": 4096,
    "max_context_tokens": 12288,
    "model_timeout_s": 150,
    "enable_thinking": True,
    "enable_subcall": True,
    "max_subcalls": 2,
    "child_reply_tokens": 4096,
    "child_total_tokens": 4096,
    "child_prompt_tokens": 2048,
    "subcall_probe": False,
}


class EpisodeRunner:
    def __init__(
        self,
        client: Any,
        adapter_name: str,
        tokenizer_path: str,
        limits: dict | None = None,
        episode_timeout_s: float = 720,
        scorer: Scorer = score_submission,
        program_command: list[str] | None = None,
        defer_scoring: bool = False,
        execution_run_id: str | None = None,
        group_id: str | None = None,
        attempt_id: str | None = None,
    ):
        if episode_timeout_s <= 0:
            raise ValueError("Use a positive episode timeout")
        self.client = client
        self.adapter_name = adapter_name
        self.tokenizer_path = tokenizer_path
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

    async def run(self, request: EpisodeRequest) -> EpisodeResult:
        task, directory, policy_version = request.task, request.directory, request.policy_version
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "task.json").write_text(json.dumps(task))
        episode_provenance = {
            "execution_run_id": self.execution_run_id,
            "group_id": f"{self.group_id}:{task['name']}" if self.group_id is not None else None,
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
            child_reply_limit=self.limits["child_reply_tokens"],
            child_total_limit=self.limits["child_total_tokens"],
            context_tokens=self.limits["max_context_tokens"],
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
                    {**episode_provenance, "policy_version": policy_version}
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
