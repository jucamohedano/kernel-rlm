"""Real RLM rollout collection through the existing exact-token REPL bridge."""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import aiohttp
from aiohttp import web
from kernel_rlm.verifier_slot import INFERENCE_ADMISSION_TIMEOUT_S, inference_request_slot
from kernel_skyrl.contract import exact_sample, has_signal
from kernel_skyrl.preflight import load_tasks

from kernel_trl.context import (
    DEFAULT_CONTEXT_TOKENS,
    LEGACY_CONTEXT_TOKENS,
    validate_context_tokens,
)
from kernel_trl.scoring import capture_generation
from kernel_trl.turn_credit import training_rows

DEFAULT_MODEL_TIMEOUT_SECONDS = 150
KernelGenerator = None


class InferenceClient:
    """Transport REPL-admitted requests without acquiring a second reader lock."""

    def __init__(self, session, endpoint, trace, context_tokens=LEGACY_CONTEXT_TOKENS):
        self.session = session
        self.endpoint = endpoint
        self.trace = trace
        self.context_tokens = validate_context_tokens(context_tokens)

    async def chat_completion(self, payload):
        return await self.request_completion(payload)

    async def request_completion(self, payload):
        body = dict(payload["json"])
        body.pop("session_id", None)
        token_request = {
            key: body[key]
            for key in ("model", "messages", "tools", "chat_template_kwargs")
            if key in body
        }
        async with self.session.post(
            self.endpoint + "/tokenize", json=token_request
        ) as token_response:
            token_response.raise_for_status()
            tokenized = await token_response.json()
        requested = body["max_tokens"]
        body["max_tokens"] = min(requested, self.context_tokens - tokenized["count"])
        if body["max_tokens"] < 128 and requested >= 128:
            raise ValueError("Insufficient exact context headroom for a useful reply")
        async with self.session.post(self.endpoint + "/v1/chat/completions", json=body) as response:
            data = await response.json()
            with self.trace.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "request": body,
                            "response": data,
                            "http_status": response.status,
                            "requested_max_tokens": requested,
                            "tokenized_count": tokenized["count"],
                        }
                    )
                    + "\n"
                )
            if response.status != 200:
                raise RuntimeError(f"Inference HTTP {response.status}: {data}")
            if data.get("prompt_token_ids") != tokenized["tokens"]:
                raise ValueError("Chat and tokenize endpoints produced different prompt IDs")
            return data

    async def finish_session(self, session_id):
        return None


def inference_admin_app(session, endpoint):
    async def pause(request):
        async with session.post(endpoint + "/sleep?level=1") as response:
            response.raise_for_status()
        return web.json_response({"sleeping": True})

    async def resume(request):
        async with session.post(endpoint + "/wake_up") as response:
            response.raise_for_status()
        return web.json_response({"sleeping": False})

    admin = web.Application()
    admin.router.add_post("/pause", pause)
    admin.router.add_post("/resume", resume)
    return admin


async def collect(arguments):
    generator_class = KernelGenerator
    if generator_class is None:
        from kernel_skyrl.generator import KernelGenerator as generator_class

    directory = arguments.output_dir
    directory.mkdir(parents=True, exist_ok=False)
    model_timeout = getattr(arguments, "model_timeout_s", DEFAULT_MODEL_TIMEOUT_SECONDS)
    request_timeout = getattr(arguments, "inference_http_timeout_s", model_timeout + 30)
    timeout = aiohttp.ClientTimeout(total=request_timeout)
    context_tokens = validate_context_tokens(
        getattr(arguments, "context_tokens", LEGACY_CONTEXT_TOKENS)
    )
    async with aiohttp.ClientSession(timeout=timeout) as session:
        client = InferenceClient(
            session, arguments.endpoint, directory / "requests.jsonl", context_tokens
        )
        started = time.monotonic()
        smoke_request = {
            "model": arguments.adapter_name,
            "messages": [
                {
                    "role": "user",
                    "content": "Write the CUDA expression for a thread's global 1D index.",
                }
            ],
            "max_tokens": 32,
            "temperature": 1.0,
            "logprobs": True,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        async with inference_request_slot(INFERENCE_ADMISSION_TIMEOUT_S):
            smoke = await client.chat_completion({"json": smoke_request})
        sample = exact_sample(smoke, context_tokens)
        (directory / "inference-smoke.json").write_text(
            json.dumps(
                {
                    "request": smoke_request,
                    "response": smoke,
                    "elapsed_seconds": time.monotonic() - started,
                    "sampled_tokens": len(sample.response_ids),
                },
                indent=2,
            )
        )

        admin = inference_admin_app(session, arguments.endpoint)
        runner = web.AppRunner(admin, shutdown_timeout=1)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 8016).start()
        tracker = None
        if getattr(arguments, "report_to", "none") == "trackio":
            import trackio

            tracker = trackio
            tracker.init(
                project="laguna-kernel-rlm",
                name=f"{directory.parents[3].name}-{directory.parents[1].name}-rollouts",
                space_id=None,
                auto_log_gpu=False,
                auto_log_cpu=False,
                embed=False,
                config={"policy_version": arguments.policy_version, "rollouts": arguments.rollouts},
            )
        generator = generator_class(
            client,
            arguments.adapter_name,
            str(arguments.checkpoint),
            directory,
            episode_timeout_s=getattr(arguments, "episode_timeout_s", 700),
            limits={
                "max_context_tokens": context_tokens,
                "model_timeout_s": getattr(
                    arguments, "model_timeout_s", DEFAULT_MODEL_TIMEOUT_SECONDS
                ),
            },
            defer_scoring=getattr(arguments, "defer_scoring", False),
        )
        tasks = {task["name"]: task for task in load_tasks(arguments.dataset)}
        attempts = []
        completed_episodes = 0
        try:
            for group_index, task_id in enumerate(arguments.task_ids):
                task = tasks[task_id]
                slots = asyncio.Semaphore(4)

                async def run_episode(index, episode_task, group_number, episode_slots):
                    nonlocal completed_episodes
                    async with episode_slots:
                        episode = await generator.run_episode(
                            episode_task,
                            directory / f"group-{group_number}-trajectory-{index}",
                            arguments.policy_version,
                        )
                        completed_episodes += 1
                        if tracker is not None:
                            tracker.log(
                                {
                                    "rollout/completed": completed_episodes,
                                    **(
                                        {"rollout/reward": episode["reward"]}
                                        if episode["reward"] is not None
                                        else {}
                                    ),
                                    "rollout/root_turns": len(episode["samples"]),
                                    "verifier/infrastructure_error": int(
                                        episode.get("verification", {}).get("verification_status")
                                        == "infrastructure_error"
                                    ),
                                }
                            )
                        return episode

                jobs = [
                    asyncio.create_task(run_episode(index, task, group_index, slots))
                    for index in range(arguments.rollouts)
                ]
                try:
                    episodes = await asyncio.gather(*jobs)
                finally:
                    for job in jobs:
                        if not job.done():
                            job.cancel()
                    await asyncio.gather(*jobs, return_exceptions=True)
                if getattr(arguments, "defer_scoring", False):
                    capture_generation(
                        directory,
                        task_id,
                        arguments.policy_version,
                        arguments.rollouts,
                        context_tokens=context_tokens,
                    )
                    print(
                        json.dumps(
                            {"status": "generated", "task": task_id, "trajectories": len(episodes)}
                        ),
                        flush=True,
                    )
                    return
                rewards = [episode["reward"] for episode in episodes]
                infrastructure_errors = [
                    {"trajectory": index, "verification": episode["verification"]}
                    for index, episode in enumerate(episodes)
                    if episode.get("verification", {}).get("verification_status")
                    in {"infrastructure_error", "runner_crash", "invalid_runner_output", "timeout"}
                ]
                admitted = not infrastructure_errors and has_signal(rewards) and max(rewards) > 0.1
                attempts.append(
                    {
                        "task": task_id,
                        "rewards": rewards,
                        "admitted": admitted,
                        "policy_version": arguments.policy_version,
                        "reason": "admitted"
                        if admitted
                        else "infrastructure_error"
                        if infrastructure_errors
                        else "weak_signal"
                        if max(rewards) <= 0.1
                        else "zero_variance",
                        "infrastructure_errors": infrastructure_errors,
                    }
                )
                (directory / "groups.json").write_text(json.dumps(attempts, indent=2))
                if infrastructure_errors:
                    raise RuntimeError(
                        "Verifier infrastructure failed; preserve the group and repair before retry"
                    )
                if max(rewards) <= 0.1:
                    raise RuntimeError(
                        "No reward above 0.1; inspect saved trajectories before continuing"
                    )
                if admitted:
                    group = training_rows(episodes, arguments.policy_version, context_tokens)
                    group["task"] = task_id
                    (directory / "training-group.json").write_text(json.dumps(group, indent=2))
                    print(
                        json.dumps(
                            {
                                "status": "admitted",
                                "rewards": rewards,
                                "root_turns": group["real_rows"],
                                "trainable_tokens": group["trainable_tokens"],
                            }
                        ),
                        flush=True,
                    )
                    return
            raise RuntimeError(
                "All bounded groups had identical terminal rewards; no update allowed"
            )
        finally:
            if tracker is not None:
                tracker.finish()
            await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8006")
    parser.add_argument("--context-tokens", type=int, default=DEFAULT_CONTEXT_TOKENS)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, choices=(4, 8), default=4)
    parser.add_argument("--model-timeout-s", type=float, default=DEFAULT_MODEL_TIMEOUT_SECONDS)
    parser.add_argument("--inference-http-timeout-s", type=float)
    parser.add_argument("--episode-timeout-s", type=float, default=700)
    parser.add_argument("--group-timeout-s", type=float)
    parser.add_argument("--policy-version", type=int, default=0)
    parser.add_argument("--adapter-name", default="kernel-base")
    parser.add_argument("--report-to", choices=("trackio", "none"), default="trackio")
    parser.add_argument("--defer-scoring", action="store_true")
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=["lvl1-seed42-0038", "lvl1-seed42-0011", "lvl1-seed42-0044"],
    )
    arguments = parser.parse_args()
    if arguments.defer_scoring and len(arguments.task_ids) != 1:
        raise ValueError("Deferred scoring requires one complete task group")
    if arguments.policy_version < 0 or (
        arguments.policy_version > 0 and arguments.adapter_name == "kernel-base"
    ):
        raise ValueError("A nonzero policy requires an explicitly loaded trained adapter")
    if len(arguments.task_ids) > 3 or not arguments.endpoint.startswith("http://127.0.0.1:"):
        raise ValueError("Use loopback inference and at most three groups")
    if arguments.inference_http_timeout_s is None:
        arguments.inference_http_timeout_s = arguments.model_timeout_s + 30
    if (
        arguments.model_timeout_s <= 0
        or arguments.inference_http_timeout_s <= 0
        or arguments.episode_timeout_s <= 0
    ):
        raise ValueError("Model and trajectory timeouts must be positive")
    if arguments.group_timeout_s is None:
        arguments.group_timeout_s = 900 * (arguments.rollouts // 4)
    if arguments.group_timeout_s <= 0:
        raise ValueError("Group timeout must be positive")
    os.environ["KERNEL_INFERENCE_ADMIN_URL"] = "http://127.0.0.1:8016"
    asyncio.run(asyncio.wait_for(collect(arguments), timeout=arguments.group_timeout_s))


if __name__ == "__main__":
    main()
