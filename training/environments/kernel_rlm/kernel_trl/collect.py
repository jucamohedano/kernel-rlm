"""Real RLM rollout collection through the existing exact-token REPL bridge."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp
from aiohttp import web
from kernel_skyrl.contract import exact_sample, has_signal
from kernel_skyrl.generator import KernelGenerator
from kernel_skyrl.preflight import load_tasks

from kernel_trl.turn_credit import training_rows


class InferenceClient:
    def __init__(self, session, endpoint, trace):
        self.session = session
        self.endpoint = endpoint
        self.trace = trace

    async def chat_completion(self, payload):
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
        body["max_tokens"] = min(requested, 12288 - tokenized["count"])
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


async def collect(arguments):
    directory = arguments.output_dir
    directory.mkdir(parents=True, exist_ok=False)
    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        client = InferenceClient(session, arguments.endpoint, directory / "requests.jsonl")
        started = time.monotonic()
        smoke_request = {
            "model": "kernel-base",
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
        smoke = await client.chat_completion({"json": smoke_request})
        sample = exact_sample(smoke, 12288)
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

        async def pause(request):
            async with session.post(arguments.endpoint + "/sleep?level=1") as response:
                response.raise_for_status()
            return web.json_response({"sleeping": True})

        async def resume(request):
            async with session.post(arguments.endpoint + "/wake_up") as response:
                response.raise_for_status()
            return web.json_response({"sleeping": False})

        admin = web.Application()
        admin.router.add_post("/pause", pause)
        admin.router.add_post("/resume", resume)
        runner = web.AppRunner(admin, shutdown_timeout=1)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 8016).start()
        generator = KernelGenerator(
            client, "kernel-base", str(arguments.checkpoint), directory, episode_timeout_s=700
        )
        tasks = {task["name"]: task for task in load_tasks(arguments.dataset)}
        attempts = []
        try:
            for group_index, task_id in enumerate(arguments.task_ids):
                task = tasks[task_id]
                jobs = [
                    asyncio.create_task(
                        generator.run_episode(
                            task, directory / f"group-{group_index}-trajectory-{index}", 0
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
                attempts.append(
                    {"task": task_id, "rewards": rewards, "admitted": has_signal(rewards)}
                )
                (directory / "groups.json").write_text(json.dumps(attempts, indent=2))
                if has_signal(rewards):
                    group = training_rows(episodes)
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
            await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8006")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=["lvl1-seed42-0038", "lvl1-seed42-0011", "lvl1-seed42-0044"],
    )
    arguments = parser.parse_args()
    if len(arguments.task_ids) > 3 or not arguments.endpoint.startswith("http://127.0.0.1:"):
        raise ValueError("Use loopback inference and at most three groups")
    asyncio.run(asyncio.wait_for(collect(arguments), timeout=900))


if __name__ == "__main__":
    main()
