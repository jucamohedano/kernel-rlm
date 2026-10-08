"""CPU-only end-to-end V1 lifecycle check against prime's pinned Verifiers."""

import json
from datetime import UTC, datetime
from pathlib import Path

import modal

from .kernel_smoke import APP_ROOT
from .rollout_smoke import rollout_image, traces

VERIFIERS_REVISION = "b2e4e8157783b2c0dffc7821044c87f29f1c3ccf"
image = (
    (
        rollout_image.apt_install("git")
        .pip_install(f"git+https://github.com/PrimeIntellect-ai/verifiers.git@{VERIFIERS_REVISION}")
        .add_local_dir(APP_ROOT / "kernel_rlm_v1", "/root/kernel_rlm_v1", copy=True)
    )
    if modal.is_local()
    else None
)
app = modal.App("kernel-rlm-v1-check")


@app.function(
    image=image,
    volumes={"/traces": traces},
    cpu=1,
    memory=4096,
    timeout=180,
    retries=0,
    max_containers=1,
)
async def check() -> dict:
    import os
    import time
    from unittest.mock import patch

    import verifiers.v1 as vf
    from aiohttp import web
    from kernel_rlm_v1 import taskset as scoring
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.configs.client import EvalClientConfig
    from verifiers.v1.envs.single_agent.env import SingleAgentEnvConfig
    from verifiers.v1.interception import InterceptionServerConfig

    requests = []
    submission = Path("/opt/kernel_rlm/submission.py").read_text()
    replies = [
        f"```repl\nprint(context)\ncandidate = {submission!r}\nfeedback = cuda_feedback(candidate)\n```",
        f"```repl\nanswer['content'] = {submission!r}\nanswer['ready'] = True\n```",
    ]

    async def respond(request: web.Request) -> web.Response:
        body = await request.json()
        index = len(requests)
        requests.append(body)
        if index >= len(replies):
            raise web.HTTPBadRequest(text="Unexpected extra model request")
        if index == 1:
            assert "REPL output:" in json.dumps(body["messages"])
            assert "CUDA feedback:" in json.dumps(body["messages"])
            assert "class Model" in json.dumps(body["messages"])
        return web.json_response(
            {
                "id": f"fixture-{index}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": replies[index]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    server = web.Application()
    server.router.add_post("/v1/chat/completions", respond)
    runner = web.AppRunner(server)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    os.environ["V1_FIXTURE_API_KEY"] = "fixture-not-a-secret"
    config = SingleAgentEnvConfig.model_validate(
        {
            "taskset": {"id": "kernel_rlm_v1", "manifest_path": "/opt/kernel_rlm/vector_add.jsonl"},
            "agent": {
                "harness": {"id": "kernel_rlm_v1", "max_turns": 2},
                "runtime": {"type": "subprocess"},
                "max_turns": 2,
                "timeout": {"rollout": 60, "scoring": 30},
            },
            "interception": InterceptionServerConfig().model_dump(),
            "timeout": {"episode": 100},
        }
    )
    environment = vf.load_environment(config)
    task = next(iter(vf.load_taskset(config.taskset)))
    context = ModelContext(
        model="fixture",
        client=EvalClientConfig(
            base_url=f"http://127.0.0.1:{runner.addresses[0][1]}/v1",
            api_key_var="V1_FIXTURE_API_KEY",
        ),
    )
    fixture = {"compiled": True, "extension_used": True, "correct": True, "speedup": 1.0}
    directory = Path("/traces/v1-check") / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with patch.object(scoring, "verify_cuda", return_value=fixture) as verifier:
            async with environment.serving():
                episode = await environment.run_episode(task, context)
            (directory / "episode.json").write_text(episode.model_dump_json(indent=2))
            (directory / "requests.json").write_text(json.dumps(requests, indent=2))
            await traces.commit.aio()
            assert episode.ok, episode.model_dump_json()
            assert len(episode.traces) == 1
            trace = episode.traces[0]
            assert len(requests) == 2 and trace.num_turns == 2
            assert verifier.call_count == 1
            assert trace.info["kernel_submission"] == submission.strip()
            assert trace.info["repl_calls"] == 2
            assert abs(trace.reward - 0.8) < 1e-6
            return {
                "status": "passed",
                "scope": "V1 eval lifecycle; scripted model and mocked verifier",
                "verifiers_revision": VERIFIERS_REVISION,
                "model_calls": len(requests),
                "verifier_calls": verifier.call_count,
                "reward": trace.reward,
                "trace_directory": str(directory),
                "gpu_allocated": False,
            }
    finally:
        await runner.cleanup()


@app.local_entrypoint()
def main() -> None:
    result = check.remote()
    directory = Path("logs/v1-check")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "latest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
