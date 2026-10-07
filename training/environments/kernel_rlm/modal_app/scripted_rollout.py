"""Exercise the real rollout lifecycle with two scripted model replies."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch


async def run_scripted_rollout(*, cuda: bool, trace_path: Path) -> dict[str, Any]:
    from aiohttp import web
    from kernel_rlm import env as kernel_env
    from pydantic_core import to_json
    from verifiers.types import ClientConfig

    submission = Path("/opt/kernel_rlm/submission.py").read_text()
    requests: list[dict[str, Any]] = []
    replies = [
        "```repl\ninspected_context = context\nprint(context)\n```",
        "```repl\nassert inspected_context == context\n"
        f"answer['content'] = {submission!r}\nanswer['ready'] = True\n```",
    ]

    async def respond(request: web.Request) -> web.Response:
        body = await request.json()
        turn = len(requests)
        requests.append(body)
        if turn >= len(replies):
            raise web.HTTPBadRequest(text="Unexpected extra model request")
        if turn == 1:
            assert "REPL output:" in json.dumps(body["messages"])
            assert "class Model" in json.dumps(body["messages"])
        return web.json_response(
            {
                "id": f"scripted-{turn}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "scripted-kernel",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": replies[turn]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    mock_app = web.Application()
    mock_app.router.add_post("/v1/chat/completions", respond)
    runner = web.AppRunner(mock_app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    os.environ["KERNEL_SMOKE_KEY"] = "scripted-no-external-api"
    environment = kernel_env.load_environment(
        manifest_path="/opt/kernel_rlm/vector_add.jsonl", max_iterations=4
    )
    environment.timeout_seconds = 60
    fixture = {
        "compiled": True,
        "extension_used": True,
        "correct": True,
        "max_diff": 0.0,
        "speedup": 1.0,
        "ref_ms": 1.0,
        "kernel_ms": 1.0,
        "error": "",
    }
    verifier_options = {"wraps": kernel_env.verify_cuda} if cuda else {"return_value": fixture}
    trace: dict[str, Any] = {
        "mode": "cuda" if cuda else "cpu-preflight-fake-verifier",
        "model_requests": requests,
        "scripted_replies": replies,
        "trace_path": str(trace_path),
        "status": "running",
    }
    try:
        with patch.object(kernel_env, "verify_cuda", **verifier_options) as verifier:
            output = await environment.run_rollout(
                input=environment.dataset[0],
                client=ClientConfig(
                    api_base_url=f"http://127.0.0.1:{port}/v1",
                    api_key_var="KERNEL_SMOKE_KEY",
                    max_retries=0,
                    timeout=30,
                ),
                model="scripted-kernel",
                sampling_args={"max_tokens": 4096},
                max_retries=0,
                state_columns=[
                    "rlm_final_answer",
                    "rlm_iterations",
                    "rlm_repl_calls",
                    "kernel_verify",
                    "rlm_backend",
                ],
            )
            trace["output"] = output
            trace["verifier_calls"] = verifier.call_count
            assert verifier.call_count == 1, f"Verifier called {verifier.call_count} times"
        assert len(requests) == 2, f"Expected two model requests, got {len(requests)}"
        assert output["rlm_final_answer"] == submission
        assert output["rlm_iterations"] == 2
        assert output["rlm_repl_calls"] == 2
        assert output["rlm_backend"] is None, "Worker was not cleaned up"
        result = output["kernel_verify"]
        assert result["compiled"] and result["correct"], result
        assert result["speedup"] is not None and not result["error"], result
        expected_reward = 0.7 + 0.1 * min(max(result["speedup"], 0.0), 3.0)
        assert abs(output["reward"] - expected_reward) < 1e-8
        assert output["metrics"]["gated_reward"] == output["reward"]
        trace["status"] = "passed"
        return trace
    except Exception as error:
        trace["status"] = "failed"
        trace["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            await environment.teardown_rlm()
            await runner.cleanup()
        finally:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_bytes(to_json(trace, indent=2))


def run(*, cuda: bool, trace_path: Path) -> dict[str, Any]:
    from pydantic_core import to_json

    return json.loads(to_json(asyncio.run(run_scripted_rollout(cuda=cuda, trace_path=trace_path))))
