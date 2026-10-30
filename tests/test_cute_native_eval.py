"""Scripted end-to-end native CuTe orchestration; no real GPU receipt or GPU execution."""

import hashlib
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from aiohttp import web
from cute_rlm.config import DATASET_ID, DATASET_REVISION, REQUIRED_GPU_CHECKS, EvalConfig
from cute_rlm.evaluate import evaluate_prepared, isolated_backend_factory, prepare
from cute_rlm.prompts import CUTE_SYSTEM_PROMPT, first_user_message


@pytest.mark.asyncio
async def test_cpu_scripted_native_tree_records_caps_logprobs_and_verification(
    tmp_path, monkeypatch
):
    project = tmp_path / "p"
    project.mkdir()
    repo = Path(__file__).parents[1]
    for relative in ("training/src/rlm_train", "training/environments/cute_rlm/cute_rlm"):
        shutil.copytree(
            repo / relative, project / relative, ignore=shutil.ignore_patterns("__pycache__")
        )
    for relative in (
        "training/environments/kernel_rlm/kernel_factory/__init__.py",
        "pyproject.toml",
        "uv.lock",
        "training/environments/cute_rlm/pyproject.toml",
    ):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / relative, path)
    source = "# reference specification only"
    task = {
        "task_id": "task",
        "reference_code": source,
        "source": {
            "repo": DATASET_ID,
            "revision": DATASET_REVISION,
            "file": "pool",
            "row": 0,
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
        },
    }
    (project / "tasks.jsonl").write_text(json.dumps(task) + "\n")
    (project / "cute.md").write_text("CuTe documentation fixture")
    (project / "docs.json").write_text(
        json.dumps(
            {
                "repo": "NVIDIA/cutlass",
                "revision": "a" * 40,
                "files": [
                    {
                        "id": "cute",
                        "path": "cute.md",
                        "sha256": hashlib.sha256((project / "cute.md").read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )
    config = EvalConfig(
        task_manifest="tasks.jsonl",
        docs_manifest="docs.json",
        seed=7,
        task_count=1,
        verification={
            "atol": 0,
            "rtol": 0,
            "seed": 7,
            "mode": "preserve",
            "correctness_runs": 1,
            "warmup_runs": 0,
            "timed_runs": 1,
            "timeout_s": 30,
        },
    )
    # The receipt is a CPU fixture, never written as a real GPU-readiness result.
    with (
        patch("kernel_factory.git_state", return_value={"commit": "cpu-fixture"}),
        patch("kernel_factory.subprocess.check_output", return_value=b""),
    ):
        readiness = prepare(project, config, "cpu-fixture", tmp_path / "runs")
    receipt = {
        "config_sha256": config.sha256(),
        "manifest_sha256": readiness["manifest_sha256"],
        "observed_context_tokens": 262144,
        "server_identity": {"tensor_parallel_size": config.tensor_parallel_size},
        "hardware": {
            "fixture": True,
            "gpu_model": config.gpu_model,
            "gpu_compute_capability": config.gpu_compute_capability,
        },
        "software": {"fixture": True, "cuda_toolkit_version": config.cuda_toolkit_version},
        "checks": {
            key: {"status": "passed", "evidence": "CPU_FIXTURE_NOT_GPU"}
            for key in REQUIRED_GPU_CHECKS
        },
    }
    payloads = []

    async def completion(request):
        payload = await request.json()
        payloads.append(payload)
        user = next(
            message["content"] for message in payload["messages"] if message["role"] == "user"
        )
        assert payload["messages"][0]["content"] == CUTE_SYSTEM_PROMPT
        if user == "child":
            code = (
                "assert read_doc('cute')['excerpt']; "
                "assert compile_candidate('child candidate')['status']=='compiled'; "
                "answer['content']='child-ok'; answer['ready']=True"
            )
        else:
            assert user == first_user_message(task, config)
            code = (
                "assert context['reference_code'] == '# reference specification only'; "
                "assert read_doc('cute')['excerpt']; import os; assert 'DO_NOT_INHERIT_API_KEY' not in os.environ; "
                "assert compile_candidate('bad candidate')['status']=='compile_error'; "
                "assert compile_candidate('revision 1')['status']=='compiled'; "
                "assert compile_candidate('revision 2')['status']=='compiled'; "
                "reply=rlm_query('child'); "
                "assert reply=='child-ok'; answer['content']='class ModelNew: pass'; answer['ready']=True"
            )
        return web.json_response(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "reasoning_content": "scripted",
                            "tool_calls": [
                                {
                                    "id": "repl",
                                    "function": {
                                        "name": "repl",
                                        "arguments": json.dumps({"code": code}),
                                    },
                                }
                            ],
                        },
                        "logprobs": {"content": [{"token": "token_id:1", "logprob": -0.1}]},
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            }
        )

    async def tokenize(request):
        data = await request.json()
        assert data["chat_template_kwargs"] == {"enable_thinking": True}
        return web.json_response({"count": 20, "max_model_len": 262144})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completion)
    app.router.add_post("/tokenize", tokenize)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    async def fake_verify(task, candidate, settings, directory, **kwargs):
        assert candidate == "class ModelNew: pass"
        return {"status": "launch_unverified", "correct": True, "speedup": None}

    feedback_calls = []

    class ScriptedFeedback:
        def __init__(self, task, settings, directory, *, worker_command):
            assert task["task_id"] == "task"
            assert settings == config.verification
            assert worker_command == ["scripted-only"]

        async def __call__(self, session_id, source):
            feedback_calls.append((session_id, source))
            return {
                "status": "compile_error" if source == "bad candidate" else "compiled",
                "detail": "scripted compiler diagnostic; no GPU execution",
            }

    # Dispatch must use the frozen rendered messages, not live module defaults.
    monkeypatch.setattr("cute_rlm.evaluate.CUTE_SYSTEM_PROMPT", "changed after preparation")

    def forbidden_render(*args):
        pytest.fail("execution must not re-render captured first-user messages")

    monkeypatch.setattr("cute_rlm.evaluate.first_user_message", forbidden_render)
    monkeypatch.setattr("cute_rlm.feedback.CompileFeedback", ScriptedFeedback)
    monkeypatch.setattr("cute_rlm.verifier.verify_submission", fake_verify)
    monkeypatch.setenv("DO_NOT_INHERIT_API_KEY", "secret")
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}/v1/", timeout=10) as client:
            factory = isolated_backend_factory(config, sys.executable)
            report = await evaluate_prepared(
                tmp_path / "runs/cpu-fixture",
                client=client,
                worker_command=["scripted-only"],
                backend_factory=factory,
                gpu_receipt=receipt,
                cute_observer=lambda *args: None,
            )
    finally:
        await runner.cleanup()
    assert len(feedback_calls) == 4
    assert len({owner for owner, _ in feedback_calls}) == 2
    assert len(payloads) == 2
    assert all(payload["max_tokens"] > 8192 for payload in payloads)
    assert all(payload["logprobs"] is True and payload["top_logprobs"] == 0 for payload in payloads)
    assert report["usage"]["root"]["completion_tokens"]["sum"] == 10
    assert report["usage"]["child"]["completion_tokens"]["sum"] == 10
    record = json.loads((tmp_path / "runs/cpu-fixture/task-00000.json").read_text())
    assert record["verification"]["status"] == "launch_unverified"
    assert not record["correct"]
    assert (
        record["child_sessions"][0]["events"][0]["token_limits"]["context_tokens"]["limit"]
        == 262144
    )
