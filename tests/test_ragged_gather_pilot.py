import asyncio
import json
import os
from pathlib import Path

import pytest

from training.environments.kernel_rlm.research.ragged_gather_reduce import pilot, task


@pytest.mark.parametrize(
    "endpoint",
    ["http://localhost:1234/v1", "https://127.0.0.1:1234/v1", "http://127.0.0.1:1234/v1?a=1"],
)
def test_only_private_literal_loopback_is_accepted(endpoint):
    with pytest.raises(ValueError, match="private"):
        pilot.require_loopback(endpoint)
    assert pilot.require_loopback("http://127.0.0.1:1234/v1") == "http://127.0.0.1:1234/v1"


def test_approved_shape_cannot_be_replaced_by_preflight():
    with pytest.raises(ValueError, match="live author-control"):
        pilot.require_author_controls({"status": "offline_preflight"})


def test_held_exact_source_and_visible_projection(tmp_path):
    state = {
        "stop_condition": "has_final_answer",
        "rlm_final_answer": "// whitespace preserved \n\n",
        "trajectory": [
            {
                "prompt": [{"role": "user", "content": "real prompt"}],
                "completion": [
                    {
                        "role": "assistant",
                        "content": "<think>private</think>visible",
                        "reasoning_content": "not exported",
                    }
                ],
            }
        ],
    }
    record = pilot.seal_state(tmp_path, state)
    assert (tmp_path / "submission.cu").read_bytes() == b"// whitespace preserved \n\n"
    assert record["status"] == "submitted_held_for_review"
    assert record["kernel_verdict"] is None
    assert "private" not in (tmp_path / "result.json").read_text()
    assert record["root_requests"][0]["completion"] == [{"role": "assistant", "content": "visible"}]


def test_no_native_answer_is_not_a_submission(tmp_path):
    record = pilot.seal_state(tmp_path, {"final_answer": "not native"})
    assert record["status"] == "no_submission"
    assert not (tmp_path / "submission.cu").exists()


def test_scripted_full_native_rollout(tmp_path, monkeypatch):
    checkout = os.environ.get("HARNESS_CG_CHECKOUT")
    if not checkout:
        pytest.skip("enable the pinned Harness-CG integration checks")
    vf = pytest.importorskip("verifiers")
    pytest.importorskip("datasets")
    upstream = pytest.importorskip("rlm_train.env")
    import httpx
    from openai import AsyncOpenAI

    assert Path(upstream.__file__).resolve().is_relative_to(Path(checkout).resolve())
    runtime = os.environ.get("RAGGED_PYTHON_RUNTIME")
    if not runtime:
        pytest.skip("set RAGGED_PYTHON_RUNTIME for the end-to-end sandbox test")
    from training.environments.kernel_rlm.research.ragged_gather_reduce.sandbox import (
        SandboxedReplBackend,
    )

    worker = pilot.require_pinned_harness(Path(checkout))
    backend = SandboxedReplBackend(worker, Path(runtime), max_sub_calls=1, wall_seconds=20)
    monkeypatch.setenv("RLM_TRAIN_SUBCALL_TIMEOUT_S", "4")
    root_actions = iter(
        (
            "```repl\nseen = 'owner entry' in context; print(seen, llm_query('stub-child'))\n```",
            '```repl\nanswer["content"] = "// pilot marker\\n" if seen else ""\n'
            'answer["ready"] = True\n```',
        )
    )
    sent = []

    def completion(request):
        assert request.url.path == "/v1/chat/completions"
        sent.append(json.loads(request.content))
        is_child = sent[-1]["messages"] == [{"role": "user", "content": "stub-child"}]
        content = "child reply" if is_child else next(root_actions)
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": "offline-script",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": content},
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            },
        )

    async def check():
        env = task.native_environment(
            backend_factory=lambda: backend,
            max_iterations=3,
            max_sub_calls=1,
            max_rollout_seconds=10,
            sub_sampling_args=pilot.sampling_args(),
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(completion)) as transport:
            async with AsyncOpenAI(
                base_url="http://127.0.0.1:1/v1",
                api_key="unused",
                max_retries=0,
                http_client=transport,
            ) as oai:
                return await pilot.generate_one(
                    tmp_path / "result",
                    env,
                    vf.OpenAIChatCompletionsClient(oai),
                    model="offline-script",
                )

    result = asyncio.run(check())
    assert len(sent) == 3
    assert all(call["chat_template_kwargs"] == {"enable_thinking": False} for call in sent)
    assert result["upstream_subcall_count"] == 1
    assert result["upstream_subcalls"][0]["response"] == "child reply"
    assert result["status"] == "submitted_held_for_review"
    assert (tmp_path / "result/submission.cu").read_bytes() == b"// pilot marker\n"


def test_receipt_requires_exact_input_and_capture():
    from training.environments.kernel_rlm.research.ragged_gather_reduce import vm_smoke

    receipt = {"schema": vm_smoke.SCHEMA, "mode": "live", "status": "author_controls_passed"}
    receipt["sources"] = {
        name: vm_smoke.sha256((task.HERE / name).read_bytes())
        for name in (*vm_smoke.SOURCE_SHA256, "reference.py", "score.py")
    }
    receipt["builds"] = {
        name: {"status": "built", "sha256": name} for name in ("control", "wrong_offset")
    }
    receipt["cases"] = [
        {
            "control": name,
            "fixture": fixture.name,
            "observed": expected,
            "input_sha256": vm_smoke.sha256(fixture.to_wire().encode()),
            "artifact_sha256": name,
            "process": {"status": "exited", "returncode": 0},
        }
        for name, fixture, expected in vm_smoke.controls_plan()
    ]
    pilot.require_author_controls(receipt)
    receipt["cases"][-1]["process"]["returncode"] = 3
    with pytest.raises(ValueError, match="failed device process"):
        pilot.require_author_controls(receipt)


def test_serving_topology_does_not_permit_one_l40s():
    manifest = {
        "model": task.MODEL_ID,
        "revision": task.MODEL_REVISION,
        "served_model": pilot.SERVED_MODEL,
        "dtype": "bfloat16",
        "gpu": "RTX PRO 6000 Blackwell",
        "tensor_parallel_size": 1,
        "max_model_len": 32768,
        "endpoint": "http://127.0.0.1:8800/v1",
    }
    assert pilot.require_serving(manifest) == manifest["endpoint"]
    manifest["gpu"] = "L40S"
    with pytest.raises(ValueError, match="topology"):
        pilot.require_serving(manifest)
    manifest["tensor_parallel_size"] = 2
    assert pilot.require_serving(manifest) == manifest["endpoint"]
