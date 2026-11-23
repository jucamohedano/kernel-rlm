import asyncio
import json
import os
from hashlib import sha256
from pathlib import Path

import pytest

from training.environments.kernel_rlm.research.blackwell_native_rollout import launcher

SOURCE = "// exact UTF-8: café 🚀\r\n\n"
DESCRIPTOR = ' { "abi":"example", "geometry":["preset"], "n":8, "bool":true, "nil":null }\n'
ANSWER = "\n" + json.dumps({"source": SOURCE, "descriptor": DESCRIPTOR}, ensure_ascii=False) + "\n"


def test_exact_transport_and_plain_types():
    result = launcher.parse_submission(ANSWER)
    assert result.answer == ANSWER.encode()
    assert result.source == SOURCE.encode()
    assert result.descriptor_text == DESCRIPTOR.encode()
    assert type(result.descriptor) is dict  # noqa: E721 - transport promises plain JSON types.
    assert type(result.descriptor["n"]) is int  # noqa: E721
    assert result.descriptor["bool"] is True
    assert result.descriptor["nil"] is None
    assert result.descriptor["geometry"] == ["preset"]


@pytest.mark.parametrize(
    ("state", "status", "reason"),
    [
        ({"final_answer": ANSWER}, "no_submission", "no_native_commit"),
        (
            {"rlm_final_answer": ANSWER, "rlm_default_answer": True},
            "no_submission",
            "fallback_answer",
        ),
        ({"error": TimeoutError(), "stop_condition": "timeout"}, "unavailable", "wall_timeout"),
        ({"rlm_final_answer": " \r\n"}, "no_submission", "empty_answer"),
        ({"rlm_final_answer": "```json\n" + ANSWER + "```"}, "invalid_submission", "fenced_answer"),
        ({"rlm_final_answer": ANSWER + "extra"}, "invalid_submission", "invalid_answer_json"),
        (
            {"rlm_final_answer": '{"source":"x","descriptor":"{}","path":"x"}'},
            "invalid_submission",
            "wrong_envelope_keys",
        ),
        (
            {"rlm_final_answer": '{"source":"x","descriptor":{}}'},
            "invalid_submission",
            "descriptor_not_string",
        ),
    ],
)
def test_failed_or_invalid_answers_do_not_export_artifacts(tmp_path, state, status, reason):
    receipt = launcher.seal_state(tmp_path, state, {})
    assert (receipt["status"], receipt["reason"]) == (status, reason)
    assert receipt["kernel_verdict"] is None
    assert sorted(p.name for p in tmp_path.iterdir()) == ["result.json"]


@pytest.mark.parametrize(
    ("source", "descriptor", "reason"),
    [
        ("\x00", "{}", "nul_in_source"),
        ("é" * (launcher.SOURCE_BYTES // 2 + 1), "{}", "source_too_large"),
        ("x", " " * launcher.DESCRIPTOR_BYTES + "{}", "descriptor_too_large"),
        ("x", '{"a":{"x":1,"x":2}}', "invalid_descriptor_json"),
        ("x", '{"n":1e999}', "invalid_descriptor_json"),
        ("x", '{"n":NaN}', "invalid_descriptor_json"),
        ("x", r'{"n":"\ud800"}', "invalid_descriptor_json"),
        ("x", "[]", "descriptor_not_object"),
    ],
)
def test_inner_transport_rejections(source, descriptor, reason):
    with pytest.raises(launcher.FormatError, match=reason):
        launcher.parse_submission(json.dumps({"source": source, "descriptor": descriptor}))


@pytest.mark.parametrize("child_calls", [0, 1])
def test_pinned_native_and_sandbox_with_scripted_http(tmp_path, monkeypatch, child_calls):
    checkout = os.environ.get("HARNESS_CG_CHECKOUT")
    runtime = os.environ.get("RAGGED_PYTHON_RUNTIME")
    if not (checkout and runtime):
        pytest.skip("set HARNESS_CG_CHECKOUT and RAGGED_PYTHON_RUNTIME for the pinned integration")
    vf = pytest.importorskip("verifiers")
    import httpx
    from openai import AsyncOpenAI

    monkeypatch.setenv("RLM_TRAIN_SUBCALL_TIMEOUT_S", "4")
    public_context = "synthetic public brief ☀\n"
    root_turns = iter(
        (
            "```repl\nseen = context == " + repr(public_context) + "\n"
            "child_reply = llm_query('stub-child'); print(seen)\n```",
            "<think>PRIVATE MARKER</think>```repl\n"
            f'answer["content"] = {ANSWER!r} if seen else ""\nanswer["ready"] = True\n```',
        )
    )
    requests = []

    def completion(request):
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        requests.append(body)
        child = body["messages"] == [{"role": "user", "content": "stub-child"}]
        content = "<think>PRIVATE MARKER</think>child reply" if child else next(root_turns)
        return httpx.Response(
            200,
            json={
                "id": "scripted",
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
        async with httpx.AsyncClient(transport=httpx.MockTransport(completion)) as transport:
            async with AsyncOpenAI(
                base_url="http://127.0.0.1:1/v1",
                api_key="unused",
                max_retries=0,
                http_client=transport,
            ) as oai:
                return await launcher.generate_one(
                    task_id="offline-example",
                    public_context=public_context,
                    budget=launcher.Budget(2, child_calls, 512, 10),
                    out_dir=tmp_path / "attempt",
                    harness_checkout=Path(checkout),
                    python_runtime=Path(runtime),
                    client=vf.OpenAIChatCompletionsClient(oai),
                    model="offline-script",
                )

    receipt = asyncio.run(check())
    assert receipt["status"] == "submitted_held_for_review", receipt
    assert receipt["kernel_verdict"] is None
    assert len(receipt["root_requests"]) == 2
    assert len(requests) == 2 + child_calls
    assert all(req.get("max_completion_tokens", req.get("max_tokens")) == 512 for req in requests)
    assert all(req["model"] == "offline-script" for req in requests)
    assert all(req["chat_template_kwargs"] == {"enable_thinking": False} for req in requests)
    assert (
        receipt["upstream_subcall_count"]
        == receipt["child_broker"]["reserved_prompts"]
        == child_calls
    )
    if child_calls:
        assert receipt["upstream_subcalls"][0]["prompt"] == "stub-child"
        assert receipt["upstream_subcalls"][0]["response"] == "child reply"
    attempt = tmp_path / "attempt"
    for name, text in {
        "answer.json": ANSWER,
        "submission.cu": SOURCE,
        "descriptor.json": DESCRIPTOR,
        "task-context.txt": public_context,
    }.items():
        assert (attempt / name).read_bytes() == text.encode()
        assert (attempt / name).stat().st_mode & 0o777 == 0o600
        if name != "task-context.txt":
            assert receipt["artifact_sha256"][name] == sha256(text.encode()).hexdigest()
    assert (attempt.stat().st_mode & 0o777) == 0o700
    assert "PRIVATE MARKER" not in (attempt / "result.json").read_text()
    intent = json.loads((attempt / "intent.json").read_bytes())
    assert intent["task_id"] == "offline-example"
    assert intent["context_sha256"] == sha256(public_context.encode()).hexdigest()
