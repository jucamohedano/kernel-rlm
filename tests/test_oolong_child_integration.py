"""CPU-only end-to-end test for the native worker/proxy/child completion path."""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from aiohttp import web

ROOT = Path(__file__).parents[1]
TRAIN = ROOT / "training"
OOLONG = ROOT / "training/environments/oolong"
for entry in (str(TRAIN / "src"), str(OOLONG)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


@pytest.fixture(autouse=True)
def worker_import_path(monkeypatch):
    entries = [str(TRAIN / "src"), os.environ.get("PYTHONPATH", "")]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, entries)))


async def _run_case(run_dir: Path, socket_path: Path, *, child_mode="stop", max_model_calls=256):
    from oolong.native_eval import evaluate

    calls = []
    root_turn = 0

    async def completions(request):
        nonlocal root_turn
        payload = await request.json()
        calls.append(payload)
        messages = payload["messages"]
        if payload.get("tools"):
            root_turn += 1
            if root_turn == 1:
                code = (
                    "one = llm_query('alpha')\n"
                    "many = llm_query_batched(['beta', 'gamma'])\n"
                    "print(one, many)\n"
                    "answer['content'] = 'synthetic gold'\n"
                    "answer['ready'] = True"
                )
                return web.json_response(
                    {
                        "choices": [
                            {
                                "finish_reason": "tool_calls",
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "reasoning_content": "root reason",
                                    "tool_calls": [
                                        {
                                            "id": "root-tool-1",
                                            "type": "function",
                                            "function": {
                                                "name": "repl",
                                                "arguments": json.dumps({"code": code}),
                                            },
                                        }
                                    ],
                                },
                            }
                        ]
                    }
                )
            raise AssertionError("answer-ready worker should terminate after first tool call")
        prompt = messages[-1]["content"]
        answer = {"alpha": "child-alpha", "beta": "child-beta", "gamma": "child-gamma"}[prompt]
        finish_reason = child_mode if prompt == "alpha" else "stop"
        return web.json_response(
            {
                "choices": [
                    {
                        "finish_reason": finish_reason,
                        "message": {
                            "role": "assistant",
                            "content": answer,
                            "reasoning_content": "child reason",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completions)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        result = await evaluate(
            [
                {
                    "id": "integration-1",
                    "context": "fixture context",
                    "root_prompt": "question",
                    "answer": "['synthetic gold']",
                    "answer_type": "ANSWER_TYPE.STRING",
                }
            ],
            base_url=f"http://127.0.0.1:{port}/v1",
            model="fixture-model",
            run_dir=run_dir,
            proxy_socket_path=socket_path,
            max_turns=2,
            max_model_calls=max_model_calls,
        )
    finally:
        await runner.cleanup()
    return result[0], calls


def _install_fake_scorer(monkeypatch):
    import types

    package = types.ModuleType("oolong")
    package.__path__ = [str(OOLONG / "oolong")]
    scoring = types.ModuleType("oolong.scoring")
    scoring.synth_score = lambda row, answer: 1.0 if answer == "synthetic gold" else 0.0
    monkeypatch.setitem(sys.modules, "oolong", package)
    monkeypatch.setitem(sys.modules, "oolong.scoring", scoring)


def test_native_wrapper_plain_queries_keep_order_and_reasoning(tmp_path, monkeypatch):
    _install_fake_scorer(monkeypatch)
    record, requests = asyncio.run(_run_case(tmp_path / "run", tmp_path / "proxy.sock"))
    assert record["error_class"] is None
    assert record["final_answer"] == "synthetic gold"
    assert record["correct"] is True
    assert not (tmp_path / "proxy.sock").exists()
    plain = [payload for payload in requests if not payload.get("tools")]
    assert [p["messages"][-1]["content"] for p in plain] == ["alpha", "beta", "gamma"]
    assert len(requests) == 4
    assert "fixture context" not in json.dumps(requests[0]["messages"])
    for payload in requests:
        assert payload["chat_template_kwargs"] == {"enable_thinking": True}
        assert payload["logprobs"] is False
    saved = json.loads((tmp_path / "run/task-000.json").read_text())
    assert saved["example_id"] == "integration-1"
    assert saved["score"] == 1.0
    events = [
        json.loads(line) for line in (tmp_path / "run/native-events.jsonl").read_text().splitlines()
    ]
    assert events


def test_truncated_plain_completion_is_not_returned_as_answer(tmp_path, monkeypatch):
    _install_fake_scorer(monkeypatch)
    record, requests = asyncio.run(
        _run_case(tmp_path / "run", tmp_path / "proxy.sock", child_mode="length")
    )
    assert record["final_answer"] == "synthetic gold"
    assert len(requests) == 4
    actions = [action for event in record["events"] for action in event.get("actions", [])]
    assert "Error:" in actions[0]["stdout"]
