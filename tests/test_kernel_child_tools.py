import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).parents[1] / "training/environments/kernel_rlm"


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from kernel_rlm import child_tools

    return child_tools


def response(action, tokens=100, finish="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(action)), finish_reason=finish
            )
        ],
        usage=SimpleNamespace(completion_tokens=tokens),
    )


def child(module, tmp_path, responses, checker=None, **kwargs):
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=responses)))
    )
    records = []
    session = module.IterativeSubcalls(
        client,
        "policy-17",
        lambda value: len(value) // 4,
        records,
        reference="unchanged reference",
        directory=tmp_path,
        checker=checker or AsyncMock(),
        **kwargs,
    )
    return session, client, records


def test_child_checks_repairs_and_returns_only_measured_status(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "validate_submission_source", lambda source: (True, ""))
    checker = AsyncMock(
        side_effect=[
            {"compiled": False, "correct": False, "error": "undeclared index"},
            {"compiled": True, "correct": True, "max_diff": 0},
        ]
    )
    session, client, records = child(
        module,
        tmp_path,
        [
            response({"action": "check", "source": "first"}),
            response({"action": "check", "source": "repaired"}),
            response(
                {"action": "finish", "source": "repaired", "advice": "Fixed index declaration"}
            ),
        ],
        checker,
    )
    result = json.loads(asyncio.run(session.query("Repair the compiler error")))
    assert result["verified_by_child_check"] is True
    assert result["final_grading_required"] is True
    assert checker.await_count == 2
    assert records[0]["completion_tokens"] == 300
    assert records[0]["child_tokens_in_root_loss"] == 0
    assert records[0]["status"] == "completed"
    requests = client.chat.completions.create.call_args_list
    assert [call.kwargs["max_tokens"] for call in requests] == [4096, 3996, 3896]
    assert "undeclared index" in requests[1].kwargs["messages"][3]["content"]
    assert all(
        call.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] for call in requests
    )


def test_child_cannot_self_certify_an_untested_source(module, tmp_path):
    session, _client, records = child(
        module,
        tmp_path,
        [response({"action": "finish", "source": "untested", "advice": "I think this works"})],
    )
    assert (
        json.loads(asyncio.run(session.query("Suggest a change")))["verified_by_child_check"]
        is False
    )
    assert records[0]["checks"] == []


def test_child_total_tokens_are_not_reset_for_each_reply(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "validate_submission_source", lambda source: (True, ""))
    session, client, records = child(
        module,
        tmp_path,
        [response({"action": "check", "source": "first"}, tokens=4000)],
        AsyncMock(return_value={"correct": True}),
    )
    with pytest.raises(ValueError, match="aggregate"):
        asyncio.run(session.query("Check this"))
    assert client.chat.completions.create.await_count == 1
    assert records[0]["status"] == "failed"


def test_child_cannot_request_a_third_check(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "validate_submission_source", lambda source: (True, ""))
    checker = AsyncMock(return_value={"correct": False})
    session, _client, _records = child(
        module,
        tmp_path,
        [response({"action": "check", "source": "candidate"}) for index in range(3)],
        checker,
    )
    with pytest.raises(ValueError, match="two checks"):
        asyncio.run(session.query("Keep trying"))
    assert checker.await_count == 2


def test_static_rejection_does_not_start_gpu_worker(module, tmp_path):
    checker = AsyncMock()
    session, _client, records = child(
        module,
        tmp_path,
        [
            response({"action": "check", "source": "import torch"}),
            response({"action": "finish", "source": "", "advice": "Need a complete extension"}),
        ],
        checker,
    )
    asyncio.run(session.query("Try a candidate"))
    checker.assert_not_awaited()
    assert records[0]["checks"][0]["result"]["verification_status"] == "static_rejected"


def test_session_timeout_cancels_the_running_checker(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "validate_submission_source", lambda source: (True, ""))
    cancelled = []

    async def checker(*args):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)

    session, _client, records = child(
        module,
        tmp_path,
        [response({"action": "check", "source": "candidate"})],
        checker,
        timeout_s=0.02,
    )
    with pytest.raises(TimeoutError):
        asyncio.run(session.query("Check this"))
    assert cancelled == [True]
    assert records[0]["status"] == "failed"


def test_stopping_worker_unwinds_its_separate_runner_group(module, tmp_path):
    pidfile = tmp_path / "runner.pid"
    script = (
        "import signal,subprocess,sys,time\n"
        "from kernel_rlm.verifier import _run_verifier_process\n"
        "def cancel(signum,frame): raise KeyboardInterrupt()\n"
        "signal.signal(signal.SIGTERM,cancel)\n"
        "_run_verifier_process([sys.executable,'-c',"
        f"\"import os,time;open({str(pidfile)!r},'w').write(str(os.getpid()));time.sleep(60)\"],"
        "timeout_s=60,cwd=None,env=None)\n"
    )

    async def run():
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            script,
            start_new_session=True,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            async with asyncio.timeout(5):
                while not pidfile.exists():
                    await asyncio.sleep(0.01)
            runner_pid = int(pidfile.read_text())
            await module.stop_verifier(process)
            with pytest.raises(ProcessLookupError):
                os.kill(runner_pid, 0)
        finally:
            await module.stop_verifier(process)

    asyncio.run(run())


def test_iterative_requests_are_excluded_from_root_loss(module, tmp_path):
    from kernel_skyrl.bridge import CompletionBridge
    from openai import AsyncOpenAI

    payload = {
        "id": "fixture",
        "object": "chat.completion",
        "created": 0,
        "model": "fixture",
        "prompt_token_ids": [2],
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "token_ids": [3],
                "message": {
                    "role": "assistant",
                    "content": '{"action":"finish","advice":"done","source":""}',
                },
                "logprobs": {"content": [{"token": "token_id:3", "logprob": -0.5}]},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    backend = SimpleNamespace(
        chat_completion=AsyncMock(return_value=payload), finish_session=AsyncMock()
    )
    bridge = CompletionBridge(backend, "policy-17", 16384, 4096, 17, tmp_path / "calls.jsonl")

    async def run():
        await bridge.start()
        try:
            async with AsyncOpenAI(base_url=bridge.endpoint, api_key=bridge.api_key) as client:
                await client.chat.completions.create(
                    model="policy-17",
                    max_tokens=4096,
                    messages=[
                        {"role": "system", "content": module.TOOL_CHILD_SYSTEM},
                        {"role": "user", "content": "check indexing"},
                    ],
                )
                assert bridge.child_calls == 1
                assert bridge.samples == []
                await client.chat.completions.create(
                    model="policy-17",
                    max_tokens=4096,
                    messages=[{"role": "system", "content": "root"}],
                )
                assert len(bridge.samples) == 1
        finally:
            await bridge.stop()

    asyncio.run(run())


def test_generator_rejects_tool_children_without_dedicated_budget(module, tmp_path):
    from kernel_skyrl.generator import KernelGenerator

    for limits in (
        {"child_tools": True},
        {
            "child_tools": True,
            "dedicated_inference": True,
            "enable_thinking": True,
            "child_prompt_tokens": 4096,
        },
    ):
        with pytest.raises(ValueError, match="explicit budgets"):
            KernelGenerator(None, "policy", "tokenizer", tmp_path, limits=limits)
    generator = KernelGenerator(
        None,
        "policy",
        "tokenizer",
        tmp_path,
        episode_timeout_s=2400,
        limits={
            "child_tools": True,
            "dedicated_inference": True,
            "enable_thinking": True,
            "child_prompt_tokens": 4096,
        },
    )
    assert generator.limits["max_subcalls"] == 2


def test_worker_lifetime_follows_its_actual_parent(module, monkeypatch):
    calls = []
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        module.ctypes,
        "CDLL",
        lambda *args, **kwargs: SimpleNamespace(prctl=lambda *args: calls.append(args) or 0),
    )
    module.install_worker_cancellation(os.getppid())
    assert calls == [(1, module.signal.SIGTERM, 0, 0, 0)]
    with pytest.raises(KeyboardInterrupt, match="parent exited"):
        module.install_worker_cancellation(-1)


def test_child_action_accepts_only_a_complete_json_object(module):
    action = {"action": "finish", "advice": "done", "source": ""}
    assert module.parse_action(json.dumps(action)) == action
    assert module.parse_action("```json\n" + json.dumps(action) + "\n```") == action
    for invalid in ("[]", 'preface {"action":"check"}', "```python\n{}\n```"):
        with pytest.raises(ValueError):
            module.parse_action(invalid)
