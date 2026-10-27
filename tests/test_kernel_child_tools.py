import asyncio
import json
import os
import re
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
    call_ids = [call.kwargs["extra_headers"]["X-Kernel-Call-Id"] for call in requests]
    assert len(set(call_ids)) == 3
    assert records[0]["child_invocation_id"]
    assert [item["call_id"] for item in records[0]["requests"]] == call_ids
    assert [call.kwargs["max_tokens"] for call in requests] == [4096, 3996, 3896]
    assert "undeclared index" in requests[1].kwargs["messages"][3]["content"]
    assert all(
        call.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] for call in requests
    )


def test_child_hardware_context_preserves_bridge_identity_and_reference(module, tmp_path):
    session, client, _records = child(
        module,
        tmp_path,
        [response({"action": "finish", "source": "", "advice": "review"})],
        hardware_context="Target RTX PRO 6000 sm_120, CUDA13.3",
    )
    asyncio.run(session.query("Review indexing"))
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["content"].startswith(
        module.TOOL_CHILD_SYSTEM.split("CHILD_TOTAL_TOKENS")[0]
    )
    assert "4096 generated tokens TOTAL" in messages[0]["content"]
    assert messages[1]["content"].startswith("Target RTX PRO 6000 sm_120")
    assert "Immutable reference source:\nunchanged reference" in messages[1]["content"]
    assert session.reference == "unchanged reference"


@pytest.mark.parametrize("max_calls", [1, 2])
def test_compiler_child_guidance_retains_contract_examples_and_real_budgets(module, max_calls):
    instruction = module.TOOL_SUBCALL_INSTRUCTION.format(max_calls=max_calls)
    assert f"up to {max_calls} times per trajectory" in instruction
    assert f"same {max_calls}-call budget" in instruction
    assert "state-dict names/shapes" in instruction
    assert "reference and hardware guidance are attached automatically" in instruction
    assert "4096 generated tokens TOTAL" in instruction
    assert "480-second session deadline" in instruction
    assert "ALTERNATIVE examples" in instruction
    assert "no reward" in instruction
    assert "submission deadline or during recovery" in instruction


@pytest.mark.parametrize("example", [0, 1, 2])
@pytest.mark.parametrize("independent_correct", [False, True])
@pytest.mark.parametrize("feedback_remaining", [0, 1])
def test_compiler_child_examples_execute_and_preserve_independent_check(
    module, example, independent_correct, feedback_remaining
):
    blocks = re.findall(r"```repl\n(.*?)\n```", module.TOOL_SUBCALL_INSTRUCTION, re.S)
    assert len(blocks) == 4
    assignments = []
    checked = []
    result = {
        "advice": "Review only the requested issue",
        "source": "repaired complete source",
        "verified_by_child_check": True,
        "final_grading_required": True,
    }

    def llm_query(assignment):
        assignments.append(assignment)
        return json.dumps(result)

    def cuda_feedback(source):
        checked.append(source)
        return {"correct": independent_correct}

    feedback = {"error": "exact compiler error", "feedback_calls_remaining": feedback_remaining}
    namespace = {
        "source": "original complete source",
        "feedback": feedback,
        "llm_query": llm_query,
        "cuda_feedback": cuda_feedback,
    }
    exec(compile(blocks[example], "compiler-child-example", "exec"), namespace)
    assert len(assignments) == 1
    assert (
        "original complete source" in assignments[0] if example else "state-dict" in assignments[0]
    )
    if example == 1:
        assert str(feedback) in assignments[0]
    if example:
        exec(compile(blocks[3], "compiler-child-independent-check", "exec"), namespace)
        assert checked == ([result["source"]] if feedback_remaining else [])
        assert namespace["source"] == (
            result["source"]
            if feedback_remaining and independent_correct
            else "original complete source"
        )
    else:
        assert namespace["contract_review"] == result
        assert checked == []


def test_child_hardware_context_counts_toward_existing_input_budget(module, tmp_path):
    session, client, _records = child(
        module,
        tmp_path,
        [],
        hardware_context="target " * 5000,
    )
    with pytest.raises(ValueError, match="4096 input tokens"):
        asyncio.run(session.query("Review indexing"))
    client.chat.completions.create.assert_not_called()
    assert session.calls_used == 0


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
    from kernel_rollout.bridge import CompletionBridge
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
    assert generator.runner.limits["max_subcalls"] == 2


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


def test_enclosing_deadlines_cover_children_feedback_and_cleanup(module):
    repl, backend = module.enclosing_timeouts(
        2, repl_base=270, backend_base=300, feedback_timeout=180, feedback_calls=2
    )
    assert repl == 2 * 480 + 2 * 180 + 30
    assert backend >= repl + 30
    assert backend < 2400
    with pytest.raises(ValueError, match="two bounded"):
        module.enclosing_timeouts(
            4, repl_base=270, backend_base=300, feedback_timeout=180, feedback_calls=2
        )


def test_fenced_child_source_needs_no_json_escaping(module):
    source = 'CPP_SOURCE = """void launch();"""\nCUDA_SOURCE = """kernel"""'
    assert module.parse_action("CHECK\n```python\n" + source + "\n```") == {
        "action": "check",
        "advice": "",
        "source": source,
    }
    assert (
        module.parse_action("FINISH\nCompiler repair passed\n```python\n" + source + "\n```")[
            "advice"
        ]
        == "Compiler repair passed"
    )


def test_malformed_action_gets_bounded_format_feedback_without_resetting_tokens(
    module, tmp_path, monkeypatch
):
    monkeypatch.setattr(module, "validate_submission_source", lambda source: (True, ""))
    replies = [response({}), response({}), response({})]
    replies[0].choices[0].message.content = 'I will check now. {"source":"broken'
    replies[1].choices[0].message.content = "CHECK\n```python\ncandidate\n```"
    replies[2].choices[0].message.content = "FINISH\nChecked candidate\n```python\ncandidate\n```"
    checker = AsyncMock(return_value={"compiled": True, "correct": True})
    session, client, records = child(module, tmp_path, replies, checker)
    result = json.loads(asyncio.run(session.query("Check the source")))
    assert result["verified_by_child_check"] is True
    assert len(records[0]["format_errors"]) == 1
    assert records[0]["completion_tokens"] == 300
    assert checker.await_count == 1
    assert [
        call.kwargs["max_tokens"] for call in client.chat.completions.create.call_args_list
    ] == [4096, 3996, 3896]


def test_tool_children_are_reachable_from_the_real_root_repl(module, tmp_path, monkeypatch):
    from kernel_skyrl.generator import KernelGenerator

    class Client:
        def __init__(self):
            self.roots = 0
            self.children = 0

        async def chat_completion(self, payload):
            messages = payload["json"]["messages"]
            if messages[0]["content"].startswith(
                module.TOOL_CHILD_SYSTEM.split("CHILD_TOTAL_TOKENS")[0]
            ):
                texts = [
                    "CHECK\n```python\nimport torch\n```",
                    "FINISH\nStatic feedback requires a complete candidate.\n```python\n\n```",
                ]
                content = texts[self.children]
                self.children += 1
            else:
                assert module.TOOL_SUBCALL_INSTRUCTION.format(max_calls=2) in messages[0]["content"]
                codes = [
                    "import json\nchild_result = json.loads(llm_query('Review this candidate'))\nprint(child_result['advice'][:80])",
                    "answer['content'] = 'fixture candidate'\nanswer['ready'] = True",
                ]
                content = "```repl\n" + codes[self.roots] + "\n```"
                self.roots += 1
            return {
                "id": "fixture",
                "object": "chat.completion",
                "created": 0,
                "model": "policy",
                "prompt_token_ids": [2],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "token_ids": [3],
                        "message": {"role": "assistant", "content": content},
                        "logprobs": {"content": [{"token": "token_id:3", "logprob": -0.5}]},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

        async def finish_session(self, session_id):
            pass

    repository = ROOT.parents[2]
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join([str(repository), str(repository / "training/src"), str(ROOT)]),
    )
    command = [
        sys.executable,
        "-c",
        "import sys,asyncio,json; "
        f"sys.path[:0]={[str(repository), str(repository / 'training/src'), str(ROOT)]!r}; "
        "from kernel_rlm_v1 import program; program.load_token_counter=lambda path:len; "
        "asyncio.run(program.run(json.load(open(sys.argv[1]))))",
        "task.json",
    ]
    client = Client()
    generator = KernelGenerator(
        client,
        "policy-17",
        "fixture-tokenizer",
        tmp_path,
        limits={
            "child_tools": True,
            "dedicated_inference": True,
            "child_prompt_tokens": 4096,
            "enable_thinking": True,
            "max_context_tokens": 16384,
            "max_turns": 10,
        },
        episode_timeout_s=2400,
        program_command=command,
        scorer=AsyncMock(return_value=(0.8, {"correct": True})),
    )

    async def run():
        return await asyncio.wait_for(
            generator.run_episode(
                {"name": "fixture-task", "context": "class Model: pass", "prompt": "Test wiring"},
                tmp_path / "episode",
                17,
            ),
            15,
        )

    result = asyncio.run(run())
    assert client.children == 2
    assert len(result["samples"]) == 2
    model_calls = result["record"]["model_calls"]
    assert len(model_calls) == 2
    assert len({call["root_turn_id"] for call in model_calls}) == 2
    assert all(call["call_id"] == call["model_call_id"] for call in model_calls)
    assert all(call["message_index"] >= 0 for call in model_calls)
    child_record = result["record"]["subcalls"][0]
    assert child_record["mode"] == "compile_repair"
    assert child_record["status"] == "completed"
    assert child_record["checks"][0]["result"]["verification_status"] == "static_rejected"
    assert child_record["child_invocation_id"]
    request_ids = [request["call_id"] for request in child_record["requests"]]
    assert len(request_ids) == 2 and len(set(request_ids)) == 2
    assert all(request["parent_turn_id"] for request in child_record["requests"])
    assert all(request["parent_tool_call_id_generated"] for request in child_record["requests"])
    assert all(request["parent_action_id"] for request in child_record["requests"])
    root_action = next(
        action
        for action in result["record"]["root_actions"]
        if action["action_id"] == child_record["parent_action_id"]
    )
    assert root_action["tool_call_id"] == child_record["parent_tool_call_id"]
    assert root_action["tool_call_id_generated"] is True
    assert root_action["observation_id"]
