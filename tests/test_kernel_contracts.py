import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

REPO = Path(__file__).parents[1]
ENVIRONMENT = REPO / "training/environments/kernel_rlm"
TRAINING = REPO / "training/src"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ENVIRONMENT))
    monkeypatch.syspath_prepend(str(TRAINING))
    monkeypatch.setenv("PYTHONPATH", f"{ENVIRONMENT}:{TRAINING}:{REPO}")
    from kernel_rlm import actions, feedback, verifier

    spec = importlib.util.spec_from_file_location(
        "kernel_program_test", ENVIRONMENT / "kernel_rlm_v1/program.py"
    )
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    return SimpleNamespace(actions=actions, feedback=feedback, verifier=verifier, program=program)


@pytest.mark.parametrize("nested", [False, True])
def test_structured_action_and_duplicate_fence(modules, nested):
    function = {"name": "repl", "arguments": json.dumps({"code": "print(context)"})}
    call = {"function": function} if nested else function
    codes, errors = modules.actions.repl_actions(
        {"content": "```repl\nprint(context)\n```", "tool_calls": [call]}, "stop"
    )
    assert codes == ["print(context)"]
    assert errors == []


@pytest.mark.parametrize(
    "call",
    [
        {"name": "bash", "arguments": "{}"},
        {"name": "repl", "arguments": "{"},
        {"name": "repl", "arguments": '{"code":42}'},
        {"name": "repl", "arguments": "[]"},
    ],
)
def test_bad_action_reports_error(modules, call):
    codes, errors = modules.actions.repl_actions({"tool_calls": [call]}, "stop")
    assert not codes
    assert errors


def test_truncated_actions_are_not_executed(modules):
    codes, errors = modules.actions.repl_actions({"content": "```repl\nprint(1)\n```"}, "length")
    assert not codes
    assert "truncated" in errors[0]


@pytest.mark.parametrize("code", ["print(source)", "print(len(source))"])
def test_raw_tool_arguments_are_rejected_with_feedback(modules, code):
    message = {"tool_calls": [{"function": {"name": "repl", "arguments": code}}]}
    codes, errors = modules.actions.repl_actions(message, "stop")
    assert codes == []
    assert errors == ["Malformed repl arguments; expected JSON with a code string."]


def test_textual_tool_call_echo_is_not_an_action(modules):
    message = {
        "content": 'Structured REPL actions:\n[{"id":"call_0","function":'
        '{"arguments":"print(source)","name":"repl"},"type":"function"}]',
        "tool_calls": None,
    }
    codes, errors = modules.actions.repl_actions(message, "stop")
    assert codes == []
    assert "No executable REPL action" in errors[0]


@pytest.mark.parametrize("transport", ["local", "http"])
@pytest.mark.parametrize("fail", [False, True])
def test_feedback_schema_budget_and_no_reward(modules, monkeypatch, transport, fail):
    invocations = []

    def verify(reference, source, **kwargs):
        invocations.append(kwargs)
        assert kwargs["benchmark"] is False
        if fail:
            raise TimeoutError("mock timeout")
        return {"compiled": True, "correct": True, "extension_used": True, "max_diff": 0}

    monkeypatch.setattr(modules.verifier, "verify_cuda", verify)
    namespace = {"answer": {"ready": False, "content": ""}}
    if transport == "local":
        code = modules.feedback.build_local_feedback_bootstrap("reference", max_calls=1)
    else:
        code = modules.feedback.build_http_feedback_bootstrap("http://unused", max_calls=1)
    exec(code, namespace)
    if transport == "http":

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(verify("reference", "source", benchmark=False)).encode()

        namespace["_kernel_feedback_urlopen"] = lambda *args, **kwargs: Response()
    first = namespace["cuda_feedback"]("source")
    exhausted = namespace["cuda_feedback"]("source")
    assert first.keys() == exhausted.keys()
    assert first["compiled"] is not fail
    assert exhausted["compiled"] is False
    assert exhausted["feedback_calls_used"] == 1
    assert exhausted["feedback_calls_remaining"] == 0
    assert len(invocations) == 1
    assert "budget exhausted" in exhausted["error"]
    assert "reward" not in first
    assert namespace["answer"] == {"ready": False, "content": ""}


def test_timeout_budget_includes_two_feedback_calls(modules):
    actions = modules.actions
    assert actions.FEEDBACK_MAX_CALLS * actions.FEEDBACK_TIMEOUT_S < actions.REPL_TIMEOUT_S
    assert actions.REPL_TIMEOUT_S < actions.BACKEND_TIMEOUT_S


def test_prompt_distinguishes_constructor_constants_from_runtime_inputs(modules):
    from kernel_rlm.submission_contract import KERNEL_TASK_INSTRUCTION

    assert "constructor values are NOT extra runtime inputs" in KERNEL_TASK_INSTRUCTION
    assert "_module.launch(x, 3.0)" in KERNEL_TASK_INSTRUCTION
    assert "Random initialized" in KERNEL_TASK_INSTRUCTION
    assert "never invent them" in KERNEL_TASK_INSTRUCTION


def test_prompt_scaffold_assembles_source_without_loading_extension(modules, capsys):
    import ast

    from kernel_rlm.submission_contract import (
        CUDA_ASSEMBLY_TEMPLATE,
        KERNEL_TASK_INSTRUCTION,
        validate_submission_source,
    )

    received = []

    def feedback(source):
        received.append(source)
        print('CUDA feedback: {"compiled": true}')
        return {"compiled": True}

    namespace = {
        "cpp_declarations": "#include <torch/extension.h>\ntorch::Tensor launch(torch::Tensor x);",
        "cuda_implementation": "#include <torch/extension.h>\ntorch::Tensor launch(torch::Tensor x) { return x; }",
        "cuda_feedback": feedback,
    }
    exec(CUDA_ASSEMBLY_TEMPLATE, namespace)
    assert capsys.readouterr().out.count("compiled") == 1
    assert received == [namespace["source"]]
    assert validate_submission_source(namespace["source"])[0]
    tree = ast.parse(namespace["source"])
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_inline"
    )
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    assert ast.literal_eval(keywords["functions"]) == ["launch"]
    assert "is_python_module" not in keywords
    assert "Never use import context" in KERNEL_TASK_INSTRUCTION
    assert "string containing the reference Python source" in KERNEL_TASK_INSTRUCTION
    assert "Model()" in KERNEL_TASK_INSTRUCTION
    assert "ast.parse(context)" in KERNEL_TASK_INSTRUCTION
    from kernel_rlm.subcalls import KERNEL_SYSTEM

    assert "not a module, Model instance, or callable" in KERNEL_SYSTEM
    assert "or execute the reference" in KERNEL_SYSTEM
    assert "turn six" not in KERNEL_SYSTEM
    assert "current run's turn budget" in KERNEL_SYSTEM
    assert KERNEL_TASK_INSTRUCTION.endswith(CUDA_ASSEMBLY_TEMPLATE)


def test_worker_exit_during_forced_cleanup(modules):
    from rlm_train.repl.subprocess import SubprocessReplBackend, WorkerProtocolError

    backend = SubprocessReplBackend()
    backend._proc = SimpleNamespace(
        returncode=None,
        terminate=Mock(),
        kill=Mock(side_effect=ProcessLookupError),
        wait=AsyncMock(side_effect=[TimeoutError(), None]),
    )
    backend._send = AsyncMock()
    backend._read_line = AsyncMock(side_effect=WorkerProtocolError("closed"))
    asyncio.run(backend.stop())
    assert backend._proc is None


@pytest.mark.parametrize("workflow", ["budget", "scaffold", "invalid_syntax"])
def test_feedback_through_real_worker_does_not_submit(modules, monkeypatch, tmp_path, workflow):
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "KERNEL_LIMITS": '{"max_turns":1,"max_tokens":128,"model_timeout_s":90}',
        "KERNEL_ENDPOINT": "http://unused",
        "KERNEL_API_KEY": "test-only",
        "KERNEL_MODEL": "fake-model",
    }.items():
        monkeypatch.setenv(key, value)
    original_bootstrap = modules.program.build_local_feedback_bootstrap

    def mock_bootstrap(*args, **kwargs):
        return original_bootstrap(*args, **kwargs) + (
            "\ndef _kernel_feedback_verify(*args, **kwargs):\n"
            "    assert kwargs['benchmark'] is False\n"
            "    return {'compiled': True, 'correct': True, 'extension_used': True}\n"
        )

    monkeypatch.setattr(modules.program, "build_local_feedback_bootstrap", mock_bootstrap)
    content = "```repl\ncuda_feedback('a')\ncuda_feedback('b')\ncuda_feedback('c')\n```"
    if workflow != "budget":
        from kernel_rlm.submission_contract import CUDA_ASSEMBLY_TEMPLATE

        code = (
            "assert compile is None\n"
            "cpp_declarations = 'torch::Tensor launch(torch::Tensor x);'\n"
            "cuda_implementation = 'torch::Tensor launch(torch::Tensor x) { return x; }'\n"
            + CUDA_ASSEMBLY_TEMPLATE
        )
        if workflow == "invalid_syntax":
            code = code.replace("import ast\n", "source = 'def broken('\nimport ast\n")
        else:
            code += '\nanswer["content"] = source\nanswer["ready"] = True\n'
        content = f"```repl\n{code}\n```"
    reasoning = "```repl\nraise RuntimeError('Never execute thoughts')\n```"
    message = SimpleNamespace(
        content=content,
        model_dump=lambda **kwargs: {"content": content, "reasoning_content": reasoning},
    )
    reply = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=reply))),
        close=AsyncMock(),
    )
    monkeypatch.setattr(modules.program, "AsyncOpenAI", lambda **kwargs: client)
    asyncio.run(modules.program.run({"context": "CPU fixture", "prompt": "test"}))
    record = json.loads((tmp_path / "submission.json").read_text())
    assert "reward" not in record
    assert not any(message.get("reasoning_content") for message in record["messages"])
    assert record["model_calls"][0]["reasoning_characters"] == len(reasoning)
    request = client.chat.completions.create.await_args.kwargs
    assert [tool["function"]["name"] for tool in request["tools"]] == ["repl"]
    assert request["tool_choice"] == "auto"
    if workflow == "budget":
        assert record["feedback_calls"] == 2
        assert record["submission"] == ""
        assert record["termination_reason"] == "turn_limit"
        assert "budget exhausted" in record["messages"][-1]["content"]
    elif workflow == "scaffold":
        assert record["feedback_calls"] == 1
        assert "CPP_SOURCE" in record["submission"]
        assert record["termination_reason"] == "submitted"
        assert '"correct": true' in record["messages"][-1]["content"]
    else:
        assert record["feedback_calls"] == 0
        assert record["submission"] == ""
        assert "SyntaxError" in record["messages"][-1]["content"]


def test_structured_repl_call_gets_matching_tool_observation(modules, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "KERNEL_LIMITS": '{"max_turns":3,"max_tokens":128}',
        "KERNEL_ENDPOINT": "http://unused",
        "KERNEL_API_KEY": "test-only",
        "KERNEL_MODEL": "fake-model",
    }.items():
        monkeypatch.setenv(key, value)
    code = "context = 'fabricated one-input reference'\nprint(context)"
    final_code = "print(context)\nanswer['content'] = 'cpu-test-source'\nanswer['ready'] = True"
    replies = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        model_dump=lambda **kwargs: {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-python-0",
                                    "type": "function",
                                    "function": {
                                        "name": "python",
                                        "arguments": json.dumps({"code": code}),
                                    },
                                }
                            ],
                        },
                    ),
                    finish_reason="tool_calls",
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        model_dump=lambda **kwargs: {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-repl-1",
                                    "type": "function",
                                    "function": {
                                        "name": "repl",
                                        "arguments": json.dumps({"code": code}),
                                    },
                                }
                            ],
                        },
                    ),
                    finish_reason="tool_calls",
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        model_dump=lambda **kwargs: {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-repl-2",
                                    "type": "function",
                                    "function": {
                                        "name": "repl",
                                        "arguments": json.dumps({"code": final_code}),
                                    },
                                }
                            ],
                        },
                    ),
                    finish_reason="tool_calls",
                )
            ]
        ),
    ]
    create = AsyncMock(side_effect=replies)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=AsyncMock()
    )
    monkeypatch.setattr(modules.program, "AsyncOpenAI", lambda **kwargs: client)
    asyncio.run(modules.program.run({"context": "CPU fixture", "prompt": "inspect"}))
    final_request = create.await_args_list[2].kwargs
    messages = final_request["messages"]
    assert not any(
        call.get("function", call).get("name") == "python"
        for message in messages
        for call in message.get("tool_calls", [])
    )
    assert any(
        message.get("role") == "user" and "only tool is repl" in message.get("content", "")
        for message in messages
    )
    call_message = next(message for message in messages if message.get("tool_calls"))
    call_id = call_message["tool_calls"][0]["id"]
    tool_replies = [message for message in messages if message.get("role") == "tool"]
    assert tool_replies[0]["tool_call_id"] == call_id == "call-repl-1"
    assert "context is read-only" in tool_replies[0]["content"]
    assert tool_replies[1]["tool_call_id"] == "call-repl-2"
    assert "CPU fixture" in tool_replies[1]["content"]
    assert "Structured REPL actions:" not in str(messages)
    record = json.loads((tmp_path / "submission.json").read_text())
    assert record["submission"] == "cpu-test-source"


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("structured", "submitted"),
        ("fenced", "submitted"),
        ("empty", "turn_limit"),
        ("length", "truncated"),
        ("error", "error"),
        ("textual_tool_echo", "turn_limit"),
        ("raw_tool_arguments", "turn_limit"),
    ],
)
def test_program_with_real_cpu_worker(modules, monkeypatch, tmp_path, kind, expected):
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "KERNEL_LIMITS": '{"max_turns":1,"max_tokens":128}',
        "KERNEL_ENDPOINT": "http://unused",
        "KERNEL_API_KEY": "test-only",
        "KERNEL_MODEL": "fake-model",
        "RLM_TRAIN_EXEC_TIMEOUT_S": "0.01",
    }.items():
        monkeypatch.setenv(key, value)
    code = "import time\ntime.sleep(0.05)\nanswer['content'] = 'cpu-test-source'\nanswer['ready'] = True"
    message = {
        "content": f"```repl\n{code}\n```"
        if kind in {"fenced", "length", "structured"}
        else "Still working"
    }
    if kind == "structured":
        message["tool_calls"] = [
            {"function": {"name": "repl", "arguments": json.dumps({"code": code})}}
        ]
    if kind == "textual_tool_echo":
        message["content"] = "Structured REPL actions:\n" + json.dumps(
            [{"function": {"name": "repl", "arguments": code}}]
        )
        message["tool_calls"] = None
    if kind == "raw_tool_arguments":
        message["tool_calls"] = [{"function": {"name": "repl", "arguments": code}}]
    reply = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=message["content"], model_dump=lambda **kwargs: message
                ),
                finish_reason="length" if kind == "length" else "stop",
            )
        ]
    )
    create = (
        AsyncMock(side_effect=RuntimeError("model unavailable"))
        if kind == "error"
        else AsyncMock(return_value=reply)
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=AsyncMock()
    )
    monkeypatch.setattr(modules.program, "AsyncOpenAI", lambda **kwargs: client)
    if kind == "error":
        with pytest.raises(RuntimeError, match="model unavailable"):
            asyncio.run(modules.program.run({"context": "CPU fixture", "prompt": "test"}))
    else:
        asyncio.run(modules.program.run({"context": "CPU fixture", "prompt": "test"}))
    record = json.loads((tmp_path / "submission.json").read_text())
    assert record["termination_reason"] == expected
    assert record["repl_calls"] == (1 if expected == "submitted" else 0)
    assert bool(record["submission"]) is (expected == "submitted")
    assert record["feedback_calls"] == 0
    if kind == "structured":
        saved = next(message for message in record["messages"] if message["role"] == "assistant")
        assert saved["tool_calls"][0]["function"]["name"] == "repl"
        assert saved["tool_calls"][0]["id"].startswith("repl_")
        assert "Structured REPL actions:" not in saved["content"]
    if kind == "textual_tool_echo":
        assert "No executable REPL action" in record["model_calls"][0]["action_errors"][0]
        assert "No executable REPL action" in record["messages"][-1]["content"]
    if kind == "raw_tool_arguments":
        assert "Malformed repl arguments" in record["messages"][-1]["content"]
    client.close.assert_awaited_once()


@pytest.mark.parametrize("subcall_probe", [False, True])
def test_enabled_subcall_through_program_and_worker(modules, monkeypatch, tmp_path, subcall_probe):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "KERNEL_LIMITS",
        json.dumps(
            {
                "max_turns": 1,
                "max_tokens": 2048,
                "enable_subcall": True,
                "subcall_probe": subcall_probe,
            }
        ),
    )
    monkeypatch.setenv("KERNEL_ENDPOINT", "http://unused")
    monkeypatch.setenv("KERNEL_API_KEY", "test-only")
    monkeypatch.setenv("KERNEL_MODEL", "fake-model")
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)
        ),
    )
    content = (
        "```repl\nchild_advice = llm_query('help with CUDA')\nprint(child_advice)\n"
        "answer['content'] = 'test-source'\nanswer['ready'] = True\n```"
    )
    root = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content, model_dump=lambda **kwargs: {"content": content}
                ),
                finish_reason="stop",
            )
        ]
    )
    child = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="use threadIdx.x"),
                finish_reason="stop",
            )
        ],
        usage=None,
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(side_effect=[root, child]))
        ),
        close=AsyncMock(),
    )
    monkeypatch.setattr(modules.program, "AsyncOpenAI", lambda **kwargs: client)
    asyncio.run(
        modules.program.run({"context": "reference", "prompt": "Do not delegate to sub-LLMs."})
    )
    record = json.loads((tmp_path / "submission.json").read_text())
    assert record["submission"] == "test-source"
    assert len(record["subcalls"]) == 1
    assert record["subcalls"][0]["response"] == "use threadIdx.x"
    assert "use threadIdx.x" in record["messages"][-1]["content"]
    assert "Do not delegate" not in str(record["messages"])
    assert "2,048" in record["messages"][0]["content"]
    assert "Example of delegating an indexing review" in record["messages"][0]["content"]
    assert "make one focused child call before coding" in record["messages"][0]["content"]
    assert str(record["messages"]).count("Example of delegating an indexing review") == 1
    assert "100k" not in str(record["messages"])
    assert "500K" not in str(record["messages"])
    assert "FINAL TURN" in str(record["messages"])
    assert client.chat.completions.create.await_count == 2
    client.close.assert_awaited_once()
