import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def subcalls(monkeypatch):
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / "training/environments/kernel_rlm"))
    monkeypatch.syspath_prepend(str(root / "training/src"))
    monkeypatch.setenv(
        "PYTHONPATH", f"{root / 'training/src'}:{root / 'training/environments/kernel_rlm'}:{root}"
    )
    from kernel_rlm.subcalls import BoundedSubcalls

    return BoundedSubcalls


def fake_client(content="CUDA advice", finish="stop"):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)],
        usage=SimpleNamespace(model_dump=lambda: {"completion_tokens": 3}),
    )
    create = AsyncMock(return_value=response)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_task_prompt_requires_a_contract_checklist(subcalls):
    from kernel_rlm.submission_contract import KERNEL_TASK_INSTRUCTION

    assert "record a short contract checklist" in KERNEL_TASK_INSTRUCTION
    assert "output type and arity" in KERNEL_TASK_INSTRUCTION
    assert "parameter or buffer names" in KERNEL_TASK_INSTRUCTION


def test_budget_and_trace(subcalls):
    client = fake_client()
    records = []
    child = subcalls(client, "same-model", len, records, max_calls=1)

    async def check():
        assert await child.query("help") == "CUDA advice"
        with pytest.raises(ValueError, match="budget exhausted"):
            await child.query("again")

    asyncio.run(check())
    assert records[0]["response"] == "CUDA advice"
    assert records[0]["usage"]["completion_tokens"] == 3
    assert records[0]["status"] == "completed"
    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == "same-model"
    assert request["max_tokens"] == 2048
    assert "tools" not in request
    assert "no REPL" in request["messages"][0]["content"]
    client.chat.completions.create.assert_awaited_once()


def test_thinking_is_requested_for_child_without_storing_reasoning(subcalls):
    client = fake_client()
    records = []
    child = subcalls(client, "same-model", len, records, enable_thinking=True)

    assert asyncio.run(child.query("derive the output index")) == "CUDA advice"
    request = client.chat.completions.create.call_args.kwargs
    assert request["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert "reasoning_content" not in records[0]


def test_private_reasoning_markup_is_not_returned_to_parent(subcalls, monkeypatch):
    root = Path(__file__).parents[1] / "training/environments/kernel_rlm"
    monkeypatch.syspath_prepend(str(root))
    from kernel_rlm.thinking import visible_response

    assert visible_response("<think>private plan</think>Visible answer") == "Visible answer"
    assert visible_response("Before <think>private</think> after") == "Before  after"


def test_oversized_input_does_not_call_model(subcalls):
    client = fake_client()
    child = subcalls(client, "same-model", len, [])
    with pytest.raises(ValueError, match="token limit"):
        asyncio.run(child.query("x" * 2049))
    assert child.calls_used == 0
    client.chat.completions.create.assert_not_called()


def test_timeout_consumes_budget_and_records_error(subcalls):
    client = fake_client()

    async def slow(**kwargs):
        await asyncio.sleep(10)

    client.chat.completions.create.side_effect = slow
    records = []
    child = subcalls(client, "same-model", len, records, timeout_s=0.01)
    with pytest.raises(TimeoutError):
        asyncio.run(child.query("help"))
    assert child.calls_used == 1
    assert records[0]["status"] == "failed"
    assert "TimeoutError" in records[0]["error"]


def test_truncation_is_visible(subcalls):
    child = subcalls(fake_client(finish="length"), "same-model", len, [])
    assert "truncated" in asyncio.run(child.query("help"))


def test_empty_child_reply_is_not_success(subcalls):
    records = []
    child = subcalls(fake_client(content=""), "same-model", len, records)
    with pytest.raises(ValueError, match="text advice"):
        asyncio.run(child.query("help"))
    assert records[0]["status"] == "failed"


def test_real_worker_proxy_single_and_batched_share_budget(subcalls):
    from rlm_train.proxy import ClientHandle, SubLLMProxy
    from rlm_train.repl.subprocess import SubprocessReplBackend

    client = fake_client()
    records = []
    child = subcalls(client, "same-model", len, records, max_calls=1)

    async def check():
        proxy = SubLLMProxy()
        proxy.register(
            "test",
            ClientHandle(client=None, model="same-model", fake_query=child.query, max_concurrent=1),
        )
        backend = SubprocessReplBackend(request_timeout=10)
        await proxy.start()
        try:
            await backend.start(proxy.url, "test")
            result = await backend.execute(
                "advice = llm_query_batched(['help', 'another'])\nprint(advice)\n"
                "print(rlm_query('grandchild?'))"
            )
            assert "CUDA advice" in result.stdout
            assert result.stdout.count("budget exhausted") == 2
            assert result.final_answer is None
        finally:
            await backend.stop()
            await proxy.stop()

    asyncio.run(check())
    client.chat.completions.create.assert_awaited_once()
    assert len(records) == 1


def test_prompt_example_runs_through_real_worker_and_proxy(subcalls):
    from kernel_rlm.subcalls import SUBCALL_INSTRUCTION
    from rlm_train.proxy import ClientHandle, SubLLMProxy
    from rlm_train.repl.subprocess import SubprocessReplBackend

    instruction = SUBCALL_INSTRUCTION.format(max_calls=2)
    example = instruction.split("```repl\n", 1)[1].split("```", 1)[0]
    client = fake_client(content="Check index < length before reading or writing.")
    records = []
    child = subcalls(client, "same-model", len, records, max_calls=2)

    async def check():
        proxy = SubLLMProxy()
        proxy.register(
            "example",
            ClientHandle(client=None, model="same-model", fake_query=child.query, max_concurrent=1),
        )
        backend = SubprocessReplBackend(request_timeout=10)
        await proxy.start()
        try:
            await backend.start(proxy.url, "example")
            result = await backend.execute(example)
            assert "Check index < length" in result.stdout
            assert not result.stderr
            stored = await backend.execute("print(child_advice)")
            assert "Check index < length before reading or writing." in stored.stdout
        finally:
            await backend.stop()
            await proxy.stop()

    asyncio.run(check())
    client.chat.completions.create.assert_awaited_once()
    assert len(records) == 1
    assert "thread-to-element mapping" in records[0]["prompt"]
    assert "if (index < length)" in records[0]["prompt"]
    assert "make one focused child call before coding" in instruction
    assert "not a kernel" in instruction
    assert "final verification measures speed" in instruction
