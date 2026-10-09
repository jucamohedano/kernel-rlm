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


def test_budget_and_trace(subcalls):
    client = fake_client()
    records = []
    child = subcalls(client, "same-model", len, records)

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


def test_oversized_input_does_not_call_model(subcalls):
    client = fake_client()
    child = subcalls(client, "same-model", len, [])
    with pytest.raises(ValueError, match="token limit"):
        asyncio.run(child.query("x" * 2049))
    assert child.used is False
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
    assert child.used is True
    assert records[0]["status"] == "failed"
    assert "TimeoutError" in records[0]["error"]


def test_truncation_is_visible(subcalls):
    child = subcalls(fake_client(finish="length"), "same-model", len, [])
    assert "truncated" in asyncio.run(child.query("help"))


def test_real_worker_proxy_single_and_batched_share_budget(subcalls):
    from rlm_train.proxy import ClientHandle, SubLLMProxy
    from rlm_train.repl.subprocess import SubprocessReplBackend

    client = fake_client()
    records = []
    child = subcalls(client, "same-model", len, records)

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
