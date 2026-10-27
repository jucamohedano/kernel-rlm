import asyncio
import json
import unittest
from types import SimpleNamespace as NS

from kernel_rlm.child_tools import IterativeSubcalls
from kernel_rlm.subcalls import BoundedSubcalls


class CompletionClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.chat = NS(completions=self)

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        item = next(self.responses)
        if isinstance(item, BaseException):
            raise item
        return item


def response(content, tokens=10, finish="stop"):
    return NS(
        choices=[
            NS(
                finish_reason=finish,
                message=NS(
                    content=content, tool_calls=None, model_dump=lambda **_: {"content": content}
                ),
            )
        ],
        usage=NS(completion_tokens=tokens, model_dump=lambda: {"completion_tokens": tokens}),
    )


def candidate(action, code="x=1"):
    return f"{action}\nfindings\n```python\n{code}\n```"


def test_multi_turn_trace_keeps_raw_actions_and_observations(tmp_path):
    client = CompletionClient([response(candidate("CHECK")), response(candidate("FINISH"))])

    async def checker(reference, source, directory, timeout):
        directory.mkdir()
        return {"compiled": True, "correct": True}

    calls = IterativeSubcalls(
        client, "model", lambda _: 1, [], reference="ref", directory=tmp_path, checker=checker
    )
    result = asyncio.run(
        calls.query("question", {"parent_turn_id": "root-turn", "parent_action_id": "root-act"})
    )
    record = calls.records[0]
    assert json.loads(result)["advice"] == "findings"
    assert record["completion_tokens"] == 20
    assert record["parent_turn_id"] == "root-turn"
    assert record["turns"][0]["outcome"] == "check_executed"
    assert record["turns"][0]["observation_id"] == record["checks"][0]["observation_id"]
    assert client.requests[0]["max_tokens"] == 4096
    assert record["termination_reason"] == "finish"


def test_malformed_action_is_preserved_and_not_executed(tmp_path):
    client = CompletionClient(
        [response("<tool_call>literal malformed"), response(candidate("FINISH"))]
    )
    calls = IterativeSubcalls(client, "model", lambda _: 1, [], reference="ref", directory=tmp_path)
    asyncio.run(calls.query("question"))
    record = calls.records[0]
    assert record["turns"][0]["raw_reply"]["content"].startswith("<tool_call>")
    assert record["turns"][0]["outcome"] == "malformed_action"
    assert not record["checks"]


def test_length_usage_failure_and_parent_link_are_recorded(tmp_path):
    client = CompletionClient([response("partial", tokens=4096, finish="length")])
    calls = IterativeSubcalls(
        client,
        "model",
        lambda _: 1,
        [],
        reference="ref",
        directory=tmp_path,
        child_reply_tokens=4096,
        child_total_tokens=4096,
    )
    with unittest.TestCase().assertRaisesRegex(ValueError, "length cap"):
        asyncio.run(calls.query("question", {"parent_turn_id": "parent"}))
    record = calls.records[0]
    assert record["termination_reason"] == "length"
    assert record["completion_tokens"] == 4096
    assert record["turns"][0]["parent_turn_id"] == "parent"


def test_missing_usage_terminates_unknown_not_zero(tmp_path):
    item = response("reply")
    item.usage = None
    client = CompletionClient([item])
    calls = IterativeSubcalls(client, "model", lambda _: 1, [], reference="ref", directory=tmp_path)
    with unittest.TestCase().assertRaisesRegex(ValueError, "usage unavailable"):
        asyncio.run(calls.query("question"))
    assert calls.records[0]["completion_tokens"] is None
    assert calls.records[0]["usage_accounting"] == "incomplete"


def test_iterative_caps_above_4096_are_configurable(tmp_path):
    client = CompletionClient([response(candidate("FINISH"), tokens=5)])
    calls = IterativeSubcalls(
        client,
        "model",
        lambda _: 1,
        [],
        reference="ref",
        directory=tmp_path,
        child_reply_tokens=8192,
        child_total_tokens=12288,
        child_prompt_tokens=5000,
        context_tokens=9000,
    )
    asyncio.run(calls.query("question"))
    assert client.requests[0]["max_tokens"] <= 8192
    assert client.requests[0]["max_tokens"] > 4096
    assert calls.records[0]["child_total_tokens"] == 12288


def test_iterative_caps_reject_bool(tmp_path):
    with unittest.TestCase().assertRaisesRegex(ValueError, "positive integers"):
        IterativeSubcalls(
            CompletionClient([]),
            "model",
            lambda _: 1,
            [],
            reference="ref",
            directory=tmp_path,
            child_reply_tokens=True,
        )


def test_bridge_routes_root_and_configured_child_caps():
    import tempfile
    from pathlib import Path

    from kernel_rollout.bridge import CompletionBridge

    bridge = CompletionBridge(
        object(),
        "adapter",
        16384,
        4096,
        0,
        Path(tempfile.gettempdir()) / "trace.jsonl",
        child_reply_limit=8192,
        child_total_limit=12288,
    )
    assert bridge.output_limit(20000, child=False) == 4096
    assert bridge.output_limit(20000, child=True) == 8192
    assert bridge.output_limit(7000, child=True) == 7000


class OneShotClient(CompletionClient):
    pass


def one_shot_response(content="reply", usage=10):
    result = response(
        content, tokens=usage if isinstance(usage, int) and not isinstance(usage, bool) else 10
    )
    result.usage = (
        None
        if usage is None
        else NS(completion_tokens=usage, model_dump=lambda: {"completion_tokens": usage})
    )
    return result


def test_one_shot_preserves_raw_reply_before_bad_usage(tmp_path):
    for usage in (None, 0, -1, True, 5000):
        client = OneShotClient([one_shot_response(usage=usage)])
        records = []
        subcalls = BoundedSubcalls(
            client, "model", lambda _: 1, records, child_reply_tokens=4096, child_total_tokens=4096
        )
        with unittest.TestCase().assertRaises(ValueError):
            asyncio.run(subcalls.query("question"))
        assert records[0]["turns"][0]["raw_reply"]["content"] == "reply"
        assert records[0]["status"] == "failed"


def test_one_shot_output_defaults_remain_4096(tmp_path):
    client = OneShotClient([one_shot_response()])
    subcalls = BoundedSubcalls(client, "model", lambda _: 1, [], prompt_tokens=2048)
    asyncio.run(subcalls.query("question"))
    assert client.requests[0]["max_tokens"] == 4096


def test_provider_failure_has_unknown_usage_and_parent_lineage(tmp_path):
    client = CompletionClient([RuntimeError("backend rejected request")])
    calls = IterativeSubcalls(client, "model", lambda _: 1, [], reference="ref", directory=tmp_path)
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "backend rejected"):
        asyncio.run(calls.query("question", {"parent_turn_id": "parent"}))
    record = calls.records[0]
    assert record["completion_tokens"] is None
    assert record["usage_accounting"] == "incomplete"
    assert record["parent_turn_id"] == "parent"
    assert record["termination_reason"] == "error"


def test_iterative_request_cap_is_limited_by_active_context(tmp_path):
    client = CompletionClient([response(candidate("FINISH"), tokens=5)])
    calls = IterativeSubcalls(
        client,
        "model",
        lambda value: len(value),
        [],
        reference="ref",
        directory=tmp_path,
        child_reply_tokens=8192,
        child_total_tokens=8192,
        child_prompt_tokens=5000,
        context_tokens=10000,
    )
    asyncio.run(calls.query("question"))
    request = client.requests[0]
    assert request["max_tokens"] < 8192
    assert calls.records[0]["turns"][0]["context_output_allowance"] < 8192
