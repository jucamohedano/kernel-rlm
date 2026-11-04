import asyncio
import json

import pytest

from cute_rlm.trajectory_journal import TrajectoryJournal
from cute_rlm.modal_collect import (
    BASE_URL, MODEL, ModalAmbiguousStop, ModalAuthenticationStop,
    ModalCollectionRuntime, ModalRatePause, ModalRunQueue, modal_manifest,
)


class Response:
    def __init__(self, status=200, body=None):
        self.status_code, self.body = status, body or {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 5}}
        self.text = json.dumps(self.body)
    def json(self): return self.body

class Client:
    def __init__(self, response=None): self.calls, self.response = [], response or Response()
    async def post(self, path, **kwargs): self.calls.append((path, kwargs)); return self.response


def runtime(tmp_path, response=None):
    obj = object.__new__(ModalCollectionRuntime)
    obj.client = Client(response)
    obj.modal_journal = TrajectoryJournal(tmp_path / "modal.jsonl", tmp_path / "modal-progress.json")
    obj.task_id, obj.episode_id = "task", "episode"
    return obj


def test_native_constructor_accepts_positional_model(tmp_path):
    journal = TrajectoryJournal(tmp_path / "journal.jsonl", tmp_path / "progress.json")
    rt = ModalCollectionRuntime(Client(), "framework", object(), journal=journal)
    assert rt.model == MODEL


def test_openai_request_and_raw_response_are_durable(tmp_path):
    async def run():
        rt = runtime(tmp_path)
        raw = await rt.fetch_response(None, {"call_id": "c1", "branch": "root"}, {
            "model": "framework", "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 12345, "chat_template_kwargs": {"enable_thinking": True},
            "tools": [{"type": "function", "function": {"name": "repl"}}],
        })
        assert raw["usage"]["prompt_tokens"] == 5
        path, call = rt.client.calls[0]
        assert path == "chat/completions"
        assert call["json"] == {"model": MODEL, "messages": [{"role": "user", "content": "x"}],
                               "tools": [{"type": "function", "function": {"name": "repl"}}],
                               "reasoning_effort": "high", "temperature": 0.3, "top_p": 0.9}
        assert call["headers"] == {"X-RLM-Call-Id": "c1"}
        rows = [json.loads(x) for x in (tmp_path / "modal.jsonl").read_text().splitlines()]
        assert rows[0]["intent"]["framework_max_tokens_removed"] == 12345
        assert rows[1]["raw_response"] == raw
        assert rows[2]["outcome"] == "response_received"
    asyncio.run(run())


@pytest.mark.parametrize("status,error", [(401, ModalAuthenticationStop), (403, ModalAuthenticationStop),
                                           (429, ModalRatePause), (500, ModalAmbiguousStop),
                                           (418, ModalAmbiguousStop)])
def test_provider_failure_is_retained_and_never_retried(tmp_path, status, error):
    async def run():
        rt = runtime(tmp_path, Response(status, {"error": "stop"}))
        with pytest.raises(error):
            await rt.fetch_response(None, {"call_id": "c1", "branch": "root"}, {"messages": []})
        assert len(rt.client.calls) == 1
        rows = [json.loads(x) for x in (tmp_path / "modal.jsonl").read_text().splitlines()]
        assert rows[-2]["raw_response"]["http_status"] == status
        assert rows[-1]["outcome"] == "http_error"
    asyncio.run(run())


def test_modal_queue_is_distinct_and_never_replays_interrupted(tmp_path):
    path = tmp_path / "modal-run.jsonl"
    queue = ModalRunQueue(path)
    task = queue.enqueue({"source": "public"})
    queue.transition(task, "running")
    restored = ModalRunQueue(path)
    assert restored.items[task]["status"] == "interrupted_uncertain"
    assert modal_manifest(task_ids=[task])["base_url"] == BASE_URL
    assert modal_manifest(task_ids=[task])["model"] == MODEL
