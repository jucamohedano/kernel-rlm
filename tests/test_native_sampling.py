import time
from types import SimpleNamespace

import pytest
from rlm_train.native import NativeRuntime


class Response:
    def raise_for_status(self):
        pass

    def json(self):
        return {"usage": {"completion_tokens": 1}, "choices": []}


class Client:
    def __init__(self):
        self.payloads = []

    async def post(self, endpoint, *, json):
        self.payloads.append(json)
        return Response()


@pytest.mark.asyncio
async def test_sampling_applies_to_root_child_and_plain_requests_and_checkpoint():
    client = Client()
    sampling = {"temperature": 0.6, "top_p": 0.95, "seed": 20261101}
    runtime = NativeRuntime(client, "model", object(), sampling_args=sampling)
    sampling["seed"] = 0
    runtime.deadline = time.monotonic() + 5
    for identity, depth, plain in (("root", 0, False), ("child", 1, False), ("plain", 1, True)):
        session = SimpleNamespace(
            id=identity,
            parent=None if not depth else "root",
            depth=depth,
            origin={},
            current_action={},
            record={"events": []},
            requests=set(),
            accepting=True,
        )
        await runtime.request(session, [], "model", plain=plain)
    for payload in client.payloads:
        assert {key: payload[key] for key in ("temperature", "top_p", "seed")} == {
            "temperature": 0.6,
            "top_p": 0.95,
            "seed": 20261101,
        }
    assert runtime.checkpoint_config()["sampling_args"]["seed"] == 20261101
    default = NativeRuntime(Client(), "model", object())
    assert default.sampling_args == {}


@pytest.mark.parametrize(
    "sampling",
    [
        {"model": "other"},
        {"max_tokens": 1},
        {"temperature": True},
        {"temperature": float("nan")},
        {"temperature": 3},
        {"top_p": 0},
        {"top_p": 1.1},
        {"seed": True},
        {"seed": -1},
        {"seed": 2**32},
    ],
)
def test_sampling_rejects_invalid_or_non_sampling_overrides(sampling):
    with pytest.raises(ValueError):
        NativeRuntime(Client(), "model", object(), sampling_args=sampling)
