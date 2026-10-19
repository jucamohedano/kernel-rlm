import asyncio
import importlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "training/environments/kernel_rlm"))
    return SimpleNamespace(
        context=importlib.import_module("kernel_trl.context"),
        memory=importlib.import_module("kernel_trl.memory_probe"),
        collect=importlib.import_module("kernel_trl.collect"),
        curriculum=importlib.import_module("kernel_trl.live_curriculum"),
        credit=importlib.import_module("kernel_trl.turn_credit"),
    )


@pytest.mark.parametrize("value", [0, 12287, 16385, 16384.0, True])
def test_context_bounds_are_explicit(modules, value):
    with pytest.raises(ValueError):
        modules.context.validate_context_tokens(value)


def test_default_leaves_margin_below_the_tight_16k_probe(modules):
    assert modules.context.DEFAULT_CONTEXT_TOKENS == 14336
    assert modules.context.validate_context_tokens(16384) == 16384


def test_server_and_credit_share_the_selected_context(modules, tmp_path):
    arguments = SimpleNamespace(
        inference_python="python", checkpoint=tmp_path, context_tokens=16384
    )
    command = modules.curriculum.server_command(arguments, 0, None)
    assert command[command.index("--max-model-len") + 1] == "16384"
    from kernel_skyrl.contract import RootSample

    episodes = [
        {"reward": reward, "samples": [RootSample([2] * 13000, [3], [-1.0], "stop")]}
        for reward in [0.1, 0.8, 0.1, 0.9]
    ]
    with pytest.raises(ValueError, match="configured limit"):
        modules.credit.training_rows(episodes)
    result = modules.credit.training_rows(episodes, context_tokens=16384)
    assert result["context_tokens"] == 16384
    assert len(result["rows"][0]["prompt_ids"]) == 13000


def test_exact_token_transport_uses_selected_limit(modules, tmp_path):
    class Response:
        def __init__(self, payload):
            self.payload = payload
            self.status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        async def json(self):
            return self.payload

    tokens = [2] * 13000
    session = SimpleNamespace(
        post=Mock(
            side_effect=[
                Response({"count": 13000, "tokens": tokens}),
                Response({"prompt_token_ids": tokens}),
            ]
        )
    )
    client = modules.collect.InferenceClient(session, "http://unused", tmp_path / "requests", 16384)
    asyncio.run(client.request_completion({"json": {"model": "policy", "max_tokens": 4096}}))
    request = session.post.call_args_list[1].kwargs["json"]
    assert request["max_tokens"] == 3384


def test_memory_probe_has_no_optimizer_step_or_checkpoint_write(modules, monkeypatch):
    group = modules.memory.probe_group(16384, 5)
    row = group["rows"][0]
    assert len(row["prompt_ids"]) + len(row["completion_ids"]) == 16384
    assert len(row["completion_ids"]) == 4096
    trainer = SimpleNamespace(
        model=SimpleNamespace(train=Mock()),
        args=SimpleNamespace(gradient_accumulation_steps=2),
        training_step=Mock(return_value=1.0),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            isfinite=math.isfinite,
            cuda=SimpleNamespace(synchronize=Mock(), mem_get_info=lambda: (100, 200)),
        ),
    )
    result = modules.memory.measure_backward(trainer, group)
    assert result["optimizer_steps"] == 0
    assert trainer.training_step.call_count == 2
    assert trainer.current_gradient_accumulation_steps == 2
    assert result["synthetic_inputs_not_rollouts"] is True


def test_resume_refuses_silent_context_expansion(modules, tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    dataset = tmp_path / "dataset"
    dataset.write_bytes(b"tasks")
    import hashlib

    (output / "state.json").write_text(json.dumps({"status": "paused"}))
    (output / "inputs.json").write_text(
        json.dumps(
            {
                "dataset_sha256": hashlib.sha256(b"tasks").hexdigest(),
                "source_commit": "source",
                "context_tokens": 12288,
            }
        )
    )
    monkeypatch.setattr(modules.curriculum, "ensure_gpus_idle", lambda: None)
    arguments = SimpleNamespace(
        output_dir=output, dataset=dataset, source_commit="source", context_tokens=16384
    )
    with pytest.raises(ValueError, match="immutable run context"):
        modules.curriculum.load_state(arguments)
