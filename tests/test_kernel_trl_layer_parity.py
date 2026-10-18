import functools
import importlib
import pickle
from pathlib import Path

import pytest


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "training/environments/kernel_rlm"))
    return importlib.import_module("kernel_trl.layer_parity")


def test_positions_cover_start_window_boundary_and_reply(monkeypatch):
    assert module(monkeypatch).capture_positions(2880, 64) == [0, 1, 512, 2879, 2880]
    assert module(monkeypatch).capture_positions(8, 8) == [0, 1, 7, 8]


@pytest.mark.parametrize("prompt,reply", [(1, 8), (8, 0)])
def test_empty_or_invalid_capture_is_rejected(monkeypatch, prompt, reply):
    with pytest.raises(ValueError):
        module(monkeypatch).capture_positions(prompt, reply)


def test_worker_callback_uses_an_importable_module_not_main(monkeypatch):
    callback = functools.partial(
        module(monkeypatch).install_vllm_capture, directory="/tmp/probe", selected=[0, 1]
    )
    restored = pickle.loads(pickle.dumps(callback))
    assert restored.func is module(monkeypatch).install_vllm_capture
    assert restored.func.__module__ == "kernel_trl.layer_parity"
