import ast
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1] / "training/environments/kernel_rlm"


@pytest.fixture
def phase_state(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    return importlib.import_module("kernel_trl.phase_state")


def test_muon_state_migration_keeps_parameter_keys(phase_state, monkeypatch):
    class Tensor:
        def to(self, device):
            return ("moved", device)

    torch = ModuleType("torch")
    torch.Tensor = Tensor
    monkeypatch.setitem(sys.modules, "torch", torch)
    parameter = object()
    optimizer = SimpleNamespace(state={parameter: {"momentum": Tensor(), "step": 1}})
    assert phase_state.move_optimizer_state(optimizer, "cpu") == 1
    assert list(optimizer.state) == [parameter]
    assert optimizer.state[parameter] == {"momentum": ("moved", "cpu"), "step": 1}


def test_muon_state_migration_rejects_unknown_structure(phase_state, monkeypatch):
    torch = ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    monkeypatch.setitem(sys.modules, "torch", torch)
    with pytest.raises(ValueError, match="Unexpected optimizer state"):
        phase_state.move_optimizer_state(SimpleNamespace(state={object(): {"momentum": []}}), "cpu")


def test_phase_probe_is_bounded_native_generation_not_custom_rollout():
    tree = ast.parse((ROOT / "kernel_trl/phase_probe.py").read_text())
    config = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "GRPOConfig"
    )
    settings = {keyword.arg: keyword.value for keyword in config.keywords}
    assert ast.literal_eval(settings["max_steps"]) == 2
    assert ast.literal_eval(settings["max_completion_length"]) == 8
    assert ast.literal_eval(settings["use_vllm"]) is False
    assert ast.literal_eval(settings["save_strategy"]) == "no"
    assert ast.literal_eval(settings["generation_kwargs"])["min_new_tokens"] == 8
    assert ast.literal_eval(settings["chat_template_kwargs"]) == {"enable_thinking": True}
    assert not any(
        isinstance(node, ast.keyword) and node.arg == "rollout_func" for node in ast.walk(tree)
    )
    assert all(
        not name.name.startswith("_")
        for name in ast.walk(tree)
        if isinstance(name, ast.FunctionDef)
    )


def test_full_prompt_stays_structured_until_native_template_tokenization():
    tree = ast.parse((ROOT / "kernel_trl/phase_probe.py").read_text())
    builder = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "tokenizer_and_prompt"
    )
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "apply_chat_template"
        for node in ast.walk(builder)
    )
    assert any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Tuple)
        and isinstance(node.value.elts[-1], ast.Name)
        and node.value.elts[-1].id == "messages"
        for node in ast.walk(builder)
    )
