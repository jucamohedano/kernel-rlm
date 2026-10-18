import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SOURCE = Path(__file__).parents[1] / "training/environments/kernel_rlm"
sys.path.insert(0, str(SOURCE))
hooks = importlib.import_module("kernel_skyrl.framework_hooks")
enable_lazy_padding_import = hooks.enable_lazy_padding_import
trl_muon_trainer = hooks.trl_muon_trainer


def test_lazy_padding_is_guarded_and_does_not_fake_flash_attention(tmp_path):
    path = tmp_path / "skyrl/backends/skyrl_train/workers/model_wrapper.py"
    path.parent.mkdir(parents=True)
    path.write_text("from flash_attn.bert_padding import pad_input, unpad_input\n")
    result = enable_lazy_padding_import(tmp_path)
    assert result["original_sha256"] != result["patched_sha256"]
    specification = importlib.util.spec_from_file_location("padding_gate", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    assert callable(module.pad_input) and callable(module.unpad_input)
    with pytest.raises(ImportError):
        module.pad_input(None)
    with pytest.raises(ValueError, match="exact pinned"):
        enable_lazy_padding_import(tmp_path)


def test_trl_public_optimizer_hook_never_uses_default_optimizer(monkeypatch):
    from kernel_skyrl import framework_hooks

    module = ModuleType("trl")
    module.GRPOTrainer = object
    monkeypatch.setitem(sys.modules, "trl", module)
    calls = []
    sentinel = object()
    monkeypatch.setattr(framework_hooks, "build_muon", lambda *args: calls.append(args) or sentinel)
    observed = []
    trainer = trl_muon_trainer("mesh", observed.append)()
    trainer.optimizer = None
    trainer.model = "model"
    trainer.args = SimpleNamespace(learning_rate=1e-5, weight_decay=0.01)
    assert trainer.create_optimizer() is sentinel
    assert trainer.create_optimizer() is sentinel
    assert calls == [("model", "mesh", 1e-5, 0.01)]
    assert observed == [sentinel]


@pytest.mark.parametrize("fail_prepare", [False, True])
def test_skyrl_suppresses_adamw_and_restores_configuration(monkeypatch, fail_prepare):
    class NativeStrategy:
        def prepare(self, pair):
            assert self.optimizer_config is None
            if fail_prepare:
                raise RuntimeError("fixture load failure")
            return pair

    strategy_module = ModuleType("skyrl.backends.skyrl_train.distributed.fsdp_strategy")
    strategy_module.FSDPStrategy = NativeStrategy
    monkeypatch.setitem(sys.modules, strategy_module.__name__, strategy_module)
    transformers = ModuleType("transformers")
    transformers.get_scheduler = lambda *args, **kwargs: "scheduler"
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    calls = []
    monkeypatch.setattr(hooks, "build_muon", lambda *args: calls.append(args) or "muon")
    strategy = hooks.skyrl_muon_strategy()()
    settings = SimpleNamespace(lr=1e-5, weight_decay=0.01, scheduler="constant", num_warmup_steps=0)
    strategy.optimizer_config = settings
    strategy.device_mesh = "mesh"
    strategy.total_training_steps = 1
    if fail_prepare:
        with pytest.raises(RuntimeError, match="fixture load"):
            strategy.prepare(("model", None, None))
        assert not calls
    else:
        assert strategy.prepare(("model", None, None)) == ("model", "muon", "scheduler")
        assert calls == [("model", "mesh", 1e-5, 0.01)]
        with pytest.raises(ValueError, match="pre-existing"):
            strategy.prepare(("model", "adamw", None))
    assert strategy.optimizer_config is settings


@pytest.mark.parametrize("rank", [0, 1])
def test_meta_buffer_fix_preserves_only_nonpersistent_metadata(monkeypatch, rank):
    class Buffer:
        def __init__(self, meta):
            self.is_meta = meta
            self.shape = (32,)
            self.dtype = "fp32"

        def detach(self):
            return self

        def cuda(self):
            assert not self.is_meta
            return self

        def cpu(self):
            return self

    buffer = Buffer(meta=rank == 1)
    registered = {}
    owner = SimpleNamespace(
        _non_persistent_buffers_set={"rope", "absent"},
        _buffers={"rope": buffer, "absent": None, "persistent": object()},
        register_buffer=lambda name, value, persistent: registered.update(
            {name: (value, persistent)}
        ),
    )
    distributed = ModuleType("torch.distributed")
    distributed.get_rank = lambda: rank
    calls = []
    distributed.broadcast = lambda value, src: calls.append((value, src))
    torch = ModuleType("torch")
    torch.distributed = distributed
    torch.empty = lambda *args, **kwargs: Buffer(meta=False)
    torch.isfinite = lambda value: SimpleNamespace(all=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    model = SimpleNamespace(named_modules=lambda: [("model.rotary_emb", owner)])
    assert hooks.materialize_nonpersistent_buffers(model) == ["model.rotary_emb.rope"]
    assert len(calls) == 1 and calls[0][1] == 0
    assert not registered["rope"][0].is_meta
    assert registered["rope"][1] is False
    assert "persistent" not in registered


def test_only_meta_adapters_receive_small_cpu_placeholder_storage(monkeypatch):
    parameters = {
        "adapter": SimpleNamespace(requires_grad=True, is_meta=True, shape=(8, 16), dtype="fp32"),
        "base": SimpleNamespace(requires_grad=False, is_meta=True, shape=(128, 128), dtype="bf16"),
        "existing_adapter": SimpleNamespace(requires_grad=True, is_meta=False),
    }
    registered = {}
    owner = SimpleNamespace(
        named_parameters=lambda recurse: list(parameters.items()),
        register_parameter=lambda name, value: registered.update({name: value}),
    )
    torch = ModuleType("torch")
    torch.empty = lambda shape, dtype, device: (shape, dtype, device)
    torch.nn = SimpleNamespace(Parameter=lambda value: value)
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert hooks.materialize_meta_adapters(SimpleNamespace(modules=lambda: [owner])) == 1
    assert registered == {"adapter": ((8, 16), "fp32", "cpu")}
