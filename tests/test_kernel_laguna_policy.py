import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SOURCE = (
    Path(__file__).parents[1] / "training/environments/kernel_rlm/kernel_training/laguna_policy.py"
)


@pytest.fixture
def policy(monkeypatch):
    specification = importlib.util.spec_from_file_location("laguna_policy_check", SOURCE)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16", float32="fp32"))
    return module


class Parameter:
    def __init__(self, trainable, dtype, shape):
        self.requires_grad = trainable
        self.dtype = dtype
        self.shape = shape
        self.ndim = len(shape)

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return 2 if self.dtype == "bf16" else 4


def model_fixture():
    parameters = [
        ("model.layers.0.self_attn.q_proj.base_layer.weight", Parameter(False, "bf16", (16, 16))),
        ("model.layers.0.self_attn.q_proj.lora_A.default.weight", Parameter(True, "fp32", (8, 16))),
        ("model.layers.0.self_attn.q_proj.lora_B.default.weight", Parameter(True, "fp32", (16, 8))),
    ]
    return SimpleNamespace(named_parameters=lambda: iter(parameters), parameters_list=parameters)


def test_storage_is_bf16_for_base_and_fp32_for_adapters(policy):
    inventory = policy.audit_storage(model_fixture())
    assert inventory["frozen_bytes"] == 512
    assert inventory["adapter_bytes"] == 1024
    assert inventory["adapter_tensors"] == 2


@pytest.mark.parametrize("index,dtype", [(0, "fp32"), (1, "bf16")])
def test_storage_rejects_wrong_master_dtype(policy, index, dtype):
    model = model_fixture()
    model.parameters_list[index][1].dtype = dtype
    with pytest.raises(ValueError):
        policy.audit_storage(model)


def test_muon_never_includes_frozen_weights(policy):
    model = model_fixture()
    groups = policy.muon_groups(model)
    assert groups[0]["params"] == [
        parameter for name, parameter in model.parameters_list if parameter.requires_grad
    ]
    assert groups[0]["algorithm"] == "muon"
    assert groups[0]["adjust_lr"] == "rms_norm"
    assert groups[0]["lr"] == 1e-5
    assert groups[0]["weight_decay"] == 0.01


def test_muon_rejects_training_dense_experts_or_nonmatrix_adapters(policy):
    parameter = Parameter(True, "fp32", (8, 16))
    model = SimpleNamespace(
        named_parameters=lambda: iter(
            [("model.layers.0.mlp.gate_proj.lora_A.default.weight", parameter)]
        )
    )
    with pytest.raises(ValueError, match="outside attention"):
        policy.muon_groups(model)
    parameter.shape = (8,)
    parameter.ndim = 1
    with pytest.raises(ValueError, match="Only attention LoRA matrices"):
        policy.muon_groups(model)


def test_pinned_muon_mesh_and_moonlight_scaling(policy, monkeypatch):
    captured = {}

    def create_muon(groups, **kwargs):
        captured.update(kwargs)
        captured["groups"] = groups
        return captured

    dion = ModuleType("dion")
    dion.Muon = create_muon
    muon = ModuleType("dion.muon")
    muon.adjust_lr_rms_norm = lambda lr, shape: lr * 0.2 * math.sqrt(max(shape))
    monkeypatch.setitem(sys.modules, "dion", dion)
    monkeypatch.setitem(sys.modules, "dion.muon", muon)
    mesh = SimpleNamespace(ndim=1, mesh_dim_names=("fsdp",))
    optimizer = policy.build_muon(model_fixture(), mesh)
    assert optimizer["world_mesh"] is mesh
    assert optimizer["distributed_mesh"] is mesh
    assert optimizer["fsdp_mesh_dim"] == 0
    assert optimizer["adjust_lr"] == "rms_norm"
    assert optimizer["mu"] == 0.95
    with pytest.raises(ValueError, match="one-dimensional"):
        policy.build_muon(model_fixture(), SimpleNamespace(ndim=2))
    muon.adjust_lr_rms_norm = lambda *args: 0.5
    with pytest.raises(ValueError, match="Moonlight RMS scaling"):
        policy.build_muon(model_fixture(), mesh)


def test_native_loader_is_strict_and_cpu_bf16(policy, monkeypatch):
    calls = []
    model = SimpleNamespace(config=SimpleNamespace(model_type="laguna"))
    model.config.rope_parameters = {"full_attention": {"rope_type": "default", "rope_theta": 10000}}
    monkeypatch.setattr(policy, "load_config", lambda path: model.config)
    information = {"missing_keys": set(), "unexpected_keys": set(), "mismatched_keys": set()}

    def load(*args, **kwargs):
        calls.append(kwargs)
        return model, information

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=load)),
    )
    returned, metadata = policy.load_base(Path("model"))
    assert returned is model
    assert metadata["missing_keys"] == []
    assert calls[0]["dtype"] == "bf16"
    assert calls[0]["device_map"] == "cpu"
    assert calls[0]["trust_remote_code"] is False
    information["missing_keys"].add("unloaded-expert")
    with pytest.raises(ValueError, match="Incomplete Laguna"):
        policy.load_base(Path("model"))


def test_loading_metadata_can_be_saved_without_weakening_validation(policy):
    assert policy.serializable_loading_info({"nested": {"missing": {"one"}}}) == {
        "nested": {"missing": ["one"]}
    }


def test_layer_specific_rope_survives_legacy_global_override(policy, monkeypatch, tmp_path):
    original = {
        "full_attention": {"rope_type": "yarn", "rope_theta": 500000, "partial_rotary_factor": 0.5},
        "sliding_attention": {
            "rope_type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 1.0,
        },
    }
    (tmp_path / "config.json").write_text(
        json.dumps({"rope_parameters": original, "partial_rotary_factor": 0.5})
    )
    parsed = json.loads(json.dumps(original))
    parsed["sliding_attention"]["partial_rotary_factor"] = 0.5
    configuration = SimpleNamespace(
        model_type="laguna",
        layer_types=["full_attention", "sliding_attention"],
        rope_parameters=parsed,
        partial_rotary_factor=0.5,
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=SimpleNamespace(from_pretrained=lambda *args, **kwargs: configuration)
        ),
    )
    result = policy.load_config(tmp_path)
    assert result.rope_parameters == original
    assert result.rope_parameters["sliding_attention"]["partial_rotary_factor"] == 1
    assert not hasattr(result, "partial_rotary_factor")
    assert json.loads((tmp_path / "config.json").read_text())["partial_rotary_factor"] == 0.5


def test_missing_per_type_rope_is_not_silently_guessed(policy, monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}")
    configuration = SimpleNamespace(model_type="laguna", layer_types=["full_attention"])
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=SimpleNamespace(from_pretrained=lambda *args, **kwargs: configuration)
        ),
    )
    with pytest.raises(ValueError, match="per-layer-type"):
        policy.load_config(tmp_path)


def test_readonly_legacy_alias_is_not_deleted(policy, monkeypatch, tmp_path):
    parameters = {"full_attention": {"rope_type": "default", "rope_theta": 10000}}
    (tmp_path / "config.json").write_text(json.dumps({"rope_parameters": parameters}))

    class Configuration:
        model_type = "laguna"
        layer_types = ["full_attention"]

        @property
        def rope_scaling(self):
            return self.rope_parameters

    parsed = Configuration()
    parsed.rope_parameters = parameters
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=lambda *args, **kwargs: parsed)),
    )
    assert policy.load_config(tmp_path).rope_scaling == parameters
