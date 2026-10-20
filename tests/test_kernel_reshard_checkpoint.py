import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1] / "training/environments/kernel_rlm"


@pytest.fixture(autouse=True)
def imports(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))


def test_one_rank_checkpoint_requires_explicit_selection(tmp_path):
    from kernel_trl.checkpoint_state import checkpoint_metadata

    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "step": 17,
                "world_size": 1,
                "frozen_weights_saved": False,
            }
        )
    )
    with pytest.raises(ValueError, match="world size"):
        checkpoint_metadata(tmp_path, 17)
    assert checkpoint_metadata(tmp_path, 17, 1)["world_size"] == 1
    with pytest.raises(ValueError, match="explicitly"):
        checkpoint_metadata(tmp_path, 17, 4)


@pytest.fixture
def source(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    directory = tmp_path / "source"
    directory.mkdir()
    full = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    name = "layer.lora_A.default.weight"
    for rank in range(2):
        shard = full[rank * 3 : (rank + 1) * 3].clone()
        torch.save(
            {
                "step": 17,
                "adapters": {name: shard},
                "momentum": {name: shard + 2},
                "scheduler": {"last_epoch": 17},
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.get_rng_state(),
                "python_rng": (),
                "numpy_rng": (),
            },
            directory / f"state-rank-{rank}.pt",
        )
    save_file({"layer.lora_A.weight": full}, directory / "adapter_model.safetensors")
    (directory / "adapter_config.json").write_text("{}")
    refresh_manifest(directory)
    return directory


def refresh_manifest(directory):
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "step": 17,
                "world_size": 2,
                "frozen_weights_saved": False,
                "adapter_tensors": 1,
                "muon_groups": [{"step": 17}],
                "sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in directory.iterdir()
                    if path.name != "metadata.json"
                },
            }
        )
    )


def test_conversion_preserves_adapters_momentum_and_source(source, tmp_path):
    import torch
    from kernel_trl.reshard_checkpoint import convert_checkpoint

    source_hash = hashlib.sha256((source / "metadata.json").read_bytes()).hexdigest()
    proof = convert_checkpoint(source, tmp_path / "converted")
    state = torch.load(tmp_path / "converted/state-rank-0.pt", weights_only=True)
    full = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    assert torch.equal(state["adapters"]["layer.lora_A.default.weight"], full)
    assert torch.equal(state["momentum"]["layer.lora_A.default.weight"], full + 2)
    assert proof["optimizer_updates"] == 0
    assert proof["scheduler_and_muon_counter_preserved"]
    assert hashlib.sha256((source / "metadata.json").read_bytes()).hexdigest() == source_hash


def test_conversion_rejects_corrupt_shard_without_output(source, tmp_path):
    import torch
    from kernel_trl.reshard_checkpoint import convert_checkpoint

    path = source / "state-rank-1.pt"
    state = torch.load(path, weights_only=True)
    state["adapters"]["layer.lora_A.default.weight"].add_(1)
    torch.save(state, path)
    refresh_manifest(source)
    with pytest.raises(ValueError, match="reconstruct"):
        convert_checkpoint(source, tmp_path / "converted")
    assert not (tmp_path / "converted").exists()


def test_replicated_momentum_must_agree():
    torch = pytest.importorskip("torch")
    from kernel_trl.reshard_checkpoint import merge_parameter

    full = torch.ones(4, 4)
    with pytest.raises(ValueError, match="Replicated momentum"):
        merge_parameter([full, full], full, [full, full * 2])
    momentum, placement = merge_parameter([full, full], full, [full, full])
    assert placement == "replicated"
    assert torch.equal(momentum, full)


def test_nonfinite_momentum_is_rejected():
    torch = pytest.importorskip("torch")
    from kernel_trl.reshard_checkpoint import merge_parameter

    full = torch.ones(4, 4)
    with pytest.raises(ValueError, match="momentum"):
        merge_parameter([full[:2], full[2:]], full, [full[:2], full[2:] * float("nan")])
