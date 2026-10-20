"""Explicit CPU conversion of a verified two-rank compact Muon checkpoint to one rank."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

from kernel_trl.checkpoint_state import checkpoint_metadata, portable_adapter_name


def merge_parameter(shards, full_adapter, momenta):
    import torch

    if len(shards) != 2 or len(momenta) != 2:
        raise ValueError("Exactly two source shards are required")
    for shard, momentum in zip(shards, momenta, strict=True):
        if shard.dtype != full_adapter.dtype or momentum.dtype != shard.dtype:
            raise ValueError("Adapter or momentum dtype differs")
        if momentum.shape != shard.shape or not torch.isfinite(momentum).all():
            raise ValueError("Invalid Muon momentum shard")
    if all(shard.shape == full_adapter.shape for shard in shards):
        if not all(torch.equal(shard, full_adapter) for shard in shards):
            raise ValueError("Replicated adapters disagree with portable weights")
        if not torch.equal(momenta[0], momenta[1]):
            raise ValueError("Replicated momentum shards disagree")
        return momenta[0].clone(), "replicated"
    if any(shard.ndim != full_adapter.ndim for shard in shards):
        raise ValueError("Source shard dimensions differ")
    merged = torch.cat(shards, dim=0)
    if merged.shape != full_adapter.shape or not torch.equal(merged, full_adapter):
        raise ValueError("Row shards do not reconstruct the portable adapter exactly")
    return torch.cat(momenta, dim=0).contiguous(), "shard_dim0"


def convert_checkpoint(source: Path, destination: Path) -> dict:
    import torch
    from safetensors.torch import load_file

    original = json.loads((source / "metadata.json").read_text())
    step = original["step"]
    metadata = checkpoint_metadata(source, step, 2)
    required = {
        "adapter_model.safetensors",
        "adapter_config.json",
        "state-rank-0.pt",
        "state-rank-1.pt",
    }
    if not required <= set(metadata.get("sha256", {})):
        raise ValueError("Conversion requires checksums for every source checkpoint file")
    states = [
        torch.load(source / f"state-rank-{rank}.pt", map_location="cpu", weights_only=True)
        for rank in range(2)
    ]
    required_state = {
        "adapters",
        "momentum",
        "step",
        "scheduler",
        "python_rng",
        "numpy_rng",
        "rng",
        "cuda_rng",
    }
    if any(not required_state <= set(state) or state["step"] != step for state in states):
        raise ValueError("Source checkpoint lacks full training state at the expected step")
    if states[0]["scheduler"] != states[1]["scheduler"]:
        raise ValueError("Source scheduler states disagree")
    if any(group.get("step") != step for group in metadata["muon_groups"]):
        raise ValueError("Muon counter disagrees with policy step")
    names = set(states[0]["adapters"])
    if any(set(state["adapters"]) != names or set(state["momentum"]) != names for state in states):
        raise ValueError("Source adapter and momentum identities disagree")
    portable = load_file(source / "adapter_model.safetensors")
    canonical = {portable_adapter_name(name): value for name, value in portable.items()}
    expected = {portable_adapter_name(name) for name in names}
    if len(canonical) != len(portable) or len(expected) != len(names) or set(canonical) != expected:
        raise ValueError("Portable adapter identities do not match the local shards")
    converted = copy.deepcopy(states[0])
    placements = {}
    for name in sorted(names):
        full = canonical[portable_adapter_name(name)]
        momentum, placement = merge_parameter(
            [state["adapters"][name] for state in states],
            full,
            [state["momentum"][name] for state in states],
        )
        converted["adapters"][name] = full.clone()
        converted["momentum"][name] = momentum
        placements[name] = placement
    proof = {
        "source": str(source.resolve()),
        "source_metadata_sha256": hashlib.sha256(
            (source / "metadata.json").read_bytes()
        ).hexdigest(),
        "source_file_sha256": metadata["sha256"],
        "source_world_size": 2,
        "target_world_size": 1,
        "step": step,
        "adapters_verified_bitwise": len(names),
        "momentum_tensors_reconstructed": len(names),
        "source_placements_verified": placements,
        "rng_policy": "Preserve former rank-zero CPU/CUDA/Python/NumPy RNG streams; not bitwise-equivalent distributed execution",
        "scheduler_and_muon_counter_preserved": True,
        "optimizer_updates": 0,
    }
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copy2(source / name, destination / name)
    torch.save(converted, destination / "state-rank-0.pt")
    (destination / "conversion.json").write_text(json.dumps(proof, indent=2))
    updated = {**metadata, "world_size": 1, "conversion": "conversion.json"}
    updated["sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(destination.iterdir())
        if path.is_file()
    }
    (destination / "metadata.json").write_text(json.dumps(updated, indent=2))
    checkpoint_metadata(destination, step, 1)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(convert_checkpoint(arguments.source, arguments.destination)), flush=True)


if __name__ == "__main__":
    main()
