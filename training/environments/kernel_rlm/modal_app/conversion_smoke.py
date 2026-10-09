"""CPU-only preservation checks for prime-rl's pinned Laguna converter."""

from __future__ import annotations

import json
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import modal

from .attention_lora import MODEL_ID, PRIME_REVISION, REVISION, image, models, reports

SOURCES = {
    "conversion_ops.py": "b487a69ba20f4163a751d2e00219022dda8368321dd0d24fd4a8de19d9d13c02",
    "laguna/converting_laguna.py": "269f0fc39a9e34c41a09ab9b83117fd9094feed80aad673c539453301d5d1102",
}
FETCH_SCRIPT = f"""
import hashlib
import urllib.request
from pathlib import Path
for relative, expected in {SOURCES!r}.items():
    url = 'https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/{PRIME_REVISION}/src/prime_rl/trainer/models/' + relative
    data = urllib.request.urlopen(url, timeout=30).read()
    assert hashlib.sha256(data).hexdigest() == expected, relative
    destination = Path('/opt/prime-conversion') / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
"""
conversion_image = image.run_commands(f"python -c {shlex.quote(f'exec({FETCH_SCRIPT!r})')}")
app = modal.App("kernel-rlm-prime-conversion-check")


def import_source(name: str, relative: str) -> Any:
    import hashlib
    import importlib.util
    import sys

    path = Path("/opt/prime-conversion") / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCES[relative]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def destination_for(key: str) -> tuple[str, int | None]:
    import re

    weights = {"gate": "w1", "down": "w2", "up": "w3"}
    expert = re.fullmatch(
        r"(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate|down|up)_proj\.weight", key
    )
    if expert:
        return f"{expert[1]}.{weights[expert[3]]}", int(expert[2])
    shared = re.fullmatch(
        r"(model\.layers\.\d+)\.mlp\.shared_expert\.(gate|down|up)_proj\.weight", key
    )
    if shared:
        return f"{shared[1]}.shared_expert.{weights[shared[2]]}.weight", None
    if key.endswith(".mlp.gate.weight"):
        return key.removesuffix("gate.weight") + "router.gate.weight", None
    if key.endswith(".mlp.experts.e_score_correction_bias"):
        return key.removesuffix("experts.e_score_correction_bias") + "expert_bias", None
    return key, None


def check_group(
    snapshot: Path, weight_map: dict[str, str], names: list[str], ops: Any, chain: Any
) -> dict[str, int]:
    from contextlib import ExitStack

    import torch
    from safetensors import safe_open

    with ExitStack() as stack:
        shards = {
            filename: stack.enter_context(
                safe_open(snapshot / filename, framework="pt", device="cpu")
            )
            for filename in {weight_map[name] for name in names}
        }
        original = {name: shards[weight_map[name]].get_tensor(name) for name in names}
        converted = ops.apply_hf_to_prime(dict(original), chain)
        assert set(converted) == {destination_for(name)[0] for name in names}
        for name, value in original.items():
            destination, expert_index = destination_for(name)
            actual = converted[destination]
            if expert_index is not None:
                actual = actual[expert_index]
            assert actual.shape == value.shape and actual.dtype == value.dtype, name
            assert torch.equal(
                actual.contiguous().view(torch.uint8), value.contiguous().view(torch.uint8)
            ), name
        ops.apply_prime_to_hf(converted, chain)
        assert set(converted) == set(original)
        for name, value in original.items():
            assert torch.equal(
                converted[name].contiguous().view(torch.uint8), value.contiguous().view(torch.uint8)
            ), name
        return {
            "tensors": len(original),
            "parameters": sum(value.numel() for value in original.values()),
            "bytes": sum(value.numel() * value.element_size() for value in original.values()),
        }


def persist(run_id: str, report: dict[str, Any]) -> None:
    path = Path("/reports/prime-conversion") / run_id / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    reports.commit()


@app.function(
    image=conversion_image,
    volumes={"/models": models, "/reports": reports},
    cpu=4,
    memory=24576,
    timeout=600,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def validate(run_id: str) -> dict[str, Any]:
    import gc
    import struct
    import time
    import traceback
    from types import SimpleNamespace

    import torch
    from huggingface_hub import snapshot_download

    started = time.monotonic()
    torch.set_num_threads(4)
    report: dict[str, Any] = {
        "status": "running",
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "prime_revision": PRIME_REVISION,
        "source_sha256": SOURCES,
        "scope": "Exact upstream converter, all checkpoint tensors; no full prime model or trainer instantiated",
        "gpu_allocated": False,
        "groups": {},
        "torch_version": torch.__version__,
    }
    persist(run_id, report)
    try:
        ops = import_source("prime_rl.trainer.models.conversion_ops", "conversion_ops.py")
        converter = import_source("laguna_conversion_check", "laguna/converting_laguna.py")
        snapshot = Path(
            snapshot_download(
                MODEL_ID,
                revision=REVISION,
                local_files_only=True,
                allow_patterns=["*.json", "*.py", "*.safetensors"],
            )
        )
        config = json.loads((snapshot / "config.json").read_text())
        chain = converter.conversion_chain(SimpleNamespace(**config))
        weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        metadata = {}
        for filename in set(weight_map.values()):
            with (snapshot / filename).open("rb") as stream:
                header_size = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(header_size))
            metadata.update(
                {name: value for name, value in header.items() if name != "__metadata__"}
            )
        assert set(metadata) == set(weight_map)
        assert all(value["dtype"] == "BF16" for value in metadata.values())
        state = {
            name: torch.empty(value["shape"], device="meta", dtype=torch.bfloat16)
            for name, value in metadata.items()
        }
        ops.apply_hf_to_prime(state, chain)
        expected_shapes = {}
        for name, value in metadata.items():
            destination, expert_index = destination_for(name)
            shape = (
                value["shape"] if expert_index is None else [config["num_experts"], *value["shape"]]
            )
            if destination in expected_shapes:
                assert expected_shapes[destination] == shape
            expected_shapes[destination] = shape
        assert set(state) == set(expected_shapes)
        assert all(list(state[name].shape) == shape for name, shape in expected_shapes.items())
        report.update(
            {
                "input_tensors": len(metadata),
                "converted_tensors": len(state),
                "all_key_shapes_checked": True,
            }
        )
        ops.apply_prime_to_hf(state, chain)
        assert set(state) == set(metadata)
        assert all(list(state[name].shape) == value["shape"] for name, value in metadata.items())
        del state
        persist(run_id, report)
        groups = {
            f"layer-{layer}": [
                name for name in weight_map if name.startswith(f"model.layers.{layer}.")
            ]
            for layer in range(config["num_hidden_layers"])
        }
        groups["non-layer"] = [name for name in weight_map if not name.startswith("model.layers.")]
        for group, names in groups.items():
            report["groups"][group] = check_group(snapshot, weight_map, names, ops, chain)
            gc.collect()
            print(
                f"Verified {group}: {report['groups'][group]['tensors']} tensors, exact values and round trip",
                flush=True,
            )
            persist(run_id, report)
        assert sum(group["tensors"] for group in report["groups"].values()) == len(weight_map)
        report.update(
            {
                "status": "passed",
                "all_values_bitwise_preserved": True,
                "all_roundtrips_bitwise_equal": True,
            }
        )
    except Exception:
        report.update({"status": "failed", "error": traceback.format_exc()})
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        persist(run_id, report)
    return report


@app.local_entrypoint()
def main() -> None:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12]
    print(f"Report: kernel-rlm-traces:prime-conversion/{run_id}/report.json")
    result = validate.remote(run_id)
    directory = Path("logs/prime-conversion") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "groups"}, indent=2))
