"""CPU/meta validation of prime-rl's native Laguna model and LoRA path."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import modal

from .attention_lora import MODEL_ID, PRIME_REVISION, REVISION, image, reports

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
RANK = 8
ALPHA = 16
PRIME_SOURCE = Path("/tmp/prime-rl-v090")

native_image = (
    image.pip_install(
        "prime-pydantic-config>=0.4.3",
        "loguru>=0.7.3",
        "https://github.com/pytorch/torchtitan/archive/23e4dfc.tar.gz",
        "ring-flash-attn>=0.1.8",
        "flash-linear-attention",
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu128torch2.11-cp312-cp312-linux_x86_64.whl",
    )
    .add_local_dir(PRIME_SOURCE / "src" / "prime_rl", "/root/prime_rl", copy=True)
    .add_local_dir(
        PRIME_SOURCE / "packages" / "prime-rl-configs" / "src" / "prime_rl",
        "/root/prime_rl",
        copy=True,
    )
    .env({"PYTHONPATH": "/root", "WORLD_SIZE": "1", "LOCAL_RANK": "0", "RANK": "0"})
)
app = modal.App("kernel-rlm-prime-native-lora-check")


def save_report(run_id: str, report: dict[str, Any]) -> None:
    path = Path("/reports/native-lora") / run_id / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    reports.commit()


@app.function(
    image=native_image,
    volumes={"/reports": reports},
    cpu=4,
    memory=32768,
    timeout=600,
    retries=0,
    max_containers=1,
)
def validate(run_id: str) -> dict[str, Any]:
    import sys
    import time
    import traceback
    import types
    from unittest.mock import patch

    import torch
    import torch.nn as nn

    models_package = types.ModuleType("prime_rl.trainer.models")
    models_package.__path__ = ["/root/prime_rl/trainer/models"]
    sys.modules["prime_rl.trainer.models"] = models_package
    from prime_rl.trainer.models.base import PreTrainedModelPrimeRL

    models_package.PreTrainedModelPrimeRL = PreTrainedModelPrimeRL
    from prime_rl.configs.trainer import LoRAConfig
    from prime_rl.trainer.lora import apply_lora_to_model, setup_lora_state
    from prime_rl.trainer.models.laguna.configuration_laguna import LagunaConfig
    from prime_rl.trainer.models.laguna.modeling_laguna import LagunaForCausalLM

    started = time.monotonic()
    report: dict[str, Any] = {
        "status": "running",
        "scope": "prime-rl native model and native LoRA wrapping; meta tensors only",
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "prime_revision": PRIME_REVISION,
        "targets": TARGETS,
        "rank": RANK,
        "alpha": ALPHA,
        "gpu_allocated": False,
        "torch_version": torch.__version__,
    }
    save_report(run_id, report)
    try:
        config = LagunaConfig.from_pretrained(MODEL_ID, revision=REVISION)
        with torch.device("meta"):
            model = LagunaForCausalLM(config)

        before = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
            and name.rsplit(".", 1)[-1] in TARGETS
            and ".self_attn." in name
        }
        assert len(before) == 160, f"Expected 160 native attention targets, got {len(before)}"
        assert all(module.weight.device.type == "meta" for module in before.values())

        lora_config = LoRAConfig(rank=RANK, alpha=ALPHA, target_modules=TARGETS)
        original_setup = setup_lora_state
        with patch(
            "prime_rl.trainer.lora.setup_lora_state",
            side_effect=lambda config, _device: original_setup(config, torch.device("meta")),
        ):
            apply_lora_to_model(model, lora_config)

        wrapped = {
            name: module
            for name, module in model.named_modules()
            if hasattr(module, "lora_A")
        }
        assert set(wrapped) == set(before)
        trainable = {
            name: param
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        assert trainable and all("lora_A" in name or "lora_B" in name for name in trainable)
        assert all(param.device.type == "meta" for param in trainable.values())
        base_trainable = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and "lora_A" not in name and "lora_B" not in name
        ]
        assert not base_trainable
        report.update(
            {
                "status": "passed",
                "native_model": type(model).__name__,
                "native_attention_targets": len(before),
                "wrapped_targets": len(wrapped),
                "trainable_adapter_parameters": sum(param.numel() for param in trainable.values()),
                "base_trainable_parameters": len(base_trainable),
                "device_override": "meta (prime apply_lora_to_model hardcodes CUDA for LoRA state)",
            }
        )
    except Exception:
        report.update({"status": "failed", "error": traceback.format_exc()})
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save_report(run_id, report)
    return report


@app.local_entrypoint()
def main() -> None:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12]
    print(f"Report: kernel-rlm-traces:native-lora/{run_id}/report.json")
    result = validate.remote(run_id)
    directory = Path("logs/native-lora") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
