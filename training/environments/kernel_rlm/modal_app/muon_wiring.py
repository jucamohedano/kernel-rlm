"""CPU-only verification of prime's Muon parameter groups and Moonlight scaling."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import modal

from .attention_lora import MODEL_ID, PRIME_REVISION, REVISION, models, reports
from .native_lora_smoke import ALPHA, RANK, TARGETS, native_image

DION_REVISION = "d891eeb"
muon_image = native_image.pip_install(
    f"https://github.com/samsja/dion/archive/{DION_REVISION}.tar.gz"
)
app = modal.App("kernel-rlm-muon-wiring")


@app.function(
    image=muon_image,
    volumes={"/models": models, "/reports": reports},
    cpu=2,
    memory=16384,
    timeout=300,
    retries=0,
    max_containers=1,
)
def verify(run_id: str) -> dict[str, Any]:
    import math
    import sys
    import tempfile
    import time
    import traceback
    import types
    from unittest.mock import patch

    import torch
    import torch.distributed as dist
    from dion.muon import adjust_lr_rms_norm
    from torch.distributed.device_mesh import init_device_mesh

    started = time.monotonic()
    report: dict[str, Any] = {
        "status": "running",
        "scope": "CPU construction and meta-parameter grouping; no optimizer step or training",
        "prime_revision": PRIME_REVISION,
        "dion_revision": DION_REVISION,
        "model_revision": REVISION,
        "gpu_allocated": False,
    }
    destination = Path("/reports/muon-wiring") / run_id / "report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        models_package = types.ModuleType("prime_rl.trainer.models")
        models_package.__path__ = ["/root/prime_rl/trainer/models"]
        sys.modules["prime_rl.trainer.models"] = models_package
        from prime_rl.trainer.models.base import PreTrainedModelPrimeRL

        models_package.PreTrainedModelPrimeRL = PreTrainedModelPrimeRL
        from prime_rl.configs.trainer import LoRAConfig, MuonConfig
        from prime_rl.trainer.lora import apply_lora_to_model, setup_lora_state
        from prime_rl.trainer.models.laguna.configuration_laguna import LagunaConfig
        from prime_rl.trainer.models.laguna.modeling_laguna import LagunaForCausalLM
        from prime_rl.trainer.optim import setup_optimizer
        from prime_rl.trainer.parallel_dims import ParallelDims

        model_config = LagunaConfig.from_pretrained(
            MODEL_ID, revision=REVISION, local_files_only=True
        )
        model_config.gating = "per-head"
        with torch.device("meta"):
            model = LagunaForCausalLM(model_config)
        with patch(
            "prime_rl.trainer.lora.setup_lora_state",
            side_effect=lambda config, device: setup_lora_state(config, torch.device("meta")),
        ):
            apply_lora_to_model(model, LoRAConfig(rank=RANK, alpha=ALPHA, target_modules=TARGETS))

        with tempfile.TemporaryDirectory() as rendezvous:
            dist.init_process_group(
                "gloo", init_method=f"file://{rendezvous}/store", rank=0, world_size=1
            )
            mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp_shard",))
            parallel = ParallelDims(
                dp_replicate=1,
                dp_shard=1,
                cp=1,
                pp=1,
                ep=1,
                world_size=1,
                _world_mesh=mesh,
            )
            config = MuonConfig(lr=1e-4, weight_decay=0.01, mu=0.95)
            optimizer, manager = setup_optimizer(config, list(model.named_parameters()), parallel)
            assert manager is None
            trainable = {
                id(param): name for name, param in model.named_parameters() if param.requires_grad
            }
            grouped = [param for group in optimizer.param_groups for param in group["params"]]
            assert len(grouped) == len(trainable) == 320
            assert {id(param) for param in grouped} == set(trainable)
            assert sum(param.numel() for param in grouped) == 8192000
            assert all("lora_A" in name or "lora_B" in name for name in trainable.values())
            groups = []
            for group in optimizer.param_groups:
                if group["params"]:
                    assert group["algorithm"] == "muon"
                    assert group["adjust_lr"] == "rms_norm"
                groups.append(
                    {
                        "algorithm": group["algorithm"],
                        "adjust_lr": group["adjust_lr"],
                        "tensor_count": len(group["params"]),
                        "parameter_count": sum(param.numel() for param in group["params"]),
                    }
                )
            shapes = sorted({tuple(param.shape) for param in grouped})
            scaling = []
            for shape in shapes:
                expected = config.lr * 0.2 * math.sqrt(max(shape))
                actual = adjust_lr_rms_norm(config.lr, shape)
                assert math.isclose(actual, expected, rel_tol=1e-12)
                scaling.append({"shape": shape, "base_lr": config.lr, "update_lr": actual})

            fixtures = [
                ("attention.lora_A.0", torch.nn.Parameter(torch.ones(8, 16))),
                ("norm.weight", torch.nn.Parameter(torch.ones(16))),
                ("embed_tokens.weight", torch.nn.Parameter(torch.ones(16, 16))),
                ("lm_head.weight", torch.nn.Parameter(torch.ones(16, 16))),
                ("frozen.weight", torch.nn.Parameter(torch.ones(16, 16), requires_grad=False)),
            ]
            fixture_optimizer, _manager = setup_optimizer(config, fixtures, parallel)
            algorithms = {
                id(param): group["algorithm"]
                for group in fixture_optimizer.param_groups
                for param in group["params"]
            }
            assert [algorithms.get(id(param)) for name, param in fixtures] == [
                "muon",
                "adamw",
                "adamw",
                "adamw",
                None,
            ]
            report.update(
                {
                    "status": "passed",
                    "optimizer_config": config.model_dump(mode="json"),
                    "config_scope": "wiring-test values, not tuned GRPO hyperparameters",
                    "groups": groups,
                    "scaling": scaling,
                    "fallback_and_frozen_checks": "passed",
                    "optimizer_step_executed": False,
                }
            )
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        report["elapsed_seconds"] = time.monotonic() - started
        destination.write_text(json.dumps(report, indent=2))
        reports.commit()
    return report


@app.local_entrypoint()
def main() -> None:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12]
    print(f"Report: kernel-rlm-traces:muon-wiring/{run_id}/report.json", flush=True)
    result = verify.remote(run_id)
    destination = Path("logs/muon-wiring") / run_id / "report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
