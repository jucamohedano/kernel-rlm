"""Single-A100 native prime-rl checkpoint and attention-LoRA smoke test."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import modal

from .attention_lora import MODEL_ID, PRIME_REVISION, REVISION, models, reports
from .muon_wiring import muon_image
from .native_lora_smoke import ALPHA, RANK, TARGETS, native_image

app = modal.App("kernel-rlm-prime-native-gpu-check")
TOKENS = 32


def save_report(run_id: str, report: dict[str, Any]) -> None:
    path = Path("/reports/native-gpu") / run_id / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    reports.commit()


@app.function(
    image=native_image,
    volumes={"/models": models},
    cpu=4,
    memory=24576,
    timeout=600,
    retries=0,
    max_containers=1,
)
def diagnose_shapes() -> dict[str, Any]:
    import json as json_module
    import struct
    import sys
    import types

    import torch
    from huggingface_hub import snapshot_download

    models.reload()
    models_package = types.ModuleType("prime_rl.trainer.models")
    models_package.__path__ = ["/root/prime_rl/trainer/models"]
    sys.modules["prime_rl.trainer.models"] = models_package
    from prime_rl.trainer.models.base import PreTrainedModelPrimeRL

    models_package.PreTrainedModelPrimeRL = PreTrainedModelPrimeRL
    from prime_rl.trainer.models.conversion_ops import apply_hf_to_prime
    from prime_rl.trainer.models.laguna.configuration_laguna import LagunaConfig
    from prime_rl.trainer.models.laguna.converting_laguna import conversion_chain
    from prime_rl.trainer.models.laguna.modeling_laguna import LagunaForCausalLM

    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=REVISION,
            local_files_only=True,
            allow_patterns=["*.json", "*.py", "*.safetensors"],
        )
    )
    config = LagunaConfig.from_pretrained(MODEL_ID, revision=REVISION)
    config.gating = "per-head"
    with torch.device("meta"):
        model = LagunaForCausalLM(config)
    model_shapes = {name: list(value.shape) for name, value in model.state_dict().items()}
    index = json_module.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    metadata: dict[str, dict[str, Any]] = {}
    for filename in set(index.values()):
        with (snapshot / filename).open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            header = json_module.loads(stream.read(header_size))
        metadata.update({name: value for name, value in header.items() if name != "__metadata__"})
    state = {
        name: torch.empty(value["shape"], device="meta", dtype=torch.bfloat16)
        for name, value in metadata.items()
    }
    apply_hf_to_prime(state, conversion_chain(config))
    mismatches = [
        {"name": name, "checkpoint": list(value.shape), "model": model_shapes.get(name)}
        for name, value in state.items()
        if name not in model_shapes or list(value.shape) != model_shapes[name]
    ]
    result = {
        "checkpoint_tensors": len(metadata),
        "converted_tensors": len(state),
        "model_tensors": len(model_shapes),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "gating": config.gating,
        "config_keys": sorted(config.to_dict().keys()),
    }
    print(json_module.dumps(result), flush=True)
    return result


@app.function(
    image=muon_image,
    volumes={"/models": models, "/reports": reports},
    gpu="A100-80GB",
    cpu=8,
    memory=98304,
    timeout=600,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def validate(run_id: str) -> dict[str, Any]:
    import gc
    import sys
    import traceback
    import types

    import torch
    import torch.distributed as dist
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    models.reload()
    started = time.monotonic()
    report: dict[str, Any] = {
        "status": "running",
        "scope": "native prime-rl Laguna converted-weight load and one attention-LoRA update",
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "prime_revision": PRIME_REVISION,
        "targets": TARGETS,
        "rank": RANK,
        "alpha": ALPHA,
        "tokens": TOKENS,
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
    }
    save_report(run_id, report)
    try:
        models_package = types.ModuleType("prime_rl.trainer.models")
        models_package.__path__ = ["/root/prime_rl/trainer/models"]
        sys.modules["prime_rl.trainer.models"] = models_package
        from prime_rl.trainer.models.base import PreTrainedModelPrimeRL

        models_package.PreTrainedModelPrimeRL = PreTrainedModelPrimeRL
        from prime_rl.configs.trainer import LoRAConfig, MuonConfig
        from prime_rl.trainer.lora import apply_lora_to_model
        from prime_rl.trainer.models.conversion_ops import apply_hf_to_prime
        from prime_rl.trainer.models.laguna.configuration_laguna import LagunaConfig
        from prime_rl.trainer.models.laguna.converting_laguna import conversion_chain
        from prime_rl.trainer.models.laguna.modeling_laguna import LagunaForCausalLM
        from prime_rl.trainer.models.layers.lora.base import set_lora_num_tokens
        from prime_rl.trainer.optim import setup_optimizer
        from prime_rl.trainer.parallel_dims import ParallelDims
        from torch.distributed.device_mesh import init_device_mesh

        snapshot = Path(
            snapshot_download(
                MODEL_ID,
                revision=REVISION,
                local_files_only=True,
                allow_patterns=["*.json", "*.py", "*.safetensors"],
            )
        )
        config = LagunaConfig.from_pretrained(MODEL_ID, revision=REVISION)
        config.gating = "per-head"
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("meta"):
                model = LagunaForCausalLM(config)
        finally:
            torch.set_default_dtype(previous_dtype)
        assert all(param.dtype == torch.bfloat16 for param in model.parameters())
        model = model.to_empty(device="cuda")
        model.init_buffers_post_meta()
        model_keys = set(model.state_dict())
        weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        state: dict[str, torch.Tensor] = {}
        for filename in sorted(set(weight_map.values())):
            with safe_open(snapshot / filename, framework="pt", device="cpu") as shard:
                for name in shard.keys():
                    state[name] = shard.get_tensor(name)
        report["checkpoint_tensors"] = len(state)
        assert len(state) == 30513
        apply_hf_to_prime(state, conversion_chain(config))
        report["converted_tensors"] = len(state)
        unexpected = sorted(set(state) - model_keys)
        missing = sorted(model_keys - set(state))
        allowed_missing = [name for name in missing if "tokens_per_expert" in name]
        assert not unexpected, f"Unexpected converted keys: {unexpected[:5]}"
        assert missing == allowed_missing, f"Missing model keys: {missing[:10]}"
        report.update({"unexpected_keys": unexpected, "missing_runtime_keys": allowed_missing})
        with torch.no_grad():
            destination = dict(model.named_parameters()) | dict(model.named_buffers())
            for name, tensor in state.items():
                destination[name].copy_(tensor)
        del state
        gc.collect()
        torch.cuda.synchronize()
        report["base_loaded_strictly"] = True
        report["base_gpu_allocated_bytes"] = torch.cuda.memory_allocated()
        save_report(run_id, report)

        apply_lora_to_model(
            model,
            LoRAConfig(rank=RANK, alpha=ALPHA, target_modules=TARGETS),
        )
        adapters = {name: param for name, param in model.named_parameters() if param.requires_grad}
        assert adapters and all("lora_A" in name or "lora_B" in name for name in adapters)
        assert sum(param.numel() for param in adapters.values()) == 8192000
        set_lora_num_tokens(torch.tensor([TOKENS], dtype=torch.int32, device="cuda"))

        def sdpa_fallback(q, k, v, _cu_seqlens_q, _cu_seqlens_k, _max_q, _max_k, **_kwargs):
            output = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
                is_causal=True,
                enable_gqa=True,
            )
            return output.squeeze(0).transpose(0, 1)

        for module in model.modules():
            if hasattr(module, "_flash_attn_call") and module._flash_attn_call is None:
                module._flash_attn_call = sdpa_fallback
        report["attention_backend"] = "torch_sdpa_gqa_fallback"
        model.train()
        torch.manual_seed(42)
        dist.init_process_group(
            "nccl", init_method=f"file:///tmp/muon-{run_id}", rank=0, world_size=1
        )
        parallel = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            pp=1,
            ep=1,
            world_size=1,
            _world_mesh=init_device_mesh("cuda", (1,), mesh_dim_names=("dp_shard",)),
        )
        optimizer_config = MuonConfig(lr=1e-4, weight_decay=0.01, mu=0.95)
        optimizer, manager = setup_optimizer(
            optimizer_config, list(model.named_parameters()), parallel
        )
        assert manager is None
        assert all(
            group["algorithm"] == "muon" and group["adjust_lr"] == "rms_norm"
            for group in optimizer.param_groups
            if group["params"]
        )
        report["optimizer"] = optimizer_config.model_dump(mode="json")
        report[
            "scope"
        ] = "one Muon update on random-token CE fixture; not GRPO or quality evaluation"
        inputs = torch.randint(32, config.vocab_size, (1, TOKENS), device="cuda")
        before = {name: param.detach().clone() for name, param in adapters.items()}
        model.config.use_cache = False
        model.enable_input_require_grads()
        report["input_ids"] = inputs.cpu().tolist()
        report["stage"] = "forward-backward"
        save_report(run_id, report)
        hidden = model.model(
            input_ids=inputs,
            seq_lens=torch.tensor([TOKENS], device="cuda"),
        ).last_hidden_state
        logits = model.lm_head(hidden)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), inputs.reshape(-1)
        )
        report["forward_path"] = "native Laguna body + native lm_head + torch cross_entropy"
        assert torch.isfinite(loss)
        loss.backward()
        assert all(
            param.grad is not None and torch.isfinite(param.grad).all()
            for param in adapters.values()
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(list(adapters.values()), 1.0)
        assert torch.isfinite(grad_norm)
        optimizer.step()
        torch.cuda.synchronize()
        assert all(torch.isfinite(param).all() for param in adapters.values())
        assert any(not torch.equal(before[name], param) for name, param in adapters.items())
        assert all(param.grad is None for param in model.parameters() if not param.requires_grad)
        report.update(
            {
                "status": "passed",
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                "adapter_parameters": sum(param.numel() for param in adapters.values()),
                "nonzero_gradient_tensors": sum(
                    int(param.grad.count_nonzero() > 0) for param in adapters.values()
                ),
                "changed_adapter_tensors": sum(
                    int(not torch.equal(before[name], param)) for name, param in adapters.items()
                ),
                "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
    except Exception:
        report.update({"status": "failed", "error": traceback.format_exc()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        report["elapsed_seconds"] = time.monotonic() - started
        if report["status"] == "passed":
            metrics_path = Path("/reports/native-gpu") / run_id / "metrics.jsonl"
            metrics_path.write_text(
                json.dumps(
                    {
                        "step": 0,
                        "time": time.time(),
                        "producer": "trainer",
                        "loss/mean": report["loss"],
                        "optim/grad_norm": report["grad_norm"],
                        "optim/lr": 1e-4,
                        "perf/peak_memory": report["peak_memory_reserved_bytes"] / 1024**3,
                        "time/check_seconds": report["elapsed_seconds"],
                        "adapters/changed_tensors": report["changed_adapter_tensors"],
                    }
                )
                + "\n"
            )
        save_report(run_id, report)
    return report


@app.local_entrypoint()
def main() -> None:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12]
    print(f"Report: kernel-rlm-traces:native-gpu/{run_id}/report.json")
    result = validate.remote(run_id)
    directory = Path("logs/native-gpu") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
