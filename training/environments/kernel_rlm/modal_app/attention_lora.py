"""Bounded attention-only PEFT feasibility test, not a prime-rl training run."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import modal

MODEL_ID = "poolside/Laguna-XS.2"
REVISION = "a397bde04501591d0ac3d74a5d9d0a4ec298ffb8"
PRIME_REVISION = "ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
RANK = 8
ALPHA = 16
TOKENS = 32

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==5.6.2", "peft==0.19.0", "accelerate==1.15.0")
    .env({"HF_HOME": "/models/hf", "TOKENIZERS_PARALLELISM": "false"})
)
app = modal.App("kernel-rlm-attention-lora-smoke")
models = modal.Volume.from_name("kernel-rlm-model-cache", create_if_missing=True)
reports = modal.Volume.from_name("kernel-rlm-traces")


def adapt(model: Any) -> Any:
    from peft import LoraConfig, get_peft_model

    return get_peft_model(
        model,
        LoraConfig(
            r=RANK,
            lora_alpha=ALPHA,
            target_modules=TARGETS,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )


def inventory(model: Any) -> dict[str, Any]:
    modules = {
        name: {"in": module.in_features, "out": module.out_features}
        for name, module in model.named_modules()
        if hasattr(module, "lora_A")
    }
    trainable = {
        name: param.numel() for name, param in model.named_parameters() if param.requires_grad
    }
    assert len(modules) == 160, f"Expected 4 projections in each of 40 layers, got {len(modules)}"
    assert all(".self_attn." in name and name.rsplit(".", 1)[-1] in TARGETS for name in modules)
    assert all(".lora_A." in name or ".lora_B." in name for name in trainable)
    assert sum(trainable.values()) == 8192000, trainable
    return {"modules": modules, "trainable_parameters": sum(trainable.values()), "targets": TARGETS}


def save_report(run_id: str, phase: str, report: dict[str, Any]) -> None:
    path = Path("/reports/attention-lora") / run_id / f"{phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    reports.commit()


@app.function(
    image=image,
    volumes={"/models": models, "/reports": reports},
    cpu=2,
    memory=8192,
    timeout=180,
    retries=0,
    max_containers=1,
)
def preflight(run_id: str) -> dict[str, Any]:
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, code_revision=REVISION, dtype=torch.bfloat16
        )
        expected_keys = set(model.state_dict())
        result = inventory(adapt(model))
    index_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json", revision=REVISION)
    checkpoint_keys = set(json.loads(Path(index_path).read_text())["weight_map"])
    result["missing_checkpoint_keys"] = sorted(expected_keys - checkpoint_keys)
    result["unexpected_checkpoint_keys"] = sorted(checkpoint_keys - expected_keys)
    compatible = expected_keys == checkpoint_keys
    result.update(
        {
            "status": "passed" if compatible else "failed",
            "mode": "meta-no-weights-no-gpu",
            "model_revision": REVISION,
            "checkpoint_keys_match_without_conversion": compatible,
        }
    )
    models.commit()
    save_report(run_id, "preflight", result)
    assert (
        compatible
    ), "Checkpoint keys need conversion; refusing weight download and GPU allocation"
    return result


@app.function(
    image=image,
    volumes={"/models": models},
    cpu=4,
    memory=16384,
    timeout=1200,
    retries=0,
    max_containers=1,
)
def prepare_weights() -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        MODEL_ID,
        revision=REVISION,
        allow_patterns=["*.json", "*.py", "*.safetensors"],
        max_workers=4,
    )
    models.commit()
    return path


@app.function(
    image=image,
    volumes={"/models": models, "/reports": reports},
    gpu="A100-80GB",
    cpu=4,
    memory=98304,
    timeout=600,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def smoke(run_id: str, model_path: str) -> dict[str, Any]:
    import time
    import traceback

    import torch
    from transformers import AutoModelForCausalLM

    started = time.monotonic()
    report: dict[str, Any] = {
        "status": "running",
        "stage": "loading",
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "prime_reference_revision": PRIME_REVISION,
        "implementation": "transformers+peft; not prime-rl trainer",
        "rank": RANK,
        "alpha": ALPHA,
        "sequence_length": TOKENS,
        "batch_size": 1,
        "dtype": "bfloat16",
        "attention": "sdpa",
        "seed": 17,
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
    }
    save_report(run_id, "gpu", report)
    try:
        models.reload()
        torch.manual_seed(17)
        torch.cuda.reset_peak_memory_stats()
        model, loading_info = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        report["loading_info"] = loading_info
        assert not any(
            loading_info.get(key)
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        ), "Checkpoint loading mismatch; refusing to test a partly reinitialized model"
        report["loaded_allocated_bytes"] = torch.cuda.memory_allocated()
        model = adapt(model)
        report.update(inventory(model))
        assert all(param.device.type == "cuda" for param in model.parameters())
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.train()
        adapters = {name: param for name, param in model.named_parameters() if param.requires_grad}
        before = {name: param.detach().clone() for name, param in adapters.items()}
        optimizer = torch.optim.AdamW(adapters.values(), lr=1e-4, weight_decay=0.0)
        inputs = torch.randint(32, model.config.vocab_size, (1, TOKENS), device="cuda")
        report["input_ids"] = inputs.cpu().tolist()
        report["stage"] = "forward-backward"
        save_report(run_id, "gpu", report)
        loss = model(input_ids=inputs, labels=inputs).loss
        assert torch.isfinite(loss), "Nonfinite loss"
        loss.backward()
        assert all(param.grad is None for param in model.parameters() if not param.requires_grad)
        assert all(
            param.grad is not None and torch.isfinite(param.grad).all()
            for param in adapters.values()
        )
        report["nonzero_gradient_tensors"] = sum(
            int(param.grad.count_nonzero() > 0) for param in adapters.values()
        )
        assert report["nonzero_gradient_tensors"] > 0, "No adapter gradient signal"
        optimizer.step()
        torch.cuda.synchronize()
        report["changed_adapter_tensors"] = sum(
            int(not torch.equal(before[name], param)) for name, param in adapters.items()
        )
        assert report["changed_adapter_tensors"] > 0, "No adapter update"
        report.update(
            {
                "status": "passed",
                "stage": "complete",
                "loss": loss.item(),
                "base_gradients_absent": True,
            }
        )
    except Exception:
        report.update({"status": "failed", "error": traceback.format_exc()})
        raise
    finally:
        report.update(
            {
                "elapsed_seconds": time.monotonic() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
        save_report(run_id, "gpu", report)
    return report


@app.local_entrypoint()
def main(gpu: bool = False) -> None:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12]
    directory = Path("logs/attention-lora") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Reports: kernel-rlm-traces:attention-lora/{run_id}")
    result = preflight.remote(run_id)
    (directory / "preflight.json").write_text(json.dumps(result, indent=2))
    print(f"Preflight passed: {result['trainable_parameters']:,} adapter parameters")
    if gpu:
        result = smoke.remote(run_id, prepare_weights.remote())
        (directory / "gpu.json").write_text(json.dumps(result, indent=2))
        print(
            json.dumps({key: value for key, value in result.items() if key != "modules"}, indent=2)
        )
