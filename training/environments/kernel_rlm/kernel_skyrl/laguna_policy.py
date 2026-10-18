"""Shared BF16 Laguna loading and Moonlight Muon wiring for the optional backend."""

import math
from pathlib import Path
from typing import Any

MODEL_REVISION = "a397bde04501591d0ac3d74a5d9d0a4ec298ffb8"
DION_REVISION = "d891eebbd950a6ff11d9b5f806282e33df83df9d"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
RANK = 8
ALPHA = 16


def serializable_loading_info(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: serializable_loading_info(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serializable_loading_info(item) for item in value]
    return value


def attach_adapters(model: Any) -> Any:
    import torch
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        model,
        LoraConfig(
            r=RANK,
            lora_alpha=ALPHA,
            target_modules=list(TARGETS),
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if ".lora_A." not in name and ".lora_B." not in name:
                raise ValueError(f"Unexpected trainable parameter: {name}")
            parameter.data = parameter.data.to(torch.float32)
    audit_storage(model)
    return model


def audit_storage(model: Any) -> dict:
    import torch

    totals = {
        "frozen_parameters": 0,
        "frozen_bytes": 0,
        "adapter_parameters": 0,
        "adapter_bytes": 0,
        "adapter_tensors": 0,
    }
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.dtype != torch.float32 or parameter.ndim != 2:
                raise ValueError(f"Adapters must be FP32 matrices: {name}")
            if ".lora_A." not in name and ".lora_B." not in name:
                raise ValueError(f"Non-adapter parameter is trainable: {name}")
            totals["adapter_parameters"] += parameter.numel()
            totals["adapter_bytes"] += parameter.numel() * parameter.element_size()
            totals["adapter_tensors"] += 1
        else:
            if parameter.dtype != torch.bfloat16:
                raise ValueError(f"Frozen parameter is not BF16: {name}: {parameter.dtype}")
            totals["frozen_parameters"] += parameter.numel()
            totals["frozen_bytes"] += parameter.numel() * parameter.element_size()
    return totals


def load_base(model_path: Path) -> tuple[Any, dict]:
    import torch
    from transformers import AutoModelForCausalLM

    model, information = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=False,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    problems = {
        key: information.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if information.get(key)
    }
    if problems:
        raise ValueError(f"Incomplete Laguna checkpoint load: {problems}")
    if model.config.model_type != "laguna":
        raise ValueError("The loader only supports Laguna")
    return model, serializable_loading_info(information)


def muon_groups(model: Any, lr: float = 1e-5, weight_decay: float = 0.01) -> list[dict]:
    matrices = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.ndim != 2 or (".lora_A." not in name and ".lora_B." not in name):
                raise ValueError(f"Only attention LoRA matrices may be optimized: {name}")
            if not any(f".self_attn.{target}." in name for target in TARGETS):
                raise ValueError(f"Adapter outside attention target list: {name}")
            matrices.append(parameter)
    if not matrices or len({id(parameter) for parameter in matrices}) != len(matrices):
        raise ValueError("Expected distinct trainable adapter matrices")
    return [
        {
            "params": matrices,
            "algorithm": "muon",
            "adjust_lr": "rms_norm",
            "lr": lr,
            "weight_decay": weight_decay,
        }
    ]


def build_muon(model: Any, mesh: Any, lr: float = 1e-5, weight_decay: float = 0.01):
    from dion import Muon
    from dion.muon import adjust_lr_rms_norm

    if mesh.ndim != 1 or mesh.mesh_dim_names != ("fsdp",):
        raise ValueError("This baseline requires a one-dimensional FSDP mesh")
    audit_storage(model)
    groups = muon_groups(model, lr, weight_decay)
    for parameter in groups[0]["params"]:
        expected = lr * 0.2 * math.sqrt(max(parameter.shape))
        if not math.isclose(adjust_lr_rms_norm(lr, parameter.shape), expected, rel_tol=1e-12):
            raise ValueError("Dion Moonlight RMS scaling does not match the pinned convention")
    return Muon(
        groups,
        fsdp_mesh_dim=0,
        world_mesh=mesh,
        distributed_mesh=mesh,
        lr=lr,
        weight_decay=weight_decay,
        mu=0.95,
        betas=(0.9, 0.95),
        adjust_lr="rms_norm",
        use_triton=False,
    )
