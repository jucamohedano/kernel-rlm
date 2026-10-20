"""Bounded optional PRIME-RL CUDA component checks, not curriculum training."""

import argparse
import copy
import importlib.metadata
import json
import time
import traceback
from datetime import timedelta
from pathlib import Path

from kernel_trl.spending import phase_deadline, spending_snapshot


def authorize_probe(configuration: dict, now: float) -> float:
    if configuration["budget_usd"] != 20:
        raise ValueError("This diagnostic requires the separately authorized RTX $20 ceiling")
    return phase_deadline(configuration, now, 600, reserve_usd=2)


def run(arguments, report: dict) -> None:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional
    from flash_attn import flash_attn_varlen_func
    from prime_rl.configs.trainer import LoRAConfig, MuonConfig
    from prime_rl.trainer.lora import apply_lora_to_model
    from prime_rl.trainer.models.laguna.configuration_laguna import LagunaConfig
    from prime_rl.trainer.models.laguna.modeling_laguna import LagunaForCausalLM
    from prime_rl.trainer.models.layers.lm_head import VanillaOutputLinear
    from prime_rl.trainer.optim import setup_optimizer
    from prime_rl.trainer.parallel_dims import ParallelDims
    from torch.distributed.device_mesh import init_device_mesh

    def save() -> None:
        arguments.output.write_text(json.dumps(report, indent=2))

    report["versions"] = {
        name: importlib.metadata.version(name)
        for name in ("prime-rl", "torch", "transformers", "flash-attn", "dion")
    }
    torch.cuda.set_device(0)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("Expose exactly one RTX PRO 6000 sm120 card to this component probe")
    report["gpu"] = torch.cuda.get_device_name()
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    query, key, value = [
        torch.randn(256, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _index in range(3)
    ]
    sequence_ends = torch.tensor([0, 256], device="cuda", dtype=torch.int32)
    actual = flash_attn_varlen_func(
        query, key, value, sequence_ends, sequence_ends, 256, 256, causal=True
    )
    expected = (
        functional.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            is_causal=True,
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    actual.float().square().mean().backward()
    if not all(
        tensor.grad is not None and torch.isfinite(tensor.grad).all()
        for tensor in (query, key, value)
    ):
        raise ValueError("FlashAttention gradients are missing or nonfinite")
    report["flash_attention_2"] = {
        "status": "passed",
        "forward_matches_sdpa": True,
        "backward_finite": True,
    }
    save()

    configuration = LagunaConfig.from_pretrained(arguments.model, local_files_only=True)
    configuration._attn_implementation = "flash_attention_2"
    adapters = LoRAConfig(rank=8, alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    with torch.device("meta"):
        full_model = LagunaForCausalLM(configuration)
        apply_lora_to_model(full_model, adapters)
    trainable = [parameter for parameter in full_model.parameters() if parameter.requires_grad]
    if len(trainable) != 320 or sum(parameter.numel() for parameter in trainable) != 8192000:
        raise ValueError(
            "Native PRIME-RL attention adapter identities differ from the current policy"
        )
    report["full_laguna_meta"] = {
        "status": "passed",
        "adapter_tensors": len(trainable),
        "adapter_parameters": sum(parameter.numel() for parameter in trainable),
        "checkpoint_weights_loaded": False,
    }
    del full_model, trainable
    save()

    small_config = LagunaConfig(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_attention_heads_per_layer=[4, 4],
        layer_types=["full_attention", "sliding_attention"],
        mlp_layer_types=["dense", "sparse"],
        sliding_window=64,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        gating=configuration.gating,
        load_balance_coeff=None,
        use_cache=False,
    )
    small_config._attn_implementation = "flash_attention_2"
    model = LagunaForCausalLM(small_config).to(device="cuda", dtype=torch.bfloat16)
    head = VanillaOutputLinear(small_config.hidden_size, small_config.vocab_size).to(
        device="cuda", dtype=torch.bfloat16
    )
    head.weight = model.lm_head.weight
    model.lm_head = head
    apply_lora_to_model(model, adapters)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    originals = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    dist.init_process_group(
        "nccl",
        init_method=f"file://{arguments.output.parent / 'rendezvous'}",
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=120),
        device_id=torch.device("cuda", 0),
    )
    try:
        mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("dp_shard",))
        parallel = ParallelDims(
            dp_replicate=1, dp_shard=1, cp=1, pp=1, ep=1, world_size=1, _world_mesh=mesh
        )
        optimizer, manager = setup_optimizer(
            MuonConfig(lr=1e-5, weight_decay=0.01, mu=0.95),
            list(model.named_parameters()),
            parallel,
        )
        if manager is not None:
            raise ValueError("Unexpected optimizer offload")
        input_ids = torch.randint(0, 512, (1, 256), device="cuda")
        targets = input_ids.roll(-1, dims=1)

        def update() -> float:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = (
                    model(
                        input_ids=input_ids,
                        labels=targets,
                        seq_lens=torch.tensor([256], device="cuda"),
                    )["loss"]
                    / 256
                )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite synthetic native loss")
            loss.backward()
            gradients = [
                parameter.grad for parameter in model.parameters() if parameter.requires_grad
            ]
            if not gradients or any(
                gradient is None or not torch.isfinite(gradient).all() for gradient in gradients
            ):
                raise ValueError("Missing or nonfinite native adapter gradient")
            optimizer.step()
            torch.cuda.synchronize()
            return float(loss.detach())

        first_loss = update()
        saved_model = copy.deepcopy(model.state_dict())
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        torch.save(
            {"model": saved_model, "optimizer": saved_optimizer},
            arguments.output.parent / "synthetic-component-state.pt",
        )
        second_loss = update()
        continued = {
            name: parameter.detach().clone() for name, parameter in model.named_parameters()
        }
        saved = torch.load(
            arguments.output.parent / "synthetic-component-state.pt", weights_only=True
        )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        resumed_loss = update()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, continued[name], rtol=0, atol=0)
            if name in frozen and not torch.equal(parameter, frozen[name]):
                raise ValueError(f"Frozen synthetic base changed: {name}")
        changed = sum(
            not torch.equal(parameter, originals[name])
            for name, parameter in model.named_parameters()
            if name in originals
        )
        if not changed:
            raise ValueError("No synthetic adapter changed")
        report["synthetic_native_laguna"] = {
            "status": "passed",
            "layers": 2,
            "tokens": 256,
            "dense_and_sparse_mlp": True,
            "full_and_sliding_attention": True,
            "adapter_dtype": "float32",
            "frozen_dtype": "bfloat16",
            "first_loss": first_loss,
            "second_loss": second_loss,
            "resumed_loss": resumed_loss,
            "changed_adapter_tensors": changed,
            "optimizer_resume_reproduced_next_update_bitwise": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "goal_optimizer_updates": 0,
            "not_grpo_or_full_model_training": True,
        }
    finally:
        dist.destroy_process_group()
    report["status"] = "passed_components_only"
    save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = json.loads(arguments.budget.read_text())
    authorize_probe(configuration, time.time())
    arguments.output.parent.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "status": "running",
        "scope": "Official native PRIME-RL components only; no RLM GRPO, full-weight loading, or checkpoint19 migration",
        "goal_optimizer_updates": 0,
    }
    try:
        run(arguments, report)
    except BaseException:
        report.update(status="failed_component", error=traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["budget"] = spending_snapshot(configuration, time.time())
        arguments.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
