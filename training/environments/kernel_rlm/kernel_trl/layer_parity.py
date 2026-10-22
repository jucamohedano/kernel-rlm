"""Read-only layer-boundary captures for a bounded base-model parity diagnostic."""

import argparse
import copy
import functools
import json
import time
from pathlib import Path


def capture_positions(prompt_length: int, reply_length: int) -> list[int]:
    if prompt_length < 2 or reply_length < 1:
        raise ValueError("A nonempty reply and a prompt of at least two tokens are required")
    return sorted({0, 1, min(512, prompt_length - 1), prompt_length - 1, prompt_length})


def install_capture(model, directory: Path, selected: list[int], backend: str) -> None:
    model.parity_records = {}
    model.parity_positions = None

    def set_positions(module, args, kwargs):
        positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
        if positions is None:
            raise ValueError("Missing vLLM token positions")
        model.parity_positions = positions.detach().cpu().reshape(-1).tolist()

    if backend == "vllm":
        model.register_forward_pre_hook(set_positions, with_kwargs=True)

    def capture(name: str, combine_residual: bool = False):
        def observe(module, args, output):
            value = output
            if isinstance(value, tuple):
                value = value[0] + value[1] if combine_residual else value[0]
            flat = value.detach().reshape(-1, value.shape[-1])
            positions = model.parity_positions if backend == "vllm" else range(flat.shape[0])
            for position in selected:
                if position not in positions:
                    continue
                index = positions.index(position) if isinstance(positions, list) else position
                key = f"{name}:position-{position}"
                model.parity_records[key] = flat[index].float().cpu().clone()

        return observe

    body = model.model
    body.embed_tokens.register_forward_hook(capture("embedding"))
    for index, layer in enumerate(body.layers):
        layer.register_forward_hook(capture(f"layer-{index:02d}.output", backend == "vllm"))
        if index < 2:
            for name in ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp"):
                getattr(layer, name).register_forward_hook(capture(f"layer-{index:02d}.{name}"))
    body.norm.register_forward_hook(capture("final_norm"))
    model.parity_directory = str(directory)
    if backend == "vllm":
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank

        def write_capture(module, args, output):
            rank = get_tensor_model_parallel_rank()
            torch.save(model.parity_records, directory / f"vllm-layer-rank-{rank}.pt")

        model.register_forward_hook(write_capture)


def install_vllm_capture(model, directory: str, selected: list[int]) -> dict:
    from vllm.distributed import get_tensor_model_parallel_rank

    install_capture(model, Path(directory), selected, "vllm")
    return {"rank": get_tensor_model_parallel_rank(), "positions": selected}


def inspect_attention(model, checkpoint: str) -> dict:
    import torch
    from transformers.models.laguna.configuration_laguna import LagunaConfig
    from transformers.models.laguna.modeling_laguna import (
        LagunaRotaryEmbedding,
        apply_rotary_pos_emb,
    )
    from vllm.distributed import get_tensor_model_parallel_rank

    raw = json.loads((Path(checkpoint) / "config.json").read_text())
    reference = LagunaRotaryEmbedding(LagunaConfig(**copy.deepcopy(raw)))
    positions = torch.tensor([[0, 1, 512, 2880]])
    records = []
    for index in (0, 1, 2, 4):
        attention = model.model.layers[index].self_attn
        rotary = attention.rotary_emb
        layer_type = raw["layer_types"][index]
        cosine, sine = reference(
            torch.zeros(1, 4, raw["hidden_size"], dtype=torch.bfloat16), positions, layer_type
        )
        width = cosine.shape[-1] // 2
        expected = torch.cat((cosine[0, :, :width], sine[0, :, :width]), dim=-1).float()
        actual = rotary.cos_sin_cache[positions[0].to(rotary.cos_sin_cache.device)].float().cpu()
        query = (
            torch.linspace(-1, 1, 4 * attention.num_heads * attention.head_dim, device="cuda")
            .reshape(4, -1)
            .bfloat16()
        )
        key = (
            torch.linspace(1, -1, 4 * attention.num_kv_heads * attention.head_dim, device="cuda")
            .reshape(4, -1)
            .bfloat16()
        )
        expected_query, expected_key = apply_rotary_pos_emb(
            query.reshape(1, 4, attention.num_heads, attention.head_dim).transpose(1, 2),
            key.reshape(1, 4, attention.num_kv_heads, attention.head_dim).transpose(1, 2),
            cosine.to(query.device),
            sine.to(query.device),
        )
        actual_query, actual_key = rotary(positions[0].to(query.device), query.clone(), key.clone())
        expected_query = expected_query.transpose(1, 2).reshape_as(actual_query)
        expected_key = expected_key.transpose(1, 2).reshape_as(actual_key)
        records.append(
            {
                "layer": index,
                "type": layer_type,
                "heads_per_rank": attention.num_heads,
                "sliding_window": attention.sliding_window,
                "backend": type(attention.attn.impl).__name__,
                "backend_window": attention.attn.impl.sliding_window,
                "rotary_type": type(rotary).__name__,
                "rotary_base": rotary.base,
                "rotary_dim": rotary.rotary_dim,
                "reference_rotary_dim": cosine.shape[-1],
                "checkpoint_rotary_dim": int(
                    raw["head_dim"]
                    * raw["rope_parameters"][layer_type].get("partial_rotary_factor", 1.0)
                ),
                "cache_shape": list(actual.shape),
                "reference_cache_shape": list(expected.shape),
                "cache_max_difference": (actual - expected).abs().max().item()
                if actual.shape == expected.shape
                else None,
                "query_rotation_max_difference": (actual_query.float() - expected_query.float())
                .abs()
                .max()
                .item(),
                "key_rotation_max_difference": (actual_key.float() - expected_key.float())
                .abs()
                .max()
                .item(),
            }
        )
    return {"rank": get_tensor_model_parallel_rank(), "layers": records}


def run(arguments) -> dict:
    import torch

    directory = arguments.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    group = json.loads(arguments.group.read_text())
    row = group["rows"][0]
    prompt = row["prompt_ids"] + row["completion_ids"][:64]
    selected = capture_positions(len(row["prompt_ids"]), len(row["completion_ids"]))
    started = time.monotonic()
    if arguments.backend == "native":
        from kernel_training.laguna_policy import load_config
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            str(arguments.checkpoint),
            config=load_config(arguments.checkpoint),
            dtype=torch.bfloat16,
            device_map="balanced",
            max_memory={0: "40GiB", 1: "40GiB"},
            attn_implementation="sdpa",
        ).eval()
        install_capture(model, directory, selected, "native")
        tokens = torch.tensor([prompt], device=model.model.embed_tokens.weight.device)
        with torch.no_grad():
            model(tokens, attention_mask=torch.ones_like(tokens), use_cache=False, logits_to_keep=1)
        torch.save(model.parity_records, directory / "native-layers.pt")
        counts = [{"captures": len(model.parity_records)}]
    else:
        from vllm import LLM, SamplingParams

        from kernel_trl.layer_parity import (
            inspect_attention as worker_inspector,
        )
        from kernel_trl.layer_parity import install_vllm_capture as worker_installer

        model = LLM(
            model=str(arguments.checkpoint),
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            max_model_len=12288,
            max_num_seqs=4,
            max_num_batched_tokens=4096,
            enable_lora=True,
            max_lora_rank=8,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            generation_config="vllm",
        )
        try:
            if arguments.backend == "metadata":
                records = model.apply_model(
                    functools.partial(worker_inspector, checkpoint=str(arguments.checkpoint))
                )
                return {
                    "status": "measured",
                    "backend": "metadata",
                    "workers": records,
                    "optimizer_steps": 0,
                    "elapsed_seconds": time.monotonic() - started,
                }
            model.apply_model(
                functools.partial(worker_installer, directory=str(directory), selected=selected)
            )
            model.generate(
                [{"prompt_token_ids": prompt}],
                SamplingParams(max_tokens=1, temperature=1.0, detokenize=False),
                use_tqdm=False,
            )
            counts = [
                {
                    "rank": rank,
                    "captures": len(
                        torch.load(
                            directory / f"vllm-layer-rank-{rank}.pt",
                            map_location="cpu",
                            weights_only=True,
                        )
                    ),
                }
                for rank in range(2)
            ]
        finally:
            model.llm_engine.engine_core.shutdown()
    return {
        "status": "captured",
        "backend": arguments.backend,
        "optimizer_steps": 0,
        "positions": selected,
        "counts": counts,
        "elapsed_seconds": time.monotonic() - started,
    }


def compare(directory: Path) -> dict:
    import torch

    native = torch.load(directory / "native-layers.pt", map_location="cpu", weights_only=True)
    workers = [
        torch.load(directory / f"vllm-layer-rank-{rank}.pt", map_location="cpu", weights_only=True)
        for rank in range(2)
    ]
    if any(set(worker) != set(native) for worker in workers):
        raise ValueError("The backends did not capture identical layer/position boundaries")
    rows = []
    for name, reference in sorted(native.items()):
        value = workers[0][name]
        delta = value - reference
        rows.append(
            {
                "boundary": name,
                "max_absolute_difference": delta.abs().max().item(),
                "rms_difference": delta.square().mean().sqrt().item(),
                "reference_rms": reference.square().mean().sqrt().item(),
                "vllm_rms": value.square().mean().sqrt().item(),
                "cosine_similarity": torch.nn.functional.cosine_similarity(
                    reference, value, dim=0
                ).item(),
                "tp_rank_max_difference": (value - workers[1][name]).abs().max().item(),
            }
        )
    return {"status": "measured", "boundaries": len(rows), "rows": rows, "optimizer_steps": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("native", "vllm", "compare", "metadata"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--group", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.backend != "compare" and (arguments.checkpoint is None or arguments.group is None):
        raise ValueError("Capture requires an explicit checkpoint and saved group")
    report = compare(arguments.output_dir) if arguments.backend == "compare" else run(arguments)
    (arguments.output_dir / f"layer-{arguments.backend}.json").write_text(
        json.dumps(report, indent=2)
    )
    print(
        json.dumps(
            report if arguments.backend != "compare" else {"boundaries": report["boundaries"]}
        )
    )


if __name__ == "__main__":
    main()
