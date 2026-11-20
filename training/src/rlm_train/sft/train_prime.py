"""CPU contract probe for pinned Prime RL SFT and its isolated file dashboard.

This is not the Prime RL CUDA trainer. No model files, remote monitors, daemon,
or service are opened by this entry point.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from rlm_train.sft.bundle import collate_group, load_bundle, masked_nll_sum

PRIME_REVISION = "4e94a8616558d99f9a39d6a1facbadbfe6233e86"
METRICS_FILE = Path("monitors/file/metrics.jsonl")


def prime_group_batches(group, bundle):
    """One outer trajectory, one unpadded Prime-shaped forward per decision.

    Inputs and targets are pre-shifted as in pinned SFTDataset. Never feed the
    outer group into stock CatDataset or treat its rows as optimizer samples.
    """
    batch = collate_group(
        group,
        pad_id=bundle.pad_id,
        member_cap=bundle.member_cap,
        padded_slots_cap=bundle.padded_slots_cap,
    )
    for i, length in enumerate(batch["attention_mask"].sum(dim=1).tolist()):
        if length == 1:  # validated position zero cannot contain a target
            continue
        width = length - 1
        yield {
            "input_ids": batch["input_ids"][i : i + 1, :width],
            "target_ids": batch["input_ids"][i : i + 1, 1:length],
            "loss_mask": batch["labels"][i : i + 1, 1:length].ne(-100),
            "position_ids": torch.arange(width, dtype=torch.long).unsqueeze(0),
            "seq_lens": torch.tensor([width], dtype=torch.long),
            "mm_kwargs": None,
            "mm_token_type_ids": None,
        }


def _probe_logits(table, ids):
    """Tiny causal CPU probe, sensitive to row prefixes and boundaries."""
    positions = torch.arange(1, ids.shape[-1] + 1, device="cpu")
    return table[ids].cumsum(dim=1) / positions[None, :, None]


def _contract_step(bundle):
    table = torch.nn.Parameter(
        torch.randn(
            (16, 16),
            device="cpu",
            dtype=torch.float32,
            generator=torch.Generator(device="cpu").manual_seed(21),
        )
    )
    # The fixture schema can be extended, but this fixed probe cannot handle
    # IDs outside its vocabulary. No truncation/modulo/remapping is allowed.
    if (
        any(token >= 16 for g in bundle.groups for m in g.members for token in m.input_ids)
        or bundle.pad_id >= 16
    ):
        raise ValueError("token outside CPU probe vocabulary")
    loss_parts = []
    count = 0
    reference_parts = []
    reference_count = 0
    for group in bundle.groups:
        for b in prime_group_batches(group, bundle):
            mask = b["loss_mask"]
            n = int(mask.sum())
            if n:
                scores = _probe_logits(table, b["input_ids"])
                loss_parts.append(
                    F.cross_entropy(scores[mask], b["target_ids"][mask], reduction="sum")
                )
                count += n
        reference = collate_group(
            group,
            pad_id=bundle.pad_id,
            member_cap=bundle.member_cap,
            padded_slots_cap=bundle.padded_slots_cap,
        )
        subtotal, selected = masked_nll_sum(
            _probe_logits(table, reference["input_ids"]), reference["labels"]
        )
        reference_parts.append(subtotal)
        reference_count += int(selected)
    if not count or count != reference_count:
        raise RuntimeError("Prime contract target count disagrees with shared objective")
    total = torch.stack(loss_parts).sum()
    loss = total / count
    reference_loss = torch.stack(reference_parts).sum() / reference_count
    torch.testing.assert_close(loss, reference_loss)
    (reference_grad,) = torch.autograd.grad(reference_loss, table)
    optimizer = torch.optim.SGD([table], lr=0.01)
    loss.backward()
    torch.testing.assert_close(table.grad, reference_grad)
    optimizer.step()
    mean = float(loss.detach())
    return {
        "step": 1,
        "time": time.time(),
        "loss/mean": mean,
        "loss/perplexity": math.exp(min(mean, 20)),
        "sft_cpu_probe/target_tokens": count,
        "sft_cpu_probe/trajectory_groups": len(bundle.groups),
    }


def run_synthetic(bundle_dir, manifest_sha256, output_dir, *, allow_synthetic=False):
    """Run one CPU-only update, then hand aggregate metrics to the dashboard."""
    for key, expected in {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}.items():
        if os.environ.get(key, expected) != expected:
            raise ValueError("only single-process CPU probe is supported")
    if torch.distributed.is_initialized():
        raise ValueError("distributed execution is not supported")
    bundle = load_bundle(bundle_dir, manifest_sha256, allow_synthetic=allow_synthetic)
    run_dir = Path(output_dir)
    if not run_dir.name or run_dir.name.startswith("."):
        raise ValueError("dashboard run directory name must not be hidden")
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError("dashboard run directory must be new")
    metrics = _contract_step(bundle)
    row = json.dumps(metrics, allow_nan=False) + "\n"
    run_dir.mkdir(mode=0o700)
    path = run_dir / METRICS_FILE
    path.parent.mkdir(parents=True)
    with path.open("x", encoding="utf-8") as writer:
        writer.write(row)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="new child of a dedicated dashboard parent"
    )
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    try:
        path = run_synthetic(
            args.bundle_dir,
            args.manifest_sha256,
            args.output_dir,
            allow_synthetic=args.allow_synthetic,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Synthetic CPU probe only (not native Prime SFT). Metrics file: {path}")


if __name__ == "__main__":
    main()
