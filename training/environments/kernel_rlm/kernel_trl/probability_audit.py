"""Compare saved sampler probabilities with the unchanged native learner policy."""

import hashlib
import json
import math
import statistics


class AuditComplete(Exception):
    pass


def group_digest(group: dict) -> str:
    return hashlib.sha256(
        json.dumps(group, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def require_probability_agreement(path, group: dict) -> dict:
    if path is None:
        raise ValueError("A passing probability audit is required before training")
    summary = json.loads(path.read_text())["probability_audit"]
    if (
        summary.get("group_sha256") != group_digest(group)
        or summary.get("policy_version") != group["policy_version"]
        or summary.get("tokens") != group["trainable_tokens"]
    ):
        raise ValueError("Probability audit is stale or does not cover this exact group")
    mean = summary["mean_absolute_log_ratio"]
    outside = summary["outside_ratio_0_8_to_1_2_fraction"]
    if not (
        math.isfinite(mean) and math.isfinite(outside) and 0 <= mean <= 0.1 and 0 <= outside <= 0.05
    ):
        raise ValueError("Sampler/learner probability disagreement exceeds readiness limits")
    return summary


def summarize_deltas(deltas: list[float]) -> dict:
    if not deltas or not all(math.isfinite(value) for value in deltas):
        raise ValueError("Probability audit requires finite token deltas")
    absolute = sorted(abs(value) for value in deltas)
    return {
        "tokens": len(deltas),
        "mean_log_ratio": statistics.mean(deltas),
        "mean_absolute_log_ratio": statistics.mean(absolute),
        "p95_absolute_log_ratio": absolute[math.ceil(0.95 * len(absolute)) - 1],
        "max_absolute_log_ratio": absolute[-1],
        "outside_ratio_0_8_to_1_2_fraction": sum(
            value < math.log(0.8) or value > math.log(1.2) for value in deltas
        )
        / len(deltas),
    }


def audit_group(trainer, group: dict, directory) -> dict:
    import torch
    import torch.distributed as dist

    rank = dist.get_rank()
    trainer.model.eval()
    records = []
    deltas = []
    with torch.no_grad():
        for row in group["rows"][rank :: dist.get_world_size()]:
            sequence = row["prompt_ids"] + row["completion_ids"]
            tokens = torch.tensor([sequence], device=trainer.accelerator.device)
            with trainer.compute_loss_context_manager():
                logps, _, _ = trainer._get_per_token_logps_and_entropies(
                    trainer.model,
                    tokens,
                    torch.ones_like(tokens),
                    len(row["completion_ids"]),
                )
            current = logps[0].float().cpu().tolist()
            difference = [
                new - old for new, old in zip(current, row["old_per_token_logps"], strict=True)
            ]
            if not row["padding"]:
                deltas.extend(difference)
                records.append(
                    {
                        "trajectory": row["trajectory"],
                        "turn": row["turn"],
                        "prompt_tokens": len(row["prompt_ids"]),
                        "reply_tokens": len(row["completion_ids"]),
                        "learner_logprobs": current,
                        "sampler_logprobs": row["old_per_token_logps"],
                        **summarize_deltas(difference),
                    }
                )
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, deltas)
    report = {
        "policy_version": group["policy_version"],
        "group_sha256": group_digest(group),
        "optimizer_steps": 0,
        "all_real_root_tokens_scored": True,
        "padding_excluded": True,
        **summarize_deltas([value for shard in gathered for value in shard]),
    }
    (directory / f"probability-audit-rank-{rank}.json").write_text(
        json.dumps({"summary": report, "rows": records}, indent=2, allow_nan=False)
    )
    return report
