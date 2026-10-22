"""Trajectory GRPO advantages attached to exact root-turn sampling contexts."""

import math
import statistics

from kernel_rollout.contract import RootSample, has_signal

from kernel_trl.context import LEGACY_CONTEXT_TOKENS, validate_context_tokens


def training_rows(
    episodes: list[dict], policy_version: int = 0, context_tokens: int = LEGACY_CONTEXT_TOKENS
) -> dict:
    validate_context_tokens(context_tokens)
    if len(episodes) not in {4, 8}:
        raise ValueError("TRL requires four or eight complete trajectories")
    if type(policy_version) is not int or policy_version < 0:
        raise ValueError("Policy version must be a non-negative integer")
    rewards = [float(episode["reward"]) for episode in episodes]
    if not has_signal(rewards):
        raise ValueError("Zero-variance terminal rewards: no optimizer update allowed")
    mean = statistics.mean(rewards)
    deviation = statistics.stdev(rewards)
    advantages = [(reward - mean) / (deviation + 1e-4) for reward in rewards]
    rows = []
    for trajectory, (episode, advantage) in enumerate(zip(episodes, advantages, strict=True)):
        if not episode["samples"]:
            raise ValueError("Cannot train a trajectory without exact root samples")
        for turn, sample in enumerate(episode["samples"]):
            if not isinstance(sample, RootSample):
                raise ValueError("Expected validated exact-token root samples")
            if (
                not sample.prompt_ids
                or not sample.response_ids
                or len(sample.logprobs) != len(sample.response_ids)
            ):
                raise ValueError("Incomplete root sample")
            if not all(math.isfinite(value) and value <= 0 for value in sample.logprobs):
                raise ValueError("Invalid sampled-token log probabilities")
            if len(sample.prompt_ids) + len(sample.response_ids) > context_tokens:
                raise ValueError("Exact sampled context exceeds the configured limit")
            rows.append(
                {
                    "prompt_ids": list(sample.prompt_ids),
                    "completion_ids": list(sample.response_ids),
                    "old_per_token_logps": list(sample.logprobs),
                    "advantage": advantage,
                    "trajectory": trajectory,
                    "turn": turn,
                    "terminal_reward": rewards[trajectory],
                    "policy_version": policy_version,
                    "padding": False,
                    "execution_run_id": episode.get("execution_run_id"),
                    "group_id": episode.get("group_id"),
                    "attempt_id": episode.get("attempt_id"),
                    "episode_id": episode.get("episode_id"),
                    "root_turn_id": sample.provenance.get("root_turn_id"),
                    "call_id": sample.provenance.get("call_id"),
                    "model_call_id": sample.provenance.get("model_call_id"),
                    "completion_token_start": 0,
                    "completion_token_end": len(sample.response_ids),
                    "loss_mask": [1] * len(sample.response_ids),
                    "effective_loss_mask": [1] * len(sample.response_ids),
                }
            )
    tokens = sum(len(row["completion_ids"]) for row in rows)
    real_rows = len(rows)
    while len(rows) % len(episodes):
        rows.append(
            {
                **rows[0],
                "advantage": 0.0,
                "effective_loss_mask": [0] * len(rows[0]["completion_ids"]),
                "padding": True,
            }
        )
    return {
        "rows": rows,
        "real_rows": real_rows,
        "trainable_tokens": tokens,
        "rewards": rewards,
        "advantages": advantages,
        "policy_version": policy_version,
        "context_tokens": context_tokens,
        "producer": "kernel_trl_training_rows",
        "execution_run_id": episodes[0].get("execution_run_id"),
        "group_id": episodes[0].get("group_id"),
        "attempt_id": episodes[0].get("attempt_id"),
        "episode_ids": [episode.get("episode_id") for episode in episodes],
        "normalization": "sum root-token PPO losses / real root tokens; no turn-count reward normalization",
        "padding": "duplicates a real root row with exactly zero advantage and no auxiliary loss",
    }


def tensors_for_turn(row: dict, total_tokens: int, device):
    import torch

    return {
        "prompt_ids": torch.tensor([row["prompt_ids"]], device=device),
        "prompt_mask": torch.ones((1, len(row["prompt_ids"])), dtype=torch.long, device=device),
        "completion_ids": torch.tensor([row["completion_ids"]], device=device),
        "completion_mask": torch.ones(
            (1, len(row["completion_ids"])), dtype=torch.long, device=device
        ),
        "old_per_token_logps": torch.tensor(
            [row["old_per_token_logps"]], dtype=torch.float32, device=device
        ),
        "advantages": torch.tensor([row["advantage"]], dtype=torch.float32, device=device),
        "num_items_in_batch": torch.tensor(float(total_tokens), device=device),
    }
