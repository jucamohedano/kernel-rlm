"""Exact-token, root-only training samples for the existing kernel RLM loop."""

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RootSample:
    prompt_ids: list[int]
    response_ids: list[int]
    logprobs: list[float]
    stop_reason: str
    provenance: dict[str, Any] = field(default_factory=dict)


def exact_sample(
    response: dict[str, Any], context_limit: int, provenance: dict[str, Any] | None = None
) -> RootSample:
    choices = response.get("choices", [])
    if len(choices) != 1:
        raise ValueError("Expected exactly one completion per model request")
    choice = choices[0]
    prompt = response.get("prompt_token_ids")
    tokens = choice.get("token_ids")
    entries = (choice.get("logprobs") or {}).get("content")
    if not prompt or not tokens or not entries:
        raise ValueError("Missing exact prompt/response token IDs or sampled-token logprobs")
    if any(type(token) is not int or token < 0 for token in [*prompt, *tokens]):
        raise ValueError("Token IDs must be non-negative integers")
    if len(tokens) != len(entries) or len(prompt) + len(tokens) > context_limit:
        raise ValueError("Token/logprob alignment or context limit violated")
    logprobs = []
    for token, entry in zip(tokens, entries, strict=True):
        if entry.get("token") != f"token_id:{token}":
            raise ValueError("Sampled-token logprob does not match the response token ID")
        value = float(entry["logprob"])
        if not math.isfinite(value) or value > 0:
            raise ValueError("Invalid sampled-token logprob")
        logprobs.append(value)
    reason = choice.get("finish_reason")
    if reason not in {"stop", "length", "tool_calls"}:
        raise ValueError(f"Unusable completion finish reason: {reason}")
    return RootSample(list(prompt), list(tokens), logprobs, reason, dict(provenance or {}))


def terminal_reward(result: dict[str, Any], speed_weight: float = 0.1) -> float:
    if not math.isfinite(speed_weight) or not 0 <= speed_weight <= 0.1:
        raise ValueError("Speed reward weight must be between zero and 0.1")
    reward = 0.1
    if result.get("compiled") and result.get("extension_used"):
        reward += 0.2
    if result.get("correct"):
        speedup = float(result.get("speedup") or 0)
        if not math.isfinite(speedup):
            raise ValueError("Verifier returned non-finite speedup")
        reward += 0.4 + speed_weight * min(max(speedup, 0), 3)
    return reward


def has_signal(rewards: list[float]) -> bool:
    if len(rewards) < 2 or not all(math.isfinite(value) for value in rewards):
        raise ValueError("GRPO requires at least two finite terminal rewards")
    return any(value != rewards[0] for value in rewards[1:])


def stepwise_output(episodes: list[dict[str, Any]], trajectory_ids: list[Any]) -> dict[str, Any]:
    if len(episodes) != len(trajectory_ids) or not has_signal(
        [episode["reward"] for episode in episodes]
    ):
        raise ValueError("Cannot train a zero-variance or incomplete group")
    output: dict[str, Any] = {
        key: []
        for key in (
            "prompt_token_ids",
            "response_ids",
            "rewards",
            "loss_masks",
            "stop_reasons",
            "rollout_logprobs",
            "trajectory_ids",
            "is_last_step",
            "env_metrics",
        )
    }
    output.update(
        rollout_metrics={},
        trajectory_generation_times=None,
        trajectory_time_splits=None,
        rollout_expert_indices=None,
        rollout_sample_support=None,
        pixel_values=None,
        image_grid_thw=None,
    )
    for episode, trajectory_id in zip(episodes, trajectory_ids, strict=True):
        samples = episode["samples"]
        if not samples:
            raise ValueError("An episode has no root samples; refusing to fabricate tokens")
        for index, sample in enumerate(samples):
            last = index == len(samples) - 1
            output["prompt_token_ids"].append(sample.prompt_ids)
            output["response_ids"].append(sample.response_ids)
            output["rollout_logprobs"].append(sample.logprobs)
            output["loss_masks"].append([1] * len(sample.response_ids))
            output["rewards"].append(episode["reward"] if last else 0.0)
            output["stop_reasons"].append(sample.stop_reason)
            output["trajectory_ids"].append(trajectory_id)
            output["is_last_step"].append(last)
            output["env_metrics"].append({"kernel_reward": episode["reward"]} if last else {})
    return output
