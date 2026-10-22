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
