"""Token counting and shared generated-token reservations for native inference."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextBudgetPolicy:
    server_context_tokens: int = 262144
    output_fraction: float = 0.9
    max_generated_tokens: int = 1_000_000
    logprobs: bool = True

    def __post_init__(self):
        if (
            not isinstance(self.server_context_tokens, int)
            or isinstance(self.server_context_tokens, bool)
        ) or self.server_context_tokens <= 0:
            raise ValueError("server_context_tokens must be a positive integer")
        if (
            isinstance(self.output_fraction, bool)
            or not math.isfinite(self.output_fraction)
            or not 0 < self.output_fraction <= 1
        ):
            raise ValueError("output_fraction must be finite and in (0, 1]")
        if (
            not isinstance(self.max_generated_tokens, int)
            or isinstance(self.max_generated_tokens, bool)
        ) or self.max_generated_tokens <= 0:
            raise ValueError("max_generated_tokens must be a positive integer")
        if not isinstance(self.logprobs, bool):
            raise ValueError("logprobs must be a boolean")


class VllmPromptCounter:
    """Count a fully rendered chat payload through vLLM's /tokenize endpoint."""

    def __init__(self, client: Any, model_context: int):
        if (
            not isinstance(model_context, int) or isinstance(model_context, bool)
        ) or model_context <= 0:
            raise ValueError("model_context must be the exact positive max_model_len")
        self.client = client
        self.model_context = model_context
        base_url = str(client.base_url).rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        self.endpoint = base_url + "/tokenize"

    async def __call__(self, payload: dict[str, Any]) -> int:
        request = {
            key: payload[key]
            for key in ("model", "messages", "tools", "chat_template_kwargs")
            if key in payload
        }
        request.update(
            add_generation_prompt=True, continue_final_message=False, add_special_tokens=False
        )
        response = await self.client.post(self.endpoint, json=request)
        response.raise_for_status()
        data = response.json()
        if data.get("max_model_len") != self.model_context:
            raise ValueError("vLLM max_model_len differs from configured model_context")
        count = data.get("count")
        if count is None and isinstance(data.get("tokens"), list):
            count = len(data["tokens"])
        if (not isinstance(count, int) or isinstance(count, bool)) or count < 0:
            raise ValueError("vLLM /tokenize response has no valid token count")
        return count


class BudgetReservationPending(RuntimeError):
    """Allowance is reserved by running requests, not yet spent."""


class GeneratedTokenBudget:
    """Episode-wide fail-closed settlement for generated token reservations."""

    def __init__(self, policy: ContextBudgetPolicy):
        self.policy = policy
        self.used = 0
        self.reserved: dict[str, int] = {}
        self.incomplete = False
        self.changed = asyncio.Event()

    @property
    def outstanding(self) -> int:
        return sum(self.reserved.values())

    def reserve(self, call_id: str, prompt_tokens: int) -> int:
        if self.incomplete:
            raise RuntimeError("generated-token budget state is incomplete")
        if (
            not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool)
        ) or prompt_tokens < 0:
            raise ValueError("prompt token count must be a nonnegative integer")
        context_remaining = self.policy.server_context_tokens - prompt_tokens
        if context_remaining <= 0:
            raise RuntimeError("prompt leaves no model context for generation")
        cap = min(
            self.policy.max_generated_tokens - self.used - self.outstanding,
            math.floor(self.policy.output_fraction * context_remaining),
        )
        if cap <= 0:
            if self.used < self.policy.max_generated_tokens and self.outstanding:
                self.changed.clear()
                raise BudgetReservationPending("allowance reserved by running requests")
            raise RuntimeError("episode generated-token budget exhausted")
        if call_id in self.reserved:
            raise ValueError("duplicate generated-token reservation")
        self.reserved[call_id] = cap
        return cap

    def settle(self, call_id: str, usage: Any) -> tuple[int | None, int | None]:
        reserved = self.reserved.get(call_id)
        if reserved is None:
            raise RuntimeError("unknown generated-token reservation")
        actual = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if (
            (not isinstance(actual, int) or isinstance(actual, bool))
            or actual < 0
            or actual > reserved
        ):
            self.incomplete = True
            self.changed.set()
            return None, reserved
        del self.reserved[call_id]
        self.used += actual
        self.changed.set()
        return actual, reserved

    def release_undispatched(self, call_id: str) -> None:
        del self.reserved[call_id]
        self.changed.set()

    def retain_uncertain(self, call_id: str) -> int:
        if call_id not in self.reserved:
            raise RuntimeError("unknown generated-token reservation")
        self.incomplete = True
        self.changed.set()
        return self.reserved[call_id]
