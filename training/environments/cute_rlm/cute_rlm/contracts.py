"""Explicit correctness, state, and timing settings for CuTe candidate checks."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class VerificationConfig:
    atol: float
    rtol: float
    seed: int
    mode: str
    correctness_runs: int
    warmup_runs: int
    timed_runs: int
    timeout_s: float
    input_device: str = "cuda"

    def __post_init__(self):
        for name in ("atol", "rtol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.input_device != "cuda":
            raise ValueError("The GPU task input placement must explicitly be cuda")
        if type(self.seed) is not int or self.mode not in {"preserve", "eval"}:
            raise ValueError("Choose an integer seed and explicit preserve/eval mode")
        for name in ("correctness_runs", "timed_runs"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.warmup_runs) is not int or self.warmup_runs < 0:
            raise ValueError("warmup_runs must be a nonnegative integer")
        if (
            isinstance(self.timeout_s, bool)
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise ValueError("timeout_s must be finite and positive")

    def payload(self):
        return asdict(self)
