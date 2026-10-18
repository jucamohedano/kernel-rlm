"""Conservative launch deadlines from the original rental-spending baseline."""

import math


def spending_snapshot(configuration: dict, now: float) -> dict:
    limit = float(configuration["budget_usd"])
    rate = float(configuration["displayed_pair_rate_usd_per_hour"])
    started = float(configuration["goal_started_unix_seconds"])
    if not all(math.isfinite(value) for value in (limit, rate, started, now)):
        raise ValueError("Spending inputs must be finite")
    if limit <= 0 or rate <= 0 or now < started:
        raise ValueError("Invalid spending ceiling, rate or original start time")
    spent = (now - started) * rate / 3600
    return {
        "budget_usd": limit,
        "estimated_spent_usd": spent,
        "estimated_remaining_usd": max(0.0, limit - spent),
        "started_unix_seconds": started,
        "assumed_pair_rate_usd_per_hour": rate,
        "estimate_not_invoice": True,
        "rental_can_bill_after_gpu_processes_exit": True,
    }


def phase_deadline(
    configuration: dict, now: float, requested_seconds: float, reserve_usd: float = 2
) -> float:
    snapshot = spending_snapshot(configuration, now)
    if not math.isfinite(requested_seconds) or requested_seconds <= 0:
        raise ValueError("Phase duration must be finite and positive")
    if not math.isfinite(reserve_usd) or reserve_usd < 0:
        raise ValueError("Cleanup reserve must be finite and nonnegative")
    available = snapshot["estimated_remaining_usd"] - reserve_usd
    needed = requested_seconds * snapshot["assumed_pair_rate_usd_per_hour"] / 3600
    if needed > available:
        raise ValueError("Insufficient original spending budget for this phase and cleanup reserve")
    return now + requested_seconds
