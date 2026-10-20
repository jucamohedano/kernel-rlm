"""CPU-only authorization checks for the optional PRIME-RL component probe."""

import pytest
from kernel_trl.prime_probe import authorize_probe


def configuration() -> dict:
    return {
        "budget_usd": 20,
        "displayed_pair_rate_usd_per_hour": 2.78,
        "goal_started_unix_seconds": 1000,
    }


def test_probe_keeps_original_spending_baseline():
    settings = configuration()
    assert authorize_probe(settings, 4600) == 5200
    assert settings["goal_started_unix_seconds"] == 1000


def test_probe_reserves_cleanup_and_whole_phase():
    with pytest.raises(ValueError, match="Insufficient"):
        authorize_probe(configuration(), 1000 + 18 * 3600 / 2.78)


def test_l40_budget_cannot_replace_rtx_authorization():
    settings = configuration()
    settings["budget_usd"] = 60
    with pytest.raises(ValueError, match="separately authorized"):
        authorize_probe(settings, 4600)
