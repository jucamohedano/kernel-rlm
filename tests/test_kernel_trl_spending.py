import importlib
import math
from pathlib import Path

import pytest


@pytest.fixture
def spending(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "training/environments/kernel_rlm"))
    return importlib.import_module("kernel_trl.spending")


def configuration():
    return {
        "budget_usd": 40,
        "displayed_pair_rate_usd_per_hour": 1.26,
        "goal_started_unix_seconds": 1000,
    }


def test_lower_budget_does_not_reset_elapsed_rental_spending(spending):
    values = configuration()
    snapshot = spending.spending_snapshot(values, 1000 + 10 * 3600)
    assert snapshot["estimated_spent_usd"] == pytest.approx(12.6)
    assert snapshot["estimated_remaining_usd"] == pytest.approx(27.4)
    values["budget_usd"] = 20
    assert spending.spending_snapshot(values, 37000)["estimated_remaining_usd"] == pytest.approx(
        7.4
    )


def test_full_phase_and_cleanup_must_fit_before_launch(spending):
    assert spending.phase_deadline(configuration(), 1000, 900) == 1900
    with pytest.raises(ValueError, match="Insufficient"):
        spending.phase_deadline(configuration(), 1000 + 31 * 3600, 3600)
    assert (
        spending.spending_snapshot(configuration(), 1000 + 40 * 3600)["estimated_remaining_usd"]
        == 0
    )


@pytest.mark.parametrize("seconds", [-1, 0, math.inf, math.nan])
def test_invalid_phase_lengths_fail_closed(spending, seconds):
    with pytest.raises(ValueError):
        spending.phase_deadline(configuration(), 1000, seconds)


def test_clock_before_original_start_is_rejected(spending):
    with pytest.raises(ValueError):
        spending.spending_snapshot(configuration(), 999)
