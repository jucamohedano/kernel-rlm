import importlib
import math
from pathlib import Path

import pytest


@pytest.fixture
def credit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "training/environments/kernel_rlm"))
    return importlib.import_module("kernel_trl.turn_credit")


def episodes():
    from kernel_skyrl.contract import RootSample

    return [
        {
            "reward": reward,
            "samples": [
                RootSample([2, 10], [20 + trajectory, 2], [-4.0, -4.0], "stop")
                for turn in range(turns)
            ],
            "child_samples": [RootSample([999], [998], [-1.0], "stop")],
        }
        for trajectory, (reward, turns) in enumerate(
            zip([0, 0.2, 0.8, 1.0], [1, 3, 1, 2], strict=True)
        )
    ]


def test_advantage_is_per_trajectory_not_per_turn(credit):
    result = credit.training_rows(episodes())
    assert result["real_rows"] == 7
    assert result["trainable_tokens"] == 14
    assert len(result["rows"]) == 8
    for row in result["rows"]:
        assert row["advantage"] == (
            0.0 if row["padding"] else result["advantages"][row["trajectory"]]
        )
        assert 998 not in row["completion_ids"]
    assert sum(result["advantages"]) == pytest.approx(0)


def test_context_is_not_reconstructed_by_concatenating_turns(credit):
    from kernel_skyrl.contract import RootSample

    values = episodes()
    values[1]["samples"][1] = RootSample([2, 50, 7], [60, 2], [-3.0, -2.0], "stop")
    result = credit.training_rows(values)
    assert result["rows"][2]["prompt_ids"] == [2, 50, 7]
    assert result["rows"][2]["completion_ids"] == [60, 2]


@pytest.mark.parametrize("reward", [0.0, 0.8, 1.0])
def test_uniform_reward_group_never_reaches_training(credit, reward):
    values = episodes()
    for episode in values:
        episode["reward"] = reward
    with pytest.raises(ValueError, match="Zero-variance"):
        credit.training_rows(values)


def test_speed_only_variance_is_retained(credit):
    values = episodes()
    for index, episode in enumerate(values):
        episode["reward"] = 0.8 + index * 1e-7
    assert credit.training_rows(values)["advantages"][-1] > 0


def test_bad_logprobs_fail_closed(credit):
    from kernel_skyrl.contract import RootSample

    values = episodes()
    values[0]["samples"][0] = RootSample([2], [4], [math.nan], "stop")
    with pytest.raises(ValueError, match="log probabilities"):
        credit.training_rows(values)
