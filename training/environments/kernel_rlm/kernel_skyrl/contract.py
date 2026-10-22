"""SkyRL stepwise output conversion for exact-token episodes."""

from typing import Any

from kernel_rollout.contract import has_signal


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
