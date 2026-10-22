"""Validate the existing TRL exact-row handoff without importing trainer libraries."""

import hashlib
import json
import math
import statistics
from pathlib import Path

from kernel_rollout.artifacts import (
    GROUP_IDENTITY,
    artifact_version,
    require_identity,
    validate_generation_identity,
)

from kernel_trl.context import validate_context_tokens


def validate_episode_samples(episodes: list[dict], policy_version: int) -> None:
    manifest = {"schema_version": 1, "policy_version": policy_version}
    manifest.update({key: episodes[0].get(key) for key in GROUP_IDENTITY})
    for episode in episodes:
        if "policy_version" in episode and (
            type(episode["policy_version"]) is not int
            or episode["policy_version"] != policy_version
        ):
            raise ValueError("Episode policy differs from sampled policy")
        record = {**episode, "policy_version": policy_version}
        calls = [{**sample.provenance, "branch": "root"} for sample in episode["samples"]]
        validate_generation_identity(manifest, record, calls)


def content_digest(group: dict) -> str:
    """Canonical group content, excluding only its artifact_sha256 field."""
    content = {key: value for key, value in group.items() if key != "artifact_sha256"}
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_training_group(group: dict) -> None:
    versioned = artifact_version(group) == 1
    if group.get("artifact_sha256") is not None and group["artifact_sha256"] != content_digest(
        group
    ):
        raise ValueError("Training group content checksum mismatch")
    if not versioned:
        for row in group.get("rows", []):
            for key in (*GROUP_IDENTITY, "policy_version"):
                if (
                    group.get(key) is not None
                    and row.get(key) is not None
                    and row[key] != group[key]
                ):
                    raise ValueError("Recorded row identity differs from training group")
            episode_ids = group.get("episode_ids")
            trajectory = row.get("trajectory")
            if (
                isinstance(episode_ids, list)
                and type(trajectory) is int
                and 0 <= trajectory < len(episode_ids)
                and episode_ids[trajectory] is not None
                and row.get("episode_id") is not None
                and row["episode_id"] != episode_ids[trajectory]
            ):
                raise ValueError("Recorded row episode differs from training group")
        return
    require_identity(group, GROUP_IDENTITY)
    if group.get("artifact_sha256") != content_digest(group):
        raise ValueError("Training group content checksum mismatch")
    version = group["policy_version"]
    if type(version) is not int or version < 0:
        raise ValueError("Invalid sampled policy version")
    context = validate_context_tokens(group["context_tokens"])
    rewards, advantages, rows = group["rewards"], group["advantages"], group["rows"]
    if not all(isinstance(value, list) for value in (rewards, advantages, rows)):
        raise ValueError("Training rewards, advantages and rows must be lists")
    if len(rewards) not in {4, 8} or not all(math.isfinite(r) for r in rewards):
        raise ValueError("Expected four or eight finite rewards")
    deviation = statistics.stdev(rewards)
    if deviation == 0:
        raise ValueError("Training group has no reward signal")
    expected = [(r - statistics.mean(rewards)) / (deviation + 1e-4) for r in rewards]
    if advantages != expected:
        raise ValueError("Trajectory advantages differ from terminal rewards")
    episode_ids = group["episode_ids"]
    if (
        len(episode_ids) != len(rewards)
        or any(not isinstance(value, str) or not value.strip() for value in episode_ids)
        or len(set(episode_ids)) != len(episode_ids)
    ):
        raise ValueError("Expected distinct complete episode identities")
    real = group["real_rows"]
    if type(real) is not int or not 0 < real <= len(rows) or len(rows) % len(rewards):
        raise ValueError("Invalid real/padded row counts")
    if len(rows) - real != (-real) % len(rewards):
        raise ValueError("Unexpected number of padding rows")
    tokens = 0
    turns = [0] * len(rewards)
    calls, root_turns = set(), set()
    previous_trajectory = 0
    for index, row in enumerate(rows):
        require_identity(
            row, (*GROUP_IDENTITY, "episode_id", "root_turn_id", "call_id", "model_call_id")
        )
        if any(row[key] != group[key] for key in GROUP_IDENTITY):
            raise ValueError("Row identity differs from training group")
        prompt, completion = row["prompt_ids"], row["completion_ids"]
        if (
            not isinstance(prompt, list)
            or not isinstance(completion, list)
            or not prompt
            or not completion
            or any(type(t) is not int or t < 0 for t in [*prompt, *completion])
        ):
            raise ValueError("Invalid exact token IDs")
        logps = row["old_per_token_logps"]
        if (
            not isinstance(logps, list)
            or len(logps) != len(completion)
            or not all(math.isfinite(v) and v <= 0 for v in logps)
        ):
            raise ValueError("Invalid token/logprob alignment")
        if len(prompt) + len(completion) > context:
            raise ValueError("Exact row exceeds context limit")
        if (
            type(row["completion_token_start"]) is not int
            or type(row["completion_token_end"]) is not int
            or row["completion_token_start"] != 0
            or row["completion_token_end"] != len(completion)
        ):
            raise ValueError("Invalid half-open completion positions")
        padding = index >= real
        if type(row["padding"]) is not bool or row["padding"] != padding:
            raise ValueError("Real rows must precede padding")
        if any(
            not isinstance(row[key], list) or any(type(value) is not int for value in row[key])
            for key in ("loss_mask", "effective_loss_mask")
        ):
            raise ValueError("Serialized masks must be integer lists")
        if row["loss_mask"] != [1] * len(completion) or row["effective_loss_mask"] != [
            int(not padding)
        ] * len(completion):
            raise ValueError("Serialized masks differ from existing root-row semantics")
        trajectory = row["trajectory"]
        if type(trajectory) is not int or not 0 <= trajectory < len(rewards):
            raise ValueError("Invalid row trajectory")
        if (
            row["episode_id"] != episode_ids[trajectory]
            or type(row["policy_version"]) is not int
            or row["policy_version"] != version
        ):
            raise ValueError("Row episode/policy differs from training group")
        if row["terminal_reward"] != rewards[trajectory]:
            raise ValueError("Row reward differs from its trajectory")
        if row["advantage"] != (0.0 if padding else advantages[trajectory]):
            raise ValueError("Row advantage differs from its trajectory")
        if padding:
            original = {
                **rows[0],
                "padding": True,
                "advantage": 0.0,
                "effective_loss_mask": [0] * len(rows[0]["completion_ids"]),
            }
            if row != original:
                raise ValueError("Padding must duplicate the first real row with zero credit")
        else:
            if trajectory < previous_trajectory:
                raise ValueError("Real rows must remain in trajectory order")
            previous_trajectory = trajectory
            if (
                type(row["turn"]) is not int
                or row["turn"] != turns[trajectory]
                or row["call_id"] != row["model_call_id"]
            ):
                raise ValueError("Invalid row turn/call identity")
            turn_identity = (row["episode_id"], row["root_turn_id"])
            if row["call_id"] in calls or turn_identity in root_turns:
                raise ValueError("Real source call/turn is repeated")
            calls.add(row["call_id"])
            root_turns.add(turn_identity)
            turns[trajectory] += 1
            tokens += len(completion)
    if (
        not all(turns)
        or type(group["trainable_tokens"]) is not int
        or group["trainable_tokens"] != tokens
    ):
        raise ValueError("Real root-token denominator differs from rows")


def finalize_training_group(group: dict, *, legacy: bool = False) -> dict:
    artifact_version(group)
    if not legacy:
        group["schema_version"] = 1
    group["artifact_sha256"] = content_digest(group)
    validate_training_group(group)
    return group


def load_training_group(path: Path) -> tuple[dict, dict]:
    """Hash the same bytes that were read, not a later rewrite of the input file."""
    data = path.read_bytes()
    group = json.loads(data)
    validate_training_group(group)
    return group, {
        "group_input_path": str(path),
        "group_input_sha256": hashlib.sha256(data).hexdigest(),
        "generation_sha256": group.get("generation_sha256"),
    }
