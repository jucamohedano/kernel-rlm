"""Persist generated groups and score them sequentially after inference exits."""

import argparse
import asyncio
import hashlib
import json
import os
import shutil
from pathlib import Path

from kernel_skyrl.contract import exact_sample, has_signal
from kernel_skyrl.generator import score_submission

from kernel_trl.context import LEGACY_CONTEXT_TOKENS, validate_context_tokens
from kernel_trl.turn_credit import training_rows

INFRASTRUCTURE_STATUSES = {
    "infrastructure_error",
    "runner_crash",
    "invalid_runner_output",
    "timeout",
}


def read_json(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def capture_generation(
    directory: Path,
    task_id: str,
    version: int,
    rollouts: int,
    context_tokens: int = LEGACY_CONTEXT_TOKENS,
) -> dict:
    validate_context_tokens(context_tokens)
    if rollouts not in {4, 8}:
        raise ValueError("Require a complete four- or eight-trajectory group")
    entries = []
    for index in range(rollouts):
        episode = directory / f"group-0-trajectory-{index}"
        files = {
            name: digest(episode / name) for name in ("task.json", "submission.json", "calls.jsonl")
        }
        entries.append({"directory": episode.name, "sha256": files})
    manifest = {"task": task_id, "policy_version": version, "episodes": entries}
    if context_tokens != LEGACY_CONTEXT_TOKENS:
        manifest["context_tokens"] = context_tokens
    destination = directory / "generation-manifest.json"
    if destination.exists() and read_json(destination) != manifest:
        raise ValueError("Generated trajectories changed since capture")
    write_json(destination, manifest)
    return manifest


def load_episodes(directory: Path, manifest: dict) -> list[dict]:
    episodes = []
    context_tokens = validate_context_tokens(manifest.get("context_tokens", LEGACY_CONTEXT_TOKENS))
    if len(manifest["episodes"]) not in {4, 8}:
        raise ValueError("Incomplete generated group")
    for entry in manifest["episodes"]:
        episode = directory / entry["directory"]
        for name, expected in entry["sha256"].items():
            if digest(episode / name) != expected:
                raise ValueError(f"Generated artifact checksum mismatch: {episode / name}")
        task = read_json(episode / "task.json")
        if task.get("name") != manifest["task"]:
            raise ValueError("Saved task differs from generated group")
        calls = [json.loads(line) for line in (episode / "calls.jsonl").read_text().splitlines()]
        if any(call.get("policy_version") != manifest["policy_version"] for call in calls):
            raise ValueError("Saved calls have a different sampled policy version")
        samples = [
            exact_sample(call["response"], context_tokens)
            for call in calls
            if call.get("branch") == "root"
        ]
        if not samples:
            raise ValueError("No exact sampled root calls")
        episodes.append(
            {
                "directory": episode,
                "task": task,
                "record": read_json(episode / "submission.json"),
                "samples": samples,
            }
        )
    return episodes


async def score_generated(
    directory: Path, version: int, task_id: str, scorer=score_submission, on_score=None
) -> dict:
    manifest_path = directory / "generation-manifest.json"
    manifest = read_json(manifest_path)
    if manifest["policy_version"] != version or manifest["task"] != task_id:
        raise ValueError("Generated group differs from requested task/policy")
    episodes = load_episodes(directory, manifest)
    state_path = directory / "scoring-state.json"
    state = (
        read_json(state_path)
        if state_path.exists()
        else {"generation_sha256": digest(manifest_path), "scores": {}}
    )
    if state["generation_sha256"] != digest(manifest_path):
        raise ValueError("Scoring state belongs to different sampled trajectories")
    repaired = []
    for index, episode in enumerate(episodes):
        score_path = episode["directory"] / "score.json"
        cached = state["scores"].get(str(index))
        if cached and cached["sha256"] != digest(score_path):
            raise ValueError("Persisted score changed since verification")
        score = read_json(score_path) if score_path.exists() else None
        if score and score.get("policy_version") != version:
            raise ValueError("Persisted score belongs to a different policy")
        attempts = cached.get("attempts", 0) if cached else 0
        while (
            score is None
            or score.get("verification", {}).get("verification_status") in INFRASTRUCTURE_STATUSES
        ):
            if attempts >= 2:
                break
            if score is not None and not (episode["directory"] / "score-original.json").exists():
                write_json(episode["directory"] / "score-original.json", score)
            reward, verification = await scorer(episode["task"], episode["record"])
            attempts += 1
            score = {
                "reward": reward,
                "verification": verification,
                "policy_version": version,
                "root_calls": len(episode["samples"]),
            }
            write_json(score_path, score)
            state["scores"][str(index)] = {"sha256": digest(score_path), "attempts": attempts}
            write_json(state_path, state)
            repaired.append(index)
        if score is None:
            raise ValueError("Missing score after bounded verification")
        state["scores"][str(index)] = {"sha256": digest(score_path), "attempts": attempts}
        write_json(state_path, state)
        episode.update(reward=score["reward"], verification=score.get("verification", {}))
        if on_score is not None:
            on_score(index, score, index not in repaired)
    errors = [
        {"trajectory": index, "verification": episode["verification"]}
        for index, episode in enumerate(episodes)
        if episode["verification"].get("verification_status") in INFRASTRUCTURE_STATUSES
    ]
    rewards = [episode["reward"] for episode in episodes]
    admitted = not errors and max(rewards) > 0.1 and has_signal(rewards)
    record = {
        "task": task_id,
        "policy_version": version,
        "rewards": rewards,
        "admitted": admitted,
        "reason": "infrastructure_error"
        if errors
        else "admitted"
        if admitted
        else "weak_signal"
        if max(rewards) <= 0.1
        else "zero_variance",
        "infrastructure_errors": errors,
        "reverified_trajectories": sorted(set(repaired)),
    }
    write_json(directory / "groups.json", [record])
    if admitted:
        group = training_rows(
            episodes, version, manifest.get("context_tokens", LEGACY_CONTEXT_TOKENS)
        )
        group.update(task=task_id, generation_sha256=digest(manifest_path))
        write_json(directory / "training-group.json", group)
    return record


def import_saved_group(
    source: Path,
    destination: Path,
    task: dict,
    version: int,
    rollouts: int,
    checkpoint: Path | None = None,
    model_revision: str | None = None,
) -> None:
    if model_revision is not None:
        from kernel_trl.checkpoint_state import checkpoint_metadata

        original_run = source.parents[3]
        inputs = read_json(original_run / "inputs.json")
        state = read_json(original_run / "state.json")
        if inputs.get("model_revision") != model_revision or state.get("policy_version") != version:
            raise ValueError("Replay model revision or active sampled policy differs")
        if version:
            if checkpoint is None:
                raise ValueError("Replay requires its sampled adapter checkpoint")
            previous = checkpoint_metadata(Path(state["checkpoint"]), version)
            current = checkpoint_metadata(checkpoint, version)
            if previous.get("sha256") != current.get("sha256"):
                raise ValueError("Replay checkpoint differs from the sampled policy")
    records = read_json(source / "groups.json")
    if (
        len(records) != 1
        or records[0]["task"] != task["name"]
        or records[0]["policy_version"] != version
        or records[0].get("admitted")
    ):
        raise ValueError("Replay requires a previously rejected group from the same task/policy")
    expected = [source / f"group-0-trajectory-{index}" for index in range(rollouts)]
    for episode in expected:
        saved = read_json(episode / "task.json")
        if saved.get("name") != task["name"] or saved.get("context") != task["context"]:
            raise ValueError("Replay task context differs from the current curriculum")
    provenance = {
        "source_directory": str(source.resolve()),
        "source_groups_sha256": digest(source / "groups.json"),
        "sampled_model_revision": model_revision,
        "sampled_checkpoint": str(checkpoint) if checkpoint else None,
        "files": {
            str(path.relative_to(source)): digest(path)
            for episode in expected
            for path in episode.iterdir()
            if path.is_file()
        },
    }
    shutil.copytree(source, destination)
    if (destination / "training-group.json").exists():
        raise ValueError("Refusing to replay an existing trainable group")
    write_json(destination / "recovery-source.json", provenance)
    shutil.copy2(destination / "groups.json", destination / "groups-original.json")
    saved_manifest = source / "generation-manifest.json"
    context_tokens = (
        read_json(saved_manifest).get("context_tokens", LEGACY_CONTEXT_TOKENS)
        if saved_manifest.exists()
        else LEGACY_CONTEXT_TOKENS
    )
    capture_generation(destination, task["name"], version, rollouts, context_tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--policy-version", type=int, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--report-to", choices=("trackio", "none"), default="trackio")
    arguments = parser.parse_args()
    from kernel_trl.live_curriculum import ensure_gpus_idle

    ensure_gpus_idle()
    os.environ.pop("KERNEL_INFERENCE_ADMIN_URL", None)
    tracker = None
    if arguments.report_to == "trackio":
        import trackio

        tracker = trackio
        tracker.init(
            project="laguna-kernel-rlm",
            name=f"{arguments.directory.parents[3].name}-{arguments.task_id}-final-scoring",
            space_id=None,
            auto_log_gpu=False,
            auto_log_cpu=False,
            embed=False,
            config={
                "policy_version": arguments.policy_version,
                "recovered_group": (arguments.directory / "recovery-source.json").exists(),
            },
        )

    def log_score(index, score, reused):
        if tracker is not None:
            tracker.log(
                {
                    "rollout/completed_final_scores": index + 1,
                    "rollout/reward": score["reward"],
                    "verification/reused": int(reused),
                    "verification/correct": int(
                        score.get("verification", {}).get("correct") is True
                    ),
                }
            )

    try:
        result = asyncio.run(
            score_generated(
                arguments.directory, arguments.policy_version, arguments.task_id, on_score=log_score
            )
        )
        print(json.dumps(result), flush=True)
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    main()
