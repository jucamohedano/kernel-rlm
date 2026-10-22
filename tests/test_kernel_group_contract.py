import asyncio
import hashlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from kernel_rollout.contract import RootSample
from kernel_trl.group_contract import (
    content_digest,
    finalize_training_group,
    load_training_group,
    validate_episode_samples,
    validate_training_group,
)
from kernel_trl.probability_audit import group_digest
from kernel_trl.scoring import capture_generation, load_episodes, score_generated
from kernel_trl.turn_credit import training_rows


@pytest.fixture
def episodes():
    result = []
    for index, reward in enumerate([0.1, 0.3, 0.7, 0.9]):
        identity = {
            "execution_run_id": "run",
            "group_id": "group",
            "attempt_id": "attempt",
            "episode_id": f"episode-{index}",
        }
        samples = []
        for turn in range(2 if index == 0 else 1):
            call = f"call-{index}-{turn}"
            samples.append(
                RootSample(
                    [1, 2],
                    [3, 4],
                    [-0.1, -0.2],
                    "stop",
                    {
                        **identity,
                        "policy_version": 2,
                        "root_turn_id": f"turn-{index}-{turn}",
                        "call_id": call,
                        "model_call_id": call,
                    },
                )
            )
        result.append({**identity, "reward": reward, "samples": samples})
    return result


@pytest.fixture
def group(episodes):
    validate_episode_samples(episodes, 2)
    return finalize_training_group(training_rows(episodes, 2))


def test_hash_scopes_and_padding_round_trip(group, tmp_path):
    path = tmp_path / "training-group.json"
    path.write_text(json.dumps(group, indent=2))
    loaded, report = load_training_group(path)
    assert loaded == group
    assert group["schema_version"] == 1
    assert group["real_rows"] == 5
    assert group["trainable_tokens"] == 10
    assert len(group["rows"]) == 8
    assert report["group_input_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert group["artifact_sha256"] == content_digest(group)
    assert group_digest(group) != content_digest(group)
    path.write_text(json.dumps(group, separators=(",", ":")))
    formatted, changed_report = load_training_group(path)
    assert group_digest(formatted) == group_digest(group)
    assert content_digest(formatted) == content_digest(group)
    assert changed_report["group_input_sha256"] != report["group_input_sha256"]
    assert all(
        row["advantage"] == 0 and not any(row["effective_loss_mask"]) for row in group["rows"][5:]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda g: g["rows"][0].update(completion_token_end=1),
        lambda g: g["rows"][0].update(loss_mask=[1]),
        lambda g: g["rows"][0].update(effective_loss_mask=[0, 0]),
        lambda g: g["rows"][-1].update(advantage=1),
        lambda g: g["rows"][-1].update(completion_ids=[9, 4]),
        lambda g: g.update(trainable_tokens=16),
        lambda g: g.update(real_rows=4),
        lambda g: g["rows"][0].update(old_per_token_logps=[-0.1]),
        lambda g: g["rows"][0].update(prompt_ids=[True]),
        lambda g: g["rows"][1].update(episode_id="other"),
        lambda g: g["rows"][1].update(group_id="other"),
        lambda g: g["rows"][1].update(call_id="call-0-0", model_call_id="call-0-0"),
        lambda g: g["advantages"].__setitem__(0, 1),
        lambda g: g["rows"][1].update(advantage=0),
        lambda g: g["rows"][1].update(turn=0),
        lambda g: g["episode_ids"].__setitem__(1, "episode-0"),
    ],
)
def test_invalid_rows_fail_even_with_matching_content_hash(group, mutation):
    mutation(group)
    group["artifact_sha256"] = content_digest(group)
    with pytest.raises(ValueError):
        validate_training_group(group)


@pytest.mark.parametrize("version", [0, 2, True, "1", None])
def test_explicit_unsupported_version_fails_before_trainer_import(
    group, version, tmp_path, monkeypatch
):
    from kernel_trl.rlm_update import run

    group["schema_version"] = version
    path = tmp_path / "training-group.json"
    path.write_text(json.dumps(group))
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError):
        run(SimpleNamespace(group=path))


def test_legacy_read_does_not_upgrade_or_invent_identity(group, tmp_path):
    del group["schema_version"]
    del group["rows"][0]["call_id"]
    group["artifact_sha256"] = content_digest(group)
    path = tmp_path / "training-group.json"
    path.write_text(json.dumps(group))
    original = path.read_bytes()
    loaded, _ = load_training_group(path)
    assert "schema_version" not in loaded
    assert "call_id" not in loaded["rows"][0]
    assert path.read_bytes() == original


def test_modified_content_fails_hash(group):
    group["rows"][0]["completion_ids"][0] = 99
    with pytest.raises(ValueError):
        validate_training_group(group)


@pytest.fixture
def generated(episodes, tmp_path):
    for index, episode in enumerate(episodes):
        directory = tmp_path / f"group-0-trajectory-{index}"
        directory.mkdir()
        identity = {
            key: episode[key]
            for key in ("execution_run_id", "group_id", "attempt_id", "episode_id")
        }
        (directory / "task.json").write_text(json.dumps({"name": "task"}))
        (directory / "submission.json").write_text(
            json.dumps({**identity, "policy_version": 2, "submission": "source"})
        )
        calls = []
        for sample in episode["samples"]:
            calls.append(
                {
                    **sample.provenance,
                    "branch": "root",
                    "response": {
                        "prompt_token_ids": sample.prompt_ids,
                        "choices": [
                            {
                                "token_ids": sample.response_ids,
                                "finish_reason": "stop",
                                "logprobs": {
                                    "content": [
                                        {"token": f"token_id:{token}", "logprob": logp}
                                        for token, logp in zip(
                                            sample.response_ids, sample.logprobs, strict=True
                                        )
                                    ]
                                },
                            }
                        ],
                    },
                }
            )
        calls.append(
            {
                **identity,
                "policy_version": 2,
                "branch": "child",
                "call_id": f"child-{index}",
                "model_call_id": f"child-{index}",
                "response": {"unused_child_tokens": [999]},
            }
        )
        (directory / "calls.jsonl").write_text("\n".join(json.dumps(call) for call in calls))
        (directory / "score.json").write_text(
            json.dumps({"policy_version": 2, "reward": episode["reward"]})
        )
    manifest = capture_generation(tmp_path, "task", 2, 4)
    return tmp_path, manifest


def test_deferred_round_trip_preserves_exact_root_rows(generated, group):
    directory, manifest = generated
    assert manifest["schema_version"] == 1
    asyncio.run(score_generated(directory, 2, "task", scorer=AsyncMock()))
    loaded, report = load_training_group(directory / "training-group.json")
    assert loaded["rows"] == group["rows"]
    assert loaded["schema_version"] == 1
    assert (
        report["generation_sha256"]
        == hashlib.sha256((directory / "generation-manifest.json").read_bytes()).hexdigest()
    )
    assert all(999 not in row["completion_ids"] for row in loaded["rows"])


@pytest.mark.parametrize(
    "target,key,value",
    [
        ("submission.json", "execution_run_id", "other"),
        ("submission.json", "episode_id", ""),
        ("submission.json", "policy_version", 1),
        ("calls.jsonl", "group_id", "other"),
        ("calls.jsonl", "episode_id", None),
        ("calls.jsonl", "call_id", ""),
        ("calls.jsonl", "root_turn_id", 7),
    ],
)
def test_cross_file_identity_failure_precedes_scoring_even_with_refreshed_hashes(
    generated, target, key, value
):
    directory, manifest = generated
    path = directory / "group-0-trajectory-0" / target
    if target == "calls.jsonl":
        calls = [json.loads(line) for line in path.read_text().splitlines()]
        calls[1][key] = value
        path.write_text("\n".join(json.dumps(call) for call in calls))
    else:
        record = json.loads(path.read_text())
        record[key] = value
        path.write_text(json.dumps(record))
    manifest["episodes"][0]["sha256"][target] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "generation-manifest.json").write_text(json.dumps(manifest))
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    scorer = AsyncMock()
    with pytest.raises(ValueError):
        asyncio.run(score_generated(directory, 2, "task", scorer=scorer))
    scorer.assert_not_called()
    assert before == {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}


def test_new_capture_requires_complete_identity_without_legacy_downgrade(generated):
    directory, _ = generated
    path = directory / "group-0-trajectory-0/submission.json"
    record = json.loads(path.read_text())
    del record["execution_run_id"]
    path.write_text(json.dumps(record))
    manifest_path = directory / "generation-manifest.json"
    manifest_path.unlink()
    with pytest.raises(ValueError):
        capture_generation(directory, "task", 2, 4)
    assert not manifest_path.exists()


def test_immediate_sample_identity_checked_before_conversion(episodes):
    sample = episodes[0]["samples"][1]
    sample.provenance["episode_id"] = "other"
    with pytest.raises(ValueError):
        validate_episode_samples(episodes, 2)


@pytest.mark.parametrize("version", [True, 0, 2, "1", None])
def test_generation_unknown_explicit_version_precedes_scorer(generated, version):
    directory, manifest = generated
    manifest["schema_version"] = version
    (directory / "generation-manifest.json").write_text(json.dumps(manifest))
    scorer = AsyncMock()
    with pytest.raises(ValueError):
        asyncio.run(score_generated(directory, 2, "task", scorer=scorer))
    scorer.assert_not_called()
    assert not (directory / "scoring-state.json").exists()


def test_legacy_generation_recapture_preserves_bytes(generated):
    directory, manifest = generated
    del manifest["schema_version"]
    path = directory / "generation-manifest.json"
    path.write_text(json.dumps(manifest, separators=(",", ":")))
    original = path.read_bytes()
    assert "schema_version" not in capture_generation(directory, "task", 2, 4)
    assert path.read_bytes() == original


def test_generation_duplicate_call_across_episodes_is_rejected(generated):
    directory, manifest = generated
    path = directory / "group-0-trajectory-1/calls.jsonl"
    calls = [json.loads(line) for line in path.read_text().splitlines()]
    calls[0].update(call_id="call-0-0", model_call_id="call-0-0")
    path.write_text("\n".join(json.dumps(call) for call in calls))
    manifest["episodes"][1]["sha256"]["calls.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        load_episodes(directory, manifest)


def test_versioned_terminal_exhaustion_is_not_a_sample(generated):
    directory, manifest = generated
    episode = directory / "group-0-trajectory-0"
    record = json.loads((episode / "submission.json").read_text())
    record.update(
        termination_reason="context_exhausted", context_exhaustion={"context_tokens": 12288}
    )
    (episode / "submission.json").write_text(json.dumps(record))
    path = episode / "calls.jsonl"
    calls = [json.loads(line) for line in path.read_text().splitlines()]
    calls.append(
        {
            **{
                key: record[key]
                for key in (
                    "execution_run_id",
                    "group_id",
                    "attempt_id",
                    "episode_id",
                    "policy_version",
                )
            },
            "call_id": "exhausted",
            "model_call_id": "exhausted",
            "root_turn_id": "last-turn",
            "branch": "root",
            "outcome": "context_exhausted",
            "context_exhaustion": record["context_exhaustion"],
        }
    )
    path.write_text("\n".join(json.dumps(call) for call in calls))
    manifest["episodes"][0]["sha256"].update(
        {
            name: hashlib.sha256((episode / name).read_bytes()).hexdigest()
            for name in ("submission.json", "calls.jsonl")
        }
    )
    assert len(load_episodes(directory, manifest)[0]["samples"]) == 2


def test_missing_generation_checksum_does_not_bypass_v1_validation(generated):
    directory, manifest = generated
    del manifest["episodes"][0]["sha256"]["calls.jsonl"]
    with pytest.raises(ValueError):
        load_episodes(directory, manifest)


def test_legacy_recorded_hash_and_identity_conflicts_are_not_unknown(group):
    del group["schema_version"]
    group["artifact_sha256"] = content_digest(group)
    validate_training_group(group)
    group["rows"][0]["group_id"] = "other"
    group["artifact_sha256"] = content_digest(group)
    with pytest.raises(ValueError):
        validate_training_group(group)
    group["rows"][0]["group_id"] = group["group_id"]
    with pytest.raises(ValueError):
        validate_training_group(group)


def test_legacy_generation_rejects_recorded_identity_conflict(generated):
    directory, manifest = generated
    del manifest["schema_version"]
    path = directory / "group-0-trajectory-0/calls.jsonl"
    calls = [json.loads(line) for line in path.read_text().splitlines()]
    calls[0]["episode_id"] = "other"
    path.write_text("\n".join(json.dumps(call) for call in calls))
    manifest["episodes"][0]["sha256"]["calls.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        load_episodes(directory, manifest)


def test_immediate_episode_policy_is_not_overwritten(episodes):
    episodes[0]["policy_version"] = 1
    with pytest.raises(ValueError):
        validate_episode_samples(episodes, 2)


def test_boolean_row_policy_is_not_integer_one(episodes):
    for episode in episodes:
        for sample in episode["samples"]:
            sample.provenance["policy_version"] = 1
    group = finalize_training_group(training_rows(episodes, 1))
    group["rows"][0]["policy_version"] = True
    group["artifact_sha256"] = content_digest(group)
    with pytest.raises(ValueError):
        validate_training_group(group)


@pytest.mark.parametrize("key,value", [("policy_version", 1), ("episode_id", "other")])
def test_legacy_known_row_policy_and_episode_conflicts(group, key, value):
    del group["schema_version"]
    group["rows"][0][key] = value
    group["artifact_sha256"] = content_digest(group)
    with pytest.raises(ValueError):
        validate_training_group(group)


def test_legacy_known_submission_policy_conflict(generated):
    directory, manifest = generated
    del manifest["schema_version"]
    path = directory / "group-0-trajectory-0/submission.json"
    record = json.loads(path.read_text())
    record["policy_version"] = 1
    path.write_text(json.dumps(record))
    manifest["episodes"][0]["sha256"]["submission.json"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError):
        load_episodes(directory, manifest)
