import asyncio
import hashlib
import importlib
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1] / "training/environments/kernel_rlm"


@pytest.fixture
def curriculum(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    return importlib.import_module("kernel_trl.live_curriculum")


def test_curriculum_rotates_after_bootstrap_without_dropping_tasks(
    curriculum, monkeypatch, tmp_path
):
    task_ids = [f"task-{index}" for index in range(50)]
    monkeypatch.setattr(
        curriculum, "load_tasks", lambda dataset: [{"name": name} for name in task_ids]
    )

    ordered = curriculum.curriculum_order(tmp_path / "dataset.jsonl", "task-17")

    assert ordered == task_ids[17:] + task_ids[:17]
    assert len(ordered) == 50
    assert len(set(ordered)) == 50


@pytest.fixture
def continuation_parent(curriculum, tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("unchanged dataset")
    order = [f"task-{index}" for index in range(50)]
    manifest = {
        "source_commit": "parent-source",
        "model_revision": curriculum.MODEL_REVISION,
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
    }
    state = {
        "run_id": "parent-run",
        "status": "needs_diagnosis",
        "policy_version": 4,
        "completed_task_ids": order[:4],
        "task_order": order,
        "checkpoint": str(parent / "groups/step-004/update/checkpoint-step-4"),
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        "results": [
            {
                "status": "trained",
                "task_id": task,
                "optimizer_step": index + 1,
                "policy_version": index,
                "rewards": [0.8, 0.1],
            }
            for index, task in enumerate(order[:4])
        ]
        + [{"status": "needs_diagnosis", "task_id": order[4], "optimizer_step": None}],
        "attempts": {order[4]: 1},
        "failure": {"task_id": order[4], "reason": "weak_signal"},
    }
    curriculum.write_json(parent / "inputs.json", manifest)
    curriculum.write_json(parent / "state.json", state)
    arguments = SimpleNamespace(parent_run_dir=parent, bootstrap_dir=None, dataset=dataset)
    return arguments, state, order


@pytest.mark.parametrize("fault", [None, "coverage", "checkpoint", "completed_group"])
def test_explicit_incomplete_run_preservation_does_not_reset_coverage(
    curriculum, continuation_parent, tmp_path, fault
):
    import copy

    arguments, original, order = continuation_parent
    directory = tmp_path / "incomplete"
    pending = directory / "groups/step-005"
    (pending / "collection/rollouts").mkdir(parents=True)
    (pending / "collection/rollouts/task.json").write_text("preserved sample")
    manifest = curriculum.read_json(arguments.parent_run_dir / "inputs.json")
    manifest["source_commit"] = "failed-source"
    stopped = copy.deepcopy(original)
    stopped.update(
        status="failed",
        pending_task_id=order[4],
        pending_directory=str(pending),
        attempts={order[4]: 2},
        continuation_parent={"run_id": "parent-run"},
        manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
    )
    state = copy.deepcopy(original)
    state["continuation_parent"] = {"run_id": "parent-run"}
    if fault == "coverage":
        stopped["completed_task_ids"] = order[:3]
    if fault == "checkpoint":
        stopped["checkpoint"] = "different-checkpoint"
    if fault == "completed_group":
        (pending / "collection/rollouts/generation-manifest.json").write_text("{}")
    curriculum.write_json(directory / "inputs.json", manifest)
    curriculum.write_json(directory / "state.json", stopped)
    before = (directory / "state.json").read_bytes()
    arguments.preserved_incomplete_run = directory
    if fault:
        with pytest.raises(ValueError):
            curriculum.preserve_incomplete_run(arguments, state)
        return
    curriculum.preserve_incomplete_run(arguments, state)
    assert state["completed_task_ids"] == order[:4]
    assert state["policy_version"] == 4
    assert state["attempts"][order[4]] == 2
    assert state["results"][-1]["status"] == "incomplete_preserved"
    assert state["preserved_incomplete_runs"][0]["preserved_files_sha256"]
    assert (directory / "state.json").read_bytes() == before


def test_new_snapshot_continuation_keeps_all_coverage_and_rejected_history(
    curriculum, continuation_parent, monkeypatch
):
    arguments, original, order = continuation_parent
    before = (arguments.parent_run_dir / "state.json").read_bytes()
    validated = []

    def validate(selected, task_order):
        validated.append(selected)
        return {
            "policy_version": 4,
            "completed_task_ids": order[3:4],
            "results": [{"rewards": [0.8, 0.1]}],
        }

    monkeypatch.setattr(curriculum, "bootstrap_state", validate)
    result = curriculum.continuation_state(arguments, order)

    assert result["completed_task_ids"] == order[:4]
    assert result["results"] == original["results"]
    assert result["attempts"] == {order[4]: 1}
    assert result["continuation_parent"]["failure_preserved"] == original["failure"]
    assert result["continuation_parent"]["rejected_group_trained"] is False
    assert validated[0].parent_run_dir is None
    assert validated[0].bootstrap_dir == Path(original["checkpoint"]).parent.parent
    assert result["status"] == "ready"
    assert (arguments.parent_run_dir / "state.json").read_bytes() == before


@pytest.mark.parametrize("fault", ["active", "unfinished", "coverage", "manifest", "dataset"])
def test_new_snapshot_continuation_rejects_inconsistent_parent(
    curriculum, continuation_parent, fault
):
    arguments, state, order = continuation_parent
    if fault == "active":
        state["status"] = "running"
    elif fault == "unfinished":
        state["pending_task_id"] = order[4]
    elif fault == "coverage":
        state["completed_task_ids"] = order[:1]
    elif fault == "manifest":
        state["manifest_sha256"] = "changed"
    else:
        arguments.dataset.write_text("changed dataset")
    curriculum.write_json(arguments.parent_run_dir / "state.json", state)
    with pytest.raises(ValueError):
        curriculum.continuation_state(arguments, order)


def rejected_group(task_id, policy_version=5, reason="weak_signal", rewards=None):
    return {
        "task_id": task_id,
        "status": "needs_diagnosis",
        "policy_version": policy_version,
        "optimizer_step": None,
        "trace_directory": f"preserved/{task_id}",
        "group_record": {
            "task": task_id,
            "policy_version": policy_version,
            "reason": reason,
            "rewards": [0.1] * 8 if rewards is None else rewards,
            "admitted": False,
            "infrastructure_errors": [],
        },
    }


@pytest.mark.parametrize(
    "reason,rewards", [("weak_signal", [0, 0.1] * 4), ("zero_variance", [0.8] * 8)]
)
def test_deferral_preserves_failed_rewards_without_advancing_policy(curriculum, reason, rewards):
    state = {"task_order": ["pending"], "completed_task_ids": [], "policy_version": 5}
    result = rejected_group("pending", reason=reason, rewards=rewards)
    original = json.dumps(result, sort_keys=True)

    assert curriculum.defer_uninformative_group(state, result)
    assert state["deferred_task_ids"] == ["pending"]
    assert state["completed_task_ids"] == []
    assert state["policy_version"] == 5
    assert state["deferred_groups"][0]["rewards"] == rewards
    assert state["deferred_groups"][0]["optimizer_update"] is False
    assert json.dumps(result, sort_keys=True) == original
    assert curriculum.defer_uninformative_group(state, result)
    assert len(state["deferred_groups"]) == 1


@pytest.mark.parametrize(
    "fault", ["infrastructure", "audit", "admitted", "signal", "nonfinite", "incomplete"]
)
def test_deferral_never_bypasses_safety_failures_or_useful_signal(curriculum, fault):
    state = {"task_order": ["pending"], "completed_task_ids": [], "policy_version": 5}
    result = rejected_group("pending")
    group = result["group_record"]
    if fault == "infrastructure":
        group["infrastructure_errors"] = [{"verification_status": "timeout"}]
    elif fault == "audit":
        result["status"] = "failed"
    elif fault == "admitted":
        group["admitted"] = True
    elif fault == "signal":
        group["rewards"] = [0.1, 0.8] * 4
    elif fault == "nonfinite":
        group["rewards"][0] = float("nan")
    else:
        group["rewards"] = [0.1] * 7
    assert not curriculum.defer_uninformative_group(state, result)
    assert not state.get("deferred_task_ids")


@pytest.mark.parametrize("handling", ["continue", "stop", "infrastructure"])
def test_curriculum_filters_weak_groups_and_caps_successful_updates(
    curriculum, tmp_path, monkeypatch, handling
):
    order = [f"task-{index}" for index in range(50)]
    state = {
        "status": "ready",
        "run_id": "test",
        "target_task_count": 50,
        "task_order": order,
        "completed_task_ids": order[:5],
        "policy_version": 5,
        "results": [],
        "checkpoint": "preserved-step-5",
    }
    budget_path = tmp_path / "budget.json"
    budget_path.write_text(
        json.dumps(
            {
                "budget_usd": 60,
                "displayed_pair_rate_usd_per_hour": 1.26,
                "goal_started_unix_seconds": time.time() - 60,
            }
        )
    )
    arguments = SimpleNamespace(
        output_dir=tmp_path / "run",
        run_id="test",
        python=sys.executable,
        source_root=tmp_path,
        cuda_arch_list="8.9",
        resume=False,
        budget_file=budget_path,
        max_groups=None,
        max_updates=3,
        continue_weak_groups=handling != "stop",
    )
    calls = []

    async def train(arguments, budget, current, task_id):
        calls.append(task_id)
        if task_id == order[5]:
            result = rejected_group(task_id)
            if handling == "infrastructure":
                result["group_record"]["reason"] = "infrastructure_error"
                result["group_record"]["infrastructure_errors"] = [
                    {"verification_status": "timeout"}
                ]
            current.update(status="needs_diagnosis", failure=result)
            current["results"].append(result)
            return False
        current["policy_version"] += 1
        current["completed_task_ids"].append(task_id)
        current["results"].append({"task_id": task_id, "status": "trained"})
        return True

    monkeypatch.setattr(curriculum, "training_environment", lambda *args: {})
    monkeypatch.setattr(curriculum, "initialize_run", lambda *args: state)
    monkeypatch.setattr(curriculum, "train_task", train)
    report = asyncio.run(curriculum.run(arguments))
    if handling == "continue":
        assert calls == order[5:9]
        assert report["optimizer_steps"] == 8
        assert report["completed_task_count"] == 8
        assert report["deferred_task_ids"] == [order[5]]
        assert order[5] not in report["completed_task_ids"]
        assert report["status"] == "paused"
    else:
        assert calls == [order[5]]
        assert report["optimizer_steps"] == 5
        assert not report["deferred_task_ids"]
        assert report["failure"]["task_id"] == order[5]


def test_server_serves_the_base_or_exact_saved_lora(curriculum, tmp_path):
    arguments = SimpleNamespace(
        inference_python="/venv/inference-python",
        checkpoint=tmp_path / "base",
    )

    base = curriculum.server_command(arguments, 0, None)
    trained = curriculum.server_command(arguments, 1, tmp_path / "checkpoint-step-1")

    assert "--lora-modules" not in base
    assert trained[trained.index("--lora-modules") + 1] == (
        f"kernel-policy-1={tmp_path / 'checkpoint-step-1'}"
    )


def test_inference_environment_exposes_venv_binaries(curriculum, tmp_path):
    system_python = tmp_path / "system/bin/python"
    system_python.parent.mkdir(parents=True)
    system_python.touch()
    system_python.chmod(0o755)
    python = tmp_path / "inference-venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(system_python)
    ninja = python.parent / "ninja"
    ninja.touch()
    ninja.chmod(0o755)
    runtime = python.parents[2]
    python_include = runtime / "python-headers/usr/include/python3.12"
    multiarch_include = runtime / "python-headers/usr/include/x86_64-linux-gnu/python3.12"
    python_include.mkdir(parents=True)
    multiarch_include.mkdir(parents=True)
    (python_include / "Python.h").touch()
    (multiarch_include / "pyconfig.h").touch()
    result = curriculum.inference_environment(str(python), {"PATH": "/usr/bin"})

    assert shutil.which("ninja", path=result["PATH"]) == str(ninja)
    assert result["VLLM_SERVER_DEV_MODE"] == "1"
    assert result["CPATH"].split(":")[:2] == [
        str(python_include),
        str(runtime / "python-headers/usr/include"),
    ]


def test_runtime_environment_exposes_repl_compiler_tools(curriculum, tmp_path):
    python = tmp_path / "training-venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    python.chmod(0o755)
    ninja = python.parent / "ninja"
    ninja.touch()
    ninja.chmod(0o755)

    result = curriculum.runtime_environment(str(python), {"PATH": "/usr/bin"})

    assert shutil.which("ninja", path=result["PATH"]) == str(ninja)


def test_python_path_and_source_hashes_include_rlm_core(curriculum, tmp_path):
    source_root = tmp_path / "repo/training/environments/kernel_rlm"
    (source_root / "kernel_trl").mkdir(parents=True)
    (source_root / "kernel_trl/__init__.py").write_text("")
    core = tmp_path / "repo/rlm"
    core.mkdir(parents=True)
    (core / "__init__.py").write_text("core")
    training_package = tmp_path / "repo/training/src/rlm_train"
    training_package.mkdir(parents=True)
    (training_package / "__init__.py").write_text("worker")
    configured = curriculum.python_path(source_root, "/existing/path")
    hashes = curriculum.source_hashes(source_root)

    assert configured.split(":") == [
        str(source_root),
        str(tmp_path / "repo/training/src"),
        str(tmp_path / "repo"),
        "/existing/path",
    ]
    assert hashes["kernel_trl/__init__.py"]
    assert hashes["rlm/__init__.py"]
    assert hashes["training/src/rlm_train/__init__.py"]


def test_training_environment_sets_arch_before_gpu_hidden_compile(curriculum, tmp_path):
    python = tmp_path / "training-venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    source_root = tmp_path / "repo/training/environments/kernel_rlm"
    (source_root / "kernel_trl").mkdir(parents=True)
    (source_root.parent.parent.parent / "rlm").mkdir(parents=True)
    (source_root.parent.parent / "src/rlm_train").mkdir(parents=True)
    header_root = tmp_path / "python-headers/usr/include"
    (header_root / "python3.12").mkdir(parents=True)
    (header_root / "x86_64-linux-gnu/python3.12").mkdir(parents=True)
    (header_root / "python3.12/Python.h").touch()
    (header_root / "x86_64-linux-gnu/python3.12/pyconfig.h").touch()

    environment = curriculum.training_environment(
        str(python), source_root, "8.9", {"PATH": "/usr/bin"}
    )

    assert environment["TORCH_CUDA_ARCH_LIST"] == "8.9"
    assert str(python.parent) in environment["PATH"]
    assert str(source_root.parent.parent.parent) in environment["PYTHONPATH"]
    assert str(header_root / "python3.12") in environment["CPATH"]


def test_collection_record_requires_one_group(curriculum, tmp_path):
    output = tmp_path / "rollouts"
    output.mkdir()
    assert curriculum.collection_record(output) is None
    (output / "groups.json").write_text(
        json.dumps([{"admitted": False, "reason": "zero_variance"}])
    )
    assert curriculum.collection_record(output)["reason"] == "zero_variance"
    (output / "groups.json").write_text(json.dumps([{}, {}]))
    with pytest.raises(ValueError, match="one group"):
        curriculum.collection_record(output)


def test_exact_group_audit_passes_and_stale_hash_is_rejected(curriculum, tmp_path):
    group = {
        "policy_version": 1,
        "trainable_tokens": 3,
        "rows": [{"completion_ids": [4, 5, 6]}],
    }
    report = {
        "probability_audit": {
            "group_sha256": hashlib.sha256(
                json.dumps(group, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest(),
            "policy_version": 1,
            "tokens": 3,
            "mean_absolute_log_ratio": 0.02,
            "p95_absolute_log_ratio": 0.05,
            "outside_ratio_0_8_to_1_2_fraction": 0.01,
            "optimizer_steps": 0,
        }
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(report))

    assert curriculum.audit_passed(path, group)["tokens"] == 3
    report["probability_audit"]["group_sha256"] = "stale"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="stale"):
        curriculum.audit_passed(path, group)


def test_bootstrap_requires_a_real_completed_update(curriculum, monkeypatch, tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    monkeypatch.setattr(
        curriculum,
        "curriculum_order",
        lambda _, first_task=None: [first_task, "task-next"],
    )
    bootstrap = tmp_path / "bootstrap"
    (bootstrap / "input").mkdir(parents=True)
    checkpoint = bootstrap / "update/checkpoint-step-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "metadata.json").write_text(
        json.dumps({"step": 1, "world_size": 2, "frozen_weights_saved": False})
    )
    (bootstrap / "input/training-group.json").write_text(
        json.dumps(
            {
                "task": "task-first",
                "policy_version": 0,
                "rewards": [0.0, 1.0],
                "trainable_tokens": 2,
                "rows": [{"completion_ids": [3, 4]}],
            }
        )
    )
    for rank in range(2):
        (bootstrap / f"update/report-rank-{rank}.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "global_step": 1,
                    "sampled_policy_version": 0,
                }
            )
        )
    audit = bootstrap / "audit"
    audit.mkdir()
    digest = hashlib.sha256(
        json.dumps(
            json.loads((bootstrap / "input/training-group.json").read_text()),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    for rank in range(2):
        (audit / f"report-rank-{rank}.json").write_text(
            json.dumps(
                {
                    "status": "audited",
                    "probability_audit": {
                        "group_sha256": digest,
                        "policy_version": 0,
                        "tokens": 2,
                        "mean_absolute_log_ratio": 0.02,
                        "p95_absolute_log_ratio": 0.05,
                        "outside_ratio_0_8_to_1_2_fraction": 0.01,
                        "optimizer_steps": 0,
                    },
                }
            )
        )

    state = curriculum.bootstrap_state(
        SimpleNamespace(bootstrap_dir=bootstrap, dataset=dataset), ["task-first", "task-next"]
    )

    assert state["policy_version"] == 1
    assert state["checkpoint"] == str(checkpoint)
    assert state["task_order"] == ["task-first", "task-next"]
    assert state["completed_task_ids"] == ["task-first"]


@pytest.mark.parametrize("mode", ["inline", "deferred", "resume_generated"])
def test_resumed_task_collects_audits_and_advances_one_policy_version(
    curriculum, monkeypatch, tmp_path, mode
):
    monkeypatch.syspath_prepend(str(ROOT))
    credit = importlib.import_module("kernel_trl.probability_audit")
    bootstrap = tmp_path / "bootstrap"
    (bootstrap / "input").mkdir(parents=True)
    (bootstrap / "input/training-group.json").write_text(
        json.dumps(
            {
                "task": "task-first",
                "policy_version": 0,
                "rewards": [0.0, 1.0],
                "trainable_tokens": 2,
                "rows": [{"completion_ids": [3, 4]}],
            }
        )
    )
    checkpoint = bootstrap / "update/checkpoint-step-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "metadata.json").write_text(
        json.dumps({"step": 1, "world_size": 2, "frozen_weights_saved": False})
    )
    for rank in range(2):
        (bootstrap / f"update/report-rank-{rank}.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "global_step": 1,
                    "sampled_policy_version": 0,
                }
            )
        )
    audit_directory = bootstrap / "audit"
    audit_directory.mkdir()
    bootstrap_group = json.loads((bootstrap / "input/training-group.json").read_text())
    audit_summary = {
        "group_sha256": credit.group_digest(bootstrap_group),
        "policy_version": 0,
        "tokens": 2,
        "mean_absolute_log_ratio": 0.02,
        "p95_absolute_log_ratio": 0.05,
        "outside_ratio_0_8_to_1_2_fraction": 0.01,
        "optimizer_steps": 0,
    }
    for rank in range(2):
        (audit_directory / f"report-rank-{rank}.json").write_text(
            json.dumps({"status": "audited", "probability_audit": audit_summary})
        )
    monkeypatch.setattr(
        curriculum, "curriculum_order", lambda dataset, first_task=None: ["task-first", "task-next"]
    )
    arguments = SimpleNamespace(
        bootstrap_dir=bootstrap,
        dataset=tmp_path / "dataset.jsonl",
        output_dir=tmp_path / "full-run",
        run_id="20261019T000000Z-test",
        python=sys.executable,
        checkpoint=tmp_path / "model",
    )
    state = curriculum.bootstrap_state(arguments, ["task-first", "task-next"])
    group = {
        "task": "task-next",
        "policy_version": 1,
        "rows": [{"prompt_ids": [2], "completion_ids": [4], "padding": False}],
        "rewards": [0.1, 0.8],
        "advantages": [-1.0, 1.0],
        "real_rows": 2,
        "trainable_tokens": 2,
    }
    arguments.output_dir.mkdir()
    phases = []

    async def collect(*args):
        phases.append("collect")
        step_directory = args[2]
        output = step_directory / "collection/rollouts"
        output.mkdir(parents=True)
        if mode == "deferred":
            (output / "generation-manifest.json").write_text("{}")
            return {"status": "generated"}
        (output / "training-group.json").write_text(json.dumps(group))
        (output / "groups.json").write_text(
            json.dumps([{"task": "task-next", "admitted": True, "rewards": group["rewards"]}])
        )
        return {"status": "admitted", "task": "task-next", "admitted": True}

    async def run_phase(arguments, budget, command, log_path, telemetry_path, timeout):
        if "kernel_trl.scoring" in command:
            phases.append("score")
            output = Path(command[command.index("--directory") + 1])
            (output / "training-group.json").write_text(json.dumps(group))
            (output / "groups.json").write_text(
                json.dumps([{"task": "task-next", "admitted": True, "rewards": group["rewards"]}])
            )
            return
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True)
        if "--audit-only" in command:
            phases.append("audit")
            summary = {
                "group_sha256": credit.group_digest(group),
                "policy_version": 1,
                "tokens": 2,
                "mean_absolute_log_ratio": 0.02,
                "p95_absolute_log_ratio": 0.05,
                "outside_ratio_0_8_to_1_2_fraction": 0.01,
                "optimizer_steps": 0,
            }
            for rank in range(2):
                (output / f"report-rank-{rank}.json").write_text(
                    json.dumps({"status": "audited", "probability_audit": summary})
                )
            return
        phases.append("update")
        checkpoint_output = output / "checkpoint-step-2"
        checkpoint_output.mkdir()
        weights = checkpoint_output / "adapter.safetensors"
        weights.write_bytes(b"weights")
        (checkpoint_output / "metadata.json").write_text(
            json.dumps(
                {
                    "step": 2,
                    "world_size": 2,
                    "frozen_weights_saved": False,
                    "sha256": {weights.name: hashlib.sha256(weights.read_bytes()).hexdigest()},
                }
            )
        )
        for rank in range(2):
            (output / f"report-rank-{rank}.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "global_step": 2,
                        "sampled_policy_version": 1,
                        "quality_improvement_claimed": False,
                    }
                )
            )

    monkeypatch.setattr(curriculum, "collect_task", collect)
    monkeypatch.setattr(curriculum, "run_torch_phase", run_phase)
    monkeypatch.setattr(curriculum, "phase_deadline", lambda *args, **kwargs: 1e20)
    if mode == "resume_generated":
        preserved = arguments.output_dir / "groups/preserved"
        output = preserved / "collection/rollouts"
        output.mkdir(parents=True)
        (output / "generation-manifest.json").write_text("{}")
        state.update(pending_task_id="task-next", pending_directory=str(preserved))

    completed = asyncio.run(
        curriculum.train_task(arguments, {"budget_usd": 40}, state, "task-next")
    )

    assert completed is True
    assert state["policy_version"] == 2
    assert state["completed_task_ids"] == ["task-first", "task-next"]
    assert Path(state["checkpoint"]).name == "checkpoint-step-2"
    assert phases == (["collect"] if mode != "resume_generated" else []) + (
        ["score"] if mode != "inline" else []
    ) + ["audit", "update"]
