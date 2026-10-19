import asyncio
import hashlib
import importlib
import json
import shutil
import sys
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
