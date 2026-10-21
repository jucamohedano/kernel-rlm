import ast
import copy
import importlib.util
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ENVIRONMENT = Path(__file__).parents[1] / "training/environments/kernel_rlm"
SPEC = importlib.util.spec_from_file_location(
    "factory_test_module", ENVIRONMENT / "kernel_factory/__init__.py"
)
factory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(factory)
PINS = {"model_id": "test-model", "model_revision": "a" * 40, "prime_revision": "b" * 40}


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    environment = root / "rlm/training/environments/kernel_rlm"
    for relative in ("notes", "rlm/rlm", "rlm/training/src/rlm_train"):
        (root / relative).mkdir(parents=True)
    for relative in (
        "dashboard",
        "configs",
        "modal_app",
        "kernel_rlm",
        "kernel_rlm_v1",
        "kernel_factory",
    ):
        (environment / relative).mkdir(parents=True)
    (root / ".gitignore").write_text("rlm/\n")
    (root / "rlm/.gitignore").write_text("training/environments/kernel_rlm/logs/\n")
    (root / "notes/tasks.jsonl").write_text(
        json.dumps({"_task_id": "task-1", "code": "reference"}) + "\n"
    )
    (root / "notes/decision.md").write_text("Why this experiment exists")
    (environment / "kernel_rlm/source.py").write_text("VALUE = 1\n")
    (environment / "modal_app/launcher.py").write_text("print('launch')\n")
    config = (
        '[model]\nname="test-model"\n[[orchestrator.train.source]]\n'
        'env.taskset.manifest_path="/root/tasks.jsonl"\n'
    )
    (environment / "configs/grpo_pilot.toml").write_text(config)
    (environment / "configs/grpo_level1.toml").write_text(config)
    registry = {
        "datasets": {
            "training": {
                "path": "notes/tasks.jsonl",
                "remote_path": "/root/tasks.jsonl",
                "upstream_revision": None,
            }
        },
        "experiments": [
            {
                "id": "review",
                "intent": "test",
                "change": "test",
                "decision": "test",
                "provenance": {
                    "code_paths": ["rlm/training/environments/kernel_rlm/kernel_rlm/source.py"],
                    "notes": ["notes/decision.md"],
                    "dataset_ids": ["training"],
                },
            }
        ],
    }
    (environment / "dashboard/experiments.json").write_text(json.dumps(registry))
    for repository in (root, root / "rlm"):
        subprocess.run(["git", "init", "-q", str(repository)], check=True)
        subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )
    return root, environment


def prepare(project, **kwargs):
    root, environment = project
    return factory.prepare_run(
        root, environment, "review", "cpu-test", pilot=True, pins=PINS, **kwargs
    )


def mounted_bundle(project, manifest, tmp_path, monkeypatch):
    bundle = factory.launch_bundle(manifest, project[1])
    mount = tmp_path / "mounted"
    for item in manifest["inputs"]:
        if item["remote_path"]:
            path = mount / item["remote_path"].removeprefix("/root/")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(bundle["input_snapshots"][item["path"]])
    monkeypatch.setattr(
        factory,
        "Path",
        lambda value: mount / value.removeprefix("/root/")
        if isinstance(value, str) and value.startswith("/root/")
        else Path(value),
    )
    return bundle, mount


def test_preparation_attests_inputs_commits_and_checkpoint_parent(project, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "credential-must-not-be-recorded")
    manifest = prepare(project, resume_run="older-run", resume_step=2)
    assert manifest["capture_kind"] == "before_launch"
    assert all(
        len(state["commit"]) == 40 and not state["dirty"]
        for state in manifest["repositories"].values()
    )
    assert manifest["datasets"]["training"]["task_count"] == 1
    assert manifest["datasets"]["training"]["upstream_revision"] is None
    assert manifest["parent"]["checkpoint_step"] == 2
    assert "credential-must-not-be-recorded" not in json.dumps(manifest)
    directory = project[1] / "logs/live-grpo/cpu-test"
    for item in manifest["inputs"]:
        assert factory.digest((directory / "inputs" / item["path"]).read_bytes()) == item["sha256"]
    original = (directory / "provenance.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare(project)
    assert (directory / "provenance.json").read_bytes() == original


def test_dirty_launch_requires_explicit_override(project):
    (project[1] / "kernel_rlm/source.py").write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="Commit both repositories"):
        prepare(project)
    manifest = prepare(project, allow_dirty=True)
    assert manifest["repositories"]["rlm"]["dirty"] is True
    assert manifest["dirty_launch_explicitly_allowed"] is True


def test_tampered_prepared_snapshot_is_rejected(project):
    manifest = prepare(project)
    item = manifest["inputs"][0]
    path = project[1] / "logs/live-grpo/cpu-test/inputs" / item["path"]
    path.write_text("tampered")
    with pytest.raises(ValueError, match="Prepared input changed"):
        factory.launch_bundle(manifest, project[1])


def test_remote_start_preserves_snapshots_and_records_actual_runtime(
    project, tmp_path, monkeypatch
):
    manifest = prepare(project)
    bundle, _mount = mounted_bundle(project, manifest, tmp_path, monkeypatch)
    directory = tmp_path / "remote"
    directory.mkdir()
    factory.start_run(bundle, directory, "cpu-test")
    assert json.loads((directory / "runtime.json").read_text())["input_validation"] == "passed"
    assert "input_snapshots" not in json.loads((directory / "provenance.json").read_text())
    assert len(list((directory / "inputs").rglob("*.*"))) == len(manifest["inputs"])
    assert json.loads((directory / "events.jsonl").read_text())["event"] == "started"
    with pytest.raises(FileExistsError):
        factory.start_run(bundle, directory, "cpu-test")


def test_remote_mounted_input_mismatch_is_audited(project, tmp_path, monkeypatch):
    manifest = prepare(project)
    bundle, mount = mounted_bundle(project, manifest, tmp_path, monkeypatch)
    (mount / "kernel_rlm/source.py").write_text("VALUE = 'stale image'\n")
    directory = tmp_path / "remote"
    directory.mkdir()
    with pytest.raises(ValueError, match="Mounted input differs"):
        factory.start_run(bundle, directory, "cpu-test")
    assert (
        json.loads((directory / "events.jsonl").read_text())["event"] == "input_validation_failed"
    )


def test_bad_manifest_and_wrong_run_id_fail_before_start(project, tmp_path):
    manifest = prepare(project)
    directory = tmp_path / "remote"
    directory.mkdir()
    with pytest.raises(ValueError, match="matching factory manifest"):
        factory.start_run(None, directory, "cpu-test")
    with pytest.raises(ValueError, match="matching factory manifest"):
        factory.start_run(manifest, directory, "another-run")
    altered = copy.deepcopy(manifest)
    altered["experiment_id"] = "changed"
    with pytest.raises(ValueError, match="integrity"):
        factory.start_run(altered, directory, "cpu-test")


def test_registry_rejects_unknown_dataset_and_missing_notes(project):
    root, environment = project
    path = environment / "dashboard/experiments.json"
    registry = json.loads(path.read_text())
    registry["experiments"][0]["provenance"]["dataset_ids"] = ["unknown"]
    path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="Unknown dataset"):
        factory.validate_registry(root, environment)
    registry["experiments"][0]["provenance"]["dataset_ids"] = ["training"]
    registry["experiments"][0]["provenance"]["notes"] = ["notes/missing.md"]
    path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="Required asset is missing"):
        factory.validate_registry(root, environment)


def test_paths_and_duplicate_task_ids_are_rejected(project):
    root, environment = project
    with pytest.raises(ValueError, match="escapes"):
        factory.checked_path(root, "../outside.md")
    dataset = root / "notes/tasks.jsonl"
    dataset.write_text(dataset.read_text() * 2)
    with pytest.raises(ValueError, match="unique"):
        factory.validate_registry(root, environment)


def test_checked_in_example_registry_and_dataset_validate():
    root = Path(__file__).parents[1]
    result = factory.validate_registry(root, root / "examples/factory_cpu")
    assert result["status"] == "passed"
    assert result["datasets"]["vector-add-cpu-contract"]["task_count"] == 1


def test_missing_provenance_still_fails_closed(project):
    root, environment = project
    path = environment / "dashboard/experiments.json"
    registry = json.loads(path.read_text())
    registry["experiments"][0]["provenance"]["code_paths"] = []
    path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="Incomplete experiment provenance"):
        factory.validate_registry(root, environment)


def test_remote_snapshot_payload_cannot_bypass_hashes(project, tmp_path, monkeypatch):
    manifest = prepare(project)
    bundle, _mount = mounted_bundle(project, manifest, tmp_path, monkeypatch)
    bundle["input_snapshots"][manifest["inputs"][0]["path"]] = "altered payload"
    directory = tmp_path / "remote"
    directory.mkdir()
    with pytest.raises(ValueError, match="Snapshot integrity"):
        factory.start_run(bundle, directory, "cpu-test")


def test_unregistered_experiment_and_model_mismatch_fail_before_capture(project):
    root, environment = project
    with pytest.raises(ValueError, match="full commit hashes"):
        factory.prepare_run(
            root, environment, "review", "test", pins={**PINS, "model_revision": "main"}
        )
    with pytest.raises(ValueError, match="registered experiment"):
        factory.prepare_run(root, environment, "missing", "test", pins=PINS)
    with pytest.raises(ValueError, match="Model pin"):
        factory.prepare_run(root, environment, "review", "test", pins={**PINS, "model_id": "wrong"})


def test_dependency_checkout_must_match_pin(project):
    root, environment = project
    with pytest.raises(ValueError, match="Prime source checkout"):
        factory.prepare_run(
            root,
            environment,
            "review",
            "test",
            pins=PINS,
            dependency_repositories={"prime": root / "rlm"},
        )


@pytest.mark.parametrize("failure", [None, "unregistered experiment", "dirty repository"])
def test_entrypoint_prepares_before_any_remote_dispatch(tmp_path, failure):
    tree = ast.parse((ENVIRONMENT / "modal_app/grpo_live_smoke.py").read_text())
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    main.decorator_list = []
    prepare_mock = Mock(return_value={"run_id": "prepared"})
    if failure:
        prepare_mock.side_effect = ValueError(failure)
    remote = Mock(return_value={"status": "fake"})
    namespace = {
        "prepare_run": prepare_mock,
        "launch_bundle": Mock(return_value={"bundle": True}),
        "event": Mock(),
        "datetime": datetime,
        "UTC": UTC,
        "uuid4": lambda: SimpleNamespace(hex="12345678"),
        "json": json,
        "PROJECT_ROOT": tmp_path,
        "APP_ROOT": tmp_path,
        "MODEL_ID": "model",
        "REVISION": "revision",
        "PRIME_REVISION": "prime",
        "PRIME_SOURCE": tmp_path / "prime",
        "dry_run": SimpleNamespace(remote=remote),
        "train": SimpleNamespace(remote=remote),
        "train_pilot": SimpleNamespace(remote=remote),
    }
    exec(compile(ast.Module(body=[main], type_ignores=[]), "entrypoint", "exec"), namespace)
    if failure:
        with pytest.raises(ValueError, match=failure):
            namespace["main"](pilot=True, experiment="review")
        remote.assert_not_called()
    else:
        namespace["main"](pilot=True, experiment="review")
        remote.assert_called_once()
        assert remote.call_args.args[-1] == {"bundle": True}
        assert prepare_mock.call_args.args[2] == "review"


def test_remote_input_failure_commits_evidence_without_starting_trainer(tmp_path):
    tree = ast.parse((ENVIRONMENT / "modal_app/grpo_live_smoke.py").read_text())
    train = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_train"
    )
    traces = Mock()
    namespace = {
        "Path": lambda value: tmp_path / value.lstrip("/"),
        "os": SimpleNamespace(environ={}),
        "start_run": Mock(side_effect=ValueError("input mismatch")),
        "traces": traces,
    }
    exec(compile(ast.Module(body=[train], type_ignores=[]), "launch-gate", "exec"), namespace)
    with pytest.raises(ValueError, match="input mismatch"):
        namespace["_train"]("run", True, 1200, 1380)
    traces.commit.assert_called_once()
