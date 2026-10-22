"""Enforce backend ownership without importing inactive trainer packages."""

import ast
import asyncio
import json
import os
import subprocess
import sys
from functools import wraps
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
ENVIRONMENT = ROOT / "training/environments/kernel_rlm"
BACKENDS = {"kernel_trl", "kernel_skyrl", "kernel_prime"}
TRAINERS = {"trl", "skyrl", "prime_rl"}


def imported_modules(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_backend_import_boundaries():
    for package in BACKENDS:
        for path in (ENVIRONMENT / package).rglob("*.py"):
            trainer = {"kernel_trl": "trl", "kernel_skyrl": "skyrl", "kernel_prime": "prime_rl"}[
                package
            ]
            forbidden = (BACKENDS - {package}) | (TRAINERS - {trainer})
            imports = {name.split(".")[0] for name in imported_modules(path)}
            assert not imports & forbidden, f"{path}: {imports & forbidden}"
    # Prime-RL also uses the existing Verifiers adapters and launch modules.
    prime_paths = [
        ENVIRONMENT / "kernel_rlm/env.py",
        ENVIRONMENT / "kernel_rlm_v1/taskset.py",
        *(ROOT / "training/src/rlm_train").rglob("*.py"),
        *(ENVIRONMENT / "modal_app").rglob("*.py"),
    ]
    for path in prime_paths:
        imports = {name.split(".")[0] for name in imported_modules(path)}
        assert not imports & {"kernel_trl", "kernel_skyrl", "trl", "skyrl"}, path
    for package in ("kernel_rollout", "kernel_training"):
        for path in (ENVIRONMENT / package).rglob("*.py"):
            imports = {name.split(".")[0] for name in imported_modules(path)}
            assert not imports & (BACKENDS | TRAINERS), f"{path}: trainer dependency"


def test_trl_collection_and_artifacts_import_without_inactive_backends():
    script = """
import importlib.abc
import sys

blocked = {"kernel_skyrl", "kernel_prime", "skyrl", "prime_rl", "trl"}

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ImportError(fullname)

sys.meta_path.insert(0, Block())
import kernel_trl.collect
import kernel_trl.rollout
import kernel_trl.scoring
import kernel_trl.artifacts
import kernel_trl.turn_credit
assert not blocked & set(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT), str(ROOT / "training/src"), str(ENVIRONMENT)]
            ),
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def run_async(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return run


class Client:
    def __init__(self):
        self.finished = []

    async def finish_session(self, session_id):
        self.finished.append(session_id)


@run_async
async def test_shared_runner_cancellation_kills_program_and_closes_session(tmp_path):
    from kernel_rollout.episode import EpisodeRequest, EpisodeRunner

    client = Client()
    script = (
        "import os, time; from pathlib import Path; "
        "Path('pid').write_text(str(os.getpid())); time.sleep(60)"
    )
    runner = EpisodeRunner(
        client, "adapter", "fixture", program_command=[sys.executable, "-c", script]
    )
    directory = tmp_path / "cancelled"
    job = asyncio.create_task(runner.run(EpisodeRequest({"name": "task"}, directory, 7)))
    # Let the actual program start before cancellation. Bound this CPU-only fixture.
    async with asyncio.timeout(10):
        while not (directory / "pid").exists():
            await asyncio.sleep(0.01)
    pid = int((directory / "pid").read_text())
    job.cancel()
    with pytest.raises(asyncio.CancelledError):
        await job
    assert len(client.finished) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    failure = json.loads((directory / "failure.json").read_text())
    assert failure["error_type"] == "CancelledError"
    assert failure["optimizer_update_allowed"] is False
    assert not (directory / "score.json").exists()


@run_async
@pytest.mark.parametrize("backend", ["shared", "trl", "skyrl"])
async def test_episode_adapters_preserve_artifact_provenance(tmp_path, backend):
    from kernel_rollout.episode import EpisodeRequest, EpisodeRunner

    client = Client()
    program = (
        "import json; from pathlib import Path; "
        "Path('submission.json').write_text(json.dumps({'submission': 'fixture'}))"
    )

    async def score(task, record):
        assert task["name"] == "task-a"
        assert record["submission"] == "fixture"
        return 0.8, {"correct": True, "fixture": "CPU only"}

    settings = {
        "scorer": score,
        "program_command": [sys.executable, "-c", program],
        "execution_run_id": "run-a",
        "group_id": "group-a",
        "attempt_id": "attempt-2",
    }
    task = {"name": "task-a"}
    directory = tmp_path / "episode-a"
    if backend == "shared":
        runner = EpisodeRunner(client, "adapter", "fixture", **settings)
        result = await runner.run(EpisodeRequest(task, directory, 9))
    else:
        if backend == "trl":
            from kernel_trl.rollout import RolloutCollector

            adapter = RolloutCollector(client, "adapter", "fixture", **settings)
        else:
            from kernel_skyrl.generator import KernelGenerator

            adapter = KernelGenerator(client, "adapter", "fixture", tmp_path, **settings)
        result = await adapter.run_episode(task, directory, 9)
    expected = {
        "execution_run_id": "run-a",
        "group_id": "group-a:task-a",
        "attempt_id": "attempt-2",
        "episode_id": "episode-a",
        "policy_version": 9,
    }
    assert all(result[key] == value for key, value in expected.items())
    record = json.loads((directory / "submission.json").read_text())
    assert all(record[key] == value for key, value in expected.items())
    assert result["samples"] == []
    assert result["reward"] == 0.8
    assert json.loads((directory / "score.json").read_text())["root_calls"] == 0
    assert len(client.finished) == 1
