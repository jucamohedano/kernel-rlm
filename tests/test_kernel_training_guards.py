import ast
import asyncio
import json
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

ROOT = Path(__file__).parents[1] / "training/environments/kernel_rlm"


def test_actual_reward_includes_speed_for_correct_kernels():
    tree = ast.parse((ROOT / "kernel_rlm_v1/taskset.py").read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef))
    method.decorator_list = []
    method.returns = None
    for argument in method.args.args:
        argument.annotation = None
    result = {"compiled": True, "extension_used": True, "correct": True, "speedup": 0.5}
    namespace = {
        "asyncio": asyncio,
        "json": json,
        "validate_submission_source": lambda source: (True, ""),
        "verify_cuda": lambda *args: result,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), "reward-test", "exec"), namespace)
    runtime = SimpleNamespace(
        read=AsyncMock(
            return_value=json.dumps(
                {
                    "submission": "candidate",
                    "repl_calls": 1,
                    "messages": [],
                }
            )
        )
    )
    task = SimpleNamespace(data=SimpleNamespace(context="reference", speed_reward_weight=0.1))
    rewards = []
    for speed in (0.5, 1.0, 3.0, 3.0):
        result["speedup"] = speed
        trace = SimpleNamespace(info={})
        rewards.append(asyncio.run(namespace["kernel_reward"](task, trace, runtime)))
        assert trace.info["kernel_verify"]["speedup"] == speed
    assert rewards[0] < rewards[1] < rewards[2]
    assert rewards[2] == rewards[3]
    result["correct"] = False
    assert (
        asyncio.run(namespace["kernel_reward"](task, SimpleNamespace(info={}), runtime))
        < rewards[0]
    )


def test_launch_deadlines_and_no_retry():
    tree = ast.parse((ROOT / "modal_app/grpo_live_smoke.py").read_text())
    train = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    options = {
        item.arg: ast.literal_eval(item.value)
        for item in train.decorator_list[0].keywords
        if item.arg in {"timeout", "startup_timeout", "retries"}
    }
    assert options == {"timeout": 900, "startup_timeout": 120, "retries": 0}
    waits = [
        ast.literal_eval(keyword.value)
        for node in ast.walk(train)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "wait"
        for keyword in node.keywords
        if keyword.arg == "timeout"
    ]
    assert waits == [780, 30, 10]
    assert sum(waits) < options["timeout"]


@pytest.mark.parametrize("outcome", ["success", "failure", "terminate", "kill"])
def test_training_exit_preserves_report_and_commits_traces(tmp_path, outcome):
    tree = ast.parse((ROOT / "modal_app/grpo_live_smoke.py").read_text())
    train = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    train.decorator_list = []
    process = Mock(pid=123)
    timeout = subprocess.TimeoutExpired("prime", 780)
    process.wait.side_effect = {
        "success": [0],
        "failure": [1],
        "terminate": [timeout, -15],
        "kill": [timeout, timeout, -9],
    }[outcome]
    popen = Mock(return_value=process)
    killpg = Mock()
    traces = Mock()
    namespace = {
        "Path": lambda value: tmp_path / value.lstrip("/"),
        "patch_prime_laguna_config": Mock(),
        "subprocess": SimpleNamespace(
            Popen=popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
        ),
        "os": SimpleNamespace(killpg=killpg),
        "signal": signal,
        "json": json,
        "traces": traces,
        "MODEL_ID": "test-model",
        "REVISION": "test-revision",
        "PRIME_REVISION": "test-prime",
        "CONFIG": "test-config",
    }
    exec(compile(ast.Module(body=[train], type_ignores=[]), "launch-test", "exec"), namespace)
    if outcome == "success":
        namespace["train"]("test-run")
    else:
        with pytest.raises(RuntimeError):
            namespace["train"]("test-run")
    report = json.loads((tmp_path / "traces/live-grpo/test-run/report.json").read_text())
    assert (
        report["status"]
        == {
            "success": "passed",
            "failure": "failed",
            "terminate": "budget_timeout",
            "kill": "budget_timeout",
        }[outcome]
    )
    traces.commit.assert_called_once()
    assert popen.call_args.kwargs["start_new_session"] is True
    assert killpg.call_count == {"success": 0, "failure": 0, "terminate": 1, "kill": 2}[outcome]
    if outcome in {"terminate", "kill"}:
        assert killpg.call_args_list[0].args == (123, signal.SIGTERM)
    if outcome == "kill":
        assert killpg.call_args_list[1].args == (123, signal.SIGKILL)
