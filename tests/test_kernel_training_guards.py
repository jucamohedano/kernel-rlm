import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

ROOT = Path(__file__).parents[1] / "training/environments/kernel_rlm"


def test_actual_reward_ignores_speed_when_configured():
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
    task = SimpleNamespace(data=SimpleNamespace(context="reference", speed_reward_weight=0.0))
    rewards = []
    for speed in (0.5, 1.0, 3.0):
        result["speedup"] = speed
        trace = SimpleNamespace(info={})
        rewards.append(asyncio.run(namespace["kernel_reward"](task, trace, runtime)))
        assert trace.info["kernel_verify"]["speedup"] == speed
    assert max(rewards) - min(rewards) == 0
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
