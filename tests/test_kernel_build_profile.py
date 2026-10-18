import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1] / "training/environments/kernel_rlm"
SPEC = importlib.util.spec_from_file_location("build_profile", ROOT / "kernel_rlm/build_profile.py")
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


@pytest.mark.parametrize(
    "tool,arguments,kind",
    [
        ("cxx", ["-c", "main.cpp", '-DMESSAGE="hello world"'], "cpp_compile"),
        ("nvcc", ["-c", "cuda.cu"], "cuda_compile"),
        ("cxx", ["-shared", "main.o"], "link"),
        ("cxx", ["-v"], "probe"),
    ],
)
def test_wrapper_preserves_arguments_exit_status_and_records_resources(
    tmp_path, tool, arguments, kind
):
    wrapper = tmp_path / tool
    driver = [sys.executable, "-c", "import json,sys; print(json.dumps(sys.argv[1:])); sys.exit(3)"]
    profile.write_wrapper(wrapper, driver, tool)
    commands = tmp_path / "commands"
    result = subprocess.run(
        [str(wrapper), *arguments],
        text=True,
        capture_output=True,
        env={**os.environ, "KERNEL_COMPILER_PROFILE_DIR": str(commands)},
    )
    assert result.returncode == 3
    assert json.loads(result.stdout) == arguments
    records = list(commands.glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["kind"] == kind
    assert record["wall_seconds"] >= 0
    assert record["cpu_user_seconds"] >= 0
    assert record["max_child_rss_kib"] > 0


def test_command_summary_does_not_claim_sum_is_wall_clock():
    records = [
        {
            "kind": "cpp_compile",
            "wall_seconds": 10,
            "cpu_user_seconds": 4,
            "cpu_system_seconds": 1,
            "max_child_rss_kib": 1024,
        },
        {
            "kind": "cuda_compile",
            "wall_seconds": 12,
            "cpu_user_seconds": 7,
            "cpu_system_seconds": 1,
            "max_child_rss_kib": 2048,
        },
    ]
    summary = profile.command_summary(records)
    assert summary["cpp_compile"]["summed_wall_seconds"] == 10
    assert summary["cuda_compile"]["summed_cpu_seconds"] == 8
    assert summary["link"]["count"] == 0
    assert summary["link"]["max_child_rss_kib"] is None


def test_counter_delta_and_missing_cgroup_data():
    assert profile.counter_delta(None, None) is None
    assert profile.counter_delta(
        "usage_usec 10\nnr_throttled 1", "usage_usec 18\nnr_throttled 3"
    ) == {
        "usage_usec": 8,
        "nr_throttled": 2,
    }


def test_parallelism_is_bounded_before_starting_compilers(tmp_path):
    with pytest.raises(ValueError):
        profile.run_batch([], tmp_path / "invalid", {}, 3, 2, 0)
    with pytest.raises(ValueError):
        profile.run_batch([], tmp_path / "invalid", {}, 2, 8, 0)


def test_modal_entrypoint_cannot_request_or_dispatch_a_gpu():
    tree = ast.parse((ROOT / "modal_app/compiler_profile.py").read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    decorator = functions["cpu_profile"].decorator_list[0]
    assert all(keyword.arg != "gpu" for keyword in decorator.keywords)
    options = {
        item.arg: ast.literal_eval(item.value)
        for item in decorator.keywords
        if item.arg in {"cpu", "memory", "timeout", "retries", "max_containers", "scaledown_window"}
    }
    assert options == {
        "cpu": (8, 8),
        "memory": (16384, 32768),
        "timeout": 720,
        "retries": 0,
        "max_containers": 1,
        "scaledown_window": 2,
    }
    remote_targets = [
        ast.unparse(node.func.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "remote"
    ]
    assert remote_targets == ["cpu_profile"]
    assert "Path('/dev').glob('nvidia[0-9]*')" in ast.unparse(functions["cpu_profile"])


def test_compiler_environment_masks_devices_and_wraps_both_tools(tmp_path, monkeypatch):
    cuda = tmp_path / "cuda"
    (cuda / "bin").mkdir(parents=True)
    (cuda / "bin/nvcc").touch()
    monkeypatch.setenv("CUDA_HOME", str(cuda))
    environment, tools = profile.compiler_environment(tmp_path / "diagnostic")
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["NVIDIA_VISIBLE_DEVICES"] == "void"
    assert environment["TORCH_CUDA_ARCH_LIST"] == "8.0"
    assert Path(environment["CXX"]).is_file()
    assert Path(environment["PYTORCH_NVCC"]).is_file()
    assert tools["nvcc"] == str(cuda / "bin/nvcc")
