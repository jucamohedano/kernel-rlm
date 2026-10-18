"""CPU-only instrumentation for compiler commands and bounded build batches."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOOL_WRAPPER = r"""
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

driver = DRIVER
tool = TOOL
started = time.monotonic()
before = resource.getrusage(resource.RUSAGE_CHILDREN)
result = subprocess.run([*driver, *sys.argv[1:]])
after = resource.getrusage(resource.RUSAGE_CHILDREN)
arguments = sys.argv[1:]
kind = ("cuda_compile" if tool == "nvcc" else "cpp_compile") if "-c" in arguments else (
    "link" if "-shared" in arguments else "probe")
record = {
    "tool": tool, "kind": kind, "arguments": arguments,
    "start_monotonic": started, "end_monotonic": time.monotonic(),
    "cpu_user_seconds": after.ru_utime - before.ru_utime,
    "cpu_system_seconds": after.ru_stime - before.ru_stime,
    "max_child_rss_kib": after.ru_maxrss, "returncode": result.returncode,
    "cwd": os.getcwd(),
}
record["wall_seconds"] = record["end_monotonic"] - started
directory = Path(os.environ["KERNEL_COMPILER_PROFILE_DIR"])
directory.mkdir(parents=True, exist_ok=True)
(directory / (uuid4().hex + ".json")).write_text(json.dumps(record))
raise SystemExit(result.returncode)
"""


def write_wrapper(path: Path, driver: list[str], tool: str) -> None:
    if tool not in {"cxx", "nvcc"} or not driver:
        raise ValueError("Invalid compiler wrapper specification")
    source = TOOL_WRAPPER.replace("driver = DRIVER", "driver = " + repr(driver))
    source = source.replace("tool = TOOL", "tool = " + repr(tool))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n" + source)
    path.chmod(0o755)


def cgroup_sample() -> dict:
    sample = {"monotonic_seconds": time.monotonic()}
    for name in ("memory.current", "memory.max", "cpu.max", "cpu.stat", "memory.events"):
        path = Path("/sys/fs/cgroup") / name
        sample[name] = path.read_text().strip() if path.exists() else None
    return sample


def counter_delta(before: str | None, after: str | None) -> dict | None:
    if before is None or after is None:
        return None
    previous = dict(line.split() for line in before.splitlines())
    current = dict(line.split() for line in after.splitlines())
    return {
        key: int(value) - int(previous[key]) for key, value in current.items() if key in previous
    }


def command_summary(commands: list[dict]) -> dict:
    return {
        kind: {
            "count": len(records := [record for record in commands if record["kind"] == kind]),
            "summed_wall_seconds": sum(record["wall_seconds"] for record in records),
            "summed_cpu_seconds": sum(
                record["cpu_user_seconds"] + record["cpu_system_seconds"] for record in records
            ),
            "max_child_rss_kib": max(
                (record["max_child_rss_kib"] for record in records), default=None
            ),
        }
        for kind in ("cpp_compile", "cuda_compile", "link", "probe")
    }


def compile_profiled(
    case: dict, directory: Path, environment: dict[str, str], timeout: float
) -> dict:
    from kernel_rlm.diagnostics import source_hash
    from kernel_rlm.verifier import _RUNNER, _run_verifier_process

    directory.mkdir(parents=True, exist_ok=True)
    commands_dir = directory / "commands"
    cache_root = directory / "cache" / source_hash(case["submission"])
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "runner.py").write_text(_RUNNER)
        (root / "submission.py").write_text(case["submission"])
        process = _run_verifier_process(
            [
                sys.executable,
                str(root / "runner.py"),
                "unused_reference.py",
                str(root / "submission.py"),
                "1",
                "0.001",
                "0.0001",
                "0",
                str(root / "phase.json"),
                "compile",
            ],
            timeout_s=timeout,
            cwd=None,
            env={
                **environment,
                "KERNEL_RLM_EXTENSION_BUILD_ROOT": str(cache_root),
                "KERNEL_COMPILER_PROFILE_DIR": str(commands_dir),
            },
        )
    (directory / "compiler.log").write_text(process.stdout + process.stderr)
    commands = [json.loads(path.read_text()) for path in sorted(commands_dir.glob("*.json"))]
    if process.returncode:
        raise RuntimeError(process.stderr[-1800:])
    result = json.loads(process.stdout.strip().splitlines()[-1])
    summary = command_summary(commands)
    if not result.get("compiled"):
        raise RuntimeError(result.get("error", "Compilation failed"))
    if not summary["cpp_compile"]["count"] or not summary["cuda_compile"]["count"]:
        raise RuntimeError("Incomplete compiler coverage: both C++ and NVCC must be captured")
    return {
        "case": case["name"],
        "seconds": time.monotonic() - started,
        "phase_events": result.get("phase_events", []),
        "commands": commands,
        "command_summary": summary,
        "source_sha256": source_hash(case["submission"]),
    }


def run_batch(
    cases: list[dict],
    directory: Path,
    environment: dict[str, str],
    parallel: int,
    jobs: int,
    deadline: float,
) -> dict:
    if parallel not in (1, 2) or jobs not in (1, 2):
        raise ValueError("Only bounded one/two build and worker counts are allowed")
    directory.mkdir(parents=True, exist_ok=False)
    samples = [cgroup_sample()]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(0.5):
            samples.append(cgroup_sample())

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    started = time.monotonic()

    def run_case(case: dict) -> dict:
        remaining = min(110, deadline - time.monotonic())
        if remaining < 10:
            raise TimeoutError("CPU diagnostic work budget exhausted")
        return compile_profiled(
            case, directory / case["name"], {**environment, "MAX_JOBS": str(jobs)}, remaining
        )

    try:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            records = list(pool.map(run_case, cases))
    finally:
        stop.set()
        sampler.join()
        samples.append(cgroup_sample())
        (directory / "resource-samples.json").write_text(json.dumps(samples))
    return {
        "label": directory.name,
        "parallel_builds": parallel,
        "max_jobs_per_build": jobs,
        "wall_seconds": time.monotonic() - started,
        "builds": records,
        "peak_cgroup_memory_bytes": max(
            (int(item["memory.current"]) for item in samples if item["memory.current"] is not None),
            default=None,
        ),
        "cpu_stat_delta": counter_delta(samples[0]["cpu.stat"], samples[-1]["cpu.stat"]),
        "memory_events_delta": counter_delta(
            samples[0]["memory.events"], samples[-1]["memory.events"]
        ),
        "resource_samples": samples,
    }


def compiler_environment(directory: Path) -> tuple[dict[str, str], dict]:
    import shlex

    compiler = shlex.split(os.environ.get("CXX", "c++"))
    compiler[0] = shutil.which(compiler[0]) or compiler[0]
    nvcc = str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc")
    if not Path(nvcc).is_file():
        raise ValueError("CUDA compiler not available in CPU image")
    wrappers = directory / "tools"
    write_wrapper(wrappers / "g++", compiler, "cxx")
    write_wrapper(wrappers / "nvcc", [nvcc], "nvcc")
    return {
        **os.environ,
        "CXX": str(wrappers / "g++"),
        "PYTORCH_NVCC": str(wrappers / "nvcc"),
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "TORCH_CUDA_ARCH_LIST": "8.0",
    }, {
        "cxx": compiler,
        "nvcc": nvcc,
        "torch_cuda_arch_list": "8.0",
        "cxx_wrapper": str(wrappers / "g++"),
        "nvcc_wrapper": str(wrappers / "nvcc"),
    }
