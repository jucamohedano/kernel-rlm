"""Frozen native-REPL ICL experiment: 12 tasks x two independent conditions.

Preparation is CPU-only. Generation requires explicit isolated REPL/worker
bindings. Validation is a separate command, after the operator stops the model.
There is no automatic retry or resume, including after an interrupted batch.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import posixpath
import random
import sys
import tempfile
import traceback
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from rlm_train.native import NativeProtocolError

from cute_rlm.icl.experiment import ExperimentConfig as ICLExperimentConfig
from cute_rlm.icl.experiment import (
    capture_metadata,
    extract_source,
    now,
    sha256,
    write_json,
)

MODE = "native-repl-cute-icl"
CONDITIONS = ("api", "api_example")
PINNED_MODELS = {
    "poolside/Laguna-XS-2.1": "c5f36269bbdbd3f27fddc9a9f9dbae0cf2cf57db",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": "a9904d24bcc1d289a1950fa9d2b978c47cf903b9",
}
PROJECT = Path(__file__).resolve().parents[5]
SOURCE_ROOTS = ("training/src/rlm_train", "training/environments/cute_rlm/cute_rlm")
CANDIDATE_FILES = (
    "__init__.py",
    "outputs.py",
    "verify_worker.py",
    "contracts.py",
    "compile_trace.py",
    "icl/__init__.py",
    "icl/worker.py",
    "icl/inputs.py",
)


@dataclass(frozen=True)
class ReplExperimentConfig(ICLExperimentConfig):
    """New experiment settings; the tool-free protocol defaults stay unchanged."""

    max_tokens: int = 16384
    request_timeout: float = 800
    episode_timeout: float = 1800
    max_generated_tokens: int = 1_000_000
    max_depth: int = 1
    max_turns: int = 64
    max_children: int = 32
    max_live_children: int = 8
    max_concurrent_requests: int = 2
    max_model_calls: int = 256
    broker_timeout: float = 800
    broker_request_bytes: int = 8 * 1024**2
    repl_exec_timeout: float = 800
    worker_startup_timeout: float = 30

    def __post_init__(self):
        super().__post_init__()
        if PINNED_MODELS.get(self.model_id) != self.model_revision:
            raise ValueError("Choose one of the two pinned BF16 model/revision pairs")
        fixed = {
            "max_tokens": 16384,
            "request_timeout": 800,
            "episode_timeout": 1800,
            "max_generated_tokens": 1_000_000,
            "max_depth": 1,
            "max_turns": 64,
            "max_children": 32,
            "max_live_children": 8,
            "max_concurrent_requests": 2,
            "max_model_calls": 256,
            "temperature": 0.6,
            "top_p": 0.95,
            "seed": 20261101,
            "broker_timeout": 800,
            "broker_request_bytes": 8 * 1024**2,
            "repl_exec_timeout": 800,
            "worker_startup_timeout": 30,
            "tokenize": False,
        }
        for name, value in fixed.items():
            if getattr(self, name) != value or isinstance(getattr(self, name), bool) != isinstance(
                value, bool
            ):
                raise ValueError(f"The frozen REPL protocol requires {name}={value}")


def paired_seed(task_id: str, seed: int) -> int:
    return (seed + int(sha256(task_id.encode())[:8], 16)) % 2**32


def render_prompt(task: dict, condition: str) -> tuple[dict, dict]:
    """Reuse the tool-free semantic/API/example packets, changing only tool instructions."""
    from cute_rlm.icl.prompts import render_prompt as render_tool_free
    from cute_rlm.prompts import ORCHESTRATOR_ADDENDUM

    if condition not in CONDITIONS:
        raise ValueError("Only api and api_example conditions are supported")
    original = render_tool_free(task, condition)
    no_tools = "There are no tools, external files, runtime context variables or compiler feedback."
    if original["user"].count(no_tools) != 1:
        raise ValueError("Tool-free prompt contract changed; review the REPL adaptation")
    user = original["user"].replace(
        no_tools,
        "Use native repl(code) for CPU reasoning and compile_candidate(source) for isolated "
        "compile/forward feedback on all four public cases. Numerical feedback is unavailable.",
    )
    orchestration = (
        ORCHESTRATOR_ADDENDUM.replace(
            "then use rlm_query(prompt, blocking=False) to delegate",
            "you may use rlm_query(prompt, blocking=False) to delegate",
        )
        .replace("Pass the reference source,", "Pass the task specification,")
        .replace("Collect the child with", "If you delegate, collect the child with")
        .replace(
            'See resources["delegation_example"] for a short example.',
            'See resources["resource_index"] for the same-condition task and packet index.',
        )
    )
    system = (
        "Implement the task using Python CuTe DSL (import cutlass.cute as cute), not the "
        "older C++ API. At the root submit a standalone module defining ModelNew(torch.nn.Module) "
        "with the supplied signatures. All required arithmetic must run in an output-producing "
        "CuTe device kernel. Torch is only for allocation, metadata and tensor/stream plumbing. "
        "No Triton, C++ extensions or torch.compile. No input mutation or timing requirement.\n\n"
        "Your only external tool is native repl(code). Its Python variables persist. The reasoning "
        "REPL is CPU-only. Each child has its own REPL and the identical resources dictionary for "
        "this condition; it does not inherit your variables. resources contains only the public "
        "task specification and the supplied API/example packets, not documentation or an oracle. "
        "SHOW_VARS(), compile_candidate(source), rlm_query, rlm_wait and rlm_result are already "
        "injected; call them directly, do not import or redefine them. compile_candidate reports "
        "compile/forward behavior for all four public cases only, not numerical correctness. "
        "Keep complete source in best_source. Root submission: answer['content'] = best_source; "
        "answer['ready'] = True. Child submission uses the same answer dictionary for its report. "
        "There are at most 64 model turns per session.\n\n" + orchestration
    )
    sources = original["sources"]
    resources = {
        "condition": condition,
        "task_spec": user.split("\nAPI packet (provided in full):", 1)[0],
        "packets": {item["id"]: item["content"] for item in sources},
        "resource_index": [
            {
                "id": item["id"],
                "kind": item["kind"],
                "characters": len(item["content"]),
                "sha256": sha256(item["content"].encode()),
            }
            for item in sources
        ],
    }
    if condition == "api" and any(item["kind"] == "example" for item in sources):
        raise ValueError("The API condition must not expose an example")
    return {"system": system, "user": user, "sources": sources}, resources


def resources_bootstrap(resources: dict) -> str:
    return (
        "import json as _icl_json\nresources = _icl_json.loads("
        + repr(json.dumps(resources))
        + ")\ndel _icl_json\n"
    )


def capture_sources(output: Path) -> dict[str, str]:
    paths = [
        path
        for name in SOURCE_ROOTS
        for path in (PROJECT / name).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    paths.extend(
        PROJECT / name
        for name in (
            "pyproject.toml",
            "uv.lock",
            "training/environments/cute_rlm/pyproject.toml",
        )
    )
    records = {}
    for path in sorted(paths):
        name = str(path.relative_to(PROJECT))
        data = path.read_bytes()
        destination = output / "source_capture" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        records[name] = sha256(data)
    write_json(output / "source_capture.json", records)
    return records


def prepare(output: str | Path, config: ReplExperimentConfig | None = None) -> dict:
    """Freeze 24 roots, all prompts/resources, task cases and complete runtime sources."""
    from cute_rlm.icl.tasks import load_tasks

    config = config or ReplExperimentConfig()
    tasks = load_tasks()
    if len(tasks) != 12 or len({task["id"] for task in tasks}) != 12:
        raise ValueError("Exactly 12 distinct public tasks are required")
    if any(len(task["cases"]) != 4 for task in tasks):
        raise ValueError("Every task must retain all four public cases")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", asdict(config))
    write_json(
        output / "protocol.json",
        {
            "mode": MODE,
            "task_count": 12,
            "conditions": list(CONDITIONS),
            "planned_roots": 24,
            "independent_roots": True,
            "same_condition_child_resources": True,
            "child_repl_at_limit": True,
            "retries": 0,
            "resume": False,
            "compaction": False,
            "per_call_tokens_including_reasoning": 16384,
            "compile_feedback": "compile/forward only; four public cases; no expected arrays or numerical feedback",
            "final_validation": "separate command after operator model-stop acknowledgement",
            "valid_solution": None,
            "source_review_required": True,
        },
    )
    entries = []
    for index, task in enumerate(tasks):
        task_path = f"tasks/{index:02d}/task.json"
        write_json(output / task_path, task)
        for condition in CONDITIONS:
            entry_id = f"{index:02d}-{condition}"
            prompt, resources = render_prompt(task, condition)
            directory = output / "prompts" / entry_id
            write_json(directory / "prompt.json", prompt)
            write_json(directory / "resources.json", resources)
            for key in ("system", "user"):
                (directory / f"{key}.txt").write_text(prompt[key])
            (directory / "bootstrap.py").write_text(resources_bootstrap(resources))
            entries.append(
                {
                    "id": entry_id,
                    "task_id": task["id"],
                    "task_path": task_path,
                    "condition": condition,
                    "prompt_path": f"prompts/{entry_id}/prompt.json",
                    "resources_path": f"prompts/{entry_id}/resources.json",
                    "bootstrap_path": f"prompts/{entry_id}/bootstrap.py",
                    "sampling_seed": paired_seed(task["id"], config.seed),
                }
            )
    random.Random(config.seed).shuffle(entries)
    capture_sources(output)
    artifacts = {
        str(path.relative_to(output)): sha256(path.read_bytes())
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    plan = {
        "mode": MODE,
        "run_id": output.name,
        "created_at": now(),
        "entries": entries,
        "artifacts": artifacts,
    }
    write_json(output / "plan.json", plan)
    (output / "plan.sha256").write_text(sha256((output / "plan.json").read_bytes()) + "\n")
    return plan


def load_plan(run: Path) -> tuple[dict, ReplExperimentConfig]:
    run = run.resolve()
    data = (run / "plan.json").read_bytes()
    if sha256(data) != (run / "plan.sha256").read_text().strip():
        raise ValueError("Frozen plan SHA256 mismatch")
    plan = json.loads(data)
    if plan["mode"] != MODE or len(plan["entries"]) != 24:
        raise ValueError("Expected the 24-root native REPL experiment")
    if len({entry["id"] for entry in plan["entries"]}) != 24 or {
        entry["condition"] for entry in plan["entries"]
    } != set(CONDITIONS):
        raise ValueError("Invalid independent root/condition matrix")
    for name, expected in plan["artifacts"].items():
        path = (run / name).resolve()
        if not path.is_relative_to(run) or sha256(path.read_bytes()) != expected:
            raise ValueError(f"Frozen artifact SHA256 mismatch: {name}")
    captured = json.loads((run / "source_capture.json").read_text())
    live_paths = {
        str(path.relative_to(PROJECT))
        for name in SOURCE_ROOTS
        for path in (PROJECT / name).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if not live_paths <= set(captured):
        raise ValueError("Runtime source file inventory changed after preparation")
    for name, expected in captured.items():
        path = (PROJECT / name).resolve()
        if not path.is_relative_to(PROJECT) or sha256(path.read_bytes()) != expected:
            raise ValueError(f"Current source differs from frozen capture: {name}")
    return plan, ReplExperimentConfig(**json.loads((run / "config.json").read_text()))


def check_worker_sources(run: Path, spec_path: Path) -> None:
    """Require exact explicit worker/runtime package bindings, never a checkout mount."""
    from cute_rlm.sandbox import SandboxSpec

    spec = SandboxSpec.read(spec_path)
    captured = json.loads((run / "source_capture.json").read_text())
    packages = {
        "/runtime/python/cute_rlm": "training/environments/cute_rlm/cute_rlm/",
        "/runtime/python/rlm_train": "training/src/rlm_train/",
    }
    for target, prefix in packages.items():
        matches = [item for item in spec.readonly_bindings if item["target"] == target]
        if len(matches) != 1:
            raise ValueError(f"Sandbox needs one explicit frozen package binding: {target}")
        package = Path(matches[0]["source"]).resolve()
        expected_names = {name.removeprefix(prefix) for name in captured if name.startswith(prefix)}
        package_paths = list(package.rglob("*"))
        if any(path.is_symlink() for path in package_paths):
            raise ValueError(f"Sandbox package must not contain symlinks: {target}")
        actual_names = {
            str(path.relative_to(package))
            for path in package_paths
            if path.is_file() and "__pycache__" not in path.parts
        }
        if actual_names != expected_names:
            raise ValueError(f"Sandbox worker package inventory differs: {target}")
        for name in expected_names:
            if sha256((package / name).read_bytes()) != captured[prefix + name]:
                raise ValueError(
                    f"Sandbox worker source differs from frozen capture: {prefix + name}"
                )
    for item in spec.readonly_bindings:
        source, target = Path(item["source"]).resolve(), item["target"]
        if target in packages:
            continue
        if source == run or source in run.parents or run in source.parents:
            raise ValueError("Sandbox must not mount experiment records or their ancestors")
        if source == PROJECT or source in PROJECT.parents or PROJECT in source.parents:
            raise ValueError("Sandbox must not mount extra checkout documents or source")
        if target == "/runtime/python" or target.startswith("/runtime/python/"):
            raise ValueError("Extra Python namespace bindings are not allowed")
        os_paths = {
            "/usr",
            "/lib",
            "/lib64",
            "/bin",
            "/etc/ld.so.cache",
            "/etc/passwd",
            "/etc/group",
            "/etc/nsswitch.conf",
            "/etc/os-release",
            "/sys/devices/system",
            "/sys/module/nvidia",
            "/sys/module/nvidia_uvm",
        }
        venv = Path(spec.runtime_python).parent.parent
        if target in os_paths and source == Path(target).resolve():
            continue
        if target == str(venv) and (source / "pyvenv.cfg").is_file():
            continue
        raise ValueError(f"Sandbox binding is not an explicit OS library or runtime venv: {target}")


def make_repl_wrapper(run: Path, spec_path: Path, repl_python: Path, host_python: str) -> dict:
    """Drop the worker's CuTe package and all GPU nodes from the reasoning namespace."""
    from cute_rlm.launch import write_repl_wrapper
    from cute_rlm.sandbox import SandboxSpec

    spec = SandboxSpec.read(spec_path)
    spec.validate_host()
    original = json.loads(spec_path.read_text())
    captured = json.loads((run / "source_capture.json").read_text())
    prefix = "training/environments/cute_rlm/cute_rlm/"
    package = run / "candidate-runtime" / "cute_rlm"
    package.mkdir(parents=True, exist_ok=False)
    candidate_files = {}
    for name in CANDIDATE_FILES:
        source = run / "source_capture" / (prefix + name)
        data = source.read_bytes()
        if sha256(data) != captured[prefix + name]:
            raise ValueError(f"Frozen candidate runtime source changed: {name}")
        destination = package / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        candidate_files[name] = sha256(data)
    candidate = {**original, "observer_root": None}
    candidate["readonly_bindings"] = [
        {**item, "source": str(package)} if item["target"] == "/runtime/python/cute_rlm" else item
        for item in original["readonly_bindings"]
        if item["target"] != "/runtime/python/rlm_train"
    ]
    candidate_spec = run / "candidate-sandbox-spec.json"
    with candidate_spec.open("x") as stream:
        json.dump(candidate, stream, indent=2)
    SandboxSpec.read(candidate_spec).validate_host()
    candidate_probe = run / "candidate-probe-python"
    write_repl_wrapper(candidate_probe, candidate_spec, host_python)
    raw = dict(original)
    raw["readonly_bindings"] = [
        item
        for item in raw["readonly_bindings"]
        if item["target"]
        not in {
            "/runtime/python/cute_rlm",
            "/sys/module/nvidia",
            "/sys/module/nvidia_uvm",
        }
    ]
    raw["gpu_devices"] = []
    raw["observer_root"] = None
    repl_spec = run / "repl-sandbox-spec.json"
    with repl_spec.open("x") as stream:
        json.dump(raw, stream, indent=2)
    SandboxSpec.read(repl_spec).validate_host()
    write_repl_wrapper(repl_python, repl_spec, host_python)
    return {
        "worker_spec": str(spec_path),
        "worker_spec_sha256": sha256(spec_path.read_bytes()),
        "repl_spec": str(repl_spec),
        "repl_spec_sha256": sha256(repl_spec.read_bytes()),
        "repl_python": str(repl_python),
        "repl_python_sha256": sha256(repl_python.read_bytes()),
        "host_python": host_python,
        "candidate_spec": str(candidate_spec),
        "candidate_spec_sha256": sha256(candidate_spec.read_bytes()),
        "candidate_probe_python": str(candidate_probe),
        "candidate_probe_python_sha256": sha256(candidate_probe.read_bytes()),
        "candidate_package": str(package),
        "candidate_files": candidate_files,
    }


def check_launch_binding(binding: dict) -> None:
    for name in (
        "worker_spec",
        "repl_spec",
        "repl_python",
        "candidate_spec",
        "candidate_probe_python",
    ):
        if sha256(Path(binding[name]).read_bytes()) != binding[f"{name}_sha256"]:
            raise ValueError(f"Launch binding changed: {name}")
    if not os.access(binding["repl_python"], os.X_OK):
        raise ValueError("The isolated REPL wrapper must remain executable")
    package = Path(binding["candidate_package"])
    paths = list(package.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Candidate runtime package must not contain symlinks")
    files = {
        str(path.relative_to(package)): sha256(path.read_bytes())
        for path in paths
        if path.is_file()
    }
    if (
        set(binding["candidate_files"]) != set(CANDIDATE_FILES)
        or files != binding["candidate_files"]
    ):
        raise ValueError("Candidate runtime package differs from its frozen allowlist")


class ExperimentInfrastructureError(RuntimeError):
    """Frozen host inputs or isolated runtime bindings changed during execution."""


class UnsupportedModelError(NativeProtocolError):
    """A model-selected child/plain model is outside this experiment's pinned identity."""


def request_guard(run: Path, binding: dict, config: ReplExperimentConfig, failures: list):
    async def guard(request: httpx.Request) -> None:
        payload = json.loads(request.content) if request.method == "POST" else None
        if payload is not None and payload.get("model") != config.model_id:
            raise UnsupportedModelError("Every root and child request must use the pinned model")
        try:
            if failures:
                raise RuntimeError(
                    "Host infrastructure already failed; no further requests"
                ) from failures[0]
            load_plan(run)
            check_launch_binding(binding)
            check_worker_sources(run, Path(binding["worker_spec"]))
            if payload is not None:
                if request.url.path.endswith("/chat/completions"):
                    if not 0 < payload.get("max_tokens", 0) <= config.max_tokens:
                        raise ValueError("Generation exceeds the fixed per-request cap")
                    if (
                        payload.get("temperature") != config.temperature
                        or payload.get("top_p") != config.top_p
                    ):
                        raise ValueError("Sampling differs from the frozen protocol")
        except Exception as error:
            failures.append(error)
            raise ExperimentInfrastructureError(
                "Pre-request frozen input/binding check failed"
            ) from error

    return guard


def validate_namespace_layout(layout: dict, spec_path: Path) -> None:
    """Check the observed mount/import layout, not only the intended launcher spec."""
    spec = json.loads(spec_path.read_text())
    targets = [item["target"] for item in spec["readonly_bindings"]]
    targets.append(spec["runtime_python"])

    def under(path, roots):
        normalized = posixpath.normpath(path)
        return any(
            normalized == root or normalized.startswith(root.rstrip("/") + "/") for root in roots
        )

    import_roots = [*targets, "/runtime/python", "/tmp"]
    if any(not under(path or "/tmp", import_roots) for path in layout["sys_path"]):
        raise ValueError("Namespace sys.path escapes declared runtime bindings")
    for name, content in layout["pth_files"].items():
        if not under(name, targets) or "__editable__" in content:
            raise ValueError("Namespace exposes an undeclared or editable .pth hook")
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "import ", "import\t")):
                continue
            resolved = posixpath.join(posixpath.dirname(name), line)
            if not under(resolved, import_roots):
                raise ValueError("Namespace .pth path escapes declared runtime bindings")
    observed = {}
    for line in layout["mountinfo"].splitlines():
        fields = line.split()
        separator = fields.index("-")
        target = fields[4].replace("\\040", " ").replace("\\134", "\\")
        filesystem = fields[separator + 1]
        observed[target] = filesystem
        if target == "/":
            if filesystem != "tmpfs":
                raise ValueError("Namespace root is not the private bubblewrap tmpfs")
        elif target == "/run/rlm-root-broker.sock" or under(
            target, [*targets, "/proc", "/dev", "/tmp"]
        ):
            continue
        else:
            raise ValueError(f"Unexpected namespace mount: {target}")
    for target, filesystem in {
        "/": "tmpfs",
        "/proc": "proc",
        "/dev": "tmpfs",
        "/tmp": "tmpfs",
        "/dev/shm": "tmpfs",
    }.items():
        if observed.get(target) != filesystem:
            raise ValueError(f"Namespace lacks its expected private mount: {target}")


async def preflight_repl(run: Path, config: ReplExperimentConfig, wrapper: Path) -> dict:
    """Inspect the actual CPU namespace before allowing any model request."""
    from rlm_train.proxy import SubLLMProxy

    from cute_rlm.evaluate import isolated_backend_factory

    code = (
        "import glob, importlib.util, json, pathlib, sys\n"
        "layout = {'sys_path': sys.path, 'mountinfo': pathlib.Path('/proc/self/mountinfo').read_text(), "
        "'gpu_devices': glob.glob('/dev/nvidia*'), 'discoverable_packages': "
        "{name: str(importlib.util.find_spec(name)) for name in ('cute_rlm', 'kernel_factory')}}\n"
        "layout['pth_files'] = {str(path): path.read_text(errors='replace') for base in sys.path "
        "for path in pathlib.Path(base).glob('*.pth') if path.is_file()}\n"
        "print(json.dumps(layout))\n"
    )
    backend = isolated_backend_factory(config, str(wrapper))()
    with tempfile.TemporaryDirectory(prefix="cute-icl-preflight-") as broker:
        proxy = SubLLMProxy(socket_path=Path(broker) / "broker.sock")
        await proxy.start()
        try:
            await backend.start(proxy.url, "namespace-preflight", depth=0)
            await backend.load_context({})
            observed = await backend.execute(code)
            if observed.stderr:
                raise ValueError(f"REPL namespace preflight failed: {observed.stderr}")
            layout = json.loads(observed.stdout)
            write_json(run / "repl-namespace-preflight.json", layout)
            validate_namespace_layout(layout, run / "repl-sandbox-spec.json")
            if layout["gpu_devices"] or any(
                value != "None" for value in layout["discoverable_packages"].values()
            ):
                raise ValueError("Reasoning namespace exposes GPU devices or task/oracle packages")
        finally:
            try:
                await backend.stop()
            finally:
                await proxy.stop()
    return layout


async def preflight_candidate_namespace(
    run: Path, binding: dict, config: ReplExperimentConfig
) -> dict:
    """Probe the helper package through the same spec, with REPL mode omitting GPU nodes."""
    from rlm_train.proxy import SubLLMProxy

    script = "\n".join(
        [
            "import glob, importlib.util, json, pathlib, sys",
            "import cute_rlm.icl.inputs as inputs",
            "import cute_rlm.icl.worker",
            "def find(name):",
            "    try: return str(importlib.util.find_spec(name))",
            "    except ModuleNotFoundError: return 'None'",
            "blocked = ('cute_rlm.icl.tasks', 'cute_rlm.icl.prompts', 'cute_rlm.docs', 'cute_rlm.resources', 'cute_rlm.dataset', 'cute_rlm.examples', 'kernel_factory')",
            "package = pathlib.Path('/runtime/python/cute_rlm')",
            "layout = {'sys_path': sys.path, 'mountinfo': pathlib.Path('/proc/self/mountinfo').read_text(), 'gpu_devices': glob.glob('/dev/nvidia*'), 'discoverable_packages': {name: find(name) for name in blocked}, 'expected_output_available': hasattr(inputs, 'expected_output'), 'package_files': sorted(str(path.relative_to(package)) for path in package.rglob('*') if path.is_file())}",
            "layout['pth_files'] = {str(path): path.read_text(errors='replace') for base in sys.path for path in pathlib.Path(base).glob('*.pth') if path.is_file()}",
            "print(json.dumps(layout))",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="cute-candidate-preflight-") as broker:
        proxy = SubLLMProxy(socket_path=Path(broker) / "broker.sock")
        await proxy.start()
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                binding["candidate_probe_python"],
                "-B",
                "-c",
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={"PATH": os.environ.get("PATH", ""), "RLM_TRAIN_PROXY_URL": proxy.url},
            )
            async with asyncio.timeout(config.worker_startup_timeout):
                stdout, stderr = await process.communicate()
            (run / "candidate-namespace-preflight.stdout").write_bytes(stdout)
            (run / "candidate-namespace-preflight.stderr").write_bytes(stderr)
            if process.returncode != 0:
                raise ValueError(
                    "Slim candidate namespace preflight failed; diagnostics saved on host"
                )
            layout = json.loads(stdout)
            write_json(run / "candidate-namespace-preflight.json", layout)
            validate_namespace_layout(layout, Path(binding["candidate_spec"]))
            if (
                layout["gpu_devices"]
                or layout["expected_output_available"]
                or any(value != "None" for value in layout["discoverable_packages"].values())
                or set(layout["package_files"]) != set(CANDIDATE_FILES)
            ):
                raise ValueError("Candidate helper namespace exposes extra task/oracle/resources")
        finally:
            try:
                if process is not None and process.returncode is None:
                    process.kill()
                    await process.wait()
            finally:
                await proxy.stop()
    return layout


def infrastructure_failure(record: dict) -> str | None:
    """Model/response failures are per-root; disk, namespace and cleanup failures are fatal."""
    for session in [record, *record.get("child_sessions", [])]:
        if session.get("checkpoint_error") or session.get("cleanup_error"):
            return f"Session evidence/cleanup failure: {session.get('session_id')}"
        if session.get("error_class") in {"checkpoint", "cleanup"}:
            return session.get("error") or "Session evidence/cleanup failure"
        failure = session.get("failure") or {}
        chain = failure.get("exception_chain", [])
        if chain and chain[0].get("type") == "UnsupportedModelError":
            continue
        if any(item.get("type") == "ExperimentInfrastructureError" for item in chain):
            return "Frozen host input or namespace binding failure"
        if chain and chain[0].get("type") in {
            "OSError",
            "PermissionError",
            "FileNotFoundError",
            "IsADirectoryError",
        }:
            return "Host filesystem failure"
        if session.get("error_class") == "infrastructure" and failure.get("stage") not in {
            "model_request",
            "response_validation",
        }:
            return session.get("error") or "Runtime infrastructure failure"
        if failure.get("stage") == "prompt_count":
            return "Server tokenizer/context infrastructure failure"
    return None


async def _generate_entries(
    run: Path,
    plan: dict,
    config: ReplExperimentConfig,
    *,
    client,
    worker_command: list[str],
    backend_factory,
    binding: dict,
    host_failures: list,
) -> dict:
    from rlm_train.native import NativeLimits, NativeRuntime
    from rlm_train.proxy import SubLLMProxy
    from rlm_train.token_budget import ContextBudgetPolicy, VllmPromptCounter

    from cute_rlm.evaluate import cute_turn_notice
    from cute_rlm.icl.repl_feedback import ReplCompileFeedback

    summary: dict[str, Any] = {
        "planned": 24,
        "attempted": 0,
        "results": [],
        "retries": 0,
        "resume": False,
    }
    try:
        for entry in plan["entries"]:
            load_plan(run)
            check_launch_binding(binding)
            check_worker_sources(run, Path(binding["worker_spec"]))
            directory = run / "completions" / entry["id"]
            directory.mkdir(parents=True, exist_ok=False)
            task = json.loads((run / entry["task_path"]).read_text())
            prompt = json.loads((run / entry["prompt_path"]).read_text())
            resources = json.loads((run / entry["resources_path"]).read_text())
            result = {
                **entry,
                "started_at": now(),
                "status": "started",
                "source_valid": False,
                "valid_solution": None,
                "source_review_required": True,
            }
            write_json(directory / "result.json", result)
            summary["attempted"] += 1
            feedback = ReplCompileFeedback(task, directory, worker_command=worker_command)
            infra = []
            control = {"runtime": None}

            async def compile_checked(
                session_id, source, _feedback=feedback, _infra=infra, _control=control
            ):
                try:
                    load_plan(run)
                    check_launch_binding(binding)
                    check_worker_sources(run, Path(binding["worker_spec"]))
                    response = await _feedback(session_id, source)
                    if response.get("status") == "infrastructure":
                        raise RuntimeError(
                            "Compile worker infrastructure failed; diagnostics saved on host"
                        )
                    return response
                except Exception as error:
                    _infra.append(error)
                    host_failures.append(error)
                    active = _control["runtime"]
                    root_task = getattr(active.root, "task", None) if active is not None else None
                    if root_task is not None and not root_task.done():
                        root_task.cancel()
                    raise

            runtime = None
            record = {}
            try:
                with tempfile.TemporaryDirectory(prefix="cute-icl-broker-") as broker:
                    proxy = SubLLMProxy(
                        socket_path=Path(broker) / "broker.sock",
                        max_request_bytes=config.broker_request_bytes,
                    )
                    await proxy.start()
                    try:
                        runtime = NativeRuntime(
                            client,
                            config.model_id,
                            proxy,
                            limits=NativeLimits(
                                max_depth=config.max_depth,
                                max_turns=config.max_turns,
                                max_tokens=config.max_tokens,
                                child_max_tokens=config.max_tokens,
                                max_children=config.max_children,
                                max_live_children=config.max_live_children,
                                max_concurrent_requests=config.max_concurrent_requests,
                                max_model_calls=config.max_model_calls,
                                request_timeout=config.request_timeout,
                                episode_timeout=config.episode_timeout,
                                max_prompt_chars=None,
                            ),
                            budget_policy=ContextBudgetPolicy(
                                server_context_tokens=config.server_context_tokens,
                                output_fraction=1,
                                max_generated_tokens=config.max_generated_tokens,
                                per_call_max_tokens=config.max_tokens,
                                logprobs=config.logprobs,
                            ),
                            prompt_counter=VllmPromptCounter(client, config.server_context_tokens),
                            sampling_args={
                                "temperature": config.temperature,
                                "top_p": config.top_p,
                                "seed": entry["sampling_seed"],
                            },
                            backend_factory=backend_factory,
                            bootstrap_code=(run / entry["bootstrap_path"]).read_text(),
                            system_prompt=prompt["system"],
                            enable_thinking=config.enable_thinking,
                            stream_responses=True,
                            child_repl_at_limit=True,
                            compile_candidate=compile_checked,
                            turn_notice=cute_turn_notice,
                            event_path=directory / "native-events.jsonl",
                            checkpoint_path=directory / "checkpoint.json",
                        )
                        control["runtime"] = runtime
                        record = await runtime.run(
                            prompt["user"],
                            {
                                "task_spec": resources["task_spec"],
                                "condition": entry["condition"],
                            },
                        )
                    finally:
                        try:
                            if runtime is not None:
                                try:
                                    await runtime.close()
                                finally:
                                    if runtime.root is not None:
                                        record = runtime.records[runtime.root.id]
                                        record["child_sessions"] = [
                                            value
                                            for key, value in runtime.records.items()
                                            if key != runtime.root.id
                                        ]
                        finally:
                            await proxy.stop()
                write_json(directory / "trajectory.json", record)
                if infra or feedback.fatal_error is not None:
                    raise RuntimeError("Compile helper infrastructure failed") from (
                        infra[0] if infra else feedback.fatal_error
                    )
                fatal = infrastructure_failure(record)
                if fatal:
                    raise RuntimeError(fatal)
                result["runtime_status"] = record.get("status")
                result["usage"] = usage_summary(record)
                if record.get("status") != "completed":
                    result.update(
                        status="session_failed",
                        error=record.get("error"),
                        error_class=record.get("error_class"),
                        failure=record.get("failure"),
                    )
                else:
                    answer = record.get("final_answer")
                    if not isinstance(answer, str):
                        result.update(
                            status="invalid_source",
                            source_error="Submitted answer must be a string",
                        )
                    else:
                        (directory / "submitted.txt").write_text(answer)
                        try:
                            source = extract_source(answer)
                        except (ValueError, SyntaxError) as error:
                            result.update(status="invalid_source", source_error=str(error))
                        else:
                            (directory / "candidate.py").write_text(source)
                            result.update(
                                status="completed",
                                source_valid=True,
                                source_sha256=sha256(source.encode()),
                            )
            except BaseException as error:
                if record:
                    write_json(directory / "trajectory.json", record)
                result.update(
                    status="infrastructure_error",
                    error={
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
            finally:
                result["finished_at"] = now()
                write_json(directory / "result.json", result)
                summary["results"].append(result)
                write_json(run / "generation_summary.json", summary)
    except BaseException:
        summary["status"] = "aborted_infrastructure"
        raise
    else:
        summary["status"] = "finished"
        write_json(
            run / "GENERATION_FINISHED.json",
            {"finished_at": now(), "attempted": summary["attempted"]},
        )
    finally:
        summary["finished_at"] = now()
        write_json(run / "generation_summary.json", summary)
    return summary


def usage_summary(record: dict) -> dict:
    calls = {
        event["call_id"]: event
        for session in [record, *record.get("child_sessions", [])]
        for event in session.get("events", [])
    }
    result = {}
    for branch in ("root", "child", "plain"):
        events = [event for event in calls.values() if event.get("branch") == branch]
        result[branch] = {}
        for metric in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usages = [event.get("response", {}).get("usage") for event in events]
            values = [usage.get(metric) if isinstance(usage, dict) else None for usage in usages]
            reported = [value for value in values if type(value) is int and value >= 0]
            result[branch][metric] = {
                "sum": sum(reported) if reported or not events else None,
                "reported": len(reported),
                "calls": len(events),
            }
    return result


async def generate(
    run: str | Path,
    base_url: str,
    *,
    sandbox_spec: str | Path,
    repl_python: str | Path,
    receipt: str | Path,
    host_python: str = sys.executable,
    api_key_env: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict:
    """Run all 24 independent roots once. An individual model failure does not stop the batch."""
    from cute_rlm.evaluate import isolated_backend_factory
    from cute_rlm.icl.repl_readiness import check_readiness
    from cute_rlm.server_identity import check_server_identity

    run = Path(run).resolve()
    plan, config = load_plan(run)
    if (run / "GENERATION_STARTED.json").exists():
        raise FileExistsError("Generation already started; no retries or resume")
    spec_path = Path(sandbox_spec).resolve(strict=True)
    wrapper = Path(repl_python).absolute()
    if wrapper.exists():
        raise FileExistsError("REPL wrapper must be a fresh host-created path")
    if api_key_env is not None and not os.environ.get(api_key_env):
        raise ValueError(f"Missing API key environment variable: {api_key_env}")
    check_worker_sources(run, spec_path)
    binding = make_repl_wrapper(run, spec_path, wrapper, host_python)
    await preflight_repl(run, config, wrapper)
    await preflight_candidate_namespace(run, binding, config)
    check_launch_binding(binding)
    headers = {"Authorization": "Bearer " + os.environ[api_key_env]} if api_key_env else {}
    host_failures: list[Exception] = []
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/") + "/",
        timeout=config.request_timeout,
        headers=headers,
        transport=transport,
        event_hooks={"request": [request_guard(run, binding, config, host_failures)]},
    ) as client:
        readiness = check_readiness(run, plan, config, Path(receipt).resolve(strict=True), base_url)
        if readiness["sandbox_spec_sha256"] != binding["worker_spec_sha256"]:
            raise ValueError(
                "Readiness receipt belongs to a different worker sandbox specification"
            )
        metadata = await capture_metadata(client, "GET", "models", run / "models.json")
        if "error" in metadata:
            raise ValueError("Cannot establish server identity; see models.json")
        cards = metadata["response"].get("data", [])
        card = next((item for item in cards if item.get("id") == config.model_id), None)
        if card is None or card.get("max_model_len") != config.server_context_tokens:
            raise ValueError("Server model ID/context differs from frozen configuration")
        await check_server_identity(client, config, readiness)
        marker = {
            "started_at": now(),
            "nonce": uuid.uuid4().hex,
            "base_url": base_url,
            "run": str(run),
            "run_id": plan["run_id"],
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "api_key_env": api_key_env,
            "binding": binding,
            "readiness": readiness,
            "readiness_receipt_sha256": readiness["receipt_sha256"],
            "server_identity": readiness["server_identity"],
            "config_sha256": plan["artifacts"]["config.json"],
            "entry_ids": [entry["id"] for entry in plan["entries"]],
        }
        with (run / "GENERATION_STARTED.json").open("x") as stream:
            json.dump(marker, stream, indent=2)
        return await _generate_entries(
            run,
            plan,
            config,
            client=client,
            worker_command=[
                host_python,
                "-m",
                "cute_rlm.sandbox",
                "--spec",
                binding["candidate_spec"],
            ],
            backend_factory=isolated_backend_factory(config, str(wrapper)),
            binding=binding,
            host_failures=host_failures,
        )


async def validate(
    run: str | Path,
    sandbox_spec: str | Path,
    *,
    model_stopped_receipt: str | Path,
    host_python: str = sys.executable,
) -> dict:
    """Independent numerical validation, explicitly gated on an operator stop receipt."""
    from cute_rlm.icl.repl_readiness import check_model_stopped
    from cute_rlm.icl.validation import validate_candidate
    from cute_rlm.sandbox import SandboxSpec

    run = Path(run).resolve()
    plan, config = load_plan(run)
    if not (run / "GENERATION_FINISHED.json").is_file():
        raise ValueError("Generation must finish before independent validation")
    started = json.loads((run / "GENERATION_STARTED.json").read_text())
    stop = check_model_stopped(
        run, config, started, Path(model_stopped_receipt).resolve(strict=True)
    )
    spec = Path(sandbox_spec).resolve(strict=True)
    if sha256(spec.read_bytes()) != started["binding"]["worker_spec_sha256"]:
        raise ValueError("Validation must use the original generation worker sandbox specification")
    check_worker_sources(run, spec)
    SandboxSpec.read(spec).validate_host()
    if (run / "VALIDATION_STARTED.json").exists():
        raise FileExistsError("Validation already started; no retries or resume")
    with (run / "VALIDATION_STARTED.json").open("x") as stream:
        json.dump(
            {
                "started_at": now(),
                "stop_receipt": stop,
                "stop_receipt_sha256": stop["receipt_sha256"],
                "worker_spec_sha256": sha256(spec.read_bytes()),
            },
            stream,
            indent=2,
        )
    spec_hash = sha256(spec.read_bytes())
    results = []
    for entry in plan["entries"]:
        load_plan(run)
        check_worker_sources(run, spec)
        if sha256(spec.read_bytes()) != spec_hash:
            raise ValueError("Validation sandbox spec changed")
        directory = run / "completions" / entry["id"]
        generation = json.loads((directory / "result.json").read_text())
        if generation["status"] != "completed":
            results.append(
                {
                    "id": entry["id"],
                    "status": "not_submitted",
                    "source_valid": False,
                    "compile_forward": None,
                    "cute_execution_observed": None,
                    "numerical_correct": None,
                    "valid_solution": None,
                    "source_review_required": True,
                }
            )
            continue
        source = (directory / "candidate.py").read_text()
        if sha256(source.encode()) != generation["source_sha256"]:
            raise ValueError(f"Candidate source SHA256 mismatch: {entry['id']}")
        extract_source(source)
        task = json.loads((run / entry["task_path"]).read_text())
        result = await validate_candidate(
            task,
            source,
            run / "validation" / entry["id"],
            worker_command=[host_python, "-m", "cute_rlm.sandbox", "--spec", str(spec)],
        )
        results.append({"id": entry["id"], "source_valid": True, **result})
        write_json(
            run / "validation_summary.json",
            {"results": results, "valid_solution": None, "source_review_required": True},
        )
        if result.get("status") == "infrastructure":
            raise RuntimeError(f"Independent validation infrastructure failed: {entry['id']}")
    summary = {"results": results, "valid_solution": None, "source_review_required": True}
    write_json(run / "validation_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--output", required=True)
    preparation.add_argument("--config", help="JSON config for either pinned model")
    generation = commands.add_parser("generate")
    generation.add_argument("--run", required=True)
    generation.add_argument("--base-url", required=True)
    generation.add_argument("--sandbox-spec", required=True)
    generation.add_argument(
        "--receipt", required=True, help="protected matching GPU readiness evidence"
    )
    generation.add_argument(
        "--repl-python", required=True, help="fresh wrapper path, created by this command"
    )
    generation.add_argument("--host-python", default=sys.executable)
    generation.add_argument("--api-key-env")
    validation = commands.add_parser("validate")
    validation.add_argument("--run", required=True)
    validation.add_argument("--sandbox-spec", required=True)
    validation.add_argument("--model-stopped-receipt", required=True)
    validation.add_argument("--host-python", default=sys.executable)
    args = parser.parse_args()
    if args.command == "prepare":
        config = (
            ReplExperimentConfig(**json.loads(Path(args.config).read_text()))
            if args.config
            else ReplExperimentConfig()
        )
        result = prepare(args.output, config)
    elif args.command == "generate":
        result = asyncio.run(
            generate(
                args.run,
                args.base_url,
                sandbox_spec=args.sandbox_spec,
                repl_python=args.repl_python,
                receipt=args.receipt,
                host_python=args.host_python,
                api_key_env=args.api_key_env,
            )
        )
    else:
        result = asyncio.run(
            validate(
                args.run,
                args.sandbox_spec,
                model_stopped_receipt=args.model_stopped_receipt,
                host_python=args.host_python,
            )
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
