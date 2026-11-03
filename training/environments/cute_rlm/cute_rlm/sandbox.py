"""Fail-closed bubblewrap launcher for Cute verification workers.

The launcher mounts only declared runtime paths, phase inputs, and GPU device
nodes. It deliberately has no host-root mount and does not pass the host
process environment to the worker.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WORKER_MODULE = "cute_rlm.verify_worker"
REPL_ENV_ALLOWLIST = frozenset(
    {
        "RLM_TRAIN_PROXY_URL",
        "RLM_TRAIN_ROLLOUT_ID",
        "RLM_TRAIN_DEPTH",
        "RLM_TRAIN_BROKER_TIMEOUT_S",
        "RLM_TRAIN_EXEC_TIMEOUT_S",
        "RLM_TRAIN_READONLY_CONTEXT",
    }
)
PHASE_ROOT = "/phase"


@dataclass(frozen=True)
class SandboxSpec:
    """Trusted host-side paths and device nodes for one sandbox launcher."""

    bwrap: str
    runtime_python: str
    readonly_bindings: tuple[dict[str, str], ...]
    gpu_devices: tuple[dict[str, str], ...] = ()
    root_broker_socket: str | None = None
    runtime_env_allowlist: tuple[str, ...] = ()
    observer_root: str | None = None
    nsys_binary: str = "nsys"
    runtime_settings: dict[str, str] | None = None
    privileged_setup: bool = False
    sandbox_uid: int = 1000
    sandbox_gid: int = 1000
    spec_path: str | None = None

    @classmethod
    def read(cls, path: str | Path) -> SandboxSpec:
        raw = json.loads(Path(path).read_text())
        return cls(
            bwrap=raw["bwrap"],
            runtime_python=raw["runtime_python"],
            readonly_bindings=tuple(raw["readonly_bindings"]),
            gpu_devices=tuple(raw.get("gpu_devices", [])),
            root_broker_socket=raw.get("root_broker_socket"),
            observer_root=raw.get("observer_root"),
            runtime_settings=raw.get("runtime_settings", {}),
            privileged_setup=raw.get("privileged_setup", False),
            sandbox_uid=raw.get("sandbox_uid", 1000),
            sandbox_gid=raw.get("sandbox_gid", 1000),
            nsys_binary=raw.get("nsys_binary", "nsys"),
            runtime_env_allowlist=tuple(raw.get("runtime_env_allowlist", REPL_ENV_ALLOWLIST)),
            spec_path=str(Path(path).resolve()),
        )

    def validate_host(self) -> None:
        if (
            type(self.privileged_setup) is not bool
            or type(self.sandbox_uid) is not int
            or type(self.sandbox_gid) is not int
            or self.sandbox_uid <= 0
            or self.sandbox_gid <= 0
        ):
            raise ValueError("Namespace setup must drop to explicit non-root UID/GID")
        bwrap = shutil.which(self.bwrap) if not os.path.isabs(self.bwrap) else self.bwrap
        if not bwrap or not os.path.isfile(bwrap) or not os.access(bwrap, os.X_OK):
            raise RuntimeError("configured bubblewrap executable is unavailable")
        python = _absolute(self.runtime_python)
        if not os.path.isfile(python) or not os.access(python, os.X_OK):
            raise RuntimeError("configured runtime Python executable is unavailable")
        if not set(self.runtime_settings or {}) <= {
            "LD_LIBRARY_PATH",
            "CUDA_HOME",
            "CUDA_PATH",
            "CUTE_DSL_CUDA_VERSION",
        }:
            raise ValueError("unsupported trusted runtime setting")
        if any(not isinstance(value, str) for value in (self.runtime_settings or {}).values()):
            raise ValueError("runtime settings must be strings")
        bindings = _bindings(list(self.readonly_bindings), "read-only")
        targets = [target for _, target in bindings]
        for index, target in enumerate(targets):
            if any(
                target == other
                or target.startswith(other + os.sep)
                or other.startswith(target + os.sep)
                for other in targets[index + 1 :]
            ):
                raise ValueError("read-only binding targets cannot overlap")
        for source, _ in bindings:
            if not os.path.exists(source):
                raise RuntimeError(f"read-only binding source is unavailable: {source}")
        if self.observer_root is not None:
            observer = Path(_absolute(self.observer_root)).resolve()
            if any(
                Path(source).resolve() == observer
                or Path(source).resolve() in observer.parents
                or observer in Path(source).resolve().parents
                for source, _ in bindings
            ):
                raise ValueError("protected observer storage cannot be mounted in the sandbox")
        if not set(self.runtime_env_allowlist) <= REPL_ENV_ALLOWLIST:
            raise ValueError("runtime environment allowlist contains unsupported keys")

    def payload(self) -> dict[str, Any]:
        return {
            "bwrap": self.bwrap,
            "python": self.runtime_python,
            "python_target": self.runtime_python,
            "readonly_bindings": list(self.readonly_bindings),
            "gpu_devices": list(self.gpu_devices),
            "root_broker_socket": self.root_broker_socket,
            "runtime_env_allowlist": list(self.runtime_env_allowlist),
            "observer_root": self.observer_root,
            "nsys_binary": self.nsys_binary,
            "runtime_settings": dict(self.runtime_settings or {}),
            "privileged_setup": self.privileged_setup,
            "sandbox_uid": self.sandbox_uid,
            "sandbox_gid": self.sandbox_gid,
        }


def phase_command(spec: SandboxSpec | str | Path, role: str | None = None) -> list[str]:
    """Return the outer supervisor command accepted by ``verifier.launch_phase``."""
    spec_path = spec.spec_path if isinstance(spec, SandboxSpec) else str(Path(spec).resolve())
    if spec_path is None:
        raise ValueError("SandboxSpec must be loaded from a persisted file")
    if role not in {None, "reference", "candidate"}:
        raise ValueError("phase role must be reference or candidate")
    return [sys.executable, "-m", "cute_rlm.sandbox", "--spec", spec_path]


def _absolute(path: str | Path) -> str:
    value = os.fspath(path)
    if not os.path.isabs(value):
        raise ValueError(f"sandbox paths must be absolute: {value}")
    return os.path.normpath(value)


def _bindings(items: list[dict[str, str]], label: str) -> list[tuple[str, str]]:
    result = []
    for binding in items:
        if set(binding) != {"source", "target"}:
            raise ValueError(f"{label} bindings require source and target")
        source, target = _absolute(binding["source"]), _absolute(binding["target"])
        # Resolve existing sources before checking or mounting them; lexical names
        # must not hide a link to an otherwise prohibited host tree.
        source = str(Path(source).resolve())
        if label != "GPU device" and source in {
            "/",
            "/home",
            "/root",
            "/tmp",
            "/var/tmp",
            "/etc",
            "/run",
            "/proc",
            "/dev",
        }:
            raise ValueError("broad host paths cannot be mounted into the sandbox")
        if (
            target in {"/", "/proc", "/dev", "/phase", "/runtime", "/run"}
            or target.startswith("/phase/")
            or target.startswith("/proc/")
            or (label != "GPU device" and target.startswith("/dev/"))
        ):
            raise ValueError(f"unsafe {label} target: {target}")
        result.append((source, target))
    return result


def map_phase_request(request: dict[str, Any]) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """Map host phase directories to a deliberately small in-namespace layout."""
    role = request.get("role")
    if role not in {"reference", "candidate"}:
        raise ValueError("phase role must be reference or candidate")
    mapped = dict(request)
    mounts: list[tuple[str, str]] = []
    output = _absolute(request["output_dir"])
    mounts.append((output, f"{PHASE_ROOT}/output"))
    mapped["output_dir"] = f"{PHASE_ROOT}/output"
    if role == "candidate":
        inputs = _absolute(request["input_dir"])
        if (
            inputs == output
            or inputs.startswith(output + os.sep)
            or output.startswith(inputs + os.sep)
        ):
            raise ValueError("candidate input and output directories must be disjoint")
        mounts.append((inputs, f"{PHASE_ROOT}/inputs"))
        mapped["input_dir"] = f"{PHASE_ROOT}/inputs"
    elif "input_dir" in request:
        raise ValueError("reference phase must not receive candidate inputs")
    return mapped, mounts


def build_command(spec: dict[str, Any], request: dict[str, Any]) -> list[str]:
    """Build (do not execute) the exact bwrap command for one verification phase.

    Spec keys: ``bwrap``, ``python``, ``python_target``, ``ro_bindings`` and
    ``gpu_devices``. Bindings are explicit ``{source,target}`` records. The
    caller must provide all required Python/runtime/module paths and GPU device
    nodes; no system-root or environment fallback is added here.
    """
    bwrap = _absolute(spec["bwrap"])
    python = _absolute(spec.get("runtime_python", spec.get("python")))
    python_target = _absolute(spec.get("python_target", python))
    if not os.path.basename(bwrap).startswith("bwrap"):
        raise ValueError("launcher must be bubblewrap")
    if not python_target.startswith("/") or python_target == "/":
        raise ValueError("python_target must be an absolute file path")
    ro = _bindings(spec.get("readonly_bindings", spec.get("ro_bindings", [])), "read-only")
    devices = _bindings(spec.get("gpu_devices", []), "GPU device")
    _, phase_mounts = map_phase_request(request)
    phase_paths = [Path(request["output_dir"]).resolve()]
    if request["role"] == "candidate":
        phase_paths.append(Path(request["input_dir"]).resolve())
    for source, _ in ro:
        resolved = Path(source).resolve()
        if any(
            resolved == phase or resolved in phase.parents or phase in resolved.parents
            for phase in phase_paths
        ):
            raise ValueError("phase paths cannot overlap runtime bindings")
    command = [
        bwrap,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--hostname",
        "cute-worker",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--perms",
        "1777",
        "--tmpfs",
        "/tmp",
        "--perms",
        "1777",
        "--tmpfs",
        "/dev/shm",
        "--clearenv",
        "--setenv",
        "PATH",
        "/usr/local/cuda-13.3/bin:/usr/bin:/bin",
        "--setenv",
        "HOME",
        "/nonexistent",
        "--setenv",
        "PYTHONPATH",
        "/runtime/python",
    ]
    destinations = {target for _, target in ro + devices + phase_mounts}
    destinations.update({python_target})
    dirs = set()
    for target in destinations:
        parent = os.path.dirname(target)
        while parent not in {"/", ".", "/dev", "/proc"}:
            dirs.add(parent)
            parent = os.path.dirname(parent)
    for directory in sorted(dirs, key=lambda item: (item.count("/"), item)):
        command.extend(("--perms", "0755", "--dir", directory))
    if request["role"] == "candidate":
        command.extend(("--dir", f"{PHASE_ROOT}/inputs"))
    for source, target in ro:
        command.extend(("--ro-bind", source, target))
    if not any(
        python_target == target or python_target.startswith(target + os.sep) for _, target in ro
    ):
        command.extend(("--ro-bind", python, python_target))
    for source, target in devices:
        command.extend(("--dev-bind", source, target))
    for source, target in phase_mounts:
        if target.endswith("/output"):
            command.extend(("--bind", source, target))
        else:
            command.extend(("--ro-bind", source, target))
    for key, value in spec.get("runtime_settings", {}).items():
        if key not in {
            "LD_LIBRARY_PATH",
            "CUDA_HOME",
            "CUDA_PATH",
            "CUTE_DSL_CUDA_VERSION",
        } or not isinstance(value, str):
            raise ValueError("unsupported trusted runtime setting")
        command.extend(("--setenv", key, value))
    command.extend(("--chdir", "/tmp", python_target, "-m", WORKER_MODULE))
    if spec.get("privileged_setup"):
        index = command.index("--unshare-all")
        command[index : index + 1] = [
            "--unshare-pid",
            "--unshare-net",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup",
        ]
        executable = command.index(python_target, command.index("--chdir"))
        command[executable:executable] = [
            "/usr/bin/setpriv",
            "--reuid=" + str(spec["sandbox_uid"]),
            "--regid=" + str(spec["sandbox_gid"]),
            "--clear-groups",
            "--bounding-set=-all",
            "--no-new-privs",
        ]
        command = ["/usr/bin/sudo", "-n", *command]
    return command


def build_repl_command(
    spec: dict[str, Any],
    *,
    broker_socket: str | None = None,
    runtime_code_bindings: list[dict[str, str]] | None = None,
    python_args: list[str] | None = None,
    runtime_env: dict[str, str] | None = None,
) -> list[str]:
    """Build the separate native-REPL wrapper with only broker/code capabilities.

    This mode never inherits environment secrets or mounts a host root. It does
    not attach GPU nodes; the root broker is an explicit read-write socket bind.
    """
    bwrap = _absolute(spec["bwrap"])
    python = _absolute(spec.get("runtime_python", spec.get("python")))
    python_target = _absolute(spec.get("python_target", python))
    runtime_code_bindings = runtime_code_bindings or []
    declared = list(spec.get("readonly_bindings", spec.get("ro_bindings", []))) + list(
        runtime_code_bindings
    )
    code = _bindings(declared, "runtime-code")
    allowed = set(spec.get("runtime_env_allowlist", REPL_ENV_ALLOWLIST))
    if not allowed <= REPL_ENV_ALLOWLIST:
        raise ValueError("runtime environment allowlist contains unsupported keys")
    runtime_env = {key: value for key, value in (runtime_env or {}).items() if key in allowed}
    proxy_url = runtime_env.get("RLM_TRAIN_PROXY_URL", "")
    if not proxy_url.startswith("unix://"):
        raise ValueError("REPL mode requires a Unix-socket root broker URL")
    broker = _absolute(proxy_url.removeprefix("unix://"))
    if not os.path.exists(broker):
        raise FileNotFoundError(f"root broker socket not found: {broker}")
    import stat

    if not stat.S_ISSOCK(os.stat(broker).st_mode):
        raise ValueError("root broker path must be a Unix socket")
    command = [
        bwrap,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--hostname",
        "cute-repl",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--perms",
        "1777",
        "--tmpfs",
        "/tmp",
        "--perms",
        "1777",
        "--tmpfs",
        "/dev/shm",
        "--clearenv",
        "--setenv",
        "PATH",
        "/usr/local/cuda-13.3/bin:/usr/bin:/bin",
        "--setenv",
        "HOME",
        "/nonexistent",
        "--setenv",
        "PYTHONPATH",
        "/runtime/python",
    ]
    destinations = {target for _, target in code}
    python_covered = any(
        python_target == target or python_target.startswith(target + os.sep) for _, target in code
    )
    if not python_covered:
        destinations.add(python_target)
    for index, target in enumerate(sorted(destinations)):
        if any(
            target == other
            or target.startswith(other + os.sep)
            or other.startswith(target + os.sep)
            for other in sorted(destinations)[index + 1 :]
        ):
            raise ValueError("runtime binding targets cannot overlap")
    dirs = set()
    for target in destinations:
        parent = os.path.dirname(target)
        while parent not in {"/", ".", "/dev", "/proc"}:
            dirs.add(parent)
            parent = os.path.dirname(parent)
    for directory in sorted(dirs, key=lambda item: (item.count("/"), item)):
        command.extend(("--perms", "0755", "--dir", directory))
    for source, target in code:
        command.extend(("--ro-bind", source, target))
    if not python_covered:
        command.extend(("--ro-bind", python, python_target))
    command.extend(("--perms", "0755", "--dir", "/run"))
    command.extend(("--bind", broker, "/run/rlm-root-broker.sock"))
    runtime_env["RLM_TRAIN_PROXY_URL"] = "unix:///run/rlm-root-broker.sock"
    for key, value in runtime_env.items():
        command.extend(("--setenv", key, value))
    command.extend(("--chdir", "/tmp", python_target, *(python_args or [])))
    if spec.get("privileged_setup"):
        index = command.index("--unshare-all")
        command[index : index + 1] = [
            "--unshare-pid",
            "--unshare-net",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup",
        ]
        executable = command.index(python_target, command.index("--chdir"))
        command[executable:executable] = [
            "/usr/bin/setpriv",
            "--reuid=" + str(spec["sandbox_uid"]),
            "--regid=" + str(spec["sandbox_gid"]),
            "--clear-groups",
            "--bounding-set=-all",
            "--no-new-privs",
        ]
        command = ["/usr/bin/sudo", "-n", *command]
    return command


def _verify_namespace_support(bwrap: str) -> None:
    if not os.path.isfile(bwrap) or not os.access(bwrap, os.X_OK):
        raise RuntimeError("configured bubblewrap executable is unavailable")


def _run(spec: dict[str, Any], request: dict[str, Any]) -> int:
    executable = shutil.which(spec["bwrap"]) if not os.path.isabs(spec["bwrap"]) else spec["bwrap"]
    if not executable or not os.path.isfile(executable) or not os.access(executable, os.X_OK):
        raise RuntimeError("configured bubblewrap executable is unavailable")
    if request.get("role") not in {"reference", "candidate"}:
        raise ValueError("invalid worker phase")
    original_output = Path(_absolute(request["output_dir"]))
    if original_output.exists():
        raise RuntimeError("worker output directory must not already exist")
    input_dir = request.get("input_dir")
    if input_dir and not Path(input_dir).is_dir():
        raise RuntimeError(f"phase input directory does not exist: {input_dir}")
    devices = spec.get("gpu_devices", [])
    if not devices:
        raise RuntimeError("no GPU device nodes were configured")
    for device in devices:
        if not os.path.exists(device["source"]):
            raise RuntimeError(f"required GPU device is unavailable: {device['source']}")
    _verify_namespace_support(executable)
    with tempfile.TemporaryDirectory(prefix="cute-phase-") as staging:
        staged_output = Path(staging) / "output"
        staged_output.mkdir(mode=0o700)
        staged_request = {**request, "output_dir": str(staged_output)}
        command = build_command({**spec, "bwrap": executable}, staged_request)
        mapped, _ = map_phase_request(staged_request)
        env = {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"}
        if request["role"] == "candidate" and spec.get("observer_root"):
            import hashlib
            import sqlite3

            profile_command = command.copy()
            profile_command[-1] = "cute_rlm.profile_worker"
            profile_request = {
                **mapped,
                "profile_compile": True,
                "nsys_inside": spec["nsys_binary"],
            }
            profiled = subprocess.run(
                profile_command,
                input=json.dumps(profile_request).encode(),
                capture_output=True,
                env=env,
                check=False,
            )
            if profiled.returncode:
                sys.stdout.buffer.write(profiled.stdout)
                sys.stderr.buffer.write(profiled.stderr)
                return profiled.returncode
            capture_root = Path(spec["observer_root"]) / request["request_id"]
            capture_root.mkdir(parents=True, mode=0o700, exist_ok=False)
            for name in ("trace.sqlite", "trace.nsys-rep"):
                source = staged_output / "nsight" / name
                if not source.is_file() or source.is_symlink():
                    raise RuntimeError("profiler output must be a regular file")
                shutil.copyfile(source, capture_root / name)
            compiler = staged_output / "worker"
            provenance = compiler / "compile-provenance.json"
            (capture_root / "compiler").mkdir()
            if provenance.is_file() and not provenance.is_symlink():
                shutil.copyfile(provenance, capture_root / "compiler/compile-provenance.json")
                artifacts = compiler / "compile-artifacts"
                if artifacts.is_dir() and not artifacts.is_symlink():
                    destination = capture_root / "compiler/compile-artifacts"
                    destination.mkdir()
                    for artifact in artifacts.iterdir():
                        if artifact.is_symlink() or not artifact.is_file():
                            raise RuntimeError("compiler artifacts must be regular files")
                        shutil.copyfile(artifact, destination / artifact.name)
            db = sqlite3.connect(f"file:{capture_root / 'trace.sqlite'}?mode=ro", uri=True)
            try:
                pids = [
                    row[0]
                    for row in db.execute(
                        "SELECT DISTINCT globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL"
                    )
                ]
            finally:
                db.close()
            if len(pids) != 1:
                raise RuntimeError("profile must contain exactly one CUDA process")
            receipt = {
                "request_id": request["request_id"],
                "candidate_sha256": request["candidate_sha256"],
                "config_sha256": request["config_sha256"],
                "process_id": pids[0],
                "start_ns": 0,
                "end_ns": 2**63 - 1,
                "trace_sha256": hashlib.sha256(
                    (capture_root / "trace.sqlite").read_bytes()
                ).hexdigest(),
                "scope": "in-namespace-observational; not resistant to malicious candidate Python",
            }
            (capture_root / "capture.json").write_text(json.dumps(receipt))
            # Run a fresh, uninstrumented process for correctness and timing.
            shutil.rmtree(staged_output)
            staged_output.mkdir(mode=0o700)
        process = subprocess.run(
            command, input=json.dumps(mapped).encode(), capture_output=True, env=env, check=False
        )
        if process.returncode == 0:
            if not staged_output.is_dir() or staged_output.is_symlink():
                raise RuntimeError("sandbox worker did not create a regular output directory")
            for path in staged_output.rglob("*"):
                if path.is_symlink() or not path.is_file():
                    raise RuntimeError(
                        "worker outputs must be regular files, without links or directories"
                    )
            os.replace(staged_output, original_output)
        sys.stdout.buffer.write(process.stdout)
        sys.stderr.buffer.write(process.stderr)
        return process.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="trusted launcher specification JSON")
    parser.add_argument("--repl", action="store_true")
    parser.add_argument("passthrough", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        sandbox_spec = SandboxSpec.read(args.spec)
        sandbox_spec.validate_host()
        spec = sandbox_spec.payload()
        if args.repl:
            tail = args.passthrough[1:] if args.passthrough[:1] == ["--"] else args.passthrough
            allowed_env = {
                key: os.environ[key]
                for key in sandbox_spec.runtime_env_allowlist
                if key in os.environ
            }
            command = build_repl_command(spec, python_args=tail, runtime_env=allowed_env)
            process = subprocess.run(
                command, env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"}, check=False
            )
            return process.returncode
        request = json.load(sys.stdin)
        return _run(spec, request)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"sandbox infrastructure error: {error}", file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
