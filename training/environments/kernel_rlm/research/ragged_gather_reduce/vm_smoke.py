"""Operator-only L40S smoke for the fixed author controls (not a candidate runner).

Use the module from the repository root. ``--preflight`` does not launch any
processes or query hardware. The live flags attest separate approval and an
exclusive disposable Linux VM with a trusted checkout, OS and CUDA toolkit.
No timing, inference, or arbitrary source input. Exit 0 for preflight or live
success, 1 for a failed check, 2 for invalid CLI arguments. Stdout is one JSON
receipt on completion; retain it in operator-controlled storage. It is not signed.
The device probe requires ``nvidia-smi`` to provide CUDA capability and process
occupancy; it fails closed on unsupported output. Process listings cannot prove
exclusive VM allocation. Binary pins do not cover transitive toolkit/OS files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import selectors
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .reference import WireFixture, development_fixtures, public_fixture
from .score import MAX_OBSERVATION_BYTES, decode_observation, score_capture

SCHEMA = "ragged_author_controls_v1"
OFFLINE_PREFLIGHT = "offline_preflight"
AUTHOR_CONTROLS_PASSED = "author_controls_passed"
AUTHOR_CONTROLS_FAILED = "author_controls_failed"
SOURCE_SHA256 = {
    "contract.h": "daf917c7f80e6aa24f1599a4d1e1c3f120e9b97c42381ebfe73c959f2a93beba",
    "device_check.cu": "6e86cc3f8bfc800db24ca42fa2c24a9d4d9499456d47c64894250de33a3f191a",
    "control.cu": "e528f27adcc89f04426918f0fa1591ac86e4449654fc5593f38719938b495085",
    "wrong_offset.cu": "db0d71d37c99b2b9be0ce7b3c8b6a93f5e39948e79716000581d9b08f9306306",
}
TOOL_NAMES = ("nvcc", "host-cxx", "nvidia-smi")
ARCH_FLAG = "--generate-code=arch=compute_89,code=sm_89"
KNOWN_WRONG = ((7,), (1,), (7,), (1,))
GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}")
BUILD_TIMEOUT = 180
CONTROL_TIMEOUT = 30
PROBE_TIMEOUT = 10
DIAGNOSTIC_CAP = 64 * 1024


class SmokeFailure(Exception):
    def __init__(self, stage: str, reason: str):
        super().__init__(f"{stage}: {reason}")
        self.stage = stage
        self.reason = reason


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ProcessResult:
    status: str
    returncode: int | None = None
    stdout: bytes = b""
    stderr: bytes = b""

    @property
    def ok(self) -> bool:
        return self.status == "exited" and self.returncode == 0

    def receipt(self) -> dict:
        return {
            "status": self.status,
            "returncode": self.returncode,
            "stdout_sha256": sha256(self.stdout),
            "stderr_sha256": sha256(self.stderr),
            "stdout_bytes": len(self.stdout),
            "stderr_bytes": len(self.stderr),
        }


def run_process(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    stdin: bytes = b"",
    stdout_cap: int = DIAGNOSTIC_CAP,
) -> ProcessResult:
    """Bound captured bytes and kill the process group if the deadline/cap is hit."""
    out, err = bytearray(), bytearray()
    status = "exited"
    deadline = time.monotonic() + timeout
    with tempfile.TemporaryFile(dir=cwd) as input_file, selectors.DefaultSelector() as io:
        input_file.write(stdin)
        input_file.seek(0)
        try:
            proc = subprocess.Popen(
                argv,
                cwd=cwd,
                env=env,
                stdin=input_file,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError:
            return ProcessResult("spawn_error")
        with proc:
            try:
                assert proc.stdout is not None and proc.stderr is not None
                io.register(proc.stdout, selectors.EVENT_READ, (out, stdout_cap))
                io.register(proc.stderr, selectors.EVENT_READ, (err, DIAGNOSTIC_CAP))
                while io.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        status = "timeout"
                        break
                    for key, _ in io.select(remaining):
                        buf, cap = key.data
                        room = cap - len(buf)
                        data = os.read(key.fd, min(65536, room + 1))
                        if not data:
                            io.unregister(key.fileobj)
                            continue
                        buf.extend(data[:room])
                        if len(data) > room:
                            status = "output_limit"
                            break
                    if status != "exited":
                        break
                if status == "exited":
                    proc.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                status = "timeout"
            except OSError:
                status = "io_error"
            finally:
                if status != "exited" or proc.returncode is None:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
    return ProcessResult(status, proc.returncode, bytes(out), bytes(err))


def verified_sources(summary: dict) -> dict[str, bytes]:
    payloads = {}
    directory = Path(__file__).parent
    for name, expected in SOURCE_SHA256.items():
        # A checked source snapshot, not a path the live compiler can re-read.
        with (directory / name).open("rb") as fh:
            data = fh.read(DIAGNOSTIC_CAP + 1)
        observed = sha256(data)
        summary["sources"][name] = observed
        if len(data) > DIAGNOSTIC_CAP or observed != expected:
            raise SmokeFailure("sources", f"source pin mismatch: {name}")
        payloads[name] = data
    for name in ("reference.py", "score.py"):
        summary["sources"][name] = sha256((directory / name).read_bytes())
    return payloads


def verified_tools(args: argparse.Namespace, summary: dict) -> dict[str, str]:
    tools = {}
    for key in TOOL_NAMES:
        dest = key.replace("-", "_")
        entered = getattr(args, dest)
        path = Path(entered).resolve(strict=True)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise SmokeFailure("toolchain", f"{key} is not an executable file")
        with path.open("rb") as fh:
            digest = hashlib.file_digest(fh, "sha256").hexdigest()
        if digest != getattr(args, dest + "_sha256"):
            raise SmokeFailure("toolchain", f"{key} binary pin mismatch")
        tools[key] = str(path)
        summary["toolchain"][key] = {"path": entered, "sha256": digest}
    return tools


def probe(stage: str, argv: list[str], cwd: Path, env: dict[str, str], receipt: dict) -> str:
    result = run_process(argv, cwd=cwd, env=env, timeout=PROBE_TIMEOUT, stdout_cap=4096)
    receipt.update(result.receipt())
    if not result.ok:
        raise SmokeFailure(stage, "tool probe failed")
    try:
        return result.stdout.decode("ascii").strip()
    except UnicodeError as e:
        raise SmokeFailure(stage, "non-ASCII tool response") from e


def select_l40s(smi: str, uuid: str, cwd: Path, env: dict[str, str], device: dict) -> None:
    query = [smi, "--id=" + uuid, "--format=csv,noheader,nounits"]
    row = probe(
        "device",
        [*query, "--query-gpu=name,uuid,driver_version,compute_cap"],
        cwd,
        env,
        device.setdefault("query_process", {}),
    )
    fields = [part.strip() for part in row.split(",")]
    if len(fields) != 4 or "\n" in row or "\r" in row:
        raise SmokeFailure("device", "expected one NVIDIA-SMI GPU row")
    name, observed_uuid, driver, compute_cap = fields
    device.update(name=name, uuid=observed_uuid, driver_version=driver, compute_cap=compute_cap)
    if (
        name not in ("NVIDIA L40S", "L40S")
        or observed_uuid.lower() != uuid.lower()
        or compute_cap != "8.9"
        or re.fullmatch(r"[0-9]{2,4}(?:\.[0-9]{1,4}){1,3}", driver) is None
    ):
        raise SmokeFailure("device", "expected the selected NVIDIA L40S, CUDA capability 8.9")
    require_idle(query, cwd, env, device, "idle_before_build_process")


def require_idle(query: list[str], cwd: Path, env: dict[str, str], device: dict, key: str) -> None:
    apps = probe(
        "device", [*query, "--query-compute-apps=pid"], cwd, env, device.setdefault(key, {})
    )
    if apps:
        raise SmokeFailure("device", "selected GPU is busy or compute occupancy is unknown")


def run_builds(tools: dict[str, str], cwd: Path, env: dict[str, str], summary: dict) -> None:
    for key in ("nvcc", "host-cxx"):
        tool = summary["toolchain"][key]
        version = probe(
            "toolchain",
            [tools[key], "--version"],
            cwd,
            env,
            tool.setdefault("version_process", {}),
        )
        if not version:
            raise SmokeFailure("toolchain", f"{key} version is empty")
        tool["version"] = version

    for control in ("control", "wrong_offset"):
        exe = f"device_check_{control}"
        flags = ["-std=c++17", "-O2", ARCH_FLAG, "--cudart=static"]
        sources = [f"{control}.cu", "device_check.cu", "contract.h"]
        result = run_process(
            [tools["nvcc"], *flags, "-ccbin", tools["host-cxx"], *sources[:2], "-o", exe],
            cwd=cwd,
            env=env,
            timeout=BUILD_TIMEOUT,
        )
        build = {
            "artifact": exe,
            "flags": flags,
            "source_sha256": {name: summary["sources"][name] for name in sources},
            "process": result.receipt(),
        }
        summary["builds"][control] = build
        output = cwd / exe
        if (
            not result.ok
            or output.is_symlink()
            or not output.is_file()
            or not os.access(output, os.X_OK)
        ):
            raise SmokeFailure("build", f"{control} did not produce an executable")
        with output.open("rb") as fh:
            artifact = fh.read(32 * 1024 * 1024 + 1)
        if not artifact or len(artifact) > 32 * 1024 * 1024:
            raise SmokeFailure("build", f"{control} executable size outside bounds")
        build.update(status="built", sha256=sha256(artifact), size_bytes=len(artifact))


def controls_plan() -> list[tuple[str, WireFixture, str]]:
    good = [("control", f, "match") for f in (public_fixture(), *development_fixtures())]
    wrong = next(f for _, f, _ in good if f.name == "mask-offset")
    return [*good, ("wrong_offset", wrong, "numerical_mismatch")]


def run_controls(
    plan: list[tuple[str, WireFixture, str]], cwd: Path, env: dict[str, str], summary: dict
) -> None:
    verified = set()
    for control, fixture, expected in plan:
        build = summary["builds"][control]
        exe = build["artifact"]
        if control not in verified:
            artifact = cwd / exe
            with artifact.open("rb") as fh:
                data = fh.read(build["size_bytes"] + 1)
            if artifact.is_symlink() or sha256(data) != build["sha256"]:
                raise SmokeFailure("capture", f"{control} executable changed after build")
            verified.add(control)
        wire = fixture.to_wire().encode("ascii")
        result = run_process(
            ["./" + exe],
            cwd=cwd,
            env=env,
            timeout=CONTROL_TIMEOUT,
            stdin=wire,
            stdout_cap=MAX_OBSERVATION_BYTES,
        )
        observed = "process_error"
        if result.ok:
            score = score_capture(fixture, result.stdout, returncode=0)
            observed = "numerical_mismatch" if score.status == "mismatch" else score.status
        case = {
            "control": control,
            "fixture": fixture.name,
            "expected": expected,
            "observed": observed,
            "input_sha256": sha256(wire),
            "input_bytes": len(wire),
            "artifact_sha256": build["sha256"],
            "process": result.receipt(),
        }
        summary["cases"].append(case)
        if observed != expected:
            raise SmokeFailure(
                "capture", f"{control}/{fixture.name}: {observed}, wanted {expected}"
            )
        if (
            expected == "numerical_mismatch"
            and decode_observation(result.stdout, fixture.rows) != KNOWN_WRONG
        ):
            raise SmokeFailure(
                "capture", "negative control did not produce the exact wrong-offset result twice"
            )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight", action="store_true", help="check plans and pins without subprocesses"
    )
    mode.add_argument(
        "--run-author-controls", action="store_true", help="enable the approved live GPU smoke"
    )
    parser.add_argument("--confirm-exclusive-disposable-l40s", action="store_true")
    parser.add_argument("--gpu-uuid")
    for name in TOOL_NAMES:
        parser.add_argument(f"--{name}", metavar="RELATIVE_EXECUTABLE")
        parser.add_argument(f"--{name}-sha256", metavar="SHA256")
    args = parser.parse_args(argv)
    if args.run_author_controls:
        if not args.confirm_exclusive_disposable_l40s or not GPU_UUID.fullmatch(
            args.gpu_uuid or ""
        ):
            parser.error("live smoke requires exclusive disposable VM approval and a full GPU UUID")
    elif args.confirm_exclusive_disposable_l40s or args.gpu_uuid is not None:
        parser.error("approval and GPU selection are live-only")

    tool_args = tuple(
        getattr(args, option.replace("-", "_"))
        for name in TOOL_NAMES
        for option in (name, name + "-sha256")
    )
    if args.run_author_controls or any(arg is not None for arg in tool_args):
        for entered, digest in zip(tool_args[::2], tool_args[1::2], strict=True):
            if not entered or Path(entered).is_absolute() or len(entered) > 512:
                parser.error(
                    "supply all three executable paths relative to the checkout working directory"
                )
            if re.fullmatch(r"[0-9a-f]{64}", digest or "") is None:
                parser.error("supply a lowercase SHA-256 binary pin for each executable")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = {
        "schema": SCHEMA,
        "mode": "live" if args.run_author_controls else "offline",
        "status": AUTHOR_CONTROLS_FAILED,
        "started_at": datetime.now(UTC).isoformat(),
        "sources": {},
        "toolchain": {},
        "device": {},
        "builds": {},
        "plan": [],
        "cases": [],
        "error": None,
    }
    try:
        files = verified_sources(summary)
        tools = verified_tools(args, summary) if args.nvcc else {}
        plan = controls_plan()
        summary["plan"] = [
            {"control": control, "fixture": fixture.name, "expected": expected}
            for control, fixture, expected in plan
        ]
        if args.run_author_controls:
            with tempfile.TemporaryDirectory(prefix="ragged-author-controls-") as directory:
                cwd = Path(directory)
                env = {"PATH": os.defpath, "HOME": directory, "TMPDIR": directory, "LC_ALL": "C"}
                select_l40s(tools["nvidia-smi"], args.gpu_uuid, cwd, env, summary["device"])
                for name, data in files.items():
                    (cwd / name).write_bytes(data)
                env["CUDA_VISIBLE_DEVICES"] = summary["device"]["uuid"]
                run_builds(tools, cwd, env, summary)
                require_idle(
                    [tools["nvidia-smi"], "--id=" + args.gpu_uuid, "--format=csv,noheader,nounits"],
                    cwd,
                    env,
                    summary["device"],
                    "idle_before_capture_process",
                )
                run_controls(plan, cwd, env, summary)
        summary["status"] = (
            AUTHOR_CONTROLS_PASSED if args.run_author_controls else OFFLINE_PREFLIGHT
        )
    except SmokeFailure as e:
        summary["error"] = {"stage": e.stage, "reason": e.reason}
    except (OSError, RuntimeError) as e:
        # Don't include uncontrolled host paths, environment or tool output in the receipt.
        summary["error"] = {"stage": "os", "reason": type(e).__name__}
    summary["completed_at"] = datetime.now(UTC).isoformat()
    print(json.dumps(summary, sort_keys=True))
    return int(summary["status"] == AUTHOR_CONTROLS_FAILED)


if __name__ == "__main__":
    raise SystemExit(main())
