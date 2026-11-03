"""Read-only Nsight Systems SQLite kernel-launch observations.

CUDA activity alone does not prove CuTe compilation. Compilation is observed only
when retained artifacts match a kernel launched inside the compiled-call and
model-forward ranges. This is non-adversarial observation, not an attestation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any


class ObserverError(ValueError):
    """The protected capture cannot support an identity-bound observation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}


def parse_nsys_sqlite(
    trace_path: str | Path, *, process_id: int, start_ns: int, end_ns: int
) -> list[dict[str, Any]]:
    """Return CUDA kernel activities in the given trusted process/time window.

    Supports Nsight Systems SQLite exports exposing CUPTI_ACTIVITY_KIND_KERNEL
    with start/end timestamps and either a string name column or StringIds.
    Input is read-only; unsupported or ambiguous schemas fail closed.
    """
    path = Path(trace_path).resolve(strict=True)
    if (
        type(process_id) is not int
        or process_id <= 0
        or type(start_ns) is not int
        or type(end_ns) is not int
        or start_ns < 0
        or end_ns <= start_ns
    ):
        raise ObserverError("invalid capture process/time bounds")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        table = "CUPTI_ACTIVITY_KIND_KERNEL"
        if table not in tables:
            raise ObserverError("Nsight SQLite has no CUPTI kernel activity table")
        cols = _table_columns(connection, table)
        required = {"start", "end", "globalPid"}
        if not required <= cols:
            raise ObserverError("kernel activity schema lacks process and timestamp fields")
        name_col = next(
            (name for name in ("demangledName", "shortName", "name") if name in cols), None
        )
        select = ["start", "end", "globalPid"]
        if name_col:
            select.append(f'"{name_col}"')
        elif "nameId" in cols and "StringIds" in tables:
            strings = _table_columns(connection, "StringIds")
            if not {"id", "value"} <= strings:
                raise ObserverError("unsupported StringIds schema")
            select.append("nameId")
        else:
            raise ObserverError("kernel activity schema has no resolvable name")
        rows = connection.execute(
            f'SELECT {", ".join(select)} FROM "{table}" WHERE globalPid=? AND start>=? AND end<=? AND end>start',
            (process_id, start_ns, end_ns),
        ).fetchall()
        result = []
        for row in rows:
            start, end, pid, name = row
            if isinstance(name, str):
                kernel_name = name
            elif type(name) is int and "StringIds" in tables:
                found = connection.execute(
                    "SELECT value FROM StringIds WHERE id=?", (name,)
                ).fetchone()
                if not found or not isinstance(found[0], str):
                    raise ObserverError("unresolved kernel name ID")
                kernel_name = found[0]
            else:
                raise ObserverError("kernel name is not text or a resolvable StringIds reference")
            result.append(
                {"process_id": pid, "start_ns": start, "end_ns": end, "kernel_name": kernel_name}
            )
        return result
    except sqlite3.DatabaseError as error:
        raise ObserverError(f"invalid Nsight SQLite database: {error}") from error
    finally:
        connection.close()


def correlate_cute_artifacts(trace_path: Path, artifact_root: Path, request_id: str) -> dict:
    """Correlate retained compiler artifacts with exact profiled call ranges.

    Evidence is observational for benchmark candidates, not an attestation against
    malicious Python that replaces the compiler, NVTX, or serialization APIs.
    """
    provenance_path = artifact_root / "compile-provenance.json"
    if not provenance_path.is_file() or provenance_path.is_symlink():
        return {"observed": False, "reason": "compile artifacts were not recorded"}
    provenance = json.loads(provenance_path.read_text())
    db = sqlite3.connect(f"file:{trace_path}?mode=ro", uri=True)
    try:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL", "StringIds"} <= tables:
            return {"observed": False, "reason": "NVTX/kernel schema unavailable"}
        strings = dict(db.execute("SELECT id,value FROM StringIds"))
        ranges = [
            (start, end, text or strings.get(text_id), tid)
            for start, end, text, text_id, tid in db.execute(
                "SELECT start,end,text,textId,globalTid FROM NVTX_EVENTS WHERE end>start"
            )
        ]
        forwards = [r for r in ranges if r[2] == "cute_forward:" + request_id]
        if not forwards:
            return {"observed": False, "reason": "candidate forward range not captured"}
        columns = _table_columns(db, "CUPTI_ACTIVITY_KIND_KERNEL")
        name_column = "mangledName" if "mangledName" in columns else "demangledName"
        kernels = db.execute(
            f'SELECT start,end,globalPid,correlationId,"{name_column}" FROM CUPTI_ACTIVITY_KIND_KERNEL'
        ).fetchall()
        runtime = {}
        if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
            for start, end, tid, correlation in db.execute(
                "SELECT start,end,globalTid,correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
            ):
                runtime[correlation] = (start, end, tid)
        artifacts = {a["id"]: a for a in provenance.get("artifacts", [])}
        matched = []
        for invocation in provenance.get("invocations", []):
            if invocation.get("status") != "completed":
                continue
            artifact = artifacts.get(invocation.get("artifact_id"), {})
            payloads = {}
            for kind in ("ptx", "cubin"):
                relative = artifact.get(kind + "_path")
                if not isinstance(relative, str):
                    continue
                path = (artifact_root / relative).resolve()
                if (
                    not path.is_relative_to(artifact_root.resolve())
                    or path.is_symlink()
                    or not path.is_file()
                ):
                    raise ObserverError("compile artifact path escapes protected capture")
                blob = path.read_bytes()
                if hashlib.sha256(blob).hexdigest() != artifact.get(kind + "_sha256"):
                    raise ObserverError("compiler artifact hash mismatch")
                payloads[kind] = blob
            if not payloads.get("ptx") or not payloads.get("cubin"):
                continue
            entries = set(
                re.findall(r"\.entry\s+([A-Za-z_$][A-Za-z0-9_.$]*)", payloads["ptx"].decode())
            )
            for call_range in [r for r in ranges if r[2] == invocation.get("range_name")]:
                if not any(
                    f[0] <= call_range[0] and call_range[1] <= f[1] and call_range[3] == f[3]
                    for f in forwards
                ):
                    continue
                for start, end, pid, correlation, name in kernels:
                    symbol = strings.get(name, name) if not isinstance(name, str) else name
                    launch = runtime.get(correlation)
                    if (
                        symbol in entries
                        and call_range[0] <= start <= end <= call_range[1]
                        and launch is not None
                        and call_range[0] <= launch[0] <= call_range[1]
                        and launch[2] == call_range[3]
                        and (pid >> 24) == (launch[2] >> 24)
                    ):
                        matched.append(
                            {
                                "artifact_id": artifact["id"],
                                "ptx_sha256": artifact["ptx_sha256"],
                                "cubin_sha256": artifact["cubin_sha256"],
                                "kernel_symbol": symbol,
                                "range_name": call_range[2],
                                "correlation_id": correlation,
                                "nsys_global_pid": pid,
                                "kernel_start_ns": start,
                                "kernel_end_ns": end,
                            }
                        )
        return {
            "observed": bool(matched),
            "matches": matched,
            "scope": "compiler artifact + exact forward/compiled-call NVTX + CUPTI correlation; non-adversarial observation",
        }
    finally:
        db.close()


class NsightObserver:
    """Protected, request-scoped observation callback for ``verify_submission``.

    The collector must write immutable capture.json beside trace.sqlite outside
    candidate-writable storage. Its capture record binds request/source/config,
    process ID and the time interval. This adapter checks those bindings and
    reports launch evidence and checks the separate retained compiler artifacts.
    """

    def __init__(self, evidence_root: str | Path):
        self.evidence_root = Path(evidence_root).resolve(strict=True)

    async def __call__(
        self, request_id: str, candidate_sha256: str, config_sha256: str
    ) -> dict[str, Any]:
        if (
            not re.fullmatch(r"[0-9a-f]{32}", request_id)
            or len(candidate_sha256) != 64
            or len(config_sha256) != 64
        ):
            raise ObserverError("invalid requested observation identity")
        receipt_path = self.evidence_root / request_id / "capture.json"
        trace_path = self.evidence_root / request_id / "trace.sqlite"
        try:
            receipt = json.loads(receipt_path.read_text())
        except (OSError, ValueError) as error:
            raise ObserverError("protected capture receipt missing or invalid") from error
        expected = {
            "request_id": request_id,
            "candidate_sha256": candidate_sha256,
            "config_sha256": config_sha256,
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ObserverError("protected capture identity mismatch")
        if receipt.get("trace_sha256") != _sha256(trace_path):
            raise ObserverError("Nsight trace digest mismatch")
        kernels = parse_nsys_sqlite(
            trace_path,
            process_id=receipt["process_id"],
            start_ns=receipt["start_ns"],
            end_ns=receipt["end_ns"],
        )
        compile_evidence = correlate_cute_artifacts(
            trace_path, receipt_path.parent / "compiler", request_id
        )
        return {
            **expected,
            "compiled": compile_evidence["observed"],
            "launched": bool(kernels),
            "evidence": {
                "trace": str(trace_path),
                "trace_sha256": receipt["trace_sha256"],
                "kernel_count": len(kernels),
                "kernels": kernels,
                "scope": "CUDA kernel activity in identity-bound captured process; CuTe compilation is not established",
            },
            "cuda_launch": {
                "observed": bool(kernels),
                "kernel_count": len(kernels),
                "kernels": kernels,
            },
            "cute_compile": compile_evidence,
        }


async def collect_nsight(
    command: list[str],
    request: dict[str, Any],
    evidence_root: str | Path,
    *,
    nsys_binary: str = "nsys",
) -> dict[str, Any]:
    """Run a worker under Nsight and export a protected SQLite capture.

    ``command`` must itself be the trusted isolation launcher (e.g. the external
    bwrap wrapper), never the candidate command directly. The request binds the
    captured candidate and verification config. Collection outputs are created
    under a fresh evidence directory, separate from candidate writable paths.
    The collector records the Nsight globalPid from the exported kernel rows; it
    deliberately does not equate it with the host/Linux PID.

    This is an adapter for GPU-host integration, not a sandbox implementation.
    Actual CLI flags/schema and CUPTI permissions need validation on the pinned
    Nsight/GPU host before use.
    """
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(arg, str) for arg in command)
    ):
        raise ObserverError("explicit trusted launcher argv is required")
    request_id = request.get("request_id")
    candidate_hash = request.get("candidate_sha256")
    config_hash = request.get("config_sha256")
    if not isinstance(request_id, str) or not re.fullmatch(r"[0-9a-f]{32}", request_id):
        raise ObserverError("invalid request_id")
    if (
        not isinstance(candidate_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", candidate_hash)
        or not isinstance(config_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", config_hash)
    ):
        raise ObserverError("invalid candidate/config identity hashes")
    binary = shutil.which(nsys_binary)
    if binary is None:
        raise ObserverError(f"Nsight Systems executable not found: {nsys_binary}")
    root = Path(evidence_root).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    capture = root / request_id
    capture.mkdir(mode=0o700, exist_ok=False)
    native = capture / "trace.nsys-rep"
    sqlite_path = capture / "trace.sqlite"
    prefix = capture / "profile"
    started = time.monotonic_ns()
    profile_cmd = [
        binary,
        "profile",
        "--trace=cuda-sw,nvtx",
        "--sample=none",
        "--force-overwrite=true",
        f"--output={prefix}",
        *command,
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *profile_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await process.communicate(json.dumps(request).encode())
        except asyncio.CancelledError:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise
    except OSError as error:
        raise ObserverError(f"failed to launch Nsight Systems: {error}") from error
    ended = time.monotonic_ns()
    if process.returncode != 0:
        raise ObserverError(
            f"Nsight worker failed ({process.returncode}): {stderr.decode(errors='replace')[-2000:]}"
        )
    # Nsight may append .nsys-rep to the supplied output prefix.
    if not native.exists():
        native = Path(str(prefix) + ".nsys-rep")
    if not native.is_file():
        raise ObserverError("Nsight produced no native report")
    export = await asyncio.create_subprocess_exec(
        binary,
        "export",
        "--type",
        "sqlite",
        "--output",
        str(sqlite_path),
        str(native),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, export_stderr = await export.communicate()
    if export.returncode != 0 or not sqlite_path.is_file():
        raise ObserverError(
            f"Nsight SQLite export failed: {export_stderr.decode(errors='replace')[-2000:]}"
        )
    # Inspect only exported globalPid values. No host PID equivalence assumption.
    db = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        if "CUPTI_ACTIVITY_KIND_KERNEL" not in {
            row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }:
            raise ObserverError("export lacks CUPTI kernel activity table")
        pids = [
            row[0]
            for row in db.execute(
                "SELECT DISTINCT globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end>start"
            )
            if type(row[0]) is int and row[0] > 0
        ]
    finally:
        db.close()
    if len(pids) != 1:
        raise ObserverError("capture must contain kernels from exactly one Nsight globalPid")
    receipt = {
        "request_id": request_id,
        "candidate_sha256": candidate_hash,
        "config_sha256": config_hash,
        "process_id": pids[0],
        "start_ns": 0,
        "end_ns": 2**63 - 1,
        "trace_sha256": _sha256(sqlite_path),
        "native_report_sha256": _sha256(native),
        "collector_started_monotonic_ns": started,
        "collector_ended_monotonic_ns": ended,
        "collector": "Nsight Systems CUDA/NVTX profile; host process identity was not inferred",
        "scope": "CUDA activity observation only; CuTe compile provenance unproven",
    }
    (capture / "capture.json").write_text(json.dumps(receipt, sort_keys=True))
    return {
        "receipt": str(capture / "capture.json"),
        "trace": str(sqlite_path),
        "worker_stdout": stdout.decode(errors="replace"),
        "worker_stderr": stderr.decode(errors="replace")[-2000:],
        "cuda_launch": {"observed": True, "nsys_global_pid": pids[0]},
        "cute_compile": {"status": "unproven"},
    }
