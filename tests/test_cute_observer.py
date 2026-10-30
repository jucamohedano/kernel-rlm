import hashlib
import json
import sqlite3

import pytest
from cute_rlm.observer import NsightObserver, ObserverError, parse_nsys_sqlite


def make_trace(path):
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, globalPid INTEGER, demangledName TEXT)"
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)",
        [
            (110, 120, 42, "cute_kernel"),
            (121, 125, 7, "other_process"),
            (90, 95, 42, "outside_window"),
        ],
    )
    connection.commit()
    connection.close()


def test_parse_filters_by_pid_and_capture_window(tmp_path):
    trace = tmp_path / "trace.sqlite"
    make_trace(trace)
    assert parse_nsys_sqlite(trace, process_id=42, start_ns=100, end_ns=130) == [
        {"process_id": 42, "start_ns": 110, "end_ns": 120, "kernel_name": "cute_kernel"}
    ]


def test_parse_fails_closed_without_official_activity_table(tmp_path):
    trace = tmp_path / "empty.sqlite"
    sqlite3.connect(trace).close()
    with pytest.raises(ObserverError, match="no CUPTI kernel"):
        parse_nsys_sqlite(trace, process_id=42, start_ns=100, end_ns=130)


@pytest.mark.asyncio
async def test_observer_binds_capture_and_keeps_compile_unproven(tmp_path):
    request_id = "a" * 32
    folder = tmp_path / request_id
    folder.mkdir()
    trace = folder / "trace.sqlite"
    make_trace(trace)
    source_hash, config_hash = "b" * 64, "c" * 64
    receipt = {
        "request_id": request_id,
        "candidate_sha256": source_hash,
        "config_sha256": config_hash,
        "process_id": 42,
        "start_ns": 100,
        "end_ns": 130,
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
    }
    (folder / "capture.json").write_text(json.dumps(receipt))
    result = await NsightObserver(tmp_path)(request_id, source_hash, config_hash)
    assert result["launched"] is True
    assert result["compiled"] is False
    assert result["cuda_launch"]["kernels"][0]["kernel_name"] == "cute_kernel"
    assert result["cute_compile"]["observed"] is False


@pytest.mark.asyncio
async def test_observer_rejects_identity_mismatch(tmp_path):
    request_id = "a" * 32
    folder = tmp_path / request_id
    folder.mkdir()
    trace = folder / "trace.sqlite"
    make_trace(trace)
    (folder / "capture.json").write_text(
        json.dumps(
            {
                "request_id": request_id,
                "candidate_sha256": "d" * 64,
                "config_sha256": "c" * 64,
                "process_id": 42,
                "start_ns": 100,
                "end_ns": 130,
                "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            }
        )
    )
    with pytest.raises(ObserverError, match="identity mismatch"):
        await NsightObserver(tmp_path)(request_id, "b" * 64, "c" * 64)


def test_real_nsight_numeric_demangled_name_resolves_string_table(tmp_path):
    trace = tmp_path / "trace.sqlite"
    connection = sqlite3.connect(trace)
    connection.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, globalPid INTEGER, demangledName INTEGER)"
    )
    connection.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    connection.execute("INSERT INTO StringIds VALUES (7, 'resolved_kernel')")
    connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1, 2, 42, 7)")
    connection.commit()
    connection.close()
    rows = parse_nsys_sqlite(trace, process_id=42, start_ns=0, end_ns=3)
    assert rows[0]["kernel_name"] == "resolved_kernel"


def test_compile_artifact_must_match_kernel_inside_forward_and_invocation(tmp_path):
    from cute_rlm.observer import correlate_cute_artifacts

    trace = tmp_path / "trace.sqlite"
    db = sqlite3.connect(trace)
    db.execute("CREATE TABLE StringIds(id INTEGER, value TEXT)")
    db.execute("INSERT INTO StringIds VALUES(1,'kernel_main')")
    db.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER,mangledName INTEGER)"
    )
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(120,125,42,5,1)")
    db.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER)"
    )
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(115,122,99,5)")
    db.execute(
        "CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,textId INTEGER,globalTid INTEGER)"
    )
    rid = "a" * 32
    db.executemany(
        "INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?)",
        [(100, 150, "cute_forward:" + rid, None, 99), (110, 130, "cute_compile:a:i", None, 99)],
    )
    db.commit()
    db.close()
    artifacts = tmp_path / "compiler"
    artifacts.mkdir()
    ptx = b".visible .entry kernel_main() {}"
    cubin = b"binary"
    (artifacts / "a.ptx").write_bytes(ptx)
    (artifacts / "a.cubin").write_bytes(cubin)
    provenance = {
        "artifacts": [
            {
                "id": "a",
                "ptx_path": "a.ptx",
                "ptx_sha256": hashlib.sha256(ptx).hexdigest(),
                "cubin_path": "a.cubin",
                "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
            }
        ],
        "invocations": [
            {"range_name": "cute_compile:a:i", "artifact_id": "a", "status": "completed"}
        ],
    }
    (artifacts / "compile-provenance.json").write_text(json.dumps(provenance))
    result = correlate_cute_artifacts(trace, artifacts, rid)
    assert result["observed"] is True and result["matches"][0]["kernel_symbol"] == "kernel_main"
    assert correlate_cute_artifacts(trace, artifacts, "b" * 32)["observed"] is False
    (artifacts / "a.ptx").write_text("changed")
    with pytest.raises(ObserverError, match="hash mismatch"):
        correlate_cute_artifacts(trace, artifacts, rid)
