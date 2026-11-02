"""CPU-only protocol tests; scripted launchers do not provide GPU evidence."""

import asyncio
import builtins
import hashlib
import importlib
import json
from pathlib import Path

import pytest
from cute_rlm.icl import repl_feedback

COMMAND = ["/host/python", "-m", "cute_rlm.sandbox", "--spec", "/trusted/spec.json"]


@pytest.fixture
def task():
    return {
        "id": "public-task",
        "interface": {"forward_args": ["x"]},
        "cases": [{"n": n, "seed": 20261101 + i} for i, n in enumerate((1, 7, 32, 129))],
    }


def executed(request):
    return {
        "status": "executed",
        "source_sha256": request["source_sha256"],
        "case_count": 4,
        "compile_forward": True,
    }


@pytest.mark.asyncio
async def test_candidate_only_all_cases_serialized_with_private_full_evidence(
    tmp_path, task, monkeypatch
):
    from cute_rlm.icl import tasks

    def forbidden_oracle(*args):
        pytest.fail("feedback must not run host inputs or a reference/oracle")

    for name in ("reference_source", "expected_output", "build_inputs"):
        monkeypatch.setattr(tasks, name, forbidden_oracle)
    calls, active = [], 0
    diagnostics = "compiler diagnostic\n" * 2000
    source = "raise AssertionError('candidate must never execute on the host')\n"

    async def launch(command, request, config):
        nonlocal active
        active += 1
        assert active == 1
        calls.append(request)
        assert command == COMMAND and config.timeout_s == 600
        assert request["role"] == "candidate"
        assert request["worker_module"] == "cute_rlm.icl.worker"
        assert request["compile_feedback"] is True and request["profile_compile"] is True
        assert request["icl_task"]["cases"] == task["cases"]
        assert len(request["icl_task"]["cases"]) == 4
        assert request["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
        inputs = Path(request["input_dir"])
        assert {p.name for p in inputs.iterdir()} == {"test-input-contract.json"}
        assert json.loads((inputs / "test-input-contract.json").read_text()) == (
            repl_feedback.input_contract(task)
        )
        output = Path(request["output_dir"])
        assert not output.exists() and inputs.parent == output.parent
        assert output not in inputs.parents and inputs not in output.parents
        output.mkdir()
        for i in range(4):
            (output / f"outputs-{i}.npz").write_bytes(b"private worker output")
        artifacts = output / "compile-artifacts"
        artifacts.mkdir()
        (artifacts / "kernel.ptx").write_bytes(b"private compiler artifact")
        await asyncio.sleep(0)
        active -= 1
        return {
            **executed(request),
            "stdout": diagnostics,
            "stderr": "warning",
            "outputs": ["PRIVATE OUTPUT VALUE"],
            "expected": ["PRIVATE EXPECTED VALUE"],
            "numerical_correct": True,
            "compile_provenance": {"artifact": "PRIVATE PROVENANCE"},
            "valid_solution": True,
        }

    monkeypatch.setattr(repl_feedback, "launch_phase", launch)
    service = repl_feedback.ReplCompileFeedback(task, tmp_path, worker_command=COMMAND)
    results = await asyncio.gather(
        *(service(owner, source) for owner in ("root", "root/child", "../../child"))
    )
    assert len(calls) == 3
    assert len({call["output_dir"] for call in calls}) == 3
    assert [result["ordinal"] for result in results] == [1, 2, 3]
    assert {result["session_id"] for result in results} == {"root", "root/child", "../../child"}
    for result, request in zip(results, calls, strict=True):
        assert result["status"] == "compiled" and result["compile_forward"] is True
        assert result["correct"] is None and result["cute_verified"] is None
        assert result["stdout"] == diagnostics and result["stderr"] == "warning"
        assert result["elapsed_s"] >= 0
        assert (
            not {"outputs", "expected", "numerical_correct", "compile_provenance", "valid_solution"}
            & result.keys()
        )
        directory = Path(request["output_dir"]).parent
        assert (directory / "candidate.py").read_text() == source
        assert (
            json.loads((directory / "request.json").read_text())["session_id"]
            == result["session_id"]
        )
        assert json.loads((directory / "worker-request.json").read_text()) == request
        raw = json.loads((directory / "worker-response.json").read_text())
        assert raw["expected"] == ["PRIVATE EXPECTED VALUE"]
        assert raw["stdout"] == diagnostics
        assert json.loads((directory / "response.json").read_text()) == result
        assert (directory / "output/outputs-3.npz").read_bytes() == b"private worker output"
        assert (
            directory / "output/compile-artifacts/kernel.ptx"
        ).read_bytes() == b"private compiler artifact"
    assert service.directory.stat().st_mode & 0o777 == 0o700
    assert not list(service.directory.glob("reference*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("status", sorted(repl_feedback.CANDIDATE_ERRORS))
async def test_candidate_errors_return_full_diagnostics_without_output_values(
    tmp_path, task, monkeypatch, status
):
    async def launch(command, request, config):
        return {
            "status": status,
            "detail": "diagnostic",
            "error_type": "CompilerError",
            "traceback": "FULL TRACEBACK",
            "stdout": "partial stdout",
            "stderr": "partial stderr",
            "launcher_stderr": "launcher log",
            "outputs": [13.0],
            "correct": True,
        }

    monkeypatch.setattr(repl_feedback, "launch_phase", launch)
    service = repl_feedback.ReplCompileFeedback(task, tmp_path, worker_command=COMMAND)
    result = await service("child", "# broken candidate")
    assert result["status"] == status and result["compile_forward"] is False
    assert result["traceback"] == "FULL TRACEBACK" and result["stdout"] == "partial stdout"
    assert result["stderr"] == "partial stderr" and result["launcher_stderr"] == "launcher log"
    assert result["correct"] is None and result["cute_verified"] is None
    assert "outputs" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["infrastructure", "identity", "case_count", "exception"])
async def test_infrastructure_is_journaled_and_propagates_without_retry(
    tmp_path, task, monkeypatch, failure
):
    calls = []

    async def launch(command, request, config):
        calls.append(request)
        if failure == "exception":
            raise OSError("launcher unavailable")
        result = executed(request)
        if failure == "infrastructure":
            result.update(status="infrastructure", detail="namespace failure")
        elif failure == "identity":
            result["source_sha256"] = "wrong source"
        elif failure == "case_count":
            result["case_count"] = 1
        return result

    monkeypatch.setattr(repl_feedback, "launch_phase", launch)
    service = repl_feedback.ReplCompileFeedback(task, tmp_path, worker_command=COMMAND)
    with pytest.raises(RuntimeError, match="infrastructure"):
        await service("root", "# source")
    assert len(calls) == 1
    saved = json.loads(next(service.directory.glob("0*/response.json")).read_text())
    assert saved["status"] == "infrastructure" and saved["traceback"]
    if failure != "exception":
        assert next(service.directory.glob("0*/worker-response.json")).is_file()
    assert isinstance(service.fatal_error, RuntimeError)
    with pytest.raises(RuntimeError, match="no retries"):
        await service("another-child", "# different source")
    assert len(calls) == 1 and service.ordinal == 1


@pytest.mark.asyncio
async def test_phase_timeout_is_nonfatal_and_next_source_can_run(tmp_path, task, monkeypatch):
    calls = []

    async def launch(command, request, config):
        calls.append(request)
        if len(calls) == 1:
            output = Path(request["output_dir"])
            output.mkdir()
            (output / "partial.txt").write_text("partial timeout artifact")
            raise TimeoutError("candidate phase deadline")
        return executed(request)

    monkeypatch.setattr(repl_feedback, "launch_phase", launch)
    service = repl_feedback.ReplCompileFeedback(task, tmp_path, worker_command=COMMAND)
    first = await service("root", "# hanging candidate")
    assert first["status"] == "compile_timeout" and first["error_type"] == "TimeoutError"
    assert first["correct"] is None and first["cute_verified"] is None
    assert first["compile_forward"] is False and service.fatal_error is None
    assert "600 seconds" in first["detail"]
    assert "TimeoutError" in first["traceback"]
    assert "can lose partial" in first["diagnostic_limit"]
    directory = Path(calls[0]["output_dir"]).parent
    assert (
        json.loads((directory / "worker-response.json").read_text())["status"] == "compile_timeout"
    )
    assert json.loads((directory / "response.json").read_text()) == first
    assert (directory / "worker-request.json").is_file()
    assert (directory / "output/partial.txt").read_text() == "partial timeout artifact"
    assert len(calls) == 1
    second = await service("root/child", "# repaired candidate")
    assert second["status"] == "compiled" and service.fatal_error is None
    assert len(calls) == 2 and calls[0]["output_dir"] != calls[1]["output_dir"]


@pytest.mark.asyncio
async def test_cancellation_retains_partial_artifacts_and_propagates(tmp_path, task, monkeypatch):
    launched = asyncio.Event()

    async def launch(command, request, config):
        output = Path(request["output_dir"])
        output.mkdir()
        (output / "partial.txt").write_text("partial compile capture")
        launched.set()
        await asyncio.Future()

    monkeypatch.setattr(repl_feedback, "launch_phase", launch)
    service = repl_feedback.ReplCompileFeedback(task, tmp_path, worker_command=COMMAND)
    attempt = asyncio.create_task(service("child", "# source"))
    await launched.wait()
    attempt.cancel()
    with pytest.raises(asyncio.CancelledError):
        await attempt
    saved = json.loads(next(service.directory.glob("0*/response.json")).read_text())
    assert saved["status"] == "cancelled" and saved["session_id"] == "child"
    assert service.fatal_error is None
    assert (
        next(service.directory.glob("0*/output/partial.txt")).read_text()
        == "partial compile capture"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("source", [None, {}, "", "  ", "source\0text", "\ud800"])
async def test_invalid_source_logged_without_worker(tmp_path, task, monkeypatch, source):
    async def forbidden(*args):
        pytest.fail("invalid source must not launch a worker")

    monkeypatch.setattr(repl_feedback, "launch_phase", forbidden)
    service = repl_feedback.ReplCompileFeedback(task, tmp_path, worker_command=COMMAND)
    result = await service("root", source)
    assert result["status"] == "invalid_source"
    assert next(service.directory.glob("0*/request.json")).is_file()
    assert not list(service.directory.glob("0*/candidate-inputs"))


@pytest.mark.asyncio
async def test_disk_failure_propagates_without_worker(tmp_path, task, monkeypatch):
    async def forbidden(*args):
        pytest.fail("disk failure must not launch a worker")

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    service = repl_feedback.ReplCompileFeedback(task, tmp_path, worker_command=COMMAND)
    monkeypatch.setattr(repl_feedback, "launch_phase", forbidden)
    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(OSError, match="disk full"):
        await service("root", "# source")
    assert isinstance(service.fatal_error, OSError)


@pytest.mark.parametrize(
    "command",
    [[], ["python", "candidate.py"], COMMAND + ["extra"], tuple(COMMAND)],
)
def test_requires_explicit_five_item_sandbox_command(tmp_path, task, command):
    with pytest.raises(ValueError, match="sandbox"):
        repl_feedback.ReplCompileFeedback(task, tmp_path, worker_command=command)
    assert not (tmp_path / "compile-feedback").exists()


def test_requires_all_four_cases_and_freezes_task(tmp_path, task):
    short = {**task, "cases": task["cases"][:1]}
    with pytest.raises(ValueError, match="four"):
        repl_feedback.ReplCompileFeedback(short, tmp_path, worker_command=COMMAND)
    service = repl_feedback.ReplCompileFeedback(task, tmp_path, worker_command=COMMAND)
    task["cases"][0]["n"] = 99
    assert service.contract["cases"][0]["case"]["n"] == 1


def test_imports_and_contract_do_not_need_gpu_or_oracle(task, monkeypatch):
    original_import = builtins.__import__

    def cpu_only_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "cutlass", "cuda"}:
            pytest.fail(f"host imported GPU dependency: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", cpu_only_import)
    module = importlib.reload(repl_feedback)
    contract = module.input_contract(task)
    assert len(contract["cases"]) == 4
    assert contract["task_id"] == task["id"]


@pytest.mark.asyncio
async def test_candidate_phase_timeout_is_feedback_not_fatal_infrastructure(
    tmp_path, task, monkeypatch
):
    async def launch(command, request, config):
        raise TimeoutError("isolated phase deadline")

    monkeypatch.setattr(repl_feedback, "launch_phase", launch)
    service = repl_feedback.ReplCompileFeedback(task, tmp_path, worker_command=COMMAND)
    result = await service("child", "# source")
    assert result["status"] == "compile_timeout"
    assert result["error_type"] == "TimeoutError"
    assert result["correct"] is None and result["cute_verified"] is None
    assert service.fatal_error is None
    saved = json.loads(next(service.directory.glob("0*/response.json")).read_text())
    assert saved == result
