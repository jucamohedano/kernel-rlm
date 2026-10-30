"""CPU-only protocol/branch checks, not GPU or CuTe execution evidence."""

import asyncio
import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from cute_rlm import feedback, sandbox, verify_worker
from cute_rlm.contracts import VerificationConfig
from cute_rlm.outputs import save_tree


def settings():
    return VerificationConfig(
        atol=0.001,
        rtol=0.002,
        seed=4,
        mode="preserve",
        correctness_runs=2,
        warmup_runs=2,
        timed_runs=3,
        timeout_s=20,
    )


def task():
    source = "class Model: pass # private reference"
    return {
        "task_id": "compile-task",
        "reference_code": source,
        "source": {"sha256": hashlib.sha256(source.encode()).hexdigest()},
    }


def write_bundle(output, config):
    output.mkdir()
    save_tree(output / "initial_state.npz", {})
    save_tree(output / "init_args.npz", [])
    for trial in range(config.correctness_runs):
        save_tree(output / f"inputs-{trial}.npz", [trial])


@pytest.mark.asyncio
async def test_feedback_prepares_once_serializes_attempts_and_keeps_full_logs(
    tmp_path, monkeypatch
):
    requests, active = [], 0
    diagnostic = "compiler detail\n" * 2000

    async def launch(command, request, config):
        nonlocal active
        active += 1
        assert active == 1
        requests.append(request)
        assert config == settings()
        assert request["config"] == config.payload()
        await asyncio.sleep(0)
        active -= 1
        if request["role"] == "reference":
            assert request["prepare_only"] is True
            assert request["source"] == task()["reference_code"]
            write_bundle(Path(request["output_dir"]), config)
            return {
                "status": "prepared",
                "source_sha256": request["source_sha256"],
                "training": True,
                "module_training": {"": True},
            }
        assert request["compile_feedback"] is True
        assert "prepare_only" not in request
        assert request["module_training"] == {"": True}
        assert "reference_code" not in json.dumps(request)
        assert {p.name for p in Path(request["input_dir"]).iterdir()} == {
            "initial_state.npz",
            "init_args.npz",
            "inputs-0.npz",
            "inputs-1.npz",
        }
        return {
            "status": "compiled",
            "source_sha256": request["source_sha256"],
            "stdout": diagnostic,
            "stderr": "warning",
            "returncode": 0,
        }

    monkeypatch.setattr(feedback, "launch_phase", launch)
    service = feedback.CompileFeedback(
        task(), settings().payload(), tmp_path, worker_command=["sandbox"]
    )
    results = await asyncio.gather(
        *(service(f"owner/{i}", f"class ModelNew: pass # attempt {i}") for i in range(12))
    )
    assert [request["role"] for request in requests] == ["reference"] + ["candidate"] * 12
    assert len({request["output_dir"] for request in requests}) == 13
    assert len({result["request_id"] for result in results}) == 12
    assert [result["ordinal"] for result in results] == list(range(1, 13))
    for i, result in enumerate(results):
        assert result["session_id"] == f"owner/{i}"
        assert result["elapsed_s"] >= 0
        assert result["status"] == "compiled"
        assert result["correct"] is None and result["cute_verified"] is None
        assert result["stdout"] == diagnostic
    paths = list((tmp_path / "compile-feedback").glob("0*/response.json"))
    assert len(paths) == 12
    assert json.loads(paths[0].read_text())["stdout"] == diagnostic
    assert (tmp_path / "compile-feedback").stat().st_mode & 0o777 == 0o700


@pytest.mark.asyncio
async def test_preparation_failure_cached_and_hidden_from_model(tmp_path, monkeypatch):
    calls = []
    hidden = {
        "status": "runtime_error",
        "detail": "private reference code",
        "traceback": "oracle",
        "stdout": "expected output",
    }

    async def launch(command, request, config):
        calls.append(request)
        return hidden.copy()

    monkeypatch.setattr(feedback, "launch_phase", launch)
    service = feedback.CompileFeedback(task(), settings(), tmp_path, worker_command=["sandbox"])
    first = await service("owner", "broken candidate")
    second = await service("child", "another candidate")
    assert len(calls) == 1
    assert first["status"] == second["status"] == "infrastructure"
    assert first["stage"] == second["stage"] == "reference"
    for value in ("private reference code", "oracle", "expected output"):
        assert value not in json.dumps(first)
    saved = json.loads(next(service.directory.glob("reference-*/response.json")).read_text())
    assert saved["traceback"] == "oracle"


@pytest.mark.asyncio
@pytest.mark.parametrize("source", [None, {}, "", "  ", "source\0text", "\ud800"])
async def test_invalid_source_is_logged_without_worker_or_host_execution(
    tmp_path, monkeypatch, source
):
    async def forbidden(*args):
        pytest.fail("invalid input must not launch a worker")

    monkeypatch.setattr(feedback, "launch_phase", forbidden)
    service = feedback.CompileFeedback(task(), settings(), tmp_path, worker_command=["sandbox"])
    result = await service("../../owner", source)
    assert result["status"] == "invalid_source"
    assert result["ordinal"] == 1
    assert len(list(service.directory.glob("0*/request.json"))) == 1
    assert not list(service.directory.glob("reference-*"))


@pytest.mark.asyncio
async def test_callback_cancellation_propagates_and_records_attempt(tmp_path, monkeypatch):
    launched = asyncio.Event()
    cancelled = False

    async def launch(command, request, config):
        nonlocal cancelled
        launched.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled = True
            raise

    monkeypatch.setattr(feedback, "launch_phase", launch)
    service = feedback.CompileFeedback(task(), settings(), tmp_path, worker_command=["sandbox"])
    attempt = asyncio.create_task(service("owner", "class ModelNew: pass"))
    await launched.wait()
    attempt.cancel()
    with pytest.raises(asyncio.CancelledError):
        await attempt
    assert cancelled
    response = json.loads(next(service.directory.glob("0*/response.json")).read_text())
    assert response["status"] == "cancelled"


@pytest.fixture
def fake_gpu(monkeypatch):
    calls = []

    class Model:
        training = True

        def __init__(self, *args):
            calls.append("init")

        def cuda(self):
            return self

        def state_dict(self):
            return {}

        def named_modules(self):
            return [("", self)]

        def load_state_dict(self, state, strict):
            assert state == {} and strict is True
            calls.append("state")

        def __call__(self, *args):
            calls.append(("forward", args))
            # Deliberately unsupported by output serialization: compile feedback
            # must ignore the output entirely, not serialize/compare it.
            return object()

    torch = SimpleNamespace(
        Tensor=type("Tensor", (), {}),
        no_grad=nullcontext,
        cuda=SimpleNamespace(synchronize=lambda: calls.append("sync")),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(verify_worker, "seed_all", lambda *args: None)
    monkeypatch.setattr(
        verify_worker,
        "load_module",
        lambda *args: SimpleNamespace(
            Model=Model,
            ModelNew=Model,
            get_inputs=lambda: [7],
        ),
    )
    return calls


def worker_request(tmp_path, role):
    source = "source loaded only by the isolated worker"
    return {
        "role": role,
        "source": source,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "output_dir": str(tmp_path / "output"),
        "config": settings().payload(),
    }


def test_prepare_only_saves_bundle_and_modes_without_any_forward(tmp_path, fake_gpu):
    request = {**worker_request(tmp_path, "reference"), "prepare_only": True}
    result = verify_worker.run(request)
    assert result["status"] == "prepared"
    assert result["training"] is True and result["module_training"] == {"": True}
    assert fake_gpu == ["init"]
    assert {p.name for p in Path(request["output_dir"]).iterdir()} == {
        "initial_state.npz",
        "init_args.npz",
        "inputs-0.npz",
        "inputs-1.npz",
    }
    assert "median_ms" not in result


def test_compile_feedback_constructs_once_launches_once_syncs_without_serializing(
    tmp_path, fake_gpu
):
    inputs = tmp_path / "inputs"
    write_bundle(inputs, settings())
    request = {
        **worker_request(tmp_path, "candidate"),
        "compile_feedback": True,
        "input_dir": str(inputs),
        "training": True,
        "module_training": {"": True},
    }
    result = verify_worker.run(request)
    assert result["status"] == "compiled"
    assert fake_gpu == ["init", "state", ("forward", (0,)), "sync"]
    assert list(Path(request["output_dir"]).iterdir()) == []
    assert result["cute_verified"] is None and result["correct"] is None
    assert "median_ms" not in result


@pytest.mark.parametrize(
    "error, expected",
    [
        (SyntaxError("syntax diagnostic"), "compile_error"),
        (NotImplementedError("candidate must define ModelNew"), "unsupported"),
        (verify_worker.InitializationError("constructor failed"), "initialization_error"),
        (type("DSLUserCodeError", (Exception,), {})("JIT diagnostic"), "runtime_error"),
    ],
)
def test_worker_json_keeps_full_streams_and_traceback(monkeypatch, capsys, error, expected):
    diagnostic = "compiler output" * 2000

    def run(request):
        print(diagnostic)
        print("compiler stderr", file=sys.stderr)
        raise error

    import io

    monkeypatch.setattr(verify_worker, "run", run)
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    verify_worker.main()
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert captured.err == ""
    assert result["status"] == expected
    assert result["stdout"] == diagnostic + "\n"
    assert result["stderr"] == "compiler stderr\n"
    assert type(error).__name__ in result["traceback"]
    assert str(error) in result["traceback"]


def test_initialization_context_classifies_constructor_error_without_hiding_cause():
    with pytest.raises(verify_worker.InitializationError) as caught:
        with verify_worker.initialization():
            raise ValueError("bad constructor args")
    assert isinstance(caught.value.__cause__, ValueError)


def test_sandbox_compile_feedback_skips_nsight_even_with_observer(tmp_path, monkeypatch):
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("executable")
    bwrap.chmod(0o755)
    device = tmp_path / "device"
    device.touch()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    launches = []

    def run(command, **kwargs):
        launches.append((command, json.loads(kwargs["input"])))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(sandbox.subprocess, "run", run)
    spec = {
        "bwrap": str(bwrap),
        "runtime_python": "/venv/bin/python",
        "readonly_bindings": [],
        "gpu_devices": [{"source": str(device), "target": "/dev/nvidia0"}],
        "observer_root": str(tmp_path / "observer"),
        "nsys_binary": "/missing/nsys",
    }
    request = {
        "role": "candidate",
        "input_dir": str(inputs),
        "output_dir": str(tmp_path / "output"),
        "compile_feedback": True,
    }
    assert sandbox._run(spec, request) == 0
    assert len(launches) == 1
    assert launches[0][0][-1] == "cute_rlm.verify_worker"
    assert launches[0][1]["compile_feedback"] is True
    assert not (tmp_path / "observer").exists()


@pytest.mark.asyncio
async def test_feedback_cancellation_waits_for_real_launcher_process_cleanup(tmp_path, monkeypatch):
    from cute_rlm import verifier

    create_process = asyncio.create_subprocess_exec
    started = asyncio.Event()
    processes = []

    async def capture_process(*args, **kwargs):
        process = await create_process(*args, **kwargs)
        processes.append(process)
        started.set()
        return process

    monkeypatch.setattr(verifier.asyncio, "create_subprocess_exec", capture_process)
    service = feedback.CompileFeedback(
        task(),
        settings(),
        tmp_path,
        worker_command=[sys.executable, "-c", "import time; time.sleep(30)"],
    )
    attempt = asyncio.create_task(service("owner", "class ModelNew: pass"))
    await asyncio.wait_for(started.wait(), timeout=5)
    attempt.cancel()
    with pytest.raises(asyncio.CancelledError):
        await attempt
    assert len(processes) == 1
    assert processes[0].returncode is not None and processes[0].returncode < 0


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_owner", ["root", "recursive-child"])
async def test_cancelled_preparation_can_retry_without_losing_evidence(
    tmp_path, monkeypatch, retry_owner
):
    launched = asyncio.Event()
    requests = []
    service = feedback.CompileFeedback(task(), settings(), tmp_path, worker_command=["sandbox"])

    async def launch(command, request, config):
        requests.append(request)
        if request["role"] == "reference":
            assert service.input_directory is None
            if len(requests) == 1:
                # Retain even partial worker evidence from the cancelled launch.
                output = Path(request["output_dir"])
                output.mkdir()
                (output / "partial-worker-log.txt").write_text("incomplete preparation")
                launched.set()
                await asyncio.Future()
            write_bundle(Path(request["output_dir"]), config)
            return {
                "status": "prepared",
                "source_sha256": request["source_sha256"],
                "training": True,
                "module_training": {"": True},
            }
        assert service.input_directory == Path(request["input_dir"])
        assert {p.name for p in service.input_directory.iterdir()} == {
            "initial_state.npz",
            "init_args.npz",
            "inputs-0.npz",
            "inputs-1.npz",
        }
        return {"status": "compiled", "source_sha256": request["source_sha256"]}

    monkeypatch.setattr(feedback, "launch_phase", launch)
    first = asyncio.create_task(service("cancelled-child", "class ModelNew: pass"))
    await launched.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert service.preparation is None and service.input_directory is None
    cancelled_directory = Path(requests[0]["output_dir"]).parent
    cancelled_response = (cancelled_directory / "response.json").read_text()
    assert json.loads(cancelled_response)["status"] == "cancelled"

    result = await service(retry_owner, "class ModelNew: pass # retry")
    assert result["status"] == "compiled"
    assert [request["role"] for request in requests] == ["reference", "reference", "candidate"]
    assert requests[0]["output_dir"] != requests[1]["output_dir"]
    assert (cancelled_directory / "response.json").read_text() == cancelled_response
    assert (
        cancelled_directory / "output/partial-worker-log.txt"
    ).read_text() == "incomplete preparation"
    assert len(list(service.directory.glob("reference-*"))) == 2
    assert service.input_directory.parent != cancelled_directory
