import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from cute_rlm.icl import validation, worker
from cute_rlm.outputs import save_tree

COMMAND = ["/host/python", "-m", "cute_rlm.sandbox", "--spec", "/trusted/spec.json"]


@pytest.fixture
def task():
    return {
        "id": "test",
        "interface": {"forward_args": ["x"]},
        "cases": [{"n": 5, "seed": 20261101}, {"n": 7, "seed": 20261102}],
        "atol": 1e-5,
        "rtol": 1e-5,
    }


@pytest.fixture
def fake_phases(monkeypatch, task):
    from cute_rlm.icl import tasks

    monkeypatch.setattr(tasks, "reference_source", lambda task: "# trusted oracle")
    calls = []
    settings = {"mismatch": False, "fail": False}

    async def launch(command, request, config):
        calls.append(request)
        assert command == COMMAND and config.timeout_s == 600
        assert request["compile_feedback"] is True
        assert request["worker_module"] == "cute_rlm.icl.worker"
        directory = Path(request["output_dir"])
        directory.mkdir()
        if request["role"] == "reference":
            (directory / "test-input-contract.json").write_text(
                json.dumps(worker.input_contract(task))
            )
            prefix = "expected"
        else:
            assert request["profile_compile"] is True
            inputs = Path(request["input_dir"])
            assert [p.name for p in inputs.iterdir()] == ["test-input-contract.json"]
            assert "expected" not in (inputs / "test-input-contract.json").read_text()
            if settings["fail"]:
                return {
                    "status": "runtime_error",
                    "traceback": "FULL ERROR",
                    "stdout": "partial log",
                    "stderr": "full stderr",
                }
            prefix = "outputs"
        for i in range(len(task["cases"])):
            value = 2 if settings["mismatch"] and prefix == "outputs" else 1
            save_tree(directory / f"{prefix}-{i}.npz", np.array([value], dtype=np.float32))
        return {
            "status": "executed",
            "case_count": len(task["cases"]),
            "source_sha256": request["source_sha256"],
            "stdout": "log",
            "stderr": "",
        }

    monkeypatch.setattr(validation, "launch_phase", launch)
    return calls, settings


def test_torch_only_numerical_pass_is_not_cute_success(tmp_path, task, fake_phases):
    result = asyncio.run(
        validation.validate_candidate(task, "# source", tmp_path / "run", worker_command=COMMAND)
    )
    assert result["numerical_correct"] is True
    assert result["compile_forward"] is True
    assert result["cute_execution_observed"] is False
    assert result["valid_solution"] is None and result["source_review_required"] is True
    assert result["timing_performed"] is False
    assert result["status"] == "review_required"
    assert len(fake_phases[0]) == 2
    assert json.loads((tmp_path / "run/result.json").read_text()) == result
    assert (tmp_path / "run/reference-request.json").is_file()
    assert (tmp_path / "run/candidate-result.json").is_file()


def test_numerical_mismatch_and_failure_preserve_errors_without_retry(tmp_path, task, fake_phases):
    calls, settings = fake_phases
    settings["mismatch"] = True
    result = asyncio.run(
        validation.validate_candidate(task, "# source", tmp_path / "wrong", worker_command=COMMAND)
    )
    assert result["numerical_correct"] is False
    assert result["status"] == "numerical_mismatch"
    settings["fail"] = True
    result = asyncio.run(
        validation.validate_candidate(task, "# source", tmp_path / "fail", worker_command=COMMAND)
    )
    assert result["status"] == "runtime_error" and result["numerical_correct"] is None
    assert result["candidate"]["traceback"] == "FULL ERROR"
    assert result["candidate"]["stdout"] == "partial log"
    assert result["candidate"]["stderr"] == "full stderr"
    assert len(calls) == 4


def test_validation_rejects_direct_python_execution(tmp_path, task):
    with pytest.raises(ValueError, match="sandbox"):
        asyncio.run(
            validation.validate_candidate(
                task, "source", tmp_path / "run", worker_command=["python", "student.py"]
            )
        )
    assert not (tmp_path / "run").exists()


def test_explicit_forward_signature_and_stride_copy(task):
    class Good:
        def forward(self, x):
            return x

        __call__ = forward

    class Bad:
        def forward(self, *args):
            return args

        __call__ = forward

    worker.check_model(Good(), task)
    with pytest.raises(NotImplementedError, match="explicit"):
        worker.check_model(Bad(), task)

    class Tensor:
        shape = (3, 5)
        dtype = "float32"

        def stride(self):
            return (10, 2)

    copied = []
    torch = SimpleNamespace(
        Tensor=Tensor,
        empty_strided=lambda shape, stride, **kw: SimpleNamespace(
            copy_=lambda value: copied.append((shape, stride, kw, value))
        ),
    )
    x = Tensor()
    actual = worker.cuda_inputs(torch, [x, 4.0])
    assert actual[1] == 4.0
    assert copied == [((3, 5), (10, 2), {"dtype": "float32", "device": "cuda"}, x)]


def test_real_compile_trace_survives_actual_sandbox_output_collection(tmp_path, monkeypatch, capfd):
    from cute_rlm import sandbox
    from cute_rlm.compile_trace import install_compile_trace

    executable = tmp_path / "bwrap"
    executable.write_text("fake")
    executable.chmod(0o755)
    device = tmp_path / "gpu"
    device.touch()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    spec = {
        "bwrap": str(executable),
        "runtime_python": "/venv/python",
        "gpu_devices": [{"source": str(device), "target": "/dev/nvidia0"}],
        "observer_root": str(tmp_path / "must-not-use-nsight"),
    }
    request = {
        "role": "candidate",
        "input_dir": str(inputs),
        "output_dir": str(tmp_path / "out"),
        "compile_feedback": True,
        "profile_compile": True,
        "worker_module": "cute_rlm.icl.worker",
    }

    class Compiled:
        __ptx__ = b".visible .entry kernel_main() {}"
        __cubin__ = b"cubin"

        def __call__(self, value):
            return value + 1

    def run(command, **kwargs):
        assert command[-1] == "cute_rlm.icl.worker"
        assert json.loads(kwargs["input"])["output_dir"] == "/phase/output"
        output = next(
            Path(command[i + 1])
            for i, arg in enumerate(command)
            if arg == "--bind" and command[i + 2] == "/phase/output"
        )
        cute = SimpleNamespace(compile=lambda *a, **k: Compiled())
        torch = SimpleNamespace(
            cuda=SimpleNamespace(
                synchronize=lambda: None,
                nvtx=SimpleNamespace(range_push=lambda name: None, range_pop=lambda: None),
            )
        )
        trace = install_compile_trace(cute, torch, output)
        try:
            assert cute.compile("kernel")(3) == 4
            trace.snapshot()
        finally:
            trace.close()
        return SimpleNamespace(returncode=0, stdout=b'{"status":"executed"}\n', stderr=b"")

    monkeypatch.setattr(sandbox.subprocess, "run", run)
    assert sandbox._run(spec, request) == 0
    assert json.loads(capfd.readouterr().out)["status"] == "executed"
    observation = validation._compile_observation(Path(request["output_dir"]))
    assert observation["observed"] is True
    assert observation["completed_invocations"][0]["status"] == "completed"
    artifact = observation["artifacts"][0]
    artifact_path = Path(request["output_dir"]) / artifact["ptx_path"]
    assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == artifact["ptx_sha256"]
    artifact_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        validation._compile_observation(Path(request["output_dir"]))


def test_worker_main_preserves_full_traceback_and_output(monkeypatch, capsys):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    def fail(request):
        print("partial stdout")
        print("partial stderr", file=__import__("sys").stderr)
        raise SyntaxError("broken")

    monkeypatch.setattr(worker, "run", fail)
    worker.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "compile_error"
    assert result["error_type"] == "SyntaxError" and "SyntaxError: broken" in result["traceback"]
    assert result["stdout"] == "partial stdout\n" and result["stderr"] == "partial stderr\n"


def test_worker_regenerates_reference_and_candidate_cases_with_strides(tmp_path, task, monkeypatch):
    import sys
    from contextlib import nullcontext

    from cute_rlm.icl import tasks

    class Tensor:
        dtype = "float32"
        device = "cuda:0"

        def numel(self):
            return self.values.size

        def untyped_storage(self):
            return SimpleNamespace(data_ptr=lambda: id(self.values))

        def __init__(self, shape, stride, values=None):
            self.shape, self._stride = shape, stride
            self.values = np.ones(shape, dtype=np.float32) if values is None else values

        def stride(self):
            return self._stride

        def copy_(self, other):
            self.values = other.values.copy()

        def contiguous(self):
            return self

    seen = []
    oracle_calls = []

    class Module:
        pass

    class Model(Module):
        def cuda(self):
            return self

        eval = cuda

        def state_dict(self):
            return {}

        def forward(self, x):
            assert x.stride() == (2,)
            return Tensor(x.shape, (1,), x.values.copy())

        __call__ = forward

    torch = SimpleNamespace(
        Tensor=Tensor,
        nn=SimpleNamespace(Module=Module),
        manual_seed=lambda seed: None,
        no_grad=nullcontext,
        empty_strided=lambda shape, stride, **kw: Tensor(shape, stride),
        cuda=SimpleNamespace(manual_seed_all=lambda seed: None, synchronize=lambda: None),
    )

    def build_inputs(task, case, torch):
        seen.append(case["seed"])
        return [Tensor((case["n"],), (2,))]

    def oracle(task, inputs, torch):
        oracle_calls.append(True)
        assert inputs[0].stride() == (2,)
        return Tensor(inputs[0].shape, (1,), inputs[0].values.copy())

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(tasks, "build_inputs", build_inputs)
    monkeypatch.setattr(tasks, "expected_output", oracle)
    monkeypatch.setattr(worker, "to_arrays", lambda torch, value: value.values)
    monkeypatch.setattr(worker, "load_module", lambda *a: SimpleNamespace(ModelNew=Model))
    source = "# captured source"
    common = {
        "source": source,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "icl_task": task,
        "profile_compile": False,
    }
    reference = tmp_path / "reference"
    assert (
        worker.run({**common, "role": "reference", "output_dir": str(reference)})["status"]
        == "executed"
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "test-input-contract.json").write_bytes(
        (reference / "test-input-contract.json").read_bytes()
    )
    candidate = tmp_path / "candidate"
    assert (
        worker.run(
            {**common, "role": "candidate", "output_dir": str(candidate), "input_dir": str(inputs)}
        )["status"]
        == "executed"
    )
    assert len(oracle_calls) == len(task["cases"])
    assert seen == [case["seed"] for case in task["cases"]] * 2
    for i in range(len(task["cases"])):
        assert (reference / f"expected-{i}.npz").is_file()
        assert (candidate / f"outputs-{i}.npz").is_file()


def test_identity_output_alias_is_rejected_by_worker_guard():
    class Tensor:
        device = "cuda:0"

        def __init__(self, pointer):
            self.pointer = pointer

        def numel(self):
            return 1

        def untyped_storage(self):
            return SimpleNamespace(data_ptr=lambda: self.pointer)

    torch = SimpleNamespace(Tensor=Tensor)
    value = Tensor(17)
    with pytest.raises(NotImplementedError, match="aliases"):
        worker.reject_cross_tree_aliases(torch, [value], value, {})
    worker.reject_cross_tree_aliases(torch, [value], Tensor(18), {})


def test_worker_rejects_identity_candidate_before_output_save(tmp_path, task, monkeypatch):
    # Exercise the actual worker call site, not only the storage helper.
    import sys
    from contextlib import nullcontext

    from cute_rlm.icl import tasks

    class Tensor:
        device = "cuda:0"
        shape = (1,)
        dtype = "float32"

        def copy_(self, other):
            return self

        def stride(self):
            return (1,)

        def numel(self):
            return 1

        def untyped_storage(self):
            return SimpleNamespace(data_ptr=lambda: id(self))

    class Module:
        def cuda(self):
            return self

        eval = cuda

        def state_dict(self):
            return {}

        def forward(self, x):
            return x

        __call__ = forward

    torch = SimpleNamespace(
        Tensor=Tensor,
        nn=SimpleNamespace(Module=Module),
        manual_seed=lambda seed: None,
        no_grad=nullcontext,
        empty_strided=lambda *a, **k: Tensor(),
        cuda=SimpleNamespace(manual_seed_all=lambda seed: None, synchronize=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(tasks, "build_inputs", lambda *a: [Tensor()])
    monkeypatch.setattr(worker, "load_module", lambda *a: SimpleNamespace(ModelNew=Module))
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "test-input-contract.json").write_text(json.dumps(worker.input_contract(task)))
    source = "# identity candidate"
    request = {
        "role": "candidate",
        "icl_task": task,
        "source": source,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "output_dir": str(tmp_path / "candidate"),
        "input_dir": str(inputs),
    }
    with pytest.raises(NotImplementedError, match="aliases"):
        worker.run(request)
    assert not (tmp_path / "candidate/outputs-0.npz").exists()
