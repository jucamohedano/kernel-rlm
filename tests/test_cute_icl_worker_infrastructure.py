import io
import json

import pytest
from cute_rlm.icl import worker


@pytest.mark.parametrize(
    "error, expected",
    [
        (OSError("candidate file operation failed"), "runtime_error"),
        (worker.WorkerInfrastructureError("trusted CUDA/input setup failed"), "infrastructure"),
        (RuntimeError("generated forward failed"), "runtime_error"),
    ],
)
def test_worker_distinguishes_infrastructure_from_candidate_errors(
    monkeypatch, capsys, error, expected
):
    def fail(request):
        raise error

    monkeypatch.setattr(worker, "run", fail)
    monkeypatch.setattr(worker.sys, "stdin", io.StringIO("{}"))
    worker.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == expected


def test_trusted_worker_stage_preserves_failure_cause():
    with pytest.raises(worker.WorkerInfrastructureError) as caught:
        with worker.trusted_worker_stage():
            raise RuntimeError("CUDA allocation failed")
    assert isinstance(caught.value.__cause__, RuntimeError)


@pytest.mark.parametrize("error", [OSError("disk full"), RuntimeError("input setup failed")])
def test_trusted_input_and_persistence_errors_remain_infrastructure(error):
    with pytest.raises(worker.WorkerInfrastructureError) as caught:
        with worker.trusted_worker_stage():
            raise error
    assert caught.value.__cause__ is error


def test_compile_trace_storage_error_remains_infrastructure(monkeypatch, capsys):
    from cute_rlm.compile_trace import CompileTracePersistenceError

    def fail(request):
        raise CompileTracePersistenceError("trusted artifact disk full")

    monkeypatch.setattr(worker, "run", fail)
    monkeypatch.setattr(worker.sys, "stdin", io.StringIO("{}"))
    worker.main()
    assert json.loads(capsys.readouterr().out)["status"] == "infrastructure"
