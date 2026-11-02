import io
import json

import pytest
from cute_rlm.icl import worker


@pytest.mark.parametrize(
    "error, expected",
    [
        (OSError("disk full"), "infrastructure"),
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
