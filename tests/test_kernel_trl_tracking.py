import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("mode", ["train", "audit", "readback"])
def test_native_tracking_is_local_and_audit_only_is_not_training(monkeypatch, tmp_path, mode):
    root = Path(__file__).parents[1] / "training/environments/kernel_rlm"
    monkeypatch.syspath_prepend(str(root))
    module = importlib.import_module("kernel_trl.rlm_update")
    arguments = SimpleNamespace(
        output_dir=tmp_path / "run" / "groups" / "step-002-task" / "update",
        audit_only=mode == "audit",
        readback_only=mode == "readback",
    )
    result = module.tracking_configuration(arguments, 1)
    assert result["report_to"] == ("trackio" if mode == "train" else "none")
    assert result["trackio_space_id"] is None
    assert result["project"] == "laguna-kernel-rlm"
    assert result["run_name"].endswith("step-2")
