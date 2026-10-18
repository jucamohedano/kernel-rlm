import hashlib
import importlib
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1] / "training/environments/kernel_rlm"


@pytest.fixture
def checkpoints(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    return importlib.import_module("kernel_trl.checkpoint_state")


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    return importlib.import_module("kernel_trl.probability_audit")


def metadata(directory, **overrides):
    value = {"step": 2, "world_size": 2, "frozen_weights_saved": False, **overrides}
    (directory / "metadata.json").write_text(json.dumps(value))


def test_checkpoint_manifest_is_checked_before_resume(checkpoints, tmp_path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"fixture")
    digest = hashlib.sha256(b"fixture").hexdigest()
    metadata(tmp_path, sha256={"adapter_model.safetensors": digest})
    assert checkpoints.checkpoint_metadata(tmp_path, 2)["step"] == 2
    (tmp_path / "adapter_model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        checkpoints.checkpoint_metadata(tmp_path, 2)


@pytest.mark.parametrize(
    "overrides", [{"step": 1}, {"world_size": 1}, {"frozen_weights_saved": True}]
)
def test_incompatible_checkpoint_is_rejected(checkpoints, tmp_path, overrides):
    metadata(tmp_path, **overrides)
    with pytest.raises(ValueError):
        checkpoints.checkpoint_metadata(tmp_path, 2)


def test_checkpoint_manifest_cannot_escape_its_directory(checkpoints, tmp_path):
    metadata(tmp_path, sha256={"../other": "unused"})
    with pytest.raises(ValueError, match="nonlocal"):
        checkpoints.checkpoint_metadata(tmp_path, 2)


def test_old_checkpoint_is_readable_without_claiming_new_manifest(checkpoints, tmp_path):
    metadata(tmp_path)
    assert "sha256" not in checkpoints.checkpoint_metadata(tmp_path, 2)


def test_probability_audit_uses_real_token_denominator(audit):
    result = audit.summarize_deltas([0, 0, math.log(2), math.log(0.5)])
    assert result["tokens"] == 4
    assert result["mean_log_ratio"] == pytest.approx(0)
    assert result["outside_ratio_0_8_to_1_2_fraction"] == 0.5
    assert result["mean_absolute_log_ratio"] == pytest.approx(math.log(2) / 2)
    assert result["p95_absolute_log_ratio"] == pytest.approx(math.log(2))


@pytest.mark.parametrize("values", [[], [math.nan], [math.inf]])
def test_probability_audit_rejects_invalid_values(audit, values):
    with pytest.raises(ValueError, match="finite"):
        audit.summarize_deltas(values)
