import json
import math

import pytest
import torch
from rlm_train.sft.bundle import load_bundle, write_synthetic_fixture
from rlm_train.sft.train_prime import METRICS_FILE, prime_group_batches, run_synthetic


@pytest.fixture
def fixture(tmp_path):
    directory = tmp_path / "bundle"
    return directory, write_synthetic_fixture(directory)


def test_prime_batches_maintain_original_rows_and_masks(fixture):
    bundle = load_bundle(*fixture, allow_synthetic=True)
    groups = [list(prime_group_batches(group, bundle)) for group in bundle.groups]
    assert [len(group) for group in groups] == [2, 1]
    assert [[row["input_ids"].tolist() for row in group] for group in groups] == [
        [[[1, 2, 3]], [[5, 0]]],
        [[[5, 1]]],
    ]
    assert [[row["target_ids"].tolist() for row in group] for group in groups] == [
        [[[2, 3, 4]], [[0, 6]]],
        [[[1, 7]]],
    ]
    assert [[row["loss_mask"].tolist() for row in group] for group in groups] == [
        [[[False, True, True]], [[False, False]]],
        [[[False, True]]],
    ]
    for group in groups:
        for row in group:
            width = row["input_ids"].shape[-1]
            assert row["position_ids"].tolist() == [list(range(width))]
            assert row["seq_lens"].tolist() == [width]
            assert row["seq_lens"].dtype == torch.long


def test_prime_probe_emits_only_aggregates(fixture, tmp_path):
    run = tmp_path / "prime-synthetic"
    path = run_synthetic(*fixture, run, allow_synthetic=True)
    assert path == run / METRICS_FILE
    assert [p for p in run.rglob("*") if p.is_file()] == [path]
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert set(row) == {
        "step",
        "time",
        "loss/mean",
        "loss/perplexity",
        "sft_cpu_probe/target_tokens",
        "sft_cpu_probe/trajectory_groups",
    }
    assert row["step"] == 1
    assert row["sft_cpu_probe/target_tokens"] == 3
    assert row["sft_cpu_probe/trajectory_groups"] == 2
    assert math.isfinite(row["loss/mean"])
    assert row["loss/perplexity"] == math.exp(row["loss/mean"])
    assert all(type(x) in (int, float) for x in row.values())
    with pytest.raises(FileExistsError):
        run_synthetic(*fixture, run, allow_synthetic=True)


def test_prime_probe_fails_closed(fixture, tmp_path, monkeypatch):
    run = tmp_path / "prime-synthetic"
    with pytest.raises(ValueError, match="production blocked"):
        run_synthetic(*fixture, run)
    with pytest.raises(ValueError, match="pin mismatch"):
        run_synthetic(fixture[0], "f" * 64, run, allow_synthetic=True)
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="single-process"):
        run_synthetic(*fixture, run, allow_synthetic=True)
    assert not run.exists()
