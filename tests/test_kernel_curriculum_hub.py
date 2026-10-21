import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "training/environments/kernel_rlm"))
from kernel_rlm import curriculum


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "input"
    directory.mkdir()
    for filename in curriculum.CONFIG_FILES.values():
        rows = [
            {"_task_id": "task-2", "code": "print('second')\n", "ops": '["torch.add"]'},
            {"_task_id": "task-1", "code": "print('first')\n", "ops": '["torch.add"]'},
        ]
        (directory / filename).write_bytes(
            ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
        )
    return directory


@pytest.fixture
def release(source, tmp_path):
    pytest.importorskip("datasets")
    directory = tmp_path / "release"
    manifest = curriculum.build_release(source, directory)
    return directory, manifest


def mock_hub(monkeypatch, directory):
    calls = []

    def download(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        return str(directory / filename)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    return calls


def test_export_preserves_original_bytes_and_parquet_rows(source, release):
    from datasets import Dataset

    directory, manifest = release
    assert manifest["upstream_revision"] is None
    for config, filename in curriculum.CONFIG_FILES.items():
        record = manifest["configs"][config]
        original = (source / filename).read_bytes()
        assert (directory / record["jsonl_path"]).read_bytes() == original
        assert record["jsonl_sha256"] == hashlib.sha256(original).hexdigest()
        assert Dataset.from_parquet(
            str(directory / record["parquet_path"])
        ).to_list() == curriculum.read_rows(original)
    assert "alternate orders" in (directory / "README.md").read_text()
    with pytest.raises(FileExistsError):
        curriculum.build_release(source, directory)


def test_hub_download_keeps_legacy_loader_contract(release, monkeypatch, tmp_path):
    from kernel_skyrl.preflight import load_tasks

    directory, manifest = release
    config = "lvl1_seed42_easy_first"
    record = manifest["configs"][config]
    calls = mock_hub(monkeypatch, directory)
    output = tmp_path / "downloaded/tasks.jsonl"
    result = curriculum.materialize("owner/data", config, "a" * 40, record["jsonl_sha256"], output)
    assert result == output
    assert output.read_bytes() == (directory / record["jsonl_path"]).read_bytes()
    assert [task["name"] for task in load_tasks(output)] == ["task-2", "task-1"]
    assert all(call[2] == {"repo_type": "dataset", "revision": "a" * 40} for call in calls)
    assert (
        curriculum.materialize("owner/data", config, "a" * 40, record["jsonl_sha256"], output)
        == output
    )
    assert len(calls) == 2


@pytest.mark.parametrize("revision", ["main", "v1", "abc123", ""])
def test_download_rejects_mutable_revisions(tmp_path, revision):
    with pytest.raises(ValueError, match="immutable"):
        curriculum.materialize("owner/data", "lvl1_seed42", revision, "a" * 64, tmp_path / "out")


def test_download_refuses_changed_existing_file(tmp_path):
    output = tmp_path / "out"
    output.write_text("keep me")
    with pytest.raises(ValueError, match="overwrite"):
        curriculum.materialize("owner/data", "lvl1_seed42", "a" * 40, "b" * 64, output)
    assert output.read_text() == "keep me"


@pytest.mark.parametrize("mutation", ["bytes", "order", "path", "schema"])
def test_download_fails_closed_on_tampering(release, monkeypatch, tmp_path, mutation):
    directory, manifest = release
    config = "lvl1_seed42"
    record = manifest["configs"][config]
    expected = record["jsonl_sha256"]
    if mutation == "bytes":
        (directory / record["jsonl_path"]).write_text("tampered")
    elif mutation == "order":
        record["ordered_task_ids_sha256"] = "0" * 64
    elif mutation == "path":
        record["jsonl_path"] = "../outside"
    else:
        manifest["schema_version"] = 2
    (directory / "manifest.json").write_text(json.dumps(manifest))
    mock_hub(monkeypatch, directory)
    output = tmp_path / "out"
    with pytest.raises(ValueError):
        curriculum.materialize("owner/data", config, "a" * 40, expected, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"_task_id": "x", "code": ""}],
        [{"code": "source"}],
        [{"_task_id": "x", "code": "source"}] * 2,
    ],
)
def test_invalid_task_contracts_rejected(rows):
    content = "\n".join(json.dumps(row) for row in rows).encode()
    with pytest.raises(ValueError):
        curriculum.read_rows(content)


def test_export_rejects_different_l1_tasks_before_writing(source, tmp_path):
    pytest.importorskip("datasets")
    path = source / curriculum.CONFIG_FILES["lvl1_seed42_easy_first"]
    rows = curriculum.read_rows(path.read_bytes())
    rows[0]["code"] = "different reference"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="same task records"):
        curriculum.build_release(source, output)
    assert not output.exists()
