import hashlib
import json

import pytest
from cute_rlm.dataset import (
    load_tasks,
    prepare_tasks,
    select_tasks,
    task_context,
)
from cute_rlm.docs import docs_bootstrap, load_docs


def task(n):
    code = f"reference {n}"
    return {
        "task_id": f"t{n}",
        "reference_code": code,
        "description": "visible instruction",
        "source": {
            "repo": "source",
            "revision": "abc",
            "file": "tasks.parquet",
            "row": n,
            "sha256": hashlib.sha256(code.encode()).hexdigest(),
        },
    }


def test_selection_reproducible_uniform_pool_and_sorted():
    pool = [task(n) for n in range(500)]
    first = select_tasks(pool, 128, 991)
    assert first == select_tasks(pool, 128, 991)
    assert len(first) == 128
    assert [x["task_id"] for x in first] == sorted(x["task_id"] for x in first)
    assert select_tasks(pool, 500, 3) == sorted(pool, key=lambda x: x["task_id"])
    with pytest.raises(ValueError):
        select_tasks(pool, 501, 0)


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        select_tasks([task(1), task(1)], 1, 1)


def test_manifest_source_integrity_and_context_excludes_labels(tmp_path):
    rows = [task(n) for n in range(3)]
    output = tmp_path / "tasks.jsonl"
    # Exercise immutable canonical manifest serialization without reading source code.
    payload = json.dumps(rows[0], sort_keys=True, separators=(",", ":")) + "\n"
    output.write_text(payload)
    loaded = load_tasks(output)
    context = task_context(loaded[0], {"sha256": "docs-hash"})
    assert context["cute_dsl_task"] == "reference 0"
    assert "reference_code" not in context
    assert context["docs"] == {"sha256": "docs-hash"}
    assert not ({"answer", "reward_model", "ground_truth", "scorer"} & context.keys())
    loaded[0]["reference_code"] += " altered"
    with pytest.raises(ValueError, match="integrity"):
        task_context(loaded[0])


def test_docs_hash_pin_and_bounded_repl_helpers(tmp_path):
    doc = tmp_path / "cuda.txt"
    doc.write_text("CUDA CuTe layout and tensor documentation.\n" * 100)
    raw = doc.read_bytes()
    manifest = tmp_path / "docs.json"
    manifest.write_text(
        json.dumps(
            {
                "repo": "NVIDIA/cutlass",
                "revision": "a" * 40,
                "files": [
                    {"id": "cute", "path": "cuda.txt", "sha256": hashlib.sha256(raw).hexdigest()}
                ],
            }
        )
    )
    docs = load_docs(manifest)
    code = docs_bootstrap(docs)
    namespace = {}
    exec(code, namespace)
    assert namespace["search_docs"]("layout")[0]["id"] == "cute"
    assert len(namespace["read_doc"]("cute", chars=20)["excerpt"]) == 20
    with pytest.raises(ValueError):
        namespace["read_doc"]("../cuda.txt")
    doc.write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        load_docs(manifest)


def test_docs_reject_manifest_path_escape(tmp_path):
    manifest = tmp_path / "docs.json"
    manifest.write_text(
        json.dumps(
            {
                "repo": "r",
                "revision": "a" * 40,
                "files": [{"path": "../outside", "sha256": "0" * 64}],
            }
        )
    )
    with pytest.raises(ValueError, match="escapes"):
        load_docs(manifest)


def test_prepare_writes_hash_receipt_and_refuses_overwrite(tmp_path, monkeypatch):
    import cute_rlm.dataset as dataset

    monkeypatch.setattr(dataset, "load_pool", lambda *_a, **_kw: [task(n) for n in range(5)])
    output = tmp_path / "selection.jsonl"
    receipt = prepare_tasks("pinned.parquet", output, 2, 22)
    raw = output.read_bytes()
    assert receipt["task_count"] == 2
    assert receipt["runtime_executed"] is False
    assert receipt["manifest_sha256"] == hashlib.sha256(raw).hexdigest()
    assert len(load_tasks(output)) == 2
    with pytest.raises(FileExistsError):
        prepare_tasks("pinned.parquet", output, 2, 22)
