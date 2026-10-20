import hashlib
import json

import pytest
from kernel_trl import checkpoint_continuation as module

from tests import test_kernel_loading_repair as fixtures


@pytest.fixture
def ready(tmp_path, monkeypatch):
    arguments = fixtures.prepared.__wrapped__(tmp_path, monkeypatch)
    from kernel_trl import loading_repair

    monkeypatch.setattr(module, "load_tasks", loading_repair.load_tasks)
    arguments.parent_source_root = tmp_path / "old-source"
    old_hashes = {"runtime.py": "same", "kernel_rlm/child_tools.py": "old"}
    new_hashes = {
        **old_hashes,
        "kernel_rlm/child_tools.py": "new",
        "kernel_trl/checkpoint_continuation.py": "new",
    }
    monkeypatch.setattr(
        module,
        "source_hashes",
        lambda path: old_hashes if path == arguments.parent_source_root else new_hashes,
    )
    monkeypatch.setattr(
        module, "docs_manifest", lambda *args: {"target_profile": "sm120", "sha256": "docs"}
    )
    inputs = module.read_json(arguments.parent_run / "inputs.json")
    inputs.update(
        source_sha256=old_hashes,
        cuda_docs={"path": str(arguments.cuda_docs), "target_profile": "sm120", "sha256": "docs"},
    )
    state = module.read_json(arguments.parent_run / "state.json")
    state.update(phase="budget_gate", budget_gate="Original phase reserve exhausted")
    for key in ("pending_task_id", "pending_directory", "failure"):
        state.pop(key, None)
    state["manifest_sha256"] = hashlib.sha256(
        json.dumps(inputs, sort_keys=True).encode()
    ).hexdigest()
    module.write_json(arguments.parent_run / "inputs.json", inputs)
    module.write_json(arguments.parent_run / "state.json", state)
    return arguments


def test_clean_budget_continuation_preserves_coverage_and_parent(ready):
    original = module.read_json(ready.parent_run / "state.json")
    original_hash = module.digest(ready.parent_run / "state.json")
    result = module.prepare(ready)
    assert result["status"] == "ready"
    assert result["policy_version"] == 3
    for key in (
        "completed_task_ids",
        "results",
        "deferred_task_ids",
        "attempts",
        "checkpoint",
        "task_order",
    ):
        assert result[key] == original[key]
    assert module.digest(ready.parent_run / "state.json") == original_hash
    assert not (ready.output_dir / "groups").exists()
    receipt = module.read_json(ready.output_dir / "provenance/checkpoint-continuation.json")
    assert receipt["new_rollouts"] == 0
    assert receipt["no_pending_group_discarded"] is True
    inputs = module.read_json(ready.output_dir / "inputs.json")
    assert (
        result["manifest_sha256"]
        == hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    )


@pytest.mark.parametrize(
    "change",
    ["pending", "active", "failure", "phase", "dataset", "source", "docs", "native", "coverage"],
)
def test_budget_continuation_rejects_unsafe_changes(ready, change, monkeypatch):
    state = module.read_json(ready.parent_run / "state.json")
    if change == "pending":
        state["pending_task_id"] = "task-3"
    elif change == "active":
        state["status"] = "running"
    elif change == "failure":
        state["failure"] = {"error": "Unresolved infrastructure"}
    elif change == "phase":
        state["phase"] = "update"
    elif change == "dataset":
        ready.dataset.write_text("changed")
    elif change == "source":
        previous = module.source_hashes
        monkeypatch.setattr(
            module,
            "source_hashes",
            lambda path: previous(path)
            if path == ready.parent_source_root
            else {**previous(path), "kernel_rlm/verifier.py": "changed"},
        )
    elif change == "docs":
        monkeypatch.setattr(
            module, "docs_manifest", lambda *args: {"target_profile": "sm120", "sha256": "changed"}
        )
    elif change == "native":
        path = (
            module.Path(state["results"][-1]["update_directory"]).parent
            / "readback/report-rank-0.json"
        )
        native = module.read_json(path)
        native["resume"]["momentum_shards_restored"] = 0
        module.write_json(path, native)
    else:
        state["completed_task_ids"].reverse()
    module.write_json(ready.parent_run / "state.json", state)
    with pytest.raises(ValueError):
        module.prepare(ready)
    assert not ready.output_dir.exists()
