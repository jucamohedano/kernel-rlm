import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from kernel_rlm.cute_curriculum import STATUS, build_curriculum, validate_and_materialize


def code_for(name: str) -> str:
    return f"def reference_{name}(x):\n    return x\n"


def fixture_selection():
    codes = [
        code_for("relu_train"),
        code_for("relu_dev"),
        code_for("relu_control"),
        code_for("add"),
        code_for("mul"),
        code_for("sum"),
        code_for("relu_sum"),
        code_for("add_relu"),
        code_for("mul_relu"),
        code_for("sum_relu"),
        code_for("long"),
    ]
    source_bytes = b"pinned synthetic parquet bytes"
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    source = {
        "id": "synthetic",
        "repo": "example/data",
        "revision": "a" * 40,
        "file": "data.parquet",
        "sha256": source_hash,
        "code_column": "code",
        "original_ops_column": "ops",
    }
    specs = [
        (0, ["relu"], "relu-pointwise", {"shape": [8]}, "primitive", "train"),
        (1, ["relu"], "relu-pointwise", {"shape": [16]}, "primitive", "dev"),
        (2, ["relu"], "relu-pointwise", {"shape": [32]}, "primitive", "primitive_control"),
        (3, ["tensor_add"], "add", {"shape": [8]}, "primitive", "train"),
        (4, ["tensor_mul"], "mul", {"shape": [8]}, "primitive", "train"),
        (5, ["axis_sum"], "sum-axis-0", {"shape": [8], "axis": 0}, "primitive", "train"),
        (
            6,
            ["relu", "axis_sum"],
            "relu-sum",
            {"shape": [8], "axis": 0},
            "short_composition",
            "train",
        ),
        (7, ["tensor_add", "relu"], "add-relu", {"shape": [8]}, "short_composition", "dev"),
        (
            8,
            ["tensor_mul", "relu"],
            "mul-relu",
            {"shape": [8]},
            "short_composition",
            "composition_eval",
        ),
        (
            9,
            ["axis_sum", "relu"],
            "sum-relu",
            {"shape": [8], "axis": 0},
            "short_composition",
            "composition_eval",
        ),
        (
            10,
            ["tensor_add", "relu", "tensor_mul"],
            "long-graph",
            {"shape": [8]},
            "long_composition",
            "train",
        ),
    ]
    rows = []
    for code in codes:
        rows.append({"code": code, "ops": "upstream-label"})
    tasks = []
    for index, ops, group, contract, stage, split in specs:
        code = codes[index]
        shape = contract.get("shape", [8])
        contract["inputs"] = [
            {"shape": shape, "dtype": "float32"}
            for _ in range(2 if ops[0] in {"tensor_add", "tensor_mul"} else 1)
        ]
        graph_nodes = []
        for node_index, op in enumerate(ops):
            if node_index == 0:
                parents = [
                    ["input", input_index]
                    for input_index in range(2 if op in {"tensor_add", "tensor_mul"} else 1)
                ]
            elif op in {"tensor_add", "tensor_mul"}:
                parents = [["node", node_index - 1], ["input", 1]]
            else:
                parents = [["node", node_index - 1]]
            graph_nodes.append({"op": op, "parents": parents})
        contract = {
            **contract,
            "inputs": [
                {"shape": [8], "dtype": "float32"}
                for _ in range(2 if ops[0] in {"tensor_add", "tensor_mul"} else 1)
            ],
            "operations": [
                {"op": op, "parents": graph_nodes[index]["parents"], "config": {}}
                for index, op in enumerate(ops)
            ],
            "state": {},
            "output_shape": shape,
            "graph_signature": {
                "inputs": 2 if ops[0] in {"tensor_add", "tensor_mul"} else 1,
                "nodes": graph_nodes,
                "output": ["node", len(graph_nodes) - 1],
            },
        }
        tasks.append(
            {
                "source_id": "synthetic",
                "row_index": index,
                "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "code": code,
                "original_ops": "upstream-label",
                "verified_ops": ops,
                "graph_group": group,
                "contract": contract,
                "stage": stage,
                "split": split,
                "review_status": "source_reviewed",
                "historical_matches": [],
            }
        )
    selection = {
        "schema_version": 1,
        "sources": [source],
        "tasks": tasks,
        "required_primitives": ["tensor_add", "tensor_mul", "relu", "axis_sum"],
        "composition_holdouts": ["mul-relu", "sum-relu"],
        "composition_holdout_edges": [["tensor_mul", "relu"], ["axis_sum", "relu"]],
        "coverage": {"note": "input counts are untrusted and derived again"},
    }
    return selection, {"synthetic": rows}, {"synthetic": source_hash}, source_bytes


def validate(selection=None, rows=None, hashes=None):
    base, base_rows, base_hashes, _ = fixture_selection()
    return validate_and_materialize(selection or base, rows or base_rows, hashes or base_hashes)


def test_materialization_preserves_source_code_and_reports_static_only():
    selection, rows, hashes, _ = fixture_selection()
    output, manifest = validate_and_materialize(selection, rows, hashes)
    assert manifest["status"] == STATUS == "STATIC_VALIDATED_NOT_RUNTIME_VALIDATED"
    assert manifest["runtime_status"] == "not_run"
    assert manifest["validation"]["reference_code_executed"] is False
    assert manifest["coverage"]["blocked_reasons"] == ["primitive_controls_missing"]
    assert manifest["coverage"]["missing_primitive_controls"] == [
        "axis_sum",
        "tensor_add",
        "tensor_mul",
    ]
    assert manifest["counts"]["total"] == len(selection["tasks"])
    assert not manifest["coverage"]["empty_roles"]
    records = [row for group in output.values() for row in group]
    original_by_index = {index: row["code"] for index, row in enumerate(rows["synthetic"])}
    assert len(records) == len(selection["tasks"])
    assert all(row["code"] == original_by_index[row["source"]["row_index"]] for row in records)
    assert all(row["runtime_status"] == "not_run" for row in records)
    assert json.loads(records[0]["ops"]) == records[0]["verified_ops"]


def test_materializer_never_executes_reference_code():
    selection, rows, hashes, _ = fixture_selection()
    code = "raise RuntimeError('reference execution is forbidden')\n"
    rows["synthetic"][0]["code"] = code
    selection["tasks"][0]["code"] = code
    selection["tasks"][0]["code_sha256"] = hashlib.sha256(code.encode()).hexdigest()
    output, manifest = validate_and_materialize(selection, rows, hashes)
    emitted = [record for records in output.values() for record in records]
    assert any(record["code"] == code for record in emitted)
    assert manifest["validation"]["reference_code_executed"] is False


def test_mismatched_source_bytes_are_rejected():
    selection, rows, hashes, _ = fixture_selection()
    with pytest.raises(ValueError, match="source .* bytes do not match"):
        validate_and_materialize(selection, rows, {"synthetic": "0" * 64})


def test_source_row_hash_is_verified_instead_of_trusting_copy():
    selection, rows, hashes, _ = fixture_selection()
    selection["tasks"][0]["code"] = "def not_the_source():\n    pass\n"
    with pytest.raises(ValueError, match="copied code differs"):
        validate_and_materialize(selection, rows, hashes)


@pytest.mark.parametrize(
    ("task_index", "mutation", "message"),
    [
        (3, "bad_arity", "requires 2"),
        (6, "nonfinal_output", "final node"),
        (6, "dead_node", "do not contribute to output"),
    ],
)
def test_invalid_reviewed_graph_signature_is_rejected(task_index, mutation, message):
    selection, rows, hashes, _ = fixture_selection()
    task = selection["tasks"][task_index]
    signature = task["contract"]["graph_signature"]
    if mutation == "bad_arity":
        signature["nodes"][0]["parents"] = [["input", 0]]
    elif mutation == "nonfinal_output":
        signature["output"] = ["node", 0]
    else:
        signature["nodes"][1]["parents"] = [["input", 0]]
    for node, operation in zip(signature["nodes"], task["contract"]["operations"], strict=True):
        operation["parents"] = node["parents"]
    with pytest.raises(ValueError, match=message):
        validate_and_materialize(selection, rows, hashes)


def test_code_hash_and_original_metadata_are_verified():
    selection, rows, hashes, _ = fixture_selection()
    selection["tasks"][0]["code_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="code_sha256"):
        validate_and_materialize(selection, rows, hashes)
    selection, rows, hashes, _ = fixture_selection()
    selection["tasks"][0]["original_ops"] = "fabricated"
    with pytest.raises(ValueError, match="original_ops"):
        validate_and_materialize(selection, rows, hashes)


def test_row_index_rejects_booleans_and_negative_values():
    for bad_index in (True, -1):
        selection, rows, hashes, _ = fixture_selection()
        selection["tasks"][0]["row_index"] = bad_index
        with pytest.raises(ValueError, match="row_index"):
            validate_and_materialize(selection, rows, hashes)


def test_unreviewed_or_inconsistent_stage_tasks_are_rejected():
    selection, rows, hashes, _ = fixture_selection()
    selection["tasks"][0]["review_status"] = "pending"
    with pytest.raises(ValueError, match="not source_reviewed"):
        validate_and_materialize(selection, rows, hashes)
    selection, rows, hashes, _ = fixture_selection()
    selection["tasks"][3]["stage"] = "short_composition"
    with pytest.raises(ValueError, match="stage does not match"):
        validate_and_materialize(selection, rows, hashes)


def test_composition_graph_leakage_and_ast_duplicate_are_rejected():
    selection, rows, hashes, _ = fixture_selection()
    leak = dict(selection["tasks"][8])
    leak.update(
        row_index=11,
        code_sha256=hashlib.sha256(code_for("leak").encode()).hexdigest(),
        code=code_for("leak"),
        graph_group="renamed-mul-relu",
        split="train",
    )
    rows["synthetic"].append({"code": leak["code"], "ops": "upstream-label"})
    selection["tasks"].append(leak)
    with pytest.raises(
        ValueError, match="canonical graph signature is assigned to multiple graph_groups"
    ):
        validate_and_materialize(selection, rows, hashes)

    selection, rows, hashes, _ = fixture_selection()
    duplicate = dict(selection["tasks"][0])
    duplicate["code"] = duplicate["code"] + "\n"
    duplicate.update(
        row_index=11, code_sha256=hashlib.sha256(duplicate["code"].encode()).hexdigest()
    )
    rows["synthetic"].append({"code": duplicate["code"], "ops": "upstream-label"})
    selection["tasks"].append(duplicate)
    with pytest.raises(ValueError, match="AST-equivalent source reused"):
        validate_and_materialize(selection, rows, hashes)


def test_reserved_operation_edge_cannot_be_hidden_in_a_long_train_graph():
    selection, rows, hashes, _ = fixture_selection()
    task = copy.deepcopy(selection["tasks"][8])
    task["row_index"] = 11
    task["code"] = code_for("long_mul_relu_add")
    task["code_sha256"] = hashlib.sha256(task["code"].encode()).hexdigest()
    task["verified_ops"] = ["tensor_mul", "relu", "tensor_add"]
    task["graph_group"] = "long-mul-relu-add"
    task["stage"] = "long_composition"
    task["split"] = "train"
    signature = task["contract"]["graph_signature"]
    signature["nodes"].append({"op": "tensor_add", "parents": [["node", 1], ["input", 1]]})
    signature["output"] = ["node", 2]
    task["contract"]["operations"].append(
        {
            "op": "tensor_add",
            "parents": [["node", 1], ["input", 1]],
            "config": {},
        }
    )
    rows["synthetic"].append({"code": task["code"], "ops": "upstream-label"})
    selection["tasks"].append(task)
    with pytest.raises(ValueError, match="uses composition evaluation edge"):
        validate_and_materialize(selection, rows, hashes)


def test_missing_roles_and_holdouts_are_reported_not_fabricated():
    selection, rows, hashes, _ = fixture_selection()
    selection["tasks"] = [
        task for task in selection["tasks"] if task["split"] != "composition_eval"
    ]
    output, manifest = validate_and_materialize(selection, rows, hashes)
    assert output["eval/composition-heldout.jsonl"] == []
    assert manifest["coverage"]["composition_holdouts_missing"] == ["mul-relu", "sum-relu"]
    assert "composition_holdouts_missing" in manifest["coverage"]["blocked_reasons"]
    assert "roles_empty" in manifest["coverage"]["blocked_reasons"]


def test_build_is_deterministic_refuses_overwrite_and_emits_legacy_loader_rows(tmp_path):
    selection, rows, _, source_bytes = fixture_selection()
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    parquet_path = tmp_path / "synthetic.parquet"
    parquet_path.write_bytes(source_bytes)
    with patch("kernel_rlm.cute_curriculum.load_parquet_rows", return_value=rows["synthetic"]):
        first = build_curriculum(
            selection_path, tmp_path / "build-a", source_paths={"synthetic": parquet_path}
        )
        second = build_curriculum(
            selection_path, tmp_path / "build-b", source_paths={"synthetic": parquet_path}
        )
    relative_files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
    assert relative_files == sorted(
        path.relative_to(second) for path in second.rglob("*") if path.is_file()
    )
    assert all((first / rel).read_bytes() == (second / rel).read_bytes() for rel in relative_files)
    with patch("kernel_rlm.cute_curriculum.load_parquet_rows", return_value=rows["synthetic"]):
        with pytest.raises(FileExistsError):
            build_curriculum(selection_path, first, source_paths={"synthetic": parquet_path})

    # Reproduce the existing loader's input contract without importing the runtime.
    import sys

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "training/environments/kernel_rlm")
    )
    try:
        from kernel_rollout.tasks import load_tasks

        loaded = load_tasks(first / "train/primitives.jsonl")
    finally:
        sys.path.remove(
            str(Path(__file__).resolve().parents[1] / "training/environments/kernel_rlm")
        )
    assert loaded and all(task["context"] for task in loaded)
    assert "CUDA extension" in loaded[0]["prompt"]
    manifest = json.loads((first / "manifest.json").read_text())
    assert "NOT CuTe runtime-ready" in manifest["artifact_note"]
    assert manifest["selection_sha256"] == hashlib.sha256(selection_path.read_bytes()).hexdigest()
    assert "train/primitives.jsonl" in manifest["artifacts"]
    assert (
        manifest["artifacts"]["train/primitives.jsonl"]["size_bytes"]
        == (first / "train/primitives.jsonl").stat().st_size
    )
