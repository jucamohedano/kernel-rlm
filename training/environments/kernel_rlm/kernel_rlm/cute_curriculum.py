"""Build statically reviewed CuTe curriculum artifacts from pinned Parquet rows.

This materializes source references and reviewed annotations; it does not produce
CuTe implementations or validate anything at runtime.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
OPERATIONS = {"tensor_add", "tensor_mul", "relu", "axis_sum"}
STAGES = {"primitive", "short_composition", "long_composition"}
SPLITS = {"train", "dev", "primitive_control", "composition_eval"}
STATUS = "STATIC_VALIDATED_NOT_RUNTIME_VALIDATED"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def source_identity(source: dict[str, Any]) -> dict[str, str]:
    required = ("id", "repo", "revision", "file", "sha256", "code_column")
    missing = [key for key in required if not isinstance(source.get(key), str) or not source[key]]
    if missing:
        raise ValueError(f"source entry is missing non-empty fields: {', '.join(missing)}")
    if len(source["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in source["sha256"]):
        raise ValueError(f"source {source['id']!r} has an invalid sha256")
    if len(source["revision"]) != 40 or any(
        c not in "0123456789abcdef" for c in source["revision"]
    ):
        raise ValueError(f"source {source['id']!r} revision must be a full lowercase commit hash")
    return {key: source[key] for key in required}


def task_id(source: dict[str, Any], row_index: int) -> str:
    identity = {key: source[key] for key in ("id", "repo", "revision", "file", "sha256")}
    identity["row_index"] = row_index
    digest = sha256_bytes(canonical_json(identity).encode("utf-8"))
    return f"cute-v1-{digest[:20]}"


def graph_signature(contract: dict[str, Any]) -> str:
    """Canonicalize reviewed computation topology, excluding config fields like axes/shapes."""
    signature = contract.get("graph_signature")
    if not isinstance(signature, dict):
        raise ValueError("contract.graph_signature must be an object")
    inputs = signature.get("inputs")
    if isinstance(inputs, bool) or not isinstance(inputs, int) or inputs <= 0:
        raise ValueError("graph_signature.inputs must be a positive integer")
    nodes = signature.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("graph_signature.nodes must be a non-empty list")
    canonical_nodes = []
    operations = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict) or node.get("op") not in OPERATIONS:
            raise ValueError(f"graph_signature node {index} has an unknown operation")
        parents = node.get("parents")
        if not isinstance(parents, list) or not parents:
            raise ValueError(f"graph_signature node {index} must have parent references")
        canonical_parents = []
        for parent in parents:
            if not isinstance(parent, list) or len(parent) != 2:
                raise ValueError(f"graph_signature node {index} has an invalid parent reference")
            kind, parent_index = parent
            if kind == "input":
                if (
                    isinstance(parent_index, bool)
                    or not isinstance(parent_index, int)
                    or not 0 <= parent_index < inputs
                ):
                    raise ValueError(f"graph_signature node {index} references an invalid input")
            elif kind == "node":
                if (
                    isinstance(parent_index, bool)
                    or not isinstance(parent_index, int)
                    or not 0 <= parent_index < index
                ):
                    raise ValueError(
                        f"graph_signature node {index} has a forward/cyclic node reference"
                    )
            else:
                raise ValueError(f"graph_signature node {index} has an unknown parent kind")
            canonical_parents.append([kind, parent_index])
        expected_parents = 2 if node["op"] in {"tensor_add", "tensor_mul"} else 1
        if len(canonical_parents) != expected_parents:
            raise ValueError(
                f"graph_signature node {index} has {len(canonical_parents)} parents; "
                f"{node['op']} requires {expected_parents}"
            )
        canonical_nodes.append({"op": node["op"], "parents": canonical_parents})
        operations.append(node["op"])
    output = signature.get("output")
    if not isinstance(output, list) or len(output) != 2:
        raise ValueError("graph_signature.output must be [input|node, index]")
    output_kind, output_index = output
    if output_kind == "input":
        if (
            isinstance(output_index, bool)
            or not isinstance(output_index, int)
            or not 0 <= output_index < inputs
        ):
            raise ValueError("graph_signature output references an invalid input")
        raise ValueError("graph_signature output must reference a computation node")
    if output_kind != "node" or isinstance(output_index, bool) or not isinstance(output_index, int):
        raise ValueError("graph_signature output has an invalid node reference")
    if output_index != len(nodes) - 1:
        raise ValueError("graph_signature output must reference the final node")
    reachable = {output_index}
    for index in range(output_index, -1, -1):
        if index not in reachable:
            continue
        reachable.update(
            parent_index
            for kind, parent_index in canonical_nodes[index]["parents"]
            if kind == "node"
        )
    if reachable != set(range(len(nodes))):
        raise ValueError("graph_signature contains nodes that do not contribute to output")
    return canonical_json(
        {"inputs": inputs, "nodes": canonical_nodes, "output": [output_kind, output_index]}
    )


def configuration_signature(contract: dict[str, Any]) -> str:
    """Keep only inputs, semantic op configs, output, and state from a reviewed contract."""
    fields = {key: contract.get(key) for key in ("inputs", "operations", "output_shape", "state")}
    return canonical_json(fields)


def ast_fingerprint(code: str, task_id: str) -> str:
    try:
        tree = ast.parse(code, filename=f"<{task_id}>")
    except SyntaxError as error:
        raise ValueError(f"{task_id}: source is not parseable Python: {error.msg}") from error
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return sha256_bytes(normalized.encode("utf-8"))


def read_selection(selection_data: bytes | Path) -> dict[str, Any]:
    try:
        raw = selection_data if isinstance(selection_data, bytes) else selection_data.read_bytes()
        selection = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read selection JSON: {error}") from error
    if (
        not isinstance(selection, dict)
        or not isinstance(selection.get("schema_version"), int)
        or isinstance(selection.get("schema_version"), bool)
        or selection["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError(f"selection schema_version must be {SCHEMA_VERSION}")
    if not isinstance(selection.get("sources"), list) or not isinstance(
        selection.get("tasks"), list
    ):
        raise ValueError("selection must contain sources and tasks arrays")
    if not isinstance(selection.get("coverage", {}), dict):
        raise ValueError("selection coverage must be an object")
    return selection


def validate_and_materialize(
    selection: dict[str, Any],
    source_rows: dict[str, list[dict[str, Any]]],
    source_hashes: dict[str, str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if (
        not isinstance(selection.get("schema_version"), int)
        or isinstance(selection.get("schema_version"), bool)
        or selection["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError(f"selection schema_version must be {SCHEMA_VERSION}")
    sources = selection.get("sources")
    tasks = selection.get("tasks")
    if not isinstance(sources, list) or not isinstance(tasks, list):
        raise ValueError("selection must contain sources and tasks arrays")
    if not tasks:
        raise ValueError("selection has no reviewed tasks")

    source_by_id: dict[str, dict[str, Any]] = {}
    for entry in sources:
        if not isinstance(entry, dict):
            raise ValueError("each source must be an object")
        identity = source_identity(entry)
        source_id = identity["id"]
        if source_id in source_by_id:
            raise ValueError(f"duplicate source id: {source_id}")
        actual_hash = source_hashes.get(source_id)
        if actual_hash != identity["sha256"]:
            raise ValueError(f"source {source_id!r} bytes do not match pinned sha256")
        if source_id not in source_rows:
            raise ValueError(f"no rows loaded for source {source_id!r}")
        source_by_id[source_id] = {**entry, **identity}
    if not source_by_id:
        raise ValueError("selection has no pinned sources")

    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"ops": None, "splits": defaultdict(list)}
    )
    output: dict[str, list[dict[str, Any]]] = {
        "train/primitives.jsonl": [],
        "train/compositions.jsonl": [],
        "dev/primitives.jsonl": [],
        "dev/compositions.jsonl": [],
        "eval/primitive-controls.jsonl": [],
        "eval/composition-heldout.jsonl": [],
    }
    seen_rows: set[tuple[str, int]] = set()
    seen_code_hashes: dict[str, str] = {}
    seen_ast: dict[str, str] = {}
    graph_group_signatures: dict[str, str] = {}
    signature_graph_groups: dict[str, str] = {}
    signature_splits: dict[str, set[str]] = defaultdict(set)
    annotations: list[dict[str, Any]] = []
    required_primitives = selection.get("required_primitives", sorted(OPERATIONS))
    holdouts = selection.get("composition_holdouts", [])
    holdout_edges = selection.get("composition_holdout_edges", [])
    if not isinstance(required_primitives, list) or any(
        op not in OPERATIONS for op in required_primitives
    ):
        raise ValueError("required_primitives must be a list of semantic operation names")
    if not isinstance(holdouts, list) or any(not isinstance(group, str) for group in holdouts):
        raise ValueError("composition_holdouts must be a list of graph_group strings")
    if len(set(holdouts)) != len(holdouts):
        raise ValueError("composition_holdouts contains duplicates")
    if not isinstance(holdout_edges, list) or any(
        not isinstance(edge, list)
        or len(edge) != 2
        or any(operation not in OPERATIONS for operation in edge)
        for edge in holdout_edges
    ):
        raise ValueError("composition_holdout_edges must be pairs of semantic operation names")
    if len({tuple(edge) for edge in holdout_edges}) != len(holdout_edges):
        raise ValueError("composition_holdout_edges contains duplicates")
    if holdouts and not holdout_edges:
        raise ValueError("composition_holdouts require explicit composition_holdout_edges")

    for item in tasks:
        if not isinstance(item, dict):
            raise ValueError("each task must be an object")
        required = (
            "source_id",
            "row_index",
            "code_sha256",
            "code",
            "original_ops",
            "verified_ops",
            "graph_group",
            "contract",
            "stage",
            "split",
            "review_status",
        )
        missing = [key for key in required if key not in item]
        if missing:
            raise ValueError(f"task is missing required fields: {', '.join(missing)}")
        source_id = item["source_id"]
        if source_id not in source_by_id:
            raise ValueError(f"task references unknown source {source_id!r}")
        row_index = item["row_index"]
        if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
            raise ValueError("row_index must be a non-negative integer")
        key = (source_id, row_index)
        if key in seen_rows:
            raise ValueError(f"duplicate source row selection: {source_id}:{row_index}")
        seen_rows.add(key)
        if item["review_status"] != "source_reviewed":
            raise ValueError(f"{source_id}:{row_index} is not source_reviewed")

        source = source_by_id[source_id]
        try:
            source_row = source_rows[source_id][row_index]
        except IndexError as error:
            raise ValueError(f"{source_id}:{row_index} is outside source rows") from error
        code_column = source["code_column"]
        code = source_row.get(code_column)
        if not isinstance(code, str) or not code.strip():
            raise ValueError(f"{source_id}:{row_index} has no source code in {code_column!r}")
        code_hash = sha256_bytes(code.encode("utf-8"))
        if item["code_sha256"] != code_hash:
            raise ValueError(f"{source_id}:{row_index} code_sha256 does not match source row")
        if "code" in item and item["code"] != code:
            raise ValueError(f"{source_id}:{row_index} copied code differs from source row")
        ops_column = source.get("original_ops_column", "ops")
        actual_original_ops = source_row.get(ops_column)
        if item["original_ops"] != actual_original_ops:
            raise ValueError(f"{source_id}:{row_index} original_ops does not match source metadata")

        verified_ops = item["verified_ops"]
        if (
            not isinstance(verified_ops, list)
            or not verified_ops
            or any(not isinstance(op, str) or op not in OPERATIONS for op in verified_ops)
        ):
            raise ValueError(f"{source_id}:{row_index} has invalid verified_ops")
        if item["stage"] not in STAGES:
            raise ValueError(f"{source_id}:{row_index} has invalid stage")
        expected_stage = (
            "primitive"
            if len(verified_ops) == 1
            else ("short_composition" if len(verified_ops) == 2 else "long_composition")
        )
        if item["stage"] != expected_stage:
            raise ValueError(f"{source_id}:{row_index} stage does not match verified_ops length")
        split = item["split"]
        if split not in SPLITS:
            raise ValueError(f"{source_id}:{row_index} has invalid split")
        if not isinstance(item["graph_group"], str) or not item["graph_group"].strip():
            raise ValueError(f"{source_id}:{row_index} needs a non-empty graph_group")
        if not isinstance(item["contract"], dict):
            raise ValueError(f"{source_id}:{row_index} contract must be an object")
        if not isinstance(item["contract"].get("inputs"), list):
            raise ValueError(f"{source_id}:{row_index} contract.inputs must be a list")
        graph_signature_value = graph_signature(item["contract"])
        signature_object = json.loads(graph_signature_value)
        if signature_object["inputs"] != len(item["contract"]["inputs"]):
            raise ValueError(
                f"{source_id}:{row_index} graph_signature input count does not match contract.inputs"
            )
        signature_operations = signature_object["nodes"]
        if [node["op"] for node in signature_operations] != verified_ops:
            raise ValueError(
                f"{source_id}:{row_index} verified_ops do not match graph_signature nodes"
            )
        contract_operations = item["contract"].get("operations")
        if not isinstance(contract_operations, list):
            raise ValueError(f"{source_id}:{row_index} contract.operations must be a list")
        stripped_operations = [
            {"op": node.get("op"), "parents": node.get("parents")} for node in contract_operations
        ]
        if stripped_operations != [
            {"op": node["op"], "parents": node["parents"]} for node in signature_operations
        ]:
            raise ValueError(
                f"{source_id}:{row_index} graph_signature does not match contract.operations"
            )
        if split == "primitive_control" and item["stage"] != "primitive":
            raise ValueError("primitive_control may contain only primitive tasks")
        if split == "composition_eval" and item["stage"] == "primitive":
            raise ValueError("composition_eval may contain only composition tasks")

        graph_group = item["graph_group"]
        known_signature = graph_group_signatures.setdefault(graph_group, graph_signature_value)
        if known_signature != graph_signature_value:
            raise ValueError(
                f"graph_group {graph_group!r} contains different canonical graph signatures"
            )
        known_group = signature_graph_groups.setdefault(graph_signature_value, graph_group)
        if known_group != graph_group:
            raise ValueError(
                f"canonical graph signature is assigned to multiple graph_groups: {known_group!r}, {graph_group!r}"
            )
        signature_splits[graph_signature_value].add(split)
        edges = [
            [parent["op"], node["op"]]
            for node_index, node in enumerate(signature_object["nodes"])
            for kind, parent_index in node["parents"]
            if kind == "node"
            for parent in [signature_object["nodes"][parent_index]]
        ]
        forbidden = [edge for edge in holdout_edges if edge in edges]
        if split != "composition_eval" and forbidden:
            raise ValueError(
                f"{source_id}:{row_index} uses composition evaluation edge {forbidden[0]} outside evaluation"
            )
        if (
            split == "composition_eval"
            and len(verified_ops) > 1
            and not any(edge in holdout_edges for edge in edges)
        ):
            raise ValueError(
                f"{source_id}:{row_index} composition_eval has no declared held-out operation edge"
            )
        graph = groups[graph_group]
        ops_tuple = tuple(verified_ops)
        if graph["ops"] is not None and graph["ops"] != ops_tuple:
            raise ValueError(f"graph_group {item['graph_group']!r} has inconsistent verified_ops")
        graph["ops"] = ops_tuple
        graph["splits"][split].append(item)

        record_id = task_id(source, row_index)
        ast_hash = ast_fingerprint(code, record_id)
        if code_hash in seen_code_hashes:
            raise ValueError(
                f"exact source code reused by {record_id} and {seen_code_hashes[code_hash]}"
            )
        if ast_hash in seen_ast:
            raise ValueError(
                f"AST-equivalent source reused by {record_id} and {seen_ast[ast_hash]}"
            )
        seen_code_hashes[code_hash] = record_id
        seen_ast[ast_hash] = record_id

        record = {
            "_task_id": record_id,
            "ops": canonical_json(verified_ops),
            "verified_ops": verified_ops,
            "original_ops": item["original_ops"],
            "code": code,
            "stage": item["stage"],
            "split": split,
            "source": {
                "id": source["id"],
                "repo": source["repo"],
                "revision": source["revision"],
                "file": source["file"],
                "sha256": source["sha256"],
                "row_index": row_index,
                "code_sha256": code_hash,
            },
            "graph_group": item["graph_group"],
            "graph_signature": json.loads(graph_signature_value),
            "contract": item["contract"],
            "review_status": "source_reviewed",
            "runtime_status": "not_run",
            "historical_matches": item.get("historical_matches", []),
        }
        if "config_id" in item:
            record["config_id"] = item["config_id"]
        if "annotations" in item:
            record["annotations"] = item["annotations"]
        if "candidate_hints" in item:
            record["candidate_hints"] = item["candidate_hints"]
        if item["stage"] == "primitive":
            if split == "train":
                output["train/primitives.jsonl"].append(record)
            elif split == "dev":
                output["dev/primitives.jsonl"].append(record)
            else:
                output["eval/primitive-controls.jsonl"].append(record)
        else:
            if split == "train":
                output["train/compositions.jsonl"].append(record)
            elif split == "dev":
                output["dev/compositions.jsonl"].append(record)
            else:
                output["eval/composition-heldout.jsonl"].append(record)
        annotations.append(record)

    # Composition graph families are partitioned strictly; primitive variants can
    # share a graph only as distinct, explicitly described configuration controls.
    composition_graphs: dict[str, set[str]] = defaultdict(set)
    primitive_graphs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for graph_group, graph in groups.items():
        ops = graph["ops"]
        for split, rows in graph["splits"].items():
            if len(ops) == 1:
                primitive_graphs[graph_group].extend(rows)
            else:
                composition_graphs[graph_group].add(split)
    for graph_signature_value, splits in signature_splits.items():
        if len(json.loads(graph_signature_value)["nodes"]) > 1 and len(splits) > 1:
            group = signature_graph_groups[graph_signature_value]
            raise ValueError(f"composition graph signature {group!r} leaks across splits")
    train_composition_groups = {g for g, s in composition_graphs.items() if "train" in s}
    dev_composition_groups = {g for g, s in composition_graphs.items() if "dev" in s}
    eval_composition_groups = {g for g, s in composition_graphs.items() if "composition_eval" in s}
    if train_composition_groups & dev_composition_groups:
        raise ValueError("composition graph_group leaks between train and dev")
    if (
        train_composition_groups & eval_composition_groups
        or dev_composition_groups & eval_composition_groups
    ):
        raise ValueError("composition evaluation graph overlaps train/dev")
    unexpected_eval = eval_composition_groups - set(holdouts)
    if unexpected_eval:
        raise ValueError(
            f"composition_eval contains groups not declared in composition_holdouts: {sorted(unexpected_eval)}"
        )
    absent_holdouts = sorted(set(holdouts) - eval_composition_groups)
    for group, rows in primitive_graphs.items():
        configurations_by_split: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            configurations_by_split[row["split"]].add(configuration_signature(row["contract"]))
        if "primitive_control" in configurations_by_split:
            for split in ("train", "dev"):
                if configurations_by_split["primitive_control"] & configurations_by_split[split]:
                    raise ValueError(
                        f"primitive graph_group {group!r} reuses a configuration between "
                        f"primitive_control and {split}"
                    )

    # The selection's coverage block is provenance only. Counts and missing roles
    # below are always computed from verified materialized rows.
    stage_counts = Counter(row["stage"] for row in annotations)
    split_counts = Counter(row["split"] for row in annotations)
    op_counts = Counter(op for row in annotations for op in row["verified_ops"])
    stage_counts_by_split = {
        split: {
            stage: sum(row["stage"] == stage and row["split"] == split for row in annotations)
            for stage in sorted(STAGES)
        }
        for split in sorted(SPLITS)
    }
    source_counts = {
        source_id: sum(row["source"]["id"] == source_id for row in annotations)
        for source_id in sorted(source_by_id)
    }
    primitive_ops_by_split = {
        split: {
            op
            for row in annotations
            if row["stage"] == "primitive" and row["split"] == split
            for op in row["verified_ops"]
        }
        for split in ("train", "dev", "primitive_control", "composition_eval")
    }
    missing_train_primitives = sorted(set(required_primitives) - primitive_ops_by_split["train"])
    missing_control_primitives = sorted(
        set(required_primitives) - primitive_ops_by_split["primitive_control"]
    )
    missing_primitives = sorted(
        set(required_primitives)
        - {op for row in annotations if row["stage"] == "primitive" for op in row["verified_ops"]}
    )
    missing_roles = [role for role in SPLITS if split_counts[role] == 0]
    configuration_controls = []
    for group, info in sorted(groups.items()):
        if len(info["ops"]) != 1 or "primitive_control" not in info["splits"]:
            continue
        by_split: dict[str, set[str]] = defaultdict(set)
        for split, rows in info["splits"].items():
            for row in rows:
                by_split[split].add(configuration_signature(row["contract"]))
        configuration_controls.append(
            {
                "graph_group": group,
                "is_unseen_graph": False,
                "unique_configurations_by_split": {
                    split: len(configs) for split, configs in sorted(by_split.items())
                },
            }
        )
    blocked = []
    if missing_primitives:
        blocked.append("required_primitives_missing")
    if missing_train_primitives:
        blocked.append("train_primitives_missing")
    if missing_control_primitives:
        blocked.append("primitive_controls_missing")
    if not primitive_ops_by_split["dev"]:
        blocked.append("dev_primitive_stage_empty")
    if absent_holdouts:
        blocked.append("composition_holdouts_missing")
    if missing_roles:
        blocked.append("roles_empty")
    if not any(row["stage"] == "short_composition" for row in annotations):
        blocked.append("short_composition_stage_empty")
    if stage_counts_by_split["train"]["short_composition"] == 0:
        blocked.append("train_short_compositions_empty")
    if not any(row["stage"] == "long_composition" for row in annotations):
        blocked.append("long_composition_stage_empty")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "eligibility": "pending_selection_gaps" if blocked else "eligible_after_runtime_gate",
        "training_approved": False,
        "runtime_status": "not_run",
        "artifact_note": (
            "JSONL rows are structurally compatible with the current kernel rollout task loader, "
            "but contain original reference Python, not CuTe implementations. They are NOT CuTe "
            "runtime-ready or training-approved. The current loader prompt requests a CUDA extension; "
            "CuTe prompt/verifier integration is separate and not included."
        ),
        "validation": {
            "source_hashes_verified": True,
            "source_row_code_hashes_verified": True,
            "source_code_preserved": True,
            "exact_code_and_ast_duplicates_rejected": True,
            "composition_graph_split_leakage_rejected": True,
            "canonical_graph_signature_checked_from_reviewed_contract": True,
            "signature_is_not_derived_from_reference_code": True,
            "primitive_graph_reuse_limited_to_distinct_contract_configurations": True,
            "reference_code_executed": False,
            "runtime_validation_performed": False,
        },
        "sources": [
            {
                key: source[key]
                for key in ("id", "repo", "revision", "file", "sha256", "code_column")
            }
            | (
                {"original_ops_column": source["original_ops_column"]}
                if "original_ops_column" in source
                else {}
            )
            for source in sorted(source_by_id.values(), key=lambda value: value["id"])
        ],
        "counts": {
            "total": len(annotations),
            "by_stage": {stage: stage_counts[stage] for stage in sorted(STAGES)},
            "by_split": {split: split_counts[split] for split in sorted(SPLITS)},
            "by_stage_by_split": stage_counts_by_split,
            "by_source": source_counts,
            "by_verified_op_occurrence": {op: op_counts[op] for op in sorted(OPERATIONS)},
        },
        "coverage": {
            "required_primitives": required_primitives,
            "missing_primitives": missing_primitives,
            "primitive_train_coverage": sorted(primitive_ops_by_split["train"]),
            "missing_train_primitives": missing_train_primitives,
            "primitive_control_coverage": sorted(primitive_ops_by_split["primitive_control"]),
            "missing_primitive_controls": missing_control_primitives,
            "dev_primitive_coverage": sorted(primitive_ops_by_split["dev"]),
            "dev_primitive_stage_empty": not bool(primitive_ops_by_split["dev"]),
            "composition_holdouts": holdouts,
            "composition_holdout_edges": holdout_edges,
            "composition_holdouts_present": sorted(eval_composition_groups),
            "composition_holdouts_missing": absent_holdouts,
            "empty_roles": missing_roles,
            "primitive_configuration_controls": configuration_controls,
            "primitive_control_interpretation": (
                "Controls share their primitive graph with training and are configuration holdouts, "
                "not unseen-graph evaluation."
            ),
            "graph_signature_interpretation": (
                "Canonical topology comes from the reviewed task contract, not source-code inference; "
                "axes, keepdim, and other config are excluded from graph identity."
            ),
            "blocked_reasons": sorted(set(blocked)),
            "source_coverage_input_preserved": selection.get("coverage", {}),
        },
        "graph_groups": {
            group: {
                "verified_ops": list(info["ops"]),
                "kind": "primitive_graph" if len(info["ops"]) == 1 else "composition_graph",
                "splits": sorted(info["splits"]),
                "task_ids": sorted(
                    task_id(source_by_id[row["source_id"]], row["row_index"])
                    for row in sum((list(v) for v in info["splits"].values()), [])
                ),
            }
            for group, info in sorted(groups.items())
        },
    }
    for rows in output.values():
        rows.sort(key=lambda row: row["_task_id"])
    return output, manifest


def load_parquet_rows(
    source: dict[str, Any], path: Path, raw: bytes | None = None
) -> list[dict[str, Any]]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "building from Parquet requires pyarrow in the selected project environment"
        ) from error
    columns = [source["code_column"]]
    ops_column = source.get("original_ops_column", "ops")
    if ops_column not in columns:
        columns.append(ops_column)
    try:
        source_file = pa.BufferReader(raw) if raw is not None else path
        table = pq.read_table(source_file, columns=columns)
    except Exception as error:
        raise ValueError(f"could not read required columns from {path}: {error}") from error
    return table.to_pylist()


def build_curriculum(
    selection_path: str | Path,
    output_dir: str | Path,
    *,
    source_paths: dict[str, str | Path] | None = None,
) -> Path:
    selection_file = Path(selection_path)
    selection_bytes = selection_file.read_bytes()
    selection = read_selection(selection_bytes)
    source_paths = source_paths or {}
    source_rows: dict[str, list[dict[str, Any]]] = {}
    source_hashes: dict[str, str] = {}
    for source in selection["sources"]:
        source_identity(source)
        source_id = source["id"]
        raw_path = source_paths.get(source_id, source.get("path"))
        if raw_path is None:
            raise ValueError(
                f"source {source_id!r} needs a local path or --source {source_id}=PATH"
            )
        path = Path(raw_path)
        raw = path.read_bytes()
        source_hashes[source_id] = sha256_bytes(raw)
        source_rows[source_id] = load_parquet_rows(source, path, raw)
    artifacts, manifest = validate_and_materialize(selection, source_rows, source_hashes)

    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        for relative, rows in artifacts.items():
            path = temp_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
        (temp_dir / "README.md").write_text(artifact_readme(manifest), encoding="utf-8")
        manifest["selection_sha256"] = sha256_bytes(selection_bytes)
        manifest["artifacts"] = {
            path.relative_to(temp_dir).as_posix(): {
                "sha256": sha256_bytes(path.read_bytes()),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(temp_dir.rglob("*"))
            if path.is_file()
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_dir, target)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return target


def artifact_readme(manifest: dict[str, Any]) -> str:
    return f"""# CuTe curriculum v1 materialization

Status: **{manifest["status"]}**. Runtime validation was not performed.

The JSONL files preserve original reviewed reference Python and carry verified semantic annotations. Their record shape is structurally compatible with the current kernel rollout task loader, but that loader's current instruction asks for a CUDA extension. These artifacts are **not CuTe runtime-ready and not approved for training**. CuTe prompt/verifier integration and runtime validation require separate work and approval.

The builder verifies pinned Parquet bytes and source-row code hashes, reads reference code directly from the Parquet source, and never imports or executes reference code. See `manifest.json` for source pins, derived counts, graph partitions, coverage gaps, and checks.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path, help="reviewed selection JSON")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="new output directory; existing paths are refused",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="local Parquet input (may be repeated)",
    )
    args = parser.parse_args(argv)
    source_paths: dict[str, str] = {}
    for value in args.source:
        if "=" not in value:
            parser.error("--source must be ID=PATH")
        source_id, path = value.split("=", 1)
        if not source_id or not path or source_id in source_paths:
            parser.error("--source IDs and paths must be non-empty and source IDs unique")
        source_paths[source_id] = path
    try:
        result = build_curriculum(args.selection, args.output, source_paths=source_paths)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"cute curriculum build failed: {error}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
