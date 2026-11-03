"""Pinned cudaLLM task preparation; reference source is visible specification, not oracle."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DATASET_REPO = "ByteDance-Seed/cudaLLM-data"
DATASET_REVISION = "8387cbead988c758f204f4ec9feaf0318f12a3a5"
DATASET_FILE = "rl_cuda_llm_0424.parquet"
DATASET_SHA256 = "a6f8a1562d25b4d00a1a8e47fc355b1e9bd367881a3dc62c85380c53d57c1b58"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _row_code(row: dict[str, Any]) -> str:
    code = row.get("py_code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("cudaLLM row has no py_code source")
    return code


def load_pool(
    input_path: str | Path, *, expected_sha256: str = DATASET_SHA256
) -> list[dict[str, Any]]:
    """Read original parquet row values; no source is imported, evaluated, or executed."""
    path = Path(input_path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("cudaLLM parquet checksum mismatch")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Install the cute_rlm project dependencies (pyarrow) to inspect parquet"
        ) from exc
    parquet = pq.read_table(path, columns=["py_code", "ops", "data_source", "ability", "prompt"])
    tasks = []
    for row_number, row in enumerate(parquet.to_pylist()):
        code = _row_code(row)
        source_hash = sha256_bytes(code.encode("utf-8"))
        task_id = f"cudaLLM-{row_number:05d}-{source_hash[:12]}"
        tasks.append(
            {
                "task_id": task_id,
                "reference_code": code,
                "description": row.get("ops") or "",
                "source": {
                    "repo": DATASET_REPO,
                    "revision": DATASET_REVISION,
                    "file": DATASET_FILE,
                    "row": row_number,
                    "sha256": source_hash,
                },
            }
        )
    return tasks


def select_tasks(rows: Iterable[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Uniform sample without replacement across all rows; stable task-id output order."""
    pool = list(rows)
    if type(count) is not int or count <= 0 or count > len(pool):
        raise ValueError(f"count must be between 1 and pool size ({len(pool)})")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    ids = [task.get("task_id") for task in pool]
    if any(not isinstance(task_id, str) or not task_id for task_id in ids):
        raise ValueError("every row must have a task_id")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task_id values")
    chosen = random.Random(seed).sample(pool, count)
    return sorted(chosen, key=lambda task: task["task_id"])


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    tasks = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [task.get("task_id") for task in tasks]
    if not tasks or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("manifest must contain task records with task_id")
    if len(set(ids)) != len(ids):
        raise ValueError("manifest contains duplicate task_id values")
    for task in tasks:
        source = task.get("source", {})
        code = task.get("reference_code")
        if not isinstance(code, str) or source.get("sha256") != sha256_bytes(code.encode("utf-8")):
            raise ValueError(f"source hash mismatch for task {task['task_id']}")
    return tasks


def task_context(
    task: dict[str, Any], docs_metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Model-visible task prompt. Intentionally excludes labels, scorer and hidden outputs."""
    source = task.get("source", {})
    code = task.get("reference_code")
    if not isinstance(code, str) or source.get("sha256") != sha256_bytes(code.encode("utf-8")):
        raise ValueError("task source integrity check failed")
    result = {
        "task_id": task["task_id"],
        "cute_dsl_task": code,
        "description": task.get("description", ""),
        "source": dict(source),
    }
    if docs_metadata is not None:
        result["docs"] = docs_metadata
    return result


def prepare_tasks(
    input_path: str | Path,
    output_path: str | Path,
    count: int,
    seed: int,
    *,
    expected_sha256: str = DATASET_SHA256,
) -> dict[str, Any]:
    """Write immutable JSONL selection and adjacent integrity receipt; never runs task code."""
    input_path, output_path = Path(input_path), Path(output_path)
    receipt_path = output_path.with_suffix(output_path.suffix + ".receipt.json")
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite task manifest or receipt")
    tasks = select_tasks(load_pool(input_path, expected_sha256=expected_sha256), count, seed)
    payload = "".join(
        json.dumps(task, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        for task in tasks
    ).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    receipt = {
        "schema": "cute-task-manifest-receipt-v1",
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "dataset_file": DATASET_FILE,
        "dataset_sha256": expected_sha256,
        "task_count": len(tasks),
        "seed": seed,
        "manifest_sha256": sha256_bytes(payload),
        "manifest_bytes": len(payload),
        "task_ids": [task["task_id"] for task in tasks],
        "runtime_executed": False,
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_tasks(args.parquet, args.output, args.count, args.seed), indent=2))


if __name__ == "__main__":
    main()
