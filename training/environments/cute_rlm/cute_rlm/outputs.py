"""Safe serialization and comparison for CPU/GPU output trees."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _array_aliases(arrays: list[np.ndarray]) -> bool:
    return any(
        np.shares_memory(first, second)
        for index, first in enumerate(arrays)
        for second in arrays[index + 1 :]
    )


def encode_tree(value: Any, arrays: dict[str, np.ndarray], prefix: str = "value") -> dict[str, Any]:
    """Encode supported Python containers/scalars and extract arrays by name.

    Array storage shared with another array in this tree is rejected rather than
    copied, since a copy would silently lose the original alias relationship.
    """
    found_arrays: list[np.ndarray] = []

    def encode(item: Any, name: str) -> dict[str, Any]:
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise TypeError("object-dtype arrays are unsupported")
            if any(np.shares_memory(item, other) for other in found_arrays):
                raise ValueError("arrays sharing memory are unsupported")
            if name in arrays:
                raise ValueError(f"array name collision: {name}")
            found_arrays.append(item)
            arrays[name] = item
            return {"type": "array", "name": name}
        if item is None:
            return {"type": "none"}
        if type(item) is bool:
            return {"type": "bool", "value": item}
        if type(item) is int:
            return {"type": "int", "value": item}
        if type(item) is float:
            if math.isfinite(item):
                return {"type": "float", "value": item}
            return {
                "type": "float",
                "value": "nan" if math.isnan(item) else ("inf" if item > 0 else "-inf"),
            }
        if type(item) is str:
            return {"type": "str", "value": item}
        if type(item) is list:
            return {
                "type": "list",
                "items": [encode(child, f"{name}_{index}") for index, child in enumerate(item)],
            }
        if type(item) is tuple:
            return {
                "type": "tuple",
                "items": [encode(child, f"{name}_{index}") for index, child in enumerate(item)],
            }
        if type(item) is dict:
            if not all(type(key) is str for key in item):
                raise TypeError("tree dictionaries must have string keys")
            return {
                "type": "dict",
                "items": [
                    [key, encode(child, f"{name}_{index}")]
                    for index, (key, child) in enumerate(item.items())
                ],
            }
        raise TypeError(f"unsupported tree value: {type(item).__name__}")

    return encode(value, prefix)


def _decode_tree(metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> Any:
    kind = metadata["type"]
    if kind == "none":
        return None
    if kind in ("bool", "int", "str"):
        return metadata["value"]
    if kind == "float":
        value = metadata["value"]
        return (
            {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}.get(value, value)
            if isinstance(value, str)
            else value
        )
    if kind == "array":
        return arrays[metadata["name"]]
    if kind == "list":
        return [_decode_tree(child, arrays) for child in metadata["items"]]
    if kind == "tuple":
        return tuple(_decode_tree(child, arrays) for child in metadata["items"])
    if kind == "dict":
        return {key: _decode_tree(child, arrays) for key, child in metadata["items"]}
    raise ValueError(f"unknown tree node type: {kind}")


def save_tree(path: str | Path, tree: Any) -> None:
    """Save a tree as compressed NPZ with JSON metadata and no pickle."""
    arrays: dict[str, np.ndarray] = {}
    metadata = encode_tree(tree, arrays)
    metadata_bytes = json.dumps(metadata, allow_nan=False, separators=(",", ":")).encode("utf-8")
    payload = {name: np.asarray(array) for name, array in arrays.items()}
    payload["__metadata__"] = np.frombuffer(metadata_bytes, dtype=np.uint8)
    with Path(path).open("wb") as file:
        np.savez_compressed(file, **payload)


def load_tree(path: str | Path) -> Any:
    """Load a tree from NPZ, explicitly disabling NumPy object pickle support."""
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(archive["__metadata__"].tobytes().decode("utf-8"))
        arrays = {name: archive[name].copy() for name in archive.files if name != "__metadata__"}
    return _decode_tree(metadata, arrays)


def compare_trees(expected: Any, actual: Any, atol: float, rtol: float) -> dict[str, Any]:
    """Compare output trees; float comparisons require finite values and tolerances."""
    if (
        isinstance(atol, bool)
        or isinstance(rtol, bool)
        or not math.isfinite(atol)
        or not math.isfinite(rtol)
        or atol < 0
        or rtol < 0
    ):
        raise ValueError("Comparison tolerances must be finite and nonnegative")
    max_abs_diff: float | None = None

    def mismatch(reason: str) -> dict[str, Any]:
        return {"correct": False, "reason": reason, "max_abs_diff": max_abs_diff}

    def compare(left: Any, right: Any, path: str) -> str | None:
        nonlocal max_abs_diff
        if type(left) is not type(right):
            return f"type mismatch at {path}: {type(left).__name__} != {type(right).__name__}"
        if isinstance(left, np.ndarray):
            if left.dtype != right.dtype:
                return f"dtype mismatch at {path}: {left.dtype} != {right.dtype}"
            if left.shape != right.shape:
                return f"shape mismatch at {path}: {left.shape} != {right.shape}"
            if left.dtype.hasobject or left.dtype.kind not in "fbiu":
                return f"unsupported array dtype at {path}: {left.dtype}"
            if left.dtype.kind == "f":
                if not np.isfinite(left).all() or not np.isfinite(right).all():
                    return f"non-finite float at {path}"
                if left.size:
                    difference = float(
                        np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))
                    )
                    max_abs_diff = (
                        difference if max_abs_diff is None else max(max_abs_diff, difference)
                    )
                if not np.allclose(left, right, atol=atol, rtol=rtol, equal_nan=False):
                    return f"float values differ at {path}"
                return None
            if not np.array_equal(left, right):
                return f"array values differ at {path}"
            return None
        if left is None:
            return None
        if type(left) is float:
            if not math.isfinite(left) or not math.isfinite(right):
                return f"non-finite float at {path}"
            difference = abs(left - right)
            max_abs_diff = difference if max_abs_diff is None else max(max_abs_diff, difference)
            if not math.isclose(left, right, abs_tol=atol, rel_tol=rtol):
                return f"float values differ at {path}"
            return None
        if type(left) in (bool, int, str):
            return None if left == right else f"value mismatch at {path}"
        if type(left) in (list, tuple):
            if len(left) != len(right):
                return f"length mismatch at {path}: {len(left)} != {len(right)}"
            for index, (lchild, rchild) in enumerate(zip(left, right, strict=True)):
                reason = compare(lchild, rchild, f"{path}[{index}]")
                if reason:
                    return reason
            return None
        if type(left) is dict:
            if left.keys() != right.keys():
                return f"dictionary keys differ at {path}"
            for key in left:
                reason = compare(left[key], right[key], f"{path}.{key}")
                if reason:
                    return reason
            return None
        return f"unsupported value at {path}: {type(left).__name__}"

    reason = compare(expected, actual, "value")
    return {
        "correct": reason is None,
        "reason": "match" if reason is None else reason,
        "max_abs_diff": max_abs_diff,
    }
