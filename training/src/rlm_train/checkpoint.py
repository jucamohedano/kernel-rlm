"""Versioned JSON checkpoints for opt-in native episode recovery.

Only explicit REPL snapshots are persisted. No pickle or action replay is used.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace the active episode checkpoint and flush it to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def read_checkpoint(value: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        data = value
    else:
        data = json.loads(Path(value).read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported native checkpoint schema")
    return data
