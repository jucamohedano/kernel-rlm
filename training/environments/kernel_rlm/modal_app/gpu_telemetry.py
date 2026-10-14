"""Low-overhead GPU sampling for bounded Modal training runs."""

from __future__ import annotations

import csv
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_QUERY = (
    "index,name,utilization.gpu,utilization.memory,memory.used,memory.total,"
    "temperature.gpu,power.draw"
)


def sample_gpu_stats() -> dict[str, Any]:
    sample: dict[str, Any] = {"timestamp_utc": datetime.now(UTC).isoformat(), "gpus": []}
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
            if len(row) != 8:
                continue
            sample["gpus"].append(
                {
                    "index": int(row[0]),
                    "name": row[1],
                    "gpu_utilization_percent": _number(row[2]),
                    "memory_utilization_percent": _number(row[3]),
                    "memory_used_mib": _number(row[4]),
                    "memory_total_mib": _number(row[5]),
                    "temperature_c": _number(row[6]),
                    "power_w": _number(row[7]),
                }
            )
        if not sample["gpus"]:
            sample["error"] = "nvidia-smi returned no parseable GPU rows"
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        sample["error"] = f"{type(error).__name__}: {error}"
    return sample


def _number(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def wait_with_gpu_sampling(
    process: subprocess.Popen,
    output_path: Path,
    *,
    timeout_s: float,
    interval_s: float = 5.0,
) -> dict[str, Any]:
    """Wait for a child process while saving timestamped per-GPU samples."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    with output_path.open("a", encoding="utf-8") as output:
        while True:
            if process.poll() is not None:
                return {"returncode": process.returncode, "timed_out": False}
            output.write(json.dumps(sample_gpu_stats(), sort_keys=True) + "\n")
            output.flush()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {"returncode": None, "timed_out": True}
            try:
                return {
                    "returncode": process.wait(timeout=min(interval_s, remaining)),
                    "timed_out": False,
                }
            except subprocess.TimeoutExpired:
                continue
