"""Import completed prime-compatible JSONL metrics into a local Trackio snapshot."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def import_metrics(source: Path, run_name: str, config: dict[str, Any]) -> dict[str, Any]:
    import trackio

    producers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in source.read_text().splitlines():
        row = json.loads(line)
        step = row.get("step")
        if step is not None and (not isinstance(step, int) or isinstance(step, bool) or step < 0):
            raise ValueError("Metric step must be a nonnegative integer or null")
        producers[str(row.get("producer", "trainer"))].append(row)
    if not producers:
        raise ValueError("No metric rows to import")
    counts = {}
    for producer, rows in producers.items():
        name = f"{run_name}--{producer}"
        trackio.init(
            project="laguna-kernel-rlm",
            name=name,
            group=run_name,
            config={**config, "producer": producer, "source": str(source), "snapshot": True},
            auto_log_gpu=False,
            auto_log_cpu=False,
            embed=False,
        )
        try:
            for sequence, row in enumerate(rows):
                metrics = {
                    key: value
                    for key, value in row.items()
                    if key not in ("step", "time", "producer")
                    and isinstance(value, int | float)
                    and math.isfinite(value)
                }
                if row.get("step") is not None:
                    metrics["source/training_step"] = row["step"]
                if isinstance(row.get("time"), int | float) and math.isfinite(row["time"]):
                    metrics["source/time"] = row["time"]
                trackio.log(metrics, step=sequence)
        finally:
            trackio.finish()
        counts[name] = len(rows)
    return {"source": str(source), "rows": sum(counts.values()), "runs": counts}
