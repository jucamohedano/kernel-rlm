"""Read the shared kernel curriculum manifest."""

import json
from pathlib import Path


def load_tasks(manifest: Path) -> list[dict]:
    from kernel_rlm.submission_contract import KERNEL_TASK_INSTRUCTION

    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows or len({row["_task_id"] for row in rows}) != len(rows):
        raise ValueError("The curriculum must contain tasks with unique IDs")
    if any(not isinstance(row["code"], str) or not row["code"].strip() for row in rows):
        raise ValueError("Every task needs reference source")
    return [
        {
            "name": str(row["_task_id"]),
            "context": row["code"],
            "prompt": KERNEL_TASK_INSTRUCTION,
            "speed_reward_weight": 0.1,
        }
        for row in rows
    ]
