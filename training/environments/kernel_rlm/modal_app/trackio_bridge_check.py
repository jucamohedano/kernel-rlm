"""CPU tests for preserving trainer/orchestrator JSONL metrics in Trackio."""

import json
from pathlib import Path

import modal

app = modal.App("kernel-rlm-trackio-bridge-check")


@app.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install("trackio==0.40.0"),
    cpu=1,
    memory=2048,
    timeout=180,
    retries=0,
    max_containers=1,
)
def verify() -> dict:
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        os.environ["TRACKIO_DIR"] = temporary
        from trackio.sqlite_storage import SQLiteStorage

        from .trackio_metrics import import_metrics

        source = Path(temporary) / "metrics.jsonl"
        rows = [
            {"producer": "trainer", "step": 1, "time": 10, "loss/mean": 2.5},
            {"producer": "trainer", "step": 1, "time": 11, "optim/lr": 0.0001},
            {"producer": "orchestrator", "step": 1, "time": 12, "reward/mean": 0.5},
            {"producer": "orchestrator", "step": None, "time": 13, "throughput": 10},
        ]
        source.write_text("\n".join(json.dumps(row) for row in rows))
        result = import_metrics(source, "fixture-only", {"training": False})
        assert result["rows"] == 4 and len(result["runs"]) == 2
        trainer = SQLiteStorage.get_all_metrics_for_run(
            "laguna-kernel-rlm", "fixture-only--trainer"
        )
        assert "loss/mean" in trainer and "optim/lr" in trainer
        assert "reward/mean" not in trainer
        values = SQLiteStorage.get_metric_values(
            "laguna-kernel-rlm", "fixture-only--trainer", "loss/mean"
        )
        assert values and values[0]["value"] == 2.5
        source.write_text('{"step": -1, "loss/mean": 0}')
        try:
            import_metrics(source, "invalid", {})
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid step was accepted")
        return {"status": "passed", "fixture_rows": 4, "gpu_allocated": False}


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(verify.remote(), indent=2))
