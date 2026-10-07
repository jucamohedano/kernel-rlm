"""Export a completed metrics file from Modal storage without occupying a GPU."""

import json
import os
from pathlib import Path

import modal

from .trackio_metrics import import_metrics

reports = modal.Volume.from_name("kernel-rlm-traces")
app = modal.App("kernel-rlm-trackio-export")


@app.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install("trackio==0.40.0"),
    volumes={"/reports": reports},
    cpu=1,
    memory=2048,
    timeout=180,
    max_containers=1,
    retries=0,
)
def export(relative_path: str, run_name: str, scope: str, rollout_trace: bool = False) -> dict:
    root = Path("/reports").resolve()
    source = (root / relative_path).resolve()
    if not source.is_relative_to(root):
        raise ValueError("Metrics must be inside the reports volume")
    reports.reload()
    if rollout_trace:
        trace = json.loads(source.read_text())
        if trace["training_updates"] != 0:
            raise ValueError("Rollout export requires a frozen-model evaluation trace")
        output = trace["output"]
        verification = trace.get("verification") or {}
        metrics = {
            "producer": "evaluation",
            "step": 0,
            "eval/reward": output["reward"],
            "eval/correct": int(bool(verification.get("correct"))),
            "eval/compiled": int(bool(verification.get("compiled"))),
            "eval/repl_calls": output["rlm_repl_calls"],
            "eval/model_calls": len(trace["model_responses"]),
            "eval/verifier_calls": trace["verifier_calls"],
            "eval/output_tokens": sum(row["completion_tokens"] for row in trace["model_responses"]),
            "time/episode_seconds": trace["elapsed_seconds"],
        }
        source = source.with_suffix(".metrics.jsonl")
        source.write_text(json.dumps(metrics) + "\n")
        scope = "Frozen-model evaluation; zero optimizer updates; not GRPO"
    os.environ["TRACKIO_DIR"] = "/reports/trackio-local"
    result = import_metrics(source, run_name, {"scope": scope})
    reports.commit()
    return result


@app.local_entrypoint()
def main(
    relative_path: str,
    run_name: str,
    scope: str = "prime-rl metrics snapshot",
    rollout_trace: bool = False,
) -> None:
    print(export.remote(relative_path, run_name, scope, rollout_trace))
