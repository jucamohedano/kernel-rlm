"""Bounded Modal integration test; no language model is loaded."""

from __future__ import annotations

import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import modal

from .kernel_smoke import APP_ROOT, image

DEPENDENCIES = {
    "verifiers",
    "datasets",
    "aiohttp",
    "anthropic",
    "google-genai",
    "openai",
    "portkey-ai",
    "python-dotenv",
    "requests",
    "rich",
    "pytest",
}


def build_image() -> modal.Image:
    repo_root = APP_ROOT.parents[2]
    lock = tomllib.loads((repo_root / "training" / "uv.lock").read_text())
    pins = [
        f"{package['name']}=={package['version']}"
        for package in lock["package"]
        if package["name"] in DEPENDENCIES
    ]
    return (
        image.pip_install(*pins)
        .env({"PYTHONPATH": "/root", "MAX_JOBS": "2", "RLM_TRAIN_EXEC_TIMEOUT_S": "30"})
        .add_local_dir(repo_root / "rlm", "/root/rlm", copy=True, ignore=["**/__pycache__/**"])
        .add_local_dir(
            repo_root / "training" / "src" / "rlm_train",
            "/root/rlm_train",
            copy=True,
            ignore=["**/__pycache__/**"],
        )
        .add_local_file(
            APP_ROOT / "kernel_rlm" / "tasks" / "vector_add.jsonl",
            "/opt/kernel_rlm/vector_add.jsonl",
            copy=True,
        )
    )


rollout_image = build_image() if modal.is_local() else None
app = modal.App("kernel-rlm-scripted-smoke")
traces = modal.Volume.from_name("kernel-rlm-traces", create_if_missing=True)


def run_saved(*, cuda: bool) -> dict[str, Any]:
    from .scripted_rollout import run

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12]
    stage = "cuda" if cuda else "preflight"
    path = Path("/traces") / run_id / f"{stage}.json"
    print(f"Rollout trace: kernel-rlm-traces:{path.relative_to('/traces')}")
    try:
        return run(cuda=cuda, trace_path=path)
    finally:
        traces.commit()


@app.function(
    image=rollout_image, volumes={"/traces": traces}, timeout=120, max_containers=1, retries=0
)
def preflight() -> dict[str, Any]:
    return run_saved(cuda=False)


@app.function(
    image=rollout_image,
    volumes={"/traces": traces},
    gpu="A100-80GB",
    timeout=360,
    cpu=2,
    memory=8192,
    max_containers=1,
    retries=0,
    scaledown_window=2,
)
def cuda_rollout() -> dict[str, Any]:
    return run_saved(cuda=True)


@app.local_entrypoint()
def main(gpu: bool = False, output_dir: str = "logs/kernel-rollout") -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stages = [("preflight", preflight)]
    if gpu:
        stages.append(("cuda", cuda_rollout))
    for name, function in stages:
        result = function.remote()
        path = directory / Path(result["trace_path"]).parent.name / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, default=str))
        print(
            json.dumps(
                {
                    "stage": name,
                    "reward": result["output"]["reward"],
                    "verifier_calls": result["verifier_calls"],
                    "trace": str(path),
                }
            )
        )
