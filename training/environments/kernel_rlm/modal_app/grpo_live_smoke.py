"""Run the bounded Prime GRPO gate on two Modal A100 GPUs."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import modal

from .attention_lora import MODEL_ID, PRIME_REVISION, REVISION, models
from .kernel_smoke import APP_ROOT
from .native_lora_smoke import PRIME_SOURCE
from .rollout_smoke import traces

CONFIG = APP_ROOT / "configs" / "grpo_smoke.toml"
live_image = (
    modal.Image.from_registry(
        "vllm/vllm-openai:v0.26.0",
        add_python="3.12",
        setup_dockerfile_commands=["RUN ln -s /usr/bin/python3 /usr/local/bin/python"],
    )
    .entrypoint([])
    .pip_install(
        "prime-pydantic-config>=0.4.3",
        "verifiers==0.3.1",
        "renderers==0.1.11",
        "datasets>=4.0.0",
        "loguru>=0.7.3",
        "aiolimiter>=1.2.1",
        "beartype>=0.21.0",
        "jaxtyping>=0.3.2",
        "msgspec>=0.18",
        "openai>=1.106.1",
        "orjson>=3.11.0",
        "psutil>=7.0.0",
        "pyzmq>=27.1.0",
        "rich>=14.0.0",
        "setproctitle>=1.3.0",
        "tenacity>=8.2.0",
        "uvloop>=0.21.0",
        "https://github.com/pytorch/torchtitan/archive/23e4dfc.tar.gz",
        "https://github.com/samsja/dion/archive/d891eeb.tar.gz",
        "ring-flash-attn>=0.1.8",
        "flash-linear-attention",
    )
    .add_local_dir(
        APP_ROOT.parents[2] / "rlm",
        "/root/rlm",
        copy=True,
        ignore=["**/__pycache__/**"],
    )
    .add_local_dir(
        APP_ROOT.parents[2] / "training" / "src" / "rlm_train",
        "/root/rlm_train",
        copy=True,
        ignore=["**/__pycache__/**"],
    )
    .add_local_dir(PRIME_SOURCE / "src" / "prime_rl", "/root/prime_rl", copy=True)
    .add_local_dir(
        PRIME_SOURCE / "packages" / "prime-rl-configs" / "src" / "prime_rl",
        "/root/prime_rl",
        copy=True,
    )
    .add_local_dir(APP_ROOT / "kernel_rlm_v1", "/root/kernel_rlm_v1", copy=True)
    .add_local_dir(APP_ROOT / "kernel_rlm", "/root/kernel_rlm", copy=True)
    .add_local_dir(CONFIG.parent, "/root/configs", copy=True)
    .env(
        {
            "PYTHONPATH": "/root",
            "HF_HOME": "/models/hf",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "MAX_JOBS": "2",
            "RLM_TRAIN_EXEC_TIMEOUT_S": "30",
        }
    )
)
app = modal.App("kernel-rlm-grpo-live-smoke")


@app.function(
    image=live_image,
    volumes={"/models": models, "/traces": traces},
    gpu="A100-80GB:2",
    cpu=16,
    memory=131072,
    timeout=1800,
    startup_timeout=300,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def train(run_id: str) -> dict[str, object]:
    output_dir = Path("/traces/live-grpo") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "python",
        "-m",
        "prime_rl.entrypoints.rl",
        "@",
        "/root/configs/grpo_smoke.toml",
        "--run.name",
        run_id,
        "--output-dir",
        "/traces/outputs",
        "--clean",
        "--no-dashboard",
    ]
    log_path = output_dir / "prime.log"
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            cwd="/root",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        returncode = process.wait()
    report = {
        "status": "passed" if returncode == 0 else "failed",
        "run_id": run_id,
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "prime_revision": PRIME_REVISION,
        "gpu_topology": "1 inference A100 + 1 trainer A100",
        "config": str(CONFIG),
        "command": command,
        "returncode": returncode,
        "log_path": str(log_path),
        "output_path": f"/traces/outputs/{run_id}",
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    traces.commit()
    if returncode:
        raise RuntimeError(json.dumps(report, indent=2))
    return report


@app.local_entrypoint()
def main() -> None:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    print(json.dumps(train.remote(run_id), indent=2))
