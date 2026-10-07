"""Resolve the first live GRPO config inside the Modal runtime, without GPUs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

from .kernel_smoke import APP_ROOT
from .native_lora_smoke import PRIME_SOURCE
from .rollout_smoke import traces

CONFIG = APP_ROOT / "configs" / "grpo_smoke.toml"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "prime-pydantic-config>=0.4.3",
        "verifiers==0.3.1",
        "renderers==0.1.11",
        "loguru>=0.7.3",
        "datasets>=4.0.0",
        "transformers==5.6.2",
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
    .env({"PYTHONPATH": "/root", "HF_HUB_OFFLINE": "1"})
)
app = modal.App("kernel-rlm-grpo-dry-run")


@app.function(
    image=image,
    volumes={"/traces": traces},
    cpu=2,
    memory=8192,
    timeout=240,
    retries=0,
    max_containers=1,
)
def resolve() -> dict[str, object]:
    output_dir = Path("/traces/grpo-dry-run")
    command = [
        "python",
        "-m",
        "prime_rl.entrypoints.rl",
        "@",
        "/root/configs/grpo_smoke.toml",
        "--dry-run",
        "--output-dir",
        str(output_dir),
        "--run.name",
        "kernel-grpo-smoke",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    report = {
        "status": "passed" if result.returncode == 0 else "failed",
        "gpu_allocated": False,
        "command": command,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }
    report_path = output_dir / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    traces.commit()
    if result.returncode:
        raise RuntimeError(json.dumps(report, indent=2))
    return report


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(resolve.remote(), indent=2))
