"""Run the bounded Prime GRPO gate on three Modal A100 GPUs."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import modal
from kernel_factory import event, launch_bundle, prepare_run, start_run

from .attention_lora import MODEL_ID, PRIME_REVISION, REVISION, models
from .compact_checkpoint import (
    check_compact_checkpoint_roundtrip,
    patch_compact_lora_checkpoints,
)
from .gpu_telemetry import wait_with_gpu_sampling
from .native_lora_smoke import PRIME_SOURCE
from .rollout_smoke import traces
from .verifiers_sampling import patch_verifiers_sampling_caps

_module_dir = Path(__file__).resolve().parent
APP_ROOT = _module_dir.parent if (_module_dir.parent / "configs").exists() else Path("/root")
REPO_ROOT = next(
    (parent for parent in APP_ROOT.parents if (parent / "rlm").exists()), Path("/root")
)
PROJECT_ROOT = REPO_ROOT if (REPO_ROOT / "notes").exists() else REPO_ROOT.parent
RLM_SOURCE = REPO_ROOT / "rlm" if (REPO_ROOT / "rlm").exists() else Path("/root/rlm")
RLM_TRAIN_SOURCE = (
    REPO_ROOT / "training" / "src" / "rlm_train"
    if (REPO_ROOT / "training" / "src" / "rlm_train").exists()
    else Path("/root/rlm_train")
)
CONFIG = APP_ROOT / "configs" / "grpo_level1.toml"
PILOT_CONFIG = APP_ROOT / "configs" / "grpo_pilot.toml"
CURRICULUM_LEVEL1 = PROJECT_ROOT / "notes" / "curriculum" / "lvl1_seed42.jsonl"


def patch_prime_laguna_config() -> None:
    clients_path = Path("/root/prime_rl/orchestrator/clients.py")
    clients = clients_path.read_text()
    if "from kernel_rlm.gpu_sharing import exclusive_weight_update" not in clients:
        for name in ("update_weights", "load_lora_adapter"):
            marker = f"async def {name}("
            if clients.count(marker) != 1:
                raise RuntimeError(f"Prime GPU coordination anchor missing: {name}")
            clients = clients.replace(marker, f"@exclusive_weight_update\n{marker}", 1)
        clients = clients.replace(
            "from __future__ import annotations\n",
            "from __future__ import annotations\n\n"
            "from kernel_rlm.gpu_sharing import exclusive_weight_update\n",
            1,
        )
        compile(clients, str(clients_path), "exec")
        clients_path.write_text(clients)
    model_source = Path("/root/prime_rl/trainer/model.py")
    source = model_source.read_text()
    marker = "    model_config.use_cache = False\n"
    compatibility = (
        '    if getattr(model_config, "model_type", "") == "laguna" '
        'and getattr(model_config, "gating", None) is True:\n'
        '        model_config.gating = "per-head"\n'
    )
    if compatibility not in source:
        if marker not in source:
            raise RuntimeError("Prime Laguna compatibility anchor is missing")
        source = source.replace(marker, compatibility + marker, 1)
        model_source.write_text(source)


live_image = (
    modal.Image.from_registry(
        "vllm/vllm-openai:v0.26.0",
        add_python="3.12",
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
        "nvidia-ml-py>=12.575.51",
        "openai>=1.106.1",
        "orjson>=3.11.0",
        "psutil>=7.0.0",
        "pyzmq>=27.1.0",
        "rich>=14.0.0",
        "setproctitle>=1.3.0",
        "tenacity>=8.2.0",
        "uvloop>=0.21.0",
        "https://github.com/PrimeIntellect-ai/router/releases/download/v0.2.0/vllm_router-0.2.0-cp38-abi3-manylinux_2_28_x86_64.whl",
        "wandb>=0.21.0,<0.27.1",
        "wandb-workspaces>=0.4.3",
        "https://github.com/PrimeIntellect-ai/prime-rl/releases/download/v0.6.0/torchao-0.17.0+git02105d46c-cp312-cp312-linux_x86_64.whl",
        "https://github.com/pytorch/torchtitan/archive/23e4dfc.tar.gz",
        "https://github.com/samsja/dion/archive/d891eeb.tar.gz",
        "ring-flash-attn>=0.1.8",
        "flash-linear-attention",
    )
    .run_commands(
        "python -m pip install --no-deps modelexpress==0.3.0 prime==0.6.19 prime-sandboxes==0.2.39"
    )
    .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("torchvision==0.26.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "flash-attn @ https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu128torch2.11-cp312-cp312-linux_x86_64.whl"
    )
    .pip_install(
        "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
    )
    .run_commands(
        "python -m pip install --force-reinstall --no-deps torchaudio==2.11.0 "
        "--index-url https://download.pytorch.org/whl/cu128"
    )
    .pip_install("nvidia-cutlass-dsl==4.5.2")
    .add_local_dir(
        RLM_SOURCE,
        "/root/rlm",
        copy=True,
        ignore=["**/__pycache__/**"],
    )
    .add_local_dir(
        RLM_TRAIN_SOURCE,
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
    .add_local_dir(PRIME_SOURCE, "/root/prime_source", copy=True)
    .run_commands("python -m pip install --no-deps -e /root/prime_source")
    .add_local_dir(APP_ROOT / "kernel_rlm_v1", "/root/kernel_rlm_v1", copy=True)
    .add_local_dir(APP_ROOT / "kernel_rlm", "/root/kernel_rlm", copy=True)
    .add_local_dir(APP_ROOT / "kernel_factory", "/root/kernel_factory", copy=True)
    .add_local_file(CURRICULUM_LEVEL1, "/root/kernel_rlm/tasks/lvl1_seed42.jsonl", copy=True)
    .add_local_dir(CONFIG.parent, "/root/configs", copy=True)
    .env(
        {
            "PYTHONPATH": "/root",
            "HF_HOME": "/models/hf",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "MAX_JOBS": "2",
            "TORCH_CUDA_ARCH_LIST": "8.0",
            "RLM_TRAIN_EXEC_TIMEOUT_S": "30",
        }
    )
    .run_function(patch_verifiers_sampling_caps)
    .run_function(patch_compact_lora_checkpoints)
)
app = modal.App("kernel-rlm-grpo-live-smoke")


@app.function(image=live_image, cpu=2, memory=8192, timeout=180, retries=0)
def checkpoint_preflight() -> dict:
    return check_compact_checkpoint_roundtrip()


def _train(
    run_id: str,
    pilot: bool,
    training_timeout: int,
    modal_timeout: int,
    resume_run: str = "",
    resume_step: int = 1,
    provenance: dict | None = None,
) -> dict[str, object]:
    os.environ["KERNEL_INFERENCE_ADMIN_URL"] = "http://127.0.0.1:8100"
    output_dir = Path("/traces/live-grpo") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        start_run(provenance, output_dir, run_id)
    except Exception:
        traces.commit()
        raise
    os.environ["KERNEL_FEEDBACK_LOG_DIR"] = str(output_dir / "feedback")
    patch_prime_laguna_config()
    config_path = "/root/configs/grpo_pilot.toml" if pilot else "/root/configs/grpo_level1.toml"
    command = [
        "python",
        "-u",
        "-m",
        "prime_rl.entrypoints.rl",
        "@",
        config_path,
        "--run.name",
        run_id,
        "--output-dir",
        "/traces/outputs",
        "--no-dashboard",
    ]
    if resume_run:
        command.extend(
            [
                "--resume.dir",
                f"/traces/outputs/{resume_run}/checkpoints/step_{resume_step}",
                "--max-steps",
                str(resume_step + 1),
            ]
        )
    else:
        command.append("--clean")
    log_path = output_dir / "prime.log"
    gpu_stats_path = output_dir / "gpu_stats.jsonl"
    pipeline = f"{shlex.join(command)} 2>&1 | tee {shlex.quote(str(log_path))}"
    process = subprocess.Popen(
        ["/bin/bash", "-o", "pipefail", "-c", pipeline],
        cwd="/root",
        start_new_session=True,
    )
    monitor = wait_with_gpu_sampling(process, gpu_stats_path, timeout_s=training_timeout)
    timed_out = bool(monitor["timed_out"])
    returncode = monitor["returncode"]
    if timed_out:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            returncode = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            returncode = process.wait(timeout=10)
    report = {
        "status": "budget_timeout" if timed_out else ("passed" if returncode == 0 else "failed"),
        "training_timeout_seconds": training_timeout,
        "modal_timeout_seconds": modal_timeout,
        "startup_timeout_seconds": 120,
        "total_wall_clock_cap_seconds": modal_timeout + 120,
        "run_id": run_id,
        "experiment_id": (provenance or {}).get("experiment_id"),
        "provenance_sha256": (provenance or {}).get("manifest_sha256"),
        "provenance_path": str(output_dir / "provenance.json"),
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "prime_revision": PRIME_REVISION,
        "gpu_topology": "1 inference A100 + 2 trainer A100s",
        "pilot": pilot,
        "resumed_from": resume_run or None,
        "resume_step": resume_step if resume_run else None,
        "checkpoint_format": "LoRA and optimizer DCP; frozen base parameters excluded",
        "config": str(PILOT_CONFIG if pilot else CONFIG),
        "command": command,
        "returncode": returncode,
        "log_path": str(log_path),
        "gpu_stats_path": str(gpu_stats_path),
        "output_path": f"/traces/outputs/{run_id}",
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    event(output_dir, report["status"], {"returncode": returncode})
    traces.commit()
    if returncode or timed_out:
        raise RuntimeError(json.dumps(report, indent=2))
    return report


@app.function(
    image=live_image,
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/models": models, "/traces": traces},
    gpu="A100-80GB:3",
    cpu=16,
    memory=131072,
    timeout=3450,
    startup_timeout=120,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def train(run_id: str, provenance: dict | None = None) -> dict[str, object]:
    return _train(
        run_id, pilot=False, training_timeout=3300, modal_timeout=3450, provenance=provenance
    )


@app.function(
    image=live_image,
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/models": models, "/traces": traces},
    gpu="A100-80GB:3",
    cpu=16,
    memory=131072,
    timeout=1380,
    startup_timeout=120,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def train_pilot(
    run_id: str, resume_run: str = "", resume_step: int = 1, provenance: dict | None = None
) -> dict[str, object]:
    return _train(
        run_id,
        pilot=True,
        training_timeout=1200,
        modal_timeout=1380,
        resume_run=resume_run,
        resume_step=resume_step,
        provenance=provenance,
    )


@app.function(
    image=live_image,
    volumes={"/traces": traces},
    cpu=8,
    memory=32768,
    timeout=900,
    startup_timeout=300,
    retries=0,
    max_containers=1,
)
def dry_run(
    run_id: str,
    pilot: bool = False,
    resume_run: str = "",
    resume_step: int = 1,
    provenance: dict | None = None,
) -> dict[str, object]:
    output_dir = Path("/traces/live-grpo") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        start_run(provenance, output_dir, run_id)
    except Exception:
        traces.commit()
        raise
    patch_prime_laguna_config()
    config_path = "/root/configs/grpo_pilot.toml" if pilot else "/root/configs/grpo_level1.toml"
    command = [
        "python",
        "-m",
        "prime_rl.entrypoints.rl",
        "@",
        config_path,
        "--run.name",
        run_id,
        "--output-dir",
        "/traces/outputs",
        "--no-dashboard",
        "--dry-run",
    ]
    if resume_run:
        command.extend(
            [
                "--resume.dir",
                f"/traces/outputs/{resume_run}/checkpoints/step_{resume_step}",
                "--max-steps",
                str(resume_step + 1),
            ]
        )
    else:
        command.append("--clean")
    result = subprocess.run(command, cwd="/root", text=True, capture_output=True, check=False)
    report = {
        "status": "passed" if result.returncode == 0 else "failed",
        "run_id": run_id,
        "experiment_id": (provenance or {}).get("experiment_id"),
        "provenance_sha256": (provenance or {}).get("manifest_sha256"),
        "pilot": pilot,
        "config": str(PILOT_CONFIG if pilot else CONFIG),
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    (output_dir / "dry-run-report.json").write_text(json.dumps(report, indent=2))
    event(output_dir, "dry_run_passed" if result.returncode == 0 else "dry_run_failed")
    traces.commit()
    if result.returncode:
        raise RuntimeError(json.dumps(report, indent=2))
    return report


@app.function(
    image=live_image,
    volumes={"/traces": traces},
    cpu=2,
    memory=8192,
    timeout=180,
    retries=0,
)
def checkpoint_readback(run_id: str, step: int = 1) -> dict[str, object]:
    import torch
    from torch.distributed.checkpoint import FileSystemReader, load

    path = Path("/traces/outputs") / run_id / f"checkpoints/step_{step}/trainer"
    metadata = FileSystemReader(str(path)).read_metadata()
    keys = list(metadata.state_dict_metadata)
    parameter = next(key for key in keys if key.startswith("app.model.") and "lora_B" in key)
    description = metadata.state_dict_metadata[parameter]
    tensor = torch.empty(description.size, dtype=description.properties.dtype)
    state = {
        "app": {
            "progress": {"step": 0, "total_tokens": 0, "total_samples": 0},
            "model": {parameter.removeprefix("app.model."): tensor},
        }
    }
    load(state, checkpoint_id=str(path))
    result = {
        "run_id": run_id,
        "status": "passed",
        "progress": state["app"]["progress"],
        "checked_adapter": parameter,
        "adapter_nonzero": int(torch.count_nonzero(tensor)),
        "adapter_max_abs": tensor.abs().max().item(),
        "optimizer_keys": sum(key.startswith("app.optimizers.") for key in keys),
        "scheduler_keys": sum(key.startswith("app.scheduler.") for key in keys),
        "checkpoint_bytes": sum(
            file.stat().st_size for file in path.parent.rglob("*") if file.is_file()
        ),
        "checkpoint_path": str(path.parent),
        "scope": "CPU read-back of progress and one adapter; no full trainer restart",
    }
    if state["app"]["progress"]["step"] != step or not result["adapter_nonzero"]:
        raise RuntimeError(f"Checkpoint does not contain a trained adapter: {result}")
    if not result["optimizer_keys"] or not result["scheduler_keys"]:
        raise RuntimeError(f"Checkpoint is missing optimizer or scheduler state: {result}")
    return result


@app.local_entrypoint()
def main(
    dry: bool = False,
    pilot: bool = False,
    checkpoint_run: str = "",
    compact_check: bool = False,
    resume_run: str = "",
    checkpoint_step: int = 1,
    experiment: str = "",
    allow_dirty: bool = False,
) -> None:
    if checkpoint_step < 1:
        raise ValueError("checkpoint_step must be positive")
    if resume_run and not pilot:
        raise ValueError("Use --pilot when resuming one additional optimizer step")
    if compact_check:
        print(json.dumps(checkpoint_preflight.remote(), indent=2))
        return
    if checkpoint_run:
        result = checkpoint_readback.remote(checkpoint_run, checkpoint_step)
        directory = Path("logs/live-grpo") / checkpoint_run
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "checkpoint-check.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    manifest = prepare_run(
        PROJECT_ROOT,
        APP_ROOT,
        experiment,
        run_id,
        pilot=pilot,
        resume_run=resume_run,
        resume_step=checkpoint_step,
        allow_dirty=allow_dirty,
        pins={"model_id": MODEL_ID, "model_revision": REVISION, "prime_revision": PRIME_REVISION},
        dependency_repositories={"prime": PRIME_SOURCE},
    )
    provenance = launch_bundle(manifest, APP_ROOT)
    event(APP_ROOT / "logs/live-grpo" / run_id, "dispatch_requested", {"dry_run": dry})
    result = (
        dry_run.remote(run_id, pilot, resume_run, checkpoint_step, provenance)
        if dry
        else train_pilot.remote(run_id, resume_run, checkpoint_step, provenance)
        if pilot
        else train.remote(run_id, provenance)
    )
    print(json.dumps(result, indent=2))
