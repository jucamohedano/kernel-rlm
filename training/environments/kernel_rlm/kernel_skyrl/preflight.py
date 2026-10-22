"""CPU-only preparation. This module cannot rent a VM or launch a training worker."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from kernel_rollout.tasks import load_tasks

SKYRL_REVISION = "8adc1e53465e2766ac8e8f5ec652e633d17e2279"
MODEL_REVISION = "a397bde04501591d0ac3d74a5d9d0a4ec298ffb8"
DION_REVISION = "d891eebbd950a6ff11d9b5f806282e33df83df9d"
SOURCE_FILES = (
    "pyproject.toml",
    "uv.lock",
    "skyrl/train/generators/base.py",
    "skyrl/train/trainer.py",
    "skyrl/train/utils/trainer_utils.py",
    "skyrl/train/entrypoints/main_base.py",
    "skyrl/train/config/config.py",
    "skyrl/backends/skyrl_train/distributed/fsdp_strategy.py",
    "skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py",
    "skyrl/backends/skyrl_train/workers/model_wrapper.py",
    "skyrl/backends/skyrl_train/inference_servers/remote_inference_client.py",
)
PENDING_GATES = (
    "Connect the validated native Laguna BF16/FP32 helpers to the SkyRL policy worker",
    "Replace SkyRL's AdamW construction with the validated Dion Moonlight factory",
    "Add compact adapter/optimizer checkpoint save and distributed resume",
    "Connect KernelGenerator to the SkyRL entrypoint and factory launch capture",
    "Validate LoRA reload, exact token protocol and phase offload on two L40S GPUs",
    "Validate verifier placement, peak memory, throughput and bounded shutdown on the VM",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(directory: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(directory), *arguments], text=True).strip()


def prepare(source: Path, manifest: Path) -> dict:
    if git_value(source, "rev-parse", "HEAD") != SKYRL_REVISION:
        raise ValueError("SkyRL checkout does not match pinned skyrl-v0.4.0")
    if git_value(source, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("SkyRL tracked source is modified")
    task_count = len(load_tasks(manifest))
    environment_root = Path(__file__).resolve().parents[1]
    rlm_root = environment_root.parents[2]
    project_root = rlm_root.parent
    captured = {}
    for directory in (
        environment_root / "kernel_skyrl",
        environment_root / "kernel_rollout",
        environment_root / "kernel_training",
        environment_root / "kernel_rlm",
        environment_root / "kernel_rlm_v1",
        rlm_root / "training/src/rlm_train",
        rlm_root / "rlm",
    ):
        for path in sorted(directory.rglob("*.py")):
            captured[str(path.relative_to(rlm_root))] = digest(path)
    return {
        "kind": "backend-preparation",
        "backend": "skyrl",
        "provider": "cloudrift",
        "training_launch_enabled": False,
        "gpu_jobs_launched": 0,
        "skyrl": {
            "tag": "skyrl-v0.4.0",
            "revision": SKYRL_REVISION,
            "source_hashes": {name: digest(source / name) for name in SOURCE_FILES},
        },
        "model": {"id": "poolside/Laguna-XS.2", "revision": MODEL_REVISION},
        "optimizer": {
            "name": "Dion Muon",
            "revision": DION_REVISION,
            "adjust_lr": "rms_norm",
            "lr": 1e-5,
            "weight_decay": 0.01,
            "implemented": False,
            "component_factory": "kernel_training.laguna_policy.build_muon",
            "scope": "Component validated; implemented=false refers to native SkyRL worker integration",
        },
        "dataset": {"sha256": digest(manifest), "examples": task_count},
        "code": {
            "rlm_commit": git_value(rlm_root, "rev-parse", "HEAD"),
            "project_commit": git_value(project_root, "rev-parse", "HEAD"),
            "rlm_dirty": bool(git_value(rlm_root, "status", "--porcelain")),
            "project_dirty": bool(git_value(project_root, "status", "--porcelain")),
            "source_hashes": captured,
        },
        "proposed_placement": {
            "gpu_count": 2,
            "gpu_type": "L40S",
            "colocate_all": True,
            "trainer_fsdp_size": 2,
            "inference_tp": 2,
            "inference_engines": 1,
            "frozen_base_dtype": "bfloat16",
            "adapter_dtype": "float32",
            "per_forward_fsdp_cpu_offload": False,
            "phase_offload": True,
            "ideal_base_weight_gib_per_gpu": 31.1,
            "fit_validated": False,
        },
        "rollout_contract": {
            "root_reply_tokens": 4096,
            "recovery_reply_tokens": 512,
            "context_tokens": 12288,
            "turns": 8,
            "submit_by_turn": 6,
            "children": 2,
            "child_reply_tokens": 4096,
            "group_size": 4,
            "prompt_batch_size": 1,
            "max_sample_batches": 3,
            "step_wise_trajectories": True,
            "merge_stepwise_output": False,
            "native_dynamic_sampling": None,
            "native_zero_variance_filter": False,
            "filtering_owner": "KernelGenerator: whole groups, terminal rewards including speed",
            "loss": "root response tokens including thoughts; never child responses",
            "lora": {"rank": 8, "alpha": 16, "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        },
        "pending_gates": list(PENDING_GATES),
        "checkpoint_policy": "Do not import, alter or delete existing Modal checkpoints",
        "evidence_scope": "Source audit and input hashes only; no training or hardware validation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skyrl-source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = prepare(arguments.skyrl_source, arguments.dataset)
    with arguments.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "report": str(arguments.output),
                "examples": report["dataset"]["examples"],
                "training_launch_enabled": False,
                "pending_gates": report["pending_gates"],
            },
            indent=2,
        )
    )
