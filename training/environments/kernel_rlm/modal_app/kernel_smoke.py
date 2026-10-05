"""Run the known-good CUDA kernel through the Modal verifier."""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = "/opt/kernel_rlm/reference.py"
SUBMISSION_PATH = "/opt/kernel_rlm/submission.py"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("ninja-build")
    .pip_install(
        "torch==2.5.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .env({"TORCH_CUDA_ARCH_LIST": "8.0"})
    .add_local_python_source("kernel_rlm", copy=True)
    .add_local_file(
        APP_ROOT / "kernel_rlm" / "tasks" / "vector_add.py",
        REFERENCE_PATH,
        copy=True,
    )
    .add_local_file(
        APP_ROOT / "examples" / "vector_add_submission.py",
        SUBMISSION_PATH,
        copy=True,
    )
)

app = modal.App("kernel-rlm-cuda-smoke")


@app.function(image=image, gpu="A100-80GB", timeout=600)
def verify_known_good() -> dict[str, object]:
    from kernel_rlm.verifier import verify_cuda

    result = verify_cuda(
        Path(REFERENCE_PATH).read_text(encoding="utf-8"),
        Path(SUBMISSION_PATH).read_text(encoding="utf-8"),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["compiled"] or not result["correct"] or result["speedup"] is None:
        raise RuntimeError(f"CUDA smoke test failed: {result}")
    return result


@app.local_entrypoint()
def main() -> None:
    result = verify_known_good.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
