"""CPU simulation of verifier ordering; never evidence of CUDA correctness."""

import json

import modal

from .kernel_smoke import image

app = modal.App("kernel-rlm-verifier-contract")


@app.function(image=image, cpu=1, memory=2048, timeout=60, retries=0)
def check() -> dict:
    import contextlib
    import inspect
    import io
    import sys
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import patch

    import torch
    import torch.utils.cpp_extension
    from kernel_rlm.verifier import _RUNNER

    signature = inspect.signature(torch.utils.cpp_extension.load_inline)
    assert "keep_debug_handles" not in signature.parameters
    reference = Path("/opt/kernel_rlm/reference.py").read_text()
    prefix = (
        "from torch.utils.cpp_extension import load_inline\nCPP_SOURCE = ''\nCUDA_SOURCE = ''\n"
    )
    variants = {
        "eager": prefix
        + "extension = load_inline()\ndef cuda_forward(*inputs):\n    return extension.add(*inputs)\n",
        "lazy": prefix + "def cuda_forward(*inputs):\n    return load_inline().add(*inputs)\n",
        "bypass": prefix + "def cuda_forward(left, right):\n    return left + right\n",
    }
    timer = SimpleNamespace(
        record=lambda: None, synchronize=lambda: None, elapsed_time=lambda other: 1
    )
    results = {}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "reference.py").write_text(reference)
        for name, source in variants.items():
            (root / "submission.py").write_text(source)
            output = io.StringIO()
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "runner",
                        str(root / "reference.py"),
                        str(root / "submission.py"),
                        "3",
                        "0.001",
                        "0.0001",
                    ],
                ),
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch, "device", return_value="cpu"),
                patch.object(torch.cuda, "synchronize"),
                patch.object(torch.cuda, "Event", return_value=timer),
                patch.object(
                    torch.utils.cpp_extension,
                    "load_inline",
                    return_value=SimpleNamespace(add=lambda left, right: left + right),
                ),
                contextlib.redirect_stdout(output),
            ):
                exec(compile(_RUNNER, "verifier-contract", "exec"), {})
            results[name] = json.loads(output.getvalue().strip().splitlines()[-1])
        assert results["eager"]["correct"] and results["lazy"]["correct"]
        assert results["lazy"]["compiled"] and results["lazy"]["extension_used"]
        assert not results["bypass"]["correct"] and not results["bypass"]["extension_used"]
    return {
        "status": "passed",
        "scope": "CPU-mocked CUDA lifecycle only",
        "gpu_allocated": False,
        "eager_and_lazy_accepted": True,
        "pure_torch_bypass_rejected": True,
        "generated_keep_debug_handles_argument_supported": False,
    }


@app.local_entrypoint()
def main() -> None:
    from pathlib import Path

    result = check.remote()
    directory = Path("logs/verifier-contract")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "latest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
