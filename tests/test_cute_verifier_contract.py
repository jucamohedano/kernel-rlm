"""CPU scripted workers exercise verifier flow, not CUDA or CuTe correctness."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from cute_rlm import verifier
from cute_rlm.contracts import VerificationConfig
from cute_rlm.outputs import save_tree


def settings():
    return VerificationConfig(
        atol=0.001,
        rtol=0.001,
        seed=9,
        mode="preserve",
        correctness_runs=2,
        warmup_runs=1,
        timed_runs=3,
        timeout_s=20,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", [None, "outputs", "mutated-inputs", "post-state"])
async def test_separate_worker_results_compare_outputs_and_state(tmp_path, monkeypatch, mismatch):
    source = "class Model: pass"
    task = {
        "task_id": "t",
        "reference_code": source,
        "source": {"sha256": hashlib.sha256(source.encode()).hexdigest()},
    }
    roles = []

    async def fake_phase(command, request, config):
        roles.append(request["role"])
        out = Path(request["output_dir"])
        out.mkdir()
        if request["role"] == "reference":
            for name in ("initial_state", "init_args", "inputs-0", "inputs-1"):
                save_tree(out / (name + ".npz"), [])
        else:
            inputs = Path(request["input_dir"])
            assert not list(inputs.glob("outputs*"))
            assert not list(inputs.glob("post-state*"))
            assert request["source"] != source
        for trial in range(config.correctness_runs):
            for kind in ("outputs", "mutated-inputs", "post-state"):
                value = 7 if request["role"] == "candidate" and kind == mismatch else 1
                save_tree(out / f"{kind}-{trial}.npz", [np.array([value], dtype=np.int32)])
        return {
            "status": "executed",
            "source_sha256": request["source_sha256"],
            "training": True,
            "module_training": {"": True},
            "median_ms": 2 if request["role"] == "reference" else 1,
        }

    monkeypatch.setattr(verifier, "launch_phase", fake_phase)
    result = await verifier.verify_submission(
        task,
        "class ModelNew: pass",
        settings(),
        tmp_path / "run",
        worker_command=["explicit-isolated-worker"],
    )
    assert roles == ["reference", "candidate"]
    assert result["status"] == ("launch_unverified" if mismatch is None else "incorrect")
    assert result["speedup"] is None
    assert result["correct"] is (mismatch is None)


@pytest.mark.asyncio
async def test_worker_deadline_reaps_process(tmp_path):
    import sys

    config = VerificationConfig(
        atol=0,
        rtol=0,
        seed=0,
        mode="eval",
        correctness_runs=1,
        warmup_runs=0,
        timed_runs=1,
        timeout_s=0.05,
    )
    with pytest.raises(TimeoutError):
        await verifier.launch_phase(
            [sys.executable, "-c", "import time; time.sleep(10)"], {}, config
        )


def test_verification_requires_explicit_mode_and_finite_tolerance():
    with pytest.raises(TypeError):
        VerificationConfig(atol=0, rtol=0, seed=0)
    with pytest.raises(ValueError):
        VerificationConfig(**{**settings().payload(), "rtol": float("nan")})
