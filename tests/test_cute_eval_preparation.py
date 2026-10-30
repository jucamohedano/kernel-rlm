"""CPU-only launch gates for the next fixed CuTe evaluation."""

import sys
from pathlib import Path

import pytest

ENV = Path(__file__).parents[1] / "training/environments/cute_rlm"
sys.path.insert(0, str(ENV))


def config():
    from cute_rlm.config import EvalConfig

    return EvalConfig(
        task_manifest="tasks.jsonl",
        docs_manifest="docs.json",
        seed=42,
        task_count=4,
        verification={
            "atol": 0.001,
            "rtol": 0.001,
            "seed": 42,
            "mode": "preserve",
            "correctness_runs": 2,
            "warmup_runs": 1,
            "timed_runs": 3,
            "timeout_s": 120,
        },
    )


def test_next_eval_keeps_full_window_reasoning_and_visible_guards():
    from cute_rlm.config import EvalConfig, serving_arguments

    cfg = config()
    args = serving_arguments("/model", cfg)
    assert args[args.index("--max-model-len") + 1] == "262144"
    assert args[args.index("--generation-config") + 1] == "vllm"
    assert args[args.index("--max-num-seqs") + 1] == "2"
    assert cfg.compaction is False and cfg.logprobs is True
    assert cfg.request_timeout == 600 and cfg.episode_timeout == 1800
    assert cfg.broker_request_bytes == 8 * 1024**2
    assert cfg.max_prompt_chars is None
    with pytest.raises(ValueError, match="silently lower"):
        EvalConfig(**{**cfg.payload(), "server_context_tokens": 65536})


def test_gpu_receipt_must_bind_observed_checks_to_frozen_inputs():
    from cute_rlm.config import REQUIRED_GPU_CHECKS, check_gpu_receipt

    cfg = config()
    receipt = {
        "config_sha256": cfg.sha256(),
        "manifest_sha256": "manifest",
        "observed_context_tokens": 262144,
        "server_identity": {"tensor_parallel_size": cfg.tensor_parallel_size},
        "hardware": {
            "gpu_model": cfg.gpu_model,
            "gpu_compute_capability": cfg.gpu_compute_capability,
        },
        "software": {"cuda_toolkit_version": cfg.cuda_toolkit_version},
        "checks": {
            key: {"status": "passed", "evidence": "receipt:" + key} for key in REQUIRED_GPU_CHECKS
        },
    }
    check_gpu_receipt(receipt, cfg, "manifest")
    receipt["checks"]["two_concurrent_long_requests"]["status"] = "pending"
    with pytest.raises(ValueError, match="two_concurrent_long_requests"):
        check_gpu_receipt(receipt, cfg, "manifest")
    with pytest.raises(ValueError, match="frozen inputs"):
        check_gpu_receipt({**receipt, "manifest_sha256": "other"}, cfg, "manifest")


def test_prepared_manifest_captures_resolved_config_and_doc_bytes(tmp_path):
    import hashlib
    import json
    import shutil

    from cute_rlm.config import DATASET_ID, DATASET_REVISION, EvalConfig
    from cute_rlm.evaluate import prepare
    from kernel_factory import start_evaluation

    project = tmp_path / "project"
    project.mkdir()
    source_root = Path(__file__).parents[1]
    for relative in ("training/src/rlm_train", "training/environments/cute_rlm/cute_rlm"):
        shutil.copytree(
            source_root / relative, project / relative, ignore=shutil.ignore_patterns("__pycache__")
        )
    for relative in (
        "training/environments/kernel_rlm/kernel_factory/__init__.py",
        "pyproject.toml",
        "uv.lock",
        "training/environments/cute_rlm/pyproject.toml",
    ):
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / relative, destination)
    code = "REFERENCE_SPEC_NOT_EXECUTED = True"
    task = {
        "task_id": "t",
        "reference_code": code,
        "source": {
            "repo": DATASET_ID,
            "revision": DATASET_REVISION,
            "file": "pool.parquet",
            "row": 0,
            "sha256": hashlib.sha256(code.encode()).hexdigest(),
        },
    }
    (project / "tasks.jsonl").write_text(json.dumps(task) + "\n")
    (project / "cute.md").write_text("Pinned CuTe documentation")
    (project / "docs.json").write_text(
        json.dumps(
            {
                "repo": "NVIDIA/cutlass",
                "revision": "a" * 40,
                "files": [
                    {
                        "path": "cute.md",
                        "sha256": hashlib.sha256((project / "cute.md").read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )
    cfg = EvalConfig(**{**config().payload(), "task_count": 1})
    # The factory's real file capture is exercised; only unrelated Git identity is scripted.
    from unittest.mock import patch

    with (
        patch("kernel_factory.git_state", return_value={"commit": "cpu-fixture"}),
        patch("kernel_factory.subprocess.check_output", return_value=b""),
    ):
        readiness = prepare(project, cfg, "cpu-fixture", tmp_path / "runs")
    run = tmp_path / "runs/cpu-fixture"
    manifest = json.loads((run / "provenance.json").read_text())
    captured = {item["path"] for item in manifest["inputs"]}
    assert {
        "cute.md",
        "docs.json",
        ".eval-inputs/docs-corpus.json",
        ".eval-inputs/resolved-config.json",
        ".eval-inputs/system-prompt.txt",
        ".eval-inputs/user-prompts.json",
    } <= captured
    from cute_rlm.prompts import CUTE_SYSTEM_PROMPT, first_user_message

    assert (run / "inputs/.eval-inputs/system-prompt.txt").read_text() == CUTE_SYSTEM_PROMPT
    assert json.loads((run / "inputs/.eval-inputs/user-prompts.json").read_text()) == {
        "t": first_user_message(task, cfg)
    }
    assert readiness["status"] == "CPU_prepared_GPU_pending"
    start_evaluation(manifest, run.parent)
    with pytest.raises(ValueError, match="already started"):
        start_evaluation(manifest, run.parent)
    (run / "inputs/cute.md").write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        start_evaluation(manifest, run.parent)


@pytest.mark.parametrize("capability", ["", "0.0", "-8.9", "8", "8.90", "sm_89", 8.9, True])
def test_gpu_compute_capability_requires_explicit_major_minor_string(capability):
    from cute_rlm.config import EvalConfig

    with pytest.raises(ValueError, match="gpu_compute_capability"):
        EvalConfig(**{**config().payload(), "gpu_compute_capability": capability})


def test_different_gpu_requires_an_explicit_compute_capability():
    from cute_rlm.config import EvalConfig, serving_arguments
    from cute_rlm.prompts import CUTE_SYSTEM_PROMPT, first_user_message

    default = config()
    assert default.gpu_compute_capability == "12.0" and default.gpu_arch == "sm_120"
    assert default.tensor_parallel_size == 1
    with pytest.raises(ValueError, match="explicitly"):
        EvalConfig(
            **{**default.payload(), "gpu_model": "NVIDIA L40S", "gpu_compute_capability": None}
        )
    cfg = EvalConfig(
        **{
            **default.payload(),
            "gpu_model": "NVIDIA L40S",
            "gpu_compute_capability": "8.9",
            "tensor_parallel_size": 2,
            "max_concurrent_requests": 3,
        }
    )
    assert cfg.gpu_arch == "sm_89"
    rendered = CUTE_SYSTEM_PROMPT + first_user_message({"task_id": "t"}, cfg)
    assert "GPU: NVIDIA L40S, compute capability 8.9 (sm_89)" in rendered
    assert "SM120" not in rendered and "sm_120" not in rendered
    args = serving_arguments("/model", cfg)
    assert args[args.index("--tensor-parallel-size") + 1] == "2"
    assert args[args.index("--max-num-seqs") + 1] == "3"
    assert args[args.index("--max-model-len") + 1] == "262144"
    assert cfg.model_revision == default.model_revision
    assert cfg.verification == default.verification
    assert cfg.max_turns == default.max_turns == 32
    assert cfg.max_generated_tokens == default.max_generated_tokens == 1000000


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_tensor_parallel_size_must_be_a_positive_integer(size):
    from cute_rlm.config import EvalConfig

    with pytest.raises(ValueError, match="tensor_parallel_size"):
        EvalConfig(**{**config().payload(), "tensor_parallel_size": size})
