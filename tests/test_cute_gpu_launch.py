"""CPU tests for explicit launch evidence and frozen GPU/software targets."""

import hashlib

import pytest
from cute_rlm.config import REQUIRED_GPU_CHECKS, EvalConfig, check_gpu_receipt
from cute_rlm.launch import check_evidence_files, write_repl_wrapper


def config():
    return EvalConfig(
        task_manifest="tasks.jsonl",
        docs_manifest="docs.json",
        seed=7,
        task_count=1,
        verification={
            "atol": 0,
            "rtol": 0,
            "seed": 7,
            "mode": "preserve",
            "correctness_runs": 1,
            "warmup_runs": 0,
            "timed_runs": 1,
            "timeout_s": 30,
        },
    )


def test_gpu_target_is_frozen_to_rtx_pro_6000_cuda_133():
    cfg = config()
    receipt = {
        "config_sha256": cfg.sha256(),
        "manifest_sha256": "m",
        "observed_context_tokens": 262144,
        "hardware": {"gpu_model": cfg.gpu_model},
        "software": {"cuda_toolkit_version": "13.3"},
        "checks": {
            key: {"status": "passed", "evidence": {"path": "e", "sha256": "h"}}
            for key in REQUIRED_GPU_CHECKS
        },
    }
    check_gpu_receipt(receipt, cfg, "m")
    receipt["software"]["cuda_toolkit_version"] = "13.0"
    with pytest.raises(ValueError, match="CUDA 13.3"):
        check_gpu_receipt(receipt, cfg, "m")


def test_readiness_evidence_is_local_and_hash_pinned(tmp_path):
    (tmp_path / "e.json").write_text('{"observed": true}')
    evidence = {
        "path": "e.json",
        "sha256": hashlib.sha256((tmp_path / "e.json").read_bytes()).hexdigest(),
    }
    receipt = {"checks": {"server": {"evidence": evidence}}}
    check_evidence_files(receipt, tmp_path)
    (tmp_path / "e.json").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        check_evidence_files(receipt, tmp_path)
    receipt["checks"]["server"]["evidence"]["path"] = "../outside"
    with pytest.raises(ValueError, match="escapes"):
        check_evidence_files(receipt, tmp_path)


def test_repl_wrapper_is_private_and_cannot_overwrite(tmp_path):
    wrapper = tmp_path / "python"
    write_repl_wrapper(wrapper, tmp_path / "sandbox.json", "/path with spaces/python")
    assert wrapper.stat().st_mode & 0o777 == 0o700
    assert "--repl" in wrapper.read_text()
    assert "cute_rlm.sandbox" in wrapper.read_text()
    with pytest.raises(FileExistsError):
        write_repl_wrapper(wrapper, tmp_path / "sandbox.json", "python")
