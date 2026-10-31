import pytest
from cute_rlm.preflight import kv_cache_evidence, probe_two_long_requests


def test_kv_capacity_comes_from_observed_log_not_requested_context():
    value = kv_cache_evidence("INFO GPU KV cache size: 123,456 tokens")
    assert value["capacity_tokens"] == 123456
    with pytest.raises(ValueError, match="observed"):
        kv_cache_evidence("max_model_len=262144")


@pytest.mark.asyncio
async def test_long_probe_rejects_small_context_before_any_inference(tmp_path):
    from types import SimpleNamespace

    class Client:
        base_url = "http://localhost/v1/"

        async def post(self, path, *, json):
            assert path == "http://localhost/tokenize"
            return SimpleNamespace(
                raise_for_status=lambda: None, json=lambda: {"count": 10, "max_model_len": 262144}
            )

    with pytest.raises(ValueError, match="near the full"):
        await probe_two_long_requests(
            Client(),
            SimpleNamespace(server_context_tokens=262144, model_id="model"),
            [[{"role": "user", "content": "short"}]] * 2,
            tmp_path / "probe",
        )
    assert not (tmp_path / "probe").exists()


def test_only_actual_inference_oom_selects_serial_preflight_fallback():
    from cute_rlm.preflight import inference_concurrency_after_preflight

    assert inference_concurrency_after_preflight({"status": "passed"})["effective_concurrency"] == 2
    serial = inference_concurrency_after_preflight(
        {"status": "failed", "failure_stage": "model_inference", "error_type": "cuda_out_of_memory"}
    )
    assert serial["effective_concurrency"] == 1 and serial["requires_serial_recheck"]
    assert serial["automatic_scored_retry"] is False
    with pytest.raises(ValueError, match="capacity"):
        inference_concurrency_after_preflight(
            {
                "status": "failed",
                "failure_stage": "compile_candidate",
                "error_type": "cuda_out_of_memory",
            }
        )
    with pytest.raises(ValueError, match="capacity"):
        inference_concurrency_after_preflight(
            {"status": "failed", "failure_stage": "model_inference", "error_type": "timeout"}
        )


def test_serial_fallback_requires_its_own_frozen_capacity_evidence():
    from cute_rlm.config import EvalConfig, check_gpu_receipt, required_gpu_checks

    from tests.test_cute_eval_preparation import config

    cfg = EvalConfig(**{**config().payload(), "max_concurrent_requests": 1})
    assert "single_long_request" in required_gpu_checks(cfg)
    assert "two_concurrent_long_requests" not in required_gpu_checks(cfg)
    receipt = {
        "config_sha256": cfg.sha256(),
        "manifest_sha256": "manifest",
        "observed_context_tokens": cfg.server_context_tokens,
        "server_identity": {"tensor_parallel_size": cfg.tensor_parallel_size},
        "hardware": {
            "gpu_model": cfg.gpu_model,
            "gpu_compute_capability": cfg.gpu_compute_capability,
        },
        "software": {"cuda_toolkit_version": cfg.cuda_toolkit_version},
        "checks": {
            key: {"status": "passed", "evidence": "CPU_ONLY"} for key in required_gpu_checks(cfg)
        },
    }
    check_gpu_receipt(receipt, cfg, "manifest")
    del receipt["checks"]["single_long_request"]
    with pytest.raises(ValueError, match="single_long_request"):
        check_gpu_receipt(receipt, cfg, "manifest")
