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
