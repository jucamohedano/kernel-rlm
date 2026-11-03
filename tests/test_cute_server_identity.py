from copy import deepcopy
from types import SimpleNamespace

import pytest
from cute_rlm.server_identity import check_server_identity, model_card_sha256


@pytest.mark.asyncio
async def test_endpoint_must_match_observed_model_card_and_weight_identity():
    card = {"id": "model", "root": "/pinned/model", "max_model_len": 262144}
    config = SimpleNamespace(
        model_id="model", model_revision="revision", server_context_tokens=262144
    )

    class Client:
        base_url = "http://127.0.0.1:8000/v1/"

        async def get(self, path):
            assert path == "models"
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"data": [card]})

    receipt = {
        "server_identity": {
            "served_model": "model",
            "model_revision": "revision",
            "base_url": "http://127.0.0.1:8000/v1",
            "model_path": "/pinned/model",
            "startup_evidence_sha256": "startup",
            "model_card_sha256": model_card_sha256(card),
        }
    }
    await check_server_identity(Client(), config, receipt)
    card["created"] = 123
    card["permission"] = [{"id": "request-generated", "created": 123}]
    await check_server_identity(Client(), config, receipt)
    card["max_model_len"] = 65536
    with pytest.raises(ValueError, match="context differs"):
        await check_server_identity(Client(), config, receipt)
    card["max_model_len"] = 262144
    card["root"] = "/other/weights"
    # Even a refreshed card hash cannot override the operator's pinned weight path.
    receipt["server_identity"]["model_card_sha256"] = model_card_sha256(card)
    with pytest.raises(ValueError, match="weight path differs"):
        await check_server_identity(Client(), config, receipt)


@pytest.mark.parametrize("field", ["id", "object", "owned_by", "root", "parent", "max_model_len"])
def test_model_card_fingerprint_rejects_serving_identity_changes(field):
    card = {
        "id": "model",
        "object": "model",
        "owned_by": "vllm",
        "root": "/pinned/model",
        "parent": None,
        "max_model_len": 262144,
        "created": 1,
        "permission": [{"id": "first", "created": 1}],
    }
    changed = deepcopy(card)
    changed[field] = "changed"
    assert model_card_sha256(changed) != model_card_sha256(card)


def test_model_card_fingerprint_ignores_only_non_identity_response_metadata():
    card = {
        "id": "model",
        "root": "/pinned/model",
        "max_model_len": 262144,
        "created": 1,
        "permission": [{"id": "first", "created": 1}],
    }
    changed = {**card, "created": 2, "permission": [{"id": "second", "created": 2}]}
    assert model_card_sha256(changed) == model_card_sha256(card)
