import hashlib
import json
from types import SimpleNamespace

import pytest
from cute_rlm.server_identity import check_server_identity


@pytest.mark.asyncio
async def test_endpoint_must_match_observed_model_card_and_weight_identity():
    card = {"id": "model", "max_model_len": 262144}
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
            "model_card_sha256": hashlib.sha256(
                json.dumps(card, sort_keys=True).encode()
            ).hexdigest(),
        }
    }
    await check_server_identity(Client(), config, receipt)
    card["max_model_len"] = 65536
    with pytest.raises(ValueError, match="context differs"):
        await check_server_identity(Client(), config, receipt)
