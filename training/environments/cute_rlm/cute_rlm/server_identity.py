"""Bind the attached serving endpoint to the GPU preflight observations."""

from __future__ import annotations

import hashlib
import json


def model_card_sha256(card):
    """Hash serving identity, not vLLM's per-response timestamps/permission IDs."""
    identity = {
        key: card.get(key)
        for key in ("id", "object", "owned_by", "root", "parent", "max_model_len")
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


async def check_server_identity(client, config, receipt):
    models = await client.get("models")
    models.raise_for_status()
    data = models.json().get("data", [])
    card = next((item for item in data if item.get("id") == config.model_id), None)
    if card is None:
        raise ValueError("Attached endpoint does not serve the frozen model ID")
    if card.get("max_model_len") != config.server_context_tokens:
        raise ValueError("Attached server context differs from frozen context")
    recorded = receipt.get("server_identity")
    if not isinstance(recorded, dict):
        raise ValueError("GPU receipt lacks observed server identity")
    if (
        recorded.get("served_model") != config.model_id
        or recorded.get("model_revision") != config.model_revision
    ):
        raise ValueError("GPU receipt model identity differs from frozen model")
    if recorded.get("base_url", "").rstrip("/") != str(client.base_url).rstrip("/"):
        raise ValueError("GPU receipt belongs to a different serving endpoint")
    # Model revisions are not part of the standard /models schema. The operator's
    # protected startup evidence must pin the actual weight path/revision/argv.
    if not recorded.get("model_path") or not recorded.get("startup_evidence_sha256"):
        raise ValueError("GPU receipt needs pinned model path and startup command evidence")
    if card.get("root") != recorded["model_path"]:
        raise ValueError("Attached serving weight path differs from frozen identity")
    if recorded.get("model_card_sha256") != model_card_sha256(card):
        raise ValueError("Attached serving model metadata changed after preflight")
    return card
