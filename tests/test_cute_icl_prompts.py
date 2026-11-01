"""Frozen prompt packets are offline and have no runtime/helper context."""

import hashlib
import json

import pytest
from cute_rlm.icl.prompts import ASSET_DIRECTORY, CONDITIONS, render_prompt, source_ledger
from cute_rlm.icl.tasks import load_tasks, reference_source


def test_exactly_36_independent_plain_prompts():
    pairs = [(task, condition) for task in load_tasks() for condition in CONDITIONS]
    assert len(pairs) == 36
    assert CONDITIONS == ("task_only", "api", "api_example")
    for task, condition in pairs:
        prompt = render_prompt(task, condition)
        assert set(prompt) == {"system", "user", "sources"}
        assert isinstance(prompt["system"], str) and isinstance(prompt["user"], str)
        assert task["interface"]["forward_signature"] in prompt["user"]
        assert reference_source(task) not in prompt["user"]
        assert "compile_candidate" not in prompt["user"]
        assert "rlm_" not in prompt["user"]
        assert '"seed"' not in prompt["user"]
        assert len(prompt["sources"]) == {"task_only": 0, "api": 2, "api_example": 3}[condition]


def test_arms_add_identical_api_then_only_related_example():
    for task in load_tasks():
        baseline = render_prompt(task, "task_only")
        api = render_prompt(task, "api")
        example = render_prompt(task, "api_example")
        assert api["user"].startswith(baseline["user"])
        assert example["user"].startswith(api["user"])
        assert example["sources"][:2] == api["sources"]
        related = example["sources"][-1]
        assert related["kind"] == "example"
        assert "BSD-3-Clause" in related["content"]
        assert "Redistribution and use" in related["content"]
        assert "relatedness_limitation" in related["origin"]
        for source in api["sources"]:
            assert "pending" in source["sufficiency"]
            assert source["upstream_sources"]


def test_bundled_digests_and_license_scope():
    ledger = source_ledger()
    assert ledger["license_review"]["status"] == "recorded_not_legal_approval_for_training"
    for source in ledger["assets"]:
        data = (ASSET_DIRECTORY / source["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == source["sha256"]
        assert len(data) == source["bytes"]
    for source in ledger["upstream_api_sources"]:
        assert len(source["sha256"]) == 64
        assert source["revision"] in source["url"]
        assert "NvidiaProprietary" in source["license"]
    for task in load_tasks():
        for source in render_prompt(task, "api_example")["sources"]:
            assert hashlib.sha256(source["content"].encode()).hexdigest() == source["sha256"]
    json.dumps(ledger)


def test_unknown_condition_rejected():
    with pytest.raises(ValueError):
        render_prompt(load_tasks()[0], "api_plus_related_example")
