"""Bind manual recovery to the original captured prompt, task and budget."""

import copy
import json
from dataclasses import asdict

import pytest
from cute_rlm.config import DATASET_ID, DATASET_REVISION
from cute_rlm.dataset import task_context
from cute_rlm.docs import docs_bootstrap
from cute_rlm.resources import build_resources, resources_bootstrap
from cute_rlm.resume import check_checkpoint_binding
from rlm_train.native import NativeLimits
from rlm_train.token_budget import ContextBudgetPolicy

from tests.test_cute_eval_preparation import config


def test_resume_requires_original_frozen_task_prompts_and_limits(tmp_path):
    cfg = config()
    task = {
        "task_id": "t",
        "reference_code": "x = 1",
        "source": {
            "repo": DATASET_ID,
            "revision": DATASET_REVISION,
            "sha256": __import__("hashlib").sha256(b"x = 1").hexdigest(),
        },
    }
    root = tmp_path / "inputs/.eval-inputs"
    root.mkdir(parents=True)
    docs = {"d": {"title": "doc", "text": "CuTe", "sha256": "doc-hash"}}
    resources = build_resources(cfg)
    (root / "docs-corpus.json").write_text(json.dumps(docs))
    (root / "resources.json").write_text(json.dumps(resources))
    (root / "system-prompt.txt").write_text("frozen system")
    (root / "user-prompts.json").write_text(json.dumps({"t": "frozen user"}))
    (tmp_path / "inputs" / cfg.task_manifest).write_text(json.dumps(task) + "\n")
    limits = NativeLimits(
        max_depth=cfg.max_depth,
        max_turns=cfg.max_turns,
        max_children=cfg.max_children,
        max_live_children=cfg.max_live_children,
        max_concurrent_requests=cfg.max_concurrent_requests,
        max_model_calls=cfg.max_model_calls,
        request_timeout=cfg.request_timeout,
        episode_timeout=cfg.episode_timeout,
        max_prompt_chars=cfg.max_prompt_chars,
    )
    policy = ContextBudgetPolicy(
        server_context_tokens=cfg.server_context_tokens,
        output_fraction=cfg.output_fraction,
        max_generated_tokens=cfg.max_generated_tokens,
        logprobs=cfg.logprobs,
    )
    saved = {
        "session": {"context": task_context(task), "prompt": "frozen user"},
        "config": {
            "model": cfg.model_id,
            "limits": asdict(limits),
            "budget_policy": asdict(policy),
            "bootstrap_code": docs_bootstrap(docs) + resources_bootstrap(resources),
            "system_prompt": "frozen system",
            "enable_thinking": True,
            "stream_responses": True,
            "child_repl_at_limit": True,
        },
    }
    assert check_checkpoint_binding(saved, tmp_path, {}, cfg) == task
    changed = copy.deepcopy(saved)
    changed["config"]["budget_policy"]["max_generated_tokens"] += 1
    with pytest.raises(ValueError, match="frozen"):
        check_checkpoint_binding(changed, tmp_path, {}, cfg)
    changed = copy.deepcopy(saved)
    changed["session"]["context"]["cute_dsl_task"] = "different task"
    with pytest.raises(ValueError, match="specification"):
        check_checkpoint_binding(changed, tmp_path, {}, cfg)
