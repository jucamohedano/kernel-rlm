import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from training.environments.kernel_rlm.research.ragged_gather_reduce import task


def test_native_row_only_carries_public_material():
    row = task.native_row()
    assert row["question"] == row["info"]["root_prompt"]
    assert row["answer"] == "" and set(row["info"]) == {"root_prompt", "context"}
    assert row["info"]["context"] == task.public_context()
    assert (task.HERE / "contract.h").read_text() in task.public_context()
    assert "rlm_replay" not in row["info"]


def test_submit_exact_bytes_from_native_state():
    raw = "// original source line \n\n"
    saved = task.source_from_state({"rlm_final_answer": raw, "final_answer": "other"})
    assert saved.source == raw.encode()
    assert saved.source_sha256 == hashlib.sha256(raw.encode()).hexdigest()
    with pytest.raises(ValueError, match="no committed"):
        task.source_from_state({"final_answer": "fallback"})


@pytest.mark.parametrize("source", [" ", "```cuda\n```", "a\x00b", "a" * (64 * 1024 + 1)])
def test_non_source_or_oversize_not_repaired(source):
    with pytest.raises(ValueError):
        task.source_from_state({"rlm_final_answer": source})


def test_requires_explicit_backend_and_limits():
    with pytest.raises(ValueError, match="isolated REPL"):
        task.native_environment(
            backend_factory=None, max_iterations=2, max_sub_calls=0, max_rollout_seconds=1
        )
    with pytest.raises(ValueError, match="rollout limits"):
        task.native_environment(
            backend_factory=lambda: None,
            max_iterations=1,
            max_sub_calls=0,
            max_rollout_seconds=1,
        )


def test_pinned_worker_retains_context_state_and_exact_submit(tmp_path):
    checkout = os.environ.get("HARNESS_CG_CHECKOUT")
    if not checkout:
        pytest.skip("set HARNESS_CG_CHECKOUT to the pinned checkout for the offline worker check")
    git_env = {"PATH": os.environ["PATH"], "GIT_CONFIG_NOSYSTEM": "1"}
    commit = subprocess.check_output(
        ["git", "-C", checkout, "rev-parse", "HEAD"], env=git_env, text=True
    ).strip()
    assert commit == task.HARNESS_CG_REVISION
    worker = Path(checkout) / "src" / "rlm_train" / "worker.py"
    status = subprocess.check_output(
        ["git", "-C", checkout, "status", "--porcelain", "--", str(worker)],
        env=git_env,
        text=True,
    )
    assert status == "", "native worker must be unchanged"
    source = "// static test marker\n"
    requests = [
        {"id": "load", "type": "load_context", "payload": task.public_context()},
        {
            "id": "read",
            "type": "exec",
            "code": "seen = 'owner entry' in context; print(seen, callable(llm_query))",
        },
        {"id": "buffer", "type": "exec", "code": f"draft_source = {source!r}; print(seen)"},
        {
            "id": "commit",
            "type": "exec",
            "code": 'answer["content"] = draft_source; answer["ready"] = True',
        },
        {"id": "stop", "type": "shutdown"},
    ]
    result = subprocess.run(
        [sys.executable, str(worker)],
        input="".join(json.dumps(request) + "\n" for request in requests),
        text=True,
        capture_output=True,
        check=True,
        cwd=tmp_path,
        env={
            "RLM_TRAIN_DEPTH": "1",
            "RLM_TRAIN_PROXY_URL": "http://127.0.0.1:1",
            "RLM_TRAIN_ROLLOUT_ID": "offline-check",
            "TMPDIR": str(tmp_path),
        },
        timeout=20,
    )
    replies = {reply.get("id"): reply for reply in map(json.loads, result.stdout.splitlines())}
    assert all(replies[request["id"]]["ok"] for request in requests)
    assert replies["read"]["stdout"].strip() == "True True"
    assert replies["buffer"]["stdout"].strip() == "True"
    assert replies["read"].get("final_answer") is None
    assert replies["buffer"].get("final_answer") is None
    sealed = task.source_from_state({"rlm_final_answer": replies["commit"]["final_answer"]})
    assert sealed.source == source.encode()


def test_upstream_env_folds_native_actions_without_inference(tmp_path, monkeypatch):
    checkout = os.environ.get("HARNESS_CG_CHECKOUT")
    if not checkout:
        pytest.skip("enable the pinned Harness-CG integration checks")
    vf = pytest.importorskip("verifiers")
    pytest.importorskip("datasets")
    upstream = pytest.importorskip("rlm_train.env")
    from rlm_train.repl.subprocess import SubprocessReplBackend

    assert Path(upstream.__file__).resolve().is_relative_to(Path(checkout).resolve())
    monkeypatch.setenv("RLM_TRAIN_REPL_SCRATCH", str(tmp_path))

    async def check():
        env = task.native_environment(
            # Trusted scripted inputs only; not for model-generated Python.
            backend_factory=SubprocessReplBackend,
            max_iterations=3,
            max_sub_calls=1,
            max_rollout_seconds=10,
        )
        dataset = env.get_eval_dataset()
        assert len(dataset) == 1
        # No SDK client: an accidental model request must fail without network I/O.
        client = vf.OpenAIChatCompletionsClient(None)
        state = await env.init_state(dataset[0], client, model="offline-script")
        state["trajectory"] = []
        try:
            await env.setup_state(state)
            prompt = await env.get_prompt_messages(state)
            assert len(prompt) == 3  # System, question/metadata, first turn; no teacher hint.
            assert "owner entry" not in prompt[1]["content"]
            steps = [
                "```repl\nseen = 'owner entry' in context; print(seen)\n```",
                '```repl\nanswer["content"] = "// offline\\n" if seen else ""\n'
                'answer["ready"] = True\n```',
            ]
            for text in steps:
                state["trajectory"].append({"completion": [{"role": "assistant", "content": text}]})
                await env.get_prompt_messages(state)
            assert state["rlm_repl_calls"] == 2
            assert task.source_from_state(state).source == b"// offline\n"
        finally:
            await env.cleanup_rlm(state)
            await env.teardown_rlm()

    asyncio.run(check())
