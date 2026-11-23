"""Bounded native generation and private evidence handoff; never executes CUDA."""

from __future__ import annotations

import asyncio
import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from . import task

SERVED_MODEL = "rlm27b-f0b3b27db57c-bf16"
ROOT_TURNS = 8
CHILD_CALLS = 4
OUTPUT_TOKENS = 2048
WALL_SECONDS = 900


def sampling_args() -> dict[str, Any]:
    return {
        "temperature": 0,
        "max_tokens": OUTPUT_TOKENS,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }


def require_loopback(endpoint: str) -> str:
    parts = urlsplit(endpoint)
    if (
        parts.scheme != "http"
        or parts.hostname != "127.0.0.1"
        or not parts.port
        or parts.path != "/v1"
        or parts.query
        or parts.fragment
        or parts.username
        or parts.password
    ):
        raise ValueError("expected a private http://127.0.0.1:<port>/v1 endpoint")
    return endpoint


def visible_message(message: Any) -> dict[str, str]:
    role = message["role"] if isinstance(message, dict) else message.role
    content = message["content"] if isinstance(message, dict) else message.content
    if not isinstance(content, str):
        raise ValueError("expected a native text message")
    if role == "assistant":
        content = content.rsplit("</think>", 1)[-1] if "</think>" in content else content
        content = content.split("<think>", 1)[0] if "<think>" in content else content
    return {"role": role, "content": content}


def write_bytes(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fp:
        fp.write(payload)
        fp.flush()
        os.fsync(fp.fileno())


def write_record(path: Path, record: dict[str, Any]) -> None:
    write_bytes(path, (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode())


def seal_state(directory: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Seal one attempt, without granting permission to compile its output."""
    record: dict[str, Any] = {
        "scope": "generation_only_no_device_verdict",
        "stop": state.get("stop_condition"),
        "error_class": type(state["error"]).__name__ if state.get("error") else None,
        "root_requests": [
            {
                "prompt": [visible_message(msg) for msg in step["prompt"]],
                "completion": [visible_message(msg) for msg in step["completion"]],
            }
            for step in state.get("trajectory", [])
        ],
        "upstream_subcall_count": state.get("rlm_sub_llm_calls", 0),
        "upstream_subcalls_scope": "bounded native proxy log; not exact wire payloads",
        "upstream_subcalls": [
            {
                **entry,
                "response": visible_message({"role": "assistant", "content": entry["response"]})[
                    "content"
                ],
            }
            for entry in state.get("rlm_sub_llm_log", [])
        ],
        "source_sha256": None,
        "kernel_verdict": None,
    }
    if state.get("error") or state.get("rlm_default_answer") or "rlm_final_answer" not in state:
        record["status"] = "no_submission"
    elif state["rlm_final_answer"] is None:
        record["status"] = "no_submission"
    else:
        try:
            artifact = task.source_from_state(state)
        except ValueError:
            record["status"] = "invalid_submission"
        else:
            write_bytes(directory / "submission.cu", artifact.source)
            record["status"] = "submitted_held_for_review"
            record["source_sha256"] = artifact.source_sha256
    write_record(directory / "result.json", record)
    return record


async def generate_one(directory: Path, env, client, *, model: str) -> dict[str, Any]:
    """Run the upstream turn loop once; no retry, score, model answer replay or fallback.

    `client` and `env` are supplied by the approved controller, never by the REPL.
    `directory` must be a fresh private controller path outside the worktree.
    """
    directory.mkdir(mode=0o700, exist_ok=False)
    context = task.public_context().encode()
    write_bytes(directory / "task-context.txt", context)
    write_record(
        directory / "intent.json",
        {
            "harness_revision": task.HARNESS_CG_REVISION,
            "checkpoint": f"{task.MODEL_ID}@{task.MODEL_REVISION}",
            "context_sha256": sha256(context).hexdigest(),
            "served_model": model,
            "caps": [ROOT_TURNS, CHILD_CALLS, OUTPUT_TOKENS, WALL_SECONDS],
            "sampling": sampling_args(),
        },
    )
    try:
        state = await asyncio.wait_for(
            env.rollout(env.get_eval_dataset()[0], client, model, sampling_args()), WALL_SECONDS
        )
        return seal_state(directory, state)
    except Exception as exc:
        if not (directory / "result.json").exists():
            write_record(directory / "failed.json", {"error_class": type(exc).__name__})
        raise
    finally:
        await env.teardown_rlm()


def require_author_controls(receipt: dict[str, Any]) -> None:
    """Check bindings in the private operator receipt, not its authenticity."""
    from .vm_smoke import SOURCE_SHA256, controls_plan

    if (
        receipt.get("schema") != "ragged_author_controls_v1"
        or receipt.get("mode") != "live"
        or receipt.get("status") != "author_controls_passed"
    ):
        raise ValueError("a passing live author-control receipt is required")
    current_sources = {
        name: sha256((task.HERE / name).read_bytes()).hexdigest()
        for name in (*SOURCE_SHA256, "reference.py", "score.py")
    }
    if receipt.get("sources") != current_sources or any(
        current_sources[name] != pin for name, pin in SOURCE_SHA256.items()
    ):
        raise ValueError("control receipt sources disagree with this checkout")
    expected = {
        (control, fixture.name): (status, sha256(fixture.to_wire().encode("ascii")).hexdigest())
        for control, fixture, status in controls_plan()
    }
    cases = receipt.get("cases", [])
    observed = {
        (case["control"], case["fixture"]): (case["observed"], case["input_sha256"])
        for case in cases
    }
    if len(cases) != len(expected) or observed != expected:
        raise ValueError("incomplete or incorrect live control receipts")
    for case in cases:
        build = receipt["builds"][case["control"]]
        if build["status"] != "built" or build["sha256"] != case["artifact_sha256"]:
            raise ValueError("control receipt does not bind capture to build")
        if case["process"]["status"] != "exited" or case["process"]["returncode"] != 0:
            raise ValueError("a failed device process is not a numerical result")


def require_serving(manifest: dict[str, Any]) -> str:
    """Verify declared pins; the operator must verify the running server separately."""
    expected = {
        "model": task.MODEL_ID,
        "revision": task.MODEL_REVISION,
        "served_model": SERVED_MODEL,
        "dtype": "bfloat16",
        "max_model_len": 32768,
    }
    topology = (manifest.get("gpu"), manifest.get("tensor_parallel_size"))
    if any(manifest.get(key) != value for key, value in expected.items()) or topology not in {
        ("L40S", 2),
        ("RTX PRO 6000 Blackwell", 1),
    }:
        raise ValueError("serving manifest does not match an approved BF16 pilot topology")
    return require_loopback(manifest["endpoint"])


def require_pinned_harness(checkout: Path) -> Path:
    from subprocess import check_output

    import rlm_train.env

    checkout = checkout.resolve(strict=True)
    upstream = checkout / "src/rlm_train"
    if Path(rlm_train.env.__file__).resolve() != upstream / "env.py":
        raise ValueError("imported rlm_train must come from the pinned Harness-CG checkout")
    commit = check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if commit != task.HARNESS_CG_REVISION:
        raise ValueError("Harness-CG checkout is not at the required revision")
    dirty = check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--", "src/rlm_train"], text=True
    )
    if dirty:
        raise ValueError("Harness-CG runtime files must be unchanged")
    return upstream / "worker.py"


async def run_live(args) -> dict[str, Any]:
    import httpx
    from openai import AsyncOpenAI
    from verifiers import OpenAIChatCompletionsClient

    from .sandbox import SandboxedReplBackend

    controls = args.author_controls.read_bytes()
    serving = args.serving_manifest.read_bytes()
    require_author_controls(json.loads(controls))
    endpoint = require_serving(json.loads(serving))
    if args.out_dir.resolve().is_relative_to(Path(__file__).resolve().parents[5]):
        raise ValueError("private run artifacts must stay outside the repository")
    if args.out_dir.exists():
        raise ValueError("refusing a second attempt in an existing output directory")
    if os.environ.get("RLM_TRAIN_SUBCALL_TIMEOUT_S") != "80":
        raise ValueError("set the trusted native proxy deadline: RLM_TRAIN_SUBCALL_TIMEOUT_S=80")
    key = os.environ.get("RAGGED_MODEL_API_KEY")
    if not key:
        raise ValueError("the trusted controller requires RAGGED_MODEL_API_KEY")
    backend = SandboxedReplBackend(
        require_pinned_harness(args.harness_checkout),
        args.python_runtime,
        max_sub_calls=CHILD_CALLS,
        wall_seconds=WALL_SECONDS,
    )
    env = task.native_environment(
        backend_factory=lambda: backend,
        max_iterations=ROOT_TURNS,
        max_sub_calls=CHILD_CALLS,
        max_rollout_seconds=WALL_SECONDS,
        sub_sampling_args=sampling_args(),
    )
    async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as transport:
        async with AsyncOpenAI(
            base_url=endpoint, api_key=key, max_retries=0, timeout=120, http_client=transport
        ) as oai:
            result = await generate_one(
                args.out_dir, env, OpenAIChatCompletionsClient(oai), model=SERVED_MODEL
            )
    write_record(
        args.out_dir / "preflight.json",
        {
            "author_controls_sha256": sha256(controls).hexdigest(),
            "serving_manifest_sha256": sha256(serving).hexdigest(),
        },
    )
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-approved-generation-only", action="store_true", required=True)
    parser.add_argument("--author-controls", type=Path, required=True)
    parser.add_argument("--serving-manifest", type=Path, required=True)
    parser.add_argument("--harness-checkout", type=Path, required=True)
    parser.add_argument("--python-runtime", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(run_live(args))
    print(result["status"])
    return 0 if result["status"] == "submitted_held_for_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
