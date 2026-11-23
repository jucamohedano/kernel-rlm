"""One generation-only attempt in the pinned native Harness-CG REPL."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..ragged_gather_reduce import pilot
from ..ragged_gather_reduce.sandbox import SandboxedReplBackend
from ..ragged_gather_reduce.task import HARNESS_CG_REVISION

SOURCE_BYTES = 128 * 1024
DESCRIPTOR_BYTES = 4 * 1024
# One JSON string may require six ASCII bytes per decoded byte, plus the outer keys.
ANSWER_BYTES = 6 * (SOURCE_BYTES + DESCRIPTOR_BYTES) + 1024
CONTEXT_BYTES = 128 * 1024
ROOT_PROMPT = (
    "Write the CUDA device source and launch descriptor specified in context. "
    'Set answer["content"] to one JSON object serialized as text with exactly two string fields: '
    '"source" (raw CUDA device source) and "descriptor" (JSON object text). '
    'Do not include Markdown fences. Set answer["ready"] = True to commit.'
)


@dataclass(frozen=True)
class Budget:
    root_turns: int
    child_calls: int
    output_tokens: int
    wall_seconds: int

    def __post_init__(self) -> None:
        for value, least, most in (
            (self.root_turns, 2, pilot.ROOT_TURNS),
            (self.child_calls, 0, pilot.CHILD_CALLS),
            (self.output_tokens, 1, pilot.OUTPUT_TOKENS),
            (self.wall_seconds, 1, pilot.WALL_SECONDS),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or not least <= value <= most:
                raise ValueError("explicit rollout budget outside the pinned pilot caps")

    def sampling(self) -> dict[str, Any]:
        return {**pilot.sampling_args(), "max_tokens": self.output_tokens}


@dataclass(frozen=True)
class Submission:
    answer: bytes
    source: bytes
    descriptor_text: bytes
    descriptor: dict[str, Any]


class FormatError(ValueError):
    """A committed answer did not pass preliminary transport parsing."""


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON key")
        obj[key] = value
    return obj


def json_object(text: str, label: str) -> dict[str, Any]:
    try:
        obj = json.loads(text, object_pairs_hook=unique_object)
        # Reject non-finite values (including overflowed exponents) and lone surrogates.
        json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, RecursionError) as exc:
        raise FormatError(f"invalid_{label}_json") from exc
    if not isinstance(obj, dict):
        raise FormatError(f"{label}_not_object")
    return obj


def checked_bytes(text: Any, cap: int, label: str) -> bytes:
    if not isinstance(text, str):
        raise FormatError(f"{label}_not_string")
    try:
        data = text.encode("utf-8")
    except UnicodeError as exc:
        raise FormatError(f"{label}_not_utf8") from exc
    if len(data) > cap:
        raise FormatError(f"{label}_too_large")
    if not text.strip():
        raise FormatError(f"empty_{label}")
    if "\x00" in text:
        raise FormatError(f"nul_in_{label}")
    if text.lstrip().startswith(("```", "~~~")):
        raise FormatError(f"fenced_{label}")
    return data


def parse_submission(text: str) -> Submission:
    """Validate transport only; no ABI semantics, source inspection or execution."""
    answer = checked_bytes(text, ANSWER_BYTES, "answer")
    envelope = json_object(text, "answer")
    if set(envelope) != {"source", "descriptor"}:
        raise FormatError("wrong_envelope_keys")
    source = checked_bytes(envelope["source"], SOURCE_BYTES, "source")
    descriptor_text = checked_bytes(envelope["descriptor"], DESCRIPTOR_BYTES, "descriptor")
    descriptor = json_object(envelope["descriptor"], "descriptor")
    return Submission(answer, source, descriptor_text, descriptor)


def seal_state(out_dir: Path, state: dict[str, Any], broker: dict[str, int]) -> dict[str, Any]:
    """Write visible native evidence; a receipt is not permission to compile."""
    error = state.get("error")
    result = {
        "scope": "generation_only_no_device_verdict",
        "status": "no_submission",
        "reason": "no_native_commit",
        "stop": state.get("stop_condition"),
        "error_class": type(error).__name__ if error else None,
        "kernel_verdict": None,
        "artifact_sha256": None,
        "root_requests": [
            {
                key: [pilot.visible_message(msg) for msg in step[key]]
                for key in ("prompt", "completion")
            }
            for step in state.get("trajectory", [])
        ],
        "upstream_subcall_count": state.get("rlm_sub_llm_calls", 0),
        "upstream_subcalls_scope": "bounded native proxy log; not exact wire payloads",
        "upstream_subcalls": [
            {
                **entry,
                "response": pilot.visible_message(
                    {"role": "assistant", "content": entry["response"]}
                )["content"],
            }
            for entry in state.get("rlm_sub_llm_log", [])
        ],
        "child_broker": broker,
    }
    if error:
        reason = "wall_timeout" if state.get("stop_condition") == "timeout" else "rollout_error"
        result.update(status="unavailable", reason=reason)
    elif state.get("rlm_default_answer"):
        result.update(reason="fallback_answer")
    elif (text := state.get("rlm_final_answer")) is not None:
        try:
            submission = parse_submission(text)
        except FormatError as exc:
            reason = str(exc)
            result.update(
                status="no_submission" if reason == "empty_answer" else "invalid_submission",
                reason=reason,
            )
        else:
            files = {
                "answer.json": submission.answer,
                "submission.cu": submission.source,
                "descriptor.json": submission.descriptor_text,
            }
            for name, data in files.items():
                pilot.write_bytes(out_dir / name, data)
            result.update(
                status="submitted_held_for_review",
                reason="preliminary_transport_parse_only",
                artifact_sha256={name: sha256(data).hexdigest() for name, data in files.items()},
            )
    pilot.write_record(out_dir / "result.json", result)
    return result


async def generate_one(
    *,
    task_id: str,
    public_context: str,
    budget: Budget,
    out_dir: Path,
    harness_checkout: Path,
    python_runtime: Path,
    client: Any,
    model: str,
    serving_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Run once with a trusted controller client (scripted in offline tests)."""
    from datasets import Dataset
    from rlm_train import RLMTrainEnv

    checked_bytes(task_id, 256, "task_id")
    if not public_context.strip() or len(context := public_context.encode("utf-8")) > CONTEXT_BYTES:
        raise ValueError("expected a nonempty bounded public task context")
    if not model:
        raise ValueError("an explicit controller model is required")
    out_dir = out_dir.resolve()
    if out_dir.is_relative_to(Path(__file__).resolve().parents[5]):
        raise ValueError("private run artifacts must stay outside the repository")
    backend = SandboxedReplBackend(
        pilot.require_pinned_harness(harness_checkout),
        python_runtime,
        max_sub_calls=budget.child_calls,
        wall_seconds=budget.wall_seconds,
    )
    sampling = budget.sampling()
    row = {
        "question": ROOT_PROMPT,
        "answer": "",
        "info": {"context": public_context, "root_prompt": ROOT_PROMPT},
    }
    env = RLMTrainEnv(
        eval_dataset=Dataset.from_list([row]),
        backend_factory=lambda: backend,
        max_iterations=budget.root_turns,
        max_sub_calls=budget.child_calls,
        max_rollout_seconds=budget.wall_seconds,
        sub_model=model,
        sub_sampling_args=sampling,
        enable_sub_llm=True,  # The pinned sandbox requires native depth 1, even when its gate cap is 0.
        orchestrator_prompt=False,
        default_answer_at_limit=False,
        code_block_feedback=False,
    )
    out_dir.mkdir(mode=0o700, exist_ok=False)
    pilot.write_bytes(out_dir / "task-context.txt", context)
    pilot.write_record(
        out_dir / "intent.json",
        {
            "task_id": task_id,
            "context_sha256": sha256(context).hexdigest(),
            "harness_revision": HARNESS_CG_REVISION,
            "served_model": model,
            "serving_manifest_sha256": serving_manifest_sha256,
            "budget": asdict(budget),
            "sampling": sampling,
        },
    )
    try:
        async with asyncio.timeout(budget.wall_seconds):
            state = await env.rollout(env.get_eval_dataset()[0], client, model, sampling)
    except TimeoutError as exc:
        state = {"error": exc, "stop_condition": "timeout"}
    except Exception as exc:
        state = {"error": exc}
    finally:
        try:
            # Also covers cancellation during native setup, before the env stores the backend.
            await backend.stop()
        finally:
            await env.teardown_rlm()
    return seal_state(out_dir, state, backend.broker_evidence)


async def run(
    *,
    task_id: str,
    public_context: str,
    budget: Budget,
    out_dir: Path,
    harness_checkout: Path,
    python_runtime: Path,
    serving_manifest: Path,
    api_key: str,
) -> dict[str, Any]:
    """Controller opt-in only: use an already running, separately verified loopback model."""
    import httpx
    from openai import AsyncOpenAI
    from verifiers import OpenAIChatCompletionsClient

    manifest_bytes = serving_manifest.read_bytes()
    manifest = json_object(manifest_bytes.decode("utf-8"), "manifest")
    endpoint = pilot.require_serving(manifest)
    if not api_key:
        raise ValueError("the controller must supply its existing API key")
    if budget.child_calls and os.environ.get("RLM_TRAIN_SUBCALL_TIMEOUT_S") != "80":
        raise ValueError("set the pinned native proxy deadline: RLM_TRAIN_SUBCALL_TIMEOUT_S=80")
    async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as transport:
        async with AsyncOpenAI(
            base_url=endpoint, api_key=api_key, max_retries=0, timeout=120, http_client=transport
        ) as oai:
            return await generate_one(
                task_id=task_id,
                public_context=public_context,
                budget=budget,
                out_dir=out_dir,
                harness_checkout=harness_checkout,
                python_runtime=python_runtime,
                client=OpenAIChatCompletionsClient(oai),
                model=manifest["served_model"],
                serving_manifest_sha256=sha256(manifest_bytes).hexdigest(),
            )
