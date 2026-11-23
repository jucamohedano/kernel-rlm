"""Public task payload and exact-source handoff for the pinned Harness-CG REPL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TASK_ID = "ragged-gather-reduce-dev-v0"
HARNESS_CG_REVISION = "5272afdcb25a156c1b9b89cb479ae0604af249d7"
MODEL_ID = "a1zhang/RLM-27B"
MODEL_REVISION = "f0b3b27db57c211234cb55ec512d87f45dabdc4c"
MAX_SOURCE_BYTES = 64 * 1024
ROOT_PROMPT = (
    "Write the CUDA C++ implementation specified in context. "
    "Submit only the source for one .cu translation unit as the string "
    'in answer["content"] and set answer["ready"] = True. '
    "Do not include Markdown fences or surrounding explanation in the submitted source."
)


def public_context() -> str:
    contract = (HERE / "contract.md").read_text(encoding="utf-8")
    header = (HERE / "contract.h").read_text(encoding="utf-8")
    return f"{contract}\n## contract.h (provided by the grader)\n\n```cpp\n{header}```\n"


def native_row() -> dict[str, Any]:
    """The upstream Verifiers row shape; gold inputs are not stored in the row."""
    return {
        "question": ROOT_PROMPT,
        "answer": "",
        "info": {"context": public_context(), "root_prompt": ROOT_PROMPT},
    }


def native_environment(
    *, backend_factory, max_iterations: int, max_sub_calls: int, max_rollout_seconds: float
):
    """Wire into Harness-CG, in its own pinned environment.

    The caller must supply an isolated worker backend. The default local
    subprocess is not a security boundary and is intentionally not selected.
    This does not configure credentials, serving, GPU feedback or live scoring.
    """
    if not callable(backend_factory):
        raise ValueError("an explicit isolated REPL backend factory is required")
    if max_iterations < 2 or max_sub_calls < 0 or max_rollout_seconds <= 0:
        raise ValueError("explicit positive rollout limits are required")

    datasets = import_module("datasets")
    upstream = import_module("rlm_train")

    return upstream.RLMTrainEnv(
        eval_dataset=datasets.Dataset.from_list([native_row()]),
        backend_factory=backend_factory,
        max_iterations=max_iterations,
        max_sub_calls=max_sub_calls,
        max_rollout_seconds=max_rollout_seconds,
        enable_sub_llm=True,
        orchestrator_prompt=False,
        default_answer_at_limit=False,
        code_block_feedback=False,
    )


@dataclass(frozen=True)
class Submission:
    source: bytes
    source_sha256: str


def source_from_state(state: dict[str, Any]) -> Submission:
    """Export the native ready-answer bytes, never a fallback chat completion."""
    text = state.get("rlm_final_answer")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("no committed native source answer")
    if "```" in text or "\x00" in text:
        raise ValueError("submit raw CUDA source, not a fenced or binary payload")
    source = text.encode("utf-8", errors="strict")
    if len(source) > MAX_SOURCE_BYTES:
        raise ValueError("submission exceeds the 64 KiB source cap")
    return Submission(source, sha256(source).hexdigest())


if __name__ == "__main__":
    # Offline inspection only. No model client or compiler is imported.
    print(json.dumps(native_row(), ensure_ascii=False))
