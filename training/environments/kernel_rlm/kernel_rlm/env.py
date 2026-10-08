"""CUDA kernel tasks wired into the local RLM training loop."""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import uuid
from typing import Any

import rlm_train
from datasets import Dataset

from kernel_rlm.feedback import build_http_feedback_bootstrap
from kernel_rlm.submission_contract import KERNEL_TASK_INSTRUCTION, validate_submission_source
from kernel_rlm.verifier import verify_cuda


def _build_dataset(*, manifest_path: str, max_tasks: int | None = None, **kwargs: Any) -> Dataset:
    """Turn a JSONL manifest into rows consumed by RLMTrainEnv."""
    rows = {"example_id": [], "prompt": [], "answer": [], "info": []}
    with open(manifest_path, encoding="utf-8") as manifest_file:
        for index, line in enumerate(manifest_file):
            if max_tasks is not None and index >= max_tasks:
                break
            example = json.loads(line)
            task_id = example.get("_task_id", str(index))
            root_prompt = (
                f"{KERNEL_TASK_INSTRUCTION}\n\n"
                f"Task {task_id}.\n"
                f"Operators: {example.get('ops', '')}."
            )
            rows["example_id"].append(task_id)
            rows["prompt"].append([{"role": "user", "content": "<placeholder>"}])
            rows["answer"].append("")
            rows["info"].append(
                json.dumps(
                    {
                        "context": example["code"],
                        "root_prompt": root_prompt,
                        "ops": example.get("ops", ""),
                    }
                )
            )
    return Dataset.from_dict(rows)


async def score(info: Any, state: Any, **_kwargs: Any) -> float:
    """Verify a CUDA submission and return a correctness-first reward."""
    metadata = json.loads(info) if isinstance(info, str) else (info or {})
    submitted = (state.get("rlm_final_answer") or state.get("final_answer") or "").strip()
    if not submitted:
        return 0.0

    is_valid, validation_error = validate_submission_source(submitted)
    if not is_valid:
        state["kernel_verify"] = {
            "compiled": False,
            "extension_used": False,
            "correct": False,
            "max_diff": None,
            "speedup": None,
            "ref_ms": None,
            "kernel_ms": None,
            "error": f"static validation: {validation_error}",
        }
        return 0.0

    result = await asyncio.to_thread(
        verify_cuda,
        str(metadata.get("context") or ""),
        submitted,
    )

    reward = 0.10
    if result.get("compiled") and result.get("extension_used"):
        reward += 0.20
    if result.get("correct"):
        reward += 0.40
        speedup = result.get("speedup")
        if speedup is not None:
            reward += 0.30 * min(max(float(speedup), 0.0), 3.0) / 3.0

    state["kernel_verify"] = {
        "compiled": bool(result.get("compiled")),
        "extension_used": bool(result.get("extension_used")),
        "correct": bool(result.get("correct")),
        "max_diff": result.get("max_diff"),
        "speedup": result.get("speedup"),
        "ref_ms": result.get("ref_ms"),
        "kernel_ms": result.get("kernel_ms"),
        "error": (result.get("error") or "")[:500],
    }
    state["kernel_submission_chars"] = len(submitted)

    trace_dir = os.environ.get("KERNEL_TRACE_DIR")
    if trace_dir:
        output_dir = pathlib.Path(trace_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rollout_id = state.get("rlm_rollout_id") or state.get("trajectory_id") or uuid.uuid4().hex
        record = {
            "rollout_id": rollout_id,
            "example_id": state.get("example_id"),
            "reward": reward,
            "verify": state["kernel_verify"],
            "submission": submitted[:20000],
            "metrics": {
                key: state.get(key)
                for key in (
                    "rlm_iterations",
                    "rlm_repl_calls",
                    "rlm_sub_llm_calls",
                    "rlm_has_final_answer",
                )
            },
            "completion": state.get("completion"),
        }
        (output_dir / f"{rollout_id}.json").write_text(
            json.dumps(record, default=str),
            encoding="utf-8",
        )
    return reward


async def verify_compiled(state: Any) -> int:
    verification = state.get("kernel_verify") or {}
    return int(bool(verification.get("compiled") and verification.get("extension_used")))


async def verify_extension_used(state: Any) -> int:
    return int(bool((state.get("kernel_verify") or {}).get("extension_used")))


async def verify_correct(state: Any) -> int:
    return int(bool((state.get("kernel_verify") or {}).get("correct")))


async def verify_speedup(state: Any) -> float:
    speedup = (state.get("kernel_verify") or {}).get("speedup")
    return round(float(speedup), 3) if speedup is not None else 0.0


async def verify_max_diff(state: Any) -> float:
    max_diff = (state.get("kernel_verify") or {}).get("max_diff")
    return round(float(max_diff), 6) if max_diff is not None else -1.0


def load_environment(
    *,
    manifest_path: str,
    max_iterations: int = 8,
    feedback_url: str | None = None,
    feedback_max_calls: int = 2,
    bootstrap_code: str | None = None,
    **kwargs: Any,
) -> Any:
    """Wire the CUDA task dataset and scorer into the local RLM environment."""
    dataset = _build_dataset(manifest_path=manifest_path, **kwargs)
    rubric = rlm_train.RLMTrainRubric(correctness=score, weight=1.0)
    rubric.add_metric(verify_compiled)
    rubric.add_metric(verify_extension_used)
    rubric.add_metric(verify_correct)
    rubric.add_metric(verify_speedup)
    rubric.add_metric(verify_max_diff)
    feedback_code = (
        build_http_feedback_bootstrap(feedback_url, max_calls=feedback_max_calls)
        if feedback_url
        else ""
    )
    combined_bootstrap = "\n\n".join(
        code for code in (bootstrap_code or "", feedback_code) if code
    )
    return rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        rubric=rubric,
        sub_sampling_args={"max_tokens": 512},
        orchestrator=False,
        bootstrap_code=combined_bootstrap,
    )
