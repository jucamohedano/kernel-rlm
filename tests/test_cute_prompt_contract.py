"""Detailed prompt rendering and executable string-assembly guidance, CPU only."""

import ast

from cute_rlm.config import EvalConfig
from cute_rlm.prompts import SOURCE_ASSEMBLY_EXAMPLE, first_user_message


def test_task_prompt_renders_the_actual_frozen_execution_contract():
    config = EvalConfig(
        task_manifest="tasks.jsonl",
        docs_manifest="docs.json",
        seed=7,
        task_count=1,
        max_turns=20,
        episode_timeout=900,
        verification={
            "atol": 0.002,
            "rtol": 0.003,
            "seed": 7,
            "mode": "eval",
            "correctness_runs": 2,
            "warmup_runs": 1,
            "timed_runs": 4,
            "timeout_s": 120,
            "input_device": "cuda",
        },
    )
    rendered = first_user_message({"task_id": "task-7"}, config)
    assert "task-7" in rendered
    assert "20 model turns" in rendered and "turn 20" in rendered
    assert "900 seconds" in rendered
    assert "atol=0.002, rtol=0.003" in rendered
    assert "module mode is eval" in rendered
    assert "2 correctness trials, 1 warmups and 4 timing samples" in rendered
    assert "sm_120" in rendered


def test_assembly_example_produces_complete_source_text_without_executing_it():
    calls = []
    namespace = {
        "kernel_source": "def kernel_placeholder():\n    return 1\n",
        "model_source": "class ModelNew: pass\n",
        "compile_candidate": lambda source: calls.append(source) or {"status": "compiled"},
    }
    exec(SOURCE_ASSEMBLY_EXAMPLE, namespace)
    ast.parse(namespace["source"])
    assert len(calls) == 1
    assert calls[0] == namespace["source"]
    assert "ModelNew" not in namespace  # The example assembles text, not candidate execution.
    assert namespace["feedback"] == {"status": "compiled"}
