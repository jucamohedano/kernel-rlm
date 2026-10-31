"""Concise prompt rendering and initial native-REPL inspection, CPU only."""

import ast

import pytest
from cute_rlm.config import EvalConfig
from cute_rlm.prompts import CUTE_SYSTEM_PROMPT, first_user_message


@pytest.mark.parametrize("max_turns", [20, None])
def test_task_prompt_inspects_reference_and_indexed_resources_once(max_turns, capsys):
    config = EvalConfig(
        task_manifest="tasks.jsonl",
        docs_manifest="docs.json",
        seed=7,
        task_count=1,
        max_turns=max_turns,
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
    expected_budget = (
        "20 model turns per REPL session"
        if max_turns is not None
        else "no fixed per-session turn limit"
    )
    assert expected_budget in rendered
    inspection = rendered.split("Start with one native REPL action:\n", 1)[1].split("\n\n", 1)[0]
    ast.parse(inspection)
    reference = "raise AssertionError('reference must remain source text')"
    resources = {
        "target": {"gpu_arch": "sm_89"},
        "validation": {"mode": "preserve"},
        "limits": {"max_turns": max_turns},
        "example_index": [{"id": "addition", "purpose": "generic API example"}],
        "api_notes": ["Use the caller's stream."],
        "examples": {"addition": "GENERIC_EXAMPLE_SOURCE_NOT_DUMPED"},
        "helper_help": {"read_doc": "HELPER_SCHEMA_NOT_DUMPED"},
    }
    exec(inspection, {"context": {"reference_code": reference}, "resources": resources})
    assert capsys.readouterr().out.splitlines() == [
        reference,
        *(
            f"{key} {resources[key]}"
            for key in ("target", "validation", "limits", "example_index", "api_notes")
        ),
    ]
    assert 'resources["examples"]' in rendered and 'resources["helper_help"]' in rendered


def test_system_finalizer_submits_the_stored_complete_source():
    final_section = CUTE_SYSTEM_PROMPT.split("even if uncertain:\n", 1)[1]
    finalizer = final_section.split("\nSetting ready", 1)[0]
    namespace = {"answer": {"content": "", "ready": False}, "best_source": "class ModelNew: pass"}
    exec(finalizer, namespace)
    assert namespace["answer"] == {"content": namespace["best_source"], "ready": True}
