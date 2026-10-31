"""Concise prompt rendering and initial native-REPL inspection, CPU only."""

import ast

import pytest
from cute_rlm.config import EvalConfig
from cute_rlm.prompts import CUTE_SYSTEM_PROMPT, ORCHESTRATOR_ADDENDUM, first_user_message


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
    assert "task-7" not in rendered
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
    inventory = {
        "best_source": "str",
        "compile_candidate": "compile_candidate(source: str)",
        "search_docs": "search_docs(query, limit=2)",
        "rlm_query": "rlm_query(prompt, blocking=True)",
    }
    inventory_calls = []

    def show_vars():
        inventory_calls.append(True)
        return inventory

    exec(
        inspection,
        {
            "SHOW_VARS": show_vars,
            "context": {"reference_code": reference},
            "resources": resources,
        },
    )
    assert inventory_calls == [True]
    assert capsys.readouterr().out.splitlines() == [
        str(inventory),
        reference,
        str(resources["target"]),
        str(resources["example_index"]),
    ]
    assert 'resources["examples"]' in rendered and 'resources["helper_help"]' in rendered


def test_prompt_compile_action_calls_injected_helper_once_and_keeps_feedback(capsys):
    action = next(
        line for line in CUTE_SYSTEM_PROMPT.splitlines() if line.startswith("feedback = ")
    )
    calls = []
    feedback = {"status": "compiled", "detail": "forward/synchronize smoke"}

    def compile_candidate(source):
        calls.append(source)
        print(feedback)
        return feedback

    source = "class ModelNew: pass"
    namespace = {
        "compile_candidate": compile_candidate,
        "best_source": source,
        "answer": {"content": "", "ready": False},
    }
    exec(action, namespace)
    assert calls == [source]
    assert namespace["feedback"] is feedback
    assert namespace["best_source"] == source
    assert namespace["answer"]["ready"] is False
    assert capsys.readouterr().out.splitlines() == [str(feedback)]


def test_system_finalizer_submits_the_stored_complete_source():
    assert CUTE_SYSTEM_PROMPT.endswith(ORCHESTRATOR_ADDENDUM)
    final_section = CUTE_SYSTEM_PROMPT.split("or on the final turn:\n", 1)[1]
    finalizer = final_section.split("\n\n", 1)[0]
    namespace = {"answer": {"content": "", "ready": False}, "best_source": "class ModelNew: pass"}
    exec(finalizer, namespace)
    assert namespace["answer"] == {"content": namespace["best_source"], "ready": True}
