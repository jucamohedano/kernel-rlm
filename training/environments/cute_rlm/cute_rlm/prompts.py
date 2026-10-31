"""Short Python CuTe DSL instructions and child-REPL orchestration."""

from __future__ import annotations

from cute_rlm.config import EvalConfig

ORCHESTRATOR_ADDENDUM = """Act as an orchestrator. At the root, inspect the task, then use rlm_query(prompt, blocking=False) to delegate a concrete implementation or compile repair to a child while you work. Pass the reference source, complete candidate if available, and exact compiler feedback: children have their own persistent REPL and do not inherit your variables. A child's context is its delegated prompt string. It can use the same injected helpers and resources. Ask it to return complete source and actual compile feedback.

Collect the child with rlm_wait(handle, timeout_s=30) or rlm_result(handle), then use its work in best_source before submitting. A wait timeout leaves it running. Children at depth 1 must not spawn further children; they finish by setting answer["content"] to their requested report and answer["ready"] = True. See resources["delegation_example"] for a short example."""

CUTE_SYSTEM_PROMPT = (
    """Implement the task using Python CuTe DSL (import cutlass.cute as cute), introduced in mid-2025—not the older C++ CUTLASS/CuTe API. At the root, return standalone Python defining ModelNew(torch.nn.Module) with the supplied constructor and forward signatures. The goal is compile/forward-smoke PASS, not benchmarking.

Use only native repl(code); variables persist and the reasoning REPL has no GPU. At the root, context["reference_code"] holds the task as source text. resources is a dictionary containing the target, indexed examples and helper help.

Helpers are already injected: SHOW_VARS(), compile_candidate(source), search_docs(query), read_doc(doc_id), and rlm_query/rlm_wait/rlm_result. Call them directly inside repl; do not import or redefine them. Store the complete candidate in best_source and run:
feedback = compile_candidate(best_source)

Feedback prints automatically. Fix the reported compile/runtime error and try again. Keep CuTe decorators directly in the source string, not inside exec. Use Torch only for state, allocation and host plumbing.

Submit from the root when ready, or on the final turn:
answer["content"] = best_source
answer["ready"] = True
"""
    + "\n"
    + ORCHESTRATOR_ADDENDUM
)


def first_user_message(task: dict, config: EvalConfig) -> str:
    turn_budget = (
        f"{config.max_turns} model turns per REPL session"
        if config.max_turns is not None
        else "no fixed per-session turn limit"
    )
    return f"""Implement context["reference_code"] in Python CuTe DSL. Start with one native REPL action:
print(SHOW_VARS())
print(context["reference_code"])
print(resources["target"])
print(resources["example_index"])

Read the needed sections from resources["examples"] using the indexed character offsets. Delegate implementation or repair to a child, compile the complete candidate, and submit best_source. Helper details are in resources["helper_help"].
Budget: {turn_budget}.
"""
