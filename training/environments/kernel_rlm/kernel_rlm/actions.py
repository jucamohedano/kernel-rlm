"""Normalize supported model actions before executing a REPL block."""

import ast
import json
from collections.abc import Callable

from rlm.utils.parsing import find_code_blocks

FEEDBACK_MAX_CALLS = 2
FEEDBACK_TIMEOUT_S = 120
REPL_TIMEOUT_S = FEEDBACK_MAX_CALLS * FEEDBACK_TIMEOUT_S + 30
BACKEND_TIMEOUT_S = REPL_TIMEOUT_S + 30
SUBMISSION_TURN = 6
RECOVERY_MAX_TOKENS = 512
RECOVERY_MIN_TOKENS = 128
REPL_TOOL = {
    "type": "function",
    "function": {
        "name": "repl",
        "description": "Run Python code in the persistent task REPL. Use only this tool.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}


def load_token_counter(model: str) -> Callable[[str], int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def recovery_budget(
    usage: dict,
    pending_messages: list[dict],
    context_tokens: int,
    max_tokens: int,
    count_tokens: Callable[[str], int],
) -> tuple[int, int | None]:
    used = usage.get("total_tokens")
    if not isinstance(used, int) or used <= 0:
        return 0, None
    pending_tokens = sum(
        count_tokens(json.dumps(message, ensure_ascii=False)) for message in pending_messages
    )
    estimated_prompt = used + pending_tokens + 256
    available = context_tokens - estimated_prompt
    budget = min(RECOVERY_MAX_TOKENS, max_tokens, available)
    return (budget if budget >= RECOVERY_MIN_TOKENS else 0), estimated_prompt


def preparation_error(code: str, compiler_names: set[str], *, recovery: bool = False) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "Recovery requires a valid Python submission action." if recovery else None
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "torch.utils.cpp_extension":
            compiler_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name in {"load_inline", "load"}
            )
        if not recovery and isinstance(node, ast.Call) and (
            isinstance(node.func, ast.Name)
            and node.func.id in compiler_names
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == "load_inline"
        ):
            return (
                "Do not compile an extension directly in the REPL. Store the complete "
                "program as a source string and call cuda_feedback(source), which coordinates "
                "compilation and GPU access."
            )
    if not recovery:
        return None
    allowed = (
        ast.Module,
        ast.Assign,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Subscript,
        ast.Constant,
        ast.Dict,
        ast.Expr,
    )
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        return "Recovery is for answer assignments only; no imports, feedback, or delegation."
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)]
    if not assignments:
        return "Submit the stored source with answer['content'] = source; answer['ready'] = True."
    for assignment in assignments:
        for target in assignment.targets:
            if isinstance(target, ast.Name) and target.id == "answer":
                continue
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "answer"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value in {"content", "ready"}
            ):
                continue
            return "Recovery may only assign answer or its content and ready fields."
    return None


def repl_actions(message: dict, finish_reason: str) -> tuple[list[str], list[str]]:
    if finish_reason == "length":
        return [], ["Response truncated; no actions executed. Retry with a complete REPL block."]
    structured = []
    errors = []
    for call in message.get("tool_calls") or []:
        function = call.get("function", call)
        if function.get("name") != "repl":
            errors.append(
                f"Unsupported tool {function.get('name')!r}. The only tool is repl; "
                'call it with JSON arguments such as {"code": "print(context)"}.'
            )
            continue
        try:
            arguments = function.get("arguments", "{}")
            arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            errors.append("Malformed repl arguments; expected JSON with a code string.")
            continue
        if not isinstance(arguments, dict) or not isinstance(arguments.get("code"), str):
            errors.append("Invalid repl arguments; expected a code string.")
            continue
        structured.append(arguments["code"].strip())
    fenced = find_code_blocks(message.get("content") or "")
    if message.get("tool_calls"):
        if any(code not in structured for code in fenced):
            errors.append("Mixed action formats: only structured calls executed; use one format.")
        return structured, errors
    if not fenced:
        errors.append(
            "No executable REPL action found. Use a ```repl``` block or an actual repl "
            'tool call with JSON arguments {"code": "..."}. Text describing tool calls '
            "is not executed. If your candidate is ready, submit it by setting "
            "answer['content'] = source and answer['ready'] = True inside a REPL block."
        )
    return fenced, errors
