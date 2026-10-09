"""Normalize supported model actions before executing a REPL block."""

import json

from rlm.utils.parsing import find_code_blocks

FEEDBACK_MAX_CALLS = 2
FEEDBACK_TIMEOUT_S = 120
REPL_TIMEOUT_S = FEEDBACK_MAX_CALLS * FEEDBACK_TIMEOUT_S + 30
BACKEND_TIMEOUT_S = REPL_TIMEOUT_S + 30


def repl_actions(message: dict, finish_reason: str) -> tuple[list[str], list[str]]:
    if finish_reason == "length":
        return [], ["Response truncated; no actions executed. Retry with a complete REPL block."]
    structured = []
    errors = []
    for call in message.get("tool_calls") or []:
        function = call.get("function", call)
        if function.get("name") != "repl":
            errors.append(
                f"Unsupported tool {function.get('name')!r}; use repl with a code string."
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
