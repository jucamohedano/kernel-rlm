"""Shared CUDA-submission instructions and cheap validation."""

from __future__ import annotations

import ast

KERNEL_TASK_INSTRUCTION = (
    "Inspect the PyTorch reference in context, then write a standalone Python source "
    "string defining CPP_SOURCE, CUDA_SOURCE, and cuda_forward(*tensors). The source "
    "must implement the reference operation with a compiled CUDA extension.\n\n"
    "The submission must import load_inline from torch.utils.cpp_extension and call "
    "torch.utils.cpp_extension.load_inline with cpp_sources=CPP_SOURCE and "
    "cuda_sources=CUDA_SOURCE. Build the extension at module import time, store the "
    "returned module in a global such as _module, and make cuda_forward call the "
    "bound extension function. Do not use torch.utils.cpp_extension.load, "
    "load_extension, build_cuda_extension, source filenames, rebuild=, "
    "extra_cuda_flags=, or extra_nvcc_flags=. Those APIs are not accepted by the "
    "verifier. Include every import and binding needed by the source.\n\n"
    "The REPL is CPU-only: inspect context and construct the source string, but do "
    "not compile or run CUDA there. A separate GPU verifier imports the submitted "
    "source, checks fresh inputs, and benchmarks only after correctness passes. Do "
    "not delegate to sub-LLMs. After inspecting context, set answer['content'] to "
    "the complete standalone Python source and answer['ready'] = True inside a "
    "```repl``` block before the final allowed turn ends."
)


def _call_name(function: ast.expr) -> str:
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        prefix = _call_name(function.value)
        return f"{prefix}.{function.attr}"
    return ""


def validate_submission_source(source: str) -> tuple[bool, str]:
    """Validate the verifier-facing Python/API contract without importing it."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return False, f"syntax error: {error.msg}"

    assigned_names = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign | ast.AnnAssign)
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name)
    }
    if "CPP_SOURCE" not in assigned_names or "CUDA_SOURCE" not in assigned_names:
        return False, "source must define CPP_SOURCE and CUDA_SOURCE"
    if not any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name == "cuda_forward"
        for node in tree.body
    ):
        return False, "source must define cuda_forward"

    load_inline_calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name == "load_inline" or name.endswith(".load_inline"):
            load_inline_calls.append(node)
        if name in {"load_extension", "build_cuda_extension"}:
            return False, f"unsupported extension builder: {name}"
        if name.endswith(".cpp_extension.load"):
            return False, "use load_inline instead of cpp_extension.load"
        keyword_names = {keyword.arg for keyword in node.keywords if keyword.arg}
        for keyword in ("rebuild", "extra_cuda_flags", "extra_nvcc_flags"):
            if keyword in keyword_names:
                return False, f"unsupported load keyword: {keyword}"

    if not load_inline_calls:
        return False, "source must call torch.utils.cpp_extension.load_inline"
    if not any(
        {keyword.arg for keyword in call.keywords if keyword.arg}
        >= {"cpp_sources", "cuda_sources"}
        for call in load_inline_calls
    ):
        return False, "load_inline must receive cpp_sources and cuda_sources"
    return True, "ok"
