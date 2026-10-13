"""Shared CUDA-submission instructions and cheap validation."""

from __future__ import annotations

import ast
import textwrap

KERNEL_TASK_INSTRUCTION = (
    "Inspect the PyTorch reference in the read-only context variable. Never assign to "
    "context or context_0; use a new variable for your work. Before coding, read the "
    "forward signature and get_inputs() to determine the exact input count, shapes, "
    "and dtypes. Match cuda_forward(*tensors) and the exported C++ launch signature "
    "to that input count. Then write a standalone Python source "
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
    "The REPL provides cuda_feedback(source), which submits a candidate to a bounded "
    "GPU verifier and returns a dictionary containing compiled, extension_used, "
    "correct, max_diff, and error. Call cuda_feedback on each serious candidate "
    "before final submission so you can use compilation errors and numerical "
    "feedback to repair the kernel. Feedback checks correctness but does not run "
    "the performance benchmark. The helper has a small call budget, so do not "
    "call it on trivial text. Read all error details. If feedback says the bound "
    "C++ call returned None, make the exported launch function return the output tensor. "
    "If the TypeError lists two input tensors, accept two tensors; do not guess from "
    "the operation's name. Do not delegate to sub-LLMs. After using feedback, "
    "set answer['content'] to the complete standalone Python source and "
    "answer['ready'] = True inside a ```repl``` block before the final allowed "
    "turn ends. Do not print the candidate instead of setting both answer fields."
)


CUDA_ASSEMBLY_TEMPLATE = textwrap.dedent(
    r"""
    source = (
        "import torch\nfrom torch.utils.cpp_extension import load_inline\n"
        + "CPP_SOURCE = " + repr(cpp_declarations) + "\n"
        + "CUDA_SOURCE = " + repr(cuda_implementation) + "\n"
        + "_module = load_inline(name='kernel_rlm_candidate', "
        + "cpp_sources=CPP_SOURCE, cuda_sources=CUDA_SOURCE, "
        + "functions=['launch'], extra_cuda_cflags=['-O3'], verbose=False)\n"
        + "def cuda_forward(*tensors):\n    return _module.launch(*tensors)\n"
    )
    import ast
    ast.parse(source, filename="<candidate>", mode="exec")
    result = cuda_feedback(source)
    print(result)
    """
).strip()

CUDA_BUILD_GUIDE = textwrap.dedent(
    """

    CUDA build rules and expected workflow:

    1. Read the read-only context once. Never overwrite context or context_0.
       Inspect the reference's forward signature and get_inputs() to confirm the
       exact argument count, shapes, dtypes, and output behavior. Implement the computation yourself in a CUDA kernel;
       do not call the reference operation or delegate the computation to PyTorch.
    2. Keep CPP_SOURCE to includes and a declaration of the exported function
       named launch. Put __global__ kernels, their definitions, the matching
       launch function definition, and every <<<...>>> kernel launch in CUDA_SOURCE.
       The ordinary C++ compiler must never receive CUDA kernel-launch syntax.
       Include <torch/extension.h> in both translation units and <cuda_runtime.h>
       in CUDA_SOURCE. Match declaration and definition names, argument types,
       and return type exactly; use Tensor.numel(), not numels().
    3. Use the binding scaffold below. functions=["launch"] generates the Python
       binding automatically: do NOT also write PYBIND11_MODULE or TORCH_LIBRARY.
       The bound C++ function must return the output tensor if cuda_forward must
       return a tensor; a void function returns None to Python. Adapt the launch
       signature to the reference inputs; the task-independent
       scaffold intentionally does not supply the kernel computation.
       Do not set is_python_module=False: cuda_forward needs a Python module,
       not a library path string. Use only the load_inline keywords shown here.
       Never add is_pure_python, keep_debug_buffers, opt_level, or invented options.
    4. Prepare complete source as a string in the REPL; do not exec it or build
       an extension directly in the REPL. Import ast and check Python syntax with
       ast.parse(source, filename="<candidate>", mode="exec") without executing it.
       The REPL disables compile, exec, and eval; do not call those builtins.
       Then call cuda_feedback(source) and print the returned dictionary.
       There are exactly TWO feedback calls per trajectory. Inspect compiled,
       extension_used, correct, and error; max_diff alone does not prove success.
       Repair the reported error before spending the second call. If correct,
       preserve that exact candidate and submit it immediately.
    5. Reserve the last turn for submission. After the second feedback call,
       make any necessary final repair and submit your best complete source,
       even if verification failed. Do not ask for exhausted feedback.
       The last REPL block must set answer["content"] = source and
       answer["ready"] = True. Printing source does not submit it.

    Assembly scaffold: run after creating task-specific cpp_declarations and
    cuda_implementation strings. This is assembly code, not the final submission.
    Submit the assembled source string. Use repl fences for actual REPL actions.
    Keep launch's signature consistent with the tensors passed to cuda_forward.
    """
)

KERNEL_TASK_INSTRUCTION += "\n\n" + CUDA_BUILD_GUIDE + "\n" + CUDA_ASSEMBLY_TEMPLATE


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
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    if "CPP_SOURCE" not in assigned_names or "CUDA_SOURCE" not in assigned_names:
        return False, "source must define CPP_SOURCE and CUDA_SOURCE"
    if not any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "cuda_forward"
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
        {keyword.arg for keyword in call.keywords if keyword.arg} >= {"cpp_sources", "cuda_sources"}
        for call in load_inline_calls
    ):
        return False, "load_inline must receive cpp_sources and cuda_sources"
    return True, "ok"
