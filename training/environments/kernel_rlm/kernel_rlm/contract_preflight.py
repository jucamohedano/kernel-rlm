"""Conservative meta-tensor interface checks without compiling or running CUDA."""

import ast
import inspect
import json
import re
import resource
import sys
from pathlib import Path
from types import SimpleNamespace


class UnsupportedContract(Exception):
    pass


class ContractFailure(Exception):
    pass


class ExtensionReached(BaseException):
    pass


def extension_signature(source: str, name: str) -> inspect.Signature:
    matches = re.findall(r"\b" + re.escape(name) + r"\s*\(([^()]*)\)\s*(?:;|\{)", source)
    if len(matches) != 1:
        raise UnsupportedContract("Expected one unambiguous C++ declaration: " + name)
    arguments = matches[0].strip()
    if any(character in arguments for character in "=[]") or "..." in arguments:
        raise UnsupportedContract("Complex or defaulted C++ signature: " + name)
    if arguments == "void":
        arguments = ""
    parts = []
    start = 0
    depth = 0
    for index, character in enumerate(arguments):
        if character == "<":
            depth += 1
        elif character == ">":
            depth -= 1
            if depth < 0:
                raise UnsupportedContract("Unbalanced C++ template signature: " + name)
        elif character == "," and depth == 0:
            parts.append(arguments[start:index])
            start = index + 1
    if depth:
        raise UnsupportedContract("Unbalanced C++ template signature: " + name)
    if arguments:
        parts.append(arguments[start:])
    parameters = []
    for index, argument in enumerate(parts):
        if not re.fullmatch(r"[\w:\s*&<>,]+", argument.strip()):
            raise UnsupportedContract("Unsupported C++ parameter: " + argument)
        parameters.append(inspect.Parameter(f"argument_{index}", inspect.Parameter.POSITIONAL_ONLY))
    return inspect.Signature(parameters)


def contract_namespace(source: str, name: str) -> dict:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [entry.name for entry in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if any(module.split(".")[0] not in {"torch", "math"} for module in modules):
                raise UnsupportedContract("Preflight supports torch/math imports only")
        elif not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.Assign, ast.AnnAssign)):
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Constant):
                raise UnsupportedContract("Unsupported module-level statement")
    namespace = {"__name__": name}
    exec(compile(tree, name, "exec"), namespace)
    return namespace


def check_contract(reference_code: str, submission_code: str) -> dict:
    import torch
    import torch.utils.cpp_extension

    from kernel_rlm.candidate import candidate_forward

    original = torch.utils.cpp_extension.load_inline

    def mock_load_inline(*args, **kwargs):
        source = kwargs.get("cpp_sources", args[1] if len(args) > 1 else "")
        if isinstance(source, list):
            source = "\n".join(source)
        functions = kwargs.get("functions", args[3] if len(args) > 3 else None)
        if isinstance(functions, str):
            functions = [functions]
        if not isinstance(source, str) or not isinstance(functions, list):
            raise UnsupportedContract("Explicit load_inline functions are required")

        def binding(name):
            signature = extension_signature(source, name)

            def call(*arguments, **keywords):
                try:
                    signature.bind(*arguments, **keywords)
                except TypeError as error:
                    raise ContractFailure(f"Extension {name}: {error}") from error
                raise ExtensionReached()

            return call

        return SimpleNamespace(**{name: binding(name) for name in functions})

    try:
        torch.utils.cpp_extension.load_inline = mock_load_inline
        with torch.device("meta"):
            reference = SimpleNamespace(**contract_namespace(reference_code, "contract_reference"))
            submission = SimpleNamespace(
                **contract_namespace(submission_code, "contract_candidate")
            )
            init_inputs = getattr(reference, "get_init_inputs", lambda: [])()
            model = reference.Model(*init_inputs).eval()
            inputs = reference.get_inputs()
            try:
                candidate = candidate_forward(submission, model, init_inputs, "meta")
            except RuntimeError as error:
                if "Error(s) in loading state_dict" not in str(error):
                    raise
                raise ContractFailure(str(error)) from error
            function = candidate.forward if isinstance(candidate, torch.nn.Module) else candidate
            try:
                inspect.signature(function).bind(*inputs)
            except TypeError as error:
                raise ContractFailure(f"Candidate wrapper: {error}") from error
            candidate(*inputs)
            raise UnsupportedContract("No supported extension call reached")
    except ExtensionReached:
        return {"status": "interface_passed", "numerical_correctness_tested": False}
    except ContractFailure as error:
        return {"status": "contract_failed", "error": str(error)}
    except Exception as error:
        return {"status": "unsupported", "error": f"{type(error).__name__}: {error}"}
    finally:
        torch.utils.cpp_extension.load_inline = original


def main():
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    result = check_contract(Path(sys.argv[1]).read_text(), Path(sys.argv[2]).read_text())
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preflight unexpectedly initialized CUDA")
    print(json.dumps({**result, "cuda_initialized": False}))


if __name__ == "__main__":
    main()
