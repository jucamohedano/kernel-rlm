import ast
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def candidate(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "training/environments/kernel_rlm"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(is_tensor=lambda value: False))
    return importlib.import_module("kernel_rlm.candidate")


def test_clone_keeps_nested_constructor_inputs_independent(candidate):
    original = [{"axes": [1, 2]}, (3, 4)]
    cloned = candidate.clone_inputs(original)
    cloned[0]["axes"][0] = 99
    assert original[0]["axes"] == [1, 2]
    assert cloned[1] == (3, 4)


@pytest.mark.parametrize(
    "expected,actual,error",
    [
        ((1, 2), [1, 2], "sequence type"),
        ([1, 2], [1], "sequence length"),
        ({"a": 1}, {"b": 1}, "mapping keys"),
        (1, True, "metadata"),
    ],
)
def test_output_structure_differences_are_not_ignored(candidate, expected, actual, error):
    correct, difference, message = candidate.compare_outputs(expected, actual, 1e-3, 1e-4)
    assert not correct and difference == 0
    assert error in message


def test_nested_metadata_can_match_without_changing_values(candidate):
    value = {"out": (1, [2, 3])}
    assert candidate.compare_outputs(value, value, 1e-3, 1e-4) == (True, 0.0, "")


def test_legacy_function_remains_the_direct_entrypoint(candidate):
    submission = SimpleNamespace(cuda_forward=lambda value: value)
    assert candidate.candidate_forward(submission, None, [], "cpu") is submission.cuda_forward


def test_modelnew_is_a_valid_standalone_submission(candidate):
    from kernel_rlm.submission_contract import validate_submission_source

    source = (
        "from torch.utils.cpp_extension import load_inline\n"
        "CPP_SOURCE='declaration'\nCUDA_SOURCE='implementation'\n"
        "module=load_inline(name='example', cpp_sources=CPP_SOURCE, cuda_sources=CUDA_SOURCE)\n"
        "class ModelNew:\n    def forward(self, x):\n        return module.launch(x)\n"
    )
    assert validate_submission_source(source) == (True, "ok")
    invalid = source.replace("def forward", "def unrelated")
    assert "cuda_forward or ModelNew.forward" in validate_submission_source(invalid)[1]


def test_reference_and_candidate_benchmarks_both_disable_autograd(candidate):
    from kernel_rlm.verifier import _RUNNER

    tree = ast.parse(_RUNNER)
    protected = [
        node
        for block in ast.walk(tree)
        if isinstance(block, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "no_grad"
            for item in block.items
        )
        for node in ast.walk(block)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "benchmark"
    ]
    assert len(protected) == 2
