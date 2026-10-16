import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1] / "training/environments/kernel_rlm"


@pytest.fixture
def recover_string():
    tree = ast.parse((ROOT / "modal_app/verifier_replay.py").read_text())
    function = next(node for node in tree.body if getattr(node, "name", None) == "literal_string")
    namespace = {"ast": ast}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "recovery", "exec"), namespace)
    return namespace["literal_string"]


def test_recovery_preserves_source_escaping_without_executing_it(recover_string):
    values = {"cuda": 'line one\n"line two"'}
    expression = ast.parse("'CUDA_SOURCE = ' + repr(cuda) + '\\n'", mode="eval").body
    recovered = recover_string(expression, values)
    assert ast.literal_eval(ast.parse(recovered).body[0].value) == values["cuda"]


@pytest.mark.parametrize(
    "expression", ["__import__('os').system('false')", "open('file').read()", "repr(x=1)"]
)
def test_recovery_rejects_executable_expressions(recover_string, expression):
    with pytest.raises(ValueError):
        recover_string(ast.parse(expression, mode="eval").body, {})
