"""Loading exceptions retain the owner across extension compilation."""

import ast
import hashlib
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("owner", ["candidate", "reference"])
def test_inline_compile_preserves_loading_owner(owner, tmp_path):
    from kernel_rlm.verifier import _RUNNER

    tree = ast.parse(_RUNNER)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "tracked_load_inline"
    )
    events = [{"phase": "load_" + owner, "owner": owner}]

    def mark_phase(phase, owner="harness"):
        events.append({"phase": phase, "owner": owner})

    namespace = {
        "phase_events": events,
        "mark_phase": mark_phase,
        "hashlib": hashlib,
        "extension_root": tmp_path,
        "extension_builds": [],
        "TrackedExtension": lambda value: value,
        "original_load_inline": lambda **kwargs: SimpleNamespace(__name__="fixture"),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "loading-owner", "exec"), namespace)
    namespace["tracked_load_inline"](name="fixture")
    assert events[-1] == {"phase": "extension_compile", "owner": owner}


@pytest.mark.parametrize(
    "owner,expected", [("candidate", "candidate_error"), ("reference", "infrastructure_error")]
)
def test_actual_loading_handler_classifies_after_inline_failure(owner, expected, tmp_path, capsys):
    from kernel_rlm.verifier import _RUNNER

    tree = ast.parse(_RUNNER)
    loading = next(
        node
        for node in tree.body
        if isinstance(node, ast.Try) and "load_reference" in ast.unparse(node)
    )
    failure = loading.handlers[0]
    body = ast.parse("raise NameError(\"name 'source' is not defined\")").body
    scoped = ast.Try(body=body, handlers=[failure], orelse=[], finalbody=[])
    result = {}
    import json

    namespace = {
        "guard": {},
        "mode": "verify",
        "result": result,
        "phase_events": [{"phase": "extension_compile", "owner": owner}],
        "json": json,
        "torch": SimpleNamespace(OutOfMemoryError=MemoryError),
    }
    with pytest.raises(SystemExit) as exited:
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[scoped], type_ignores=[])),
                "loading-handler",
                "exec",
            ),
            namespace,
        )
    assert exited.value.code == 0
    assert result["verification_status"] == expected
    assert (result.get("failure_kind") == "reference_loading") is (owner == "reference")
    assert "NameError" in result["error"]
