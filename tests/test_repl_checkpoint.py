"""Data-only REPL state recovery without replaying prior tool actions."""

import ast
import json

import pytest
from rlm_train.repl.snapshot import decode_namespace, encode_namespace
from rlm_train.worker import Worker


def test_snapshot_preserves_supported_types_aliases_and_ast():
    shared = [1, 2]
    data = {
        "source": "class ModelNew: pass",
        "contract": {"shape": (64, 128)},
        "tree": ast.parse("x = 1"),
        "ast_module": ast,
        "left": shared,
        "right": shared,
        "raw": b"abc",
        "seen": {"x", "y"},
    }
    snapshot = json.loads(json.dumps(encode_namespace(data)))
    restored = decode_namespace(snapshot)
    assert restored["left"] is restored["right"]
    assert restored["contract"]["shape"] == (64, 128)
    assert ast.dump(restored["tree"]) == ast.dump(data["tree"])
    assert restored["ast_module"] is ast
    assert restored["raw"] == b"abc" and restored["seen"] == {"x", "y"}


def test_snapshot_reports_unsupported_values_instead_of_serializing_code():
    snapshot = encode_namespace({"source": "candidate", "fn": lambda: None})
    assert snapshot["unsupported"][0]["name"] == "fn"
    with pytest.raises(ValueError, match="unsupported"):
        decode_namespace(snapshot)


def test_worker_restore_retains_helpers_and_does_not_execute_source():
    original = Worker("http://unused", "old")
    original.execute(
        "import ast; source = 'raise RuntimeError()'; tree = ast.parse(source); best_source = source"
    )
    snapshot = original.snapshot()
    restored = Worker("http://unused", "new")
    restored.load_context({"reference_code": "reference"})
    restored.restore(json.loads(json.dumps(snapshot)))
    result = restored.execute(
        "assert best_source == source; print(len(tree.body)); answer['content']=source; answer['ready']=True"
    )
    assert not result["stderr"]
    assert result["final_answer"] == "raise RuntimeError()"
    assert callable(restored.globals["compile_candidate"])
    assert restored.locals["context"] == {"reference_code": "reference"}


@pytest.mark.asyncio
async def test_subprocess_snapshot_restore_roundtrip(monkeypatch):
    import os
    from pathlib import Path

    from rlm_train.repl.subprocess import SubprocessReplBackend

    src = str(Path(__file__).parents[1] / "training/src")
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join(filter(None, (src, os.environ.get("PYTHONPATH"))))
    )
    first, second = SubprocessReplBackend(), SubprocessReplBackend()
    try:
        await first.start("http://unused", "old", depth=0)
        await first.load_context("reference")
        await first.execute("contract={'shape':(64,128)}; best_source='candidate'; counter=3")
        snapshot = await first.snapshot()
        await first.stop()
        await second.start("http://unused", "new", depth=0)
        await second.load_context("reference")
        await second.restore(snapshot)
        result = await second.execute(
            "counter+=1; print(counter,contract['shape']); answer['content']=best_source; answer['ready']=True"
        )
        assert result.final_answer == "candidate" and "4 (64, 128)" in result.stdout
    finally:
        await first.stop()
        await second.stop()


def test_immutable_cycle_is_explicitly_unsupported_not_falsely_recoverable():
    items = []
    loop = (items,)
    items.append(loop)
    snapshot = encode_namespace({"loop": loop})
    assert snapshot["unsupported"]
    with pytest.raises(ValueError, match="unsupported"):
        decode_namespace(snapshot)
