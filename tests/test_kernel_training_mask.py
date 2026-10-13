import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_child_mask_without_external_artifacts(monkeypatch):
    root = Path(__file__).parents[1] / "training/environments/kernel_rlm"
    monkeypatch.syspath_prepend(str(root))
    from kernel_rlm.subcalls import CHILD_SYSTEM
    from kernel_rlm.training_mask import mask_child_branches

    nodes = [
        SimpleNamespace(
            parent=parent,
            message=SimpleNamespace(content=content),
            mask=[sampled],
            token_ids=[index],
            logprobs=[-0.1] if sampled else [],
        )
        for index, (parent, content, sampled) in enumerate(
            [
                (None, "root system", False),
                (0, "root action", True),
                (None, CHILD_SYSTEM, False),
                (2, "child answer", True),
                (1, "child reply as observation", False),
                (4, "root submission", True),
            ]
        )
    ]
    trace = SimpleNamespace(nodes=nodes, info={})
    mask_child_branches(trace)
    assert [node.mask for node in nodes] == [[False], [True], [False], [False], [False], [True]]
    assert nodes[3].message.content == "child answer"
    assert trace.info["child_loss_exclusions"]["3"]["logprobs"] == [-0.1]


def test_saved_child_branch_is_excluded_but_root_is_unchanged(monkeypatch):
    root = Path(__file__).parents[1] / "training/environments/kernel_rlm"
    monkeypatch.syspath_prepend(str(root))
    from kernel_rlm.training_mask import mask_child_branches

    path = root / "logs/frozen-probe/20261009T112652Z-298b409b/traces.jsonl"
    if not path.exists():
        pytest.skip("Saved GPU trace is not downloaded")
    data = json.loads(path.read_text().splitlines()[0])["traces"][0]
    trace = SimpleNamespace(nodes=[SimpleNamespace(**node) for node in data["nodes"]], info={})
    for node in trace.nodes:
        node.message = SimpleNamespace(**node.message)
    originals = [list(node.mask) for node in trace.nodes]
    mask_child_branches(trace)
    excluded = trace.info["child_loss_exclusions"]
    assert list(excluded) == ["7"]
    assert sum(excluded["7"]["mask"]) == 902
    assert sum(sum(node.mask) for node in trace.nodes) == 1816
    for index, node in enumerate(trace.nodes):
        if index != 7:
            assert node.mask == originals[index]
    assert trace.nodes[7].message.content == data["nodes"][7]["message"]["content"]
    mask_child_branches(trace)
    assert sum(excluded["7"]["mask"]) == 902


def test_unknown_branch_fails_closed(monkeypatch):
    root = Path(__file__).parents[1] / "training/environments/kernel_rlm"
    monkeypatch.syspath_prepend(str(root))
    from kernel_rlm.training_mask import mask_child_branches

    nodes = [
        SimpleNamespace(parent=None, message=SimpleNamespace(content=content))
        for content in ["root", "unknown"]
    ]
    with pytest.raises(ValueError, match="Unknown independent"):
        mask_child_branches(SimpleNamespace(nodes=nodes, info={}))


def test_delegation_and_final_turn_instructions(monkeypatch):
    root = Path(__file__).parents[1] / "training/environments/kernel_rlm"
    monkeypatch.syspath_prepend(str(root))
    from kernel_rlm.subcalls import KERNEL_SYSTEM, SUBCALL_INSTRUCTION, turn_budget

    assert "specific subproblem" in SUBCALL_INSTRUCTION
    assert "Never send bare context" in SUBCALL_INSTRUCTION
    assert "100k" not in KERNEL_SYSTEM + SUBCALL_INSTRUCTION
    prompt = turn_budget(4, 5, 2, True, True)
    assert "Feedback calls remaining: 0" in prompt
    assert "Child calls remaining: 0" in prompt
    assert "FINAL TURN" in prompt
