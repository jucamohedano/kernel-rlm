"""Run against the patched Prime runtime, or explicitly test extracted source."""

import ast
import os
from pathlib import Path

import pytest
import torch
from rlm_train.sft.bundle import collate_group, load_bundle, masked_nll_sum, write_synthetic_fixture
from rlm_train.sft.train_prime import _probe_logits
from torch.nn import functional as F


@pytest.fixture
def seam():
    if os.environ.get("XS21_PRIME_NATIVE_IMPORT") == "1":
        from prime_rl.trainer.sft.data import grouped_synthetic_sft_loss, setup_dataloader

        return setup_dataloader, grouped_synthetic_sft_loss
    if file := os.environ.get("XS21_PRIME_PATCHED_SOURCE"):
        source = ast.parse(Path(file).read_text(), filename=file)
        wanted = {
            "setup_dataloader",
            "cat_collate",
            "_grouped_synthetic_batches",
            "grouped_synthetic_sft_loss",
        }
        selected = [
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        ]
        assert {node.name for node in selected} == wanted
        future = ast.parse("from __future__ import annotations").body
        module = ast.fix_missing_locations(ast.Module(body=future + selected, type_ignores=[]))
        ns = {"torch": torch}  # stock loader symbols absent: using one would fail immediately
        exec(compile(module, file, "exec"), ns)
        return ns["setup_dataloader"], ns["grouped_synthetic_sft_loss"]
    pytest.skip("set XS21_PRIME_NATIVE_IMPORT or XS21_PRIME_PATCHED_SOURCE")


def test_grouped_loader_and_loss_parity(seam, tmp_path):
    loader, group_loss = seam
    pin = tmp_path, write_synthetic_fixture(tmp_path)
    groups = tuple(loader(None, None, 1, grouped_synthetic=pin))
    assert [len(g) for g in groups] == [2, 1]
    assert [sum(int(b["loss_mask"].sum()) for b in g) for g in groups] == [2, 1]
    assert [b["input_ids"].tolist() for g in groups for b in g] == [[[1, 2, 3]], [[5, 0]], [[5, 1]]]
    table = torch.randn(16, 16, generator=torch.Generator().manual_seed(7), requires_grad=True)
    seen = []

    def compute_loss(batch):
        mask = batch["loss_mask"]
        ids = batch["input_ids"]
        seen.append(ids.tolist())
        assert ids.shape == (1, ids.shape[-1])
        assert batch["position_ids"].tolist() == [list(range(ids.shape[-1]))]
        assert batch["seq_lens"].tolist() == [ids.shape[-1]]
        assert batch["seq_lens"].dtype == torch.long
        return F.cross_entropy(
            _probe_logits(table, ids)[mask], batch["target_ids"][mask], reduction="sum"
        ), mask.sum()

    got = group_loss(groups, compute_loss)
    assert seen == [[[1, 2, 3]], [[5, 1]]]
    bundle = load_bundle(*pin, allow_synthetic=True)
    reference = []
    for group in bundle.groups:
        b = collate_group(
            group,
            pad_id=bundle.pad_id,
            member_cap=bundle.member_cap,
            padded_slots_cap=bundle.padded_slots_cap,
        )
        reference.append(masked_nll_sum(_probe_logits(table, b["input_ids"]), b["labels"]))
    expected = sum(loss for loss, _ in reference) / sum(n for _, n in reference)
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(
        torch.autograd.grad(got, table)[0], torch.autograd.grad(expected, table)[0]
    )
    with pytest.raises(ValueError, match="exact selected token count"):
        group_loss(groups, lambda _: (torch.tensor(0.0), torch.tensor(0)))


def test_grouped_loader_rejection(seam, tmp_path, monkeypatch):
    loader, _ = seam
    pin = tmp_path, write_synthetic_fixture(tmp_path)
    for args in [(None, None, 2), (None, object(), 1), (object(), None, 1)]:
        with pytest.raises(ValueError, match="rejects stock dataset"):
            loader(*args, grouped_synthetic=pin)
    with pytest.raises(ValueError, match="pin mismatch"):
        loader(None, None, 1, grouped_synthetic=(tmp_path, "d" * 64))
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="one CPU process"):
        loader(None, None, 1, grouped_synthetic=pin)
