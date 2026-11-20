"""Pinned native TRL parity and separate real/mocked Trackio checks."""

import json
import sqlite3
from importlib.util import find_spec

import pytest
import torch
from rlm_train.sft.bundle import load_bundle, write_synthetic_fixture
from rlm_train.sft.train_trl import (
    LR,
    METRICS,
    TRACKIO_PROJECT,
    _local_trackio,
    _make_trainer,
    _tiny_model,
    run_smoke,
)


@pytest.fixture
def fixture_bundle(tmp_path):
    directory = tmp_path / "bundle"
    digest = write_synthetic_fixture(directory)
    return directory, digest, load_bundle(directory, digest, allow_synthetic=True)


def test_native_dataloader_loss_and_sgd_parity(fixture_bundle, tmp_path):
    _, _, bundle = fixture_bundle
    trainer = _make_trainer(bundle, tmp_path / "trainer")
    expected_rows = {
        2: (
            [[1, 2, 3, 4], [5, 0, 6, 0]],
            [[1, 1, 1, 1], [1, 1, 1, 0]],
            [[-100, -100, 3, 4], [-100, -100, -100, -100]],
        ),
        1: ([[5, 1, 7]], [[1, 1, 1]], [[-100, -100, 7]]),
    }
    assert len(trainer.get_train_dataloader()) == 2
    seen = set()
    refs = {}
    expected_models = {}
    for batch in trainer.get_train_dataloader():
        count = int(batch["labels"].ne(-100).sum())
        seen.add(count)
        assert (
            tuple(batch[k].tolist() for k in ("input_ids", "attention_mask", "labels"))
            == expected_rows[count]
        )
        actual = trainer.compute_loss(trainer.model, batch)
        # Independent token loop: no shared masking helper or cross-entropy reduction.
        # Run each reference row independently, without padding, to catch cross-row leakage.
        reference = _tiny_model(bundle)
        terms = []
        for ids, attention, labels in zip(
            expected_rows[count][0], expected_rows[count][1], expected_rows[count][2], strict=True
        ):
            if all(label == -100 for label in labels):
                continue
            logits = reference(
                input_ids=torch.tensor([ids[: sum(attention)]]), use_cache=False
            ).logits[0]
            for pos, label in enumerate(labels):
                if label != -100:
                    terms.append(-logits[pos - 1].log_softmax(dim=-1)[label])
        loss = torch.stack(terms).mean()
        torch.testing.assert_close(actual, loss)
        refs[count] = float(loss.detach())
        loss.backward()
        with torch.no_grad():
            for p in reference.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-LR)
        expected_models[count] = reference
    assert seen == {1, 2}
    old = trainer.model.transformer.wte.weight.detach().clone()
    result = trainer.train()
    assert result.global_step == 1
    selected = trainer.step_metrics["target_tokens"]
    assert trainer.step_metrics["masked_nll"] == pytest.approx(refs[selected])
    reference = expected_models[selected]
    assert not torch.equal(old, trainer.model.transformer.wte.weight)
    for key, expected in reference.state_dict().items():
        torch.testing.assert_close(
            trainer.model.state_dict()[key], expected, rtol=1e-5, atol=2e-7, msg=key
        )


def test_fail_closed_before_loading_or_logging(fixture_bundle, tmp_path, monkeypatch):
    directory, digest, bundle = fixture_bundle
    with pytest.raises(ValueError, match="production blocked"):
        run_smoke(directory, digest, tmp_path / "no-opt-in")
    assert not (tmp_path / "no-opt-in").exists()
    with monkeypatch.context() as context:
        context.setenv("WORLD_SIZE", "2")
        with pytest.raises(ValueError, match="single local CPU"):
            _make_trainer(bundle, tmp_path / "distributed")
    with monkeypatch.context() as context:
        context.setenv("TRACKIO_SERVER_URL", "https://do-not-contact.invalid")
        with pytest.raises(ValueError, match="remote Trackio"):
            with _local_trackio(tmp_path):
                pytest.fail("invalid remote configuration was accepted")
    assert not (tmp_path / "trackio").exists()


def test_mock_numeric_logging(fixture_bundle, tmp_path, monkeypatch):
    from contextlib import nullcontext

    from rlm_train.sft import train_trl

    class Run:
        def __init__(self):
            self.events = []

        def log(self, metrics, step):
            self.events.append((step, metrics.copy()))

    stub = Run()
    monkeypatch.setattr(train_trl, "_local_trackio", lambda _: nullcontext(stub))
    directory, digest, _ = fixture_bundle
    result = run_smoke(directory, digest, tmp_path / "mock", allow_synthetic=True, track=True)
    assert stub.events == [(1, {name: result[name] for name in METRICS})]
    assert all(type(v) in (int, float) for v in stub.events[0][1].values())


@pytest.mark.skipif(find_spec("trackio") is None, reason="optional pinned Trackio isn't installed")
def test_real_trackio_sqlite(fixture_bundle, tmp_path):
    directory, digest, _ = fixture_bundle
    output_dir = tmp_path / "actual-trackio"
    result = run_smoke(directory, digest, output_dir, allow_synthetic=True, track=True)
    db = (output_dir / "trackio" / f"{TRACKIO_PROJECT}.db").resolve()
    with sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True) as connection:
        metrics = connection.execute("SELECT step, metrics, space_id FROM metrics").fetchall()
        assert len(metrics) == 1
        step, raw, space = metrics[0]
        assert step == 1 and space is None
        assert json.loads(raw) == {name: result[name] for name in METRICS}
        for table in ("system_metrics", "traces", "pending_uploads"):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)
