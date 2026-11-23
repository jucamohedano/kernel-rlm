import hashlib
import importlib
import os
from pathlib import Path

import pytest

TASK = (
    Path(__file__).resolve().parents[1]
    / "training/environments/kernel_rlm/research/kernelbench-gemm-groupnorm-hardtanh"
)


@pytest.fixture
def kb30(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(TASK))
    return importlib.import_module("qualify")


def test_pinned_loader_refuses_untrusted_host_code(kb30, tmp_path):
    marker = tmp_path / "executed"
    source = tmp_path / "spoof.py"
    source.write_text(f"from pathlib import Path; Path({str(marker)!r}).touch()")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        kb30.load_pinned(source)
    assert not marker.exists()


def test_independent_equation_has_per_row_population_stats_affine_and_clamp(kb30):
    torch = kb30.torch
    x = torch.tensor([[1.0, -1.0], [2.0, 1.0]])
    w = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    z = kb30.equation_linear(x, w, torch.zeros(4))
    gamma = torch.tensor([3.0, 1.0, 1.0, 1.0])
    beta = torch.tensor([0.0, 0.25, 0.375, -0.25])
    actual = kb30.equation_finish(z, gamma, beta, groups=2)
    known = torch.tensor([[2.0, -0.749995, 0.375, -0.25], [2.0, -0.749980, 0.375, -0.25]])
    torch.testing.assert_close(actual, known, atol=1e-6, rtol=0)
    rejected = kb30.checked(kb30.equation_finish(z, gamma, beta, 2, "sample_variance"), known)
    assert rejected["interior_mismatched"] == 2


def test_shape_dtype_nonfinite_and_saturation_do_not_pass_negatives(kb30):
    torch = kb30.torch
    ref = torch.tensor([2.0, -2.0, 0.25, 1.0])
    for bad in (ref.reshape(2, 2), ref.double(), ref * float("nan")):
        with pytest.raises(RuntimeError, match="expected finite failure on interior"):
            kb30.require_negative_rejected(kb30.checked(bad, ref), "bad")
    saturated_only = kb30.checked(torch.tensor([-2.0, 2.0, 0.25, 1.0]), ref)
    assert saturated_only["mismatched"] == 2 and saturated_only["interior_mismatched"] == 0
    with pytest.raises(RuntimeError, match="expected finite failure on interior"):
        kb30.require_negative_rejected(saturated_only, "wrong_at_bounds_only")


def test_exact_mutation_check_and_fp32_fingerprint(kb30):
    torch = kb30.torch
    x = torch.zeros(4)
    expected = hashlib.sha256(bytes(16)).hexdigest()
    assert kb30.fp32_sha256(x) == expected
    state = torch.ones(3)

    def mutate(_):
        state[1] = 1.001
        return x.clone()

    with pytest.raises(RuntimeError, match="mutated input or explicit state"):
        kb30.validate_call(mutate, x, (state,), x.clone(), None, {}, "mutator")


def test_reduced_pinned_original_and_three_finite_negatives(kb30, monkeypatch):
    upstream = os.getenv("KERNELBENCH_L2_30_UPSTREAM")
    if not upstream:
        pytest.skip("set KERNELBENCH_L2_30_UPSTREAM to the pinned bytes")

    def forbidden():
        raise AssertionError("CPU smoke must not probe CUDA")

    monkeypatch.setattr(kb30.torch.cuda, "is_available", forbidden)
    args = kb30.argparse.Namespace(mode="cpu-smoke", cpu_threads=4, upstream=Path(upstream))
    receipt = {}
    kb30.qualify(args, receipt)
    assert receipt["source_verified"]
    assert receipt["dimensions_m_k_n_g"] == kb30.SMOKE
    assert receipt["state_fp32_sha256"]["upstream"] != receipt["state_fp32_sha256"]["diagnostic"]
    for case in receipt["cases"].values():
        assert case["eager"]["repeat"]["comparison"]["passed"]
    diagnostic = receipt["cases"]["diagnostic_17"]
    assert diagnostic["input_fp32_sha256"] != diagnostic["stale_input_fp32_sha256"]
    negatives = diagnostic["equation_negatives"]
    assert set(negatives) == {
        "interleaved_groups",
        "sample_variance",
        "skipped_current_producer_stale_buffer",
    }
    assert all(
        not result["passed"] and result["interior_mismatched"] for result in negatives.values()
    )
