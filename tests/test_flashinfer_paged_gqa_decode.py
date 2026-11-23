"""CPU-only qualification adapter tests; definition bytes are supplied out of tree."""

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "training/environments/kernel_rlm/research/flashinfer-paged-gqa-decode/qualify.py"
)
spec = importlib.util.spec_from_file_location("flashinfer_paged_decode_qualify", SCRIPT)
assert spec is not None and spec.loader is not None
qualify = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = qualify
spec.loader.exec_module(qualify)


@pytest.fixture(autouse=True)
def single_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_closed_form_and_controls():
    case, expected = qualify.analytic_probe()
    got = qualify.oracle(case)
    torch.testing.assert_close(got[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(got[1], expected[1], atol=1e-6, rtol=0)
    controls = qualify.controls(case, got, expected)
    assert controls["interleaved_kv_head"]["output"]["mismatches"] == 8
    assert controls["interleaved_kv_head"]["lse"]["mismatches"] == 8
    assert controls["natural_log_lse"]["output"]["mismatches"] == 0
    assert controls["natural_log_lse"]["lse"]["mismatches"] == 16


def test_page_boundaries_and_empty_row_sentinels():
    ragged = qualify.irregular()
    assert ragged.token_counts() == [1, 63, 64, 65, 153]
    # A single valid token means O is exactly its V for every head and LSE is its logit / ln2.
    o, lse = qualify.oracle(ragged)
    torch.testing.assert_close(o[0], ragged.v[5, 0].repeat_interleave(8, dim=0), atol=0, rtol=0)
    logit = (ragged.q[0, 8].double() * ragged.k[5, 0, 1].double()).sum()
    assert lse[0, 8] == pytest.approx(float(logit) * ragged.scale / math.log(2), abs=1e-6)
    empty = qualify.mixed_empty(ragged)
    want = qualify.oracle(empty)
    torch.testing.assert_close(want[0][1:], o[1:])
    torch.testing.assert_close(want[1][1:], lse[1:])
    assert qualify.compare(want, want, empty.token_counts())["empty_rows"] == 1
    for row, value, message in [(0, 0, "empty sentinel"), (1, -math.inf, "nonfinite on nonempty")]:
        bad_lse = want[1].clone()
        bad_lse[row, 0] = value
        with pytest.raises(ValueError, match=message):
            qualify.compare((want[0], bad_lse), want, empty.token_counts())
    bad_o = want[0].clone()
    bad_o[0, 0, 0] = 1
    with pytest.raises(ValueError, match="output bad empty sentinel"):
        qualify.compare((bad_o, want[1]), want, empty.token_counts())


def test_timing_tables_share_qkv_but_change_page_ids_and_lengths():
    a, b = qualify.timing_cases()
    assert a.q is b.q and a.k is b.k and a.v is b.v
    assert a.indptr is b.indptr
    assert torch.count_nonzero(a.indices == b.indices) == 0
    assert all(x != y for x, y in zip(a.token_counts(), b.token_counts(), strict=True))
    assert b.indices.dtype == b.tail.dtype == torch.int32
    assert (
        qualify.compare(qualify.oracle(a), qualify.oracle(b), a.token_counts())["output"][
            "mismatches"
        ]
        > 0
    )


def test_cli_opt_in_and_definition_is_pinned(tmp_path):
    definition = tmp_path / "bad.json"
    definition.write_text(json.dumps({"reference": "raise AssertionError('must not run')"}))
    staging = tmp_path / "stage"
    args = [
        sys.executable,
        "-B",
        str(SCRIPT),
        "--definition",
        str(definition),
        "--stage-dir",
        str(staging),
        "--mode",
        "cpu-reference",
    ]
    run = subprocess.run(args, capture_output=True, text=True, check=False)
    assert run.returncode != 0 and "--author-only" in run.stderr
    run = subprocess.run([*args, "--author-only"], capture_output=True, text=True, check=False)
    assert run.returncode != 0 and "SHA-256 mismatch" in run.stderr
    assert not staging.exists()
