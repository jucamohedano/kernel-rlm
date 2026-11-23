import pytest

from training.environments.kernel_rlm.research.ragged_gather_reduce.reference import (
    development_fixtures,
    public_fixture,
)
from training.environments.kernel_rlm.research.ragged_gather_reduce.score import (
    MAX_OBSERVATION_BYTES,
    OBSERVATION_HEADER,
    score_capture,
)


def capture(*arrays):
    return (
        OBSERVATION_HEADER
        + b"\n"
        + b"".join(" ".join(map(str, row)).encode("ascii") + b"\n" for row in arrays)
    )


def test_score_both_device_observations():
    fixture = public_fixture()
    sums, counts = fixture.expected_sums, fixture.expected_counts
    assert score_capture(
        fixture, capture(sums, counts, sums, counts), returncode=0
    ).matches_reference
    bad = score_capture(fixture, capture(sums, counts, sums, (3, 1, 2)), returncode=0)
    assert (bad.status, bad.detail) == ("mismatch", "capture 2, row 1: count")


def test_wrong_mask_computes_valid_but_wrong_numbers():
    fixture = next(f for f in development_fixtures() if f.name == "mask-offset")
    bad = score_capture(fixture, capture((7,), (1,), (7,), (1,)), returncode=0)
    assert (bad.status, bad.detail) == ("mismatch", "capture 1, row 0: sum")


def test_one_matching_initializer_cannot_hide_unwritten_output():
    # Match the first observation; on the second run an untouched slot differs.
    fixture = public_fixture()
    sums, counts = fixture.expected_sums, fixture.expected_counts
    assert not score_capture(
        fixture, capture(sums, counts, (53, 0x18D6, 5), counts), returncode=0
    ).matches_reference


def test_zero_rows_and_maximum_fixture():
    fixtures = {f.name: f for f in development_fixtures()}
    zero, maximum = fixtures["zero-rows"], fixtures["max"]
    assert score_capture(zero, capture((), (), (), ()), returncode=0).matches_reference
    sums, counts = maximum.expected_sums, maximum.expected_counts
    assert score_capture(
        maximum, capture(sums, counts, sums, counts), returncode=0
    ).matches_reference


@pytest.mark.parametrize(
    "output",
    [
        b"match\n",  # Process stdout is not a verdict.
        capture((1, 2), (1, 2, 3), (1, 2, 3), (1, 2, 3)),
        capture((-(2**31) - 1, 2, 3), (1, 2, 3), (1, 2, 3), (1, 2, 3)),
        capture(("-0", 2, 3), (1, 2, 3), (1, 2, 3), (1, 2, 3)),
        b"x" * (MAX_OBSERVATION_BYTES + 1),
    ],
)
def test_reject_incomplete_or_malformed_capture(output):
    assert score_capture(public_fixture(), output, returncode=0).status == "invalid_capture"


def test_nonzero_exit_never_counts_as_match():
    fixture = public_fixture()
    outputs = (fixture.expected_sums, fixture.expected_counts) * 2
    assert score_capture(fixture, capture(*outputs), returncode=3).status == "capture_error"
