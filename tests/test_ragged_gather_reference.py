from itertools import pairwise

import pytest

from training.environments.kernel_rlm.research.ragged_gather_reduce.reference import (
    LogicalCase,
    development_fixtures,
    public_fixture,
)


def test_public_wire_matches_hand_example():
    # Entries 31,32,33 keep values 30,30,-7. Row 1 is empty. Entries 34,36
    # keep 12,-7. Dropped entries 30 and 35 have IDs outside the value view.
    assert public_fixture().to_wire() == (
        "3 101 3 2\n30 34 34 37\n-1 103 103 101 102 262144 101\n2147483648 23\n-7 12 30\n"
    )
    assert public_fixture().expected_sums == (53, 0, 5)
    assert public_fixture().expected_counts == (3, 0, 2)


def test_development_profiles_and_limits():
    fixtures = {f.name: f for f in development_fixtures()}
    assert fixtures["zero-rows"].to_wire().split() == ["0", "0", "0", "0", "0"]
    assert fixtures["all-dropped"].expected_counts == (0, 0, 0)
    lengths = [b - a for a, b in pairwise(fixtures["short"].row_offsets)]
    assert min(lengths) == 0 and 0 < max(lengths) < 32
    lengths = [b - a for a, b in pairwise(fixtures["skewed"].row_offsets)]
    assert max(lengths) > sum(lengths) // 2 and sum(n < 32 for n in lengths) > len(lengths) // 2
    max_case = fixtures["max"]
    assert (max_case.rows, len(max_case.gather_index), max_case.row_offsets[-1]) == (
        4096,
        65536,
        262144,
    )
    assert max_case.value_base + len(max_case.values) == 262144
    assert max_case.expected_sums == (-(2**31),) + (0,) * 4095
    assert max_case.expected_counts == (65536,) + (0,) * 4095


def test_view_relative_mask_yields_wrong_numbers_without_invalid_reads():
    case = next(f for f in development_fixtures() if f.name == "mask-offset")
    assert case.row_offsets == (1, 4)
    assert all(case.value_base <= i < case.value_base + len(case.values) for i in case.gather_index)
    wrong = [
        case.values[i - case.value_base]
        for relative, i in enumerate(case.gather_index)
        if case.keep_words[relative // 32] >> (relative % 32) & 1
    ]
    assert (case.expected_sums, case.expected_counts) == ((34,), (2,))
    assert (sum(wrong), len(wrong)) == (7, 1)


def test_development_wire_agrees_with_logical_oracle():
    for case, repeat in zip(development_fixtures(), development_fixtures(), strict=True):
        wire = case.to_wire()
        assert wire == repeat.to_wire()
        header, offsets, ids, words, values = (
            list(map(int, line.split())) for line in wire.splitlines()
        )
        sums, counts = case.expected_sums, case.expected_counts
        rows, base, nvalues, nwords = header
        assert len(offsets) == rows + 1 and len(ids) == offsets[-1] - offsets[0]
        assert len(words) == nwords == (offsets[-1] + 31) // 32 and len(values) == nvalues
        assert len(sums) == len(counts) == rows
        for r, (start, end) in enumerate(pairwise(offsets)):
            kept = [
                ids[p - offsets[0]] - base
                for p in range(start, end)
                if words[p // 32] >> (p % 32) & 1
            ]
            assert all(0 <= j < nvalues for j in kept)
            assert (sum(values[j] for j in kept), len(kept)) == (sums[r], counts[r])


@pytest.mark.parametrize(
    ("entry_base", "value_base", "values", "rows"),
    [
        (262144, 0, (), ((None,),)),  # Nonempty entry view past the bound
        (0, 262144, (0,), ()),  # Nonempty value view past the bound
        (0, 0, (32768,), ()),
        (0, 1, (3,), ((0,),)),  # Negative Python indices must not wrap
        (0, 0, (), ((),) * 4097),
        (0, 0, (), ((None,) * 65537,)),
    ],
    ids=["entry-end", "value-end", "sample-range", "kept-id", "too-many-rows", "too-many-entries"],
)
def test_reject_invalid_logical_cases(entry_base, value_base, values, rows):
    with pytest.raises(ValueError):
        LogicalCase("invalid", entry_base, value_base, values, rows)
