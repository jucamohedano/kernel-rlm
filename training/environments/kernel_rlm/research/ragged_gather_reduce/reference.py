"""CPU reference and reproducible author-development inputs (no held-out cases)."""

from dataclasses import dataclass
from itertools import accumulate

MAX_ROWS = 4096
MAX_ENTRIES = 65536
OWNER_END = 262144


@dataclass(frozen=True)
class WireFixture:
    name: str
    value_base: int
    row_offsets: tuple[int, ...]
    gather_index: tuple[int, ...]
    keep_words: tuple[int, ...]
    values: tuple[int, ...]
    expected_sums: tuple[int, ...]
    expected_counts: tuple[int, ...]

    @property
    def rows(self) -> int:
        return len(self.row_offsets) - 1

    def to_wire(self) -> str:
        """Encode device inputs only; the comparator keeps expected outputs."""
        groups = (
            (self.rows, self.value_base, len(self.values), len(self.keep_words)),
            self.row_offsets,
            self.gather_index,
            self.keep_words,
            self.values,
        )
        return "".join(" ".join(map(str, group)) + "\n" for group in groups)


@dataclass(frozen=True)
class LogicalCase:
    """Each row lists owner-value IDs in entry order; None means dropped."""

    name: str
    entry_base: int
    value_base: int
    values: tuple[int, ...]
    rows: tuple[tuple[int | None, ...], ...]

    def __post_init__(self) -> None:
        entries = sum(map(len, self.rows))
        if len(self.rows) > MAX_ROWS or entries > MAX_ENTRIES:
            raise ValueError("too many rows or entries")
        if type(self.entry_base) is not int or not 0 <= self.entry_base <= OWNER_END - entries:
            raise ValueError("entry view outside owner bounds")
        if type(self.value_base) is not int or not 0 <= self.value_base <= OWNER_END - len(
            self.values
        ):
            raise ValueError("value view outside owner bounds")
        if any(type(v) is not int or not -32768 <= v <= 32767 for v in self.values):
            raise ValueError("samples must be signed int16")
        end = self.value_base + len(self.values)
        for row in self.rows:
            for owner in row:
                if owner is not None and (
                    type(owner) is not int or not self.value_base <= owner < end
                ):
                    raise ValueError("kept value ID outside the value view")

    def fixture(self, *, poison_drops: bool = True) -> WireFixture:
        if not poison_drops and not self.values:
            raise ValueError("in-range dropped IDs require a nonempty value view")
        # Oracle depends only on the logical rows, never on offsets or mask packing.
        sums = tuple(
            sum(self.values[i - self.value_base] for i in row if i is not None) for row in self.rows
        )
        counts = tuple(sum(i is not None for i in row) for row in self.rows)

        offsets = tuple(accumulate(map(len, self.rows), initial=self.entry_base))
        words = [0] * ((offsets[-1] + 31) // 32)
        gather = []
        for start, row in zip(offsets[:-1], self.rows, strict=True):
            for p, owner in enumerate(row, start):
                if owner is None:
                    poison = OWNER_END if p % 2 else -1
                    gather.append(poison if poison_drops else self.value_base)
                else:
                    words[p // 32] |= 1 << (p % 32)
                    gather.append(owner)
        return WireFixture(
            self.name,
            self.value_base,
            offsets,
            tuple(gather),
            tuple(words),
            self.values,
            sums,
            counts,
        )


def public_fixture() -> WireFixture:
    return LogicalCase(
        "public", 30, 101, (-7, 12, 30), ((None, 103, 103, 101), (), (102, None, 101))
    ).fixture()


def development_fixtures() -> tuple[WireFixture, ...]:
    """Author controls, not private scorer inputs. No random state or external data."""
    value_base = 223
    values = (-32768, 0, 19, 32767, -83, 4, 51)

    def mixed_row(r: int, length: int) -> tuple[int | None, ...]:
        return tuple(
            None if (j + r) % 5 == 0 else value_base + (3 * j + r) % len(values)
            for j in range(length)
        )

    short = tuple(mixed_row(r, r % 9) for r in range(257))
    skewed = tuple(mixed_row(r, 8193 if r == 95 else r % 4) for r in range(193))
    # One maximal row reaches exactly INT32_MIN. Both owner ranges end at the allowed bound.
    last_id = OWNER_END - 1
    maximal = ((last_id,) * MAX_ENTRIES,) + ((),) * (MAX_ROWS - 1)
    cases = (
        LogicalCase("zero-rows", 0, 0, (), ()),
        LogicalCase("all-dropped", 63, 77, (), ((None,) * 3, (), (None,))),
        LogicalCase("short", 11, value_base, values, short),
        LogicalCase("skewed", 97, value_base, values, skewed),
        LogicalCase("max", OWNER_END - MAX_ENTRIES, last_id, (-32768,), maximal),
    )
    # Keep every gather legal so a view-relative mask bug yields a wrong number,
    # not an invalid read. Correct owner bits 1,3 select ID 42 twice.
    mask_case = LogicalCase("mask-offset", 1, 41, (7, 17), ((42, None, 42),))
    return (mask_case.fixture(poison_drops=False), *(case.fixture() for case in cases))
