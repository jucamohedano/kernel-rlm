"""Compare capture output with the CPU reference, without executing a candidate."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .reference import MAX_ROWS, WireFixture

OBSERVATION_HEADER = b"ragged-observation-v1"
# Four rows of int32 decimal tokens (at most 12 bytes per token and delimiter).
MAX_OBSERVATION_BYTES = 4 * MAX_ROWS * 12 + 128
INT32_TOKEN = re.compile(rb"(?:0|-?[1-9][0-9]{0,9})")


@dataclass(frozen=True)
class Score:
    status: str
    detail: str = ""

    @property
    def matches_reference(self) -> bool:
        """A matching capture is not proof of genuine execution or isolation."""
        return self.status == "match"


def decode_observation(capture: bytes, rows: int) -> tuple[tuple[int, ...], ...]:
    if len(capture) > MAX_OBSERVATION_BYTES:
        raise ValueError("oversize capture")
    lines = capture.split(b"\n")
    if len(lines) != 6 or lines[0] != OBSERVATION_HEADER or lines[-1] != b"":
        raise ValueError("invalid capture framing")
    arrays = []
    for line in lines[1:-1]:
        tokens = line.split(b" ") if line else []
        if len(tokens) != rows or any(not INT32_TOKEN.fullmatch(token) for token in tokens):
            raise ValueError("invalid output array")
        values = tuple(map(int, tokens))
        if any(value < -(2**31) or value >= 2**31 for value in values):
            raise ValueError("output outside int32")
        arrays.append(values)
    return tuple(arrays)


def score_capture(fixture: WireFixture, capture: bytes, *, returncode: int) -> Score:
    if returncode != 0:
        return Score("capture_error", f"capture exited {returncode}")
    try:
        observed = decode_observation(capture, fixture.rows)
    except ValueError as error:
        return Score("invalid_capture", str(error))
    for i, (got, expected) in enumerate(
        zip(observed, (fixture.expected_sums, fixture.expected_counts) * 2, strict=True)
    ):
        for row, (actual, target) in enumerate(zip(got, expected, strict=True)):
            if actual != target:
                field = "count" if i % 2 else "sum"
                return Score("mismatch", f"capture {1 + i // 2}, row {row}: {field}")
    return Score("match")
