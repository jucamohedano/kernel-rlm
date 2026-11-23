# Fused gather–reduce for a ragged view

Implement `ragged_gather_reduce` from `contract.h` in one CUDA C++ translation
unit for NVIDIA L40S (`sm_89`). For each row, write the sum and count of its
kept gathered values. Empty rows and rows with no kept entries produce `(0, 0)`.
Repeated gather indices count as separate entries.

- `row_offsets` has `rows + 1` nondecreasing offsets in **owner entry** positions.
  Write `b = row_offsets[0]`, `end = row_offsets[rows]` and `n = end - b`.
  Row `r` has owner entry positions `p` in `[row_offsets[r], row_offsets[r+1])`.
- `keep_words` has `ceil(end / 32)` 32-bit words covering owner entry positions,
  with bit `p % 32` of word `p / 32` indicating whether to keep entry `p`.
- `gather_index` has `n` entries starting at the view, so the index for a kept
  entry `p` is `gather_index[p - b]`. This is an **owner value** coordinate,
  not an owner entry coordinate. The gathered value is
  `values[gather_index[p - b] - value_base]`. Do not dereference a value for a
  dropped entry; its gather index is unspecified.
- `rows` is in `[0, 4096]`, `0 <= b <= end <= 262144`, `n <= 65536`, and the
  value view lies within owner value positions `[0, 262144)`. The caller
  guarantees that every kept coordinate belongs to that view. Values are
  signed 16-bit integers; all sums and partial sums fit in signed 32 bits.
- All buffers are naturally aligned, non-null, disjoint device pointers,
  even for zero logical length. Outputs have `rows` entries each. Change only
  those entries. There is no invalid-input or aliasing case.
- Enqueue work on the supplied nonblocking `stream` and return `cudaSuccess`
  or an immediate launch error. Do not synchronize, allocate, copy to/from
  the host, use another stream, or call a library that implements the gather
  or reduction. CUDA runtime and standard C++/CUDA headers are available.
  `rows == 0` is a no-op. There is no required launch plan or throughput floor.

Example: `row_offsets = [30, 34, 34, 37]`, `value_base = 101`,
`values = [-7, 12, 30]`, `gather_index = [-1, 103, 103, 101, 102, 262144, 101]`,
and `keep_words = [2147483648, 23]`. The expected sums are `[53, 0, 5]` and
the counts are `[3, 0, 2]`.

The evaluator checks correctness first. If a same-device baseline is qualified,
latency covers all submitted GPU work for the operation. Aim for low latency on
both mostly short rows and skewed rows; judge correctness under the full
contract. Use the ABI unchanged.
