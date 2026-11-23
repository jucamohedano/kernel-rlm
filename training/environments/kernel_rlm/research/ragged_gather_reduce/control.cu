// Operator-side author control, not part of the solver context. Target: sm_89.
#include "contract.h"
#include <cuda_runtime.h>

namespace {
constexpr int WARP = 32;
constexpr int THREADS = 128;
constexpr int WARPS = THREADS / WARP;

template <bool block_per_row>
__global__ void reduce_rows(int rows, const int* offsets, const int* ids,
                            const uint32_t* mask, int value_base, const int16_t* values,
                            int* sums, int* counts) {
    const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const int row = block_per_row ? int(blockIdx.x) : int(blockIdx.x) * WARPS + warp;
    if (row >= rows) return;  // uniform across each participating warp

    const int b = offsets[0], begin = offsets[row], end = offsets[row + 1];
    int sum = 0, count = 0;
    for (int p = begin + (block_per_row ? int(threadIdx.x) : lane); p < end;
         p += block_per_row ? THREADS : WARP) {
#ifdef RAGGED_NEGATIVE_CONTROL
        const int bit = p - b;  // intentional bug: mask belongs to owner, not view
#else
        const int bit = p;
#endif
        if ((mask[bit / 32] >> (bit % 32)) & 1u) {
            sum += values[ids[p - b] - value_base];
            ++count;
        }
    }
    sum = __reduce_add_sync(0xffffffffu, sum);
    count = __reduce_add_sync(0xffffffffu, count);
    if (block_per_row) {
        __shared__ int warp_sums[WARPS], warp_counts[WARPS];
        if (lane == 0) {
            warp_sums[warp] = sum;
            warp_counts[warp] = count;
        }
        __syncthreads();
        if (warp != 0) return;
        sum = __reduce_add_sync(0xffffffffu, lane < WARPS ? warp_sums[lane] : 0);
        count = __reduce_add_sync(0xffffffffu, lane < WARPS ? warp_counts[lane] : 0);
    }
    if (lane == 0) {
        sums[row] = sum;
        counts[row] = count;
    }
}
}  // namespace

extern "C" cudaError_t ragged_gather_reduce(
    int32_t rows, const int32_t* row_offsets, const int32_t* gather_index,
    const uint32_t* keep_words, int32_t value_base, const int16_t* values,
    int32_t* row_sums, int32_t* row_counts, cudaStream_t stream) {
    if (rows == 0) return cudaSuccess;
    // Heuristic, not a measured crossover. Few rows can afford more threads per row.
    if (rows <= 256) {
        reduce_rows<true><<<rows, THREADS, 0, stream>>>(
            rows, row_offsets, gather_index, keep_words, value_base, values, row_sums, row_counts);
    } else {
        reduce_rows<false><<<(rows + WARPS - 1) / WARPS, THREADS, 0, stream>>>(
            rows, row_offsets, gather_index, keep_words, value_base, values, row_sums, row_counts);
    }
    return cudaGetLastError();
}
