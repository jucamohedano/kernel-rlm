#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

extern "C" cudaError_t ragged_gather_reduce(
    int32_t rows, const int32_t* row_offsets, const int32_t* gather_index,
    const uint32_t* keep_words, int32_t value_base, const int16_t* values,
    int32_t* row_sums, int32_t* row_counts, cudaStream_t stream);
