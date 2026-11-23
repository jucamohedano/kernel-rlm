// Reviewed author fragment only: head128, one block per row/head.
__device__ __forceinline__ void fi_decode_body(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const int* I, const int* J, const int* T, float scale, int B, int P, int M,
    __nv_bfloat16* O, float* L) {
    const int d = threadIdx.x, h = blockIdx.x, b = blockIdx.y;
    const int lane = d % 32, warp = d / 32;
    const int row_head = b * 16 + h;
    const int out = row_head * 128 + d;
    const int begin = I[b], end = I[b + 1];
    const float neg_inf = -__int_as_float(0x7f800000);
    if (begin == end) {
        O[out] = __float2bfloat16_rn(0.0f);
        if (d == 0) L[row_head] = neg_inf;
        return;  // block-uniform, including M=0: no J/K/V access
    }
    const int kv = h / 8;
    const float q = __bfloat162float(Q[out]);
    const float log2e = 1.4426950408889634f;
    float m = neg_inf, mass = 0.0f, accum = 0.0f;
    __shared__ float dots[4];
    for (int j = begin; j < end; ++j) {
        const int page = J[j];
        const int stop = j + 1 == end ? T[b] : 64;
        for (int t = 0; t < stop; ++t) {
            const int offset = ((page * 64 + t) * 2 + kv) * 128 + d;
            float dot = q * __bfloat162float(K[offset]);
            for (int step = 16; step > 0; step /= 2)
                dot += __shfl_down_sync(0xffffffffu, dot, step);
            if (lane == 0) dots[warp] = dot;
            __syncthreads();
            dot = (dots[0] + dots[1]) + (dots[2] + dots[3]);
            __syncthreads();  // no warp can clobber next token's shared dots early

            const float logit = (dot * scale) * log2e;
            const float next_m = fmaxf(m, logit);
            const float old = mass == 0.0f ? 0.0f : exp2f(m - next_m);
            const float add = exp2f(logit - next_m);
            accum = accum * old + add * __bfloat162float(V[offset]);
            mass = mass * old + add;
            m = next_m;
        }
    }
    O[out] = __float2bfloat16_rn(accum / mass);
    if (d == 0) L[row_head] = m + log2f(mass);
}
