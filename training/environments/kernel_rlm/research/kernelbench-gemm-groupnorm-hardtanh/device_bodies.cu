// Trusted body template adapted from L2/30 controls.cu SHA-256
// 3e53d3484eb996260a88ed6b35feb4499ca81329ae05659d949807653e0d3446.
namespace {
constexpr int K = 8192, N = 8192, Groups = 16, Width = N / Groups;
constexpr int Tile = 16, Threads = 256;
}

// --- LINEAR / FINISH BOUNDARY ---
// grid=(N/Tile,M/Tile,1), block=(Tile,Tile,1), dynamic shared bytes=0.
__device__ __forceinline__ void kb_linear_body(const float* x, const float* w,
                                       const float* b, float* z) {
    __shared__ float xs[Tile][Tile + 1], ws[Tile][Tile + 1];
    const int i = threadIdx.y, j = threadIdx.x;
    const int row = blockIdx.y * Tile + i, col = blockIdx.x * Tile + j;
    float sum = 0.0f;
    for (int k = 0; k < K; k += Tile) {
        xs[i][j] = x[row * K + k + j];
        ws[i][j] = w[(blockIdx.x * Tile + i) * K + k + j];
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < Tile; ++t) sum = fmaf(xs[i][t], ws[j][t], sum);
        __syncthreads();
    }
    z[row * N + col] = sum + b[col];
}

// --- LINEAR / FINISH BOUNDARY ---
// grid=(Groups,M,1), block=(Threads,1,1), dynamic shared bytes=0.
// Wrong grouping uses interleaved channels. Wrong variance divides by Width-1.
template <bool Interleaved, bool SampleVariance>
__device__ __forceinline__ void finish_group(const float* z, const float* scale,
                                         const float* offset, float* y) {
    __shared__ float a[Threads];
    __shared__ float mean;
    int t = threadIdx.x, row = blockIdx.y, group = blockIdx.x;
    int c0 = Interleaved ? t * Groups + group : group * Width + t;
    int c1 = Interleaved ? c0 + Threads * Groups : c0 + Threads;
    float p = z[row * N + c0], q = z[row * N + c1];
    a[t] = p + q;
    __syncthreads();
    for (int s = Threads / 2; s > 0; s >>= 1) {
        if (t < s) a[t] += a[t+s];
        __syncthreads();
    }
    if (t == 0) mean = a[0] * (1.0f / Width);
    __syncthreads();
    p -= mean;
    q -= mean;
    a[t] = p * p + q * q;
    __syncthreads();
    for (int s = Threads / 2; s > 0; s >>= 1) {
        if (t < s) a[t] += a[t+s];
        __syncthreads();
    }
    const float inv = rsqrtf(a[0] / (SampleVariance ? (Width - 1) : Width) + 1e-5f);
    float yp = (p * inv) * scale[c0] + offset[c0];
    float yq = (q * inv) * scale[c1] + offset[c1];
    // Preserve NaN poison (fmin/fmax would otherwise turn some poison into a bound).
    y[row * N + c0] = yp < -2.f ? -2.f : yp > 2.f ? 2.f : yp;
    y[row * N + c1] = yq < -2.f ? -2.f : yq > 2.f ? 2.f : yq;
}

__device__ __forceinline__ void kb_finish_body(const float* z, const float* gamma,
                                               const float* beta, float* y) {
    finish_group</*VARIANT*/>(z, gamma, beta, y);
}
