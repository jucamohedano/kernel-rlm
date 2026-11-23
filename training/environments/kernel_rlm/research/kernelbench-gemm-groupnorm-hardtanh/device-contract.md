# KernelBench L2/30 — `kb30_f32_device_v1`

This device-body ABI specifies one full FP32 Linear → GroupNorm → Hardtanh
forward. It is not the upstream Python/host calling convention. No submitted
code is admitted for execution by the descriptor/parser implementation.

## Two separate byte payloads

- `submission.cu`: at most 131,072 bytes, nonempty UTF-8, no embedded NUL or
  Markdown fences. Preserve the source exactly; do not trim or reformat it.
- `descriptor.json`: at most 4,096 bytes, UTF-8 JSON. Exactly `abi`, `plan` and
  `geometry` are required; duplicate, missing or extra fields are errors. There
  are no defaults. Preserve the request exactly and hash a canonical typed
  descriptor separately. Geometry is a fixed-length ordered list of presets.

```json
{"abi":"kb30_f32_device_v1","plan":"staged","geometry":["tile16x16","group256"]}
```

## Tensor and numerical contract

All pointers address disjoint, contiguous, row-major IEEE FP32 arrays aligned
at least 256 bytes. Input snapshots are immutable and already cast. No shapes,
strides, pointers, scalars, or sizes are submitted in the descriptor.

- `X[1024,8192]`, `W[8192,8192]` with `W[out,in]`, `bias[8192]`.
- `gamma[8192]`, `beta[8192]`, `Y[1024,8192]`.
- Linear reduces over 8192 input channels and then adds the per-output bias.
  In the staged plan its entire FP32 result is `Z[1024,8192]`: exactly
  33,554,432 bytes, overwritten on every call. The final stage reads it.
- For each row, normalize **independently** over each of 16 contiguous groups
  of 512 output channels. Use population variance with `1e-5` added before
  reciprocal square root. Apply `gamma` and `beta` per channel, then clamp
  to `[-2,2]`. Every output position must be written and finite, and satisfy
  `abs(ref-got) <= 1e-4 + 1e-4 * abs(got)`.

## Fixed launch slots

The fragment supplies the matching device-body functions. The trusted side
supplies the externally visible CUDA entry wrappers and packs the ordered
pointer arguments. No dynamic shared memory. A staged forward always launches
the producer before the consumer on the same stream, using no external or
precomputed `Z`.

1. `__device__ __forceinline__ void kb_linear_body(const float* X, const float* W, const float* bias, float* Z)`
   - `tile16x16`: grid `(512,64,1)`, block `(16,16,1)`.
   - `tile32x8`: grid `(256,128,1)`, block `(32,8,1)`.
2. `__device__ __forceinline__ void kb_finish_body(const float* Z, const float* gamma, const float* beta, float* Y)`
   - `group128`: grid `(16,1024,1)`, block `(128,1,1)`.
   - `group256`: grid `(16,1024,1)`, block `(256,1,1)`.

`{"abi":"kb30_f32_device_v1","plan":"fused","geometry":["group256"]}` selects a
reserved, **unsupported** future single-slot plan. It would call
`kb_forward_body(const float* X, const float* W, const float* bias, const float* gamma, const float* beta, float* Y)`
with either group preset and no `Z`. It must not fall back to the staged plan.

The source contract is device functions and local device helpers, not global
entry definitions, host code, includes, external dependencies, persistent state,
device allocation, cooperative launch or device-side launch. No cubins, paths,
compiler flags or target overrides are submitted. Native target is base
`sm_120`, not PTX, fatbin, JIT, or a fallback. Bounded static shared memory is
permitted (at most 48 KiB per CTA); total explicit device allocations including
scratch/alignment/guards are capped at 384 MiB. These limits do not establish
runtime isolation or qualify a submission.
