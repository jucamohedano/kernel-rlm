# Paged GQA decode — direct device ABI v1

The submission consists of two independent byte strings: `submission.cu` and
`descriptor.json`. Send raw UTF-8 bytes, not Markdown fences. The CUDA fragment
must be nonempty, at most 128 KiB and NUL-free. The descriptor is strict UTF-8
JSON at most 4 KiB, with no duplicate, extra, or missing keys. Its only allowed
value is:

```json
{"abi":"fi_gqa_h16_kv2_d128_p64_device_v1","geometry":["head128"],"plan":"direct"}
```

JSON key order and whitespace do not affect descriptor identity. The receiver
retains both exact byte strings and can hash the canonical descriptor separately.
Packet validation is not source admission or execution permission.

## Fragment signature and launch

Define the following device body. The trusted translation unit provides
`cuda_bf16.h`, `math.h`, the matching declaration, and the sole exported CUDA
entry `fi_decode` that calls the body. Do not provide an entry or a host wrapper.

```cuda
__device__ __forceinline__ void fi_decode_body(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const int* I, const int* J, const int* T, float scale, int B, int P, int M,
    __nv_bfloat16* O, float* L);
```

One launch, grid `(16,B,1)`, block `(128,1,1)`, no dynamic shared memory or
scratch buffer. `blockIdx.x` is the query head; `blockIdx.y` is the batch row.
Static shared memory (≤48 KiB per block), local device helpers and CUDA device
math are allowed. No includes, host code, external linkage/dependencies, module
globals/persistent state, device-side launch or allocation. There are no caller
path, symbol, pointer, offset, flag or launch-size overrides. The target is
base `sm_120`. This is neither FlashInfer's host wrapper ABI nor a ragged ABI.

## Buffers and math

All buffers are contiguous and disjoint; first six arguments are read-only.
`Q[B,16,128]`, `K[P,64,2,128]`, `V[P,64,2,128]`, and `O[B,16,128]` are BF16.
K and V are separate, `[physical page, token slot, KV head, channel]` (NHD).
`I[B+1]`, `J[M]`, `T[B]` are signed int32. `L[B,16]` is FP32. `scale` is a finite
positive FP32 argument, followed by `B`, `P`, `M` as int32 arguments by value.

- Domain: `1≤B≤16`, `1≤P≤512`, `0≤M≤1024`, at most 128 pages per row. Fixture
  values are finite and in a safe FP32 accumulation range. The host validates
  byte lengths and page metadata before any launch.
- `I[0]=0`, `I[B]=M`, and `I` is nondecreasing. Row `b` consumes logical page
  indices `J[I[b]:I[b+1]]` in order. Each physical page ID is in `[0,P)`. Repeated,
  shared, and reordered IDs are legal. Full pages have 64 tokens; consume only
  the first `T[b]` tokens of the final page (`1≤T[b]≤64` for nonempty rows).
  Empty rows have `T[b]=0`. If `M=0`, the `J` address is a guarded dummy; do not
  dereference it.
- For head `h` use KV head `h//8`. For each logical token form the FP32-scaled
  logit `x = scale * dot(Q[b,h,:], K[page,slot,h//8,:])`. Set every channel of
  `O[b,h,:]` to the softmax-weighted reduction of V (FP32 accumulation, BF16
  result). Set `L[b,h] = log2(sum(exp(x)))`, a **base-2** LSE of the logits.
- Empty rows write exact BF16 `+0` for all of `O` and FP32 `-inf` for all of `L`.
  Otherwise both outputs must be finite. Every element is checked independently:
  reject if both absolute error and `abs_error/(abs(reference)+1e-8)` exceed
  `0.01`. Overwrite both entire outputs on each invocation.
