# Proposed native artifact v1: KernelBench L2/30 FP32

Interface proposal for a future isolated runner, **not** an accepted model input format in `qualify.py`. The pinned task and checked-in author controls do not depend on this schema being adopted by another runner.

- Payload: one separately staged `forward.cubin` (standalone `sm_120` ELF64 device module, not a host library, PTX, fatbin or archive) and one JSON plan. The runner resolves only exact C-linkage kernel symbols in that module. It pins the module bytes by SHA-256, checks physical target and resource limits, then uses CUDA Driver API; never calls a candidate host launcher. Use an approved, sandboxed compile path if device source is to be accepted later.
- Plan: `schema` is `kb30-sm120-fp32-launch/v1`. `scratch_bytes` is an integer between 0 and 268435456 inclusive. `launches` is a nonempty ordered list of at most 16 entries. Each entry names `symbol`, 3-element positive integer `grid` and `block` dimensions, `dynamic_shared_bytes` (nonnegative integer), and `args` (ordered). The runner validates dimensions, dynamic shared limits and symbol metadata before launching; no indirect or value-dependent launch plan.
- Argument descriptors use exactly one tag: `{ "ptr": "x" }`, `{ "ptr": "weight" }`, `{ "ptr": "bias" }`, `{ "ptr": "scale" }`, `{ "ptr": "offset" }`, `{ "ptr": "y" }`, `{ "ptr": "scratch" }`, `{ "i32": 0 }`, or `{ "u32": 0 }` (scalar values are bounded by their respective type). Buffers have their shape/layout from the task; all pointers are 64-bit CUDA device addresses. Scratch is one optional aligned byte buffer treated as undefined every forward; kernels may compute byte offsets within the declared extent. No pointer literals, host callbacks, arbitrary file paths, allocations, graph construction, additional streams, device-side kernel launches, library work (e.g. ATen/cuBLAS/Triton), or persistent mutable globals. Scalar constants are passed by value with exact 32-bit representation. Read-only inputs may not be modified even transiently.
- A fused launch or multiple raw-CUDA launches are allowed. Every launch is enqueued in order on the harness-owned stream. The timed span includes the whole list including any workspace initialization, staging, packing or synchronization needed by the operation. The harness supplies fresh/poisoned workspace/output for correctness and checks errors after enqueue and completion before evaluating the full output. Claim timing only for a tested artifact on the same workload/settings as the tuned baseline.

For example, the positive **author-control** launch plan (fixed in the shim; not parsed at runtime):

```json
{
  "schema": "kb30-sm120-fp32-launch/v1",
  "scratch_bytes": 33554432,
  "launches": [
    {
      "symbol": "kb30_linear", "grid": [512, 64, 1], "block": [16, 16, 1], "dynamic_shared_bytes": 0,
      "args": [{"ptr":"x"}, {"ptr":"weight"}, {"ptr":"bias"}, {"ptr":"scratch"}]
    },
    {
      "symbol": "kb30_epilogue", "grid": [16, 1024, 1], "block": [256, 1, 1], "dynamic_shared_bytes": 0,
      "args": [{"ptr":"scratch"}, {"ptr":"scale"}, {"ptr":"offset"}, {"ptr":"y"}, {"i32":0}]
    }
  ]
}
```

Validation and cubin-only loading are not a sandbox for untrusted device code. Any future candidate executor must separately enforce process/GPU isolation, wall/VRAM budgets, and artifact eligibility. Do not reuse the trusted shim for this purpose.
