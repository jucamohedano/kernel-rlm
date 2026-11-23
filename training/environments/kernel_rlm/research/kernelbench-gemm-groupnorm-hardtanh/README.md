# KernelBench L2/30 — Blackwell author qualification

This opt-in development gate uses the **unmodified** public KernelBench model as its eager correctness reference. It is not the official KernelBench evaluator, a model-candidate runner, or a training task. The source is not vendored: stage `KernelBench/level2/30_Gemm_GroupNorm_Hardtanh.py` at commit [423217d9fda91e0c2d67e4a43bf62f96f6d104f1](https://github.com/ScalingIntelligence/KernelBench/blob/423217d9fda91e0c2d67e4a43bf62f96f6d104f1/KernelBench/level2/30_Gemm_GroupNorm_Hardtanh.py), SHA-256 `461335ed064d25b193afd8cee6645eb12b37540ce15baa5170f3d61b765d2a90`. `qualify.py` verifies bytes before importing any upstream code.

## Forward and proposed native artifact

One inference forward, contiguous row-major FP32 CUDA tensors, no gradients:

- `x[1024,8192]`, `weight[8192,8192]` indexed `[out,in]`, `bias[8192]`; `scale[8192]`, `offset[8192]`. All five inputs are read-only and disjoint. `y[1024,8192]` is a disjoint harness-allocated output; write every element for each call.
- `z = x @ weight.T + bias`, then 16 **contiguous** 512-channel groups independently for each row; mean and population variance (`/512`) over channels. `y = clamp(scale * (z - mean) / sqrt(var + 1e-5) + offset, -2, 2)` with affine per channel and clamp last.
- Float32 only in v1; IEEE FP32 math requested, TF32 disabled for PyTorch. No FP16/BF16 claims. Forward-only; no packing, transposition, or preprocessing of input/state excluded from a native timing span. No persistent derived state or reading prior output/scratch.

Proposed candidate interface (not loaded by this gate): one exact `sm_120` CUDA ELF cubin plus a declarative launch plan. See [native-artifact.md](native-artifact.md). The checked-in `controls.cu` and the fixed Driver API shim `native_controls.py` illustrate the mechanism, but only for trusted author use. No model-emitted host source, imports, shared object, arbitrary file, or launch plan can enter the present feedback path. A cubin is device code, **not** a security boundary; candidate execution must be separately authorized and isolated before any such loader is enabled.

## What the checks establish

- Full modes use the pinned constructor and input generator at state seed 42 and input seeds 108 and 810. Original eager is independently checked using FP64 dot and centered population variance, with FP32 casts at the Linear and GroupNorm boundaries. Compatibility cases are not synthetic replacements for the upstream shape or state. `cpu-smoke` alone shrinks the generated cases to `(M,K,N,G)=(8,32,64,4)`.
- Synthetic diagnostics use the same shape, seeds 17 and 31, constant and near-constant groups, distinct rows, nontrivial Linear bias and GroupNorm scale/offset, and both clamp bounds. A minimum of one-quarter of all expected outputs must have `abs(y) <= 1.75`. Interleaved groups, sample variance, and a skipped **current** producer with finite stale scratch from the other input must each disagree on interior values (both outputs at least 0.25 away from the clamp limits). A crash, NaN or saturation-only mismatch does not qualify a negative control. Native positive/repeat and three native negatives use this same state on GPU; native positive/repeat also runs on the first original input.
- The harness shares explicit parameter storage, records SHA-256 of contiguous FP32 input/state bytes, checks same shape/device/dtype/contiguity and finiteness, and compares **all** elements using `atol=rtol=1e-4` (`torch.isclose(reference, actual)`). It checks unchanged inputs/state after completion. Raw author output and new scratch on positives start as NaN, not a prior valid output. No hidden tests, datasets, model artifacts or VM logs are shipped here.

## Stage on the authorized VM

Stage this directory plus a separate pinned upstream file. Existing example environment (operator supplies paths and GPU selection):

```bash
base=/var/tmp/kernel-rlm-rtx-serving-f0b3b27d
stage=/var/tmp/kb30-stage   # contains this directory's files and staged upstream bytes
h="$base/ubuntu-headers/usr/include"
export CPATH="$h/python3.12:$h/x86_64-linux-gnu/python3.12:$h${CPATH:+:$CPATH}"
export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=0                # choose the authorized RTX device; no competing workload
bash "$stage/run_gpu.sh" "$base/.serve/bin/python" "$stage/kernelbench-L2-30.upstream.py" \
  "/var/tmp/kb30-full-$(date +%s)" gpu-full
```

Use `gpu-reference` for eager/equation/CPU-style finite checks on target with no NVCC or timing. `gpu-controls` adds native build/load/launch, native positives and negatives, still no timing. `gpu-full` also requires a separate tuned baseline: PyTorch Dynamo `fullgraph=True`, `dynamic=False` with Inductor **max-autotune-no-cudagraphs** tuning options. A counter records actual successful full-graph Inductor calls. There is no eager fallback for this baseline. Inductor may choose a GEMM library or its own implementation; the harness does not attribute its GEMM to cuBLAS or report a cuBLAS-only result. Native author controls are not a speed baseline; eager is the pinned reference comparator, not the tuned competitor. If the target compiler or tuned baseline fails, keep the failure receipt; do not substitute eager as a competitor.

GPU mode requires device name RTX PRO 6000 Blackwell and `get_device_properties()` capability `(12,0)`. The native command is `nvcc --cubin --generate-code=arch=compute_120,code=sm_120 --std=c++17 -O3 controls.cu -o ...`; compiler arg injections are refused. Requires CUDA ELF64, loads exact build output through CUDA Driver API in the PyTorch device context, and launches on the harness's non-default CUDA stream. No `compute_120` PTX, fatbinary/PTX JIT fallback, `sm_120a`, `sm_100`, or forward-compat substitution. This native artifact policy is distinct from how PyTorch/Inductor deliver their trusted internal kernels.

`receipt.json` is write-once in a fresh `/var/tmp` directory: schema `kernelbench-l2-30-author-qualification/v1`, with phase/status/error, source and harness hashes, fixture hashes, precision/device/runtime, separate NVCC build and module load, enqueue versus stream completion, and full-output comparisons. Only successful `gpu-full` can establish a tuned timing baseline. Its timing starts after validation again with restored upstream state/input and ten warmup calls per backend; defaults to 25 alternating CUDA-event samples/backend with four full forwards per sample on the same stream. Reports raw per-forward ms and median/min/max. Event windows include device work and any stream idle gaps between launches, not first-time compilation, uploads, validation or serialization. Clock/thermal stability and no competing GPU work are operator responsibilities. An absent/empty receipt is not a pass.

## Local CPU checks

From the worktree root, using the existing PyTorch CPU environment:

```bash
p=/home/juancm/miniconda3/bin/python
up=/home/juancm/.local/share/kernel-rlm/analyses/pre-sft-research-handoff-20261121T162754Z/sm120-reference-readiness-v1/kernelbench-L2-30.upstream.py
d=training/environments/kernel_rlm/research/kernelbench-gemm-groupnorm-hardtanh
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
KERNELBENCH_L2_30_UPSTREAM="$up" "$p" -m pytest -q -p no:cacheprovider tests/test_kernelbench_gemm_groupnorm_hardtanh.py
"$p" "$d/qualify.py" --mode cpu-full --upstream "$up" --receipt "/var/tmp/kb30-cpu-$(date +%s).json"
```

Local results: full exact-shape CPU eager/FP64 equation agreement, all three finite negatives rejected on interior outputs, and five focused tests passed (`torch 2.14.1+cpu`). CPU runs do not qualify NVCC, Driver API execution, GPU numerics, or performance; no GPU claims are encoded in these files.
