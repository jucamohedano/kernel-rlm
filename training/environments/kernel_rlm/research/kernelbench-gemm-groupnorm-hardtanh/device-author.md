# Private KB30 author qualification handoff

Keep this page, fixture implementation and CUDA bodies separate from the
submission-facing `device-contract.md`. The staged port is based on the
previously qualified `kb-qual-final/controls.cu` (`3e53d3484eb996260a88ed6b35feb4499ca81329ae05659d949807653e0d3446`).
The previous author qualification used a different ABI. The port is not a
candidate and results cannot be used for training, scoring or speedup claims.

## Parent-only first hardware check

Only after explicit parent approval and idle/exclusive RTX preflight: verify the
GPU UUID, driver, compiler/header closure, trusted checkout and file pins. Use
the existing qualified PyTorch interpreter, NVCC V13.3.73 and its trusted host
C++ compiler. The CLI uses no model file, plug-in, external cubin or host C++
source and requires `/usr/bin/prlimit`. Run in a clean environment with no
secret variables or extra descriptors. From the worktree root:

```sh
KB=training/environments/kernel_rlm/research/kernelbench-gemm-groupnorm-hardtanh
sha256sum -c "$KB/device-author.sha256"
env -i PATH=/usr/bin:/bin /usr/bin/timeout --kill-after=5s 180s \
  /absolute/qualified-pytorch-python -E -s -B "$KB/device_author.py" \
  --approved-author-smoke --nvcc /absolute/nvcc --ccbin /absolute/g++ \
  --device-uuid GPU-... --fixture upstream --control positive
```

Inspect the receipt before continuing. A successful first run requires both
seed 108 and changed seed 810 to pass with the same allocated ranges, fresh
scratch/output poison, and both ordered slots executed. Next use
`--fixture diagnostic --control positive`. For separate process invocations
use the same command with `--fixture diagnostic` and each of the following:

- `--control interleaved_groups`: both launches complete, guards/input pass,
  the finite output fails numerics on interior positions.
- `--control sample_variance`: same expected independent numerical rejection.
- `--control missing_producer`: no-op producer leaves fresh scratch poisoned;
  epilogue propagates it, and the independent check rejects the NaN output.
- `--control stale_producer`: first full positive populates scratch; changed X
  (17→31), freshly poisoned Y and a separate no-op producer body leave the
  byte-identical prior Z. Second finite output must fail interior numerics.

Expected negative behavior returns 0 only if all required non-numerical stages
pass and its `numerics` stage records `fail`. Archived upstream constructor
seed 42 and input/state hashes are checked before opening libcuda. The oracle
is a trusted local functional transcription of FP32 eager Linear, GroupNorm
and Hardtanh on CPU, not an execution of the pinned model source or its prior
GPU eager timing baseline. Reference bytes do not enter the Driver adapter.
A hash mismatch, unavailable dependency or CPU oracle failure is preflight
`unavailable`, not a result. An unexpected runtime stage, timeout, lost or
quarantined device, missing receipt or cleanup failure requires a stop and
parent recovery check (known author vec-add); no automatic retry/reset.

The process loads immutable exact bytes after native ELF checks and verifies
function parameter layouts and resources. It keeps launches on one private
nondefault stream. Staged scratch is exactly 32 MiB; total explicit guarded GPU
allocation is 369,200,640 bytes. Prepare/load/validation are not timing lanes.
The toolkit include-*tree* digest binds its staged CUDA headers; this is not a
complete hash of the host system headers or OS, which remain parent preflight
inputs. This patch provides no candidate compiler/runner or runtime isolation
claim. No GPU/VM access occurred during authorship.
