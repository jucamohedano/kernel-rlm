# Parent-only direct GQA author capture

The public handoff is `DEVICE_ABI.md` and `direct.descriptor.json`. Keep this
runbook, the author source, fixtures, tests, and receipts out of model context.
Only the fixed author body and three fixed numeric controls can reach NVCC or the
CUDA Driver. There is no candidate admission/load interface or timing stage.

## CPU result and next hardware step

From this worktree's root, in the existing project environment:

```sh
/home/juancm/projects/kernel-rlm/.venv/bin/python -B -m pytest -q \
  -p no:cacheprovider tests/test_flashinfer_direct_device_abi.py
sha256sum -c training/environments/kernel_rlm/research/flashinfer-paged-gqa-decode/direct-author.sha256
```

The tests validate contract and exact byte preservation, an independent FP64
oracle vs analytic math and a CPU specification of the FP32 online recurrence,
and the guarded capture using a fake Driver API. They do not execute CUDA,
compile NVCC source, or validate barrier behavior on hardware. The oracle is a
stdlib implementation and runs after driver teardown; neither it nor any
baseline implementation is linked into the author body. `analytic` covers a
singleton and two tokens with a nondefault scale, distinct KV heads, shared page
and padding that exposes a tail mistake. `irregular` covers 1/63/64/65/153 tokens,
reordered and repeated page IDs. `mixed-empty` adds an empty row; `all-empty`
exercises `M=0`. These are separate from archived FA2 and vector-only receipts.

First real hardware wire: establish the pinned NVCC V13.3.73 toolchain and
header closure, host compiler, loaded driver build, GPU UUID and exclusive
allocation using parent preflight. The author receipt records paths/version and
flags; bind it to the external toolchain/driver closure receipt. Inspect the
source and exact checksums before staging. From the trusted staged worktree
root, with only the assigned RTX PRO 6000 Blackwell visible as ordinal 0 and
only after explicit parent approval (replace variables by trusted assignments):

```sh
umask 077
env -i PATH=/usr/bin:/bin /usr/bin/timeout --kill-after=5s 75s \
  "$APPROVED_PYTHON" -B -m \
  training.environments.kernel_rlm.research.flashinfer-paged-gqa-decode.author_capture \
  --approved-author-smoke --nvcc "$APPROVED_NVCC" --ccbin "$APPROVED_CCBIN" \
  --device-uuid "$APPROVED_GPU_UUID" --case analytic --variant positive \
  > fi-direct-analytic-positive.json
```

One command builds one cubin and launches one forward. The unresolved hardware
check is that NVCC emits the expected sole `sm_120` text entry and the Driver
reports the exact parameter layout/count (including unused scalar bounds), then
full BF16 O and FP32 L match with intact guards and immutable inputs. Expect
`positive_match`, numerics/cleanup `pass`, `timing=not_run`. Read the first
receipt before running any more. Run `irregular`, `mixed-empty` and `all-empty`
with `--variant positive`, one approved fresh command/context per case. For
controls, repeat the same bounded command with case `analytic` and exactly one
of these variants, using a distinct output file each time:

- `--variant wrong-head` → O and L mismatch (interleaved heads).
- `--variant wrong-lse` → O passes; only L mismatches (natural-log LSE).
- `--variant wrong-tail` → O and L mismatch (padded tokens included; in bounds).

The controls are separately compiled and captured; `expected_numeric_rejection`
means intact execution/cleanup and the expected output-role mismatch only. An
expected rejection exits zero but numerics remains `fail`, never a qualifying
result. The receipt includes complete little-endian O/L bytes in hex. No
benchmark, training, timing comparison, stock FlashInfer-Bench result or
candidate execution is authorized by this check.

Stop on an unexpected stage, missing receipt, or timeout. Quarantine on timeout,
CUDA device/Driver error, integrity failure, or failed cleanup; no retry/reset.
A missing receipt does not establish device health. Parent should verify
recovery, including known vector author smoke, before further execution. Guards
are diagnostics, not a containment boundary. Stale-state checks in reused
allocations, repeat policy, long/timing fixtures and matched baseline comparison
remain separate work. Prior control receipts do not qualify this ABI.

## Later split/merge — not implemented or accepted by the direct parser

The future `split` descriptor has ordered geometry `["head128","head128"]` and
an exact integer `splits` in `{2,4,8,16}`. Let `A` be direct's first ten arguments.
Two ordered slots: `fi_partial(A,S,U)` grid `(16,B,S)` then
`fi_merge(U,B,S,O,L)` grid `(16,B,1)`, both block `(128,1,1)`. With `n` row pages,
split `j` consumes logical pages `[floor(j*n/S),floor((j+1)*n/S))` with the
actual final tail. FP32 `U[B,16,S,130]` stores max in base-2-logit units, shifted
mass, and 128 unnormalized shifted V sums. Empty partials overwrite all fields
with `(-inf,0,0...)`; merge ignores zero mass, and all empty yields exact O/L
sentinels. Scratch is `B*16*S*130*4` bytes (at most 2,129,920), fully overwritten
per call. No direct scratch. Do not silently broaden the current author CLI
or treat a direct result as a split result.
