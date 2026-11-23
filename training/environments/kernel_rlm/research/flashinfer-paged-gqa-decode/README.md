# Paged GQA decode: author qualification

This is a local author control, not an official FlashInfer-Bench port, a model task/trace, or a benchmark result. The CPU gate needs only PyTorch. The native gate probes the **already installed** FlashInfer runtime separately. Eager PyTorch is a correctness reference only; it is not competitive performance evidence. Neither mode runs a submitted candidate.

## Source and contract

Supply the exact 4,872-byte `flashinfer-scratch/gqa_paged_decode_h16_kv2_d128_ps64.json` from the `sm120-reference-readiness-v1` handoff. Its [definition](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace/blob/da915083d4c7c5e61aa3005e3d17ae488e0fc71c/definitions/gqa_paged/gqa_paged_decode_h16_kv2_d128_ps64.json) is pinned to `da915083d4c7c5e61aa3005e3d17ae488e0fc71c`, SHA-256 `6f05748fd5fa488f593009adbe86e4f260fb176ea69a073951b50b79b894c899`. The CLI requires `--author-only`; it rejects other bytes before executing anything from the definition. It stages the exact JSON and the untouched embedded `run` as standalone `pinned_reference.py` in a **new** scratch directory. The definition is supplied out of tree, with no bundled dataset or workload. No permission for redistribution or training ingest is inferred from the pin.

Tensor/scale arguments, in order:

```
q                 BF16  [B,16,128]
k_cache, v_cache  BF16  [P,64,2,128] each, separate NHD pages
kv_indptr         int32 [B+1]
kv_indices        int32 [M]
kv_last_page_len  int32 [B]
sm_scale          host float32 scalar (explicit; default value 1/sqrt(128))
=> output         BF16  [B,16,128]
=> lse            FP32  [B,16], log-sum-exp in base 2
```

For a subsequent native candidate, use dense contiguous buffers (Q/O strides `2048,128,1`; K/V `16384,256,128,1`; LSE `16,1`, all in elements). Freeze definition order and return order; a destination-passing binding appends `output,lse` to the inputs, then exposes `B,P,M`, any declared reusable workspace and a caller stream as adapter parameters. All buffers are device-resident except the host scalar; inputs are read-only, outputs caller-owned, and buffers nonoverlapping. `B>=1`, `kv_indptr[0]=0`, offsets are nondecreasing, `kv_indptr[-1]=M`, and referenced physical pages are in `[0,P)`. On a nonempty row the tail is in `[1,64]`, including `64` for a full last page. For an empty row the harness passes tail `0` (unused). First scope excludes malformed inputs, strides other than contiguous, nonfinite input/logits, and nonpositive scale. B, P and M are runtime values; do not specialize on fixture contents.

Each query head `h` uses KV head `h//8`. Read logical pages in listed order; a repeated/shared physical page is read as many times as listed. For all nonfinal pages use all 64 tokens; for the final page use only the given tail. Apply no extra mask or positional transform. Natural logits are `scale*dot(q,k)` (FP32 reference accumulation), O is the softmax-weighted V sum rounded to BF16, and LSE is `logsumexp(logits)/ln(2)`. Empty rows require exact zero O and negative-infinity LSE. **Choose your own CUDA implementation strategy.** The FlashInfer wrapper is an external comparator, not a required implementation or candidate dependency. No plan details, scheduling strategy or baseline library entrypoints are part of the candidate contract.

## Quick CPU smoke

From repository root, use an existing CPU PyTorch interpreter (`/home/juancm/miniconda3/bin/python` on the development host) and the local pinned definition (path below in the handoff layout). Outputs stay in `/var/tmp`.

```bash
set -o pipefail
bundle=training/environments/kernel_rlm/research/flashinfer-paged-gqa-decode
DEFINITION="$HOME/.local/share/kernel-rlm/analyses/pre-sft-research-handoff-20261121T162754Z/sm120-reference-readiness-v1/flashinfer-scratch/gqa_paged_decode_h16_kv2_d128_ps64.json"
RUN=$(mktemp -d /var/tmp/paged-gqa-cpu.XXXXXXXX)
PYTHON_BIN=/home/juancm/miniconda3/bin/python
CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" -B "$bundle/qualify.py" --author-only \
  --definition "$DEFINITION" --stage-dir "$RUN/reference" --mode cpu-reference \
  | tee "$RUN/cpu-reference.jsonl"
CUDA_VISIBLE_DEVICES= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON_BIN" -B -m pytest \
  -p no:cacheprovider -q tests/test_flashinfer_paged_gqa_decode.py
```

The gate checks both O and LSE: a two-token, two-head-group closed-form probe uses a nondefault scale, then rejects an intentionally interleaved KV-head mapping and natural-log LSE. An independent FP64 token-address oracle uses stabilized `exp2` for the irregular fixture (`T=1,63,64,65,153`, shuffled/shared/repeated page IDs, poisoned unused slots). The separate mixed-empty fixture checks exact sentinels before comparing remaining finite rows. Per output, every finite element must pass `abs_error <= 0.01` **or** `abs_error/(abs(want)+1e-8) <= 0.01`; receipts give separate maxima/counts. This is the finite tolerance of FlashInfer-Bench `15618a67e8d7b7f49ca61b5ba3b9f26a876d0bee`, not a run of its evaluator. **Do not use the stock reject-any-Inf check for valid empties.**

## Existing SM120 stack: operator plan

On the approved VM, stage this directory plus the exact definition JSON under `/var/tmp`. Set `PYTHON_BIN` to the preflight-tested `.serve/bin/python`, `CPATH` to the ordered preflight-verified Ubuntu 3.12 Python include paths (Python.h, multiarch and the include root), and `DEFINITION` to the staged JSON. The CPATH is needed even for eager `torch.bmm` on this stack when PyTorch uses Triton. Use the approved RTX PRO 6000 without concurrent workloads; expose it as CUDA device 0. The launcher records driver and nvcc, and the script checks compute capability `(12,0)`, records GPU UUID/PyTorch as available, disables TF32 via the launcher, and uses a single explicit PyTorch stream. If the operator cannot permit network access, run under the existing network restrictions; the script does not install or download dependencies, but it does not sandbox third-party runtime internals. Preserve preprovisioned cache settings only with known provenance.

```bash
set -o pipefail
# Preflight supplies PYTHON_BIN, CPATH, DEFINITION and the staged bundle path.
export PYTHON_BIN CPATH
export QUAL_SCRATCH=$(mktemp -d /var/tmp/paged-gqa-sm120.XXXXXXXX)
bash "$bundle/run-sm120.sh" sm120-reference \
  --definition "$DEFINITION" --stage-dir "$QUAL_SCRATCH/reference" \
  2>&1 | tee "$QUAL_SCRATCH/reference.log"

# Cheap check for an existing runtime; no import/JIT/install here.
"$PYTHON_BIN" -B -c 'from importlib.metadata import version; print(version("flashinfer-python"))'

# Only if present and approved. Use separate processes and receipt files.
bash "$bundle/run-sm120.sh" sm120-native --backend fa2-tensor \
  --definition "$DEFINITION" --stage-dir "$QUAL_SCRATCH/fa2-tensor" \
  2>&1 | tee "$QUAL_SCRATCH/fa2-tensor.log"
bash "$bundle/run-sm120.sh" sm120-native --backend cuda-core \
  --definition "$DEFINITION" --stage-dir "$QUAL_SCRATCH/cuda-core" \
  2>&1 | tee "$QUAL_SCRATCH/cuda-core.log"
```

Expect `reference_pass` for the eager-only control. For the native gate, distribution metadata is not enough: the script imports `flashinfer.decode`, logs its installed version/path/source digest, requests the CUDA-core `use_tensor_cores=False` path or the tensor-core `use_tensor_cores=True, backend="fa2"` path, then reports plan/JIT-or-load completion, API launch return, stream completion, numerical checks and timings separately. First plan wall time can include JIT/build or module loading; it is not proof of a fresh build without compiler receipts. There is no `auto` backend or eager timing fallback. Native outputs are NaN-filled before correctness calls to catch unwritten destinations, and both supplied destinations must be returned. The installed distribution can be compared with the proposed `flashinfer-python==0.6.4` source pin `f1e6fdcb8f65104047697f022b5d055ef022d763`; a passing receipt qualifies the **observed** installation and explicit dispatch, not that pin by assumption.

Accept a native comparator only with terminal `native_baseline_pass` and its timing scope. The native gate covers the nonempty probe, irregular fixture and two tables at `B=16,P=512,M=496` (roughly 0.5k–4k tokens per row). It does not claim native empty-row support or general serving performance. Pin the winning passing backend per compared timing scope; don't combine a fixed-table numerator with a changing-table denominator. If neither explicit backend passes (including missing dependencies), report the exact blocker and keep the reference-only result. Do not recommend dependency changes before checking the installed stack; any pin change, new native dependency, or independently written native comparator needs its own approval and same-device qualification.

## Timing and resources

- **Fixed table:** warmed CUDA events around complete `run` on one stream, with plan reuse. This is stream elapsed time, not individual kernel time or synchronized host-wall API latency. Include all native run device work, auxiliary launches and both writes. No CUDA graph, no cache flush, no per-call correctness fills. Keep reuse permissions equal for future candidates.
- **Changing table:** alternate two pre-resident page ID arrays and last-page lengths for the same Q/K/V and page counts, in synchronized host-wall `plan+run`. Include plan/metadata processing, internal copies, launches, synchronization and both writes; exclude producer/input creation and H2D input staging. This does not exercise changing capacity or Q/K/V refresh. No offline expansion/memoization permitted. Use the same ready inputs and scopes for any later comparison.
- Default `--warmup 10 --repeats 50 --workspace-mib 128`. Logs contain raw samples and min/median/p90/max. One-time build/JIT, inputs, workspace and output allocation are outside both timing intervals. Bytes include the explicit caller workspace/outputs, input storage and additional table metadata; the PyTorch allocator view is supplementary and does **not** account for pinned host planning buffers, all external GPU allocations, compiler/cache disk use or total process memory. No training or inference resource claim follows. Retain stdout/stderr and run exit codes under scratch, not Git.
