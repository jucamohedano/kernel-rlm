# Modal smoke test

## One frozen-model episode

From `training/environments/kernel_rlm`:

```bash
modal run -m modal_app.model_rollout
modal run -m modal_app.model_rollout --gpu
```

The first command checks cached checkpoint shards and the chat template on CPU.
The second additionally runs one real Laguna-XS.2 episode using vLLM 0.26.0,
the original pinned checkpoint, and the local RLM training environment. It is
not a prime-rl training run. No optimizer or LoRA update runs here.

The CPU coordinator owns the REPL and records full requests and responses. A
single inference A100 serves at most three calls, each capped at 2,048 output
tokens. The inference function has a 600-second timeout and is cancelled before
the one final verification call. Verification uses a separate A100 with a
180-second timeout, blocked network access, and no model/trace volumes mounted.
The extension-call guard is a heuristic; it does not prove resistance to cheating.

Traces are committed incrementally to
`kernel-rlm-traces:model-rollout/<run-id>/trace.json` and copied locally to
`logs/model-rollout/<run-id>/trace.json`. A failed episode is recorded as failed;
it is never silently retried. Inspect `status`, not just the CLI exit code.
The requested compute envelope is estimated below $1, including startup
allowances, at the 2026-10-07 published rates. Image builds, actual resource use,
and storage also affect billing; timeouts are not a financial account limit.

### First real-model result: 2026-10-07

Run `ap-1TwEMLvOssKAdhq6FfZRnW` loaded the frozen checkpoint and generated
three replies (1,105 output tokens). Two REPL blocks executed, but no final
answer was submitted: reward 0, `max_turns_reached`, zero verifier calls,
zero training updates. The third generated block was not processed before
the turn limit. The model also mistook the CPU REPL's CUDA-runtime warning
for a kernel problem. Fix final-turn handling and clarify CPU preparation
versus GPU verification before paying for another episode.

Full trace: `model-rollout/20261007T095005Z-f97e72d76181/trace.json` in the trace
volume. Coordinator elapsed time was 400.4 seconds; inference initialization
inside the worker took 229.3 seconds. Actual billing is unverified. The app
stopped with zero active tasks. Tokenizer-regex and RoPE warnings from vLLM
also need review before drawing conclusions about model quality.

## Prime-rl checkpoint conversion gate

```bash
modal run -m modal_app.conversion_smoke
```

This CPU-only test uses the two unmodified conversion modules from pinned
prime-rl v0.9.0 and verifies their SHA-256 hashes. It imports them directly to
avoid importing the full trainer and GPU kernels. No converter logic is replaced.

The cached Laguna checkpoint index and safetensors headers drive an all-tensor
key/shape check on meta tensors. Each layer's real BF16 weights is then loaded
on CPU, converted, and compared byte for byte against independently specified
destination names and expert indices. Reverse conversion must recover every
original key and tensor byte. Embeddings, norms, and the output head are also
included. Layer-by-layer processing avoids holding two complete model copies.

No GPU, network model download, or converted-checkpoint write is performed.
The function has a 600-second timeout, four CPU cores, 24 GiB RAM, and no
automatic retries. Reports commit after each group to
`kernel-rlm-traces/prime-conversion/<run-id>/report.json`; successful reports
also download to `logs/prime-conversion/<run-id>/report.json`.

Scope: this validates the actual converter on the checkpoint, not full native
model loading, logit equivalence, native LoRA, vLLM, or trainer integration.
Those need separate tests; a conversion pass must not revalidate the earlier
partly initialized Transformers/PEFT run.

Validated on 2026-10-05: `ap-4u8ZTmk4S7PrQSBjC7TQiC`, report ID
`20261005T160336Z-8936311c740a`. All 30,513 original tensors became 678 prime-layout
tensors with exact forward-mapping and round-trip value preservation. All 40
layers and the three non-layer tensors passed in 125.3 seconds on CPU.

## Native prime-rl Laguna + LoRA gate

```bash
modal run -m modal_app.native_lora_smoke
```

This CPU/meta-only check imports the pinned prime-rl v0.9.0 `LagunaForCausalLM`
and calls its native `apply_lora_to_model`, targeting only `q_proj`, `k_proj`,
`v_proj`, and `o_proj`. It verifies all 160 projections across 40 layers,
8,192,000 rank-8 adapter parameters, and zero trainable base parameters. The
LoRA state is redirected to meta tensors because prime-rl's current helper
hardcodes a CUDA device; this is an explicit test-only override, not a production
behavior change. No checkpoint weights or GPU are allocated.

Validated on 2026-10-05: Modal run `ap-8mNS9Lihz4iDNu4g71Pzjt`, report ID
`20261005T163316Z-ad6d34b6b936`, status `passed`. This establishes that the
native model and native adapter target selection are compatible. The single-GPU
load/update gate is documented below; this is not yet full RL training.

## Native prime-rl single-GPU update

```bash
modal run -m modal_app.native_gpu_smoke
```

The successful run `ap-MB8BYkIHFxyj8zIIoBDsMJ` (`20261005T190110Z-2cbbb7b487b1`)
loaded all 30,513 BF16 checkpoint tensors after conversion to 678 native tensors;
all 678 native shapes matched and no keys were missing or unexpected. The native
model occupied 66,885,336,576 allocated GPU bytes. Rank-8 attention LoRA added
8,192,000 trainable parameters, produced finite loss (`5.25`), nonzero gradients
for all 160 adapter tensors, and changed all 160 after one AdamW update.

Two compatibility details are explicit in the report. The checkpoint's boolean
`gating=true` corresponds to per-head gate weights, so the smoke harness sets
`config.gating="per-head"`; prime-rl v0.9.0 otherwise constructs 40 incompatible
per-element gate tensors. Also, the image's FlashAttention symbol was unavailable,
so the test used a Torch SDPA GQA fallback. The native Laguna body, native head,
checkpoint load, and LoRA gradients were exercised, but FlashAttention kernel
performance was not measured. The top-level `LagunaForCausalLM` call also passes
`temperature` to plain `nn.Linear`; the test computes equivalent cross-entropy
from the native body and head and records that workaround.

## Attention-only LoRA feasibility

**Current status: this direct PEFT loader is blocked by checkpoint-format mismatch.**
The separate prime-rl conversion gate now passes; native model/LoRA integration
has not yet been tested. The first GPU run
(`ap-uAuPWZbGrJrnDVjXZ9JtR4`) completed an update, but its pretrained loading was
invalid: individual expert checkpoint tensors were not mapped into the fused
expert tensors expected by the remote implementation. Missing tensors were
initialized instead. The original `gpu.json` says `passed` because the initial
test omitted loading validation; its sibling `validation.json` explicitly
invalidates that claim. Both are retained under run ID
`20261005T153205Z-0e4cf1d1fe22` locally and on the traces Volume.

The CPU preflight now compares checkpoint index keys with meta-model state keys
and fails closed before weight download/GPU allocation. The GPU loader also
rejects missing, unexpected, or mismatched keys. This deliberately requires a
verified conversion before proceeding. Prime-rl's pinned Laguna conversion chain
handles per-expert/fused and singular/plural shared-expert layouts; validate that
path before another GPU test rather than trusting a successful optimizer step.

```bash
modal run -m modal_app.attention_lora
modal run -m modal_app.attention_lora --gpu
```

The default uses a CPU meta-model to verify all 160 q/k/v/o projections and
8,192,000 trainable adapter parameters (rank 8, alpha 16). No model weights or
GPU are required for that check. The GPU option downloads the pinned original
Laguna-XS.2 weights on CPU before allocating one A100-80GB, then attempts one
AdamW update on a batch of 32 synthetic tokens with gradient checkpointing.
It checks finite loss/gradients, absent frozen-base gradients, and changed
adapter tensors, and records peak allocated/reserved GPU memory.

This uses Transformers 5.6.2, PEFT 0.19.0, and PyTorch 2.11.0 (CUDA 12.8).
It does **not** use the prime-rl trainer or vLLM and is not a full RL test.
The pinned prime-rl v0.9.0 reference is commit
`ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1`; its full RL deployment defaults
to one trainer GPU plus one inference GPU. Success here does not establish
useful-context memory capacity, rollout throughput, or adapter transfer.

The model/code revision is `a397bde04501591d0ac3d74a5d9d0a4ec298ffb8`.
The original BF16 weights occupy roughly 67 GB in `kernel-rlm-model-cache`;
keep this reusable cache distinct from discarded training checkpoints.
Reports are committed to `kernel-rlm-traces/attention-lora/<run-id>/` and
successful reports copied into local `logs/attention-lora/<run-id>/`.
Reports include synthetic input IDs, settings, target names, status, and errors.
The test intentionally saves no trained checkpoint from meaningless random data.

Limits: 180 seconds for CPU preflight, 1200 seconds for CPU download, and
600 seconds for the single GPU function, with no automatic function retries.
At the inspected $2.50/GPU-hour rate, the GPU function's time cap is about
$0.42, excluding CPU/RAM, startup/shutdown, image building, and storage.
Timeouts are not an account-wide dollar cap. Hard termination may leave only
the last committed stage report; ordinary exceptions save their traceback.
Direct dependency pins are explicit; the transitive dependency tree is not frozen.

## CUDA verifier

Run the known-good vector-add submission on an A100-80GB GPU:

```bash
cd /home/juancm/projects/laguna-rlm/rlm/training/environments/kernel_rlm
modal run -m modal_app.kernel_smoke
```

The image pins CUDA 12.4.1 and PyTorch 2.5.1. The test compiles the submission
in the verifier's fresh process, checks correctness, and records the benchmark
speedup. It does not run RLM training.

## Scripted rollout integration test

Run the real training-environment lifecycle on Modal without loading a model:

```bash
modal run -m modal_app.rollout_smoke
modal run -m modal_app.rollout_smoke --gpu
```

The default runs a CPU-only preflight with a fake verifier. `--gpu` first repeats
that preflight and, only if it passes, runs one A100-80GB rollout with the real
CUDA verifier. Neither command runs experiments on the laptop.

`rollout_smoke.py` owns the cloud image, resource limits, and persistent trace saving.
`scripted_rollout.py` owns the test protocol:

1. A loopback HTTP server supplies two prepared model replies, with no external
   inference requests. Reported token usage is a placeholder, not a measurement.
2. `RLMTrainEnv` sends the system/task prompts and executes the first reply through
   `SubprocessReplBackend`: save and print the reference code from `context`.
3. The second request must contain the REPL output. Its reply checks that the
   saved variable survived, then submits the known-good CUDA source.
4. The environment stops, cleans up its worker, and the rubric verifies the source
   exactly once. The gated-reward metric reuses that score instead of benchmarking
   again. Reward is `0.1 + 0.2*compiled + 0.4*correct + 0.1*clip(speedup, 0, 3)`
   for this nonempty, successful submission.
5. The cloud result becomes plain JSON before crossing back to the laptop, so
   local training-framework dependencies are not needed to read the trace.

Traces are committed to the dedicated Modal Volume `kernel-rlm-traces` before
the remote function returns, under `<UTC timestamp>-<unique id>/<stage>.json`.
Successful results are also downloaded to `logs/kernel-rollout/<run-id>/<stage>.json`
(git-ignored). Repeated runs do not overwrite earlier traces. Each trace contains
the actual model requests, scripted replies, final output, metrics, and pass/fail
status; exceptions during rollout or validation preserve the partial trace and error.
Startup failures and hard process termination can still prevent a trace from being
written. Failed traces can be downloaded from the volume rather than the CLI return.

```bash
modal volume ls kernel-rlm-traces
modal volume get kernel-rlm-traces <run-id>/preflight.json ./preflight.json
```

The original passing traces were backfilled under
`ap-2kALiMvhQ3mGA5ap2YJXuA/{preflight,cuda}.json` without rerunning the GPU job.
This volume holds only small traces, not model weights or checkpoints. Keep it
separate from the historical `laguna-runs` volume so checkpoint cleanup cannot
delete the new rollouts.

Assertions cover two model requests, two REPL calls, the exact final source,
worker cleanup, one verifier invocation, correctness, and reward consistency.

Budget controls: no automatic rollout/function retries, at most one container per
function, a 120-second CPU function timeout, a 360-second GPU function timeout,
and a two-second GPU scale-down window. Image builds and CPU work also have costs;
these timeouts are not an account-wide dollar cap. Direct dependencies use versions
from `training/uv.lock`; transitive dependencies are not fully frozen by this image.

This is an integration check, not evidence of model ability, training compatibility,
or a reliable kernel speedup. The tiny task's timing is noisy. The next checkpoint
is to inspect this trace before attempting a bounded real-model rollout; LoRA/GRPO
training remains a later, separately budgeted step.

### Validated checkpoint: 2026-10-05

Modal run `ap-2kALiMvhQ3mGA5ap2YJXuA` passed both stages. The CUDA stage recorded
two REPL calls, one verifier invocation, successful compilation, exact correctness
(`max_diff=0`), speedup `1.0`, and reward `0.8`. The app completed and stopped.
This does not establish an optimization gain. Billing had not yet included this
completed run when first checked, so its final cost was not confirmed.
