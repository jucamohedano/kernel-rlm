# kernel_rlm

CUDA kernel tasks for the local RLM training loop.

The environment keeps Python as the persistent REPL language. A submission is Python source
that defines CPP_SOURCE, CUDA_SOURCE, and cuda_forward(*tensors). The submission compiles
the CUDA implementation through torch.utils.cpp_extension and calls the resulting extension.

The verifier imports the submission, compiles the extension, checks correctness on fresh CUDA
inputs, and benchmarks only after correctness passes. Verification runs in a separate process.

The tiny vector-add task is in kernel_rlm/tasks/vector_add.jsonl. A known-good submission is
in examples/vector_add_submission.py.

The package is intentionally compatible with the existing rlm_train.RLMTrainEnv. The generic
RLM loop and the historical triton_rlm environment remain separate.

## Muon verification

From this directory, run `modal run -m modal_app.muon_wiring` for a bounded CPU-only
check of prime-rl v0.9.0's real optimizer constructor against Laguna's meta model.
It checks all 320 attention LoRA tensors (8,192,000 parameters), frozen-parameter
exclusion, AdamW fallback, and Moonlight RMS scaling for every adapter shape.
The report is saved in `kernel-rlm-traces:muon-wiring/<run-id>/report.json` and
downloaded to `logs/muon-wiring/<run-id>/report.json`.

This does not execute an optimizer step, load checkpoint weights, or validate
GRPO. It uses the pinned prime source checkout at `/tmp/prime-rl-v090` inherited
from the native-model smoke image. Learning rate and decay in the check are test
values, not tuned RL hyperparameters. A GPU update check is still required.

## Monitoring the first GRPO run

### Current choice: Trackio, local viewer

Prime-hosted logging is paused. Trackio is pinned to 0.40.0. The private live
Hugging Face Space attempt returned 402: this account requires PRO for Gradio/Docker
cpu-basic hosting. No subscription was purchased. Trackio's static Space export
requires public data, so we do not use it for this private project.

Run `modal run -m modal_app.trackio_smoke` for a CPU-only persistence test. It logs
only `preflight/connectivity`, with `training=False`, under
`kernel-rlm-traces:trackio-local`. The report is saved separately under
`trackio-smoke/<run-id>/report.json`. No real training curves exist yet.
The optional `--hosted` path attempts private Space setup and requires `HF_TOKEN`
in the caller environment; the module currently passes that token via an ephemeral
Modal Secret, not a command-line argument or tracked file.

To download a snapshot and view it locally, from this directory:

```bash
mkdir -p logs/trackio-viewer
modal volume get kernel-rlm-traces trackio-local/laguna-kernel-rlm.db logs/trackio-viewer/laguna-kernel-rlm.db
TRACKIO_DIR="$PWD/logs/trackio-viewer" uv run --no-project --python 3.12 --with trackio==0.40.0 python -c 'import trackio; trackio.show(project="laguna-kernel-rlm", host="127.0.0.1", server_port=7860, share=False, open_browser=False, block_thread=True)'
```

Open http://127.0.0.1:7860. This is only a CPU metrics viewer, not a local model
experiment. Stop the viewer before replacing an existing snapshot; use
`modal volume get --force` for an intentional refresh. The snapshot does not update
automatically. Preserve Modal logs as the source of truth. Integration with prime's
trainer/orchestrator metrics is still pending; the W&B backend is not assumed to be
drop-in compatible with Trackio.

### Hosted Prime access check

Store `PRIME_API_KEY` in the Modal Secret `prime-intellect-secret` and
`PRIME_TEAM_ID` in `prime-intellect-team-id-secret`. Both are attached to the check. Run
`modal run -m modal_app.prime_monitoring` from this directory before any paid
training launch. It uses CPU only and follows the external-run endpoints used by
our pinned prime monitor. If authorized, it creates a clearly labeled connectivity
record, logs one preflight metric (not a training loss/reward), and closes the record.
It saves a sanitized result to `kernel-rlm-traces:prime-monitoring/<run-id>/report.json`
and downloads a copy to `logs/prime-monitoring/<run-id>/report.json`.

The 2026-10-07 check found the API key in the Secret, but external-run registration
returned HTTP 401, `Invalid token`. There was no surrounding whitespace and no
`PRIME_TEAM_ID`. No platform run was created and no GPU was allocated. Correcting
authentication is necessary before testing team authorization; the missing team
field alone is not established as the cause of the 401. Sample upload is not covered
by this connectivity check and will need validation with the first saved episode.

### Curves and traces

The subsequent two-secret check loaded both fields and authenticated successfully
(200), but registration returned 403, `You are not a member of this team`.
Check that the supplied team ID belongs to the account owning the API key before
investigating external-run allowlisting. No platform run was created.

Latest retry (2026-10-07) supersedes the initial authentication diagnosis:
Bearer-only `GET /api/v1/user/whoami` succeeded (200); external-run registration
returned 403, `External training runs are only enabled for allowlisted teams`.
The secret still has no team ID. The key is valid; an intended, enabled team must
be selected before hosted logging can work. The probe now checks identity before
registration and does not expose identity response data. No platform run was created.

The training launcher is not ready yet: this package still returns the legacy
`RLMTrainEnv`, while pinned prime-rl consumes Verifiers V1 tasksets and harnesses.
The previous model rollout also failed to submit a final answer. Resolve these
gates before launching paid GRPO, rather than treating optimizer construction as
an end-to-end training success.

Prime's file monitor is enabled by default. Mount the eventual run output directory
on a Modal Volume and commit it periodically and on shutdown. Preserve the resolved
config, `metrics.jsonl`, logs, and all/effective per-step rollout traces, including
failures and zero-advantage groups. Keep these files even when W&B is enabled.

For live W&B charts, the eventual RL configuration uses:

```toml
[monitors.wandb]
project = "laguna-kernel-rlm"

[monitors.file]
path = "metrics.jsonl"
```

These are configuration fragments, not a runnable training config. Supply
`WANDB_API_KEY` through a Modal Secret only to trusted training/orchestration
processes, never to generated-code workers. W&B is optional and currently not
configured in this project. Prime logs the trainer and orchestrator to one shared
run; its shared mode does not support offline W&B logging.

Without W&B, copy the saved run directory locally and use prime's read-only
`uv run dashboard /path/to/downloaded/run-output` to inspect curves and traces.
This serves a CPU-only viewer (normally port 7788), not a local model experiment.
A downloaded snapshot is not live unless updated. The project HTML tracker is
the curated learning report, not a live telemetry endpoint.

Watch these panels:

- Outcomes: reward and CUDA correctness separately, compilation and submission
  rates, speedup only for correct kernels, with sample counts.
- Learning signal: group reward spread, zero-advantage fraction, and all versus
  effective rollouts. An all-equal group supplies no relative-reward gradient.
- Stability: policy loss, entropy, gradient norm, learning rate, clipping, and
  available KL diagnostics. Policy loss need not decrease monotonically; trainer/
  inference mismatch KL is not automatically KL to the original base model.
- Budget: wall time, generated tokens, throughput, peak GPU memory, and estimated
  cost labeled separately from Modal's billed usage.

Prime supplies standard training metrics; CUDA outcome metrics and group diagnostics
must be checked in our eventual V1 integration. GRPO curves do not exist yet:
optimizer checks and rollout-only runs are not GRPO training runs.

### Actual metrics in Trackio

Trackio 0.40.0 is the selected local viewer; Prime hosted logging is paused.
`modal_app.native_gpu_smoke` saves measured `metrics.jsonl` alongside its report.
The 2026-10-07 Muon check passed one update on a 32-random-token cross-entropy
fixture: loss 4.0625, pre-clipping gradient norm 17.625, 160 adapter tensors
changed, and 67.56 GB peak reserved GPU memory. This validates optimizer
execution, not kernel quality or GRPO learning.

From this directory, export a completed snapshot on Modal CPU:

```bash
modal run -m modal_app.trackio_export \
  --relative-path native-gpu/20261007T114150Z-bb860d098b55/metrics.jsonl \
  --run-name muon-update-20261007T114150Z \
  --scope "One real Muon update on random-token CE fixture; not GRPO"
modal volume get --force kernel-rlm-traces \
  trackio-local/laguna-kernel-rlm.db logs/trackio-viewer/laguna-kernel-rlm.db
```

Stop the local viewer before replacing its database. Restart with `TRACKIO_DIR`
set to the absolute `logs/trackio-viewer` directory and
`trackio.show(project="laguna-kernel-rlm", host="127.0.0.1",
server_port=7860, share=False)` in the pinned Trackio environment.

The importer separates trainer/orchestrator producers into different runs. Its
x-axis is log-row sequence; `source/training_step` retains the original step,
preserving multiple records at the same optimizer step. This is a snapshot
bridge, not live logging or an integrated GRPO launcher. Export each completed
snapshot once with a distinct name. `modal_app.trackio_bridge_check` tests it
on Modal CPU using temporary fixtures, not the real dashboard database.

The scripted CPU regression now verifies submission on the final allowed turn:
two replies, two REPL executions, one verifier call. Reward is mocked. A real
model CUDA submission and training-mode V1 integration remain gates before GRPO.

### Verifiers V1 adapter (CPU lifecycle verified)

`kernel_rlm_v1` exports a taskset and a harness without replacing the local fork
or its legacy environment. The boundary follows the pinned Verifiers revision
`b2e4e8157783b2c0dffc7821044c87f29f1c3ccf` from prime-rl v0.9.0.

```text
prime / V1 SingleAgentEnv
  -> KernelTaskset.load(): manifest -> typed KernelData / KernelTask
  -> KernelHarness.launch(): runtime starts kernel_rlm_v1.program
     -> OpenAI-compatible requests through V1 interception
     -> SubprocessReplBackend -> existing rlm_train.worker
     -> submission.json (source + complete REPL transcript)
  -> KernelTask.kernel_reward(): trusted verifier scores submitted source
  -> Episode / Trace: intercepted messages, rewards, verification, submission
```

The bounded program is root-only: no sub-LLM endpoint is connected. It uses the
fork's prompts, code-block parser, answer protocol, and persistent worker.
The harness runtime must already contain `rlm`, `rlm_train`, `kernel_rlm_v1`, and
their dependencies. The Modal check uses subprocesses **inside the Modal CPU
container**, not the laptop. Subprocesses are not a security sandbox; production
generated-code execution must remain separated from trainer credentials and
trusted scoring. `NEEDS_CONTAINER=False` permits this explicit test runtime,
not a claim that host execution is safe for arbitrary generated code.

Run `modal run -m modal_app.v1_check` from this directory. The passing CPU test
uses two scripted responses and a mocked verifier, checks final-turn submission,
and saves `episode.json` and `requests.json` on `kernel-rlm-traces` under
`v1-check/20261007T121404Z`. Its reward 0.8 is a fixture, not a model result.
The first test failed only on the fixture's trailing-newline comparison; the
corrected test matches the scorer's source stripping.

This verifies V1 **evaluation** lifecycle compatibility. Training-client token
IDs/logprobs, native prime batch construction, adapter synchronization, and the
separate GPU verifier route still need verification before a GRPO launch. The
current V1 scorer calls the local verifier function; the CPU test patches that
boundary deliberately. Do not deploy this as an already-complete RL setup.
The legacy package remains on its existing lock; the dedicated Modal V1 image
pins the newer dependency. Lazy `rlm_train` exports let the standalone worker
load without importing the removed legacy Verifiers environment API.

### Real-model submission and verifier ordering

The frozen-model episode `20261007T121226Z-7f429dc77f13` reached submission on
turn 3 after repairing a Python quoting error. Its saved result is **failed**:
reward 0.1 (submission credit), no compilation/correctness, one verifier call,
231.4 seconds episode time, and zero training updates. Full prompts, replies,
and submitted source live in `model-rollout/<run-id>/trace.json` on the traces
volume and in local logs.

The verifier incorrectly required a build during import, before lazy compilation
inside `cuda_forward` could happen. It now checks after the first invocation,
while still requiring an extension call for every trial. Run
`modal run -m modal_app.verifier_contract` for the Modal CPU regression: mocked
CUDA tests eager/lazy ordering and rejects a pure-PyTorch bypass. This is not a
security guarantee or real CUDA correctness test. The generated submission
also uses unsupported `keep_debug_handles=True`; no further GPU run was spent
on this known-invalid source. The original failed measurement stays unchanged.

Export completed frozen-model trace metrics with `modal_app.trackio_export`
using `--rollout-trace --relative-path model-rollout/<run-id>/trace.json
--run-name <unique-name>`. These use a separate `--evaluation` run suffix and
an explicit no-training scope. Keep raw reward separate from correctness:
submission credit does not mean the CUDA task passed.

### Training-data contract (verified on Modal CPU)

Run `modal run -m modal_app.training_contract`. This uses the cached, pinned
Laguna tokenizer and the actual Verifiers training client, renderer, interception
server, our REPL harness, and prime-rl trajectory conversion / GRPO / batch packer.
The generation endpoint and CUDA verifier are explicitly synthetic fixtures.
No model weights, GPU, optimizer step, or real sampling probabilities are used.

The passing run `20261007T131557Z-042cb5fd` verifies:

- All 498 supplied completion token IDs and logprobs survive exactly, once.
- Prompt and observation tokens remain masked out of the policy loss.
- The initial rendered prompt matches the checkpoint chat template; four text/code
  probes match the checkpoint's raw `tokenizer.json`. Existing Transformers
  tokenizer/config warnings remain; these probes do not establish universal parity.
- Equal-reward groups `[0, 0]` and `[1, 1]` have zero advantages and no retained
  RL-only samples. Raw rewards remain unchanged. Auxiliary CE loss is not discarded.
- Mixed rewards `[0.1, 0.8]` yield approximately `[-0.35, +0.35]`, retain two
  training samples, and pack into trainer microbatches without losing sampled tokens.
- Missing, count-mismatched, wrong-token, nonfinite, and sentinel logprobs are rejected.
- Prime's MessagePack transport round-trips the training samples.

Consecutive user messages (REPL observation followed by a separate turn reminder)
prevented incremental rendering and produced two branches. The harness now merges
adjacent user messages, preserving the same sampled tokens in one branch. This
avoids redundant context branches; it is not a measured GPU-cost reduction.

Evidence on `kernel-rlm-traces`: `training-contract/<run-id>/report.json`,
`episode.json`, `engine-fixtures.json`, and `microbatches.msgpack`. Local summary:
`logs/training-contract/<run-id>/report.json`. Do not import these synthetic rewards
or logprobs into real training/evaluation curves. The test imports W&B because
prime utilities require it, but sets `WANDB_MODE=disabled`; Trackio stays selected.

Prime's baseline GRPO uses reward minus group mean (no standard-deviation division
in this tested configuration). Its existing zero-advantage pruning removes RL
credit, not the raw reward or saved trajectory. The remaining gate is a bounded
live sampling / trainer integration check with real logprobs and adapter state;
we do not require every frozen-model CUDA submission to be correct first.

### First live GRPO gate configuration

`configs/grpo_smoke.toml` is the intentionally small first live configuration:
one inference A100, one trainer A100, two rollouts in one GRPO group, one trainer
step, rank-8 attention-only LoRA, and Muon. It keeps the file monitor enabled so
Prime writes `metrics.jsonl`, resolved configs, and rollout traces under the run
directory. It is a wiring check: a successful step proves that the real model,
V1 taskset, vLLM logprobs, adapter synchronization, Muon optimizer, and local
trace storage connect end to end; it is not enough data to judge learning.

The first run must be launched only after a Prime `--dry-run` confirms that the
custom `kernel_rlm_v1` taskset resolves inside the Modal image. The taskset
manifest path resolves to `/root/kernel_rlm/tasks/vector_add.jsonl` in that
image. The renderer ID is the registered `laguna-xs.2` variant, not the
human-readable checkpoint spelling.

### CPU-tested rollout contracts

The V1 loop accepts fenced REPL blocks or structured `repl` calls. When both
appear in a response, structured calls take precedence, so a duplicated fence
does not execute twice. Unsupported/malformed calls produce feedback. Truncated
responses execute no actions and receive a retry message.

Local intermediate verification is limited to two calls of 120 seconds each.
The enclosing REPL budget is 270 seconds (both calls plus overhead), and the
backend watchdog is 300 seconds. Feedback always includes the same status,
error, timing, and call-budget fields, including on exhaustion. Saved records
distinguish `submitted`, `turn_limit`, `truncated`, and `error` termination.

Intermediate `cuda_feedback` checks use `benchmark=False`; they do not award
reward or submit the answer. Final scoring independently validates and verifies
only the submitted source. There is no best-of-intermediate-candidates reward.

From the repository root, run `python -m pytest tests/test_kernel_contracts.py -q`.
These tests use mocked model responses and CUDA checks, including execution
through the real CPU subprocess worker. They do not establish GPU compilation
success or validate Prime's complete distributed training path.
