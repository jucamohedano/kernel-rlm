# Exact-turn RLM training with TRL

**18 October: one real update passed.** vLLM TP2 generated four trajectories on
L1 subtraction task `lvl1-seed42-0038`. Native TRL/Muon trained 22 root replies
(10,631 target tokens) and exported a 94.21 MiB adapter/local-state checkpoint.
CPU read-back and a fresh vLLM process loading that trained adapter passed.
The raw run is `20261018T212420Z-50f5a8cb`, using source commit `9f73e79`.

**Quality did not pass:** rewards were `[0.1, 0, 0.1, 0.1]`; three final sources,
zero correct kernels. Two bindings omitted constructor-derived arguments, one
submission failed compilation, and one trajectory ended truncated. Both limited
calls were 512-token recovery replies. Four intermediate feedback checks timed
out, including three explicit GPU-slot waits. No child call occurred.

Keep the working framework route, but seek positive correctness signal before
longer training. Fix the stateful-task interface and verifier feedback queue;
audit the native first-update clipping statistic (about 0.306) against sampler
probabilities. That statistic is averaged across microbatches and includes
zero-credit padding in metrics. It is not a measured fraction of all real tokens.

## Credit is not a separate reward per turn

For four trajectories, compute one GRPO advantage from each final verifier reward,
including speed. Attach that same advantage to every root reply in its trajectory.
Keep each reply's exact sampled prompt, token IDs and behavior log probabilities.
Do not join turns into one sequence: old reasoning is absent from later contexts.
REPL observations and child responses are context, not root loss targets.

The native TRL DAPO clipped-PPO loss sums root-token terms and divides by the
global count of real root tokens. Unequal turn counts do not change how the group
reward mean or standard deviation is calculated. Extra rows needed for distributed
alignment duplicate a real row with zero advantage; auxiliary router loss, entropy
bonus and KL penalty are disabled, so these rows have exactly zero gradient.
Native entropy/clip metrics include these padding rows; use the recorded real-row
counts for interpreting this bounded adapter.

`training_rows` rejects uniform final rewards. Whole groups can be oversampled at
most three times; exhausted or corrupted groups stop without an optimizer update.
The existing correctness/speed verifier and its submission guards are unchanged.

## Feedback repair status

The documentation-assisted continuation uses `--max-turns 10`: eight normal
turns, with its first deadline reminder at turn 8, then two short recovery turns.
The reply/context limits remain 4,096/14,336 tokens and recovery stays at 1,024.
Use `--cuda-docs <corpus.json> --cuda-docs-sha256 <digest>` to pin the offline
NVIDIA CUDA 13.3 archive collection matching this VM's compiler. The downloader
is `python -m kernel_rlm.cuda_docs --download <private-cache-directory>`.

Only collection metadata and search instructions enter the initial prompt.
`search_cuda_docs` and `read_cuda_doc` run inside the existing REPL, return bounded
cited excerpts, and log retrievals. Raw manuals remain in a private cache, outside
the Git source archive. The corpus hash/version and turn limit are immutable on
resume. Delegation instructions and the indexing-review example occur once.

Root and child requests explicitly enable thinking. Full sampled root token IDs,
including reasoning, receive GRPO credit; child outputs and REPL observations do
not. Earlier private reasoning is absent from later conversation messages, while
raw responses remain in traces. `reasoning_characters` helps distinguish real
reasoning from an API that omits its reasoning-token counter.

The completed baseline through policy 8 remains immutable. New source adds
CPU/meta contract preflight, bridge-owned cancellable inference admission and
early feedback draining. See `kernel_rlm/STATEFUL_TASKS.md` for exact budgets
and limitations. Training is not restarted by these CPU-tested repairs.

## Two-GPU scheduling priorities

Both BF16 inference (TP2) and native training need the L40S pair. Permanently
assigning one GPU to verification does not fit the current model layout.
CPU preflight and extension compilation do not need a GPU and can overlap model
inference; CUDA correctness checks and uncontended timing require GPU access.

The next optimizations to validate, in order, are:

1. The warm-engine component can reuse a fixed policy, but production final
   scoring requires zero resident GPU processes. The driver therefore closes
   inference before final scoring, even with `--reuse-inference`. Retaining a
   sleeping engine across groups is disabled until the scorer can independently
   validate its ownership and memory isolation. Do not weaken the idle guard.
   Closing the parent process does not prove that CUDA workers have released
   their contexts. Cleanup polls the unchanged GPU-idle guard for at most 30
   seconds after process-group termination; persistent PIDs still fail closed.
2. `--precompile-finals` starts bounded CPU-only preparation as each rollout
   submits, while siblings continue. The worker hides CUDA and shares the existing
   source-isolated build cache. Its reports are not grades. Final correctness and
   timing still run independently, including after failed CPU preparation.
3. Only after measuring residual GPU memory, test keeping sleeping inference
   workers across learner phases. Sleep can release weights/KV cache, but CUDA
   contexts and communication buffers may remain. Learner allocation retries
   make simultaneous residency unsafe to assume.

After a scoring-startup failure, a new immutable continuation can use
`--recover-generated-run FAILED_RUN` with the same validated checkpoint parent.
It verifies all eight captured episodes, exact token samples, task data,
documentation hash and turn/context protocol. It copies durable scoring attempts
without resetting them or regenerating trajectories. Audits still gate updates.

Both flags are opt-in and immutable on resume. Per-episode `precompile.json` and
`precompile-error.json`, plus `inference-lifecycle.jsonl`, record their actual work.
Cancelling preparation terminates the worker and its compiler process group;
a real CPU subprocess regression checks cleanup and credential stripping.
The fixed-policy ABBA scheduling diagnostic measures component savings; its
short synthetic requests do not measure complete GRPO throughput or quality.

On 19 October, the policy-8 GPU diagnostic measured 66.13 seconds for process
startup and 2.11 seconds for median same-process wake. Its serial/overlap/overlap/
serial trials averaged 99.51 versus 69.85 seconds per fixed-workload cycle
(29.8% less time). All four independent CUDA checks passed, with no optimizer
updates. The same candidate used fresh extension caches in each trial; actual
training with already-cached feedback builds may gain less. This is not a full
eight-rollout RLM or GRPO throughput measurement. See
`notes/results/20261019-scheduling-probe.json` in the project notes.

Keep inference and optimizer updates phase-separated until those memory checks
pass. The last bounded block spent about 955 seconds in native rank-0 training
out of 12,557 seconds elapsed (7.6%). The rest includes useful inference,
verification, audits, startup and waits; it is not all idle time. Improve
end-to-end useful-group throughput, not learner utilization alone.

## Calling flow

```text
vLLM 0.25.1, TP2, separate inference environment
└── existing KernelGenerator.run_episode × four
    ├── exact-token CompletionBridge
    ├── existing V1 Python REPL program and optional child calls
    ├── cuda_feedback → shared-GPU sleep/wake admin relay
    └── trusted final CUDA verifier → terminal reward
         ↓ exact root-turn group with trajectory advantages
stop inference workers and release GPUs
         ↓
RLMGRPOTrainer (subclass of native TRL GRPOTrainer)
├── native FSDP2 preparation and Dion Muon optimizer factory
├── public get_train_dataloader: balanced exact-turn microbatches
├── public training_step: no stock single-sequence rollout preparation
├── native compute_loss: DAPO PPO clipping against sampled log probabilities
└── one optimizer update → compact adapter export and local Muon/RNG shards
```

The explicit backward scaling cancels Accelerate's automatic accumulation
division: DAPO already uses the global token denominator. CPU proofs compare the
actual native TRL update with a direct reference gradient, on one process and two
Gloo ranks. Their SGD optimizer is only an analytic test harness; production
always uses the validated Dion Muon factory.

## Limits of this first run

- Existing two L40S GPUs only. No concurrent inference/trainer GPU copies.
- One optimizer update; 30-minute original wall deadline, including failed GPU
  startups. No automatic retries or synthetic-reward fallback.
- Four trajectories per task, at most three candidate groups from the unchanged
  materialized L1 curriculum. Root cap 4,096, context 12,288, up to eight root
  turns and two text-only children. Recovery caps remain 512.
- The client uses the inference tokenizer endpoint to cap outputs to exact context
  headroom and checks chat/tokenize prompt-ID equality. Actual requests are saved.
- A local admin relay sleeps vLLM for exclusive verifier access and wakes it again.
  Existing file locks drain model calls before exclusive verification. This is
  process coordination, not a security sandbox for hostile model code.
- After generation the server stops, then native TRL runs from exact saved turns.
  This first adapter is a bounded one-group update, not an asynchronous pipeline.
- Compact export saves only trainable adapters, local Muon momentum and RNG state.
  Frozen base weights are never duplicated in the new checkpoint. Fresh-process
  resume and post-update inference reload still require explicit verification.

All requests, root/child responses, REPL observations, submissions, feedback,
verifier results, rejected groups and training inputs must remain inspectable.
One completed optimizer step is not evidence of quality improvement.

## Preparing the full L1 pass

- Eight-rollout groups use a rolling queue with at most four active episodes;
  a finished episode frees its slot immediately for the next one. Every member uses the same
  explicit adapter and policy version. Four-rollout diagnostics remain supported.
- The native two-rank CPU gradient proof passed with eight trajectories and policy
  version seven: maximum parameter error was about `7.3e-12`. This validates credit
  and accumulation, not model quality or distributed GPU checkpoint resume.
- Collection stops for diagnosis if no reward exceeds 0.1. Uniform rewards,
  including equal speed rewards, remain filtered. Rejected rewards are saved intact.
- Each trainer process consumes exactly one fresh group. Its local step starts at
  zero; the global policy version comes from a matching compact checkpoint.
- New checkpoint support saves full LoRA plus local Muon momentum, its step counter,
  scheduler and Python/NumPy/Torch RNG states. A checksum manifest marks completed
  saves. Fresh two-rank GPU readback and re-export now pass; an actual resumed
  optimizer update still requires verification.
- Legacy checkpoints lack full RNG/scheduler state. Their limited restore is
  explicitly reported, never presented as an exact full-Trainer resume.
- An audit-only mode scores every real root target with native TRL without an
  optimizer update; padding is excluded. The first measured audit found 93.25% of
  token probability ratios outside 0.8–1.2, with mean absolute log-ratio about 5.17.
  This is a blocking discrepancy, not ordinary rounding evidence. Diagnose it
  before using these samples for prolonged curriculum training.
- Training now requires a passing `--audit-report` for the exact group hash,
  policy version and real token count. Conservative readiness limits are mean
  absolute log-ratio at most 0.1 and at most 5% outside ratios 0.8–1.2. These are
  diagnostic safety thresholds, not claims about model quality. Missing, stale
  or failing audits stop before model loading; audit/readback modes never update.
- A separate unsharded forward reproduces a large discrepancy on 64 saved tokens
  (mean absolute log-ratio about 5.10). This rules out FSDP as the sole cause; it
  does not establish that FSDP and unsharded execution are identical.
- Direct CPU comparisons found identical BF16 RoPE tables at six tested positions
  for global and sliding attention. Nested config warnings alone are not proof
  of incorrect rotary tables. The original router-bias tensors are already BF16.
- The first resume attempt caught activation-checkpoint wrapper names in the old
  adapter export. Portable names now remove those wrappers. The corrected GPU
  restore checks all 320 adapter shards; a CPU comparison confirms all 1,280 local
  adapter/momentum tensors and 320 exported full adapters are bitwise unchanged.
  All original files are preserved and the old compact checkpoint is hash-verified
  in ignored local backup storage. Prior vLLM reload only proved that generation
  ran; adapter tensor coverage must be checked explicitly before claiming sync.

The full 50-task GPU pass has not started. Sampling remains phase-alternated;
this change does not enable simultaneous vLLM and trainer model copies.

## Saved state and evidence

For the next run, recovery replies allow up to 1,024 tokens instead of 512.
The normal reply cap remains 4,096 and the total context cap remains 12,288.
Recovery still requires context headroom and permits only final submission, not
new compilation, feedback or delegation. The historical first run above used 512.

The checkpoint contains full LoRA matrices plus each rank's local adapter,
momentum and Torch RNG tensors. It is not yet a full native Trainer checkpoint:
fresh-process optimizer/scheduler/data-sampler resume has not been validated.
Frozen base tensors are absent. This is a new baseline from the pinned base,
not a continuation of the old Prime checkpoint. Back it up before releasing the VM.

The local importer verifies saved source hashes and an optional equivalent Git
commit. It maps the run into the existing dashboard without changing trajectory
design. Full traces remain on the VM and ignored local log directories; aggregate
results are in `notes/results/20261018-trl-live.json` in the parent project.
The single- and two-rank native CPU proofs matched direct reference gradients;
the two-rank maximum parameter error was about `7.3e-12`.

## Resumable CloudRift curriculum driver

`live_curriculum.py` runs one L1 task group at a time. It alternates between
two phases because vLLM inference and FSDP2 training cannot fit together safely
on these two L40S GPUs:

1. Start TP2 vLLM and load the current LoRA adapter by its local checkpoint path.
2. Generate eight trajectories for one dataset task through a four-slot rolling queue.
3. Stop inference. Keep the complete request, REPL, feedback and verifier traces.
4. Run the no-update, exact-group sampler/learner probability audit.
5. Run one TRL/Muon update only when the group has valid reward variance and the
   exact audit passes.
6. Save a compact adapter/Muon checkpoint, advance the policy version, and use
   that checkpoint for the next task.

The driver records each task result, policy version, reward group, audit summary,
checkpoint, telemetry and spend estimate. It stops for diagnosis when a group
has weak or zero-variance reward by default. With `--continue-weak-groups`, it
instead defers a fully scored group with weak or zero-variance reward and no
final infrastructure errors. Deferred tasks remain untrained and visible in the
report; their rewards, sampled policy and traces are preserved. The next task
uses the same checkpoint. Failed audits and infrastructure errors still stop
the driver. Neither option changes the reward function or admission rules.

Use `--max-updates 3` to cap successful optimizer updates independently of weak
groups. `--max-groups` caps attempted groups instead. Both caps apply per driver
invocation; after a pause, subtract completed updates from the authorized block
before resuming. If all remaining tasks are deferred, the pass pauses; it does
not claim completion or retry the deferred tasks automatically.

`--resume` continues a paused run from its
latest validated checkpoint and reuses saved groups; a run lock prevents a
second driver from launching against the same directory. Do not resume until
the VM has no leftover GPU worker processes.

The base 150-second model-call deadline proved too short after policy-1 adapter
loading: one root response took 138.5 seconds and another hit the 150-second
timeout. The collector's HTTP client also had a separate fixed 180-second cap.
The live driver now allows 600 seconds for a model request, 630 seconds for
the HTTP client, 1,500 seconds for a trajectory and 2,400 seconds for a group.
That baseline used a 4,096-token reply cap and a 12,288-token context cap.

## Longer context for new TRL runs

Exact context exhaustion is a typed episode outcome. The bridge returns a
specific HTTP 400 code with the exact prompt count, not a transport failure.
The root saves its completed sampled replies and terminates without inventing
a model reply. A missing submission keeps the existing zero terminal reward.
Durable scoring accepts only a matching final exhaustion marker and excludes
that marker from token targets. Real bridge/transport failures still block
training. The collector lets independent siblings finish and records every
infrastructure failure before rejecting the group; the overall group deadline
still cancels outstanding work.

After an explicitly authorized retry of an incomplete failed attempt, select
the validated saved-checkpoint ancestor with `--parent-run-dir` and record the
failed descendant with `--preserved-incomplete-run`. Its checkpoint, coverage,
dataset, policy version and ancestor must match. Complete generated groups
cannot be discarded through this option. Old partial files are hashed and kept,
attempt counts are retained, and a fresh complete group is generated under the
new context/source. No partial group is trained or counted as completed.

The new-run CLI default is 14,336 total prompt-plus-reply tokens. Use
`--context-tokens 14336` explicitly in the launch record. Both 14,336 and 16,384
passed two native backward passes with real checkpoint-5 Muon state. The smaller
limit used about 1.1 GiB less peak allocated VRAM. Both triggered allocator
retries; device-free memory alone excludes reclaimable PyTorch cache. This is a
conservative measured default, not a guarantee for all routing patterns. Root replies stay at
4,096; recovery stays at 1,024; children keep their existing 2,048-token caps.
The same context value reaches vLLM, exact token counting, the RLM generator,
durable generation manifests, final scoring, training rows and dashboard
provenance. Historical manifests without this field still mean 12,288 tokens.
Resuming an existing immutable run with a different context is rejected; keep
`--context-tokens 12288` for those old runs and prepare a new snapshot for a
larger-context continuation.

Inference KV-cache headroom is not proof that training fits. Before enabling a
larger limit on the two L40S GPUs, run a bounded native FSDP2/TRL loss memory
check with `kernel_trl.rlm_update --memory-probe-tokens 14336 --resume-from
<checkpoint>`. This mode restores real adapters and Muon state, accumulates two
synthetic full-length backward passes, performs zero optimizer steps, disables
training tracking and writes no checkpoint. It reports allocated/reserved VRAM
and free device memory. Synthetic inputs test one routing pattern, not every
possible real trajectory. Keep the independent context-exhaustion and sibling
cancellation repairs as open work; a larger cap cannot guarantee termination.

The runtime must expose `ninja` on `PATH` for both REPL compilation and vLLM
LoRA kernels. The driver adds the training virtual environment's `bin` directory
to worker processes and the inference virtual environment's `bin` directory to
vLLM. Compile-only feedback hides CUDA devices, so it cannot auto-detect the
compiler target. The L40S driver sets `TORCH_CUDA_ARCH_LIST=8.9` (`sm_89`); set
the matching architecture when you move this workflow to different hardware.
The VM's cached Python 3.12 headers are also required by Triton's generated
launcher; the driver adds their normal and multiarch include directories to
`CPATH`. This uses the cached header package, not a system package install.
The training Python path includes the environment source, the repository's
`training/src` package tree (`rlm_train`), and the RLM repository root.
This keeps vLLM's runtime-update API disabled and the local service bound to
loopback.

## Inference admission ownership

Root requests acquire their reader slot in `kernel_rlm_v1.program`; child
requests acquire it in `kernel_rlm.subcalls`. The slot remains held across the
loopback bridge and the inference HTTP response. `InferenceClient` is transport
only: it must not acquire a second reader slot. The standalone startup smoke
request acquires its own slot in the collector.

A CPU regression reproduces the previous race: the root holds a reader lock,
the verifier acquires writer intent, and the collector tries to acquire another
reader lock. The collector waits for the verifier, while the verifier waits for
the root's HTTP request to finish. Removing the duplicate acquisition breaks
this cycle without removing verifier priority or allowing inference during
exclusive verification. This fix is not a measured GPU-throughput improvement;
the next bounded live check must still validate scheduling behavior.

## New snapshot after a diagnosed failure

Use `--parent-run-dir` in a new output directory when an explicitly authorized
retry needs corrected source. The parent must be stopped, with no pending group.
Its dataset bytes, all 50 task IDs and order, manifest checksum and complete
optimizer/coverage history must match. The latest checkpoint is independently
validated through the existing two-rank bootstrap reports and probability audit.

The new run keeps every completed task, prior result and attempt count. It
records the parent's state hash, source commit and rejected group separately;
it never edits the parent's failed state or trains its rejected samples. Use
`--max-groups 1` for the first fresh eight-rollout validation. Pass the updated
authorization file with the original spending baseline. This is not a way to
relax probability audits, reset the budget or resume unfinished infrastructure
failures blindly.

For a bounded continuation test, use `--bootstrap-dir` with a successful
step-1 run and `--max-groups 1`. Remove `--max-groups` for the remaining L1
curriculum. Run the same command with `--resume` after a clean paused boundary.
Keep `--budget-file` pointed to the operator-approved authorization record with
the original spending baseline; each
GPU phase checks the remaining allowance before it starts.

An authorization record may also set `stop_at_unix_seconds`. A phase starts only
when its full requested duration plus 30 seconds for cleanup fits before that
deadline. This preserves the original spending baseline and avoids starting a
new phase near the user's working-time cutoff.
