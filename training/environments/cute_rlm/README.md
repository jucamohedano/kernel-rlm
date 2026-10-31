# Native CuTe evaluation

CPU preparation for a new model-first cudaLLM baseline. This package does not rent
compute or launch a server. The prior draft fence environment is not used.

## What is ready on CPU

- Original task source selection, hashes and pinned documentation in REPL memory.
- Native root/child Python sessions, async child handles and separate histories.
- An opt-in output budget of 90% of the remaining rendered context for every model
  request, with shared generated-token reservations and selected token logprobs.
  The old root/child output defaults remain available for unchanged OOLONG callers.
- Explicit verifier input/state/output contracts and a worker launched only through
  an external isolation command. CPU tests use scripted models and worker results.
- Frozen experiment input capture and a launch gate requiring later GPU evidence.

These checks do **not** prove CuTe correctness, performance, isolation on the target
machine, or 262k serving capacity. They must be checked when a GPU is available.

## Preparation

Use the full pinned cudaLLM pool, not a primitive ladder or a fixed 177-task subset.
Choose a sample size and seed explicitly. Original reference Python is the task
specification and stays available in `context`; hidden expected outputs stay in the
verifier. Do not repair or resize the source to make a task run.

`config.EvalConfig` records model/dataset revisions, selection, docs, verification
settings and every limit. Verifier tolerances, training/eval mode, state and timing
settings must be chosen explicitly, not inherited from old hardware or silently
relaxed. Use `python -m cute_rlm.evaluate --help` for offline input capture.

The next-eval plan requests:

- Fresh BF16 Laguna-XS-2.1 at `c5f36269bbdbd3f27fddc9a9f9dbae0cf2cf57db`.
- A 262,144-token server window and two concurrent inference requests. The supplied
  vLLM arguments disable model generation defaults so an old 32k default does not
  silently replace the per-request policy. Pin the server build at GPU validation.
- No compaction. Task source, docs, logs and intermediate results stay outside the
  model prompt until the model inspects them.
- Per-call maximum output `floor(0.9 * (server_context_tokens - rendered_prompt_tokens))`.
  Tokenization must include tools, template and preserved reasoning. All branches
  use the same rule. This is a ceiling, not a target. It does not enlarge context.
- One million generated tokens per task tree is the initial **provisional** setting.
  Parallel calls reserve distinct allowances. Unknown usage after dispatch is
  charged conservatively rather than released for reuse. Input tokens are logged
  but are not part of this generated-token budget.
- The CuTe preset allows 64 turns per REPL session, 32 child admissions and 256
  model requests per task tree. These guards stay explicit and optional in the
  manifest; they are ceilings, not requirements. Root depth 0 can create REPL
  children at depth 1; children cannot create grandchildren. Generic native
  callers keep their existing depth-cutoff behavior.
- The old 65,536-character child-prompt screen is disabled for this preset: exact
  token counting applies instead. The local broker has an explicit 8 MiB request
  body limit, rather than silently retaining aiohttp's 1 MiB default. Other callers
  keep their existing defaults.
- Request/task deadlines remain 600/1,800 seconds initially. Queue, inference, REPL,
  verifier and timing measurements inform later changes. An explicit child wait
  timeout does not cancel the child. Freeze any change under a new run ID.

## Model interface and compile feedback

The concise system prompt and first user message are defined in
`cute_rlm/prompts.py`. The first action prints `SHOW_VARS()` for the actual
session's helper inventory and signatures, then reads the authoritative reference
and a small resource index. The model reads one relevant example section and
starts a candidate. Its current diagnostic goal is a standalone CuTe `ModelNew`
that passes compile/forward-smoke; reference behavior, constructor/forward
signatures and the configured target remain requirements. There is no new lookup
quota, early-check deadline or automatic submission. Keep the complete candidate
in `best_source`; reserve the final turn for the two answer assignments.

The only model tool is native `repl(code)`. `compile_candidate`, `search_docs`,
`read_doc`, `SHOW_VARS` and `rlm_*` helpers are injected callables: call them
directly inside the REPL, without importing or redefining them. `resources` and
`resources["helper_help"]` are data dictionaries, not objects with helper methods.
Delegation is optional.

The shared `resources` dictionary is preloaded in root and recursive child REPLs:

- `target`, `validation`, `limits`: allowlisted views of the captured config, not
  hidden verifier inputs/outputs or a second source of settings. Limits are
  configured ceilings, not live remaining balances.
- `example_index`, `examples`: character-indexed, exact historical generic CuTe
  addition source. NVIDIA's BSD-3-Clause notice is retained. That exact source
  passed the separate L40S/SM89, CuTe 4.8.1 numerical/launch/timing fixture in
  `20261030T200639Z-xs21-cute-l40s-detailed4`; its provenance records the source
  and result hashes. This is not a benchmark task solution, a task-specific
  implementation plan or readiness evidence for a new VM/dtype/layout/stream.
- `api_notes`, `helper_help`: short launch/adaptation/source-inspection notes and
  documented helper bounds and return schemas; `SHOW_VARS()` shows live signatures.
  The indexed `imports` section shows the exact `cutlass.cute` import paths without
  changing the verified example source.
- `delegation_example`: executable root REPL code for a bounded complete-`ModelNew`
  repair by a recursive child, followed by structured result collection and review.

Preparation captures exact system/user messages and `.eval-inputs/resources.json`
as hashed inputs; execution loads the captured bytes, not current defaults.
The same generic resources are bootstrapped into children; their task context
remains the delegated prompt string. Large examples stay in REPL variables rather
than the initial prompt. Inspect the indexed sections instead of dumping a whole
CLI example.

Documentation search returns match-centered excerpts with `start`/`end` character
offsets, not the first file header ranked by raw repetition. `read_doc` still
returns a dictionary: use its `excerpt` and `more` fields; `chars` is 1..1600.
These CPU changes require a new frozen run and target-VM checks. This contract
clarification changes model-facing guidance and tool discovery, not verifier or
scoring behavior. Historical prompts, inputs and results are not rewritten.

The task is already in the root REPL as a Python dictionary. Start with
`print(SHOW_VARS())` and `print(context["reference_code"])`; do not search the
filesystem for the task.
Use `search_docs(...)` and `read_doc(...)` for pinned CuTe documentation. The
reasoning REPL has no GPU. Recursive children receive their supplied prompt as
context, not the root dictionary; pass the needed task snippets explicitly.

Keep a complete Python source string defining `ModelNew` in `best_source`.
Inside the native REPL, call:

```python
feedback = compile_candidate(best_source)
# Feedback is automatically printed once; do not print it again.
# Revise best_source and call compile_candidate again when useful.
```

This host-backed helper runs the source in the configured isolated GPU worker.
CuTe compiles lazily, so the check includes constructing the model, loading the
reference initialization/state contract, one example forward and synchronization.
It returns compiler/runtime diagnostics. It does **not** return reference outputs,
compare numerical results, measure speed, or establish CuTe launch provenance.
Each call and its source hash, owner session, diagnostics and duration are saved.
The complete returned dictionary remains in REPL memory and host logs; the existing
12,000-character tool-observation display limit is unchanged. Inspect selected
fields or excerpts from the saved dictionary if a long diagnostic is truncated.

There is **no feedback-call quota**. Repeat within the existing turn, token,
REPL, worker and episode deadlines, including multiple checks in one REPL action.
Checks are not final submissions and do not set `answer["ready"]`. Submit only
when finished:

```python
answer["content"] = best_source
answer["ready"] = True
```

With the 64-turn preset, the host warns before model turn **60/64**, counting
turns from one. Warnings continue through turn 63 so the model can correct and
submit before the last turn. On turn 64 it is told to submit immediately rather
than start another check.
The warning is guidance, not an automatic submission or an extra turn. The same
rule applies to recursive REPL sessions. Generic callers without this CuTe hook
keep their prior prompts and behavior. The historical two-feedback-call cap is
not used. No old experiment snapshot or result is changed by this protocol.
This feedback path is CPU-tested with scripted models/workers. Its new broker-to-GPU
path still needs a target-VM smoke under a new frozen run before a model evaluation.
The earlier generic GPU verifier precheck is not evidence that this new path ran.

### Recursive child kernel-development example

Delegation is available, not required. Use it for a bounded repair or an independent
stage audit while the root continues useful work. Both root and child have their
own persistent REPL, `compile_candidate`, `search_docs`, `read_doc` and captured
`resources`. They share the task tree's token, model-call and time budgets. Child
context is its prompt string: it does **not** inherit the root's variables or task
dictionary. Pass the authoritative reference, complete candidate, contract and
exact diagnostics explicitly. A kernel fragment alone cannot construct `ModelNew`
with the trusted task inputs used by compile feedback.

Read `print(resources["delegation_example"])` after storing `best_source`, `contract`
and `feedback` in the root. The example assembles a child prompt from those values
and `context["reference_code"]`, then calls:

```python
repair_handle = rlm_query(repair_prompt, blocking=False)
# Keep working independently, then collect in a later root action:
child_report = json.loads(rlm_wait(repair_handle, timeout_s=30))
child_source = child_report["source"]
ast.parse(child_source)
```

The child must repair the **complete** source and call `compile_candidate` in its
own REPL. It returns JSON containing `source`, actual feedback dictionaries in
`checks`, `changes` and `remaining_risks`, using its own answer assignments. A failed
repair returns its real diagnostics, not a success claim. `compiled` means only a
forward/synchronize smoke; it does not establish numerical correctness or full-CuTe
execution. The root reviews the source and checks before choosing whether to set
`best_source = child_source`, and submits separately. The example does not accept
or submit a child result automatically.

`rlm_status(handle)` inspects progress; `rlm_result(handle)` collects an already
finished child. A timed-out wait does not cancel the child, so retain its handle
and collect later. Root submission stops unfinished children. At depth 1 a child
still has its REPL and compilation helper, but a grandchild request is rejected.
`llm_query` is a plain completion, not a substitute for this workflow.

`tests/test_cute_delegation_example.py` executes the example through real root and
child subprocess REPLs, a real broker and a scripted model/compilation callback.
It checks child-owned feedback, complete-source return, root integration, rejected
grandchildren and confirmed worker exit. This CPU protocol test is **not** evidence
of a GPU compilation, correct kernel or a model choosing to delegate. No task
solution or hidden expected output is added to the example.

### Laguna thinking setting

The native request uses `chat_template_kwargs={"enable_thinking": true}` and
preserves returned reasoning in later messages. It does not send
`reasoning_effort="high"`. The pinned Laguna chat template exposes boolean
thinking, not a documented low/medium/high effort tier; accepting a generic
vLLM API field does not create model-specific tier support. Keep thinking on.
The last run's provider `reasoning_tokens=0` field did not mean thinking was off:
all 128 raw responses contained reasoning text. No effort setting is changed by
this concise-prompt condition.

## GPU handoff

Before any scored model task, verify and save evidence for:

1. Pinned model/server/tokenizer/CuTe/driver/software and actual GPU.
2. Startup at 262,144, KV-cache capacity and two concurrent long requests; no silent
   fallback to a smaller window.
3. Rendered tokenizer counts versus actual response prompt usage; selected-token
   logprob support and preserved reasoning on native tool turns.
4. A real CuTe compile/launch, independent reference/candidate state and inputs,
   correctness trials and timing warmup/repetitions on that GPU.
5. Worker isolation, secret removal and no access to hidden verifier outputs.

`readiness.json` starts with these checks pending. A later GPU receipt must bind the
same frozen configuration and manifest before `evaluate_prepared` starts. It is an
experiment gate, not a security attestation. No CPU-generated or mock receipt may
be used as evidence of a real GPU check.

The supported first worker contract uses `ModelNew`, the reference constructor
arguments, exact parameter/buffer keys and shapes, and fresh inputs/state for every
correctness and timing call. Input tensors are placed on CUDA explicitly; their
original dtype is retained. Noncontiguous or aliased tensor storage is reported as
unsupported rather than silently changed. `preserve` mode keeps the reference's
training flag; `eval` mode is an explicit configuration choice. No tolerance is
chosen by this package. Both are required fields in the verifier configuration.
Nested outputs, changed inputs and changed state are checked separately. CUDA event
timing measures the forward call, after state/input reset and warmup. The end event is recorded directly after forward, then the device is synchronized.
The timed forward must join any other streams back to the caller stream; validate
that contract with the GPU trace before interpreting timing.
Compile/setup and process wall time are reported separately. Validate this timing
protocol on the target GPU before interpreting a speedup. Final verification
remains independent of the model-visible compile feedback described below.

The GPU worker has passed a generic addition smoke on RTX PRO 6000 with CUDA
13.3 and CuTe DSL 4.8.1. This does not validate every cudaLLM reference or candidate.
GPU dependencies (Torch/CUTLASS/CuTe and the matching compiler/driver) belong in the
later GPU worker image, not the CPU test environment. Warmup and timing settings,
input placement, module modes, state-key constraints and tolerance choices are all
part of the frozen verifier contract. Do not silently relax an unsupported case.
A protected GPU-launch observer must bind compile/launch evidence to the request,
candidate and config. Without that evidence a numerical pass is `launch_unverified`,
not a correct CuTe result and never a speedup. The configured launcher must place
reference and candidate in separate boundaries and mount only candidate inputs;
the protocol's input path alone does not enforce filesystem separation.

Candidate workers must run behind a separately reviewed sandbox boundary. A local
Python process, AST check or imported `cutlass.cute` name cannot prove isolation or
CuTe execution. Do not report speedup for a candidate without trusted launch evidence
and correctness. Unsupported source/state/output contracts remain explicit outcomes.

Use the repository environment from the project root:

```bash
uv sync --group dev --group test
PYTHONPATH=training/src:training/environments/kernel_rlm:training/environments/cute_rlm \
  uv run --no-sync python -m cute_rlm.dataset /path/to/pinned.parquet selected.jsonl \
  --count TASK_COUNT --seed SEED
PYTHONPATH=training/src:training/environments/kernel_rlm:training/environments/cute_rlm \
  uv run --no-sync python -m cute_rlm.evaluate --project . --config config.json \
  --run-id NEW_RUN_ID --output-root /path/to/experiments
```

The second command only prepares a frozen bundle. No CLI auto-starts a model or
GPU job. `evaluate_prepared` is a host API for the later GPU launcher after its
checks pass. `preflight.probe_two_long_requests` and `kv_cache_evidence` are prepared
for those explicit checks, not executed during CPU preparation.

Run CPU checks from the project root with `make check`. No existing OOLONG evidence
or active VM is changed by this package. Track #21 (execution/budgets) and #24
(model-first CuTe baseline); this branch is stacked on #28 and #27.

## Explicit launch wiring

The scored host entry point is `python -m cute_rlm.launch RUN_DIRECTORY --base-url
URL --receipt GPU_RECEIPT --sandbox-spec SPEC --observer-root EVIDENCE_DIRECTORY`.
It does not start the serving process itself. It verifies the attached `/models`
record against the protected startup evidence before sending scored requests.

The default target remains **RTX PRO 6000 Blackwell, compute capability 12.0
(sm_120), with the CUDA 13.3 toolkit**. For a new GPU, freeze its actual model name
in `gpu_model` and explicitly set `gpu_compute_capability` as a positive
major.minor string. For example, an L40S run uses `"gpu_model": "NVIDIA L40S"` and
`"gpu_compute_capability": "8.9"`. These fields render the task's execution target.
Readiness requires matching observed `hardware.gpu_model` and
`hardware.gpu_compute_capability` in the GPU receipt, not just requested settings.
Do not reuse an old hardware receipt or change historical experiment snapshots.

CuTe JIT uses the worker's actual CUDA device to select its compilation target;
this configuration does not invent or set a CuTe architecture environment option.
The `cute_compile_launch` readiness evidence must confirm compilation/launch on
the configured device and compute capability. A prompt or startup argument alone
does not establish that match. Do not force an architecture copied from another
GPU. No Torch extension architecture override is needed by the CuTe-only worker.

Serving `tensor_parallel_size` defaults to 1 and is separate from
`max_concurrent_requests`. Set `"tensor_parallel_size": 2` explicitly for a
2-GPU serving setup; the generated vLLM arguments include `--tensor-parallel-size`.
The receipt must record the observed integer `server_identity.tensor_parallel_size`
matching the frozen configuration, backed by the server startup evidence.
This does not change the 262,144-token context, BF16 model, concurrency, scoring
or task budgets, and does not prove that the GPU pair has enough memory. The
operator must validate actual serving capacity before a scored run.

A CUDA version printed by `nvidia-smi` is driver capability, not proof that the
13.3 toolkit and compiler are installed. Pin actual `nvcc`, Torch, CuTe and driver
versions in the later GPU receipt.

`sandbox.SandboxSpec` describes explicit read-only runtime mounts and GPU device
nodes. The namespace starts with no host root, no network and no inherited secret
environment. REPL workers receive only their Unix broker socket and allowlisted
rollout/time settings, with no GPU. Reference and candidate phases get separate
writable output mounts; candidate inputs contain no reference outputs. Validate
actual Python/library paths, namespace support and device permissions on the VM.
A live VM smoke checked hidden host files, absent REPL GPU devices, a Unix broker
round trip and persistent Python state. Reference/candidate phases also passed a
CuTe addition smoke. These checks do not establish resistance to every malicious
candidate or certify all task/runtime combinations.

`compile_trace` retains exact PTX/CUBIN artifacts and records compiled-call NVTX
ranges during an extra profile pass. `NsightObserver` requires matching artifact
hashes, PTX entry symbols and CUPTI launch correlations within the exact forward
and compiled-call ranges. CUDA activity alone is not CuTe compiler provenance.
The profile runs in an isolated namespace and is copied to host evidence afterward;
these are observational records, **not tamper-proof attestation** against malicious
Python that replaces instrumentation. Numerical results and timings come from a
fresh unprofiled process. The integrated capture passed the generic addition smoke on the actual VM.
Readiness for a scored run still needs its exact task/config/document bundle and
operator-reviewed evidence receipt. Traced probe timings are never scored.


Example sandbox shape (replace all paths after inspecting the actual VM):

```json
{
  "bwrap": "/usr/bin/bwrap",
  "runtime_python": "/opt/cute-worker/bin/python",
  "readonly_bindings": [
    {"source": "/usr", "target": "/usr"},
    {"source": "/lib", "target": "/lib"},
    {"source": "/lib64", "target": "/lib64"},
    {"source": "/opt/cute-worker", "target": "/opt/cute-worker"},
    {"source": "/opt/eval-worker-modules", "target": "/runtime/python"}
  ],
  "gpu_devices": [{"source": "/dev/nvidia0", "target": "/dev/nvidia0"}],
  "observer_root": "/opt/protected-captures",
  "nsys_binary": "/opt/nvidia/nsight-systems/2026.1.3/target-linux-x64/nsys",
  "privileged_setup": true,
  "sandbox_uid": 1000,
  "sandbox_gid": 1000
}
```

Keep the Python environment at the same path inside the namespace so venv and
shared-library resolution are preserved. Mount only code/runtime trees, not the
project's run directory, home directory, model secrets, or protected captures.
List every required allocated NVIDIA device and library explicitly after VM
inspection. The example is not a validated launch specification.

The tested VM requires privileged namespace setup. `sudo` runs only bubblewrap's
mount/namespace setup; `setpriv` drops to the configured non-root UID/GID, removes
capabilities and sets no-new-privs before Python starts. The live spec also binds
`/bin`, small read-only account/loader files, the installed Nsight runtime and
read-only `/sys/devices/system`, `/sys/module/nvidia` and `/sys/module/nvidia_uvm`
metadata. `/tmp` and `/dev/shm` are private writable mounts. These paths were needed
for the Nsight helper; do not replace them with a broad host-root/home mount.

Live serving capacity: the 262,144-token model window started successfully, with
477,395 tokens of reported KV cache (1.82 full-window requests). Two simultaneous
238,051-token prompts completed with matching tokenizer usage and selected-token
logprobs. That is not a guarantee that two full 262k prompts plus arbitrary output
fit at once. Queue/preemption and memory pressure must stay visible during eval.
Torch is `2.13.0+cu130` (built with CUDA 13.0); the toolkit compiler is CUDA
13.3.73 and the 610.57.04 compatibility library is used over kernel driver
580.159.03. Preserve those distinctions in the run record.

## Failure evidence and manual recovery

Every failed task and child record is retained, including partial progress and
cleanup outcomes. CuTe model requests stream: append-only `stream_chunk` records
save raw wire bytes before parsing, and `stream_delta` records preserve partial
reasoning, text, tool arguments and selected logprobs. Incomplete tool calls are
never executed. Complete responses keep the normal canonical response/usage ledger.

Failures record the active stage, local deadline, connect/read phase when known,
HTTP status, exception chain and last progress. A model-request deadline is not a
compiler deadline. A client-side trace cannot prove an unknown server root cause;
keep `server_root_cause=unknown` when the evidence does not establish one. Preserve
server logs and metrics alongside the per-request stream journal on the VM.

A data-only `checkpoint-NNNNN.json` saves supported REPL variables, source,
contract, feedback, messages and spent budgets before cleanup. No pickle and no
replay of prior tool actions is used. Manual recovery is a **new linked attempt**,
not a continuation of provider generation state. It retries only the interrupted
request from the last complete REPL state; it does not execute partial generated
code or overwrite the failed trajectory. A model timeout/connect/read failure or
retryable HTTP failure may be eligible; compilation, REPL, protocol and budget
failures are not silently retried.

Recovery is opt-in after inspecting the failure and confirming worker cleanup.
The first implementation supports root sessions without descendant sessions and
only the supported data-only variable/import/AST snapshot types. Checkpoints list
unsupported state or descendant handles explicitly and refuse recovery rather
than falsely reconstructing a session. This limitation does not remove any failed
child trajectory or stream evidence. Old non-streaming runs have no such live
checkpoints and cannot be retroactively recovered exactly.

For an eligible checkpoint, explicitly choose a new output directory and a new
wall-time allowance (example values are operator choices, not preset changes):

```bash
PYTHONPATH=training/src:training/environments/kernel_rlm:training/environments/cute_rlm \
  python -m cute_rlm.resume ORIGINAL_RUN ORIGINAL_RUN/checkpoint-00000.json \
  --output NEW_RECOVERY_DIRECTORY --additional-seconds SECONDS \
  --base-url URL --receipt CURRENT_RECEIPT --sandbox-spec SPEC \
  --observer-root CAPTURES --charge-uncertain-reservation
```

The last flag is required if usage is unknown: it conservatively spends the full
uncertain reservation, **not** an invented actual token count. Prior spent turns,
model calls and generated allowance remain spent. Model, prompts, captured
resources and other frozen limits must match. The explicit added wall time and
`resume_of` lineage are recorded; no automatic replay or hidden budget reset.
The 600-second per-request and 1,800-second initial episode defaults are unchanged.

## Async inference and VRAM fallback

Root and child model calls use async I/O, with at most two inference requests in
flight by default. Python work, compilation and child waits do not hold an
inference slot. Async calls can overlap, but do not promise two full context
windows or free compiler memory. Validate actual VRAM/KV capacity with the model
and verifier resident before scored dispatch.

`inference_concurrency_after_preflight` selects one request only for a structured
**model-inference CUDA OOM** result. It does not reinterpret compiler failures or
ordinary timeouts as OOM. If preflight OOM occurs, preserve that evidence, recheck
serial operation, then freeze `max_concurrent_requests=1` and a
`single_long_request` readiness check. The server's max-num-seqs and native
semaphore derive from that same config. Otherwise retain async concurrency 2 and
its concurrent-request readiness gate. No scored task is automatically restarted
or silently retried following an OOM.

## What counts as a correct full-CuTe implementation?

Keep three questions separate:

1. **Numerically correct:** outputs, input mutations and post-state pass the
   independent verifier across fresh trials with unchanged dtype/mode/tolerances.
2. **CuTe observed:** retained compiler artifacts and exact forward/compiled-call
   NVTX/CUPTI correlations establish the individual CuTe launches.
3. **Numerical path reviewed:** source and forward launch inventory account for
   every numerical operation and required state update, including work not
   attributed to CuTe. PyTorch allocation, DLPack and ordinary data/state copies
   are allowed plumbing, not blanket violations.

The current `cute_verified` backend gate is an **any-matching-launch observation**,
not automatic full-CuTe coverage. Until all numerical work is accounted for,
label coverage `unknown/review required`; do not claim full CuTe from one observed
kernel, an import or a `compiled` smoke. Do not reject legitimate allocation or
memset just because it is outside CuTe. No complete all-launch classifier or
malicious-code-resistant attestation is implemented here.

For example, the saved incorrect convolution/noise submission had 24 directly
correlated forward kernels across three trials: 3 attributed to its retained
CuTe noise kernel and 21 other source-supported numerical kernels, including
convolution and BMM. It was demonstrably mixed-backend, not full CuTe. Kernel-count
or time fractions are diagnostics, not numerical-operation coverage or a scoring
threshold. A scoped full-CuTe claim currently needs reviewer sign-off in addition
to correctness and trace evidence. Timing remains unprofiled and correctness-gated.
