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
- Existing 32-turn, 32-child and 256-model-request guards stay explicit and optional
  in the manifest; they are not model requirements. No hidden limit is removed.
- The old 65,536-character child-prompt screen is disabled for this preset: exact
  token counting applies instead. The local broker has an explicit 8 MiB request
  body limit, rather than silently retaining aiohttp's 1 MiB default. Other callers
  keep their existing defaults.
- Request/task deadlines remain 600/1,800 seconds initially. Queue, inference, REPL,
  verifier and timing measurements inform later changes. An explicit child wait
  timeout does not cancel the child. Freeze any change under a new run ID.

## Model interface and compile feedback

The task is already in the root REPL as a Python dictionary. Start with
`print(context["reference_code"])`; do not search the filesystem for the task.
Use `search_docs(...)` and `read_doc(...)` for pinned CuTe documentation. The
reasoning REPL has no GPU. Recursive children receive their supplied prompt as
context, not the root dictionary; pass the needed task snippets explicitly.

Keep a complete Python source string defining `ModelNew` in a variable such as
`source`. Inside the native REPL, call:

```python
feedback = compile_candidate(source)  # Prints diagnostics and returns a dictionary.
# Revise source and call compile_candidate again when useful.
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
answer["content"] = source
answer["ready"] = True
```

The host warns before model turn **28/32**, counting turns from one. Warnings
continue through turn 31 so the model can correct and submit before the last turn.
On turn 32 it is told to submit immediately rather than start another check.
The warning is guidance, not an automatic submission or an extra turn. The same
rule applies to recursive REPL sessions. Generic callers without this CuTe hook
keep their prior prompts and behavior. The historical two-feedback-call cap is
not used. No old experiment snapshot or result is changed by this protocol.
This feedback path is CPU-tested with scripted models/workers. Its new broker-to-GPU
path still needs a target-VM smoke under a new frozen run before a model evaluation.
The earlier generic GPU verifier precheck is not evidence that this new path ran.

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

The current target is **RTX PRO 6000 Blackwell with the CUDA 13.3 toolkit**. A
CUDA version printed by `nvidia-smi` is driver capability, not proof that the 13.3
toolkit and compiler are installed. Pin actual `nvcc`, Torch, CuTe and driver
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
