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
timing measures the forward call, after state/input reset and warmup. A device-wide
synchronization precedes the end event so work on other streams is not omitted;
this includes synchronization overhead and is not a pure single-kernel latency.
Compile/setup and process wall time are reported separately. Validate this timing
protocol on the target GPU before interpreting a speedup. This is
final-submission verification, not a model-visible intermediate test tool.

The GPU worker code is implemented, but its CuTe/driver execution is unvalidated.
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
