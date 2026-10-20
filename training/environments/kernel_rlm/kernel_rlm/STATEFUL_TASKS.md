# Stateful CUDA tasks

The first L1 curriculum includes constructor constants, initialized parameters and
structured outputs. A runtime input list alone does not contain all of that state.

## Submission entrypoints

- Use `cuda_forward(*inputs)` for stateless operations. If a task has a visible
  fixed constructor constant, the wrapper can pass that constant to the extension.
- Use `ModelNew(torch.nn.Module)` for constructor-dependent or weighted operations.
  Match the reference constructor arguments and its parameter/buffer names and
  shapes. Its `forward` method must call the compiled CUDA extension.
- Both paths still define `CPP_SOURCE` and `CUDA_SOURCE` and build the extension
  with `load_inline` at import time. `ModelNew` takes precedence if both exist.
- `nn` modules may register state; their PyTorch numerical forward is not the
  implementation being requested. The existing extension-use guard still applies.

## What the verifier does

1. Build the reference from the trusted `get_init_inputs()` result.
2. Run it in evaluation/no-grad mode. This also materializes lazy parameters.
3. Build `ModelNew` with an independent copy of those constructor arguments.
4. Copy the reference state with strict key/shape matching; do not pass the
   reference model or its forward function into the candidate.
5. Run the candidate on separate cloned inputs. Check outputs, dtypes, device,
   nested tuple/list/dict structure and input-mutation behavior.
6. Require a tracked extension call in every correctness trial. Benchmark only
   after correctness passes, as before. Reward coefficients are unchanged.

Input and state copies prevent ordinary tensor aliasing between candidate and
reference. They do not turn in-process Python into a hostile-code sandbox. The
model must still obey the no-reference-execution and CUDA-only computation rules.

## Evidence

`kernel_rlm.candidate_probe` checks real CPU Torch state copying, strict mismatch
rejection, tensor independence and nested/dtype/nonfinite/complex outputs with
GPUs hidden. CPU unit tests also retain the old function entrypoint.

The real L40S check uses `examples/stateful_linear_submission.py`: three trials
with independently initialized reference weights pass, with maximum error about
`3e-7`. It returns a tensor plus nested shape metadata. A reference that adds one
to the result is rejected with an error about one, using the same cached extension.
This verifies the constructor/structured-output path, not model-generated kernel
quality. Benchmark calls now disable autograd for both reference and candidate.

## CPU contract preflight

The saved-candidate CPU diagnostic caught six of the eight old task-0018
interfaces without compiling an extension. It extracted candidate definitions,
used real CPU Torch state copying, and replaced the extension with an
argument-count stub. This proves that those specific errors can be caught
cheaply; it is not a general numerical verifier or a safe arbitrary-code loader.

New source runs `contract_preflight` before compilation. It uses meta tensors in
a fresh GPU-hidden subprocess, with a wall deadline, 10 CPU seconds and a 16 GiB
address-space limit. Simple `load_inline(functions=[...])` bindings are replaced
with argument-count stubs. Strict state keys/shapes, the Python wrapper signature
and supported C++ binding arities can fail before a CUDA build.

Defaulted or ambiguous C++ signatures, unsupported imports, data-dependent meta
operations and preflight resource failures fall back to the full verifier. A
positive preflight never establishes numerical correctness or extension use.
Independent CUDA correctness and speed checks remain mandatory. This bounded
subprocess is not a hardened hostile-code sandbox.

## Baseline feedback deadline diagnostic

Compilation, GPU admission and execution currently share a 120-second feedback
deadline. A 54-second compile leaves 66 seconds for both admission and execution.
An already-active model request can exceed that remaining allowance even after
the duplicate-reader lock bug is removed. Writer priority blocks new readers;
it does not preempt requests that already hold a reader lock.

CPU tests reproduce the remaining-budget calculation and a timeout behind one
active reader without launching CUDA. Recorded relay times had a median of
87 seconds and a maximum of 209 seconds, but those aggregates do not establish
exact overlap for every observed timeout. A queue timeout means unknown
correctness, not a failed numerical test.

Do not increase only the feedback deadline. The base REPL and backend limits are
270 and 300 seconds. With two enabled 90-second child calls, this baseline
reserves another 180 seconds, giving 450 and 480 seconds respectively. Two
feedback calls, any child calls and cleanup must still fit those limits. Queue,
compile, execution, wake cleanup and episode deadlines need a coordinated budget
policy. These numbers describe the immutable completed three-update baseline;
its samples and source have not been changed.

## Repaired scheduling and cancellation

- The loopback completion bridge owns inference admission for bridged root and
  child requests. The program does not acquire a second reader lock. Direct
  non-bridge integrations retain their caller-owned admission.
- Disconnects cancel the bridge's transport coroutine. Its reader lock remains
  held through coroutine cleanup. Cancelled child calls are logged and excluded
  from root samples; cancelled root calls still reject incomplete collection.
- Exclusive intermediate feedback announces writer intent before CPU compilation.
  Active inference can finish while the CPU builds; new requests wait. CUDA
  execution still requires the exclusive GPU slot. Qualified guarded overlap
  retains its separate eligibility checks; final benchmarks stay exclusive.
- Every queue/cache-lock wait reduces the same absolute verification deadline.
  Queue timeout still means unknown correctness, not an incorrect kernel.
- Feedback has a 180-second deadline, plus a 30-second cleanup allowance per call.
  Two feedback calls fit a 450-second base REPL and 480-second backend. Two
  optional 90-second children raise those enclosing bounds to 630/660 seconds.
  Turn/output/context caps and the outer group deadline are unchanged.

This policy bounds waiting; it cannot guarantee that every long active inference
request drains before a feedback deadline. CPU cancellation and lock tests are
not measurements of live GPU abort latency, throughput or numerical correctness.
Use a new source snapshot for GPU validation; do not resume the old immutable run
with edited code.

## BatchNorm verification protocol: batchnorm-state-v2

Standard BatchNorm1d/2d/3d and SyncBatchNorm modules receive reproducible,
nondefault affine parameters and running statistics before every correctness
trial. Running variance stays positive; absent parameters/buffers, input shapes,
evaluation mode, numerical tolerances and reward formulas are unchanged. Other
module types and state are not changed. A dedicated CPU generator derives its
seed from the protocol, module path and trial without changing the global RNG.

The reference evaluates this state, and the existing strict state handoff copies
it to ModelNew. Every numerical trial must pass before benchmarking. Timing uses
the same BatchNorm state as the last successful trial, not fresh default state.
Results record the protocol, challenge seeds/state digests and benchmark state.
This catches the saved identity-return shortcut even for a one-trial feedback
check; compiling/calling an extension alone is still not proof of a CUDA launch.

The RTX driver requires matching protocol tags on cached submitted candidates.
Legacy results fail closed with an explicit revalidation requirement; no original
scores, attempts, trajectories or checkpoint history are rewritten. A new scorer
returning a wrong protocol is durably recorded as an attempt before failing.
Revalidation must use a separate evidence directory and immutable source revision.

This is expanded numerical coverage, not a hardened security boundary against
arbitrary hostile Python or a guarantee of generalization over all module states.
The deterministic challenge seeds are reproducibility metadata, not hidden tests.
It does not repair the separate task0037 timeout, undo the update at checkpoint28,
or authorize restarting training from a different checkpoint.

An operator-approved repair branch can be prepared with
`python -m kernel_trl.reward_repair`. It verifies the unchanged 50-task dataset,
parent lineage, base checkpoint hashes/native readback, exact saved generation,
revalidation score hashes/reward formula, and matching verifier runtime source.
It copies only collection artifacts, never the discarded descendant's update or
audit outputs. Original scores and ledger are retained alongside versioned scores;
the new ledger adds the explicit revalidation without resetting old attempts.
One operator-selected pending infrastructure group may be deferred with a reason
and original manifest/ledger hashes. This is not automatic weak-group filtering.

The prepared new directory uses the driver's existing `--resume` path. The parent
directory and all previous checkpoints remain immutable. The new branch inherits
only updates through its chosen base, and must pass fresh probability audit,
optimizer restore and native readback before accepting its rebuilt update.

## Hardware-scoped CUDA documentation

Generate a new collection with `python -m kernel_rlm.cuda_docs --download DIR
--target-profile sm120`. It retains pinned CUDA13.3 programming, best-practices
and nvcc references, adds the NVIDIA Blackwell tuning/compatibility guides, and
does not include Ada as the target tuning guide. Original HTML, URLs, notices
and SHA256 hashes remain in the private cache. Never overwrite a corpus used by
an immutable run: a new collection requires a new checksum and source receipt.

The RTX driver requires an SM120-profiled corpus and checks the recorded docs
metadata on resume. The root prompt verifies this profile against the supplied
`TORCH_CUDA_ARCH_LIST=12.0` (or12.0+PTX), identifies RTXPRO6000/CC12.0, and warns
that SM100/B200 features or compiler flags do not automatically apply to SM120.
Every search/read result carries the target scope. Generic archived examples
are reference material, not permission to override the job's compiler target.

Both child types receive the same hardware context in their user assignment,
within their existing input-token cap. Their system identities, bridge routing,
generated-token budgets and exclusion from root loss are unchanged. This is a
prompt/retrieval correction, not a security mechanism against a candidate that
deliberately changes environment variables or compiler flags.

Stage new source and docs while a healthy old driver runs; do not hot-patch it
or relabel its existing trajectories as using the new prompt. Activate only in
an explicitly recorded, checkpoint-safe immutable continuation with the original
spending baseline, source/docs pins and ordinary safety gates preserved.
