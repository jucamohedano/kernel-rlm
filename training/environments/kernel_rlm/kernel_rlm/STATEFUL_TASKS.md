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
