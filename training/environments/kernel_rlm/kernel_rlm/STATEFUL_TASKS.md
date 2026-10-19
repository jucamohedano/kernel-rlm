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

## CPU preflight is not integrated yet

The saved-candidate CPU diagnostic caught six of the eight old task-0018
interfaces without compiling an extension. It extracted candidate definitions,
used real CPU Torch state copying, and replaced the extension with an
argument-count stub. This proves that those specific errors can be caught
cheaply; it is not a general numerical verifier or a safe arbitrary-code loader.

The production verifier still compiles before checking candidate state and
runtime arguments. A future preflight needs a bounded, GPU-hidden subprocess
and an explicit unsupported result for constructors or bindings it cannot safely
analyze. Unsupported code must reach the existing verifier, not be falsely
rejected. Independent CUDA correctness and speed checks remain mandatory.

## Feedback deadline diagnostic

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
policy. This is follow-up infrastructure work; it is not deployed into the
unchanged three-update baseline block.
