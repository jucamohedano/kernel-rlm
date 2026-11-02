# Tool-enabled CuTe comparison

This experiment compares the pinned Laguna XS-2.1 and NVIDIA Nemotron 3.5 Lightning
BF16 checkpoints on the same 12 public CuTe tasks used in the tool-free pilot.
It has 24 independent root sessions per model: `api` and `api_example`, one root
per task/condition. There is no `task_only` arm. Do not combine these results with
older runs or describe this as an equal-budget tool-free comparison.

## Model interface

Each root receives simple CuTe instructions plus the existing orchestrator
addendum, adapted so delegation is available rather than forced. The task,
API notes, and optional worked example remain those of the tool-free pilot.
Only the tool/submission instructions change. Root and depth-one children have
separate persistent CPU REPLs, the same condition's resources, and the injected
`compile_candidate` helper. Children cannot spawn grandchildren. The API-only
arm does not gain a worked example through a child or filesystem mount.

The reasoning namespace has no GPU devices, project checkout, task oracle, or
full documentation corpus. Actual namespace/source-binding checks are required
before model calls; CPU tests alone do not prove a remote sandbox is sound.

`compile_candidate(source)` invokes the isolated GPU worker on all four public
cases. It reports compile/forward errors, not numerical correctness or reference
values. It saves source, diagnostics, and compiler artifacts privately. Generated
code may print its own values; this is not an oracle channel. Compiling, returning
from forward, observed CuTe invocation, and correct computation are separate facts.

Final submission uses `answer["content"] = best_source` and
`answer["ready"] = True`. Incomplete tool arguments are never executed or repaired
by the host. Final sources are syntax-checked, not imported on the host.

## Frozen limits

The same limits apply to both models:

- BF16, 262,144-token server context, thinking and selected-token logprobs enabled.
- At most 16,384 generated tokens per model call, including reasoning/tool arguments.
  Root, child, and plain helper calls share this cap. Context and tree remainder
  can reduce it. This is a new fixed cap, not the old 90%-of-context policy.
- Temperature 0.6, top_p 0.95, seed 20261101 with task-paired seeds.
- 800-second model requests and 1,800-second total root-tree episodes.
- Shared 1,000,000 generated-token ceiling per root tree; 64 turns per session,
  32 child admissions, 8 live children, 256 model calls, 2 concurrent requests.
- Isolated GPU phases retain the existing 600-second worker deadline.

The tree/turn/child guards retain the earlier recursive experiment settings;
the model request deadline and fixed per-call output cap are explicit changes.
The 24 sequential roots have a maximum of 12 generation-hours per model,
excluding setup, final grading, and backup. At the currently listed $0.63 per
L40S-hour, four GPUs for both models in parallel would cost up to about $30.24
for generation at this time ceiling. This is not a total invoice cap or estimate
of actual duration.

## Execution and lifecycle

Freeze all prompts, tasks, resources, source inventory and config hashes before
calls. Keep raw SSE, partial failures, root/child traces, helper diagnostics,
compiler artifacts, usage and checkpoints. A model/response failure ends only
that root; infrastructure, persistence, source-binding or cleanup failures abort
the run. No automatic retries/resume or extra root sessions are authorized.

Stop each model server before independent final numerical grading. Bind the stop
receipt to the run/model/generation nonce and actual owned process/port/GPU
observations. Use the unchanged four-case ICL grader. Numerical and CuTe evidence
still require source review; passing four cases is not general-domain proof.
There is no speedup/timing/Nsight scoring in this experiment.

## Resource boundary limits

Compiler workers necessarily expose the installed Torch/CuTe runtime to candidate
Python. This is not protection against deliberate introspection of installed
library internals. The project oracle, other task definitions, and project
API/example assets are excluded from the candidate helper package, and all model
feedback remains subject to source review. Do not claim malicious-code attestation.

The operator checks CloudRift L40S availability every 20 minutes. Prefer four
GPUs split two per model, in one four-GPU VM or two two-GPU VMs, with at most four
owned GPUs total. If only two are available, run both models sequentially on one
two-GPU VM. Do not rent until CPU readiness is complete. Remote GPU/tool/sandbox
preflight is required before experimental roots, including compile feedback while
the model server is loaded. On a four-GPU VM, each worker receives only its owned
model pair's device bindings and an explicit trusted `CUDA_VISIBLE_DEVICES` value;
the two model servers use disjoint pairs. Device UUIDs and visibility must be
checked on the actual VM before roots start.

After each VM's assigned work: transfer and verify the complete backup, delete
only the owned VM and disks through the provider's supported lifecycle, terminate
the rental, and confirm compute resources released. Termination alone may retain
a saved disk; a deletion acknowledgement must be tracked separately. Never touch
unrelated resources or silently replace L40S with another GPU.
