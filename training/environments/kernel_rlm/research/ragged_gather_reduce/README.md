# Ragged gather–reduce development task

Local research task for [Kernel RLM #43](https://github.com/jucamohedano/kernel-rlm/issues/43).
This is a real CUDA C++ development problem, not a synthetic environment,
training set, PMPP-Hard item, or a sealed transfer test.

## Model-facing interface

Use Harness-CG at `5272afdcb25a156c1b9b89cb479ae0604af249d7` and
`a1zhang/RLM-27B@f0b3b27db57c211234cb55ec512d87f45dabdc4c`.
`task.py` embeds only `contract.md` and `contract.h` in the native `context`
string and provides the short `root_prompt`. Use the pinned depth-1 Python REPL
and its normal `repl` blocks, stdout, and optional `llm_query` /
`llm_query_batched`. The upstream depth-1 `rlm_query` aliases are not a second
recursive planner. Set `answer["content"]` to raw CUDA source and then set
`answer["ready"] = True` in Python. Don't wrap it in `CUDA_SOURCE` or provide a
PyTorch extension. Do not add strategy hints, a forced solve/review route,
fake children or an automatic final response at the turn limit.

The task adapter builds an upstream `RLMTrainEnv` row, accepts an explicit
worker backend and caps from the controller, and exports only the committed
`rlm_final_answer` as exact UTF-8 bytes plus SHA-256 (64 KiB maximum). The
adapter does not set up an inference endpoint or judge. There is no live
compile/run tool in this first adapter. REPL actions and the final artifact
are separate: the offline tests use fixed mock turns, not model inference.
No changes to the historical inline-PyTorch environment or the Harness-CG fork
are needed for the adapter.

## Local/offline checks

From the repo root, using its project Python:

```sh
python training/environments/kernel_rlm/research/ragged_gather_reduce/task.py
python -m pytest -q tests/test_ragged_gather_{native,reference,score}.py
```

Set `HARNESS_CG_CHECKOUT` to the unchanged pinned checkout to enable the
additional offline subprocess check of the actual upstream REPL worker. To
also test the full `RLMTrainEnv` using scripted turns, use the Harness-CG
project environment with `datasets` and `verifiers`, and put its `src/` and
its pinned `rlm` dependency on the import path before Kernel RLM's legacy
`training/src`. With pytest, use `-o pythonpath=` to disable the legacy path
in Kernel RLM's pytest configuration. The integration test checks which
implementation was imported. A skip does not establish native behavior.
CPU or object-build checks are not GPU evidence.

## Capture and score (not a live runner)

`reference.py` builds expected sums and counts from logical rows, separately
from offset/mask encoding. `WireFixture.to_wire()` encodes only the device
inputs: expected outputs never enter the candidate process through this wire.
Checked-in fixtures are authoring controls, not hidden scorer inputs.

`device_check.cu` links one trusted author control, invokes it twice with
different output initializers, checks input and guard integrity after each
invocation, and emits four observed arrays. It does not see gold or say
whether the numerical result passed. Outside that process, `score.py`
checks the capture status/shape and compares both observations to the CPU
reference. No device capture in this patch provides timing. Capture output
and source receipts would need to be bound by the controller in a live run.

Future device smoke: compile these author controls for `sm_89`. The known-good
control must match on the development cases. The wrong-offset control must
produce a numerical mismatch on the dedicated `mask-offset` case. Do not run
that negative on fixtures with out-of-range dropped IDs. After smoke, qualify
a separate timing path and a baseline on the same device and stack, with the
same ABI and timing scope. Report exact GPU, source hashes, toolkit/driver,
inputs, caps and receipts. No device correctness or performance claim follows
from the offline checks.

## Deployment gates

- **REPL:** launch the unchanged pinned worker in a fresh isolated context, not
  the author's checkout. Preserve the upstream JSONL turn protocol, private
  persistent state, native answer handoff and optional child calls. Only the
  public task packet crosses into the REPL. Keep credentials, scorer assets,
  candidate compilation and evaluator filesystem out of the worker; use a
  narrow per-attempt child-call channel rather than general network access.
- **CUDA:** transport only exact committed `.cu` source out of the REPL.
  Compile and execute in its own disposable restricted GPU environment with
  inputs and the ABI, never with the further authority to choose correctness.
  The current C++ runner is for trusted controls only. Because it links a
  candidate in one address space, its output is not trustworthy evidence from
  arbitrary model code. Splitting gold and comparison into Python does not fix
  that boundary. Do not send private comparison data or trust a candidate's
  output until a separately reviewed capture path is in place. Neither the
  REPL sandbox nor that GPU capture path is provisioned by this change.
- **Approval:** pin exact model serving and the time/cost cap before any live
  model or device work. Exercise lifecycle and the source-only boundary with
  benign controls before opening a rollout. Private run records stay outside
  Git; no sandbox, rollout, training, or benchmark is authorized here.
