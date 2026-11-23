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
python -m pytest -q tests/test_ragged_gather_native.py tests/test_ragged_gather_reference.py
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

## Device smoke (not authorized by this change)

`reference.py` computes gold sums and counts from logical rows, independently
of the offset/mask encoding. The checked-in cases are authoring controls, not
hidden grader cases. `device_check.cu` is a device comparator for trusted
controls and an optional timing probe. `control.cu` is a candidate same-device
comparison point; it is not yet a measured or qualified performance baseline.
`wrong_offset.cu` is a deliberate near-twin negative control. Run the negative
only on a discriminator where all gathered coordinates are safe even if the
wrong mask is used. Do not run it on a general valid-input suite that may have
unusable coordinates behind a cleared mask.

The smoke plan is: compile for `sm_89`; run the good source on the developer
cases; require a numerical mismatch (not a crash/build failure) from the
wrong-offset control on its dedicated discriminator. Only then check latency
on short and skewed regimes. Warm up and compare both implementations on the
same device and stack, with the same ABI and timing scope. Report exact GPU,
source hashes, toolkit/driver, inputs, caps, and per-regime receipts. The
driver's repeated hot-case event timings are diagnostic, not evidence of a
competitive kernel or a qualified baseline. Do not report an L40S speed claim
without that separate measurement and baseline qualification.

### Before any model rollout

The upstream local REPL subprocess is **not** a sandbox. The supplied device
driver links a candidate and a comparator into one address space and is not a
security boundary. Do not run model-generated Python or CUDA in the author
checkout or against privileged files. Live deployment still needs a reviewed
isolated REPL and a separate trusted scorer, with only committed source
crossing the boundary. Provision no shared credentials, reference material,
private test cases, or outputs in the model workspace. Any public probes must
be deliberately scoped and use a separate execution lane. Private run records
stay outside Git. A CUDA pass needs exact-device evidence; the approval for
that run, checkpoint serving and cap are separate from this task change.
