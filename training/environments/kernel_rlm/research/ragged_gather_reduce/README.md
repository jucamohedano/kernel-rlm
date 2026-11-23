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

## VM qualification (commands require separate approval unless marked offline)

Use one full L40S for the author-control smoke. Qualify the unchanged BF16
RLM-27B on either an RTX PRO 6000 Blackwell 96 GB (`TP=1`) or two full L40S
devices in the same VM (`TP=2`), not two independent one-card rentals. RTX
serving does not change the L40S device-verification target. The model's BF16
tensors occupy about 51 GiB before runtime. Confirm driver/GPU/PCIe/NCCL, storage, and package pins on the
rented VM; do not switch dtype, tokenizer, task hardware, or model to make a
fit check pass. The checker uses a CUDA development toolkit with native
`sm_89` output. Do not run the smoke on a GPU currently serving the model.

1. **Offline:** from the repo root, invoke
   `python -m training.environments.kernel_rlm.research.ragged_gather_reduce.vm_smoke --preflight`.
   It hashes the task files and lists eight planned controls without querying
   a GPU. `offline_preflight` does not count as device evidence.
2. **Approved device smoke:** use the same module with
   `--run-author-controls --confirm-exclusive-disposable-l40s` and
   `--gpu-uuid GPU-…`. Supply `--nvcc`, `--host-cxx`, `--nvidia-smi` paths
   relative to the working directory and their corresponding `--*-sha256`
   pins from the reviewed VM image. Redirect the JSON receipt outside the
   checkout. Require all seven positives and the one exact negative. An
   unsupported device or inconclusive process result stops the pilot.
3. **Approved private serving check:** verify rights and materialize the exact
   model and tokenizer revision, using BF16 without quantization for this
   pilot. Use a private loopback endpoint, 32,768 tokens, no speculative
   decoding or reasoning parser, the pinned model's chat template, and actual
   device/startup/headroom receipts. The serving manifest declares `model`,
   `revision`, `served_model` (`rlm27b-f0b3b27db57c-bf16`), `dtype`
   (`bfloat16`), `gpu` (`RTX PRO 6000 Blackwell` or `L40S`),
   `tensor_parallel_size` (`1` or `2`, respectively), `max_model_len` (`32768`),
   and `endpoint` (`http://127.0.0.1:<port>/v1`). A matching manifest or
   `/v1/models` alias is not identity proof. Check exact weights, template and
   tokenizer parity on the running service.
   The proposed amd64 serving image is
   `vllm/vllm-openai@sha256:cf5b2fdd087eb4f6d261736967cc5cde974bd22ff4df50365f8fa419b6b4c484`
   (vLLM 0.34.0/CUDA 13.0.3; Linux driver ≥580.126.20 for that stack).
   Start with an offline, read-only snapshot: `vllm serve <snapshot>` with
   `--load-format safetensors --model-impl vllm --language-model-only`, the
   snapshot as `--tokenizer`, its `chat_template.jinja`, `--dtype bfloat16`,
   `--host 127.0.0.1`, the manifest alias/port, and the chosen TP/model-length
   pins above. Qualify with a short benign request and a near-limit prompt;
   require at least 4 GiB of measured headroom on every serving rank, stable
   response, and no fallback, OOM or Xid. For dual L40S verify both ranks and
   NCCL with `--disable-custom-all-reduce`. Flags and image are proposed,
   not live-qualified. Do not give the REPL direct access to the service.
4. **Approved one-attempt diagnostic:** use
   `python -m training.environments.kernel_rlm.research.ragged_gather_reduce.pilot`
   with `--run-approved-generation-only`, `--author-controls`,
   `--serving-manifest`, `--harness-checkout`, `--python-runtime`, and a new
   private `--out-dir`. Use the Harness-CG controller Python environment
   with the pinned `rlm_train` available before the legacy path. The API key
   is visible only to the trusted controller (`RAGGED_MODEL_API_KEY`). Set
   `RLM_TRAIN_SUBCALL_TIMEOUT_S=80` for that controller so native proxy calls
   expire before the isolated bridge deadline. The worker does not get the key. The script requests at most eight
   root turns, four child prompts, 2,048 output tokens per response, and a
   900-second wall cap, with no automatic retry. It writes the native visible
   trajectory, exact committed source bytes and receipts privately. It never
   executes model source. Verify the retained source/evidence off-VM and stop.

This is one **generation-only** diagnostic after trusted controls pass, not
scored evidence about an arbitrary model kernel. The trusted author driver
cannot be reused as an automatic scorer for unreviewed model output. The
result remains held; timing, candidate correctness and SFT are not part of
this qualification.

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
  output until a separately reviewed capture path is in place. The rootless
  REPL and its scoped child bridge pass scripted tests locally. Recheck the
  namespace/timeout behavior as a non-root user on the intended VM before
  connecting it to the private model service. The GPU candidate capture
  path is not provisioned by this change.
- **Approval:** pin exact model serving and the time/cost cap before any live
  model or device work. Exercise lifecycle and the source-only boundary with
  benign controls before opening a rollout. Private run records stay outside
  Git; no sandbox, rollout, training, or benchmark is authorized here.
