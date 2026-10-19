# TRL compatibility check

## Shared-GPU verifier control

The live `kernel_trl.collect` command exposes the local pause/resume proxy on
port 8016 and exports `KERNEL_INFERENCE_ADMIN_URL` to its REPL workers and final
scorer. Model requests hold the shared inference admission lock. Exclusive
verification waits for active requests, sleeps vLLM, checks the kernel, then
wakes vLLM even if verification fails. Eligible intermediate checks retain the
existing guarded-overlap path; final timing remains exclusive.

The sigmoid run on 19 October exposed missing control wiring: five compiled
submissions exhausted verifier memory while inference stayed resident. CPU
regressions cover the fix; a live L40S sleep/wake and correctness replay is still
required before claiming the OOM issue is solved. Historical gates follow.

TRL is a candidate, not yet a Laguna training backend. Keep this separate from
the Prime and SkyRL environments. Pin v1.9.2 at
`33f9e462728b98f7f91d38b99328e81adde2faa0`.

The later [two-GPU native framework gates](../kernel_skyrl/FRAMEWORK_VALIDATION.md)
test BF16 Laguna, Accelerate FSDP2 and real Dion Muon through `GRPOTrainer.train`.
That is separate from the earlier CPU/SGD injection check described below. It
does not yet port the RLM's changing-prefix, multi-turn credit to TRL.

The [single-policy phase gate](PHASE_VALIDATION.md) then tests native Transformers
generation, two Muon updates and CPU offload/reload on the same FSDP2 model.
It avoids a second inference copy; it does not validate vLLM colocation.

The real CPU probe uses an in-memory random one-layer Llama, local tokenizer,
attention LoRA rank 8/alpha 16 and four synthetic rewards. It calls TRL's custom
rollout hook, verifies feedback-token masking, injects an optimizer, and completes
one optimizer step. Four adapter tensors changed. No pretrained weights, model
API, vLLM engine, CUDA kernel or GPU is used. This does not measure learning quality.

The injected optimizer is **SGD solely to test the hook**, not our project optimizer.
That CPU probe did not validate pinned Dion Muon: its CPU import requires Triton even when Triton
execution is disabled. No AdamW or SGD replacement is proposed for project training.

## Reproduce in the isolated CPU-check environment

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=training/environments/kernel_rlm /tmp/kernel-trl-cpu-venv/bin/python \
  -m kernel_trl.cpu_probe --trl-source /tmp/kernel-trl-v192 \
  --work-dir /tmp/new-trl-work --output /tmp/new-trl-report.json
```

The check requires TRL 1.9.2. The measured environment used CPU Torch 2.14.1,
Transformers 5.18.0, Accelerate 1.12.0, datasets 4.7.0 and PEFT 0.18.1.
These are diagnostic versions, not a validated CUDA training image. Disable
dependency resolution when reusing the installed CPU Torch; otherwise a package
resolver can download CUDA Torch and NVIDIA libraries unnecessarily.

## What the earlier CPU check established

- **Yes for optimizer injection:** TRL accepts an optimizer directly. SkyRL's
  pinned FSDP path constructs AdamW internally and needs a worker extension.
- **Yes for simple rollouts:** custom completions and environment masks work.
- **Not yet for this RLM:** TRL's normal batch has one prompt/completion sequence
  per trajectory. We remove old thoughts between turns. Later sampling prefixes
  therefore are not necessarily extensions of earlier ones. A loss mask removes
  target loss; it does not remove those tokens from attention context. A correct
  port needs step-wise samples and trajectory-level GRPO credit, not concatenation
  or decoded-text re-tokenization. The CPU probe includes a synthetic counterexample.
- **Not automatically for two L40S:** vLLM sleep removes inference weights during
  training. The standard path does not also offload the trainer during inference.
  Both BF16 base copies would need roughly 62 GiB per GPU before overhead. FSDP
  CPU offload is a possible alternative, but Muon/offload and reload need tests.
- **Weight sync matters:** the pinned TRL PEFT path merges adapters and syncs base
  weights. That is different from adapter-only reload; it needs a memory/transfer
  check for Laguna before billing for a full-model run.

Next: continue TRL first, as requested. Keep SkyRL as another option. Both passed
short native full-model backward/Muon gates; TRL also passed native generation
and manual CPU phase moves with two updates. Native decoding was slow in that
short test. Exact per-turn RLM credit, faster inference, realistic context and
checkpoint restart still need validation before full RLM training.

Sources: [custom rollouts](https://huggingface.co/docs/trl/grpo_trainer),
[vLLM integration](https://huggingface.co/docs/trl/vllm_integration),
[pinned sync implementation](https://github.com/huggingface/trl/blob/33f9e462728b98f7f91d38b99328e81adde2faa0/trl/generation/vllm_generation.py).
