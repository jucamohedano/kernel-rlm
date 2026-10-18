# Native framework gates: SkyRL and TRL

These are bounded maintenance checks on the existing two-L40S CloudRift VM.
They are not kernel-generation experiments. Modal and Prime remain unchanged.

## What the code connects

`framework_hooks.py` supplies two public hooks:

- **SkyRL:** a native `FSDPStrategy.prepare` extension. Upstream loads/shards the
  supplied BF16 policy but does not construct its default AdamW. The extension
  creates distributed Dion Muon and the requested scheduler instead. Native
  backward, gradient clipping and optimizer-step methods remain unchanged.
- **TRL:** a native `GRPOTrainer.create_optimizer` extension. It creates Muon
  before Accelerate FSDP2 preparation. Accelerate remaps optimizer parameters to
  the sharded DTensors. A step hook verifies that Muon actually receives those
  DTensors, finite nonzero autograd gradients and only FP32 attention adapters.

Both use `laguna_policy.load_base` and `attach_adapters`: frozen BF16 base,
attention LoRA rank 8 / alpha 16, FP32 adapters, Moonlight RMS Muon, learning rate
`1e-5`, momentum `0.95`, decay `0.01`. There is no AdamW fallback. A full-model
check must retain all 320 adapter tensors / 8,192,000 trainable parameters.

`framework_probe.py` runs one optimizer update per process group:

1. Start exactly two distributed CUDA processes.
2. Build a tiny random Laguna, or load the cached full pretrained checkpoint.
3. Pass it through the selected native framework, not the earlier component-only
   sharding helper.
4. Use four short synthetic samples with rewards `[0, 1/3, 2/3, 1]`. Mask the
   middle observation token out of the target loss. SkyRL uses native GRPO
   advantages, PPO loss, model wrapper and strategy. TRL calls native `train()`
   with its custom rollout and environment-mask API. Its fixture logprobs come
   from a real current-policy forward pass.
5. Verify real gradients, a finite changed adapter set, optimizer class and base
   storage. Save per-rank reports, failures, source copies and stage events.
6. Exit without saving a trained checkpoint. Discard the synthetic updates.

Initial LoRA B matrices are zero. Nonzero first-step gradients are expected in
B, while A can still change through weight decay. A changed tensor alone is not
evidence of task learning. The fixtures and native auxiliary-loss handling differ;
this is not an objective-parity or speed comparison between frameworks.

## Pins and the SDPA compatibility patch

- SkyRL `skyrl-v0.4.0`: `8adc1e53465e2766ac8e8f5ec652e633d17e2279`.
- TRL `v1.9.2`: `33f9e462728b98f7f91d38b99328e81adde2faa0`.
- Shared core: Torch `2.13.0+cu130`, Transformers `5.16.1`, PEFT `0.18.1`,
  Accelerate `1.15.0`. This is a minimal trainer-check runtime, not a complete
  production dependency lock for both frameworks.

SkyRL's model-wrapper module eagerly imports FlashAttention padding helpers even
when padded SDPA is selected. `enable_lazy_padding_import` changes only that
import in a private checkout. The helpers still import real FlashAttention when
called; no fake implementation or substitute module is supplied. The function
requires the exact original import once, and records original/patched SHA-256.
Do not apply it to a different revision or claim the checkout is unmodified.
The CPU preparation launcher still rejects modified upstream tracked source.

TRL enables gradient checkpointing by default. These checks explicitly disable
that option and enable FSDP activation checkpointing instead; enabling both makes
Transformers reject the configuration before training.

The first full TRL attempt stalled after rank-0 loading and was stopped. Unlike
the first tiny fixture, the full model creates rank 1 on meta. Accelerate 1.15.0
restores each rank's local nonpersistent buffers with `.to(device)`, without
broadcasting the real data first. Four tiny RoPE buffers on rank 1 have no data
to copy. A CPU-only full-config reproduction confirms that error. The loader now
broadcasts those buffers from rank 0 before native Accelerate preparation. No
frozen weights are copied eagerly to GPU. The corrected tiny test also creates
rank 1 on meta, so it covers the actual loading pattern rather than both-rank CPU
initialization. Keep failed and corrected attempts separate in the evidence.

That meta-initialized tiny check exposed a second issue: Accelerate remaps
optimizer parameters by their old storage pointers. All meta tensors have pointer
zero, so an optimizer constructed from those adapters can map multiple entries
to the same sharded parameter on rank 1. The loader now gives only the FP32 LoRA
adapters distinct empty CPU storage before optimizer construction (32.8 MB for
the full model). Rank 0 still supplies all actual values through native FSDP
broadcast; frozen base weights remain meta on rank 1. The step guard requires
exact identity coverage of every trainable adapter, with no duplicate entries.
Host-side exceptions are recorded before distributed cleanup, and a failing rank
exits promptly so torchrun can stop its sibling instead of waiting in cleanup.

## Bounded invocation

Use a private source upload and the same CPATH documented in
[LAGUNA_VALIDATION.md](LAGUNA_VALIDATION.md). Never run both frameworks at once.
Start with a fresh tiny output directory, then a fresh full-model directory only
after the tiny gate succeeds:

```bash
timeout --signal=TERM --kill-after=5s 600s python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 --max_restarts=0 \
  -m kernel_skyrl.framework_probe trl --output-dir /fresh/trl-tiny
timeout --signal=TERM --kill-after=5s 600s python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 --max_restarts=0 \
  -m kernel_skyrl.framework_probe skyrl --checkpoint /cached/pinned/checkpoint \
  --output-dir /fresh/skyrl-full
```

Replace the backend argument to test the other path. Capture one-second NVML
telemetry externally. Keep every failed attempt separate. Download only small
reports, telemetry, logs and exact sources, never the base weights.

## Still outside these gates

- SkyRL Ray policy-worker initialization and complete trainer orchestration.
- vLLM generation, phase offload/onload and adapter reload.
- Real RLM multi-turn rollouts, child exclusion and changing-prefix loss credit.
- CUDA verifier execution and contention during inference.
- Compact checkpoint save/restart and realistic-context VRAM/throughput.
- Quality, reward improvement and a production GRPO run.

TRL's normal single-sequence trajectory does not preserve our removed old-thought
prefixes. Its native environment loss mask is not a solution to that attention
context mismatch. Keep the per-turn adapter gate explicit before porting RLM.

Results and exact executed sources are kept in the parent project's
`notes/diagnostics/20261018-framework-trainers/`. A passing short backward gate
must not be described as complete two-GPU training readiness.
