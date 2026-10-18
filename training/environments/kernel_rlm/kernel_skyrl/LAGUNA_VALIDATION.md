# Laguna BF16 and Muon component gates

These checks run only on the allocated CloudRift VM. They do not launch GRPO,
inference rollouts, Ray workers or vLLM. Prime and Modal are unchanged.

The full pretrained two-GPU component check passed on 18 October 2026. It used
BF16 frozen weights and all 320 FP32 attention LoRA tensors, with a real Dion
synthetic-gradient update and finite four-token logits on each rank. Maximum
observed NVML use was 37.9 GiB per GPU (one-second samples); PyTorch's own peak
allocation was 36.3 GiB and peak reservation 37.2 GiB. These are different metrics.

The later [native framework gates](FRAMEWORK_VALIDATION.md) extend this component
check to actual SkyRL strategy/loss and TRL GRPO trainer paths. Keep the earlier
synthetic-gradient evidence separate from those real-backward diagnostics.

## What changed

`laguna_policy.py` provides reusable loader and optimizer functions:

- `load_base`: use native Transformers Laguna, CPU loading, BF16 storage and
  strict missing/unexpected/mismatched-key checks. Do not execute remote code.
- `attach_adapters`: attention-only rank 8 / alpha 16; FP32 LoRA masters and
  frozen BF16 base parameters. Reject other trainable weights.
- `audit_storage`: check actual parameter storage dtype, not merely compute dtype.
- `build_muon`: pinned Dion Muon, momentum 0.95, learning rate `1e-5`, decay
  `0.01`, Moonlight `rms_norm` scaling and a one-dimensional FSDP mesh.

All our trainable parameters are matrices. Non-matrix base parameters are frozen,
so no AdamW fallback group is needed for this baseline. The helper rejects an
unexpected trainable vector or MLP adapter rather than silently changing policy.

Transformers 5.16.1 has a native conversion path for the old Laguna checkpoint
layout: per-expert gate/up/down tensors become fused expert tensors, router bias
moves to the native router, and shared-expert names change. The CPU fixture checks
every tensor bit-for-bit, including double-digit expert IDs. No new converter or
random initialization of missing pretrained weights is accepted.

## Stages and meaning

1. **CPU fixture:** tiny random legacy-format checkpoint, exact conversion,
   BF16/FP32 storage, finite logits. No GPU is needed.
2. **Tiny two-GPU fixture:** native FSDP2 load/broadcast, real backward pass,
   gradients only for adapters, real distributed Dion update, finite logits.
3. **Full checkpoint:** download the pinned pretrained revision first; load on
   CPU rank 0, initialize rank 1 on meta, shard before CUDA allocation, and
   broadcast parameter shards. Confirm 33,442,617,088 frozen parameters and
   320 adapter tensors / 8,192,000 adapter parameters. Exercise Muon with synthetic
   gradients, then one four-token forward. Save peak CUDA allocation/reservation.

The full-model synthetic update is a wiring test, not a policy-learning update.
It does not establish full-model backward performance, 12k-token activation fit,
reward signal, quality improvement, phase offload, adapter reload or resume.
Native component success is not proof that the SkyRL/Ray trainer is connected:
its default FSDP worker still needs our loader/optimizer hooks.

## Runtime

Core pins match the selected SkyRL components: Torch `2.13.0+cu130`, Transformers
`5.16.1`, PEFT `0.18.1`, Accelerate `1.15.0`, Triton `3.7.1`, Hub `1.32.0`,
safetensors `0.8.0`. Dion is pinned to
`d891eebbd950a6ff11d9b5f806282e33df83df9d`.
This is a private component-check environment, not the complete frozen SkyRL image.
Keep its dependency freeze with the reports.

The VM's system Python lacks development headers. Triton's imported host helper
needs `Python.h` even though Muon uses its Torch Newton–Schulz implementation.
Matching Ubuntu `libpython3.12-dev` was downloaded and extracted privately,
without sudo or system installation. Pass its include directories through CPATH.

```bash
export PYTHONPATH=/path/to/uploaded/source
export CPATH=/path/to/python-headers/usr/include/python3.12:/path/to/python-headers/usr/include:/path/to/python-headers/usr/include/x86_64-linux-gnu/python3.12
export OMP_NUM_THREADS=4
timeout --signal=TERM --kill-after=5s 120s python -m kernel_skyrl.laguna_probe \
  cpu-fixture --output-dir /path/to/fresh/cpu-report
timeout --signal=TERM --kill-after=5s 180s python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 --max_restarts=0 \
  -m kernel_skyrl.laguna_probe distributed --output-dir /path/to/fresh/tiny-report
timeout --signal=TERM --kill-after=5s 600s python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 --max_restarts=0 \
  -m kernel_skyrl.laguna_probe distributed --checkpoint /path/to/pinned/checkpoint \
  --output-dir /path/to/fresh/full-report
```

The full checkpoint is approximately 66.9 GB. Keep it on the VM, never on the
laptop or in Git. Download revision `a397bde04501591d0ac3d74a5d9d0a4ec298ffb8`.
Use a bounded download and a fresh report directory. No automatic GPU retries.
The probe saves source snapshots, hashes, stage events, per-rank reports and
separate failure records. FSDP may leave root parameters unsharded after forward;
memory reporting handles those ordinary tensors explicitly.

Stop only the owned diagnostic processes on timeout. A completed process does not
stop VM billing. Preserve existing Modal checkpoints and the immutable base files.
