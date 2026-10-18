# SkyRL backend: rollout adapter first

Status: CPU-tested rollout integration. **Training launch is not enabled.**
This is an additional backend. Prime-RL, Modal and their checkpoints are unchanged.
No VM is rented by this package.

The [Laguna BF16/Muon gates](LAGUNA_VALIDATION.md) validate native model storage
and distributed optimizer components separately. These helpers are not yet
connected to a SkyRL/Ray training entrypoint.

The [native framework gates](FRAMEWORK_VALIDATION.md) connect the shared BF16/Muon
helpers to SkyRL's native strategy and TRL's native GRPO trainer. They test one
short synthetic update, separately from the RLM/Ray/vLLM training entrypoint.

## Why SkyRL

SkyRL supports shared training/inference GPUs and phase offload. That is a better
starting point for two 48 GB L40S cards than our pinned Prime launcher's separate
GPU allocations. This is not a measured speed or cost improvement.

Pin `skyrl-v0.4.0` at `8adc1e53465e2766ac8e8f5ec652e633d17e2279`.
Do not confuse this tag with `skyrl_train-v0.4.0`. Use its own Python 3.12+
environment and frozen upstream lockfile, not the Prime environment. The pin uses
Torch 2.13, vLLM 0.30 and PEFT 0.18.1. Driver and CUDA compatibility need a VM check.

## Calling flow

```text
SkyRL trainer (entrypoint connection still pending)
└── KernelGenerator.generate: one prompt, four trajectories
    ├── CompletionBridge: explicit LoRA adapter, exact tokens and logprobs
    ├── isolated program subprocess: existing kernel_rlm_v1.program
    │   └── existing persistent Python REPL worker
    │       ├── cuda_feedback: compilation and correctness feedback
    │       └── llm_query: optional text-only child through the same bridge
    ├── trusted final verifier: correctness and speed reward
    └── root-only per-turn training samples → SkyRL step-wise advantages
```

Root thoughts remain in the sampled tokens for loss. The existing program removes
them from later conversation history. Child calls are saved, but produce no loss
rows. REPL outputs are next-turn prompt tokens, not target tokens. Only the final
root turn receives terminal reward; SkyRL broadcasts the trajectory advantage.
Decoded text is never re-tokenized to manufacture training samples.

The adapter keeps whole groups only when their terminal rewards differ, including
speed. It can try at most three groups using supplied replacement tasks. Exhaustion
raises an error before an update. Do not also enable native dynamic sampling or
native zero-variance filtering: at this pin they operate on flattened response
rows, which is unsafe for our variable-length step-wise trajectories.

## Saved evidence

Each generation batch has a fresh directory. Every attempted trajectory keeps
`task.json`, raw root/child requests and responses in `calls.jsonl`, `program.log`,
`submission.json`, feedback JSONL, and `score.json`. `groups.json` records rejected
and admitted groups. Dropped groups are not erased. These files do not yet appear
automatically in the factory dashboard; launch capture/import is a remaining gate.
The loopback relay uses a fresh credential, does not save it, and removes common
provider credentials from the program environment. This is process separation,
not a security sandbox for hostile code.

## CPU preparation

From `rlm/`, with the pinned source already checked out:

```bash
PYTHONPATH=training/environments/kernel_rlm:training/src .venv/bin/python \
  -m kernel_skyrl --skyrl-source /tmp/kernel-skyrl-v040 \
  --dataset ../notes/curriculum/lvl1_seed42.jsonl \
  --output /tmp/kernel-skyrl-preparation.json
```

This command verifies the source pin, hashes dataset/code/config inputs and writes
a preparation report. It does not import Torch, start Ray, download model weights,
create a research experiment, call an API, or allocate a GPU. It refuses to overwrite
an existing output. A preparation is not a successful training run.

## Before full training on the two L40S cards

1. Connect the validated native Laguna loader to the SkyRL worker. The component
   check loaded the full pretrained base on two L40S GPUs; it was not a Ray run.
2. Keep BF16 **storage** for frozen weights and FP32 adapters. The shared helper
   enforces this distinction; the worker must actually call it.
3. Select the validated `LagunaMuonStrategy` in the policy worker. Its public
   preparation hook replaces the standard AdamW path with Moonlight RMS Muon;
   the unmodified upstream worker does not select that hook automatically.
4. Add compact checkpoint save/resume. Native FSDP saves frozen model shards too.
5. Connect the entrypoint and provider-neutral factory input capture.
6. On the VM, test adapter reload, offload, verifier placement and peak memory,
   then one bounded update. Only then compare throughput or learning quality.

Ideal BF16 base storage is approximately 31.1 GiB per GPU. A full-base synthetic
Muon update and four-token forward passed, with about 37.9 GiB observed NVML use
per GPU. Activations at training context length, KV cache, phase offload and
verification still need checks. Full training memory fit is unproven.
Inference and training must alternate; this is not the previous asynchronous
disaggregated topology. Changing frameworks also changes the loss implementation;
do not treat the first SkyRL run as a bitwise continuation of the Prime run.

Sources: [placement](https://docs.skyrl.ai/docs/tutorials/placement),
[agent integration](https://docs.skyrl.ai/docs/tutorials/agent-integration),
[pinned FSDP worker](https://github.com/NovaSky-AI/SkyRL/blob/8adc1e53465e2766ac8e8f5ec652e633d17e2279/skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py).
