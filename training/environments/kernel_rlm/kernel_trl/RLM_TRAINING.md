# Exact-turn RLM training with TRL

## Credit is not a separate reward per turn

For four trajectories, compute one GRPO advantage from each final verifier reward,
including speed. Attach that same advantage to every root reply in its trajectory.
Keep each reply's exact sampled prompt, token IDs and behavior log probabilities.
Do not join turns into one sequence: old reasoning is absent from later contexts.
REPL observations and child responses are context, not root loss targets.

The native TRL DAPO clipped-PPO loss sums root-token terms and divides by the
global count of real root tokens. Unequal turn counts do not change how the group
reward mean or standard deviation is calculated. Extra rows needed for distributed
alignment duplicate a real row with zero advantage; auxiliary router loss, entropy
bonus and KL penalty are disabled, so these rows have exactly zero gradient.
Native entropy/clip metrics include these padding rows; use the recorded real-row
counts for interpreting this bounded adapter.

`training_rows` rejects uniform final rewards. Whole groups can be oversampled at
most three times; exhausted or corrupted groups stop without an optimizer update.
The existing correctness/speed verifier and its submission guards are unchanged.

## Calling flow

```text
vLLM 0.25.1, TP2, separate inference environment
└── existing KernelGenerator.run_episode × four
    ├── exact-token CompletionBridge
    ├── existing V1 Python REPL program and optional child calls
    ├── cuda_feedback → shared-GPU sleep/wake admin relay
    └── trusted final CUDA verifier → terminal reward
         ↓ exact root-turn group with trajectory advantages
stop inference workers and release GPUs
         ↓
RLMGRPOTrainer (subclass of native TRL GRPOTrainer)
├── native FSDP2 preparation and Dion Muon optimizer factory
├── public get_train_dataloader: balanced exact-turn microbatches
├── public training_step: no stock single-sequence rollout preparation
├── native compute_loss: DAPO PPO clipping against sampled log probabilities
└── one optimizer update → compact adapter export and local Muon/RNG shards
```

The explicit backward scaling cancels Accelerate's automatic accumulation
division: DAPO already uses the global token denominator. CPU proofs compare the
actual native TRL update with a direct reference gradient, on one process and two
Gloo ranks. Their SGD optimizer is only an analytic test harness; production
always uses the validated Dion Muon factory.

## Limits of this first run

- Existing two L40S GPUs only. No concurrent inference/trainer GPU copies.
- One optimizer update; 30-minute original wall deadline, including failed GPU
  startups. No automatic retries or synthetic-reward fallback.
- Four trajectories per task, at most three candidate groups from the unchanged
  materialized L1 curriculum. Root cap 4,096, context 12,288, up to eight root
  turns and two text-only children. Recovery caps remain 512.
- The client uses the inference tokenizer endpoint to cap outputs to exact context
  headroom and checks chat/tokenize prompt-ID equality. Actual requests are saved.
- A local admin relay sleeps vLLM for exclusive verifier access and wakes it again.
  Existing file locks drain model calls before exclusive verification. This is
  process coordination, not a security sandbox for hostile model code.
- After generation the server stops, then native TRL runs from exact saved turns.
  This first adapter is a bounded one-group update, not an asynchronous pipeline.
- Compact export saves only trainable adapters, local Muon momentum and RNG state.
  Frozen base weights are never duplicated in the new checkpoint. Fresh-process
  resume and post-update inference reload still require explicit verification.

All requests, root/child responses, REPL observations, submissions, feedback,
verifier results, rejected groups and training inputs must remain inspectable.
One completed optimizer step is not evidence of quality improvement.
