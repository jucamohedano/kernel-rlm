# Single-policy generation and training

The next TRL maintenance gate uses native Transformers generation on the same
FSDP2 policy used for training. It does not create a second inference model or
install vLLM. This is a low-friction memory route, not a throughput claim.

## Calling flow

```text
GRPOTrainer.train (no custom rollout function)
├── native Transformers generate: four sampled completions
├── synthetic fixture rewards and native GRPO loss
├── Dion Muon update 1
├── on_step_end: phase_roundtrip
│   ├── reshard all FSDP2 units
│   ├── move model and Muon momentum to CPU
│   ├── record released CUDA memory
│   └── reload; verify values and Parameter identities
├── native Transformers generate with the updated, reloaded policy
├── native GRPO loss and Muon update 2
└── save reports and exact generation traces; exit
```

The shared BF16 loader, rank-1 meta-buffer fix, small CPU adapter placeholders,
FP32 attention LoRA rank 8 / alpha 16 and Moonlight Muon are unchanged. Only the
trainable adapters and momentum are copied for bitwise comparisons. Frozen base
weights are not cloned for comparison or saved as a new checkpoint.

`phase_state.py` moves optimizer state explicitly; `model.to(...)` alone does not
move it. It verifies that parameter identities and optimizer coverage survive.
The second native backward/update must work with momentum from the first step.
At full-model scale, more than 1 GiB left allocated during offload fails the
strict memory-release gate. No second model is loaded even if this check passes.

## Deliberate limits

- Exactly two allocated GPUs, two optimizer updates and one CPU roundtrip.
- Four completions per group, microbatch one per GPU, two accumulated microbatches.
- Exactly eight new tokens per completion. Minimum and maximum are equal so all
  ranks take the same bounded decode path. This is not the curriculum reply cap.
- Full-model input uses the pinned real tokenizer and chat template, with thinking
  enabled. Tiny input uses a local random-model tokenizer.
- Rewards are `[0, 1/3, 2/3, 1]` fixture values, not kernel correctness or speed.
- Truncation masking is off for this diagnostic only. The eight-token samples
  intentionally reach the cap; a clipped-ratio metric of one is expected here.
- No model checkpoint, trained adapter, API upload or factory experiment is made.
- Native generation is instrumented after FSDP preparation; exact prompt and
  completion IDs, decoded output and timing are saved per rank.

Changing stochastic completions after the first update does not establish a
policy-quality gain. There is no RLM REPL, CUDA verifier or child call in this
probe. Exact changing-prefix credit, real context lengths and compact restart
remain separate gates. vLLM sleep/weight sync also remain untested; native
single-policy generation avoids that dependency rather than validating it.

## Invocation on the existing VM

Use the private runtime and CPATH from the shared framework validation guide.
Run tiny first, then full only if tiny passes, in fresh output directories:

```bash
timeout --signal=TERM --kill-after=5s 600s python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 --max_restarts=0 \
  -m kernel_trl.phase_probe --output-dir /fresh/tiny-phase
timeout --signal=TERM --kill-after=5s 600s python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 --max_restarts=0 \
  -m kernel_trl.phase_probe --checkpoint /cached/pinned/checkpoint \
  --output-dir /fresh/full-phase
```

Capture one-second NVML telemetry externally. Download only reports, traces,
telemetry and source snapshots; preserve failed attempts. The cached base and
earlier Modal checkpoints stay untouched. VM billing continues after workers exit.

Sources: [pinned native generation path](https://github.com/huggingface/trl/blob/33f9e462728b98f7f91d38b99328e81adde2faa0/trl/trainer/grpo_trainer.py),
[pinned FSDP2 generation context](https://github.com/huggingface/trl/blob/33f9e462728b98f7f91d38b99328e81adde2faa0/trl/distributed.py).
