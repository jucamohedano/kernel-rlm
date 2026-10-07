# kernel_rlm

CUDA kernel tasks for the local RLM training loop.

The environment keeps Python as the persistent REPL language. A submission is Python source
that defines CPP_SOURCE, CUDA_SOURCE, and cuda_forward(*tensors). The submission compiles
the CUDA implementation through torch.utils.cpp_extension and calls the resulting extension.

The verifier imports the submission, compiles the extension, checks correctness on fresh CUDA
inputs, and benchmarks only after correctness passes. Verification runs in a separate process.

The tiny vector-add task is in kernel_rlm/tasks/vector_add.jsonl. A known-good submission is
in examples/vector_add_submission.py.

The package is intentionally compatible with the existing rlm_train.RLMTrainEnv. The generic
RLM loop and the historical triton_rlm environment remain separate.

## Muon verification

From this directory, run `modal run -m modal_app.muon_wiring` for a bounded CPU-only
check of prime-rl v0.9.0's real optimizer constructor against Laguna's meta model.
It checks all 320 attention LoRA tensors (8,192,000 parameters), frozen-parameter
exclusion, AdamW fallback, and Moonlight RMS scaling for every adapter shape.
The report is saved in `kernel-rlm-traces:muon-wiring/<run-id>/report.json` and
downloaded to `logs/muon-wiring/<run-id>/report.json`.

This does not execute an optimizer step, load checkpoint weights, or validate
GRPO. It uses the pinned prime source checkout at `/tmp/prime-rl-v090` inherited
from the native-model smoke image. Learning rate and decay in the check are test
values, not tuned RL hyperparameters. A GPU update check is still required.

## Monitoring the first GRPO run

The training launcher is not ready yet: this package still returns the legacy
`RLMTrainEnv`, while pinned prime-rl consumes Verifiers V1 tasksets and harnesses.
The previous model rollout also failed to submit a final answer. Resolve these
gates before launching paid GRPO, rather than treating optimizer construction as
an end-to-end training success.

Prime's file monitor is enabled by default. Mount the eventual run output directory
on a Modal Volume and commit it periodically and on shutdown. Preserve the resolved
config, `metrics.jsonl`, logs, and all/effective per-step rollout traces, including
failures and zero-advantage groups. Keep these files even when W&B is enabled.

For live W&B charts, the eventual RL configuration uses:

```toml
[monitors.wandb]
project = "laguna-kernel-rlm"

[monitors.file]
path = "metrics.jsonl"
```

These are configuration fragments, not a runnable training config. Supply
`WANDB_API_KEY` through a Modal Secret only to trusted training/orchestration
processes, never to generated-code workers. W&B is optional and currently not
configured in this project. Prime logs the trainer and orchestrator to one shared
run; its shared mode does not support offline W&B logging.

Without W&B, copy the saved run directory locally and use prime's read-only
`uv run dashboard /path/to/downloaded/run-output` to inspect curves and traces.
This serves a CPU-only viewer (normally port 7788), not a local model experiment.
A downloaded snapshot is not live unless updated. The project HTML tracker is
the curated learning report, not a live telemetry endpoint.

Watch these panels:

- Outcomes: reward and CUDA correctness separately, compilation and submission
  rates, speedup only for correct kernels, with sample counts.
- Learning signal: group reward spread, zero-advantage fraction, and all versus
  effective rollouts. An all-equal group supplies no relative-reward gradient.
- Stability: policy loss, entropy, gradient norm, learning rate, clipping, and
  available KL diagnostics. Policy loss need not decrease monotonically; trainer/
  inference mismatch KL is not automatically KL to the original base model.
- Budget: wall time, generated tokens, throughput, peak GPU memory, and estimated
  cost labeled separately from Modal's billed usage.

Prime supplies standard training metrics; CUDA outcome metrics and group diagnostics
must be checked in our eventual V1 integration. None of these training curves exists
yet: optimizer checks and rollout-only runs are not training runs.
