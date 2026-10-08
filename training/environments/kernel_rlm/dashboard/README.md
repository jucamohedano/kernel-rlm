# Kernel-RLM experiment dashboard

This is a local, CPU-only viewer for downloaded Modal artifacts. It keeps the
curated `PROJECT_TRACKER.html` separate from raw experiment evidence while
showing the same trace progression as a PostTrainBench-style run page:

- reward, loss, and entropy curves from `metrics.jsonl`;
- run metadata and status;
- RLM rollout traces, CUDA feedback, verifier results, and `max_diff`;
- saved Verifiers episode records.

The run view uses a PostTrainBench-inspired three-column layout: run summary,
trajectory timeline, and saved metrics. Select any saved Prime trajectory (not
just the first twelve), then switch between Run trace, Verification, and Artifacts.
Messages, system prompts, reasoning, structured calls, and raw records are folded
by default. The expand control opens the selected trajectory's messages.
Structured calls are labelled as emitted, not as proof of execution. Token-limit
flags and missing submissions remain visible even when the agent process completed.
GPU telemetry is not yet connected; missing series are not shown as zero.

This is an independent implementation inspired by https://posttrainbench.com/traces/.
The public benchmark repository is https://github.com/aisa-group/PostTrainBench;
its main source tree did not contain the deployed website viewer at review time.

Start it from `training/environments/kernel_rlm`:

```bash
python -m dashboard.server --root logs --port 7789
```

To inspect a downloaded Modal run, place its directory below `logs/` and reload
the page. The viewer never starts a model, verifier, or GPU job.
