# Kernel-RLM experiment dashboard

This is a local, CPU-only viewer for downloaded Modal artifacts. It keeps the
curated `PROJECT_TRACKER.html` separate from raw experiment evidence while
showing the same trace progression as a PostTrainBench-style run page:

- reward, loss, and entropy curves from `metrics.jsonl`;
- run metadata and status;
- RLM rollout traces, CUDA feedback, verifier results, and `max_diff`;
- saved Verifiers episode records.

Start it from `training/environments/kernel_rlm`:

```bash
python -m dashboard.server --root logs --port 7789
```

To inspect a downloaded Modal run, place its directory below `logs/` and reload
the page. The viewer never starts a model, verifier, or GPU job.
