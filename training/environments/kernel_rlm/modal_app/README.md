# Modal smoke test

Run the known-good vector-add submission on an A100-80GB GPU:

```bash
cd /home/juancm/projects/laguna-rlm/rlm/training/environments/kernel_rlm
modal run -m modal_app.kernel_smoke
```

The image pins CUDA 12.4.1 and PyTorch 2.5.1. The test compiles the submission
in the verifier's fresh process, checks correctness, and records the benchmark
speedup. It does not run RLM training.
