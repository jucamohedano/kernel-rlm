# CuTe curriculum v1

This builder creates task-reference JSONLs from an explicit, source-reviewed selection. It preserves original PyTorch reference code and adds verified operation labels. It does not generate CuTe solutions or start training.

## Scope

The initial vocabulary is tensor addition, tensor multiplication, ReLU, and explicit-axis sum. Group assignments describe reviewed dataflow. They are not inferred or proven by the builder. Original upstream labels remain separate provenance.

The local selection and built data are ignored under `.research-local/`. They are not published artifacts. Use pinned source files; do not replace the historical curriculum.

## Build

Run from the repository root in an existing environment with `pyarrow`:

```bash
PYTHONPATH=training/environments/kernel_rlm python -m kernel_rlm.cute_curriculum \
  --selection .research-local/curriculum-v1-selection.json \
  --source cudaLLM=/path/to/rl_cuda_llm_0424.parquet \
  --source ops6k=/path/to/data.parquet \
  --output .research-local/curriculum-v1
```

The output directory must not exist. The builder verifies source-file and row-code hashes, checks reviewed graph partitions, and writes counts, coverage gaps, and artifact hashes in `manifest.json`.

## Read the output

- `train/primitives.jsonl`: individual computations.
- `train/compositions.jsonl`: available training compositions.
- `dev/`: development references; do not use reserved evaluation outcomes to tune settings.
- `eval/primitive-controls.jsonl`: configuration controls, not novel graphs.
- `eval/composition-heldout.jsonl`: reserved composition groups and their variants.

An empty file means coverage is missing. Do not pass an empty file to the legacy task loader. Row counts and shape variants are not counts of independent graph families.

Status is `STATIC_VALIDATED_NOT_RUNTIME_VALIDATED`. The record structure is compatible with the existing loader, but that loader still requests CUDA-extension submissions. These files are **not CuTe runtime-ready or approved for training**. CuTe submission/verifier integration, target hardware, correctness, numerical tolerances, and resource validation remain separate gates.

A resource-gated selection may contain unchanged, larger Ops-6K references. It is a separate optional build, not an automatic addition to the core training set. Static tensor-byte estimates are not peak-memory or runtime guarantees.

Implementation and remaining acceptance criteria: [issue #24](https://github.com/jucamohedano/kernel-rlm/issues/24).
