# Versioned Hugging Face curricula without breaking JSONL training

The public dataset is [jucamohedano/kernel-rlm-curriculum](https://huggingface.co/datasets/jucamohedano/kernel-rlm-curriculum). Use immutable revision `c5bfb92c4f4dfb5aa865fca035e2b3113a6824f3` to reproduce the preserved curricula. Anonymous reads are supported; no Hugging Face login is required to download this public release.

The source is [CUDA-Agent-Ops-6K](https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K). The uploaded dataset card includes attribution, the source's declared CC-BY-4.0 license, and our selection/order/format-conversion changes. The historical upstream download revision was not recorded; it is not fabricated. Exact preserved materializations are authoritative for reproducing previous runs.

## Configurations

- `lvl1_seed42`: 50 single-op tasks in the original selected order.
- `lvl1_seed42_easy_first`: the same 50 task records in the saved easy-first order.
- `lvl2_seed42`: 100 selected two-op tasks.
- `lvl3_seed42`: 50 selected three-to-five-op tasks under our builder's classification, not KernelBench levels.

Each configuration has one `train` split. Configurations are not independent train/test splits. Selection does not mean every task received an accepted optimizer update. No held-out evaluation material, raw model responses, or checkpoint weights are uploaded.

## Native Datasets access

Install this environment's existing `datasets` dependency. Authentication is not required for public downloads; publishing changes still requires an authorized account. Do not put tokens into commands, code, or the dataset card.

```python
from datasets import load_dataset

tasks = load_dataset(
    "jucamohedano/kernel-rlm-curriculum",
    name="lvl1_seed42_easy_first",
    split="train",
    revision="c5bfb92c4f4dfb5aa865fca035e2b3113a6824f3",
)
```

Parquet preserves all source columns, task IDs, code strings, constructor semantics, and row order. Do not sort, shuffle, resample, normalize references, or replace original constructor arguments when reproducing a saved run.

## Existing JSONL training commands remain supported

Current training drivers require a local `--dataset` path and hash its actual file bytes. Loading a Dataset and reserializing it can change whitespace or key ordering, causing a legitimate resume hash failure. The Hub release therefore also retains exact original files under `original/`.

From this environment directory, materialize the original bytes:

```bash
python -m kernel_rlm.curriculum download \
  --repo-id jucamohedano/kernel-rlm-curriculum \
  --config lvl1_seed42_easy_first \
  --revision c5bfb92c4f4dfb5aa865fca035e2b3113a6824f3 \
  --expected-sha256 dade5ab7fe2391fcd565a091bc44ec3c374d26cc715a2bce651df814606ed6d7 \
  --output /tmp/kernel-rlm-inputs/lvl1_seed42_easy_first.jsonl
```

Pass that path to an existing training command only under a separate compute authorization. The download command is CPU-only and does not execute reference programs, alter prompts/rewards/shapes, or launch training.

The materializer requires a full Hub commit and an expected original SHA256, checks task IDs/content/order against the manifest, refuses differing existing outputs, and never silently falls back to mutable `main`. An already present, hash-matching file can be reused offline. Existing JSONL readers, checkpoint manifests, dataset SHA gates, and historical resume semantics are unchanged.

## Exporting a reviewed release

```bash
python -m kernel_rlm.curriculum export \
  --source-dir /path/to/preserved/curriculum \
  --output-dir /tmp/new-curriculum-release
```

The input directory must contain all four original JSONL files. Export refuses an existing destination, validates the task contract, checks the two L1 configurations contain identical records, writes Parquet/config metadata, and verifies local roundtrips. It does not sample the upstream dataset or upload anything. Publication uses an explicit separate `hf upload --repo-type dataset` operation after review.

The release was tested with `datasets==4.0.0`; unit tests cover unchanged-byte download, legacy-loader compatibility, mutable revision rejection, tampering, and refusal to overwrite another curriculum. Other legacy scripts that assume the old parent workspace's location still require a separate portability pass; this distribution path does not silently rewrite all their paths.
