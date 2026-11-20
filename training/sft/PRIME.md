# Prime RL: synthetic grouped SFT contract probe

[Prime RL upstream](https://github.com/PrimeIntellect-ai/prime-rl) source pin: `4e94a8616558d99f9a39d6a1facbadbfe6233e86` (`prime-rl` 0.10.0). The audit and extracted-source test use committed source from this revision, not working-tree files.

- `src/prime_rl/trainer/sft/data.py`: `SFTDataset` renders messages then shifts IDs and masks. `setup_dataloader` uses `CatDataset` to pack rows before `cat_collate`. The stock path does not preserve the pretokenized grouped bundle as optimizer samples.
- `src/prime_rl/trainer/sft/train.py`: the native CUDA path evaluates a summed loss with ignored targets masked out and scales gradients by global selected-token count (accounting for accumulation and gradient division). This is source verification, not a distributed numerical test.
- `src/prime_rl/dashboard/server.py` and `src/prime_rl/utils/pathing.py`: the scanner recognizes a child with a `monitors` directory and reads `monitors/file/metrics.jsonl`. The pinned reader functions were exercised in isolation against probe output; the FastAPI app, browser, and native monitor did not run.

## Local use

From this repository root, using `SFT_PYTHON` and a synthetic fixture as described in [README.md](README.md):

```sh
"$SFT_PYTHON" -m rlm_train.sft.train_prime \
  --bundle-dir "$SYNTHETIC_BUNDLE" --manifest-sha256 "$MANIFEST_SHA256" \
  --output-dir "$DEDICATED_OUTPUT_PARENT/prime-cpu-probe" --allow-synthetic
```

The output parent must already exist; the run child must not exist. The probe uses one CPU-only, in-memory SGD step at LR `0.01` on a seeded `16 × 16` synthetic score table. It checks loss and gradient against the shared grouped objective before stepping, normalizing once by all selected targets in the synthetic batch. It logs one numeric aggregate JSONL row. It does not launch a dashboard, load a checkpoint, upload data, or enable a remote monitor.

Once the pinned dashboard runtime is separately available, point its installed CLI at this **dedicated parent**, not the run child:

```sh
dashboard --host 127.0.0.1 --isolated "$DEDICATED_OUTPUT_PARENT"
```

`--isolated` skips the Prime dashboard directory registry and daemon discovery. Use the Metrics tab for `loss/mean`, `loss/perplexity`, and `sft_cpu_probe/*`. No live server was started. The native dashboard runtime was not available: the offline Prime workspace had a missing dependency checkout, and its workspace settings were incompatible with the available uv version.

The native CUDA trainer remains **blocked**, not a selectable backend of this module. Production needs a loader/loop integration that preserves trajectory groups without re-rendering or cross-decision packing, native model/attention and gradient parity, and all required bundle attestations. Do not pass the bundle to stock `sft` or enable native remote monitors for this workflow.

Probe tests: `OMP_NUM_THREADS=1 "$SFT_PYTHON" -m pytest -q training/tests/sft/test_prime.py`.

## Pinned Prime RL loader seam (test-only)

The patch modifies upstream `src/prime_rl/trainer/sft/data.py` to add the synthetic loader seam; the upstream [Apache 2.0 license and attributions](LICENSE-prime-rl.txt) accompany the patch.

`prime-rl-4e94a8616-grouped-synthetic.patch` adds an explicit branch in Prime RL's `src/prime_rl/trainer/sft/data.py:setup_dataloader`, before `CatDataset`. The stock caller in `trainer/sft/train.py`, under “Set up the dataset and dataloader”, runs `setup_dataset` first. The test caller deliberately replaces **both** calls with `setup_dataloader(None, None, 1, grouped_synthetic=(bundle_dir, pin))`. No CLI or native trainer configuration exposes this path. The keyword requires a validated synthetic bundle and rejects stock dataset/config, packing, context parallelism and distributed rank/process modes.

The loader returns an iterator with one tuple per trajectory. Each tuple contains native `cat_collate` batches with one unpadded request per forward, position IDs starting at zero and one `seq_lens` segment. The test helper takes the native trainer's `(summed NLL, token_count)` callback convention and divides once across all selected tokens, with no optimizer activity. Loss/gradient parity with the shared objective is checked using causal CPU logits. This validates the loader and denominator, not Prime RL GPU attention or LM-head execution.

To reproduce without editing Prime RL, run from this repository root with `PRIME_CHECKOUT` set to a local Prime RL repository containing the pinned commit:

```sh
pinned=4e94a8616558d99f9a39d6a1facbadbfe6233e86
stage=$(mktemp -d)
file=src/prime_rl/trainer/sft/data.py
mkdir -p "$stage/$(dirname "$file")"
git -C "$PRIME_CHECKOUT" show "$pinned:$file" > "$stage/$file"
git -C "$stage" apply --check "$PWD/training/sft/prime-rl-4e94a8616-grouped-synthetic.patch"
git -C "$stage" apply "$PWD/training/sft/prime-rl-4e94a8616-grouped-synthetic.patch"
OMP_NUM_THREADS=1 XS21_PRIME_PATCHED_SOURCE="$stage/$file" \
  "$SFT_PYTHON" -m pytest -q training/tests/sft/test_prime_native_seam.py
```

The extracted-source mode runs the patched function bodies and native `cat_collate`, not the full module import. With an already installed and patched Prime runtime, set `PRIME_PYTHON` to its Python executable and run `XS21_PRIME_NATIVE_IMPORT=1 "$PRIME_PYTHON" -m pytest -q training/tests/sft/test_prime_native_seam.py` from this repository root instead. Full import here was blocked by missing Prime dependencies (`jaxtyping` first), in addition to the existing offline workspace issue above. No native CUDA trainer, server, production data or network was used.
