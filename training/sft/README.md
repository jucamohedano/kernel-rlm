# XS-2.1 SFT CPU preflight

Both paths use the same local, hash-pinned `manifest.json` and `groups.jsonl`. Each original trajectory is one record, with each decision on its own causal row. A supplied label mask selects loss positions; neither backend should rerender, pack across decisions, or truncate. This checkout accepts **synthetic fixtures only**. It does not read or materialize a production dataset.

## Run the checks

From the repository root, set `SFT_PYTHON` to the Python executable in an existing pinned environment. This repository does not bundle that environment or add its dependencies to the base training harness. If the training package is not installed there, expose the source from the repository root:

```sh
export PYTHONPATH="$PWD/training/src${PYTHONPATH:+:$PYTHONPATH}"
```

```sh
"$SFT_PYTHON" -m pytest -q -p no:cacheprovider \
  training/tests/sft/test_bundle.py training/tests/sft/test_trl.py training/tests/sft/test_prime.py
```

The tested Python 3.11 environment used TRL `1.19.0.dev0` at `68ad0529f6b79b4091b4d0f14cbc721f5c0c1b63`, Transformers `5.23.0`, Accelerate `1.16.0`, and PyTorch `2.15.0+cpu`. The TRL probe checks these versions and the `SFTTrainer` source hash. The tests also used Datasets `5.2.0`, Tokenizers `0.23.3`, and pytest `9.2.0`.

The real Trackio SQLite test skips when Trackio is unavailable. It passed offline with cached Trackio `0.40.0` on `PYTHONPATH`. This is not a dependency lock: the tested environment had `huggingface-hub` `2.4.0` and Pillow `13.0.0`, outside Trackio `0.40.0`'s declared `<2` and `<13.0.0` ranges, respectively. Only the local SQLite path was verified.

For a manual probe, set `BUNDLE` to a fresh directory, set `NEW_TRL_RUN` and `NEW_PRIME_RUN` to unused output paths, and capture the fixture writer’s returned SHA-256:

```sh
PIN=$("$SFT_PYTHON" -c 'import sys; from rlm_train.sft.bundle import write_synthetic_fixture; print(write_synthetic_fixture(sys.argv[1]))' "$BUNDLE")
"$SFT_PYTHON" -m rlm_train.sft.train_trl \
  --bundle "$BUNDLE" --manifest-sha256 "$PIN" \
  --output-dir "$NEW_TRL_RUN" --allow-synthetic --trackio
"$SFT_PYTHON" -m rlm_train.sft.train_prime \
  --bundle-dir "$BUNDLE" --manifest-sha256 "$PIN" \
  --output-dir "$NEW_PRIME_RUN" --allow-synthetic
```

- **TRL / Trackio:** uses the actual `SFTTrainer` with an intact-group collator and explicit selected-token loss. The optimizer probe is one step, one rank, and no gradient accumulation. Optional Trackio writes only aggregate metrics to local SQLite, with no Space or upload.
- **Prime RL / Prime dashboard:** `train_prime` is a CPU contract probe, not the native SFT launcher. It writes local numeric metrics in the native reader's layout. See [PRIME.md](PRIME.md) for the pinned Prime seam and dashboard limits.

Before production: approve the exact training data and split, freeze the materialized bundle, verify target backend loss/gradient and checkpoint resume, and measure member/batch caps. Distributed and accumulated gradients must normalize by the total number of selected positions, not by the number of groups. No GPU, network access, or telemetry upload is needed for these CPU checks.
