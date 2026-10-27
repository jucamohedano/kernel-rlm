# Local research dashboard

Use this dashboard to inspect campaign records and exported trajectories. Use Writing to edit approved research documents. The dashboard does not start training or publish evidence.

## Start the server

Run this command from the repository root:

```bash
uv run python -m dashboard.server \
  --evidence-root /path/to/logs/live-grpo \
  --port 7790
```

Open <http://127.0.0.1:7790>. The **Dark mode** toggle overrides your system theme and remembers your choice in this browser.

- Set `--evidence-root` to an existing export directory. Use `logs/live-grpo` to browse multiple runs. Use a run directory to view one run.
- Add `--campaign-receipt /path/to/receipt.json` to load a campaign summary.
- Add `--task-catalog /path/to/catalog.jsonl` to show task titles and reference source.
- Keep the server on localhost. Do not expose private evidence to a network.
- Do not use your home directory or a checkpoint backup as the evidence root.

## Start Trackio

Trackio shows saved trainer metrics. It does not show the trajectory viewer. Existing histories are unchanged; new logging rules apply to future runs.

Use a review copy. The Trackio viewer can write to its database. Set `TRACKIO_SOURCE` to the saved `laguna-kernel-rlm.db` file. For the final campaign, use `logs/trackio-final-local`, not the older `logs/trackio-viewer` snapshot.

```bash
export TRACKIO_SOURCE=/path/to/logs/trackio-final-local/laguna-kernel-rlm.db
export TRACKIO_DIR="$(mktemp -d)"
python3 -c 'import os, sqlite3; source = sqlite3.connect("file:" + os.environ["TRACKIO_SOURCE"] + "?mode=ro", uri=True); target = sqlite3.connect(os.environ["TRACKIO_DIR"] + "/laguna-kernel-rlm.db"); source.backup(target); target.close(); source.close()'
uv run --no-project --python 3.12 --with trackio==0.40.0 python -c 'import trackio; trackio.show(project="laguna-kernel-rlm", host="127.0.0.1", server_port=7861, share=False, open_browser=False, block_thread=True)'
```

Open <http://127.0.0.1:7861/?project=laguna-kernel-rlm>. This command can download Trackio and Python if they are not cached. It does not start training or publish metrics. Do not log training into the review copy.

Use separate terminals for the two dashboards. Press **Ctrl+C** in each terminal to stop its server. Use a different port if one is already in use.

### Review RL training metrics

Open the [accepted-history RL view](http://127.0.0.1:7861/?project=laguna-kernel-rlm&runs=laguna-l1-verifier-v2-learning-curves&metrics=outcomes%2Fmean_reward%2Coutcomes%2Freward_std%2Coutcomes%2Fcorrect_fraction%2Coutcomes%2Ffaster_correct_fraction%2Coutcomes%2Fsubmitted_fraction%2Ctrain%2Fentropy%2Ctrain%2Fclip_ratio%2Fregion_mean%2Ctrain%2Fgrad_norm%2Ctrain%2Floss&x_axis=curves%2Foptimizer_step&smoothing=0&plots_per_row=3). It selects only `laguna-l1-verifier-v2-learning-curves`, uses optimizer step on the x-axis, and shows reward, reward spread, correctness, faster-correct fraction, submission rate, entropy, clipping, gradient norm, and loss.

Do not select all records for this review. Older probes use different metric names and can appear as unrelated summary bars. These curves cover accepted groups on changing tasks, not held-out improvement. Steps 1–19 used L40S; later steps used RTX. The history includes inherited updates and verifier changes.

## Read the results

- A completed trajectory has a submitted candidate. It has not necessarily passed verification.
- Lineage edges show research follow-ups. They do not prove checkpoint inheritance.
- Saved status labels are historical records, not live job status.
- Missing measurements remain unknown. Do not treat them as zero.
- Trajectory details require a supported `traces.jsonl` export.

The dashboard reads filesystem exports. It does not query Trackio directly.

## Edit research documents

Open **Literature** for the maintained, project-linked review. Use **Source / Edit** to update it with the same backup and conflict checks as Writing.

Writing can save only these files in `docs/research/`:

- `README.md`
- `first-campaign.md`
- `protocol.md`
- `literature-review.md`
- `artifact-publication.md`
- `experiment-tracking.md`

Select **Save** to write changes. There is no automatic save. Saves reject version conflicts and retain backups in `~/.prime-agent/kernel-rlm/dashboard-writing-backups`. The document size limit is 1 MiB.

## References

- [Research index](../docs/research/README.md)
- [Provenance and metrics plan](https://github.com/jucamohedano/kernel-rlm/issues/8)
- Preserved historical guide: `/archive/guide`
