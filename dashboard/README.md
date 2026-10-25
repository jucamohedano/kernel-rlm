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
