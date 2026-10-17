# Project-level factory

This is a small offline provenance layer, not a scheduler or training framework.
It uses the standard library. It never starts compute by itself.

## Record the question first

Register an experiment in `dashboard/experiments.json` with intent, change,
decision, code paths, note paths and dataset IDs. Dataset definitions name the
materialized JSONL file, its upstream source and its intended role. Keep unknown
upstream revisions null; do not manufacture historical provenance.

Validate all references and dataset task IDs from this environment directory:

```bash
python -m kernel_factory validate
```

## Prepare without launching

Commit both the project and nested RLM repositories. Then:

```bash
python -m kernel_factory prepare --experiment factory-provenance --pilot \
  --run-id <unique-local-preparation-id>
```

This creates `logs/live-grpo/<id>/` with:

- `provenance.json`: experiment ID, both Git revisions/dirty states, pinned model
  and Prime revisions, dataset SHA-256/task count/task-ID hash, parsed config,
  input hashes, checkpoint parent and declared artifact locations.
- `inputs/`: exact small source, configuration, dataset, registry and note files.
- `events.jsonl`: append-only lifecycle records. Preparation is not execution.

Normal capture rejects dirty repositories. `--allow-dirty` is an explicit
exception: the manifest labels it and preserves selected working-tree bytes.
Run IDs and snapshot paths are checked. Existing preparations are not overwritten.
No environment dump, credential values, model weights or optimizer tensors are
collected. Snapshots contain selected project source/data/notes; keep those free
of embedded secrets and treat the saved run directory as private project data.

## Launch integration

Future `modal_app.grpo_live_smoke` launches require `--experiment <registered-id>`.
The local entrypoint captures inputs before dispatching a GPU function. It also
checks the actual Prime checkout against its pinned revision and requires that
dependency checkout to be clean. No new retry or budget limit is introduced.

The remote function persists the manifest and snapshots, compares shipped
Python source, selected config and dataset against prepared hashes, records
installed package versions, then starts the existing Prime command. A mismatch
stops before the trainer subprocess, although Modal has already allocated the
requested container by this point. Failure reports and lifecycle events remain
in the trace volume. This gate still needs its first live cloud validation.

The local-only preparation command records declared pins but does not inspect
the Prime checkout. That additional check belongs to the actual launch path.
Checkpoint read-back and tiny CPU optimizer checks remain separate commands.

Download the small run directory, including `inputs/`, `runtime.json`,
`provenance.json` and `events.jsonl`; leave checkpoint weights on Modal. The
dashboard's Provenance view opens captured input snapshots and verifies their
hashes. Old runs explicitly show missing attestation rather than borrowing the
current file's hash. The registry keeps documented change commits separate from
exact launch revisions.

## Limits of the evidence

- Hashes provide content identity/integrity, not a trusted signature.
- Dataset identity attests materialized bytes; it does not recover the missing
  original upstream revision, prove licensing, or prove train/eval separation.
- The image tag is versioned, but no immutable registry image digest is captured.
- Actual package versions are recorded on execution; they do not replace a full
  dependency lock or complete environment replay.
- Only the GRPO launch path is gated initially. Older launchers/probes remain
  visibly outside this coverage. No historical manifest is backfilled as fact.
- Training correctness, checkpoint durability and policy quality need their own
  tests. A complete manifest is not a successful experiment result.
