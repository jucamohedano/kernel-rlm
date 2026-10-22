# Local research dashboard

A private viewer for selected campaign evidence, with local editing of allowlisted research Markdown. It does not launch experiments, modify training records, or publish artifacts.

From the repository root, use the project's existing environment:

```bash
uv run python -m dashboard.server \
  --evidence-root /path/to/private/evidence \
  --campaign-receipt /path/to/campaign-receipt.json \
  --task-catalog /path/to/lvl1_seed42.jsonl \
  --port 7790
```

`--evidence-root` is required and must name an existing directory. Use the campaign export collection (`logs/live-grpo`) to browse runs across the L40S and RTX stages, rather than just its final-run child directory. An individual run directory intentionally limits the viewer to that run. Avoid broad home directories and complete retirement/checkpoint backups; those include redundant copies and artifacts this viewer does not need. `--campaign-receipt` is optional and accepts a JSON object. The server binds to localhost by default on port 7790; `--port` overrides the port. Keep it local and do not expose it to a network or use it to publish private evidence.

Writing documents use a fixed repository allowlist; there is no `--notes-root` option. Source/preview editing saves only these files. Saves require a same-origin token and matching SHA-256 version, reject symlinks, and retain previous content in `~/.prime-agent/kernel-rlm/dashboard-writing-backups`. The UTF-8 document limit is 1 MiB. No save occurs automatically. The Writing IDs are `README`, `first-campaign`, `protocol`, `artifact-publication`, and `experiment-tracking`, served from `docs/research/{README.md,first-campaign.md,protocol.md,artifact-publication.md,experiment-tracking.md}`. The preserved legacy guide is available at `/archive/guide` for historical reference, not current instructions. See the [research index](../docs/research/README.md).

## Implementation status

The trajectory detail uses compact Benchmark/Model/Run metadata, a selectable step list, a flat timeline option, and a shared verifier panel. Raw events and complete code remain accessible; no text is silently truncated. The light theme uses a white canvas and warm-neutral content surfaces. This is a custom local implementation, not a verified copy of Poolside’s component library.

The app separates campaign records, execution workloads, and individual trajectory attempts. The Trajectories page lists attempts, not checkpoint continuations. Distinct attempts on the same task remain separate; duplicate exported IDs within a run collapse, while cross-run rescoring records retain their provenance. A “completed” tag means a final candidate was submitted, not that it passed verification.

Category colors identify training continuations (green), fix validations (amber), setup validations (blue), research pilots (purple), and tooling preparation (slate). These are purpose labels, not quality or live status. The middle pane resizes by drag or arrow keys and remembers its width; minimum width is 220 px, with 420 px reserved for the main pane when space permits. Mobile layout switches at 700 px.

The Lineage view draws the registry’s research follow-up graph. Edges are not inferred checkpoint inheritance. Pan and zoom do not modify records. This uses the existing plain JavaScript/SVG frontend; no inferred Poolside stack or external packages were installed.

`--task-catalog` is optional. It provides operation titles and reference source using exact `_task_id` matches. It is display metadata, not a new historical input attestation. Benchmark labels come from recorded dataset membership. Peak input context uses the largest root prompt token count; prompt-cache percentage requires recorded cached-token counts for every call. Model identity and episode start time remain unavailable when absent from exports. The original visualizer and historical HTML remain intact.

Validation includes focused new and legacy dashboard tests, Python lint/format checks, JavaScript syntax checks, and Chrome headless checks against the final-run archive. This is a local research viewer, not a public deployment or a replacement for the full legacy feature set.

Current limits: the adapter reads exported `traces.jsonl` episodes; other historical formats may have a run record but no supported episode export. Missing model/task-source metadata stays unavailable. Child records are shown separately when the archive lacks a reliable parent-turn link. Campaign notes are edited for readability while preserving IDs, evidence links and historical meaning. Records are categorized as training continuations, fix validations, setup validations, research pilots or tooling preparation under the L1 post-training campaign. These categories describe purpose, not measured benefit. Original status labels remain visible as saved historical status, not live activity; unlabeled status dots are no longer used. Repository-source links outside the Writing allowlist are displayed as paths rather than served. Use the `logs/live-grpo` export collection for historical coverage. Run discovery does not load metric histories; detail requests include full events/raw records only for the selected trajectory. Historical status labels are saved observations, not live job status.
