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

The trajectory detail uses compact Benchmark/Model/Run metadata, a selectable step list, a turn-grouped timeline option, and a shared verifier panel. Raw events and complete code remain accessible; no text is silently truncated. The light theme uses a white canvas and warm-neutral content surfaces. This is a custom local implementation, not a verified copy of Poolside’s component library.

The app separates campaign records, execution workloads, and individual trajectory attempts. The Trajectories page lists attempts, not checkpoint continuations. Distinct attempts on the same task remain separate; duplicate exported IDs within a run collapse, while cross-run rescoring records retain their provenance. A “completed” tag means a final candidate was submitted, not that it passed verification.

Category colors identify training continuations (green), fix validations (amber), setup validations (blue), research pilots (purple), and tooling preparation (slate). These are purpose labels, not quality or live status. The middle pane resizes by drag or arrow keys and remembers its width; minimum width is 220 px, with 420 px reserved for the main pane when space permits. Mobile layout switches at 700 px.

The Lineage view keeps the resizable experiment list open and synchronizes selection with graph nodes. Selecting a record centers its node; selecting a node reveals its list entry (clearing a hiding filter if necessary). “Open experiment” opens the full record separately. The Lineage view draws the registry’s research follow-up graph. Edges are not inferred checkpoint inheritance. Pan and zoom do not modify records. This uses the existing plain JavaScript/SVG frontend; no inferred Poolside stack or external packages were installed.

`--task-catalog` is optional. It provides operation titles and reference source using exact `_task_id` matches. It is display metadata, not a new historical input attestation. Benchmark labels come from recorded dataset membership. Peak input context uses the largest root prompt token count; prompt-cache percentage requires recorded cached-token counts for every call. Model identity and episode start time remain unavailable when absent from exports. The original visualizer and historical HTML remain intact.

Validation includes focused new and legacy dashboard tests, Python lint/format checks, JavaScript syntax checks, and Chrome headless checks against the final-run archive. This is a local research viewer, not a public deployment or a replacement for the full legacy feature set.

Current limits: the adapter reads exported `traces.jsonl` episodes; other historical formats may have a run record but no supported episode export. Missing model/task-source metadata stays unavailable. Child records are shown separately when the archive lacks a reliable parent-turn link. Campaign notes are edited for readability while preserving IDs, evidence links and historical meaning. Records are categorized as training continuations, fix validations, setup validations, research pilots or tooling preparation under the L1 post-training campaign. These categories describe purpose, not measured benefit. Original status labels remain visible as saved historical status, not live activity; unlabeled status dots are no longer used. Repository-source links outside the Writing allowlist are displayed as paths rather than served. Use the `logs/live-grpo` export collection for historical coverage. Run discovery does not load metric histories; detail requests include full events/raw records only for the selected trajectory. Historical status labels are saved observations, not live job status.

The interface calls the work “L1 simple kernels · Stage 1”; historical campaign IDs and receipts are unchanged. Child call panels render assignment/reply text only, while token arrays remain in Raw events. The timeline groups root turns with subsequent observations without inventing child-parent links. Inter is bundled locally for UI and Writing; JetBrains Mono is bundled for code and technical metadata. Both use the SIL Open Font License; license texts, upstream URLs, and SHA-256 hashes are in `dashboard/fonts/`. No third-party font requests are made. Poolside’s public sites use Untitled Sans and JetBrains Mono; the private app font is unverified, and Untitled Sans is not bundled. Active highlights use rounded white-on-#006b45 green states.

## Agreed next iteration: provenance and metrics

Keep TRL/Trackio for trainer metrics; this viewer currently reads filesystem exports rather than querying Trackio directly. Extend the existing records and exporter instead of introducing a competing metrics store. Connect Trackio run IDs and optimizer steps to explicit execution, episode, group, root-turn, model/tool/child-call, verification, and checkpoint IDs.

The target is token provenance from sampling to training: preserve exact token sequences (with tokenizer revision and artifact hashes), call identity, role/branch, sampled policy, and the mapping to training rows, token positions, loss masks, and optimizer updates. Count inference usage separately from tokens contributing to loss; child outputs are not directly trained root outputs. Repeated context, padding, replay, rescoring, and inherited updates must remain distinguishable. Do not claim historical token-to-update attribution where the archived join is missing.

Priorities for the follow-up PR:

- Preserve explicit child parent-turn/tool-call IDs at dispatch and through export; never infer causality from arrival order.
- Preserve runtime model/checkpoint/config identity rather than importer defaults. Use exact admitted group/episode IDs, not task names, when selecting training evidence.
- Expose recorded root/child usage, call/admission durations, verifier timing, and recorded run elapsed time with coverage counts. Derived viewer IDs are navigation identities, not recovered source IDs. Missing usage, timestamps, cache details, and reasoning-token splits remain unknown.
- Version the run manifest and event/export schemas; keep raw evidence separate from rebuildable summary indexes and authored explanations. Retain failures and excluded samples.
- Link candidate hashes and verification attempts to turns; separate intermediate feedback from final grading and training results from held-out evaluation.
- Let an optional agent draft evidence-linked interpretation after deterministic calculations, with human review. An agent is not needed to load or count results, and prose must not become the measurements' source of truth.

No training settings, loss/reward semantics, or historical artifacts are changed by this dashboard PR.


## Recorded metrics

Trajectory detail presents recorded root and child model-call input, output, and total token counts separately. Root calls use `info.kernel_model_calls` when present, otherwise the historical `calls` list; node/transcript renderings are not counted again. Child usage reads `response.usage` and supports older direct `usage` records. Each metric shows its known sum and recorded-call coverage. A partial sum is not presented as a complete total; missing values remain unavailable, while explicitly recorded zero remains zero.

Run elapsed time is shown only when recorded by the report and is distinct from summed per-call elapsed durations (which may overlap and are not wall time). Verifier timing is displayed only from the selected trajectory's final recorded verification object, with missing breakdowns called unavailable. The dashboard does not infer parent joins or aggregate across duplicate exports from different runs.
