# Kernel-RLM experiment dashboard

This is a local, CPU-only viewer for downloaded Modal artifacts. The canonical
project plan and learning report is the repository-root `PROJECT_TRACKER.html`.
This viewer organizes questions, run attempts and trajectory evidence:

- reward, loss, and entropy curves from `metrics.jsonl`;
- run metadata and status;
- RLM rollout traces, CUDA feedback, verifier results, and `max_diff`;
- saved Verifiers episode records.

The Poolside-inspired workspace uses Python's standard HTTP server, HTML, CSS,
plain browser JavaScript and inline SVG. There is no frontend build step, extra
package, hosted service, model call or GPU job. TypeScript/React is not needed
for this first version. The learning report has a quieter editorial layout,
inspired by Microsoft's pre-training article, inside the same navigation.

## Views and evidence

- `/`: current milestone, next decision and all downloaded attempts.
- `/?view=experiments`: curated experiments with intent, change, conclusion,
  limitations and related runs. Search the experiment list independently.
- `/?view=runs`: timestamp-sorted attempts with search and execution filters.
- `/?view=lineage`: explicit checkpoint resume and retry links; chronological
  order alone never creates a dependency. A failed restore is labelled requested.
- `/?view=configs`: saved component revisions, separate from current config files.
- `/guide`: the canonical root HTML report, rendered inside the workspace.

`experiments.json` stores curated research questions and decisions, not another
copy of rollout metrics. Add related run IDs and explicit retry relationships
there. Historical runs remain in the attempt library even before they are
assigned to a curated experiment. Keep failed attempts in the same experiment.
`run_labels.json` provides readable labels without changing original artifacts.

Summaries normalize report groups, older trace files and Prime traces. Stable
trace IDs prevent double-counting `all` and `effective` exports; admitted counts
remain separate. The latest resumed run has eight saved episodes, four admitted
episodes and three correct final kernels. Date sorting extracts UTC timestamps
instead of sorting by category-prefixed strings. Legacy deep links still work.

The run view uses a three-column layout: run summary,
trajectory timeline, and saved metrics. Select any saved Prime trajectory (not
just the first twelve), then switch between Run trace, Verification, and Artifacts.
Messages, system prompts, reasoning, structured calls, and raw records are folded
by default. The expand control opens the selected trajectory's messages.
Structured calls are labelled as emitted, not as proof of execution. Token-limit
flags and missing submissions remain visible even when the agent process completed.
GPU memory/utilization charts read `gpu_stats.jsonl`. Axes include units, series
labels and source ranges. Trainer charts recognize Prime metric names. A single
saved optimizer point is a scalar, not a fabricated learning curve. Missing
measurements are unavailable, never inferred as zero. Telemetry spans exclude
queue time; they are not billed durations. A passed process is not evidence of
held-out model improvement.
Use **Expand charts** on a run to read the measurements in a wider grid; **Back
to trace** restores the trajectory view without changing the selected episode.

This is an independent implementation inspired by https://posttrainbench.com/traces/.
The public benchmark repository is https://github.com/aisa-group/PostTrainBench;
its main source tree did not contain the deployed website viewer at review time.
Other design references: the user-provided Poolside Model Factory screenshots,
https://arxiv.org/pdf/2605.27605 (section 2), and
https://microsoft.ai/news/frontier-pre-training-at-microsoft-ai/.

Start it from `training/environments/kernel_rlm`:

```bash
python -m dashboard.server --root logs --port 7790
```

The server must stay running while the browser uses `http://127.0.0.1:7790/`.
If the page does not load, restart the command above. It does not start by
opening `PROJECT_TRACKER.html` directly.

Open `http://127.0.0.1:7790/guide` for the consolidated project plan and
learning report. The dashboard navigation links between the guide and saved
experiment reports.

To inspect a downloaded Modal run, place its directory below `logs/` and reload
the page. The viewer never starts a model, verifier, or GPU job.
