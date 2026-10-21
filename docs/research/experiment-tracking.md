# Experiment tracking and a public research dashboard

Research date: 21 October 2026. This is a decision proposal, not a deployed public dashboard. Evidence below comes from public primary documentation and source code; supported integrations do not establish an organization's universal internal stack.

## Recommendation

Keep Trackio as the existing experiment record and first improve its presentation over a reviewed publication export. Consider W&B if we need its report-authoring, Vega charts, and collaboration workflow enough to justify migration and account/storage administration. Choose MLflow primarily if model-registry or deployment-governance workflows become requirements. Do not rerun training or change loss/reward semantics to change tracking tools.

The archived project uses Trackio 0.40.0. Local signature inspection confirmed that exact version has `trackio.show(frontend_dir=...)` and `trackio.sync(frontend_dir=..., sdk=...)`. This verifies API availability, not a fully tested custom-frontend deployment. Existing private archives and the local dashboard were not modified.

## Improving Trackio

The [custom-frontend workflow](https://huggingface.co/docs/trackio/launch) serves a user-owned static frontend using Trackio's API. This permits a project-specific presentation without replacing the logger or forking its database implementation. [Grouping runs and configs](https://huggingface.co/docs/trackio/track) helps compare experiments coherently.

[Environment variables](https://huggingface.co/docs/trackio/environment_variables) control plot order, themes, and light/dark logos. These are useful visual improvements, but are not substitutes for clearly defined metrics.

[Deployment and embedding](https://huggingface.co/docs/trackio/deploy_embed) supports syncing existing projects to Spaces and creating browser-only static snapshots. Static snapshots require public backing data, so an iframe does not conceal private data. Use a new sanitized snapshot, not a blind sync of the private SQLite archive. Hosting/account/storage constraints need checking before deployment; this proposal starts no hosted service, subscription, or GPU job.

## Alternatives

### Weights & Biases

W&B supports logged tables and [Vega custom charts](https://docs.coreweave.com/products/wandb/app/features/custom-charts). [Reports](https://docs.coreweave.com/products/wandb/reports/create-a-report) combine selected experiments, visualizations, and narrative, with saved run filters and programmatic authoring. It is attractive if we prefer an established report editor over maintaining a publication frontend.

The tradeoff is a new data destination, privacy configuration, and potentially plan-dependent storage/hosting. A historical import must preserve timestamps, steps, lineage, missing values, and failure/weak-group status. Native trainer W&B reporting alone would not automatically include all custom verifier/audit metrics. Do not present imported historical runs as newly executed experiments.

### MLflow and TensorBoard

MLflow's [tracking server](https://www.mlflow.org/docs/latest/self-hosting/architecture/tracking-server/) combines a tracking API and UI, with metadata/artifact stores and model-registry integration. This fits lifecycle/governance or existing MLflow infrastructure, but adds service and storage administration for a small public research gallery.

TensorBoard remains useful for event-file diagnostics and is integrated by the public frameworks below. It does not by itself provide our curated artifact-release narrative and lineage explanation. GPU profilers such as Nsight answer kernel/timeline questions; they complement rather than replace experiment tracking.

## What NVIDIA and other labs publicly expose

- **NVIDIA NeMo RL:** [logger design](https://docs.nvidia.com/nemo/rl/latest/design-docs/logger.html) integrates W&B, TensorBoard, MLflow, and SwanLab, including distributed metrics and system measurements. There is no single NVIDIA-only tracker required by this framework.
- **NVIDIA Megatron Bridge:** [logging documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/logging.html) likewise documents multiple logger integrations.
- **PyTorch/Meta-associated TorchTitan:** its [public project](https://github.com/pytorch/torchtitan) lists TensorBoard or W&B for loss, memory, throughput, FLOPs, and utilization. This is project evidence, not proof about every Meta team.
- **AllenAI OLMo-core:** its [training example](https://github.com/allenai/OLMo-core/blob/main/src/examples/llm/train.py) has optional W&B and Comet callbacks; supported options are not company-wide policy.
- **Google DeepMind XManager:** its [repository](https://github.com/google-deepmind/xmanager) covers packaging, launching, and tracking experiments, but explicitly excludes the server/UI needed for its cloud private preview. It is not an off-the-shelf public dashboard replacement.

These examples support keeping metric production independent from presentation. They do not justify claims about undisclosed internal tooling at OpenAI, Anthropic, Google, or other organizations.

## Proposed visualization layout

1. **Campaign overview:** accepted coverage, deferred scope, model/dataset/source revisions, verification protocol, hardware, and artifact links. Separate inherited and new updates.
2. **Rollout outcomes:** correct and faster-correct fractions for every fully scored group, including weak groups, with group denominator and sampled policy. Accepted-only curves hide failed attempts if presented without this context.
3. **Kernel performance:** conditional-on-correctness speedup distributions, slower candidates, and ATen-delegation caveats. If custom-kernel provenance is absent, label it unknown rather than inventing a success rate.
4. **Optimization:** native loss and actually recorded entropy/gradient/clipping metrics versus accepted optimizer step. Loss is not accuracy. Do not fabricate reference KL or missing instrumentation.
5. **Verification reliability:** recorded compile, contract, numerical, candidate-crash, and infrastructure outcomes, retaining original attribution and independent repair receipts.
6. **Comparisons:** explicit hardware/source/verifier boundaries and meaningful wall-time or optimizer-step axes. Genuine improvement claims require fixed held-out evaluation, not changing-task training outcomes.

Every panel needs units, definitions, legends, source/date captions, and accessible labels. Show unsmoothed values by default; label optional smoothing and do not synthesize uncertainty bands without repeated measurements. Offer reviewed scalar-data downloads, not raw private archives.

## Publication acceptance criteria

- Build a new dataset/database from allowlisted scalar keys and reviewed config fields. Do not copy or sync the complete private SQLite archive.
- Exclude raw requests/replies/reasoning, credentials, user identifiers, local paths, VM addresses, unreviewed attachments, and optimizer pickle files.
- Deduplicate inherited history using stable campaign/branch/update IDs. Preserve the old shortcut-reward branch separately; identical checkpoint numbers across branches are not the same state.
- Keep rollout/final-score history separate from accepted-only optimizer curves; preserve missing values and sampled-policy versus post-update-checkpoint identities.
- Compare exported counts and representative values with immutable backups. Test unknown-field rejection, nonfinite values, duplicate history keys, and inherited-point inflation.
- Preview locally and inspect both pages and downloadable backing files. Verify read-only behavior and access controls independently of chart appearance.
- Publish only after content/visibility review, then link immutable dashboard, dataset, and model releases from the README. Preserve private backups.

## Review sequence

The weight-pointer/publication-checklist PR is documentation-only. The HF curriculum compatibility PR changes code and must not be treated as cosmetic. This decision PR changes no logger, trainer, database, or visibility. A follow-up implementation PR should add the sanitized export and a local Trackio frontend prototype; actual publication or W&B migration remains a separate decision.

Before making the source repository public, review its reachable history as well as its current files. Deleting a file in a new commit does not erase earlier versions. Source, model, dataset, and dashboard publication each need an explicit scope and appropriate attribution.
