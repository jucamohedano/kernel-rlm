# Public research artifacts

This checklist separates a public source repository, a model release, a dataset release, and a metrics dashboard. Changing GitHub visibility does not publish a private Hugging Face model, and a link to private weights is not a public release.

## Checkpoint35 adapter release

The first completed L1 campaign ended with 35 accepted task updates and 15 deferred tasks. Its final accepted checkpoint is preserved at immutable archive commit `833f460dfbd71918a0ad7403e1245233113d4b71`, directory `checkpoints/20261020T230200Z-final-l1-pass/checkpoint-step-35`. This is the corrected reward lineage, not the historical shortcut-reward checkpoint28 on a different branch.

Before creating a separate public model release:

- Review the pinned base model's license and conditions, dataset attribution, and grant acknowledgment requirements.
- Export only the reviewed adapter weights and a portable adapter configuration. Do not expose the whole private checkpoint archive or optimizer/RNG pickle files.
- Set the portable base reference to `poolside/Laguna-XS.2`, and document its exact revision `a397bde04501591d0ac3d74a5d9d0a4ec298ffb8`; do not present adapters as a full model.
- Preserve tensor hashes and record any metadata-only configuration changes separately. Test loading the released adapter format before claiming it is ready for inference.
- Include a model card describing training inputs, rank8/alpha16 attention adapters, the final immutable execution source, verifier limitations including ATen delegation, deferred coverage, and the absence of held-out improvement evidence.
- Keep all original private backups and branch histories. Publish a new curated model repository rather than changing the archive's visibility.
- Replace the README's private archive pointer with the new immutable public release link only after the release is verified.

The archive's uploaded file addresses and sizes were verified against the remote commit. This does not mean a full remote binary download readback was performed. Local checkpoint file hashes and native readback were verified separately.

## Public metrics dashboard

Do not upload the complete private Trackio database unchanged. Build a new publication dataset from an explicit allowlist of scalar metrics and reviewed run metadata. Exclude credentials, environment dumps, local paths, VM addresses, user identities, raw requests/replies, reasoning traces, and unreviewed source attachments.

Keep lineage identifiers so inherited optimizer history is not counted as new training in each continuation. Separate rollout outcomes, accepted updates, weak deferred groups, diagnostics, and held-out evaluation. Mark the verifier/source/hardware boundaries and historical shortcut-reward branch explicitly. Do not fill absent metrics with zero, call probability agreement reference KL, or present changing-task curves as causal learning evidence.

Preview the exported dashboard locally, verify representative values against the immutable backups, and obtain publication approval before changing artifact visibility. A static dashboard is a read-only snapshot, not live monitoring.

## Dataset release

Publish a versioned materialization of the exact selected task rows and order, not a freshly sampled approximation of the upstream source. Preserve original task IDs, reference source bytes, constructor behavior, and original JSONL hashes. Different levels and orderings are configurations, not invented train/test splits. The easy-first L1 ordering contains the same selected tasks as the original L1 configuration.

Attribute [CUDA-Agent-Ops-6K](https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K), retain its published CC-BY-4.0 attribution information, describe sampling/reordering/format conversion, and do not invent an upstream source revision that was not recorded. Review the derivative dataset card before public release. Keep historical JSONL loading operational while adding Hugging Face distribution.

## Pull request boundaries

1. Documentation, attribution, and private weight pointer: no training or deployment changes.
2. Dataset packaging/download compatibility: code and tests; no rewards, shapes, sampling, or order changes.
3. Tracking/publication design: compare tools and define export acceptance criteria before deploying or replacing the logger.

Merge reviewed changes normally. Do not rewrite historical commits or automatically make repositories, weights, datasets, or dashboards public.
