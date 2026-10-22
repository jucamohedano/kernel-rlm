# Evidence reading and reporting

These notes apply to present evidence review and reporting. They are not a full training protocol or a proposal to rerun training, change the verifier, or automatically execute decisions found in historical records.

## Reading evidence

- Treat the campaign as **35 accepted task updates and 15 deferred tasks**. Do not describe it as all-50-task completion.
- Keep provenance attached to each reported result: task, run, checkpoint/lineage, hardware, verifier, and source artifact where available. Missing fields are unknown, not inferred.
- Report accepted outcomes as verifier outcomes, not proof of custom CUDA: some passing candidates delegate computation to ATen.
- Do not infer held-out improvement, broad generalization, mastery, or causal hardware effects from this campaign.
- Distinguish current protocol from historical guide text. Historical decisions are context, not automatic next-experiment instructions.

## Reviewing a new claim

State the exact claim and the evidence that supports it. Link to the underlying run/artifact and identify limitations or unavailable fields. Separate measured observations from interpretation. If evidence is insufficient, say so rather than filling gaps with assumptions. Any new experiment requires a separately reviewed question, protocol, and authorization; this document launches nothing.

## Artifact and privacy boundary

The checkpoint archive remains private. A planned public release is not a published release. Follow the separate [publication checklist](artifact-publication.md) before any release or hosting decision. The local dashboard is read-only and should be configured with an explicit evidence root; it does not publish data.
