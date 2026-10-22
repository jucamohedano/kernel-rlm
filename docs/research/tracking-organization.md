# Trackio organization and evaluation metrics

## Diagnosis

The existing Trackio project is `laguna-kernel-rlm`. It contains several kinds of records, not one uniform kind of training run:

- The RTX coordinator uses the L1 execution `run_id` and logs collection, scoring, budget, and update progress. Historically it also replayed inherited losses with `history/inherited=1`.
- Each native TRL process performs one optimizer update and has its own Trackio run, historically named `<run_id>-step-N`. Its local Trainer step counter resets; it is not the global policy step.
- Collection and final-scoring processes log separate phase records.
- The learning-curve observer and JSONL imports create derived views. They are not additional training executions.

The final-pass local snapshot confirms this distinction: the coordinator has 959 metric rows, but the pass added only five optimizer updates after 30 inherited updates. Metric rows are observations, not update counts. Historical data remains unchanged; its inherited flags must remain visible to readers.

## Target organization

Keep TRL/Trackio, and connect its records to the artifact manifest rather than creating a competing metric store.

```text
Stage 1 / experiment record
  execution_run_id = exact L1 run_id
    coordinator: name = execution_run_id
    collection/scoring: producer + exact group/attempt identity
    native trainer: producer + exact group/attempt/update identity
      trainer_local_step (process-local)
      policy_global_step (checkpoint lineage position)
    episodes -> turns -> model/tool/child calls -> token artifacts
    admitted training rows -> masks/token positions -> optimizer update

Derived lineage curve: separately labeled, references source execution IDs
```

An execution is a launch/workload, not a root turn or optimizer update. Preserve separate IDs for those objects. Repeated native trainer processes must not share a Trackio name and overwrite/reset each other's step counters. A resumed process may resume its existing record only when it represents the same execution/update identity; a new retry needs a distinct attempt identity.

New coordinator runs should log new work only. Preserve parent checkpoint/run and history artifact references instead of replaying inherited points as current work. Derived historical curves may include inherited points when explicitly labeled. Keep `space_id=None`; this proposal does not authorize remote publication or rewriting historical databases.

## Metrics and populations

Separate four kinds of measurements:

1. **Training:** native loss, available entropy/clipping/gradient metrics, learning rate, sampled policy, reward distribution, group advantages, admitted/excluded counts, real loss-bearing root tokens, and optimizer update identity. Do not manufacture reference KL when the objective does not compute it.
2. **Execution cost:** root/child input/output/total tokens with coverage; admission/request/verifier times; recorded run elapsed time. Repeated prompt context contributes to inference usage. Summed request durations are not wall-clock with concurrency; verifier stage timers can overlap.
3. **Training-rollout outcomes:** submitted/compiled/correct/faster counts with sampled/scored/unknown/excluded denominators, candidate and verifier identity. These describe changing training tasks, not held-out learning gains.
4. **Fixed evaluation:** correctness and speed metrics over a declared task set, frozen policy, sample selection rule, GPU, reference baseline, and verifier protocol.

### KernelBench-derived definitions

The original KernelBench paper defines `fast_p` as the fraction of the full evaluated task set whose selected candidate is correct and has `reference_runtime / candidate_runtime > p`. The official implementation uses strict `>`: a correct exactly-1x candidate does not qualify for `fast_1`.

- `fast_0`: correctness rate for a fully assessed task set.
- `fast_1`: correct and strictly faster than reference.
- `fast_2`: correct and strictly more than 2x faster.
- Correct-only geometric mean speedup: include slow correct candidates and report the number included. It does not penalize incorrect candidates and must accompany correctness coverage.
- `pass@k`: probability of at least one correct result among k samples, using the declared sampling protocol and adequate samples per task. This is not a speed metric. A speed-qualified best-of-k metric needs its own stated criterion.

Do not silently exclude missing/infrastructure-error outcomes from denominators or label them numerical failures. Show evaluated, missing, and infrastructure counts; with incomplete assessment, label a partial result or bounds rather than a complete benchmark score. Do not aggregate different GPUs/verifier versions/checkpoints as a controlled comparison. Repeated tasks and acceptance filtering must not bias the population.

Our Stage 1 tasks are a CUDA Agent Ops-6K-derived curriculum, not the original KernelBench task set. Use the above definitions as explicitly labeled local curriculum metrics, not a claim of a reproduced KernelBench score. There is no single universal timing setup across the original paper and the evolving repository.

### Additional evidence to retain

Record the dataset/taskset revision and task hash, policy/tokenizer revisions, candidate hash, baseline (eager/compiled), GPU and software versions, correctness trials/seeds/tolerances, timing method, warmup/repetition counts, summary statistic and dispersion or raw timing samples. Keep compilation/contract/numerical/candidate-crash/infrastructure outcomes separate. Preserve intermediate feedback separately from final grading.

Token-to-update attribution requires exact token artifact references and source call IDs, plus the training row/token-position/loss-mask and update consumption mapping. Usage totals alone cannot establish it. Child outputs are not directly trained root outputs; repeated context, padding, replay and repeated consumption remain distinct. Historical missing parent links and loss attribution must not be backfilled by guesses.

## Sources

- [KernelBench paper, arXiv:2502.10517](https://arxiv.org/abs/2502.10517), especially metric definitions and evaluation appendix.
- [Official score implementation](https://github.com/ScalingIntelligence/KernelBench/blob/main/src/kernelbench/score.py).
- [Official evaluation analysis](https://github.com/ScalingIntelligence/KernelBench/blob/main/scripts/benchmark_eval_analysis.py).
- [Official timing implementation](https://github.com/ScalingIntelligence/KernelBench/blob/main/src/kernelbench/timing.py). Pin an exact revision for a reproduction; `main` evolves.
- Local implementation: `kernel_trl/rlm_update.py`, `rtx_curriculum.py`, `collect.py`, `scoring.py`, `learning_curves.py`, and `modal_app/trackio_metrics.py` under `training/environments/kernel_rlm/`.

This document describes the intended organization and metric definitions. Implementation and remaining gaps are tracked in issue #8; it does not claim that every field or metric is already produced.

## Current PR scope and remaining work

The first implementation exposes archived token/duration coverage and verifier timings, separates coordinator/history roles, corrects admitted export selection, and carries execution/call identifiers through the active TRL collection and deferred-scoring paths. Native trainer names use explicit identity when supplied; metadata is also retained in reports. This is not a full Trackio migration or retroactive database rewrite.

Remaining issue #8 work includes a fully versioned manifest/event contract, immutable tokenizer/model revision attestations throughout the pipeline, complete candidate-to-verification event linkage, validated native Trackio config propagation for all producers, and fixed-evaluation metric panels. This PR documents KernelBench-derived metrics; it does not introduce a new benchmark run or claim benchmark results. Full GPU/inference end-to-end validation still requires a separately authorized run. Historical missing links remain unknown.
