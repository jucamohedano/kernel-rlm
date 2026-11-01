# CuTe in-context-learning pilot

## Scope

Test whether a pinned Laguna XS-2.1 checkpoint can use relevant Python CuTe DSL
information supplied directly in its prompt. This is not an RLM run: generation
has no tools, REPL, child sessions, retrieval or compiler feedback. Implementing
or preparing the pilot does not authorize live generation, validation on GPUs,
training or another rental.

## Fixed comparison

Twelve stateless FP32 tasks, four operation families, and three prompt conditions:

- `task_only`: operation, interface and input/output contract.
- `api`: the same task plus relevant version-pinned API notes.
- `api_example`: the same API notes plus a related worked example.

One completion per task/condition gives 36 outputs. Do not automatically regenerate
failures, repair malformed answers, select a replacement output, or use compiler
feedback to produce another completion. An explicitly authorized rerun needs a
new run directory and separately reported results; original evidence stays
unchanged. `generate --entry-id ID` selects entries (repeatable, in supplied order)
from the frozen 36-entry plan. Omit it to attempt all entries. A total request
deadline is recorded before moving to the next selected entry; a connection or
other HTTP transport failure stops the batch. Neither case retries an entry.
Seed pairing and deterministic interleaving are recorded in the prepared plan.
The first screen is descriptive, not a reliable estimate of generalization or
statistical significance.

Target Python CuTe DSL 4.8.1 and NVIDIA L40S/SM89. Hardware-dependent instructions
from Hopper/Blackwell must not enter an SM89 API packet. API/examples must have
recorded provenance and license terms; source visibility is not a training
license. Preserve NVIDIA notices and the distinction between BSD examples and
CuTe implementation files covered by different terms.

## Prompt quality and leakage

Review task interfaces, specified shape domains, dtype/layout assumptions and
API-packet sufficiency before a live run. An insufficient API packet makes a
failed completion inconclusive about in-context-learning ability. CPU tests do
not certify that a kernel example compiles on the target GPU.

Do not put oracle code, expected outputs or an equivalent target implementation
in the prompt. A worked example must teach a related API pattern rather than
merely rename the target computation. Relatedness is recorded, not assumed to be
a family-held-out generalization test. Keep original examples and close
variations in the same family when constructing later SFT/evaluation splits.

## Independent generation and grading

Freeze rendered prompts, task definitions, API/example contents, generation
settings and source hashes before dispatch. Every model request is a fresh
system/user conversation with no tools field. Preserve raw SSE before parsing,
reasoning if emitted, selected-token log probabilities, final usage and partial
failed streams. Missing final usage is unknown, not zero. Never execute partial
tool arguments or arbitrary generated code in the host process.

Offline reference and candidate execution are separate sandbox phases. The
candidate cannot read reference outputs. Task inputs are reconstructed from the
same declared case and fixed data seed; positive strides are constructed
explicitly instead of losing their contract through serialization. Generated
code must define the exact `ModelNew` interface.

Report independent outcomes:

1. Source extraction/parsing and interface admission.
2. Candidate execution and synchronization.
3. Observed CuTe compilation and completed invocation, with retained artifacts.
4. Numerical agreement with independent Torch reference outputs.
5. Source review of whether CuTe performs the required output computation.

Numerical agreement and a CuTe launch together do not prove the computation used
CuTe: Torch could compute the answer while a dummy kernel launches. Keep
`valid_solution` unknown and source review required rather than manufacturing a
full-CuTe certificate. Torch-only implementations must not count as CuTe
solutions. This pilot has no timing, speedup or Nsight requirement.

## Interpretation

A benefit from API notes or examples supports a practical in-context remedy; it
does not show persistent knowledge or establish the benefit of SFT. A gain only
with worked examples can indicate missing implementation patterns rather than
missing signatures. No improvement is inconclusive until context sufficiency,
output truncation and environment compatibility have been checked.

Report per-task paired outcomes and family breakdowns. Do not count shape cases
as independent implementation tasks. Larger sample batches and training require
separate authorization. Repository examples are seeds, not already-complete
agent trajectories; generate actual teacher interactions instead of inventing
retrospective reasoning or tool observations.

## Relationship to prior runs

This pilot is independent of the completed compile-forward RLM diagnostics.
Numerical validation here does not retroactively score those runs. Their raw
artifacts and reported outcomes remain unchanged. The two-L40S VM was explicitly
kept by the user; do not terminate it or launch this pilot merely because this
code exists.
