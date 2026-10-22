# First campaign: building and completing an L1 pass

## Why this project exists

Recursive Language Models (RLMs), introduced by [Alex L. Zhang, Tim Kraska, and Omar Khattab](https://arxiv.org/abs/2512.24601), let a model work with context through a programmatic REPL and recursive language-model calls. Read the [paper](https://arxiv.org/abs/2512.24601) or [project blog post](https://alexzhang13.github.io/blog/2025/rlm/) for the original idea. This project explores a different application of that harness: asking a model to produce CUDA kernels, then checking candidates against a reference. The aim of the first campaign was to establish a training and verification path over a bounded L1 task set—not to demonstrate general kernel-writing ability.

## From early trials to native training

The project first explored Prime/Modal GRPO trials, then moved to a native TRL/Muon route. In the completed native route, training credited exact sampled root tokens; child outputs were context, not loss targets. TRL was a later route, not the campaign’s origin. Tasks exposed constructor state as well as runtime tensors, so a candidate needed to reproduce the reference interface and implement its operation through a compiled CUDA extension. This motivated a verifier that checked correctness before benchmarking, and made constructor-aware task contracts important; see the [stateful task notes](../../training/environments/kernel_rlm/kernel_rlm/STATEFUL_TASKS.md).

The first bounded training attempt was a setup milestone, not a quality result. The later campaign continued from preserved checkpoints and revised guidance. The immutable continuation code restricts what could change, including child-tool and program guidance; see [checkpoint continuation](../../training/environments/kernel_rlm/kernel_trl/checkpoint_continuation.py). These changes improved the guidance and the resulting usage of child calls, but that is not evidence that delegation improved correctness or performance. A call being available or used is not itself a benefit.

## The hardware handoff

The native TRL campaign ran first on L40S GPUs and later continued on RTX PRO 6000 GPUs. Earlier Modal trials used A100 hardware. These were sequential campaign stages, not a controlled hardware comparison. Do not attribute changes in outcomes to the GPU generation or treat the stages as an isolated benchmark.

## Why the verifier caveat matters

A passing candidate could satisfy the extension-use check while delegating computation to ATen. Such outcomes are verifier passes, but do not establish that the candidate implemented the operation as a custom CUDA kernel. Separately, a BatchNorm identity shortcut was found: default-state BatchNorm could behave as an identity, allowing a trivial candidate to exploit the reward. A targeted verifier change made nondefault BatchNorm weights, bias, and running statistics reject that saved identity shortcut; genuine CUDA controls passed. The corrected branch rebuilt an update from all eight original samples and separately validated terminal rewards, while preserving the old checkpoint 28 and scores. This is distinct from the ATen limitation: one is a specific reward shortcut that was patched; the other is the broader fact that extension use did not guarantee all computation was custom CUDA. The final result belongs to the corrected branch, not the old shortcut-reward checkpoint.

## What finished

The final L1 pass recorded **35 accepted task updates and 15 deferred tasks**. The final run included **30 inherited updates and 5 new updates**; this describes checkpoint continuation, not 35 fresh updates in that one run. Thus the campaign did not complete all 50 tasks. The final accepted adapter is checkpoint-step 35 at private archive commit [`833f460dfbd71918a0ad7403e1245233113d4b71`](https://huggingface.co/jucamohedano/laguna-xs2-kernel-rlm-checkpoints/tree/833f460dfbd71918a0ad7403e1245233113d4b71/checkpoints/20261020T230200Z-final-l1-pass/checkpoint-step-35). It requires `poolside/Laguna-XS.2` revision [`a397bde04501591d0ac3d74a5d9d0a4ec298ffb8`](https://huggingface.co/poolside/Laguna-XS.2/tree/a397bde04501591d0ac3d74a5d9d0a4ec298ffb8); these are adapter weights, not a full model. The archive is private and access-controlled. A planned release is not a published release; see the separate [artifact publication checklist](artifact-publication.md).

## Closing and preservation

The work concluded with a short closure after the final run and provider shutdown. The historical closure record reports a verified backup of 9,504 regular files, five symlinks, 44 native checkpoint manifests, and consistent SQLite integrity. Receipts and archive contents are preserved, but this summary has not rechecked them. The closeout also records that no GPU jobs remained and the rental was released.

This research was supported by a **CloudRift compute grant**; see [ACKNOWLEDGMENTS.md](../../ACKNOWLEDGMENTS.md) for the full acknowledgment and caveat. The archive and receipts preserve source and execution records; this short account does not replace them. No held-out improvement or broad generalization is established. Given the ATen shortcut caveat and 15 deferred tasks, neither “all tasks solved” nor “custom CUDA mastery” is an accurate summary.

## For contributors

To inspect the implementation without launching compute, start at the [kernel environment README](../../training/environments/kernel_rlm/README.md), then read its task/verifier code in `training/environments/kernel_rlm/kernel_rlm/`, the [stateful task contract](../../training/environments/kernel_rlm/kernel_rlm/STATEFUL_TASKS.md), and the [TRL training notes](../../training/environments/kernel_rlm/kernel_trl/RLM_TRAINING.md). Find relevant tests under `tests/` and the kernel environment before editing; run narrow CPU-compatible tests with the existing project environment. A dashboard or source review requires no GPU run. Keep the [evidence-reading guidance](protocol.md) in mind when describing results. The [dated legacy guide](../../training/environments/kernel_rlm/dashboard/guide.html) remains available as historical context; its past decisions are not automatic recommendations for new work.
