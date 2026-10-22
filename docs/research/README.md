# Start here: kernel research

This repository combines the Recursive Language Models (RLM) framework with a CUDA-kernel training environment. RLMs were introduced by [Alex L. Zhang, Tim Kraska, and Omar Khattab](https://arxiv.org/abs/2512.24601); see the [paper](https://arxiv.org/abs/2512.24601) and [project blog post](https://alexzhang13.github.io/blog/2025/rlm/). The core idea is to let a language model inspect and operate on context through a REPL, including recursive language-model calls. This project applies that harness to generating and verifying CUDA kernels. It is an independent extension, not a claim of authorship over the original RLM work.

## Find your way around

- [First campaign](first-campaign.md): what was built, how the campaign changed, and what its results do—and do not—show.
- [Evidence reading and reporting](protocol.md): how to describe observed outcomes without overstating them.
- [Dashboard guide](../../dashboard/README.md): browse explicitly selected local evidence without launching compute.
- [Kernel environment](../../training/environments/kernel_rlm/README.md): task, verifier, and training code orientation.
- [Training implementation notes](../../training/environments/kernel_rlm/kernel_trl/RLM_TRAINING.md) and [stateful task contract](../../training/environments/kernel_rlm/kernel_rlm/STATEFUL_TASKS.md): details for code-level investigation.
- [Dated legacy guide](../../training/environments/kernel_rlm/dashboard/guide.html): preserved historical record, not current instructions by default.

## Contributor orientation

The framework lives in `rlm/`; kernel task and verification code is under `training/environments/kernel_rlm/kernel_rlm/`; training integration is under `training/environments/kernel_rlm/kernel_trl/`. Start by reading the task contract and verifier before changing prompts or training behavior. Existing tests are under `tests/` and the kernel environment's test directories; identify the narrow relevant test and run it in the project's existing environment. CPU-only checks are available for contracts that do not require CUDA. Do not launch a GPU job simply to browse or understand the project. Training and cloud execution require their own explicit authorization and environment setup.

The first campaign is a historical result, not a recommendation to repeat its launch plan. The final checkpoint archive is private, and this repository makes no claim of held-out improvement. Keep claims tied to their specific evidence and preserve the distinction between current guidance and the historical archive.
