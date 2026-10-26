# Literature review for Kernel RLM

A kernel can pass a test without doing the right computation. A training curve can rise without showing a better policy. And an agent can call children without learning useful delegation. This review follows those distinctions through our kernel-generation pipeline.

Start with the thematic sections for the story. Open a source record for its finding, project connection and limits. These are research connections, not claims that we reproduced each method or obtained its results.

![Research map connecting recursive agents, kernel verification, root-policy training and held-out evaluation](/literature/research-map.svg)

Original project schematic: how the research questions connect, not a comparison of paper results.

Last reviewed: **25 October 2026**. The year below is the first preprint/report year unless a conference year is stated. A preprint label means no independent venue was verified for this review, not that the work is necessarily unpublished. Review depth is explicit; a reading lead is not a methods review.

## A root agent, a REPL, and the choice to delegate

Imagine the root agent inspecting an operator, writing a CUDA candidate, and revising it after execution feedback. Recursive Language Models supplies the persistent-REPL foundation; Recursive Agent Optimization asks how root and child trajectories receive training credit. Our current loss targets exact root outputs, not child outputs, although shared weights can still change child behavior indirectly. The agent-scaling studies add a necessary control: useful delegation must be weighed against actual root-plus-child tokens, tool calls and elapsed time.

![RLM paper Figure 2: the prompt lives in a REPL variable; the model inspects it and makes recursive calls](/literature/rlm-figure-2.png)

Figure from **Recursive Language Models**, arXiv:2512.24601v3, Figure 2. Alex L. Zhang, Tim Kraska, and Omar Khattab. [Original figure and caption](https://arxiv.org/html/2512.24601v3#S1.F2) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Image unchanged; scaled for display. This architecture does not imply that our child outputs receive direct policy loss.

### Recursive Language Models

[Primary source](https://arxiv.org/html/2512.24601v3) · 2025. **Status:** arXiv preprint; v3 inspected (11 May 2026); no venue asserted

**Review:** Prior primary review + current abstract metadata and targeted §2/§3.2/Appendix A reinspection **Location:** §2; §3.2 fine-tuning; Appendix A Additional Training Details

**Source finding:** Treats long input as external environment data accessed through a persistent REPL and recursive LM calls. Its 1,000-filtered-trajectory experiment distills teacher root turns with their full histories as SFT samples. Appendix A separately describes an MRCRv2 RL experiment.

**Project connection:** Architectural and upstream-code foundation for persistent REPL, external context and recursive delegation; motivates focus on root behavior without claiming identical training.

**Limits:** The 1,000-teacher-trajectory result is supervised distillation, not this project’s direct GRPO baseline. Do not imply the entire paper contains no RL. Its root-training motivation is not evidence that child policy loss is used here.

### Recursive Agent Optimization

[Primary source](https://arxiv.org/html/2605.06639v2) · 2026. **Status:** arXiv preprint; v2 (2 October 2026); no venue asserted

**Review:** Prior primary full-text section review retained; current primary abstract metadata rechecked **Location:** §2.2.1–2.2.2; §4; §5.0.1/Fig.8; §5.0.2/Fig.9

**Source finding:** Jointly optimizes root and child trajectories using node-local success signals and depth-level weighting. Examines child-success-rate delegation reward; a success-count bonus encouraged unnecessary easy delegation.

**Project connection:** Direct methodological contrast with the supplied root-token-only project objective. Low child use alone cannot separate low utility, limited exploration and lack of direct child credit.

**Limits:** “Only Root Rewards” still trains child tokens with propagated root reward; it is NOT a root-token-only-loss ablation. Shared weights can change child behavior indirectly even without direct child loss. Evidence is not CUDA; implementation or new ablation plans belong in issues, not claimed as completed.

### Towards a Science of Scaling Agent Systems

[Primary source](https://arxiv.org/html/2512.08296v3) · 2025. **Status:** arXiv preprint; v3 (8 April 2026); no venue asserted

**Review:** Prior primary review + current primary metadata and targeted setup/results/controls inspection **Location:** §3; §4.1–4.4; §5; Appendix E.4

**Source finding:** Controlled comparisons vary agent architecture while maintaining task/tool controls and computational budgets; coordination benefit depends on domain and task structure, and coordination can add substantial overhead.

**Project connection:** Supports budget-aware comparison of sequential root work and delegation, with actual tokens, tool calls and elapsed time reported rather than assuming more agents imply better decomposition.

**Limits:** Tool-using benchmark evidence is not CUDA-kernel evidence. Equal budget controls do not imply equal actual consumed compute; reported thresholds/effect sizes are not universal laws. Latest v3 has expanded benchmarks/results, so avoid copying unversioned older headline statistics.

### Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets

[Primary source](https://arxiv.org/html/2604.02460v2) · 2026. **Status:** arXiv preprint; v2 (11 April 2026); no venue asserted

**Review:** Prior primary section review retained; current primary abstract metadata rechecked **Location:** §4–5; Appendix C

**Source finding:** Compares single-agent and multi-agent reasoning systems under thinking-token upper caps and discusses accounting limitations.

**Project connection:** Extra child tokens must not be mistaken for architectural benefit; literature supports distinguishing budget caps from actual root-plus-child usage.

**Limits:** Text-only multi-hop reasoning; tools and vision outside scope. Caps are not enforced equal consumed tokens; Gemini accounting is approximate. The title must not be repeated as a universal claim about CUDA agents.

## What does a passing kernel actually prove?

Suppose a candidate passes because a module’s default state makes an operation look like an identity. Our BatchNorm repair closed one such shortcut, but did not prove that every required operation ran in custom CUDA. KernelBench connects correctness and speed; the verification studies then examine what finite tests miss, including state coverage, numerical contracts and semantic hacks. Their stronger checks are research references, not suites we can assume this project has implemented.

![KernelBench paper Figure 1: generate a kernel from a PyTorch task, check correctness, and measure performance](/literature/kernelbench-figure-1.png)

Figure from **KernelBench: Can LLMs Write Efficient GPU Kernels?**, arXiv:2502.10517v1, Figure 1. Anne Ouyang, Simon Guo, Simran Arora, Alex L. Zhang, William Hu, Christopher Ré, and Azalia Mirhoseini. [Original figure and caption](https://arxiv.org/html/2502.10517v1#S1.F1) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Image unchanged; scaled for display. A benchmark diagram is not evidence that our campaign reproduced its evaluation.

### KernelBench: Can LLMs Write Efficient GPU Kernels?

[Primary source](https://arxiv.org/html/2502.10517v1) · 2025. **Status:** ICML 2025; verified in official PMLR proceedings.

[Official venue record](https://proceedings.mlr.press/v267/ouyang25a.html).

**Review:** Targeted full-primary-text review of the named sections; not a reproduction or exhaustive paper review. **Location:** §3.3 metric design; §4.4 hardware variation; §5.1 repeated sampling and iterative refinement; Appendix B evaluation methodology.

**Source finding:** fast_p measures the fraction of assigned workloads whose generated kernels are correct and exceed speed threshold p relative to the stated reference. Hardware, baseline and sampling/refinement budgets determine the meaning of the score.

**Project connection:** Provides the task-level correctness-and-speed metric for interpreting this project; its changing-task, accepted-group training outcomes are not this benchmark score.

**Limits:** fast_1 is a speed threshold, not pass@1. Original finite correctness checks are not a certificate of general semantics, and current project results are not held-out policy evaluation.

### Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization

[Primary source](https://arxiv.org/html/2509.14279v1) · 2025. **Status:** arXiv preprint; no independent conference venue verified in this review.

**Review:** Targeted full-primary-text review of the named sections; not a reproduction or exhaustive paper review. **Location:** §3 robust-kbench; §4 soft verification; §§5–6 agentic search; Appendix A.1 compromised-task filtering.

**Source finding:** Adds varied initialization states and input configurations, forward/backward workloads and multiple timing/profiling checks. LLM verification is a screening stage before hardware verification; its own reported aggregate speedup decreases from 3.13× to 1.49× after compromised tasks are excluded.

**Project connection:** An earlier verifier-loophole reference in the guide, directly relevant to default-state identity shortcuts and weak baselines. Passing the project’s current tests does not mean a candidate has passed robust-kbench.

**Limits:** The 3.13×→1.49× change is its experiment, not a correction factor for this project. Its LLM screening is fallible, and benchmark heuristics are not universal correctness thresholds.

### KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch?

[Primary source](https://arxiv.org/html/2607.16241v1) · 2026. **Status:** arXiv preprint; no independent conference venue verified in this review.

**Review:** Targeted full-primary-text review of the named sections; not a reproduction or exhaustive paper review. **Location:** §§2.2–2.5 TF32, hidden tests, memory and metrics; §3.4 audit; §4 limitations; Appendix L.4 residual hacks.

**Source finding:** Uses a TF32-enabled reference, four value distributions and peak-memory metrics while retaining fixed task shapes. Four genuine semantic hacks survive the expanded finite tests; two other audit flags are legitimate shape specializations.

**Project connection:** The guide already cites it as a verifier-audit reference, not proof that its complete suite is implemented. The project’s BatchNorm state checks and CUDA execution guard address narrower failure modes.

**Limits:** Four deterministic transforms do not prove semantics or vary module lifecycle/state, and feedback can leak hidden cases in a multi-turn loop. TF32 must respect the task numerical contract; the arXiv page unusually lists 26 June 2026 for a 2607 identifier, so no month is inferred from the ID.

### The Correctness Illusion in LLM-Generated GPU Kernels

[Primary source](https://arxiv.org/html/2606.20128v2) · 2026. **Status:** arXiv preprint; v2 reviewed, with no independent conference venue verified.

**Review:** Targeted full-primary-text review of the named sections; not a reproduction or exhaustive paper review. **Location:** §§3.1–3.5 schemas, oracle, corpus and assumptions; §4 evaluation including cross-GPU sweep; §6 limitations.

**Source finding:** Seeded, schema-aware tests expose author-planted transcription bugs that fixed-shape small-sample checks accept, while keeping positive controls clean in the studied corpus. Tests preserve shared symbolic dimensions and record numerical-error distributions against an fp64 mathematical reference rounded to the target dtype.

**Project connection:** Supports distinguishing kernel semantics from the project’s separate sampler/learner probability audit. The guide labels it verification reading, not a reproduced fuzzer or a measured bug rate in this model.

**Limits:** Author-seeded bugs and CPU stand-ins do not estimate deployed-model failure prevalence. Shape expansion is valid only under the declared task contract, and fp64 mathematical semantics need not match the project’s PyTorch implementation contract.

## Iterative kernel search is not held-out evaluation

A multi-turn agent gets several chances to improve a kernel: its final submission, best candidate in one history and best of several trajectories can yield different scores. Kevin and Dr. Kernel make those reporting choices concrete; CUDA Agent and DRTriton connect training data, curriculum and search. TimeRLM adds a persistent-REPL transfer example, but its depth-0 setting cannot demonstrate useful child delegation.

![Separate paths for collecting verified rollouts, training on replayed root outputs and evaluating checkpoints on held-out tasks](/literature/train-evaluate.svg)

Original project schematic: training feedback and held-out evaluation answer different questions.

Our Stage 1 campaign changed tasks, hardware and verification. Its accepted-group curve describes selected training experience, not a controlled base-versus-checkpoint result. No held-out gain is established; [First campaign](first-campaign.md) records that boundary.

### Kevin: Multi-Turn RL for Generating CUDA Kernels

[Primary source](https://arxiv.org/html/2507.11948v1) · 2025 preprint; 2026 conference. **Status:** ICLR 2026; verified in official conference proceedings.

[Official venue record](https://proceedings.iclr.cc/paper_files/paper/2026/hash/86ba836d4c5dd859d795a172911745e2-Abstract-Conference.html).

**Review:** Targeted full-primary-text review of the named sections; not a reproduction or exhaustive paper review. **Location:** §§3.1, 4.1–4.4 and 5.1–5.4; Appendix A task and split construction.

**Source finding:** Uses per-turn training samples with actual histories, drops old long reasoning while retaining summaries/code/feedback, and aggregates discounted future kernel scores. The main evaluation uses 16 trajectories × 8 turns; 82% correctness is best@16, while avg@16 is 46%, and trajectory selection already uses the best kernel within that history.

**Project connection:** An earlier direct CUDA multi-turn-RL reference in the guide. The project’s documented exact-turn replay and final-outcome trajectory credit are not a reproduction of Kevin’s discounted per-turn return.

**Limits:** Training uses 180 KernelBench tasks; evaluation uses 20 reserved original tasks plus 80 newly constructed tasks. The headline is neither single-attempt nor final-submission-only accuracy, and does not demonstrate this project’s held-out quality.

### Dr. Kernel: Reinforcement Learning Done Right for Triton Kernel Generations

[Primary source](https://arxiv.org/html/2602.05885v2) · 2026. **Status:** ICML 2026; verified in official PMLR proceedings.

[Official venue record](https://proceedings.mlr.press/v306/liu26fd.html).

**Review:** Targeted full-primary-text review of the named sections; not a reproduction or exhaustive paper review. **Location:** §3.3 hacking checks; §4.2 TRLOO; §§5.1–5.3 mismatch/lazy optimization; §§6.1–6.3 evaluation and test-time scaling.

**Source finding:** KernelGYM checks reward hacking, while TRLOO, mismatch correction, profiling rewards and profiling-based rejection sampling target separate training failures. Main comparisons sample eight outputs per task with up to three turns; sequential scaling reports last-turn and best-of-history results separately.

**Project connection:** An existing verifier/training reference in the guide; an extension-call guard alone is not proof that all required CUDA computation ran. The guide explicitly does not claim TRLOO or profiling rewards are implemented here.

**Limits:** Its cold-start SFT, Triton and H100 results do not establish transfer to this CUDA/SM120 project. Some baseline comparisons use their best earlier turn, and benchmark-based checkpoint/turn selection is development feedback rather than untouched final-test evidence.

### CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation

[Primary source](https://arxiv.org/pdf/2602.24286) · 2026. **Status:** arXiv preprint; no independent conference venue verified in this review.

**Review:** Targeted full-primary-text review of the named sections; not a reproduction or exhaustive paper review. **Location:** §§3.1–3.3 data, tool loop and RL; §4 and Table 1 evaluation; Appendix A data contamination check.

**Source finding:** Combines synthetic CUDA-Agent-Ops-6K data, tool-based verification/profiling and staged actor/critic training. KernelBench results report pass rate, faster rate and geometric mean speedup against both eager and compiled PyTorch.

**Project connection:** The existing guide credits Ops-6K as a training-data source and an influence on data-separation policy. This does not establish that the project reproduces its PPO, critic, warm-ups or long-context tool protocol.

**Limits:** The AST decontamination threshold of 0.9 excludes measured structural overlap, not every semantic overlap. Abstract faster-rate headlines differ from Table 1, so unqualified headline-rate comparisons are unsafe.

### DRTriton: Large-Scale Synthetic Data Driven Reinforcement Learning for Triton Kernel Generation

[Primary source](https://arxiv.org/html/2603.21465v2) · 2026. **Status:** arXiv preprint; no independent conference venue verified in this review.

**Review:** Targeted full-primary-text review of the named sections; not a reproduction or exhaustive paper review. **Location:** §3 CSP-DAG; §§4.1–4.5 verifier, SFT, rewards, curriculum and search; §§5.1–5.4 evaluation.

**Source finding:** Controls synthetic operator-composition difficulty and advances curriculum when held-out current-level Pass@1 exceeds its threshold. Reports a separate 400-program synthetic evaluation and KernelBench evaluation, with test-time composition search as an additional mechanism.

**Project connection:** The guide records an earlier curriculum/data-design influence, not implementation of CSP-DAG, its SFT warm start or search. Easy-first accepted training groups do not by themselves show transfer to unseen operator compositions.

**Limits:** Results use Triton, synthetic programs and eight H100s, not this project’s direct CUDA/RTX setup. A NeurIPS checklist in the manuscript is not verification of conference acceptance.

### TimeRLM: Recursive Language Models Enable Precise Anomaly Localization in Long-Context Time-Series

[Primary source](https://arxiv.org/html/2608.03391v2) · 2026. **Status:** arXiv preprint; v2 (6 September 2026); no venue asserted

**Review:** Prior primary review + current primary metadata, §4.1 and Appendix A targeted inspection **Location:** §3.1; §4.1–4.4; Appendix A; Appendix D

**Source finding:** Uses external Python access to long time-series and group-relative RL, reports final-checkpoint results on AnomalyXL plus ARFBench and unseen clinical recordings. Uses mean-centered Dr.GRPO advantages without reward-standard-deviation normalization. Appendix A states recursive sub_rlm calls are disabled (depth 0) throughout.

**Project connection:** Related work for RL-trained persistent-REPL agents and held-out/transfer evaluation, not evidence of trained child delegation. Newly reviewed related work should not be relabeled as original implementation provenance.

**Limits:** Time-series anomaly tasks are not CUDA. Its token-cost and forced-final penalties and Dr.GRPO normalization are not claimed implemented here; depth-0 evaluation cannot demonstrate benefits of recursive child agents.

## Which outputs receive learning credit?

A short successful root trajectory and a long debugging trajectory need not receive the same weight under every objective. DeepSeekMath and PPO explain the algorithmic lineage; DAPO and Dr. GRPO distinguish response averaging, realized-token averaging, constant denominators and reward scaling. Those distinctions matter when the training targets are exact root outputs, while child responses and execution feedback remain context. ProRL adds evidence about entropy and reference-policy resets, not a validated reset recipe for our native TRL/Muon pipeline.

### DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

[Primary source](https://arxiv.org/html/2402.03300v3) · 2024. **Status:** arXiv preprint; v3 (27 April 2024); no venue asserted

**Review:** Prior foundational citation + current primary metadata and §4.1 locator verification **Location:** §4.1.1–4.1.4; Appendix A.1.6

**Source finding:** Introduces Group Relative Policy Optimization, estimating relative advantage from sampled response groups rather than a separately trained critic; distinguishes outcome and process supervision.

**Project connection:** Algorithmic origin of the project’s GRPO lineage.

**Limits:** The citation does not mean every original reward, KL regularization, normalization or training-data choice is reproduced; mathematical-reasoning results do not validate the kernel verifier.

### Proximal Policy Optimization Algorithms

[Primary source](https://arxiv.org/abs/1707.06347) · 2017. **Status:** arXiv preprint; v2 (28 August 2017); no venue asserted

**Review:** Prior foundational citation + current primary abstract metadata/abstract; HTML body unavailable (failed conversion) **Location:** Abstract; clipped-surrogate foundation (not freshly full-text reviewed)

**Source finding:** Proposes policy-gradient surrogate objectives with repeated minibatch optimization after interaction-based sampling.

**Project connection:** Clipped-surrogate ancestry for group-relative policy optimization, not a claim that this project runs actor-critic PPO.

**Limits:** Do not label GRPO as the paper’s complete PPO implementation. Current HTML only exposes a conversion-failure placeholder, so no new equation-level or implementation verification is claimed.

### DAPO: An Open-Source LLM Reinforcement Learning System at Scale

[Primary source](https://arxiv.org/html/2503.14476v2) · 2025. **Status:** NeurIPS 2025 main conference; official proceedings verified

[Official venue record](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a4277440d50f1f15d2cb4c14f7e0c0d2-Abstract-Conference.html).

**Review:** Targeted primary-body review; prior 2026-10-20 record and relevant sections rechecked 2026-10-25. **Location:** §3.2 Dynamic Sampling (Eq. 11); §3.3 Token-Level Policy Gradient Loss (Eq. 12); §3.4 Overlong Reward Shaping (Eq. 13)

**Source finding:** Token loss uses the realized group token count, unlike averaging each response first. Dynamic sampling excludes all-zero/all-one correctness groups; overlong loss masking and soft overlong punishment are separate mechanisms.

**Project connection:** An implementation reference for interpreting the root-policy token denominator, unequal response/turn lengths, and selected accepted-group curves. Deferring a weak mixed group is not the paper’s zero-gradient filtering.

**Limits:** The project is not a reproduction of all DAPO techniques. Math-specific truncation treatment does not establish the correct CUDA task-budget objective; child/observation tokens and changed historical prefixes require separate project accounting.

### Understanding R1-Zero-Like Training: A Critical Perspective

[Primary source](https://arxiv.org/html/2503.20783v2) · 2025. **Status:** arXiv preprint; no conference venue asserted

**Review:** Targeted primary-body review; prior 2026-10-20 record and §§3.1–3.2 rechecked 2026-10-25. **Location:** §3.1 GRPO Leads to Biased Optimization, Eq. 3 and implementation box; §3.2 Dr. GRPO: Group Relative Policy Optimization Done Right

**Source finding:** Analyzes per-response length normalization and within-question reward-standard-deviation scaling as distinct reweighting biases. Dr. GRPO removes both and uses a constant loss denominator, unlike DAPO’s realized-token denominator.

**Project connection:** Clarifies how group reward scaling and unequal root lengths affect the intended objective. Citing Dr. GRPO or TimeRLM does not make the project’s current advantage normalization Dr. GRPO.

**Limits:** Main evidence is mathematical reasoning with Qwen2.5-1.5B, not multi-turn CUDA, the project optimizer, or evidence that a normalization change would help. Increased length alone is not demonstrated reasoning improvement.

### ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models

[Primary source](https://arxiv.org/html/2505.24864v1) · 2025. **Status:** arXiv preprint; no conference venue asserted

**Review:** Targeted primary-body review, narrower than a full-paper review; prior record and §2.3.1 rechecked 2026-10-25. **Location:** §2.3.1 KL Regularization and Reference Policy Reset, Eq. 4

**Source finding:** Adds reference-policy KL for its distilled starting model and periodically resets the reference to a recent online policy while reinitializing optimizer state, to address entropy loss and KL-dominated updates during prolonged training.

**Project connection:** A qualified comparison for entropy/KL discussions; rising selected-group rewards alone do not establish the paper’s claim of expanded reasoning boundaries in this CUDA project.

**Limits:** The source starts from DeepSeek-R1-Distill-Qwen-1.5B; its KL/reset recipe is neither claimed implemented nor validated for the project optimizer. Weak groups in a short run are not evidence that reference or optimizer resets are needed.

## Replay the conversation that was actually sampled

A root turn written after compiler feedback must not be trained as though it came from a different history. Exact prefixes, sampled token IDs, behavior log probabilities and loss masks define that replay; they do not guarantee equal probabilities across inference and training stacks. The nondeterminism studies and Beyond Precision examine numerical mismatch, AReaL addresses stale policy versions, and TMax and the tool-calling study show why harness details matter. Serial sampling and learning remove overlap, not necessarily mismatch; our probability audit is also separate from kernel verification.

### Defeating Nondeterminism in LLM Inference

[Primary source](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) · 2025. **Status:** Primary technical research report/blog, Thinking Machines Lab, 10 September 2025; not an arXiv paper

**Review:** Targeted primary-report review; prior 2026-10-20 record and named sections rechecked 2026-10-25. **Location:** Batch invariance and “determinism”; How do we make kernels batch-invariant? (RMSNorm, matrix multiplication, attention); Experiments: Performance; Experiments: True on-policy RL

**Source finding:** Explains why deterministic operations can still be batch-size dependent. In its Bigmath RLVR experiment, unmatched numerics without importance weighting collapse, whereas matched sampler/trainer numerics or importance weighting train smoothly.

**Project connection:** Direct motivation for exact sampled IDs, behavior logprobs, root prefixes and probability checks. Serial inference/learning removes overlap but does not itself remove backend numerical mismatch.

**Limits:** Report-specific evidence, not a theorem that every small mismatch collapses training. Bitwise matching can cost throughput. The project is not claimed to implement these batch-invariant kernels or the report’s complete training stack.

### Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference

[Primary source](https://arxiv.org/html/2506.09501v2) · 2025. **Status:** arXiv preprint; no conference venue asserted

**Review:** Primary-body targeted review performed 2026-10-25; no earlier deep-review evidence found in supplied prior records. **Location:** §3.1 experiment setup and reproducibility metrics; §3.2 Greedy Decoding ≠ Deterministic Output; §3.3 Random Sampling Has Reproducibility Issue, Too; §3.4 runtime-configuration ablation; §4 Near-Perfect Deterministic Reproduction: LayerCast

**Source finding:** Across its tested GPU types/counts, batching and precision settings, greedy outputs and sampled aggregate accuracy can vary. Higher precision reduces but does not universally eliminate divergence; LayerCast stores linear weights in BF16 and upcasts them for FP32 computation.

**Project connection:** Supports reporting actual inference backend, batching, hardware and precision alongside seeds. A greedy transcript is not enough to establish sampler/learner probability equality, and kernel-program correctness remains a separate question.

**Limits:** An inference reproducibility study, not a project RL stability experiment or a guarantee of determinism. LayerCast is not claimed implemented. Configuration-dependent token flips differ from stale-policy data, and this study does not by itself establish cross-stack probability parity.

### Beyond Precision: Training-Inference Mismatch is an Optimization Problem and Simple LR Scheduling Fixes It

[Primary source](https://arxiv.org/html/2602.01826v1) · 2026. **Status:** arXiv preprint; no conference venue asserted

**Review:** Targeted primary-body review; prior detailed record and method/experiment sections rechecked 2026-10-25. **Location:** §3.1 Why LR Scheduler, Eqs. 7–8; §3.2 Traditional Scheduler is Not Feasible; §3.3 Our Method: Decay by Response Length, Algorithm 1; §4.1–§4.3 scheduler, decay-period and importance-sampling experiments; §4.4 FP16 precision ablation

**Source finding:** In Qwen3 math RL, mismatch growth accompanies instability; lower update size and a response-length-informed LR decay stabilize the reported runs. Importance sampling and precision changes alone do not uniformly prevent collapse in their experiments.

**Project connection:** Provides a hypothesis linking sampler/learner discrepancy, response length and update magnitude. It supports interpreting the existing exact-prefix probability audit as more than checkpoint bookkeeping, without replacing that audit.

**Limits:** Not a Laguna/Muon/multi-turn CUDA result. Its scheduler is not claimed implemented here; changing curriculum length can confound the signal, and the paper’s loss-landscape explanation is a hypothesis rather than a general causal proof.

### AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning

[Primary source](https://arxiv.org/html/2505.24298v2) · 2025. **Status:** arXiv preprint; no conference venue asserted

**Review:** Targeted primary-body review of v2, not the later v5; prior detailed record and §§5.1–5.2 rechecked 2026-10-25. **Location:** §5.1 Staleness-Aware Training, Eq. 3; §5.2 Decoupled PPO Objective, Eqs. 4–5, proposition and Practical Remark

**Source finding:** Bounds policy-version staleness and separates sampling behavior policy from the recent proximal policy used for the clipping trust region. Its implementation recomputes proximal token probabilities on arrival of the training batch.

**Project connection:** Distinguishes stale rollout provenance from same-checkpoint numerical mismatch. If old deferred groups are reused after updates, their behavior logprobs are not automatically the current proximal probabilities.

**Limits:** The project is not claimed to implement full asynchronous AReaL or decoupled PPO. Multi-version generation assumes the actual sampled states; it does not justify rewriting prefixes or reusing rewards after verifier changes. Throughput gains are system-specific.

### TMax: A simple recipe for terminal agents

[Primary source](https://arxiv.org/html/2606.23321v1) · 2026. **Status:** arXiv preprint; no conference venue asserted

**Review:** Historical guide documents targeted primary review; §§4.1 and 5.2 rechecked 2026-10-25. **Location:** §4.1 Experimental Setup: algorithm, infrastructure, evaluation, hyperparameters; §5.2 Terminal agent training is difficult to stabilise

**Source finding:** Plots maximum inference/trainer logprob differences and reports model-dependent benefits from an FP32 LM head. Its terminal-agent experiments show backend/timeouts affect evaluation; its small-KL ablation trades lower collapse severity against lower reward.

**Project connection:** A close tool-using-agent comparison for probability mismatch and harness-dependent evaluation. The guide’s proposed monitoring lessons are not claims that the project implements DPPO or TMax’s async recipe.

**Limits:** Terminal tasks, cluster infrastructure and different models/budgets. The project is not claimed to implement DPPO, FP32-head changes or 32-rollout groups. The source does not establish the right KL coefficient for Laguna.

### On Effectiveness and Efficiency of Agentic Tool-calling and RL Training

[Primary source](https://arxiv.org/html/2606.00135v2) · 2026. **Status:** arXiv paper; author metadata states ICML 2026, but official proceedings not independently verified in this review

[Author-reported venue metadata](https://arxiv.org/abs/2606.00135).

**Review:** Historical guide documents targeted primary review; relevant evaluation and efficiency sections rechecked 2026-10-25. **Location:** §3.1–§3.4 seed, multi-turn template, thinking-history and system-prompt variance; §4.1 online pre-rollout filtering of consecutively all-correct prompts; §4.2 max-variance rollout down-sampling

**Source finding:** Shows BFCL results depend on seeds and history/prompt serialization. It skips prompts with recent all-correct streaks and selects low/high-reward rollout extremes to reduce update work in its tested setting.

**Project connection:** Explains why exact turn prefixes and frozen/reportable prompt and history policies matter. Kernel correctness saturation is not necessarily reward saturation because the project also rewards speed.

**Limits:** Tool-calling benchmark evidence is not a measured CUDA throughput benefit. Filtering and rollout down-sampling remain literature methods, not implementation claims; changing retained examples also changes the sampled training distribution.

## The backbone we use, and ideas not yet adopted

Laguna and LoRA explain the model and adaptation foundations, not a validated recipe for our attention-only rank-8 kernel adapters. For sparse rewards, the Pedagogical RL post asks how to obtain successful trajectories a student can learn from; its teacher/student design is not our sampler/learner probability audit. The education paper is a different work, while DSL-Monkeys and CudaPerf remain lighter reading leads rather than adopted methods.

Read **Project connection** with **Limits**, then consult the linked primary section before reusing a claim. Revisiting a source should update its review date, inspected version, depth and caveats explicitly. Implementation proposals belong in [issue #8](https://github.com/jucamohedano/kernel-rlm/issues/8) or a focused issue; the [historical guide](../../training/environments/kernel_rlm/dashboard/guide.html) remains a dated campaign account.

### DSL-Monkeys: Self-Generated In-Context Examples for Low-Resource GPU DSL Kernels

[Primary source](https://openreview.net/pdf?id=WJiJs9k2Lo) · 2026. **Status:** Workshop paper per previously indexed primary PDF header (ICLR 2026 workshop, not main conference); direct retrieval remains unverified

**Review:** Lead only: prior indexed primary PDF title/workshop header evidence; no detailed methods review **Location:** Title and indexed primary PDF header only; methods/settings unreviewed

**Source finding:** Title identifies self-generated in-context examples for low-resource GPU DSL kernel generation.

**Project connection:** Reading lead for examples/retrieval, not evidence that current model learned delegation or that a reproduction exists.

**Limits:** Prior direct PDF endpoint was blocked by browser verification. No detailed reproduction settings, numerical results, author expansion or main-conference status asserted.

### LoRA: Low-Rank Adaptation of Large Language Models

[Primary source](https://arxiv.org/html/2106.09685v2) · 2021. **Status:** arXiv preprint; v2 (16 October 2021); no venue asserted

**Review:** Prior foundational citation + current primary metadata and method-section locator inspection **Location:** §4.1–4.2; §7.1–7.2

**Source finding:** Adapts frozen pretrained weights with trainable low-rank update matrices; studies placement and rank of updates.

**Project connection:** Low-rank-adaptation foundation for project adapters.

**Limits:** Attention-only rank-8 adapters and the project optimizer/kernel task are project configuration, not a validated recipe inherited from this paper. Year is first arXiv submission; no venue inferred.

### Pedagogical RL: Teaching Models to Teach Themselves from Privileged Information

[Primary source](https://noahziems.com/pedagogical-rl) · 2026. **Status:** Author research post / early experimental results; standalone paper publication not verified

Authors: Souradip Chakraborty, Noah Ziems, Furong Huang, Meng Jiang, Amrit Singh Bedi, and Omar Khattab.

**Review:** Targeted primary-source review **Location:** Sections 1–3: nearest-success sampling; spike-aware pedagogy reward; surprisal-gated assimilation

**Source finding:** A privileged self-teacher is trained with correctness multiplied by a spike-aware learnability score. The student then learns through surprisal-gated imitation.

**Project connection:** Our weak or zero-variance rollout groups motivate studying how to obtain successful trajectories that remain learnable to the student. Execution feedback is a possible form of privileged context, but this method is not implemented in our pipeline.

**Limits:** Early results on two reasoning tasks, not CUDA kernels or recursive child training. This requires a separate teacher/student sampling and training design; our sampler/learner probability audit is not its learnability score. Do not infer effectiveness for Laguna from the reported gains.

### Rewarding How Models Think Pedagogically: Integrating Pedagogical Reasoning and Thinking Rewards for LLMs in Education

[Primary source](https://arxiv.org/html/2601.14560v1) · 2026. **Status:** arXiv preprint; no conference venue asserted

**Review:** Historical guide documents a targeted method review; named prompt/reward/experiment sections rechecked 2026-10-25. **Location:** §2.3 Pedagogical Reasoning Prompting; §2.4 Reward Design, Eq. 1; §3.1 Experimental Setup

**Source finding:** Combines domain-specific Polya-style tutor prompts, student outcome improvement and GPT-4o-mini judges for answer leakage, helpfulness and pedagogical thinking quality. The outcome component averages eight student attempts.

**Project connection:** An indirect example of separating domain-specific instructions from reward design. It is not evidence to replace executable CUDA verification with a thought judge or to add pedagogical thinking rewards to this project.

**Limits:** Education/student-simulation evidence, not kernel generation. It is a separate work from the May 2026 Pedagogical RL research post, Teaching Models to Teach Themselves from Privileged Information; do not merge their identity or evidence.

### Laguna M.1/XS.2 Technical Report

[Primary source](https://arxiv.org/html/2605.27605v1) · 2026. **Status:** Primary technical report on arXiv; v1 (26 May 2026); no conference venue asserted

**Review:** Prior foundational citation + current primary metadata and targeted §3.1.1/§4.1.3/§4.4.1 inspection **Location:** §3.1.1–3.1.2; §4.1.3; §4.4.1–4.4.5

**Source finding:** Describes Laguna model/training recipe, Muon-family optimization, token-in/token-out chat-template alignment, and agentic RL. Its RL objective uses CISPO ratio clipping with length-weighted leave-one-out group-relative advantages.

**Project connection:** Backbone and optimizer/model-recipe foundation; exact token and template alignment is relevant context for the project’s multi-turn sampling audit.

**Limits:** Not evidence that the report validated this attention-only rank-8 kernel LoRA configuration or the project’s direct GRPO objective. Report uses a different RL objective; Moonlight scaling also differs between M.1 and XS.2. No transfer gains are inferred.

### Multi-turn RL with Structural and Performance Aware Rewards for CUDA Kernel Generation

[Primary source](https://arxiv.org/abs/2607.20908) · 2026. **Status:** arXiv preprint; title and abstract verified only, not a methods review.

**Review:** Metadata and abstract only; reading lead, not methods-reviewed evidence. **Location:** Abstract and submission metadata only.

**Source finding:** The abstract describes CudaPerf, combining execution rewards with structural/performance-aware rewards and iterative CUDA refinement.

**Project connection:** A lighter structural-reward reading lead, not an established implementation influence or adopted reward design.

**Limits:** Detailed evaluation and reward construction were not reviewed, and no venue or reproduced result is asserted.
