# Literature review for Kernel RLM

A kernel can pass a test without doing the right computation. A training curve can rise without showing a better policy. And an agent can call children without learning useful delegation. This review follows those distinctions through our kernel-generation pipeline.

Start with the thematic sections for the story. Open a source record for its finding, project connection and limits. These are research connections, not claims that we reproduced each method or obtained its results.

![Research map connecting recursive agents, kernel verification, root-policy training and held-out evaluation](/literature/research-map.svg)

Original project schematic: how the research questions connect, not a comparison of paper results.

Sources reviewed: **25 October 2026**. Inline figures and explanations refreshed: **26 October 2026**. The year below is the first preprint/report year unless a conference year is stated. A preprint label means no independent venue was verified for this review, not that the work is necessarily unpublished. Review depth is explicit; a reading lead is not a methods review.

## A root agent, a REPL, and the choice to delegate

Imagine the root agent inspecting an operator, writing a CUDA candidate, and revising it after execution feedback. Recursive Language Models supplies the persistent-REPL foundation; Recursive Agent Optimization asks how root and child trajectories receive training credit. Our current loss targets exact root outputs, not child outputs, although shared weights can still change child behavior indirectly. The agent-scaling studies add a necessary control: useful delegation must be weighed against actual root-plus-child tokens, tool calls and elapsed time.


### Recursive Language Models

The RLM harness keeps the long input in a REPL variable. The root can inspect small pieces and delegate focused work without copying the entire input into every prompt.

![RLM Figure 2: the input prompt is stored in a REPL variable, inspected in pieces, and passed to recursive subcalls](/literature/rlm-figure-2.png)

Figure 2 from **Recursive Language Models**, arXiv:2512.24601v3. Alex L. Zhang, Tim Kraska, and Omar Khattab. [Original figure](https://arxiv.org/html/2512.24601v3#S1.F2) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Reproduced unchanged; scaled for display.

The figure shows the root, its REPL state, and recursive subcalls. It illustrates context management, not evidence that child outputs receive direct training loss in our implementation.


[Primary source](https://arxiv.org/html/2512.24601v3) · 2025. **Status:** arXiv preprint; v3 inspected (11 May 2026); no venue asserted

**Source finding:** Treats long input as external environment data accessed through a persistent REPL and recursive LM calls. Its 1,000-filtered-trajectory experiment distills teacher root turns with their full histories as SFT samples. Appendix A separately describes an MRCRv2 RL experiment.

**Project connection:** Architectural and upstream-code foundation for persistent REPL, external context and recursive delegation; motivates focus on root behavior without claiming identical training.

**Limits:** The 1,000-teacher-trajectory result is supervised distillation, not this project’s direct GRPO baseline. Do not imply the entire paper contains no RL. Its root-training motivation is not evidence that child policy loss is used here.

*Source and review notes:*

**Review:** Prior primary review + current abstract metadata and targeted §2/§3.2/Appendix A reinspection **Location:** §2; §3.2 fine-tuning; Appendix A Additional Training Details

### Recursive Agent Optimization

RAO asks how a shared policy can learn both to solve a task and to delegate useful subproblems. The three figures below show the reward design and the ablations we cited. These are results on TextCraft-Synth, not measurements from our CUDA campaign.

**How rewards reach the tree.** Each node receives local success credit and a bonus based on its children’s success rate. This differs from our current root-output-only loss.

![RAO Figure 2: local success rewards and child-success-rate delegation bonus](/literature/rao-reward.svg)

**Does the credit design matter?** Figure 8 compares reward density and trajectory weighting. Its reported evaluation success rates are 0.89 for dense/weighted, 0.75 for dense/unweighted, 0.79 for sparse/weighted, and 0.61 for sparse/unweighted. The image below is the training-curve panel; these four values are from the evaluation table in the same figure.

![RAO Figure 8 training-curve panel: reward density and trajectory weighting ablations](/literature/rao-credit-ablation.svg)

**Why not reward the number of children?** Figure 9 compares success-count and success-rate bonuses. A count bonus can encourage extra easy delegations. For our project, the relevant question is useful work per budget—not how often a child is called.

![RAO Figure 9: success, sub-agent count, and training time under count-versus-rate delegation bonuses](/literature/rao-delegation-ablation.svg)

Figures 2, 8 (plot panel), and 9 from Apurva Gandhi, Satyaki Chakraborty, Xiangjun Wang, Aviral Kumar, and Graham Neubig, **Recursive Agent Optimization**, arXiv:2605.06639v2. [Figure 2](https://arxiv.org/html/2605.06639v2#S2.F2) · [Figure 8 and its evaluation table](https://arxiv.org/html/2605.06639v2#S5.F8) · [Figure 9](https://arxiv.org/html/2605.06639v2#S5.F9) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). SVG bytes unchanged; resized for display. Click an image for full size.

**Project connection:** Low child use alone cannot separate low utility, limited exploration and lack of direct child credit. Shared weights can change child behavior indirectly even without direct child loss.

**Important distinction:** The “Only Root Rewards” ablation still trains child tokens with propagated root reward. It is not our root-token-only-loss baseline. No CUDA result, implementation, or completed ablation is claimed here.

**Source and review notes:** [Primary source](https://arxiv.org/html/2605.06639v2), 2026 preprint, v2 (2 October 2026); no venue asserted. Prior primary section review and current metadata check cover §2.2.1–2.2.2, §4, and §5.0.1–5.0.2. Node-local signals and depth-level weighting are jointly studied; the plots do not isolate every design choice or establish transfer to our task.

### Towards a Science of Scaling Agent Systems

Controlled comparisons vary agent architecture while maintaining task/tool controls and computational budgets; coordination benefit depends on domain and task structure, and coordination can add substantial overhead.

The useful comparison is not agent count alone: it is whether a coordination structure helps a particular task after its extra work is counted. The schematic makes the controls and measurements explicit without importing older unversioned headline statistics.

![Original project schematic. Hold task and tools fixed, vary the architecture, and record actual token, tool-call and elapsed-time costs. This is a conceptual comparison design, not a plot of the paper’s results.](/literature/lit-2-schematic.svg)

**Original project schematic.** Hold task and tools fixed, vary the architecture, and record actual token, tool-call and elapsed-time costs. This is a conceptual comparison design, not a plot of the paper’s results.

**Project connection:** Supports budget-aware comparison of sequential root work and delegation, with actual tokens, tool calls and elapsed time reported rather than assuming more agents imply better decomposition.

**Limits:** Tool-using benchmark evidence is not CUDA-kernel evidence. Equal budget controls do not imply equal actual consumed compute; reported thresholds/effect sizes are not universal laws. Latest v3 has expanded benchmarks/results, so avoid copying unversioned older headline statistics.

[Primary source](https://arxiv.org/html/2512.08296v3) · 2025. **Status:** arXiv preprint; v3 (8 April 2026); no venue asserted

**Review:** Prior primary review + current primary metadata and targeted setup/results/controls inspection **Location:** §3; §4.1–4.4; §5; Appendix E.4

### Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets

Compares single-agent and multi-agent reasoning systems under thinking-token upper caps and discusses accounting limitations.

Figure 1 contrasts one continuous reasoning trajectory with a serial multi-agent pipeline that communicates through explicit messages. The planner, staged agents and aggregator are visible on the right; this is an architecture illustration, not an accuracy plot or proof of equal actual consumption.

![Paper Figure 1: Single-agent continuous trajectory versus sequential multi-agent reasoning pipeline.](/literature/lit-3-figure-1.png)

**Paper Figure 1.** Sequential is the closest MAS comparator to SAS. Both are serial reasoning pipelines over the same question and the same global budget. The essential difference is that Sequential MAS externalizes intermediate reasoning into explicit messages between steps, whereas SAS keeps those intermediates latent within one continuous reasoning trajectory. Credit: Dat Tran and Douwe Kiela, *Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets*. [Source figure](https://arxiv.org/html/2604.02460v2#S4.F1) · [Creative Commons Attribution 4.0 International (CC BY 4.0)](http://creativecommons.org/licenses/by/4.0/). Rendered PDF page 6 at 150 dpi and cropped to the complete Figure 1, including both panels and full caption. Page header and unrelated text removed. No labels, data or figure content altered. No endorsement implied.

**Project connection:** Extra child tokens must not be mistaken for architectural benefit; literature supports distinguishing budget caps from actual root-plus-child usage.

**Limits:** Text-only multi-hop reasoning; tools and vision outside scope. Caps are not enforced equal consumed tokens; Gemini accounting is approximate. The title must not be repeated as a universal claim about CUDA agents.

[Primary source](https://arxiv.org/html/2604.02460v2) · 2026. **Status:** arXiv preprint; v2 (11 April 2026); no venue asserted

**Review:** Prior primary section review retained; current primary abstract metadata rechecked **Location:** §4–5; Appendix C

## What does a passing kernel actually prove?

Suppose a candidate passes because a module’s default state makes an operation look like an identity. Our BatchNorm repair closed one such shortcut, but did not prove that every required operation ran in custom CUDA. KernelBench connects correctness and speed; the verification studies then examine what finite tests miss, including state coverage, numerical contracts and semantic hacks. Their stronger checks are research references, not suites we can assume this project has implemented.


### KernelBench: Can LLMs Write Efficient GPU Kernels?

KernelBench turns a reference PyTorch program into a kernel-generation task, then checks correctness before timing the result. Its score depends on the baseline, hardware, and allowed sampling budget.

![Figure 1: KernelBench evaluates LMs’ ability to generate performant GPU Kernels . Overview of tasks in KernelBench: KernelBench tasks LMs with generating optimized CUDA kernels for a given target PyTorch model architecture and conducts automated evaluation](/literature/kernelbench-figure-1.png)

Figure 1 from **KernelBench: Can LLMs Write Efficient GPU Kernels?**, arXiv:2502.10517v1. Anne Ouyang, Simon Guo, Simran Arora, Alex L. Zhang, William Hu, Christopher Ré, and Azalia Mirhoseini. [Original figure](https://arxiv.org/html/2502.10517v1#S1.F1) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Reproduced unchanged; scaled for display.

Read the figure from left to right: reference computation, generated implementation, correctness checks, then performance measurement. Our changing-task training curve is not this held-out benchmark.


[Primary source](https://arxiv.org/html/2502.10517v1) · 2025. **Status:** ICML 2025; verified in official PMLR proceedings.

**Source finding:** fast_p measures the fraction of assigned workloads whose generated kernels are correct and exceed speed threshold p relative to the stated reference. Hardware, baseline and sampling/refinement budgets determine the meaning of the score.

**Project connection:** Provides the task-level correctness-and-speed metric for interpreting this project; its changing-task, accepted-group training outcomes are not this benchmark score.

**Limits:** fast_1 is a speed threshold, not pass@1. Original finite correctness checks are not a certificate of general semantics, and current project results are not held-out policy evaluation.

*Source and review notes:*

[Official venue record](https://proceedings.mlr.press/v267/ouyang25a.html).

**Review:** Targeted full-primary-text review of the named sections; not a reproduction or exhaustive paper review. **Location:** §3.3 metric design; §4.4 hardware variation; §5.1 repeated sampling and iterative refinement; Appendix B evaluation methodology.

### Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization

This paper asks whether an apparently fast CUDA kernel remains correct under stronger checks. Its benchmark changes input and initialization conditions, then uses LLM screening before hardware verification.

![Figure 2: translation, full-versus-filtered speedup, and proposal scaling in the paper.](/literature/lit-2509.14279-figure-2.png)

**Figure source:** Lange, Robert Tjarko; Sun, Qi; Prasad, Aaditya; Faldor, Maxence; Tang, Yujin; Ha, David. *Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization*, arXiv:2509.14279v1, Figure 2. [Source figure](https://arxiv.org/html/2509.14279v1#S3.F2) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original image bytes unchanged; resized for display. No author endorsement is implied.

**Read the figure:** The middle panel is the key verifier lesson: the paper’s aggregate optimized-kernel speedup falls from 3.13× on the full set to 1.49× after compromised tasks are filtered. The left panel concerns translation success; the right concerns proposal-budget scaling. These are the paper’s KernelBench experiments, not project measurements.

**Source finding:** Adds varied initialization states and input configurations, forward/backward workloads and multiple timing/profiling checks. LLM verification is a screening stage before hardware verification; its own reported aggregate speedup decreases from 3.13× to 1.49× after compromised tasks are excluded.

**Project connection:** An earlier verifier-loophole reference in the guide, directly relevant to default-state identity shortcuts and weak baselines. Passing the project’s current tests does not mean a candidate has passed robust-kbench.

**Limits:** The 3.13×→1.49× change is its experiment, not a correction factor for this project. Its LLM screening is fallible, and benchmark heuristics are not universal correctness thresholds.

*Citation, status and review notes*

[Primary source](https://arxiv.org/html/2509.14279v1) · 2025. **Status:** arXiv preprint; no independent conference venue verified in this review.

**Review:** Targeted full-primary-text review of the named sections; not a reproduction or exhaustive paper review. **Location:** §3 robust-kbench; §4 soft verification; §§5–6 agentic search; Appendix A.1 compromised-task filtering.

### KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch?

This audit separates two reasons a claimed speedup can disappear: a stronger PyTorch reference and harder correctness tests. The figure compares those changes rather than treating them as one correction.

![Figure 1: TF32-reference and hidden-test contributions to recalibrated speedups.](/literature/lit-2607.16241-figure-1.svg)

**Figure source:** Yunxiang Zhang; Ping Yu; Jianyu Wang; Max (Xiangjun) Fan; Julian Reed; Azalia Mirhoseini; Will Su. *KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch?*, arXiv:2607.16241v1, Figure 1. [Source figure](https://arxiv.org/html/2607.16241v1#S3.F1) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original image bytes unchanged; resized for display. No author endorsement is implied.

**Read the figure:** The four bars per model compare a naïve reference, added hidden tests, a TF32-enabled baseline, and both changes. The middle panel illustrates that baseline choice can dominate the change in measured Level-2 speedup. The dashed line is parity with the reference; the plot does not make TF32 appropriate for every task contract.

**Source finding:** Uses a TF32-enabled reference, four value distributions and peak-memory metrics while retaining fixed task shapes. Four genuine semantic hacks survive the expanded finite tests; two other audit flags are legitimate shape specializations.

**Project connection:** The guide already cites it as a verifier-audit reference, not proof that its complete suite is implemented. The project’s BatchNorm state checks and CUDA execution guard address narrower failure modes.

**Limits:** Four deterministic transforms do not prove semantics or vary module lifecycle/state, and feedback can leak hidden cases in a multi-turn loop. TF32 must respect the task numerical contract; the arXiv page unusually lists 26 June 2026 for a 2607 identifier, so no month is inferred from the ID.

*Citation, status and review notes*

[Primary source](https://arxiv.org/html/2607.16241v1) · 2026. **Status:** arXiv preprint; no independent conference venue verified in this review.

**Review:** Targeted full-primary-text review of the named sections; not a reproduction or exhaustive paper review. **Location:** §§2.2–2.5 TF32, hidden tests, memory and metrics; §3.4 audit; §4 limitations; Appendix L.4 residual hacks.

### The Correctness Illusion in LLM-Generated GPU Kernels

A kernel can pass a small fixed-shape test while implementing the wrong operation. This paper studies that gap with a controlled corpus of correct kernels and deliberately buggy variants.

![Figure 2: failure rates on the author-constructed corpus across five GPU classes.](/literature/lit-2606.20128-figure-2.png)

**Figure source:** Sarkar, Dipankar. *The Correctness Illusion in LLM-Generated GPU Kernels*, arXiv:2606.20128v2, Figure 2. [Source figure](https://arxiv.org/html/2606.20128v2#S4.F2) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original image bytes unchanged; resized for display. No author endorsement is implied.

**Read the figure:** Columns are kernels and rows are GPU classes; color encodes each kernel’s test-failure rate. Positive controls remain at zero, while seeded buggy variants fail to varying degrees across all five GPU classes. The plot is evidence about this constructed corpus, not a prevalence estimate for deployed LLM-generated kernels.

**Source finding:** Seeded, schema-aware tests expose author-planted transcription bugs that fixed-shape small-sample checks accept, while keeping positive controls clean in the studied corpus. Tests preserve shared symbolic dimensions and record numerical-error distributions against an fp64 mathematical reference rounded to the target dtype.

**Project connection:** Supports distinguishing kernel semantics from the project’s separate sampler/learner probability audit. The guide labels it verification reading, not a reproduced fuzzer or a measured bug rate in this model.

**Limits:** Author-seeded bugs and CPU stand-ins do not estimate deployed-model failure prevalence. Shape expansion is valid only under the declared task contract, and fp64 mathematical semantics need not match the project’s PyTorch implementation contract.

*Citation, status and review notes*

[Primary source](https://arxiv.org/html/2606.20128v2) · 2026. **Status:** arXiv preprint; v2 reviewed, with no independent conference venue verified.

**Review:** Targeted full-primary-text review of the named sections; not a reproduction or exhaustive paper review. **Location:** §§3.1–3.5 schemas, oracle, corpus and assumptions; §4 evaluation including cross-GPU sweep; §6 limitations.

## Iterative kernel search is not held-out evaluation

A multi-turn agent gets several chances to improve a kernel: its final submission, best candidate in one history and best of several trajectories can yield different scores. Kevin and Dr. Kernel make those reporting choices concrete; CUDA Agent and DRTriton connect training data, curriculum and search. TimeRLM adds a persistent-REPL transfer example, but its depth-0 setting cannot demonstrate useful child delegation.

![Separate paths for collecting verified rollouts, training on replayed root outputs and evaluating checkpoints on held-out tasks](/literature/train-evaluate.svg)

Original project schematic: training feedback and held-out evaluation answer different questions.

Our Stage 1 campaign changed tasks, hardware and verification. Its accepted-group curve describes selected training experience, not a controlled base-versus-checkpoint result. No held-out gain is established; [First campaign](first-campaign.md) records that boundary.

### Kevin: Multi-Turn RL for Generating CUDA Kernels

Kevin trains a model to improve a CUDA kernel over several turns, not just generate a better isolated answer. The training diagram shows how earlier turns can receive credit for a useful later kernel.

![Figure 1: multi-turn context construction and discounted future-score credit.](/literature/lit-2507.11948-figure-1.png)

**Figure source:** Baronio, Carlo; Marsella, Pietro; Pan, Ben; Guo, Simon; Alberti, Silas. *Kevin: Multi-Turn RL for Generating CUDA Kernels*, arXiv:2507.11948v1, Figure 1. [Source figure](https://arxiv.org/html/2507.11948v1#S1.F1) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original image bytes unchanged; resized for display. No author endorsement is implied.

**Read the figure:** Each turn carries forward code, evaluation feedback and summarized earlier reasoning. The bottom row assigns rewards using the current and future kernel scores; the shown kernel scores are an explanatory example, not an additional benchmark result. This is why a failed early kernel can still contribute to a useful later one.

**Source finding:** Uses per-turn training samples with actual histories, drops old long reasoning while retaining summaries/code/feedback, and aggregates discounted future kernel scores. The main evaluation uses 16 trajectories × 8 turns; 82% correctness is best@16, while avg@16 is 46%, and trajectory selection already uses the best kernel within that history.

**Project connection:** An earlier direct CUDA multi-turn-RL reference in the guide. The project’s documented exact-turn replay and final-outcome trajectory credit are not a reproduction of Kevin’s discounted per-turn return.

**Limits:** Training uses 180 KernelBench tasks; evaluation uses 20 reserved original tasks plus 80 newly constructed tasks. The headline is neither single-attempt nor final-submission-only accuracy, and does not demonstrate this project’s held-out quality.

*Citation, status and review notes*

[Primary source](https://arxiv.org/html/2507.11948v1) · 2025 preprint; 2026 conference. **Status:** ICLR 2026; verified in official conference proceedings.

[Official venue record](https://proceedings.iclr.cc/paper_files/paper/2026/hash/86ba836d4c5dd859d795a172911745e2-Abstract-Conference.html).

**Review:** Targeted full-primary-text review of the named sections; not a reproduction or exhaustive paper review. **Location:** §§3.1, 4.1–4.4 and 5.1–5.4; Appendix A task and split construction.

### Dr. Kernel: Reinforcement Learning Done Right for Triton Kernel Generations

Dr. Kernel treats reward hacking, biased multi-turn credit and lazy optimization as different problems. The schematic separates its verification environment, training interventions and evaluation selection rules.

![Original project schematic: verification, training credit and candidate selection are distinct.](/literature/lit-2602.05885-schematic.svg)

Original project schematic — verification, training credit and candidate selection are distinct.

**Figure source:** Original project schematic, drawn for this literature review from the reviewed facts below. Not paper artwork, a reproduction of its code, or a measured result. No numerical result curve is reconstructed.

**Read the schematic:** A stronger execution check does not implement TRLOO, mismatch correction, profiling rewards or profiling-based rejection sampling. Likewise, a last-turn output and the best candidate anywhere in a trajectory answer different evaluation questions. The arrows organize concepts; they do not reproduce the paper’s implementation graph.

**Source finding:** KernelGYM checks reward hacking, while TRLOO, mismatch correction, profiling rewards and profiling-based rejection sampling target separate training failures. Main comparisons sample eight outputs per task with up to three turns; sequential scaling reports last-turn and best-of-history results separately.

**Project connection:** An existing verifier/training reference in the guide; an extension-call guard alone is not proof that all required CUDA computation ran. The guide explicitly does not claim TRLOO or profiling rewards are implemented here.

**Limits:** Its cold-start SFT, Triton and H100 results do not establish transfer to this CUDA/SM120 project. Some baseline comparisons use their best earlier turn, and benchmark-based checkpoint/turn selection is development feedback rather than untouched final-test evidence.

*Citation, status and review notes*

[Primary source](https://arxiv.org/html/2602.05885v2) · 2026. **Status:** ICML 2026; verified in official PMLR proceedings.

[Official venue record](https://proceedings.mlr.press/v306/liu26fd.html).

**Review:** Targeted full-primary-text review of the named sections; not a reproduction or exhaustive paper review. **Location:** §3.3 hacking checks; §4.2 TRLOO; §§5.1–5.3 mismatch/lazy optimization; §§6.1–6.3 evaluation and test-time scaling.

### CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation

CUDA Agent links synthetic task generation, execution feedback and staged RL. Using its data is only one part of that system; the schematic keeps the full paper method separate from this project’s data provenance.

![Original project schematic: synthetic data, execution feedback and staged RL are distinct components.](/literature/lit-2602.24286-schematic.svg)

Original project schematic — synthetic data, execution feedback and staged RL are distinct components.

**Figure source:** Original project schematic, drawn for this literature review from the reviewed facts below. Not paper artwork, a reproduction of its code, or a measured result. No numerical result curve is reconstructed.

**Read the schematic:** Ops-6K is a data source, verification/profiling supplies feedback, and staged actor/critic training is a separate part of the method. Reporting pass rate, faster rate and geometric mean speedup against both eager and compiled PyTorch keeps different success criteria visible. The boxes are a project explanation, not the paper’s architecture artwork.

**Source finding:** Combines synthetic CUDA-Agent-Ops-6K data, tool-based verification/profiling and staged actor/critic training. KernelBench results report pass rate, faster rate and geometric mean speedup against both eager and compiled PyTorch.

**Project connection:** The existing guide credits Ops-6K as a training-data source and an influence on data-separation policy. This does not establish that the project reproduces its PPO, critic, warm-ups or long-context tool protocol.

**Limits:** The AST decontamination threshold of 0.9 excludes measured structural overlap, not every semantic overlap. Abstract faster-rate headlines differ from Table 1, so unqualified headline-rate comparisons are unsafe.

*Citation, status and review notes*

[Primary source](https://arxiv.org/pdf/2602.24286) · 2026. **Status:** arXiv preprint; no independent conference venue verified in this review.

**Review:** Targeted full-primary-text review of the named sections; not a reproduction or exhaustive paper review. **Location:** §§3.1–3.3 data, tool loop and RL; §4 and Table 1 evaluation; Appendix A data contamination check.

### DRTriton: Large-Scale Synthetic Data Driven Reinforcement Learning for Triton Kernel Generation

DRTriton controls the difficulty of synthetic operator compositions, then uses a held-out signal to advance training. Test-time composition search is an additional mechanism, not the same thing as the training curriculum.

![Original project schematic: controlled compositions, held-out curriculum progression and separate evaluation/search.](/literature/lit-2603.21465-schematic.svg)

Original project schematic — controlled compositions, held-out curriculum progression and separate evaluation/search.

**Figure source:** Original project schematic, drawn for this literature review from the reviewed facts below. Not paper artwork, a reproduction of its code, or a measured result. No numerical result curve is reconstructed.

**Read the schematic:** CSP-DAG controls compositions before training. The curriculum advances only after held-out current-level Pass@1 clears its threshold. A separate evaluation and optional composition search then test different claims. The boxes explain existing reviewed facts; they are not a reproduced DAG generator, training trace or result plot.

**Source finding:** Controls synthetic operator-composition difficulty and advances curriculum when held-out current-level Pass@1 exceeds its threshold. Reports a separate 400-program synthetic evaluation and KernelBench evaluation, with test-time composition search as an additional mechanism.

**Project connection:** The guide records an earlier curriculum/data-design influence, not implementation of CSP-DAG, its SFT warm start or search. Easy-first accepted training groups do not by themselves show transfer to unseen operator compositions.

**Limits:** Results use Triton, synthetic programs and eight H100s, not this project’s direct CUDA/RTX setup. A NeurIPS checklist in the manuscript is not verification of conference acceptance.

*Citation, status and review notes*

[Primary source](https://arxiv.org/html/2603.21465v2) · 2026. **Status:** arXiv preprint; no independent conference venue verified in this review.

**Review:** Targeted full-primary-text review of the named sections; not a reproduction or exhaustive paper review. **Location:** §3 CSP-DAG; §§4.1–4.5 verifier, SFT, rewards, curriculum and search; §§5.1–5.4 evaluation.

### TimeRLM: Recursive Language Models Enable Precise Anomaly Localization in Long-Context Time-Series

TimeRLM uses code to inspect long time-series stored outside the prompt. Its architectural figure shows the external-data/REPL loop, but also illustrates sub-agent calls that were disabled in the reported experiments.

![Figure 1: external time-series access through a REPL; drawn recursion was disabled in experiments.](/literature/lit-2608.03391-figure-1.png)

**Figure source:** Zumarraga, Nicolas; Steno, Lorenzo; Wang, Ning; Rosenblattl, Max; Kaar, Thomas; Xu, Maxwell A.; O'Sullivan, Kevin; Kreft, Markus; Fleisch, Elgar; Schmiedmayer, Paul; Langer, Patrick; Jakob, Robert. *TimeRLM: Recursive Language Models Enable Precise Anomaly Localization in Long-Context Time-Series*, arXiv:2608.03391v2, Figure 1. [Source figure](https://arxiv.org/html/2608.03391v2#S1.F1) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original image bytes unchanged; resized for display. No author endorsement is implied.

**Read the figure:** The signal lives in the REPL environment, where code can select channels, compute statistics and plot segments. The drawn Sub-RLM and `sub_rlm(...)` call illustrate the architecture only: Appendix A says recursive calls were disabled at depth 0 throughout the experiments. The image therefore does not show measured child-agent benefits.

**Source finding:** Uses external Python access to long time-series and group-relative RL, reports final-checkpoint results on AnomalyXL plus ARFBench and unseen clinical recordings. Uses mean-centered Dr.GRPO advantages without reward-standard-deviation normalization. Appendix A states recursive sub_rlm calls are disabled (depth 0) throughout.

**Project connection:** Related work for RL-trained persistent-REPL agents and held-out/transfer evaluation, not evidence of trained child delegation. Newly reviewed related work should not be relabeled as original implementation provenance.

**Limits:** Time-series anomaly tasks are not CUDA. Its token-cost and forced-final penalties and Dr.GRPO normalization are not claimed implemented here; depth-0 evaluation cannot demonstrate benefits of recursive child agents.

*Citation, status and review notes*

[Primary source](https://arxiv.org/html/2608.03391v2) · 2026. **Status:** arXiv preprint; v2 (6 September 2026); no venue asserted

**Review:** Prior primary review + current primary metadata, §4.1 and Appendix A targeted inspection **Location:** §3.1; §4.1–4.4; Appendix A; Appendix D

## Which outputs receive learning credit?

A short successful root trajectory and a long debugging trajectory need not receive the same weight under every objective. DeepSeekMath and PPO explain the algorithmic lineage; DAPO and Dr. GRPO distinguish response averaging, realized-token averaging, constant denominators and reward scaling. Those distinctions matter when the training targets are exact root outputs, while child responses and execution feedback remain context. ProRL adds evidence about entropy and reference-policy resets, not a validated reset recipe for our native TRL/Muon pipeline.

### DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

DeepSeekMath introduces Group Relative Policy Optimization (GRPO): sampled responses to the same prompt supply a reward baseline, so the method does not need a separately trained critic. The paper distinguishes outcome supervision from process supervision.

Read the schematic from left to right: multiple responses to one prompt provide the comparison group, reward scores define relative advantages, and the policy update avoids a separately trained critic. The drawing explains the algorithmic relationship only; it does not show a training result.

![Original schematic of GRPO group sampling, reward comparison and policy update without a separate critic](/literature/lit-13-schematic.svg)

**Figure note:** Original project schematic, not a paper figure or measured result. Paper artwork reuse permission was not verified.

**Project connection:** Algorithmic origin of the project’s GRPO lineage.

**Limits:** The citation does not mean every original reward, KL regularization, normalization or training-data choice is reproduced; mathematical-reasoning results do not validate the kernel verifier.

**Citation and status:** [Primary source](https://arxiv.org/html/2402.03300v3) · 2024. **Status:** arXiv preprint; v3 (27 April 2024); no venue asserted

**Review notes:** Prior foundational citation + current primary metadata and §4.1 locator verification **Location:** §4.1.1–4.1.4; Appendix A.1.6. Targeted figure/license inspection and illustration work added 2026-10-26; not a new full-paper review.

### Proximal Policy Optimization Algorithms

PPO proposes policy-gradient surrogate objectives that can be optimized repeatedly in minibatches after interaction-based sampling. Its clipped-surrogate idea is the ancestry cited here, rather than a claim about this project’s complete training algorithm.

The schematic separates sampling, the surrogate objective, and repeated minibatch updates. It marks the limited connection used here: clipped-surrogate ancestry, not adoption of the paper’s full actor-critic system. No equation was newly verified for this illustration.

![Original schematic of PPO sampling, clipped surrogate and repeated minibatch optimization, with limited project ancestry](/literature/lit-14-schematic.svg)

**Figure note:** Original project schematic, not a paper figure or measured result. Paper artwork reuse permission was not verified.

**Project connection:** Clipped-surrogate ancestry for group-relative policy optimization, not a claim that this project runs actor-critic PPO.

**Limits:** Do not label GRPO as the paper’s complete PPO implementation. Current HTML only exposes a conversion-failure placeholder, so no new equation-level or implementation verification is claimed.

**Citation and status:** [Primary source](https://arxiv.org/abs/1707.06347) · 2017. **Status:** arXiv preprint; v2 (28 August 2017); no venue asserted

**Review notes:** Prior foundational citation + current primary abstract metadata/abstract; HTML body unavailable (failed conversion) **Location:** Abstract; clipped-surrogate foundation (not freshly full-text reviewed). Targeted figure/license inspection and illustration work added 2026-10-26; not a new full-paper review.

### DAPO: An Open-Source LLM Reinforcement Learning System at Scale

DAPO changes both which rollout groups enter training and how their tokens contribute to the policy loss. Its token loss uses the realized group token count rather than averaging each response first. Dynamic sampling excludes all-zero/all-one correctness groups, while overlong loss masking and soft overlong punishment remain separate mechanisms.

The schematic keeps three mechanisms apart. Group filtering determines which questions contribute useful correctness variation; the token denominator controls how unequal response lengths are weighted; truncation handling changes treatment of overlong outputs. These are not interchangeable explanations for a selected-group reward curve.

![Original schematic separating DAPO dynamic sampling, realized-token loss denominator and overlong treatment](/literature/lit-15-schematic.svg)

**Figure note:** Original project schematic, not a paper figure or measured result. Paper artwork reuse permission was not verified.

**Project connection:** An implementation reference for interpreting the root-policy token denominator, unequal response/turn lengths, and selected accepted-group curves. Deferring a weak mixed group is not the paper’s zero-gradient filtering.

**Limits:** The project is not a reproduction of all DAPO techniques. Math-specific truncation treatment does not establish the correct CUDA task-budget objective; child/observation tokens and changed historical prefixes require separate project accounting.

**Citation and status:** [Primary source](https://arxiv.org/html/2503.14476v2) · 2025. **Status:** NeurIPS 2025 main conference; official proceedings verified

[Official venue record](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a4277440d50f1f15d2cb4c14f7e0c0d2-Abstract-Conference.html).

**Review notes:** Targeted primary-body review; prior 2026-10-20 record and relevant sections rechecked 2026-10-25. **Location:** §3.2 Dynamic Sampling (Eq. 11); §3.3 Token-Level Policy Gradient Loss (Eq. 12); §3.4 Overlong Reward Shaping (Eq. 13). Targeted figure/license inspection and illustration work added 2026-10-26; not a new full-paper review.

### Understanding R1-Zero-Like Training: A Critical Perspective

This paper analyzes two distinct ways GRPO can reweight its objective: per-response length normalization and within-question reward-standard-deviation scaling. Dr. GRPO removes both and uses a constant loss denominator, unlike DAPO’s realized-token denominator.

Figure 1 puts the two removed normalizers in red on the left and compares reward against output length on the right. The right panel summarizes training progress rather than wall-clock speed: higher reward at shorter output is the paper’s token-efficiency comparison. It does not establish a CUDA speedup or a benefit from changing this project’s objective.

![Dr. GRPO Figure 1 comparing removed normalization terms and reward against output length](/literature/lit-16-figure-1.svg)

Understanding R1-Zero-Like Training: A Critical Perspective, arXiv:2503.20783v2, Figure 1, by Liu, Zichen; Chen, Changyu; Li, Wenjun; Qi, Penghui; Pang, Tianyu; Du, Chao; Lee, Wee Sun; Lin, Min. [Original figure](https://arxiv.org/html/2503.20783v2#S0.F1) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Image file unchanged; resized for display.

**Project connection:** Clarifies how group reward scaling and unequal root lengths affect the intended objective. Citing Dr. GRPO or TimeRLM does not make the project’s current advantage normalization Dr. GRPO.

**Limits:** Main evidence is mathematical reasoning with Qwen2.5-1.5B, not multi-turn CUDA, the project optimizer, or evidence that a normalization change would help. Increased length alone is not demonstrated reasoning improvement.

**Citation and status:** [Primary source](https://arxiv.org/html/2503.20783v2) · 2025. **Status:** arXiv preprint; no conference venue asserted

**Review notes:** Targeted primary-body review; prior 2026-10-20 record and §§3.1–3.2 rechecked 2026-10-25. **Location:** §3.1 GRPO Leads to Biased Optimization, Eq. 3 and implementation box; §3.2 Dr. GRPO: Group Relative Policy Optimization Done Right. Targeted figure/license inspection and illustration work added 2026-10-26; not a new full-paper review.

### ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models

ProRL studies prolonged reinforcement learning from a distilled starting model. It adds reference-policy KL and periodically resets the reference to a recent online policy while reinitializing optimizer state, aiming to address entropy loss and KL-dominated updates.

The schematic shows the source recipe’s two distinct state changes: a recent online policy becomes the reference, and optimizer state is reinitialized. It does not plot entropy, reward or reasoning boundaries, and does not imply this project should reset either state.

![Original schematic of ProRL reference-policy KL, periodic reference update and optimizer reset](/literature/lit-17-schematic.svg)

**Figure note:** Original project schematic, not a paper figure or measured result. Paper artwork reuse permission was not verified.

**Project connection:** A qualified comparison for entropy/KL discussions; rising selected-group rewards alone do not establish the paper’s claim of expanded reasoning boundaries in this CUDA project.

**Limits:** The source starts from DeepSeek-R1-Distill-Qwen-1.5B; its KL/reset recipe is neither claimed implemented nor validated for the project optimizer. Weak groups in a short run are not evidence that reference or optimizer resets are needed.

**Citation and status:** [Primary source](https://arxiv.org/html/2505.24864v1) · 2025. **Status:** arXiv preprint; no conference venue asserted

**Review notes:** Targeted primary-body review, narrower than a full-paper review; prior record and §2.3.1 rechecked 2026-10-25. **Location:** §2.3.1 KL Regularization and Reference Policy Reset, Eq. 4. Targeted figure/license inspection and illustration work added 2026-10-26; not a new full-paper review.

## Replay the conversation that was actually sampled

A root turn written after compiler feedback must not be trained as though it came from a different history. Exact prefixes, sampled token IDs, behavior log probabilities and loss masks define that replay; they do not guarantee equal probabilities across inference and training stacks. The nondeterminism studies and Beyond Precision examine numerical mismatch, AReaL addresses stale policy versions, and TMax and the tool-calling study show why harness details matter. Serial sampling and learning remove overlap, not necessarily mismatch; our probability audit is also separate from kernel verification.

### Defeating Nondeterminism in LLM Inference

The report explains why deterministic operations can still depend on batch size. In its Bigmath RLVR experiment, unmatched numerics without importance weighting collapse, while matched sampler/trainer numerics or importance weighting train smoothly.

The schematic starts with the same checkpoint and prefix, then separates execution-dependent arithmetic from the audit needed to compare token probabilities. The report’s point is stronger than “set the seed”: batch invariance is a property of the operations and execution path, not just the random-number generator.

![Original schematic distinguishing execution-dependent token probabilities from exact-prefix sampler–trainer audits](/literature/lit-18-schematic.svg)

Original project schematic, created for this review; not a figure from the source and not a measured result. Primary report page contains a copyright notice and no reusable figure license was found.

**Project connection:** Direct motivation for exact sampled IDs, behavior logprobs, root prefixes and probability checks. Serial inference/learning removes overlap but does not itself remove backend numerical mismatch.

**Limits:** Report-specific evidence, not a theorem that every small mismatch collapses training. Bitwise matching can cost throughput. The project is not claimed to implement these batch-invariant kernels or the report’s complete training stack.

**Citation and status:** [Primary source](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) · 2025. **Status:** Primary technical research report/blog, Thinking Machines Lab, 10 September 2025; not an arXiv paper

**Review notes:** Targeted primary-report review; prior 2026-10-20 record and named sections rechecked 2026-10-25. **Location:** Batch invariance and “determinism”; How do we make kernels batch-invariant? (RMSNorm, matrix multiplication, attention); Experiments: Performance; Experiments: True on-policy RL. Targeted figure/license inspection and illustration work added 2026-10-26; not a new full-paper review.

### Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference

This study tests how GPU types and counts, batching, and precision change inference reproducibility. Both greedy outputs and sampled aggregate accuracy can vary across its tested settings. Higher precision reduces but does not universally eliminate divergence; LayerCast stores linear weights in BF16 and upcasts them for FP32 computation.

Figure 4 compares variation in the top-1 token probability across 12 runtime settings, for four models and five tasks. Blue BF16 bars are visibly larger than FP16 and FP32 bars; the FP32 insets show small residual variation rather than a universal zero-error guarantee. The vertical axis is a probability-variation metric, not task accuracy or RL reward.

![Numerical nondeterminism Figure 4 showing variation in top-1 probabilities across precisions and runtime settings](/literature/lit-19-figure-4.svg)

Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference, arXiv:2506.09501v2, Figure 4, by Yuan, Jiayi; Li, Hao; Ding, Xinheng; Xie, Wenya; Li, Yu-Jhe; Zhao, Wentian; Wan, Kun; Shi, Jing; Hu, Xia; Liu, Zirui. [Original figure](https://arxiv.org/html/2506.09501v2#S3.F4) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Image file unchanged; resized for display.

**Project connection:** Supports reporting actual inference backend, batching, hardware and precision alongside seeds. A greedy transcript is not enough to establish sampler/learner probability equality, and kernel-program correctness remains a separate question.

**Limits:** An inference reproducibility study, not a project RL stability experiment or a guarantee of determinism. LayerCast is not claimed implemented. Configuration-dependent token flips differ from stale-policy data, and this study does not by itself establish cross-stack probability parity.

**Citation and status:** [Primary source](https://arxiv.org/html/2506.09501v2) · 2025. **Status:** arXiv preprint; no conference venue asserted

**Review notes:** Primary-body targeted review performed 2026-10-25; no earlier deep-review evidence found in supplied prior records. **Location:** §3.1 experiment setup and reproducibility metrics; §3.2 Greedy Decoding ≠ Deterministic Output; §3.3 Random Sampling Has Reproducibility Issue, Too; §3.4 runtime-configuration ablation; §4 Near-Perfect Deterministic Reproduction: LayerCast. Targeted figure/license inspection and illustration work added 2026-10-26; not a new full-paper review.

### Beyond Precision: Training-Inference Mismatch is an Optimization Problem and Simple LR Scheduling Fixes It

In the paper’s Qwen3 math-RL runs, increasing training–inference mismatch accompanies instability. Lower update size and response-length-informed learning-rate decay stabilize the reported runs. Importance sampling and precision changes alone do not uniformly prevent collapse in those experiments.

The schematic separates a measured probability discrepancy, an optimizer-side control, and the source’s response-length signal for learning-rate decay. These are a hypothesis and intervention tested in the paper’s setting, not evidence that probability mismatch has been fixed here.

![Original schematic separating probability mismatch, update size and response-length learning-rate signal](/literature/lit-20-schematic.svg)

**Figure note:** Original project schematic, not a paper figure or measured result. Paper artwork reuse permission was not verified.

**Project connection:** Provides a hypothesis linking sampler/learner discrepancy, response length and update magnitude. It supports interpreting the existing exact-prefix probability audit as more than checkpoint bookkeeping, without replacing that audit.

**Limits:** Not a Laguna/Muon/multi-turn CUDA result. Its scheduler is not claimed implemented here; changing curriculum length can confound the signal, and the paper’s loss-landscape explanation is a hypothesis rather than a general causal proof.

**Citation and status:** [Primary source](https://arxiv.org/html/2602.01826v1) · 2026. **Status:** arXiv preprint; no conference venue asserted

**Review notes:** Targeted primary-body review; prior detailed record and method/experiment sections rechecked 2026-10-25. **Location:** §3.1 Why LR Scheduler, Eqs. 7–8; §3.2 Traditional Scheduler is Not Feasible; §3.3 Our Method: Decay by Response Length, Algorithm 1; §4.1–§4.3 scheduler, decay-period and importance-sampling experiments; §4.4 FP16 precision ablation. Targeted figure/license inspection and illustration work added 2026-10-26; not a new full-paper review.

### AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning

AReaL bounds policy-version staleness and separates the sampling behavior policy from the recent proximal policy used for the clipping trust region. Its implementation recomputes proximal token probabilities when a training batch arrives.

The schematic separates the policy that generated the rollout, the recent policy used as the clipping reference, and the current trainable policy. Staleness is bounded by policy versions; proximal probabilities are recomputed for the arriving batch. None of these steps permits changing the prefix that actually generated the tokens.

![Original schematic separating AReaL behavior, proximal and current policies under bounded staleness](/literature/lit-21-schematic.svg)

Original project schematic, created for this review; not a figure from the source and not a measured result. Exact v2 source is CC BY-NC-ND 4.0. Its figure is deliberately not reused because noncommercial/no-derivatives restrictions were not approved for the project.

**Project connection:** Distinguishes stale rollout provenance from same-checkpoint numerical mismatch. If old deferred groups are reused after updates, their behavior logprobs are not automatically the current proximal probabilities.

**Limits:** The project is not claimed to implement full asynchronous AReaL or decoupled PPO. Multi-version generation assumes the actual sampled states; it does not justify rewriting prefixes or reusing rewards after verifier changes. Throughput gains are system-specific.

**Citation and status:** [Primary source](https://arxiv.org/html/2505.24298v2) · 2025. **Status:** arXiv preprint; no conference venue asserted

**Review notes:** Targeted primary-body review of v2, not the later v5; prior detailed record and §§5.1–5.2 rechecked 2026-10-25. **Location:** §5.1 Staleness-Aware Training, Eq. 3; §5.2 Decoupled PPO Objective, Eqs. 4–5, proposition and Practical Remark. Targeted figure/license inspection and illustration work added 2026-10-26; not a new full-paper review.

### TMax: A simple recipe for terminal agents

TMax uses terminal-agent training to study numerical mismatch and training stability. It plots maximum inference/trainer logprob differences and reports model-dependent benefits from an FP32 LM head. Backend choices and timeouts affect evaluation; its small-KL ablation trades lower collapse severity against lower reward.

Figure 3 plots the maximum inference/trainer logprob difference against the first 100 training steps. The Qwen 3.5 run without an FP32 LM head has larger spikes; Qwen 3.5 with that head and Qwen 3 8B without it behave differently. This is model-dependent numerical evidence, not an evaluation-score plot or a universal prescription to change precision.

![TMax Figure 3 maximum inference versus trainer logprob difference during the first 100 steps](/literature/lit-22-figure-3.svg)

TMax: A simple recipe for terminal agents, arXiv:2606.23321v1, Figure 3, by Ivison, Hamish; Yin, Junjie Oscar; Shao, Rulin; Xiao, Teng; Lambert, Nathan; Hajishirzi, Hannaneh. [Original figure](https://arxiv.org/html/2606.23321v1#S4.F3) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Image file unchanged; resized for display.

**Project connection:** A close tool-using-agent comparison for probability mismatch and harness-dependent evaluation. The guide’s proposed monitoring lessons are not claims that the project implements DPPO or TMax’s async recipe.

**Limits:** Terminal tasks, cluster infrastructure and different models/budgets. The project is not claimed to implement DPPO, FP32-head changes or 32-rollout groups. The source does not establish the right KL coefficient for Laguna.

**Citation and status:** [Primary source](https://arxiv.org/html/2606.23321v1) · 2026. **Status:** arXiv preprint; no conference venue asserted

**Review notes:** Historical guide documents targeted primary review; §§4.1 and 5.2 rechecked 2026-10-25. **Location:** §4.1 Experimental Setup: algorithm, infrastructure, evaluation, hyperparameters; §5.2 Terminal agent training is difficult to stabilise. Targeted figure/license inspection and illustration work added 2026-10-26; not a new full-paper review.

### On Effectiveness and Efficiency of Agentic Tool-calling and RL Training

This tool-calling study shows that BFCL results depend on seeds and history/prompt serialization. It also skips prompts with recent all-correct streaks and selects low/high-reward rollout extremes to reduce update work in its tested setting.

The displayed left panel of Figure 3 compares native and context templates on BFCL multi-turn accuracy. Its grouped boxplots make serialization a visible experimental variable rather than a cosmetic formatting choice. The source’s other Figure 3 panels concern thinking history and system prompts; they are not shown here. This panel does not measure kernel speed or training efficiency.

![Tool-calling Figure 3 left panel comparing BFCL multi-turn accuracy under native and context templates](/literature/lit-23-figure-3-left.png)

On Effectiveness and Efficiency of Agentic Tool-calling and RL Training, arXiv:2606.00135v2, Figure 3 (left panel), by Liu, Tong; Qian, Cheng; Cief, Matej; He, Yuan; Dan, Daniele; Aletras, Nikolaos; Kazai, Gabriella. [Original figure](https://arxiv.org/html/2606.00135v2#S3.F3) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Image file unchanged; resized for display. Only the separately supplied left-panel image is shown; the other panels are not reproduced.

**Project connection:** Explains why exact turn prefixes and frozen/reportable prompt and history policies matter. Kernel correctness saturation is not necessarily reward saturation because the project also rewards speed.

**Limits:** Tool-calling benchmark evidence is not a measured CUDA throughput benefit. Filtering and rollout down-sampling remain literature methods, not implementation claims; changing retained examples also changes the sampled training distribution.

**Citation and status:** [Primary source](https://arxiv.org/html/2606.00135v2) · 2026. **Status:** arXiv paper; author metadata states ICML 2026, but official proceedings not independently verified in this review

[Author-reported venue metadata](https://arxiv.org/abs/2606.00135).

**Review notes:** Historical guide documents targeted primary review; relevant evaluation and efficiency sections rechecked 2026-10-25. **Location:** §3.1–§3.4 seed, multi-turn template, thinking-history and system-prompt variance; §4.1 online pre-rollout filtering of consecutively all-correct prompts; §4.2 max-variance rollout down-sampling. Targeted figure/license inspection and illustration work added 2026-10-26; not a new full-paper review.

## The backbone we use, and ideas not yet adopted

Laguna and LoRA explain the model and adaptation foundations, not a validated recipe for our attention-only rank-8 kernel adapters. For sparse rewards, the Pedagogical RL post asks how to obtain successful trajectories a student can learn from; its teacher/student design is not our sampler/learner probability audit. The education paper is a different work, while DSL-Monkeys and CudaPerf remain lighter reading leads rather than adopted methods.

Read **Project connection** with **Limits**, then consult the linked primary section before reusing a claim. Revisiting a source should update its review date, inspected version, depth and caveats explicitly. Implementation proposals belong in [issue #8](https://github.com/jucamohedano/kernel-rlm/issues/8) or a focused issue; the [historical guide](../../training/environments/kernel_rlm/dashboard/guide.html) remains a dated campaign account.

### DSL-Monkeys: Self-Generated In-Context Examples for Low-Resource GPU DSL Kernels

Title identifies self-generated in-context examples for low-resource GPU DSL kernel generation.

Only the title-level idea is visualized below. It suggests a reading direction around example construction and prompt context; it does not establish the actual generation, filtering, retrieval or evaluation procedure.

![Original project schematic, title-level only. The boxes unpack “self-generated in-context examples” and “low-resource GPU DSL kernels”; the arrows are an explanatory reading aid, not a verified reproduction of the paper’s method.](/literature/lit-24-schematic.svg)

**Figure note:** Original project schematic; no paper artwork or measured result is reproduced.

**Project connection:** Reading lead for examples/retrieval, not evidence that current model learned delegation or that a reproduction exists.

**Limits:** Prior direct PDF endpoint was blocked by browser verification. No detailed reproduction settings, numerical results, author expansion or main-conference status asserted.

[Primary source](https://openreview.net/pdf?id=WJiJs9k2Lo) · 2026. **Status:** Workshop paper per previously indexed primary PDF header (ICLR 2026 workshop, not main conference); direct retrieval remains unverified

**Review:** Lead only: prior indexed primary PDF title/workshop header evidence; no detailed methods review **Location:** Title and indexed primary PDF header only; methods/settings unreviewed

**Figure review status:** Original title-level schematic only. Direct PDF and forum access still stop at browser verification; no paper figure or methods evidence was obtained.

### LoRA: Low-Rank Adaptation of Large Language Models

Adapts frozen pretrained weights with trainable low-rank update matrices; studies placement and rank of updates.

The schematic separates what stays frozen from what is trained. A low-rank update can change the adapted operation without retraining every pretrained weight; this does not select the project’s adapter rank or target modules.

![Original project schematic. A frozen base matrix W₀ is combined with a trainable low-rank update BA. The rank and placement remain choices; no paper figure or experimental result is reproduced.](/literature/lit-25-schematic.svg)

**Original project schematic.** A frozen base matrix W₀ is combined with a trainable low-rank update BA. The rank and placement remain choices; no paper figure or experimental result is reproduced.

**Project connection:** Low-rank-adaptation foundation for project adapters.

**Limits:** Attention-only rank-8 adapters and the project optimizer/kernel task are project configuration, not a validated recipe inherited from this paper. Year is first arXiv submission; no venue inferred.

[Primary source](https://arxiv.org/html/2106.09685v2) · 2021. **Status:** arXiv preprint; v2 (16 October 2021); no venue asserted

**Review:** Prior foundational citation + current primary metadata and method-section locator inspection **Location:** §4.1–4.2; §7.1–7.2

### Pedagogical RL: Teaching Models to Teach Themselves from Privileged Information

A privileged self-teacher is trained with correctness multiplied by a spike-aware learnability score. The student then learns through surprisal-gated imitation.

The distinction is between finding a correct trajectory and finding one that the student can imitate. The schematic shows the post’s teacher/student separation, rather than presenting a project training loop or claiming its early results transfer.

![Original project schematic of the author post’s teacher–trajectory–student design. Correctness and spike-aware learnability guide the teacher; surprisal-gated imitation guides assimilation. No experimental curve is reproduced.](/literature/lit-26-schematic.svg)

**Figure note:** Original project schematic; no paper artwork or measured result is reproduced.

**Project connection:** Our weak or zero-variance rollout groups motivate studying how to obtain successful trajectories that remain learnable to the student. Execution feedback is a possible form of privileged context, but this method is not implemented in our pipeline.

**Limits:** Early results on two reasoning tasks, not CUDA kernels or recursive child training. This requires a separate teacher/student sampling and training design; our sampler/learner probability audit is not its learnability score. Do not infer effectiveness for Laguna from the reported gains.

[Primary source](https://noahziems.com/pedagogical-rl) · 2026. **Status:** Author research post / early experimental results; standalone paper publication not verified

Authors: Souradip Chakraborty, Noah Ziems, Furong Huang, Meng Jiang, Amrit Singh Bedi, and Omar Khattab.

**Review:** Targeted primary-source review **Location:** Sections 1–3: nearest-success sampling; spike-aware pedagogy reward; surprisal-gated assimilation

### Rewarding How Models Think Pedagogically: Integrating Pedagogical Reasoning and Thinking Rewards for LLMs in Education

Combines domain-specific Polya-style tutor prompts, student outcome improvement and GPT-4o-mini judges for answer leakage, helpfulness and pedagogical thinking quality. The outcome component averages eight student attempts.

The conceptual diagram keeps the tutor’s domain-specific instructions separate from the feedback used to train it. Outcome improvement and judge-based pedagogical criteria answer different questions; neither is a kernel-correctness test.

![Original project schematic. Domain-specific tutor prompting and the composite reward are separate parts of the education method. It is not a diagram of the May 2026 privileged-information research post and not a CUDA reward implementation.](/literature/lit-27-schematic.svg)

**Original project schematic.** Domain-specific tutor prompting and the composite reward are separate parts of the education method. It is not a diagram of the May 2026 privileged-information research post and not a CUDA reward implementation.

**Project connection:** An indirect example of separating domain-specific instructions from reward design. It is not evidence to replace executable CUDA verification with a thought judge or to add pedagogical thinking rewards to this project.

**Limits:** Education/student-simulation evidence, not kernel generation. It is a separate work from the May 2026 Pedagogical RL research post, Teaching Models to Teach Themselves from Privileged Information; do not merge their identity or evidence.

[Primary source](https://arxiv.org/html/2601.14560v1) · 2026. **Status:** arXiv preprint; no conference venue asserted

**Review:** Historical guide documents a targeted method review; named prompt/reward/experiment sections rechecked 2026-10-25. **Location:** §2.3 Pedagogical Reasoning Prompting; §2.4 Reward Design, Eq. 1; §3.1 Experimental Setup

### Laguna M.1/XS.2 Technical Report

Describes Laguna model/training recipe, Muon-family optimization, token-in/token-out chat-template alignment, and agentic RL. Its RL objective uses CISPO ratio clipping with length-weighted leave-one-out group-relative advantages.

Figure 2 is a narrow systems illustration: time runs left to right while token dispatch overlaps grouped GEMM work across SMs. It illustrates one implementation detail in the broader model report; it is not a chart of our LoRA performance or a diagram of our RL objective.

![Paper Figure 2: Timeline overlapping token dispatch and grouped GEMM across SMs.](/literature/lit-28-figure-2.svg)

**Paper Figure 2.** Our strategy for overlapping the dispatch with the grouped GEMM compute. Time flows from left to right. SMs numbered from 0 to 7 copy tokens. SM* illustrates all SMs running the grouped GEMM. Credit: Julien Abadji et al. (full author list in the source and provenance manifest), *Laguna M.1/XS.2 Technical Report*. [Source figure](https://arxiv.org/html/2605.27605v1#S3.F2) · [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](http://creativecommons.org/licenses/by-sa/4.0/). None; original SVG bytes downloaded directly from the exact-version HTML. Scaled for display only. Reused figure remains under CC BY-SA 4.0. No endorsement implied.

**Project connection:** Backbone and optimizer/model-recipe foundation; exact token and template alignment is relevant context for the project’s multi-turn sampling audit.

**Limits:** Not evidence that the report validated this attention-only rank-8 kernel LoRA configuration or the project’s direct GRPO objective. Report uses a different RL objective; Moonlight scaling also differs between M.1 and XS.2. No transfer gains are inferred.

[Primary source](https://arxiv.org/html/2605.27605v1) · 2026. **Status:** Primary technical report on arXiv; v1 (26 May 2026); no conference venue asserted

**Review:** Prior foundational citation + current primary metadata and targeted §3.1.1/§4.1.3/§4.4.1 inspection **Location:** §3.1.1–3.1.2; §4.1.3; §4.4.1–4.4.5

### Multi-turn RL with Structural and Performance Aware Rewards for CUDA Kernel Generation

The abstract describes CudaPerf, combining execution rewards with structural/performance-aware rewards and iterative CUDA refinement.

Targeted Figure 1 inspection adds only a workflow-level view to the abstract: context examples and feedback surround candidate generation, an execution harness and a structural-feature branch, which feed reward and policy updates. The figure does not by itself validate reward construction or establish evaluation gains.

![Paper Figure 1: CUDAPerf workflow linking candidate generation, execution, structural features, rewards and policy updates.](/literature/lit-29-figure-1.png)

**Paper Figure 1.** CUDAPerf workflow. The framework generates multiple CUDA candidates from sequential programs, evaluates them using execution-based verifiable rewards and structure-aware performance features, and iteratively refines the policy using reflection-driven feedback and reinforcement learning updates. Credit: Quazi Ishtiaque Mahmud, Nesreen K. Ahmed and Ali Jannesari, *Multi-turn RL with Structural and Performance Aware Rewards for CUDA Kernel Generation*. [Source figure](https://arxiv.org/html/2607.20908v1#S2.F1) · [Creative Commons Attribution 4.0 International (CC BY 4.0)](http://creativecommons.org/licenses/by/4.0/). None; original PNG bytes downloaded directly from exact-version HTML. Scaled for display only. No endorsement implied.

**Project connection:** A lighter structural-reward reading lead, not an established implementation influence or adopted reward design.

**Limits:** Detailed evaluation and reward construction were not reviewed, and no venue or reproduced result is asserted.

[Primary source](https://arxiv.org/abs/2607.20908) · 2026. **Status:** arXiv preprint; title and abstract verified only, not a methods review.

**Review:** Metadata and abstract only; reading lead, not methods-reviewed evidence. **Location:** Abstract and submission metadata only.

**Figure review update (2026-10-26):** Exact-version Figure 1 and caption inspected as a narrow workflow illustration. The earlier title/abstract-only status above records the prior review scope; detailed methods, reward construction and evaluation remain unreviewed.
