# Triage Research Strategy: From Heuristic Escalation to Intervention Value

**Status:** research strategy, not a paper claim.  
**Audit snapshot:** `9017edd794f2a74c981ae2a67739f91e990bad83` (2026-07-22).  
**Target:** ML-systems hybrid venue (MLSys/ICLR), moderate academic budget, hybrid black-box API and controlled local-serving evaluation.

## Checkpoint protocol and current research state

Every major phase must end in a durable checkpoint before the next phase starts. A checkpoint is complete only when the repository is runnable, the strategy state below is current, and all artifacts needed to reproduce or resume the phase are present in versioned paths.

| Checkpoint requirement | Persistent artifact |
|---|---|
| Code and configuration | Committed or otherwise preserved tracked changes; immutable config snapshot with model/provider versions and random seeds |
| Experiment execution | Command line, environment/version manifest, stdout/stderr logs, raw outputs, costs, timing, and failure records |
| Intermediate analysis | Versioned tables/figures, trace schema version, data split identifiers, and script that regenerates each derived result |
| Research decision | This section updated with completion evidence, decisions, rejected hypotheses, remaining tasks, and one exact resumption command or action |
| Consistency gate | `git status`, configuration validation, relevant tests, and an artifact-existence check recorded before proceeding |

**Checkpoint 0 — repository and research memo audit (complete).**

- **Completed:** cloned and audited public revision `9017edd794f2a74c981ae2a67739f91e990bad83`; inspected routing, signals, retrieval, verification, telemetry, evaluation scripts, receipts, configuration, tests, README, CHARTER, and history; completed the primary-source literature audit recorded below.
- **Evidence:** the current policy is a fixed heuristic response-aware cascade; `data/rigor_gsm8k.json` records 57.8% escalation and 51.1% accuracy, while `data/rigor_gsm8k_fixed.json` records 29.1% higher cost than always-big after the threshold fix. `pytest -q` passed: 42 tests.
- **Decision:** do not claim novelty for the existing Triage implementation. Pursue CA-MVOI only after the novelty, signal, and sequentiality gates pass.
- **Failure/limitation:** present telemetry records final requests rather than randomized action branches, so it cannot identify action-specific causal repair value. Existing benchmark receipts are too small or saturated for a paper claim.
- **Remaining tasks:** create the trace/data contract; complete full-text closest-work audit; run the signal gate with a frozen action-sensitive pilot; choose whether to proceed to sequential policy training.
- **Exact next resumption step:** implement the checkpointed trace contract described in the CA-MVOI repository-work section, starting with `Action`, `StateSnapshot`, `ActionOutcome`, `Trajectory`, `Budget`, and `PolicyDecision` schemas, then run their unit tests before collecting any paid traces.

## Executive research thesis

Triage is not presently a novel routing algorithm. It is a well-engineered, response-aware heuristic cascade: a small model answers, proxies of answer reliability trigger a fixed retrieve/verify path, and a weighted risk threshold optionally escalates to a big model. That framing overlaps materially with **AutoMix**'s response verification plus POMDP routing and with recent model cascades. The published GSM8K failure is therefore more valuable than the existing positive plots: an answer can be estimated as risky while *no available intervention has positive expected value*. Escalating in that situation incurs both small- and large-model cost and can lower quality.

The defensible research question is:

> Given an observed draft response and a finite action budget, which *next intervention*, if any, has sufficiently positive **action-conditioned marginal value** to justify its measured cost and risk?

The proposed contribution is **CA-MVOI**: a calibrated, conservative sequential controller that distinguishes (i) probability that the current answer is unacceptable from (ii) the conditional distribution of improvement from `RESAMPLE`, `RETRIEVE`, `VERIFY`, `TOOL`, `STRONGER_MODEL`, `SPECIALIST_MODEL`, or `ABSTAIN`. It should stop when the lower confidence bound on every action's marginal value is non-positive. This is a conditional research bet, not a novelty assertion: it proceeds only after the novelty gates below pass.

The causal story to test is precise. Scalar reliability fails because correctness and repairability are different latent variables. Fixed threshold policies confuse them, so they over-escalate when a strong model is not conditionally better or when a verifier/retriever is unhelpful. Learning intervention effects from response state, updating that state after each action, and accounting for remaining resources should reduce harmful interventions while preserving useful ones.

## Current Triage decomposition

### Confirmed implementation

The public repository is a Python/FastAPI product prototype. `app/main.py` exposes chat, telemetry, anomaly, ingest, and benchmark endpoints. A provider registry wraps OpenAI-compatible APIs, NVIDIA, Groq, OpenRouter, Hugging Face, Ollama, and a synthetic mock provider. The retrieval store is an in-process or pgvector cosine index, but its default evidence corpus is only 16 general-knowledge sentences. The long-context route is explicitly a detect-only flag, not a long-context inference method.

`route_and_answer` in `app/router/router.py` executes the following fixed program:

1. A deterministic calculator may answer pure arithmetic before any model call.
2. A lexical prompt prefilter may bypass the small model for predicted-hard prompts or suppress deep probes for predicted-easy prompts.
3. The chosen small model produces a complete answer. Token-logprob uncertainty is used if available; otherwise uncertainty requires resampling.
4. If a gate fires, Triage takes three resamples and one self-contradiction probe. It derives uncertainty, sample instability, contradiction, retrieval disagreement, and evidence sufficiency from heuristic text/embedding rules.
5. It retrieves when uncertainty is high and no conflict is detected; it verifies when conflict or evidence rules fire. The action order is fixed, not chosen.
6. A weighted multi-signal abstention score is formed. A risk threshold then escalates to one large model after the small-model retrieve/verify path; it does not choose among models, tools, or interventions.
7. It records final request-level telemetry. `compute_units = input_tokens + 10 * output_tokens` is a transparent proxy, not measured FLOPs, energy, GPU-seconds, queueing cost, or cache reuse.

The optional semantic memory stores nearest prompts and final routing fields. The optional online controller changes only `risk_high` toward a target escalation rate. It receives no correctness label: it uses whether escalation occurred, small risk, and big risk. It is neither contextual-bandit learning nor offline/online RL.

### What is engineering versus science

| Component | Status | Research value now |
|---|---|---|
| Provider abstraction, FastAPI UI, BYO key handling, deployment files | Product engineering | Useful substrate; not a contribution |
| Calculator and seed retrieval store | Deterministic utility features | Baselines/actions, not novelty |
| Logprob, resample, self-check, and embedding proxies | Hand-designed signals | Candidate state features; uncalibrated and not validated as scientific signals |
| Fixed tier map and thresholds in `router.yaml` | Heuristic policy | Baseline only |
| Semantic memory and threshold nudging | Hypotheses | The committed 10-prompt ablation is accuracy-neutral and near cost-neutral |
| SQLite telemetry and cost anomaly endpoint | Observability engineering | Necessary for trace data, currently insufficient for causal/off-policy learning |
| MT-Bench/GSM8K scripts and receipts | Initial evidence | Useful regression fixtures, insufficient for a paper |

### Evaluation and negative results

The repository's strongest positive GSM8K receipt is five 30-item runs of `gpt-4o-mini -> gpt-4o`: Triage mean accuracy 0.960, big-model mean 0.900, mean cost saving 85.14%, and mean escalation 8.64%. It is too small and the big model is not consistently better, so it does **not** establish repair from escalation. The README correctly reports this limitation.

The decisive negative is the older 90-item GSM8K `ministral-8b -> gpt-oss-120b` run: the heuristic escalated 57.8%, achieved 51.1% accuracy, versus 93.3% small and 91.1% big. After a threshold change, a 30-item run matched the small model at 86.7% but cost 29.1% more than always-big. This is evidence against using risk as an escalation value estimate, not evidence of a mere threshold bug.

MT-Bench receipts use a first-turn LLM judge and 50/50 tune/test threshold selection; the held-out halves contain 40 items and the free model is near-saturated on some pairs. The README already disclaims the old in-sample headline. Unit tests are mock-only and validate contracts and invariants, not live reliability, retrieval quality, policy value, or serving behavior.

## Literature and novelty audit

### What is substantially solved

| Work | Prior result | Consequence for Triage |
|---|---|---|
| Chen, Zaharia, Zou, **FrugalGPT** (TMLR 2024; [arXiv:2305.05176](https://arxiv.org/abs/2305.05176)) | Learns cost-aware combinations/cascades of API LLMs. | A small-to-large cost cascade is not novel. |
| Ong et al., **RouteLLM** (ICLR 2025; [arXiv:2406.18665](https://arxiv.org/abs/2406.18665)) | Preference-trained prompt router with cross-pair transfer. | Prompt-only learned routing is a required baseline. |
| Aggarwal et al., **AutoMix** (NeurIPS 2024; [arXiv:2310.12963](https://arxiv.org/abs/2310.12963)) | Small-answer self-verification and POMDP/threshold routing to a larger LM. | Triage's observe-then-escalate claim is already close prior art. |
| Chen et al., **RouterDC** (NeurIPS 2024; [arXiv:2409.19886](https://arxiv.org/abs/2409.19886)) | Query/LLM embeddings with dual contrastive routing and OOD tests. | Specialist-model selection and model substitution are established. |
| Wang et al., **MixLLM** (NAACL 2025; [arXiv:2502.18482](https://arxiv.org/abs/2502.18482)) | Continual contextual-bandit query-to-model routing under quality/cost/latency trade-offs. | A generic contextual-bandit model router is insufficiently new. |
| Dekoninck, Baader, Vechev, **A Unified Approach to Routing and Cascading for LLMs** (ICML 2025; [arXiv:2410.10347](https://arxiv.org/abs/2410.10347)) | Formal optimal routing/cascading and unified cascade routing. | Do not claim a generic optimal cascade or merely combine routing and cascading. |
| Ding et al., **BEST-Route** (ICML 2025; [arXiv:2506.22716](https://arxiv.org/abs/2506.22716)) | Chooses model and number of sampled responses by query difficulty. | Adaptive sample count alone is solved. |
| Panda et al., **Adaptive LLM Routing under Budget Constraints / PILOT** (Findings of EMNLP 2025; [arXiv:2508.21141](https://arxiv.org/abs/2508.21141)) | Preference-informed contextual bandit plus online budget policy. | Budgeted contextual-bandit framing alone is solved. |
| Wei et al., **Learning to Route LLMs from Bandit Feedback** (2025 preprint; [arXiv:2510.07429](https://arxiv.org/abs/2510.07429)) | Preference-tunable prompt-level bandit policy under partial feedback. | Online bandit feedback is a required comparison, not the main novelty. |
| Li et al., **LLMRouterBench** (Findings of ACL 2026; [arXiv:2601.07206](https://arxiv.org/abs/2601.07206)) | 400K instances, 21 datasets, 33 models, and ten router baselines; reports that simple baselines often remain competitive. | Use it for model-routing comparability, but it cannot by itself evaluate sequential intervention outcomes. |

### Closest threat and required distinction

Yin and Zhang, **Failure-mode-aware uncertainty intervention routing for large language models** (Information Sciences, 2026, [article](https://www.sciencedirect.com/science/article/pii/S0950705126014115)) predicts one of commit, deliberation, retrieval-augmented regeneration, or defer from a behavioral uncertainty signature. It is the closest prior work. A method that merely chooses among resampling, retrieval, and abstention is therefore not publishable.

CA-MVOI must differ empirically and mathematically in all of these ways:

- Its policy is **sequential**: an action changes the observed state and the remaining action set, rather than making a single intervention choice from a precomputed signature.
- It estimates **conditional incremental utility of each action**, including strong/specialized models and tools, rather than classifying a failure mode or response confidence.
- It is **conservative and resource-constrained**: stopping is selected when no lower-confidence action benefit clears cost, latency, and reliability constraints.
- It directly tests action complementarity and negative marginal interventions with randomized/exhaustive intervention traces.

If full-paper audit reveals this exact combination in prior work, CA-MVOI is not a paper direction. The fallback is a narrower systems paper on policy-aware local serving, or no paper claim.

### Adjacent foundations and controls

- Wang et al., **Self-Consistency Improves Chain of Thought Reasoning in Language Models** (ICLR 2023; [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)): resampling can help but is an action with a cost, not free evidence.
- Lewis et al., **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (NeurIPS 2020; [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)): retrieval is an intervention whose value depends on knowledge need and corpus quality.
- Kuhn, Gal, Farquhar, **Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation** (ICLR 2024; [arXiv:2302.09664](https://arxiv.org/abs/2302.09664)): semantic uncertainty is a stronger uncertainty baseline than surface agreement.
- TARo, **Token-level Adaptive Routing for LLM Test-time Alignment** (Findings of ACL 2026; [paper](https://aclanthology.org/2026.findings-acl.50.pdf)), and R2R, **Efficiently Navigating Divergent Reasoning Paths with Small-Large Model Token Routing** (2025; [paper](https://nicsefc.ee.tsinghua.edu.cn/%2Fnics_file%2Fpdf%2Fc660550f-13f6-4bb6-8b37-440a66b51879.pdf)): token-level delegation is already active work. It is an optional local-serving study, never the headline unless it adds a clearly distinct cache-aware contribution.

## Candidate directions

Scores are 1–5, where higher novelty/depth/fit/paper probability is better and higher burden is worse.

| Rank | Direction | Novelty | Depth | Fit | Burden | Paper probability | Verdict |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | CA-MVOI sequential conservative intervention policy | 4 | 5 | 5 | 4 | 4 | Proceed only after closest-work gate |
| 2 | Shift-robust, reliability-constrained action routing | 4 | 5 | 4 | 5 | 3 | Strong alternative; needs calibrated labels and shift suite |
| 3 | Load/KV-cache-aware intervention allocation | 3 | 5 | 3 | 5 | 3 | MLSys alternative, needs real cluster evidence |
| 4 | Action-complementarity benchmark and trace dataset | 4 | 4 | 4 | 4 | 3 | Valuable companion artifact, weak alone |
| 5 | Causal repairability estimator for escalation | 3 | 4 | 5 | 4 | 3 | May be a component of CA-MVOI, too narrow alone |
| 6 | Failure-mode classifier over retrieve/resample/abstain | 1 | 3 | 5 | 3 | 1 | Substantially covered by Yin and Zhang |
| 7 | Contextual-bandit routing under cost budget | 1 | 4 | 3 | 3 | 1 | Covered by PILOT/BaRP/WISERouter |
| 8 | Response-aware small-to-large cascade | 1 | 3 | 5 | 2 | 1 | Covered closely by AutoMix |
| 9 | Model plus sample-count allocation | 1 | 3 | 3 | 3 | 1 | Covered by BEST-Route |
| 10 | Token-level small/large handoff | 2 | 5 | 2 | 5 | 2 | Strong prior work and difficult cache compatibility |
| 11 | Conformal abstention/calibration only | 2 | 3 | 4 | 3 | 2 | Useful constraint/control, insufficient core claim |
| 12 | Specialized-model selection/model substitution | 1 | 3 | 3 | 3 | 1 | Covered by RouterDC and router benchmarks |

## Top three proposals

### 1. CA-MVOI — recommended primary direction

**Problem and hypothesis.** Given a response state `s_t` and remaining budget `b_t`, choose an intervention or stop. The hypothesis is that calibrated lower-confidence estimates of *action-specific* incremental utility identify harmful escalations that scalar uncertainty and fixed cascades cannot, producing better reliability–cost–latency Pareto frontiers under model and task shift.

**Formulation.** Let `y` be the latent task outcome, `q(z,y)` quality, `c(a,s)` observed vector cost, and `A(s)` feasible actions. An intervention transitions `s_{t+1} ~ P_a(.|s_t)`. Optimize

`max_pi E[ q(z_T,y) - lambda_c C_T - lambda_l L_T ]`

subject to `E[C_T] <= B`, `Pr(unsafe or incorrect | STOP) <= epsilon`, and `L_T <= deadline` when required. Learn

`Delta_a(s) = E[V(s_{t+1}) - V(s) | s,a] - lambda^T E[c(a,s)]`.

Use a pessimistic estimate `LCB_a(s) = DeltaHat_a(s) - beta * sigmaHat_a(s)`. Stop iff every feasible action has non-positive LCB; otherwise choose the feasible action with the greatest LCB. Dual variables adapt the measured budget constraint, rather than a hand-set escalation threshold.

**Algorithm and architecture.** Collect action-branch traces, fit calibrated response-state encoders and per-action distributional value models, then train a conservative finite-horizon fitted-Q policy with support penalties. A myopic action-value controller and a contextual-bandit controller are mandatory ablations. The executor is an action registry around the existing provider/retrieval/verifier interfaces. API experiments expose actions at request boundaries; local experiments add exact timing, token, GPU, and queue state.

**Repository work.** Add `Action`, `StateSnapshot`, `ActionOutcome`, `Trajectory`, `Budget`, and `PolicyDecision` schemas; a legacy Triage adapter; explicit action executors; action-level telemetry; trace generation; policy training/evaluation; and a local vLLM/SGLang adapter. Do not reuse current final-request telemetry as causal data: it lacks unchosen outcomes and intermediate state.

**Data, baselines, and metrics.** Use GSM8K/MATH or AIME-style exact reasoning; Natural Questions/PopQA-style controlled knowledge QA with a frozen corpus; HotpotQA/MuSiQue-style multi-hop retrieval; HumanEval/LiveCodeBench for tool/code; an abstention set; and LLMRouterBench for one-step model-selection comparisons. Baselines: always-small/big, fixed cascades, current Triage, prompt-only RouteLLM-style, response-risk threshold, AutoMix-style self-verification, BEST-Route-style sample allocation, contextual bandit, and oracle action planner. Report task quality, exact/pass score, human or blinded pairwise quality, cost, p50/p95 latency, router overhead, GPU-seconds, calibration, risk–coverage, abstention quality, action distribution, constraint violations, and Pareto hypervolume.

**Ablations.** Remove response state; collapse to scalar risk; remove action-specific outcome heads; remove sequential state update; remove LCB conservatism; replace measured with token proxy cost; remove resource state; remove individual actions; randomize action order; and compare one-step against two-step policy.

**Failure modes and falsifiers.** Trace coverage may be poor, verifier/judge labels may be biased, actions can damage good answers, and expensive interventions may be dominated by strong-model calls. Falsify if response state does not predict incremental benefit beyond prompt features; CA-MVOI does not beat calibrated myopic action selection on held-out Pareto frontiers; action ordering has no measurable effect; or gains vanish under model substitution/shift. A convincing paper needs significant, replicated Pareto gains across at least three action-sensitive domains, demonstrated negative-intervention avoidance, constraint satisfaction, and an oracle-gap analysis.

### 2. Robust reliability-constrained intervention routing

**Problem and hypothesis.** API model upgrades, price changes, task shifts, and retrieval-corpus changes invalidate fixed thresholds. The hypothesis is that conformalized lower bounds on action benefit and a worst-group reliability constraint reduce unsafe stopping/escalation under shifts with acceptable cost.

**Formulation.** Solve the CA-MVOI objective with `inf_{P in U} E_P[q]` and groupwise `Pr_P(error | STOP, g) <= epsilon_g`; calibrate action-benefit intervals on an exchangeable held-out calibration split. Select an action only when its benefit LCB is positive in the relevant group.

**Algorithm/repository work.** Add model-pair/task/corpus shift tags, calibration partitions, group-aware conformal predictors, policy fallback to abstain or strong model, and reproducible substitution experiments. The policy remains one or two steps; claims about long horizons require evidence.

**Experiments and ablations.** Hold out domains, model pairs, retrievers, prompt templates, and pricing/load conditions. Compare uncalibrated CA-MVOI, post-hoc calibrated risk, conformal stopping, RouteLLM transfer, RouterDC, and robust bandit baselines. Measure coverage/error guarantees, conditional calibration error, costs, quality, violation rate, and degradation relative to in-distribution.

**Falsifier and contribution bar.** Falsify if intervals fail their advertised empirical coverage or robust policy loses Pareto efficiency without reducing worst-group failures. A top-tier result requires reliable reductions in severe-error/constraint violations under multiple realistic shifts, not merely an average-cost improvement.

### 3. Cache- and load-aware intervention allocation

**Problem and hypothesis.** API token price is an inadequate systems cost. The hypothesis is that the same intervention has different value under queue load, prefix/KV reuse, and prefill/decode contention; a controller that observes these states improves tail latency and goodput under reliability budgets.

**Formulation.** Extend state with server queues, predicted output length, cache-hit/reuse indicators, model residency, and deadline. Reward is successful constrained completion minus measured GPU-time/energy and deadline penalties. This is a constrained semi-MDP; actions include model placement/selection only where actual serving support exists.

**Algorithm/repository work.** Build a controlled local serving lane with fixed versions, workload replay, tracing, and GPU/queue instrumentation. Do not claim cross-model KV-cache continuation unless the tested models and engine actually support it. Add a serving simulator only after validating it against measurements.

**Experiments, ablations, and falsifier.** Replay bursty and stationary workloads across 1–4 GPUs, compare static model pools, current Triage, load-agnostic CA-MVOI, shortest-queue routing, and load-aware policy. Measure p50/p95/p99, deadline miss, throughput, GPU utilization, energy/GPU-seconds, cache metrics, cost, and task quality. Falsify if local scheduling gains disappear when measured overhead is included or if load state adds no decision value. A paper needs real, reproducible serving gains at equal reliability, not token estimates.

## Experimental matrix and publication gates

| Axis | Required condition |
|---|---|
| Quality labels | Exact scoring where possible; blinded human/pairwise subset for open-ended tasks; report judge agreement |
| Counterfactual evidence | Randomized or exhaustive feasible action branches on a training trace set; document action support and missingness |
| Generalization | Task, model-pair, retriever/corpus, and prompt-template held-out tests |
| Cost | Actual API receipts plus local GPU-seconds and end-to-end latency; separately report routing overhead |
| Statistics | Frozen tune/calibration/test partitions, paired bootstrap confidence intervals, at least five policy seeds, per-domain tables |
| Fairness | Same model pool, prompts, action caps, timeout, judge, and budgets across all methods |
| Pareto analysis | Plot quality/risk against money, compute, and latency; report hypervolume and budget-slice results |
| Reproducibility | Versioned trace schema, raw outputs where licensing permits, config hashes, seeds, provider/model snapshots, and evaluation code |

### Decision gates

1. **Novelty gate:** read the full closest intervention-routing, adaptive-compute, and sequential-agent papers. Stop if any already learns calibrated sequential action-conditioned repair value under equivalent constraints.
2. **Signal gate:** before RL, show that response state predicts action benefit beyond prompt-only difficulty with held-out confidence intervals.
3. **Sequentiality gate:** show action values change after a prior action and that two-step decisions beat a strong one-step policy. Otherwise publish only a myopic method, if its novelty survives.
4. **Systems gate:** use measured cost/latency. Abandon load/token claims if replay instrumentation cannot support them.
5. **Robustness gate:** no main claim unless effects survive one model substitution and one task/corpus shift.

## Expected reviewer objections and responses

| Objection | Required evidence or response |
|---|---|
| “This is AutoMix plus RAG.” | Show sequential action-conditioned value, lower-confidence stopping, and randomized action-effect analysis; otherwise concede and stop. |
| “The reward is a judge artifact.” | Use exact labels, human/pairwise validation, inter-rater agreement, and judge-model sensitivity. |
| “Offline RL is unsupported.” | Publish action-support diagnostics, conservative policy constraints, randomized branch data, and on-policy held-out confirmation. |
| “The method just buys more compute.” | Compare against equal-budget BEST-Route/self-consistency/fixed cascades and report all overhead. |
| “Triage’s results are saturated or cherry-picked.” | Use broad action-sensitive tasks, frozen splits, per-domain failures, and report negative results. |
| “Token cost is not systems cost.” | Restrict API claims to bills/latency; make hardware claims only from local measured traces. |
| “Model pairs change.” | Report substitution and price/model version snapshots, then use robust calibration or narrow the claim. |

## Publication-risk assessment

**Current project:** high risk as a paper; good engineering foundation, insufficient novelty and evidence.  
**CA-MVOI:** medium-high risk but potentially publishable if it clears the novelty and sequentiality gates. Its strongest possible contribution is a validated principle: *routing should allocate interventions by calibrated marginal repair value, not by answer risk*.  
**Robust variant:** high methodological burden but stronger ML story.  
**Load-aware variant:** high systems burden and dependent on real infrastructure, but potentially strongest MLSys fit.

The correct outcome may be that Triage remains an open-source engineering system rather than a top-tier research paper. The strategy explicitly treats that outcome as success over forcing an unsupported novelty claim.
