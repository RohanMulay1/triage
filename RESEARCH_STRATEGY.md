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
- **Superseded by:** Checkpoint 1 below.

**Checkpoint 1 — trace/data contract (complete, reviewed).**

- **Schema version:** `1.1.0`. `SUPPORTED_SCHEMA_VERSIONS = {1.0.0, 1.1.0}`; the reader accepts 1.0.0 explicitly and refuses anything else rather than coercing it.
- **Completed:** built `app/trace/` — `contract.py` (`Action`, `StateSnapshot`, `ActionOutcome`, `Budget`, `PolicyDecision`, `Trajectory`, plus `ActionCost`, `ActionScore`, `TraceStep`, `TerminalRecord`, `TraceManifest`), `store.py` (single-use run ids, append-only JSONL, immutable manifest, schema gate, `validate_run`), `adapter.py` (`TraceRecorder`, the legacy-router bridge) and `support.py` (action support / missingness). `scripts/collect_traces.py` collects runs and forces the mock provider unless `--live` is passed. The router takes an optional `recorder` argument threaded through `route_and_answer` and `_assess`; every call site is a guarded side-effect append, so `recorder=None` reproduces the pre-tracing behaviour exactly.
- **Evidence:** `pytest -q` passes **121 tests** (42 pre-existing plus 79 new across `tests/test_trace_contract.py`, `tests/test_trace_store.py`, `tests/test_trace_adapter.py`). No committed receipt in `data/*.json` changed. A parametrised regression test asserts identical tier, escalation, abstention, retrieval, verification, status, model and call count with and without a recorder. Per-step costs reconcile exactly with `ChatResponse.cost` on the single-pass, deep-signal, tool and escalation paths.
- **Fixture:** `data/traces/fixture-mock/` — 20 GSM8K items, mock provider, **$0**, `analysis_grade: false`. Collection log at `data/traces/fixture-mock.collect.log`. Regenerated after the review fixes below so the manifest matches the code that produced it; it had not been committed at any earlier version.
- **Support finding (the checkpoint's substantive result):** on the fixture run `abstain`, `verify` and `tool` are **feasible in 20–41 decisions and chosen in none** (coverage 0.0), and every chosen action carries propensity 1.0 because the heuristic is deterministic. Heuristic-only traces therefore cannot support off-policy claims about those actions: any estimate of their value would be extrapolation, on exactly the actions the research question is about. This is a measurement, not an assertion, and it is the argument for randomized branch collection.
- **Review fixes applied before commit:** (1) `ActionCost.__add__` raised when summing costs of different provenance — it labelled the sum with the weaker source while carrying dollars, which the validator rejects; provenance now degrades to `mixed`, which may carry dollars. (2) `CostSource` split `measured_api` into `billed_api` (a provider invoice) and `listed_price` (measured tokens x the price in `config/models.yaml`, snapshotted in the manifest); the router only ever produces `listed_price`, so no figure it emits is presented as an invoice. (3) `store.create_run` makes run ids single-use, so a re-run cannot silently append a second collection behind one manifest. (4) `validate_run`'s `chain_ok` reported structural integrity correctly instead of being confounded with cost errors.
- **Known limitations (carried forward):** counterfactual branching is **declared but unwired** — `trace.record_counterfactual` exists in `config/router.yaml` and does nothing, and `ActionOutcome.is_counterfactual` is never set true by the current collector. `STRONGER_MODEL` cost currently represents the **full big-model pass**, not the marginal increment over the already-incurred small-model trajectory. The rare pre-filter fallback path folds into one `ANSWER`. Under a forced mock provider the router refuses to escalate, so the escalation path is covered with a stand-in provider label rather than a live one. The fixture is mock-only and is not evidence about model behaviour.
- **Remaining tasks:** Decision Gate 1 (full-text novelty audit); then, only if it does not return STOP, wire counterfactual branching, behaviour policies and split manifests.
- **Exact next resumption command:** `python -m pytest -q && git log --oneline -1` then run Decision Gate 1 — read the closest intervention-routing, adaptive-compute and sequential-agent papers in full and record the claim-by-claim comparison in the literature section below. Collect no paid traces until that gate resolves.
- **Superseded by:** Checkpoint 2 below.

**Checkpoint 2 — Decision Gate 1, full-text novelty audit (complete). Verdict: NARROW.**

- **Completed:** primary-source audit of 18 works — every source named in the gate brief plus six 2026 works the previous audit did not contain, found by targeted search on the exact claim rather than on the project's own vocabulary. Recorded per work: title, authors, year, venue, canonical identifier, action space, observability, sequentiality, whether action value is learned, calibration, conservative stopping, constraints, and trace/propensity evidence. Full table in the literature section above.
- **Answer to the gate question:** No prior work does the whole combination. But **four of the six clauses are individually solved**, two of them by 2026 preprints absent from the previous memo, so the honest verdict is NARROW rather than GO.
- **Decisive findings:** (1) [arXiv:2603.19896](https://arxiv.org/abs/2603.19896) already publishes sequential utility-guided selection over `{respond, retrieve, tool_call, verify, stop}` with `argmax_a Gain - λ₁StepCost - λ₂Uncertainty - λ₃Redundancy`; its `Gain` is, in the authors' words, *"a heuristic self-estimated signal rather than a calibrated probability"*, it uses no confidence bound, and it **loses to ReAct** on its own 200-item HotpotQA evaluation. (2) [arXiv:2604.18419](https://arxiv.org/abs/2604.18419) already establishes calibrated value-threshold stopping (`abstain iff V_β < r_⊥`, isotonic calibration, dominance proposition) for the binary continue/abstain case. (3) The Aug-2026 survey [arXiv:2608.17084](https://arxiv.org/abs/2608.17084) §9.2 names action-aware evaluation with explicit action costs as an open problem and its survey finds methods that *trigger* actions from uncertainty, none that *value* them.
- **Surviving contribution:** replace the heuristic self-estimated gain with an outcome-supervised, **calibrated, per-action** marginal value learned from **randomized action-outcome traces with recorded propensities**, and test whether that beats the heuristic-gain utility policy, calibrated risk-threshold escalation, and prompt-only routing at equal measured cost. This is one clause of the original six, plus the trace dataset the survey asks for.
- **Corrections to the previous memo (both were wrong):** the closest-work paper is in **Knowledge-Based Systems**, not *Information Sciences*; and the memo's description of its method was **unverified** — no primary source for it could be obtained. Both are fixed above.
- **Failure/limitation — the gate is provisional:** Yin & Zhang (KBS 2026, DOI 10.1016/j.knosys.2026.116685) is confirmed to exist via Crossref but is **closed access and could not be read** — ScienceDirect returns HTTP 403, Semantic Scholar reports `openAccessPdf.status = "CLOSED"` with a null abstract, and no preprint surfaced. It is the single closest work by title. If it already learns calibrated per-action repair value from outcome data, the surviving contribution collapses and this verdict becomes STOP.
- **Decisions:** retire the name CA-MVOI — "marginal value of information" overstates what survives; the work is *calibrated action-conditioned gain estimation for intervention policies*. Four claim classes are now forbidden and listed in the audit section. Proceed to the Gate 2 pilot, which is worth running for its own sake, but obtain Yin & Zhang before any paper claim.
- **Remaining tasks:** obtain Yin & Zhang by institutional access or author request; then Checkpoint 3 (wire counterfactual branching, behaviour policies, split manifests, action executors) and Gate 2 (signal and repairability pilot).
- **Exact next resumption command:** `python scripts/collect_traces.py --n 40 --dataset gsm8k --split pilot --run-id gate2-pilot-mock` — but only after Checkpoint 3 wires randomized behaviour policies, since a run under the deterministic heuristic reproduces the zero-coverage result already recorded in Checkpoint 1 and cannot answer Gate 2.
- **Superseded by:** Checkpoint 3 below.

**Checkpoint 3 — trace and experiment substrate (complete).**

- **Schema version:** `1.2.0` (`SUPPORTED = {1.0.0, 1.1.0, 1.2.0}`; older traces still read). Adds `OutcomeStatus`, `cumulative_cost`, per-action execution `detail`/`seed`, and behaviour-policy provenance (`policy_version`, `feasible_set_definition`, `exploration_seed`) on `PolicyDecision`.
- **Completed:**
  - `app/trace/actions.py` — typed `InterventionExecutor` over `ANSWER, STOP, RESAMPLE, SELF_CHECK, RETRIEVE, VERIFY, TOOL, STRONGER_MODEL, SPECIALIST_MODEL, ABSTAIN`. Sits **beside** the router, not in it: `app/router/router.py` keeps its own control flow, and the executor is what lets a collector run an arbitrary action from an arbitrary state.
  - `app/trace/policies.py` — seven behaviour policies (legacy heuristic via the Checkpoint 1 recorder, random-feasible, epsilon-greedy, balanced exploration, fixed cascade, prompt-only, response-risk threshold), each returning a true propensity.
  - `app/trace/branching.py` — `collect_fanout`: one shared ANSWER prefix, then **every available action executed once from that same state**, each carried to a terminal and labelled. Support is complete at the branch point by construction, so Δ(a) vs STOP is measured rather than imputed. `RunBudget` enforces a spend cap *before* each action.
  - `app/trace/splits.py` — leakage-safe splits assigned to a **group**, never a row, keyed on task family, model pair, retriever version, prompt template and a prompt fingerprint. Split manifests persisted per run.
  - `app/trace/support.py` — coverage, propensity distribution, **effective sample size**, missingness by status, provider splits, positivity violations, and `assert_estimable`, which **raises** rather than returning a warning.
  - `scripts/collect_traces.py` — `--mode served|fanout`, `--policy`, split manifests, and a live gate that requires **both** `--live` and an explicit `--max-usd`.
- **Evidence:** `pytest -q` passes **188 tests** (42 pre-existing + 146 trace tests across seven files). No committed receipt in `data/*.json` changed, and a test now asserts both that (a) every writable path used by the suite resolves outside the repository and (b) routing does not mutate a committed receipt. Fixture at `data/traces/fixture-fanout/` — 8 mixed items, mock provider, **$0**, with `manifest.json`, `splits.json`, `trajectories.jsonl` and `collect.log`.
- **Fixture contents:** 8 prefix trajectories + 66 branches; **104 counterfactual outcomes**, 16 `unavailable`, 58 `stop`. Exercises TOOL available *and* unavailable, ABSTAIN as a terminal, SPECIALIST_MODEL as permanently unavailable, and one CHOSEN branch per item against COUNTERFACTUAL siblings.
- **Decisions:** (1) `SPECIALIST_MODEL` reports UNAVAILABLE always — there is no such integration here, and listing it as feasible would put an action in every support denominator that nothing could execute. (2) A step the collector **forced** (the prefix ANSWER, the terminal STOP) records a feasible set of exactly one action; listing the whole space there would count occasions on which nothing could have been chosen differently. (3) An action that failed, returned nothing usable, or was served by the mock standing in for a live call is recorded with its real status and is **not labelled** — entering a provider failure as evidence that an action does not help is the specific way this dataset could lie.
- **Defects found and fixed by the tests:** `GroupKey.digest()` included `item_id`, so every item formed its own group and the split logic silently failed to prevent leakage between near-duplicate prompts — the exact failure the module exists to prevent. Also: `RunBudget` was needed because `Budget.charge` returns a new object, so a cap checked against a fresh `Budget` each time never bound; and appending "Give the final numeric answer." to arithmetic prompts left `calculator.solve` unable to parse them, silently removing TOOL from the feasible set for the items it exists to serve.
- **Measured result:** on the mock fixture, `tool` has mean Δ = **+0.50** vs STOP (n=2, 95% CI [−0.48, +1.48]) — the calculator genuinely repaired an arithmetic answer the model got wrong — and `abstain` has mean Δ = **−0.125** (n=8, +3/−4/=1). Every other action has Δ = 0.000 across all 8 items. That is the expected and uninformative result for a mock provider whose answers do not change under intervention, and it is the reason Gate 2 **cannot be answered on mock data**.
- **Failure/limitation:** the fan-out gives complete support at **depth 1 only** — nothing downstream of a first intervention is explored, so it can answer the myopic question (is action a better than stopping?) but not the sequential one. `attempted`, `failed` and `synthetic_fallback` statuses are exercised by unit tests but do not appear in the mock fixture, because a forced-mock run cannot produce a provider failure; they are not fabricated into the fixture. The run remains `analysis_grade: false`.
- **Remaining tasks:** Gate 2 needs a live run — the mock provider cannot produce action-conditioned variation. That requires the Yin & Zhang paper first (Checkpoint 2's provisional condition) and an approved budget.
- **Exact next resumption command:** `python scripts/collect_traces.py --n 60 --dataset gsm8k --mode fanout --policy balanced --live --max-usd 5.00 --run-id gate2-pilot-live` — **do not run this until** (a) Yin & Zhang has been read and Checkpoint 2's verdict confirmed, and (b) the spend is explicitly approved. Until then the honest command is the mock equivalent without `--live`, which reproduces the null result above.

## Executive research thesis

> **Superseded in part by Checkpoint 2 (2026-09-06).** The full-text audit returned
> **NARROW**, not GO. Sequential utility-guided selection over heterogeneous
> interventions ([arXiv:2603.19896](https://arxiv.org/abs/2603.19896)) and calibrated
> value-threshold stopping ([arXiv:2604.18419](https://arxiv.org/abs/2604.18419)) are
> both already published. Read the sections below as the original hypothesis; the
> surviving contribution is stated in the audit section and is narrower than what
> follows. The name CA-MVOI is retired.

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

## Literature and novelty audit (Decision Gate 1, full text, 2026-09-06)

Audited from primary sources. Where only the abstract and metadata page could be
reached, the row says so. **One source could not be obtained at all and the gate
is provisional on it — see "Unresolved source" below.**

### The novelty question

> "Has prior work already learned a calibrated, sequential, action-conditioned marginal repair-value policy over observed response state, with conservative stopping and explicit reliability/cost/latency constraints, evaluated through action-outcome traces?"

**Answer: No — not in combination. But four of the six clauses are individually
solved, two of them by 2026 work the previous version of this memo did not
contain.** The verdict is **NARROW**, not GO.

### Claim-by-claim comparison

Columns are the six clauses of the novelty question. `~` means partial.

| Work | Year / venue | Action space | Observes response | Sequential | Learns action-conditioned value | Calibrated | Conservative stop | Cost/latency/reliability constraints | Action-outcome traces |
|---|---|---|---|---|---|---|---|---|---|
| **Utility-Guided Agent Orchestration** ([arXiv:2603.19896](https://arxiv.org/abs/2603.19896)) Liu, Zhao, Xu | 2026-03, preprint | `{respond, retrieve, tool_call, verify, stop}` | yes | **yes** | **no** — heuristic LLM self-estimate | **no** (states so explicitly) | no | step budget only | no |
| **Knowing When to Quit** ([arXiv:2604.18419](https://arxiv.org/abs/2604.18419)) Davidov et al. | 2026-07, preprint | `{continue, abstain}` | yes (prefix) | yes (token-level) | one value function, **not per-action** | **yes**, isotonic | **yes**, with dominance proof | no | on-policy, no propensities |
| **Agentic Abstention** ([arXiv:2606.28733](https://arxiv.org/abs/2606.28733)) Luo, Wen, Wang | 2026-06, preprint | `{ANSWER, ABSTAIN, ACT}` | yes | yes (POMDP) | no | no | ~ (learned stopping rules) | 10-turn budget | trajectories, no propensities |
| **Failure-mode-aware uncertainty intervention routing** (KBS, DOI [10.1016/j.knosys.2026.116685](https://doi.org/10.1016/j.knosys.2026.116685)) Yin, Zhang | 2026, Knowledge-Based Systems | **UNVERIFIED** | ? | ? | ? | ? | ? | ? | ? |
| **AutoMix** ([arXiv:2310.12963](https://arxiv.org/abs/2310.12963)) Aggarwal et al. | NeurIPS 2024 | `{small, large}` | **yes** | no (one decision) | no — self-verification of correctness | no | no | cost only | no |
| **RACER** ([arXiv:2603.06616](https://arxiv.org/abs/2603.06616)) Hao, Zeng, Wei, Jing | 2026-02, preprint | model **sets** + abstain | ~ | no | no — risk control | **yes**, finite-sample | ~ (risk-controlled) | misrouting risk | no |
| **CP-Router** ([arXiv:2505.19970](https://arxiv.org/abs/2505.19970)) Su et al. | 2025-05, preprint | `{LLM, LRM}` | yes | no | no — conformal set size | **yes**, conformal | no | implicit token cost | no |
| **A2RAG** ([arXiv:2601.21162](https://arxiv.org/abs/2601.21162)) Liu et al. | 2026-01, preprint | gate / retrieve / rewrite+retry | yes | **yes** | no — threshold `TripleCheck` | no (binary validators) | no | ≤ `I_max` retries | no |
| **FrugalGPT** ([arXiv:2305.05176](https://arxiv.org/abs/2305.05176)) Chen, Zaharia, Zou | 2023 arXiv / TMLR 2024 | model cascade | yes | ~ (over models) | no | no | no | cost | no |
| **RouteLLM** ([arXiv:2406.18665](https://arxiv.org/abs/2406.18665)) Ong et al. | ICLR 2025 | `{weak, strong}` | **no** (prompt only) | no | no — preference | no | no | cost | no |
| **Unified Routing and Cascading** ([arXiv:2410.10347](https://arxiv.org/abs/2410.10347)) Dekoninck, Baader, Vechev | ICML 2025 | model selection | yes | ~ (over models) | no — quality per model | no | no | cost | no |
| **BEST-Route** ([arXiv:2506.22716](https://arxiv.org/abs/2506.22716)) Ding et al. | ICML 2025 | model + **#samples** | yes | no | no — difficulty/quality | no | no | cost | no |
| **PILOT** ([arXiv:2508.21141](https://arxiv.org/abs/2508.21141)) Panda et al. | Findings of EMNLP 2025 | model selection | bandit feedback | no | per-model reward | **LinUCB bound** | no | **budget (knapsack)** | bandit logs |
| **BaRP** ([arXiv:2510.07429](https://arxiv.org/abs/2510.07429)) Wei et al. | 2025-10, preprint | model selection | no (prompt + prefs) | no | per-model reward | no | no | cost dial | bandit logs |
| **RouterDC** ([arXiv:2409.19886](https://arxiv.org/abs/2409.19886)) Chen et al. | NeurIPS 2024 | model selection | **no** | no | no — contrastive score | no | no | none stated | no |
| **MixLLM** ([arXiv:2502.18482](https://arxiv.org/abs/2502.18482)) Wang et al. | NAACL 2025 | model selection | yes (predicted) | ~ (evolving pool) | per-model reward | no | no | **quality+cost+latency** | bandit logs |
| **LLMRouterBench** ([arXiv:2601.07206](https://arxiv.org/abs/2601.07206)) Li et al. | 2026-01, Findings of ACL 2026 | benchmark, model routing only | n/a | no | n/a | n/a | n/a | latency-aware analysis | no |
| **Uncertainty-Aware Decision Making in MLLMs (survey)** ([arXiv:2608.17084](https://arxiv.org/abs/2608.17084)) Boudiaf, Hussain, Javed | 2026-08 | survey | n/a | n/a | **names the gap** | n/a | n/a | n/a | **names the gap** |

### What this changes

The 2024–2025 model-routing literature is not the binding constraint, and never
was: FrugalGPT, RouteLLM, RouterDC, MixLLM, BEST-Route, PILOT, BaRP, Unified
Cascade Routing and LLMRouterBench all choose **which model answers**. None
chooses **which intervention to apply next**, and none estimates an action's
marginal value against a stopping baseline.

Two 2026 preprints do bind, and neither was in the previous audit:

1. **Utility-Guided Agent Orchestration** already publishes the framing this memo
   was going to claim. It selects sequentially over `{respond, retrieve,
   tool_call, verify, stop}` by
   `a* = argmax_a Gain(a|s) - λ₁StepCost - λ₂Uncertainty - λ₃Redundancy`.
   So "utility-guided selection over heterogeneous interventions including an
   explicit STOP" is **no longer a novel framing** and must not be claimed.
   What it does not do is exactly the interesting part, and the authors say so
   plainly: *"Gain(a|s_t) measures the self-estimated marginal value of taking
   action a. In our implementation, this term is a heuristic self-estimated
   signal rather than a calibrated probability."* There is no learning from
   outcomes, no calibration, no confidence bound, no cost constraint beyond a
   step budget, and the evaluation is 200 HotpotQA examples on which the policy
   **loses to ReAct** (F1 0.2360 vs 0.2662).

2. **Knowing When to Quit** already establishes calibrated value-threshold
   stopping — abstain iff `V_β(x, y_{1:t-1}; π) < r_⊥`, calibrated by isotonic
   regression, with Proposition 4.2 proving dominance over never abstaining. So
   "conservative stopping when estimated value falls below the stop baseline" is
   **solved for the binary continue/abstain case** and must not be claimed as
   novel either. It estimates one value function for continuing, not per-action
   values, and models no cost or latency constraint.

The survey that closes the loop is [arXiv:2608.17084](https://arxiv.org/abs/2608.17084)
§9.2, which names the missing piece as an open problem: benchmarks need *"tasks
where the desired output may be an answer, an abstention, a clarification, a
retrieval step, a prediction set, a self-check, or an escalation"* with the next
priority being to *"make action costs explicit so that methods can be compared by
downstream utility."* Its survey of the field turns up methods that **trigger**
actions from uncertainty and none that **value** them.

### Verdict: NARROW

The surviving contribution is one clause of the original six, plus the artifact:

> **Replace the heuristic self-estimated gain with an outcome-supervised, calibrated, per-action marginal value learned from randomized action-outcome traces with recorded propensities — and test whether doing so actually beats the heuristic-gain utility policy, calibrated risk-threshold escalation, and prompt-only routing at equal measured cost.**

That is testable, it is not done anywhere audited, and there is a concrete reason
to think it matters: the one published utility-guided intervention policy uses an
uncalibrated self-estimate of gain and underperforms a simple ReAct baseline.
Whether calibrated, outcome-supervised gains fix that is an open empirical
question with a real chance of answering "no", which is what makes it worth
running.

**Claims now forbidden by this audit:**

- first/novel utility-guided selection over heterogeneous interventions — no ([arXiv:2603.19896](https://arxiv.org/abs/2603.19896));
- novel conservative value-threshold stopping — no ([arXiv:2604.18419](https://arxiv.org/abs/2604.18419));
- novel sequential intervention routing — no ([arXiv:2603.19896](https://arxiv.org/abs/2603.19896), [arXiv:2601.21162](https://arxiv.org/abs/2601.21162), [arXiv:2606.28733](https://arxiv.org/abs/2606.28733));
- novel abstention-as-action, calibrated routing, or budgeted routing — no (Knowing When to Quit; CP-Router/RACER; PILOT).

**The name CA-MVOI should be retired** — "marginal value of information" overstates
what survives. The work is now: *calibrated action-conditioned gain estimation for
intervention policies*.

### Unresolved source (the gate is provisional on this)

**Yin, S. and Zhang, R., "Failure-mode-aware uncertainty intervention routing for
large language models", Knowledge-Based Systems, 2026, DOI
[10.1016/j.knosys.2026.116685](https://doi.org/10.1016/j.knosys.2026.116685).**

Existence, title, authors, year, journal and DOI are confirmed via the Crossref
API. The full text could **not** be obtained: ScienceDirect returns HTTP 403 to
both the article and abstract URLs, Semantic Scholar records the paper with
`openAccessPdf.status = "CLOSED"` and a null abstract, and web search does not
surface an accessible copy or preprint.

Two consequences, recorded rather than papered over:

1. The previous version of this memo asserted that this paper "predicts one of
   commit, deliberation, retrieval-augmented regeneration, or defer from a
   behavioral uncertainty signature" and used that to set the novelty bar. **That
   description is unverified**; no primary source for it was obtainable. It has
   been removed from the comparison table, whose row for this paper is marked
   UNVERIFIED.
2. The same memo cited the venue as *Information Sciences*. It is
   **Knowledge-Based Systems**. Corrected.

This is the single closest work by title and it is the one source the audit could
not read. The NARROW verdict therefore stands **provisionally**: if that paper
already learns calibrated per-action repair value from outcome data, the surviving
contribution collapses and the verdict becomes STOP. Obtaining it — via
institutional access, interlibrary loan, or an author request — is a blocking
prerequisite before any paper claim, though not before the Gate 2 pilot, which is
worth running for its own sake.

## Candidate directions

> **Superseded in part by Checkpoint 2 (2026-09-06).** The full-text audit returned
> **NARROW**, not GO. Sequential utility-guided selection over heterogeneous
> interventions ([arXiv:2603.19896](https://arxiv.org/abs/2603.19896)) and calibrated
> value-threshold stopping ([arXiv:2604.18419](https://arxiv.org/abs/2604.18419)) are
> both already published. Read the sections below as the original hypothesis; the
> surviving contribution is stated in the audit section and is narrower than what
> follows. The name CA-MVOI is retired.

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

> **Superseded in part by Checkpoint 2 (2026-09-06).** The full-text audit returned
> **NARROW**, not GO. Sequential utility-guided selection over heterogeneous
> interventions ([arXiv:2603.19896](https://arxiv.org/abs/2603.19896)) and calibrated
> value-threshold stopping ([arXiv:2604.18419](https://arxiv.org/abs/2604.18419)) are
> both already published. Read the sections below as the original hypothesis; the
> surviving contribution is stated in the audit section and is narrower than what
> follows. The name CA-MVOI is retired.

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
**CA-MVOI (as originally scoped):** ruled out by Checkpoint 2 — its framing and its stopping rule are both published. **The narrowed successor** (calibrated, outcome-supervised per-action gain estimation, learned from randomized action-outcome traces) remains medium-high risk and is publishable only if it clears the sequentiality and robustness gates AND Yin & Zhang turns out not to have done it. Its strongest possible contribution is a validated principle: *routing should allocate interventions by calibrated marginal repair value, not by answer risk*.  
**Robust variant:** high methodological burden but stronger ML story.  
**Load-aware variant:** high systems burden and dependent on real infrastructure, but potentially strongest MLSys fit.

The correct outcome may be that Triage remains an open-source engineering system rather than a top-tier research paper. The strategy explicitly treats that outcome as success over forcing an unsupported novelty claim.
