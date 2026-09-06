# Research strategy addendum: measured changes to readable prior work

Adopted 2026-09-07. This supersedes the manuscript-access gate in Checkpoint 5;
it does not supersede the pre-live defect specification or the charter.

## Governing decision

Build and evaluate a specific change to a fully read base method. Permission to
investigate a candidate is not a novelty or performance finding. Keep the
forbidden claims from Gate 1: no first intervention routing, novel sequential
routing, novel utility-guided framing, or generic novel calibrated stopping.

Yin and Zhang is related work with a disclosed overlap risk, **not a prerequisite
for repairs, experiments, or a narrowly scoped claim against a readable base**.
The publisher preview supports learned intervention selection from response
behavior. It does not settle the exact target or calibration procedure. Cite only
that supported scope and disclose the unread methods. If the manuscript becomes
available and contains the chosen change, retire that candidate and evaluate the
next one. Do not manufacture a distinction. If no candidate survives, report
that outcome. Continue public manuscript searches opportunistically; author
outreach still requires explicit authorization. The existing access record and
receipts remain historical evidence and are not rewritten.

## Base methods and reproduction contract

- [Utility-Guided Agent Orchestration, arXiv:2603.19896](https://arxiv.org/abs/2603.19896):
  primary base for the gain-estimator comparison. The earlier audit identifies
  a heuristic gain in a heterogeneous-action utility rule.
- [Knowing When to Quit, arXiv:2604.18419](https://arxiv.org/abs/2604.18419):
  base for investigating a per-action extension of calibrated binary stopping.

Before implementation, read each relevant full text, pin its version, and record
section references for the method and experimental setup. Inspect any released
code and pin its revision and license. Document prompts, action semantics, gain
scale, utility weights, cost definition, stopping rules, and tie-breaking.
Distinguish a faithful reproduction from an adaptation where details are missing
or Triage uses a different environment. No baseline is implemented from an
abstract, and the previous audit alone is not an executable specification.

## Ranked candidates and required evidence

These are proposed differentiators, not established contributions. C1 is the
highest-ranked experiment until evidence rules it out. The intended narrative
combines C2's question about information use, C4's measurement artifact, and C1's
gain-estimator experiment if it survives. No evidence currently ranks their
novelty risks as low, or establishes that any is robust to unknown prior work.

| Candidate | Deliberate change | Required controlled comparison | Failure decision |
|---|---|---|---|
| C1 | Replace the heuristic gain with an outcome-supervised, held-out calibrated per-action gain | Same actions, state information, utility weights, stopping rule, budget, and evaluator; change only the gain estimator. Paired held-out utility difference and CI; include estimation/feature-acquisition overhead. | No supported effect: do not call this the differentiator; examine the next candidate. |
| C2 | Value information-acquiring actions through a continuation that can use their output | Same items/actions at depth 1 and depth 2, plus a continuation control with the acquired information withheld and a direct-continuation control; compare at equal total cost. | A structural zero followed by a nonzero composite effect alone establishes harness semantics, not novelty or improved allocation. |
| C3 | Replace point-value selection with per-action lower confidence bounds | Equal-budget point-versus-LCB selection; measure harmful interventions, utility, coverage/abstention and constraint satisfaction with paired uncertainty. | No supported risk/utility benefit: decline the contribution. No multi-action dominance theorem is inherited from the binary base. |
| C4 | Collect counterfactual outcomes with explicit support and refusal rules | Same learning/evaluation procedure using served-path versus counterfactual training data, checked against an independent supported reference. Unsupported served-path estimates must refuse. | Different conclusions alone are insufficient; require demonstrated coverage, identifiability, or estimation improvement. |

Freeze the estimand, action set, feature availability, train/calibration/test
groups, stopping criteria, weights, and budget comparison before inspecting
held-out results. Candidate switching after evaluation must be recorded; a new
candidate needs fresh confirmatory data or a justified multiplicity procedure.
Report sample sizes and paired CIs for every empirical comparison. An interval
including no effect is inconclusive, not proof of equivalence or no effect.

## C2: what the implementation establishes, and what remains a hypothesis

The current RESAMPLE, SELF_CHECK, and RETRIEVE executors preserve the candidate
answer. Forced STOP after these actions therefore cannot expose answer repair.
This is a property of these executor definitions, not a universal property of
actions named retrieval or resampling. Voting and retrieval-conditioned
regeneration can change an answer when explicitly included in an action.

Define a consumer of each acquired observation. Existing resamples are not
automatically consumed by a later generation, and adding a second step alone
does not fix that. Distinguish information value under a fixed continuation from
the total value of buying both acquisition and continuation. Charge both steps,
failed attempts, and policy/feature overhead. A state-only action may have zero
immediate correctness gain and negative immediate net utility because it costs
resources. Do not report those as the same quantity.

The nulls in Checkpoint 4 motivate this investigation; they do not establish its
novelty or explain why the published utility policy scored as it did. That causal
explanation needs an experiment on the base implementation. Depth-2 branching is
a measurement prerequisite for these candidates, not evidence that a sequential
policy beats a strong myopic policy. Gate 3's equal-budget and shift tests remain.

For C3, distinguish predictive spread from a valid confidence bound on expected
gain and account for selection across actions. For C4, neither a randomized
designation of a served branch nor a recorded positive propensity alone proves
conditional support. These distinctions belong in the defect repairs and tests.

## Execution order and current blocker

1. Obtain and implement the authoritative D1–D10 defect spec, retaining refusal
   paths, router equivalence, immutable receipts, and isolated tests. Map each
   requirement to its behavioral tests and evidence.
2. Collect a fresh mock run after the repairs, with complete run artifacts.
   Validate the changed machinery; synthetic outcomes cannot authorize a
   scientific GO verdict.
3. Implement and validate the heuristic-gain base comparator from the full text
   and any usable released code, with explicit reproduction/adaptation labeling.
4. Implement the chosen candidate's experimental variant and ablation. This is
   authorized experimental work, not a production-controller deployment or a
   supported research claim. Gate 2 evidence still governs claims of learned
   response-conditioned repair value.
5. Collect live evidence only after the preceding work and explicit approval of
   a finite dollar cap. No cap has been approved.

The authoritative [pre-live spec](pre-live-defect-spec.md) and
[strategy](novelty-strategy.md) were supplied during this turn and are committed
at 8170e03. This document adds experimental interpretation notes; the supplied
spec is the requirements source. Its scope includes D11 in addition to D1-D10.
