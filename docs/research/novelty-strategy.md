# Addendum to the defect spec — the novelty strategy

**Path:** `docs/research/novelty-strategy.md`
**Companion:** `docs/research/pre-live-defect-spec.md` (the ten-defect spec)
**For:** Codex, `research/trace-substrate`, PR https://github.com/ishaannk/triage/pull/1
**Read with:** `docs/research/pre-live-defect-spec.md`, the ten-defect spec. This does not replace it; the repairs
still come first.

## The strategy, stated plainly

Build the contribution as a **delta on prior work we can read in full**, not as a
clean-sheet claim. Take an existing published method as the base, change one part
deliberately, and show that the change is what produces the result. This is a
normal and defensible way to do research. Two rules make it work rather than fail:

1. **Only claim a delta against a paper you have read in full.** You cannot know
   whether your tweak is already inside a method you have only seen the abstract
   of. This is why the base must be openly available.
2. **The delta must be measurable.** A tweak that changes the method but not the
   results is a difference, not a contribution. Every candidate below comes with
   the ablation that would demonstrate it matters — if that ablation shows no
   effect, the tweak is not the differentiator and you move to the next one.

## Stop treating Yin & Zhang as a blocker

The closed-access paper (KBS 2026, `10.1016/j.knosys.2026.116685`) was gating
progress because the original plan positioned the work as "first to do
intervention routing". That framing is dead anyway — Gate 1 already found two
2026 preprints that establish it.

So change what the paper is *for*. It is no longer a gate on whether to proceed;
it is **a related-work citation and a disclosed risk**. Build the delta against
the two works that are fully readable on arXiv, cite Yin & Zhang from its
publisher preview for what the preview actually supports, and state in the
limitations that its method section could not be obtained. That is honest,
standard practice, and it unblocks everything.

Keep pursuing the manuscript in the background. If it arrives and it already
contains the chosen delta, you switch to the next candidate below — that is why
there is a ranked list rather than one bet.

## The two readable base works

Both fully accessible, both already audited:

- **[arXiv:2603.19896](https://arxiv.org/abs/2603.19896)** — Utility-Guided Agent
  Orchestration. Sequential selection over `{respond, retrieve, tool_call,
  verify, stop}` by `argmax_a Gain − λ₁StepCost − λ₂Uncertainty − λ₃Redundancy`.
  **Its `Gain` is, in the authors' own words, "a heuristic self-estimated signal
  rather than a calibrated probability."** No confidence bound. No learning from
  outcomes. And it **loses to ReAct** (F1 0.2360 vs 0.2662) on its own 200-item
  HotpotQA evaluation.
- **[arXiv:2604.18419](https://arxiv.org/abs/2604.18419)** — Knowing When to
  Quit. Calibrated value-threshold stopping, `abstain iff V_β < r_⊥`, isotonic
  calibration, dominance proposition. **Binary** continue/abstain only, one value
  function, no per-action values, no cost or latency constraint.

The gap between them is the opening: one has heterogeneous actions with an
uncalibrated gain, the other has calibration and a stopping guarantee but only
two actions. Nobody in the audit has both.

## Ranked candidate differentiators

Pick the highest one that survives what you know. Each is a *one-part change* to
a readable base, with the ablation that proves it.

### C1 — Replace the heuristic gain with a calibrated, outcome-supervised one
**Base:** arXiv:2603.19896. **Change:** `Gain(a|s)` stops being an LLM
self-estimate and becomes `Δ̂_a(s)`, trained on observed per-action outcomes from
randomized counterfactual traces, and calibrated on a held-out split.
**Why it is a real delta:** the base paper's own framing invites it, and the base
policy underperforms a simple baseline — so there is a concrete result to improve.
**Ablation that proves it:** same action space, same λ weights, same budget; swap
only the gain estimator. Report the paired difference with a bootstrap CI.
**Risk:** this is the candidate most likely to overlap Yin & Zhang.

### C2 — Separate information-acquiring actions from answer-changing ones
**Base:** either. **Change:** recognise that RESAMPLE, SELF_CHECK and RETRIEVE
**cannot change the answer** — they change the *state*. Their value is realisable
only through a subsequent action that consumes the information, so a one-step
utility model scores them at exactly zero by construction and any policy fitted
that way will never take them.
**Why it is a real delta:** this was found empirically while building the
measurement apparatus (see the Checkpoint 4 corrections), and no audited work
distinguishes the two action classes or values them at depth 2. It is also a
concrete explanation for why the base paper's utility policy underperforms.
**Ablation that proves it:** depth-1 versus depth-2 valuation of the same three
actions on the same items. If depth-2 assigns them non-zero value and depth-1
assigns exactly zero, the distinction is demonstrated rather than asserted.
**Risk:** low. This is the most defensible candidate and the least likely to be
scooped, because it emerges from the trace machinery rather than from the
literature.

### C3 — Extend calibrated conservative stopping from binary to heterogeneous actions
**Base:** arXiv:2604.18419. **Change:** replace the single value function with
per-action lower confidence bounds; stop iff every feasible action's LCB is
non-positive. `ActionScore` already carries `delta_hat / sigma_hat / lcb`.
**Why it is a real delta:** the base proves dominance for two actions; the
multi-action analogue is not established anywhere audited.
**Ablation that proves it:** LCB selection versus point-estimate selection at
equal budget, measuring harmful-intervention rate.
**Risk:** medium.

### C4 — The counterfactual action-outcome protocol as the contribution
**Base:** none; this is an artifact/benchmark contribution.
**Change:** the randomized per-action trace format with recorded propensities,
support and positivity diagnostics, and a refusal gate — most of which is already
built in `app/trace/`.
**Why it is a real delta:** the Aug-2026 survey
[arXiv:2608.17084](https://arxiv.org/abs/2608.17084) §9.2 explicitly names this
as an open problem, and its own survey finds methods that *trigger* actions from
uncertainty but none that *value* them.
**Ablation that proves it:** show that a policy fitted on served-path traces and
one fitted on counterfactual traces reach different conclusions about the same
actions.
**Risk:** lowest. Survives almost any content in the unread paper, because it is
about how evidence is collected rather than what the policy computes.

**Recommended combination: C2 as the intellectual contribution, C4 as the
supporting artifact, C1 as the headline experiment if it survives.** C2 and C4
together are robust to Yin & Zhang containing almost anything.

## What this does not license

- Do not claim novelty for the framing, for sequential intervention routing, for
  utility-guided selection, or for calibrated stopping. Those were ruled out at
  Gate 1 and the list of forbidden claims stands.
- Do not describe a change as a contribution before the ablation shows it changes
  the result. "Different from prior work" and "better than prior work" are
  separate claims and only the second is worth publishing.
- Do not reimplement a baseline from its abstract. If the base method is not
  readable in full, it cannot be a base — it can only be related work.
- The pre-live defect repairs still come first. None of this is measurable until
  the budget, support-gate and prompt-feature defects are fixed, and C2's whole
  ablation depends on depth-2 branching (defect D8) existing.

## Order of work

1. Repairs D1–D10 from the defect spec.
2. Mock re-run to confirm the machinery changed the right things.
3. Implement the base policy from arXiv:2603.19896 — the heuristic-gain
   comparator. It is the thing every candidate above is a delta against, so
   without it there is no measurable claim.
4. Then the chosen differentiator, with its ablation.
5. Live evaluation only after that, with an approved cap.
