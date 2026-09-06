# Pre-live defect spec — Triage trace substrate

**Path:** `docs/research/pre-live-defect-spec.md`
**Companion:** `docs/research/novelty-strategy.md`
**For:** Codex, working on `research/trace-substrate`
(PR https://github.com/ishaannk/triage/pull/1).
**Baseline:** `7479ff2` (this document is committed at the tip of that branch; if the branch has moved since, rebase this list onto the tip and re-verify each line reference before starting). Pull before starting — the invalid prompt-only AUC claim
has already been retracted in `RESEARCH_STRATEGY.md`, so do not re-report it.
**Status:** all eleven defects below were independently verified against the code on
2026-09-07. Your Checkpoint 5 write-up was correct on every point it raised, and
D8 (informational actions) is a better catch than anything in the original audit.

**Scope discipline.** These are repairs to an existing substrate. Do not add a
controller, do not make a performance claim, and do not run anything with
`--live` until D1–D3 are fixed and their tests pass. Gate 1 remains provisionally
NARROW and gates the *paper claim*, not this work — none of these repairs depends
on obtaining the Yin & Zhang manuscript.

Suggested commit split: one commit for D1–D3 (money), one for D4–D6 + D9–D10
(inference validity), one for D7–D8 (measurement validity). Each with its tests.

---

## Severity 1 — money. Nothing runs `--live` until these four are done.

### D1. The budget is charged *after* the action executes

`app/trace/branching.py:244-245` and `:300-301`

```python
result = await executor.execute(answer, ctx)   # money already spent
run_budget.charge(result.cost)                 # cap checked afterwards
```

`RunBudget.charge` raises `BudgetExceeded` only once the provider call has
already been made and billed. The docstring on `RunBudget` claims it "refuses an
action *before* it is executed" — that is false and should be corrected along
with the behaviour.

**Required:** estimate the cost of an action before executing it and refuse when
the estimate would breach the cap. A defensible estimate is
`tokens_in ≈ len(prompt)/4` plus `max_tokens` completion at the model's
`cost_out`, i.e. the worst case, from the prices already snapshotted in the run
manifest. Charge the *actual* cost after execution, and treat the estimate as
admission control only.

**Acceptance test:** with a cap that admits exactly one action, a fan-out over an
item requiring two must raise `BudgetExceeded` with **`llm_calls` on the run
equal to 1, not 2**. Assert the call count, not just the exception — the current
code would pass an exception-only test while still having spent the money.

### D2. The budget resets on every item

`app/trace/branching.py:232` (`run_budget = RunBudget(budget)`) against
`scripts/collect_traces.py:175` and `:200`, which construct one `Budget` and pass
the same starting value into every `collect_fanout` call.

Spend does not accumulate across items. **`--max-usd 5.00` over 120 items was a
$5-per-item cap, i.e. up to ~$600.** This is the defect most likely to have cost
real money.

**Required:** the run budget is owned by the caller and threaded through, not
reconstructed per item. Pass a single mutable `RunBudget` into `collect_fanout`
(accept either and adapt), or return the updated state and reassign. `Budget`
stays immutable; `RunBudget` is the accumulator.

**Acceptance test:** a two-item collection with a cap sized for 1.5 items stops
during item 2, and `run_budget.state.spent_usd` after item 1 is strictly greater
than 0 when item 2 begins. Assert accumulation directly, not just termination.

### D3. Served mode enforces no cap at all

`scripts/collect_traces.py`, the `--mode served` branch calls
`route_and_answer(req, recorder=recorder)` with no budget parameter.

**Required:** either thread a budget through served mode, or — simpler and
honest — **refuse `--mode served --live` outright** with a message saying served
mode has no cost control. Do not leave an uncapped live path reachable.

**Acceptance test:** `--mode served --live --max-usd 1.00` either respects the cap
or exits non-zero with that message. A test asserting the refusal is fine.

### D11. `--max-usd inf` passes validation and lifts the cap entirely

*Found by Codex, 2026-09-07. Verified.*

`scripts/collect_traces.py:59`

```python
if args.live and (args.max_usd is None or args.max_usd <= 0):
    ap.error("--live requires an explicit positive --max-usd cap")
```

`argparse` with `type=float` accepts `inf` and `nan`, and neither is `<= 0`, so
both pass. Measured consequence:

| `--max-usd` | passes CLI check | `can_afford($1e9)` |
|---|---|---|
| `inf` | yes | **True — unlimited spend** |
| `nan` | yes | False — every action refused |
| `5.0` | yes | False |

`inf` is the dangerous half: the run reports a cap, the operator believes there
is one, and there is not. `nan` is merely fail-closed and confusing.

**Required:** reject any `--max-usd` that is not finite and strictly positive, in
the same check. `math.isfinite` covers both cases. Apply the same validation to
any other numeric CLI argument that gates spend or sample size.

**Acceptance test:** `--live --max-usd inf` and `--live --max-usd nan` both exit
non-zero with the cap error, and a finite positive value still succeeds.

**Note on scope:** fixing this does **not** make live collection ready. D1 (charge
after execution), D2 (per-item reset) and D3 (uncapped served mode) are the
run-wide enforcement defects, and all three remain. A validated cap that is then
enforced per item is still a per-item cap.

---

## Severity 2 — inference validity. These make a wrong answer look convincing.

### D4. `gate2_report` never calls `assert_estimable`

`app/trace/support.py` defines it; `grep -rn assert_estimable app/ scripts/`
returns nothing outside the definition. The project's own hard rule is that every
OPE or causal estimate routes through it. A refusal gate nothing calls is
decoration.

**Required:** `gate2_report` calls `assert_estimable` before producing any
estimate and returns `{"verdict": "REFUSED", "support_error": str(e), ...}` on
`SupportError` rather than propagating. Add an explicit
`thresholds: SupportThresholds | None = None` parameter so a caller can widen
them deliberately and visibly, never by accident.

**Acceptance test:** a run whose trajectories include an unsupported action
yields `verdict == "REFUSED"` and produces no `feature_set_comparison`.

### D5. Gate 2 accepts synthetic, non-analysis-grade input

`gate2_report` never consults `store.validate_run`, so it will happily analyse a
forced-mock run — which is exactly what produced the Checkpoint 4 numbers.

**Required:** call `validate_run(run_id)` and, when `analysis_grade` is false,
either refuse or stamp every returned figure with
`"analysis_grade": false, "not_evidence": true`. The verdict must never be `GO`
from a non-analysis-grade run.

**Acceptance test:** `gate2_report("gate2-pilot-mock")` reports
`analysis_grade == False` and a verdict that is not `GO`.

### D6. The item-ID split fallback bypasses the leakage-safe splits

`app/trace/analysis.py:264-272`. When `train` or `test` is empty it falls back to
a half-split ordered by `item_id`, which discards the group assignment that
`app/trace/splits.py` exists to compute. Two branches of one item, or two
near-duplicate prompts, can land on opposite sides.

**Required:** delete the fallback. If the configured splits do not populate both
sides, return `{"status": "insufficient_split_coverage", ...}`. Refusing is the
correct behaviour; a fabricated split is not a weaker answer, it is a different
one.

**Acceptance test:** rows carrying only `split="train"` return
`insufficient_split_coverage` and never an AUC.

### D9. The verdict compares point AUCs with no paired CI

`app/trace/analysis.py:334`:

```python
if (c["response_only"]["auc"] or 0) > (c["prompt_only"]["auc"] or 0)
```

A raw point comparison. `paired_bootstrap_diff` is implemented and **never
called** in production. On the sample sizes in play, 0.62 vs 0.58 is noise, and
this rule would return `GO` on it.

**Required:** the `GO` branch requires a paired bootstrap on per-item scores whose
95% CI **excludes zero** — `paired_bootstrap_diff(...)["excludes_zero"]` is
already the right shape. Record the interval in the report next to the verdict.

**Acceptance test:** two feature sets with a small point difference and
overlapping intervals produce `NARROW`, not `GO`.

### D10. Calibration rows are pooled into the test set

`app/trace/analysis.py:265`: `r["split"] in ("test", "calib")`.

`calib` exists so calibration is fitted on data that is neither train nor test.
Pooling it into test destroys that separation and inflates the test set with rows
reserved for another purpose.

**Required:** `train` fits, `calib` calibrates, `test` scores — three disjoint
roles. If calibration is not yet implemented, `calib` is simply held out and
unused; do not spend it as test data.

**Acceptance test:** no row with `split == "calib"` appears in the scored test
indices.

---

## Severity 3 — measurement validity. Without these a live run measures nothing.

### D7. Fan-out writes no prompt features, so the control condition is empty

`app/trace/branching.py`, `collect_fanout(prompt_features=...)` receives only
`{"task_family": ...}` from `scripts/collect_traces.py:203`.

`FEATURE_SETS["prompt_only"]` needs `message_chars`, `predicted_difficulty` and
`approx_prompt_tokens`. In `gate2-pilot-mock` all three are **identically 0.0
across all 40 items** — verify with:

```python
from app.trace.analysis import assemble
rows = assemble("gate2-pilot-mock")
{f: sorted({r["features"][f] for r in rows})
 for f in ("message_chars", "predicted_difficulty", "approx_prompt_tokens")}
# -> every one is [0.0]
```

This is why AUC 0.500 appeared, and it is why that claim is now retracted. The
Signal Gate's whole question is whether response state beats prompt-only; with a
constant control there is no comparison to make.

**Required:** populate the same prompt features the served path already records
in `TraceRecorder.begin` / `update_features` — `message_chars`,
`approx_prompt_tokens`, `long_context`, and `predicted_difficulty` plus
`prefilter_route` from `app/router/prefilter.py`. Reuse the served path's code so
the two collection modes cannot drift apart.

**Acceptance test:** a fan-out run produces **more than one distinct value** for
each `prompt_only` feature across items. Assert non-degeneracy, not presence — a
key holding 0.0 everywhere would pass a presence check.

### D8. Depth-1 cannot measure the informational actions, on any provider

Your finding, and the sharpest one. RESAMPLE, SELF_CHECK and RETRIEVE update
signals and evidence but **do not modify the candidate answer** — see
`app/trace/actions.py`, where `_resample`, `_self_check` and `_retrieve` all
return `answer=ctx.answer`. Under the depth-1 fan-out, `<action> → STOP` returns
the same answer as `STOP` alone, so **Δ = 0 by construction** for those three,
live or mock.

Consequence: three of the seven measurable actions cannot be evaluated at all by
the current design, and a live depth-1 run would reproduce three of those zeros
and mean nothing by them. Their value is only realisable through a *subsequent*
action that consumes the new information — VERIFY reading retrieved evidence,
or an escalation triggered by observed instability.

**Required:** extend `collect_fanout` to depth 2 for the informational actions:
after `a ∈ {RESAMPLE, SELF_CHECK, RETRIEVE}`, branch again over the actions that
can change the answer (`VERIFY`, `STRONGER_MODEL`, `ANSWER`, `TOOL`) plus `STOP`,
and difference against the matching depth-1 branch so the increment attributable
to the information is isolated. Extend rather than replace — the depth-1 path is
what the committed fixtures contain, and it stays correct for the four
answer-changing actions.

Cost scales multiplicatively; gate the depth with a flag (`--depth 1|2`, default
1) and make the live budget interaction explicit.

**Acceptance test:** in a depth-2 run, `retrieve → verify → STOP` is recorded as a
distinct branch whose parent is the `retrieve` state, and its Δ is differenced
against `retrieve → STOP` rather than against the root `STOP`.

---

## Acceptance checklist for the whole spec

- [ ] `python -m pytest -q` passes and the count increased; each defect above has
      a test that **fails on `7479ff2`** and passes after. State the pre-fix
      failure in the commit message.
- [ ] `python -m pyflakes app/ scripts/ tests/` clean; worktree clean.
- [ ] `tests/test_trace_adapter.py::test_tracing_does_not_change_the_routing_decision`
      still passes — the legacy router's decisions remain untouched.
- [ ] No `data/*.json` receipt changed. Committed fixtures under `data/traces/`
      are **not** regenerated; if D7/D8 change what a run records, write a **new**
      run id and leave the old ones as the historical record of what depth-1
      collection produced.
- [ ] `RESEARCH_STRATEGY.md` gains a Checkpoint 6 block: what was repaired, the
      evidence, what is still limited, and one exact next command.
- [ ] Every number reported carries a CI and a sample size, and you state plainly
      what the result does not show.
- [ ] Still no `--live` run, no controller, no performance claim.

## After the repairs

The first honest re-run is **mock**, to confirm the machinery changed the right
things — expect prompt-only features to vary, `verdict` to be `REFUSED` or
non-`GO` on mock input, and depth-2 branches to exist. Only then is a live run
worth proposing, and it needs an explicit approved cap and a corrected D1–D3.
