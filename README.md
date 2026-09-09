# Triage

Triage is a reliability-aware text LLM router and an experimental action-outcome
trace substrate. The production router uses black-box signals from an actual
model response to choose among serving, retrieving, verifying, using a calculator,
escalating, or abstaining to PENDING_REVIEW. Its heuristic decisions are preserved.

**No performance claim is currently supported by this research program.**
NVIDIA generation works with explicit current-model settings, but its latest
smoke contains Ultra service failures and is excluded from evidence. The Groq
fallback (GPT-OSS 20B to 120B) completed a live, analysis-grade smoke after fixing
score-JSON envelope handling. Gate 2 remains INCONCLUSIVE at smoke-test size.
The full Groq collection refuses because its conservative planning estimate
exceeds the approved cap. Calibration and C1/C2/C4 remain unestablished.
See [provider recovery evidence](docs/research/checkpoint10/provider-notes.md).

Research model IDs: `nim-nemotron-lightning-30b` / `nim-nemotron-ultra-550b`,
with `groq-gpt-oss-20b` / `groq-gpt-oss-120b` as a separately recorded fallback
pair. Pass them explicitly as `--small` and `--big`; historical router defaults
are preserved. Credentials belong in ignored `.env` (`NVIDIA_API_KEY`,
`GROQ_API_KEY`). Successful catalog access does not guarantee generation availability.

The historical automatic default points to a retired hosted Llama model. For
an explicit current application default, set `TRIAGE_DEFAULT_MODEL=groq-gpt-oss-20b`
in `.env`; its escalation target is `groq-gpt-oss-120b`. An unknown override
refuses configuration. Leaving the override unset preserves historical routing.

## Run the application

```powershell
python -m pip install -r requirements-dev.txt
$env:TRIAGE_FORCE_MOCK = '1'
python -m uvicorn app.main:app --reload
```

The default application address is http://localhost:8000. Provider configuration
is described in `.env.example`. Mock mode makes no paid calls. Research collection
requires an explicit `--live` flag and a finite positive `--max-usd` to enable live
providers; live served-mode collection is refused because the legacy router does
not enforce a run-wide cap.

## Trace and research pipeline

`app/trace/` records content-addressed states, canonical actions, outcomes,
propensities, provider provenance, immutable manifests, costs, and grouped splits.
Schema 1.3.0 reads historical 1.0.0-1.2.0 runs. Run IDs are single-use.

Depth-2 fan-out shares one draft, executes each feasible intervention, then adds
answer-changing continuations after RESAMPLE, SELF_CHECK and RETRIEVE. These three
information actions preserve the draft; their depth-1 quality gain is zero by
construction. Continuing after information is distinct from the causal value of
acquiring it. The C2 ablation includes a matched continuation without information.

Run the assembled, explicitly synthetic diagnostic pipeline:

```powershell
python scripts/run_gate2.py --n 8 --dataset mixed --depth 2 --policy balanced --diagnostic-scores --run-id my-new-mock-run
```

The command prints a planning cost estimate before generation, collects under one
run-wide accumulator, freezes the retrieval corpus, validates traces, checks
support, runs Gate 2, calibrates eligible action estimators, and attempts C1/C2/C4.
Live collection defaults to `--rps 0.6` across the run, with bounded rate-limit
and transient 502/503/504 retries. Research provider failures never silently become
mock answers. Exhausted retries remain unlabelled failures. Every live attempt,
including a retry, reserves cost against the same pinned model-price ledger.
Unknown usage retains its reservation and is disclosed separately from reported
tokens. A finite cap is admission control against listed prices, not an invoice.
`--max-input-bytes` optionally sets an enforced request-size budget; oversized
requests refuse before execution and are never truncated. It does not override
completion limits or guarantee that a whole dataset fits the approved cap.
Artifacts under `data/traces/<run-id>/` include:

- `manifest.json`, `splits.json`, `trajectories.jsonl`, and `collect.log`;
- `preflight.json` and the frozen `corpus.json`;
- `calibration.json` with fitted model coefficients and isotonic maps when estimable;
- `report.json`, the consolidated verdicts, intervals, refusals, and cost accounting.

Live runs also preserve request-budget journals and per-attempt raw-response
journals before starting the next request. The final operational report includes
latency intervals clustered by item, actual reported tokens, failure counts,
and unknown-usage reservations. Production-router overhead is not inferred from
the separate fan-out executor. Provider outages and daily quotas can still stop
a run; completed trajectories remain evidence with their actual statuses.

`--diagnostic-scores` explicitly enables seeded synthetic score JSON through the
comparator's real client path. It is forbidden with `--live`. Ordinary mock scoring
still fails closed on malformed output. The comparator is an equation-level Triage
adaptation of arXiv:2603.19896, not a reproduction of its released search/stop experiment.

Gain models fit only on `train`; isotonic maps fit only on `calib`; reliability
bins and expected calibration error are evaluated only on `test`. The target is
normalized marginal quality gain, not correctness probability. Missing split
coverage, degenerate targets, and missing action support produce explicit refusals.
An interval with null endpoints and n=0 means unavailable, not a zero effect.

## Research status and limits

- **Gate 1: provisionally NARROW.** Deltas must be established against fully read
  prior work. Yin & Zhang (Knowledge-Based Systems, DOI 10.1016/j.knosys.2026.116685)
  remains an unread related-work limitation and overlap risk.
- **Gate 2: undecided on real data.** Synthetic runs cannot return GO.
- **C1:** the gain-replacement experiment is implemented; insufficient calibration
  or comparator coverage refuses a numerical comparison.
- **C2:** mock nulls test the apparatus. They do not refute real information value.
- **C4:** the same estimator is fitted to matched served and counterfactual data
  when support permits. Unsupported served actions cannot be assigned invented
  values. Ranking disagreement is not policy superiority.
- **Gate 3 and a learned controller:** still gated on real evidence.

The approved program cap is $2.00, with earlier listed-price spend accounted
against it. New live commands require an explicit remaining cap and fresh run id;
the pipeline never infers authorization from a configured credential:

```powershell
python scripts/run_gate2.py --n 120 --dataset gsm8k --depth 2 --policy balanced --small nim-nemotron-lightning-30b --big nim-nemotron-ultra-550b --live --max-usd <APPROVED_REMAINING_CAP> --run-id <FRESH_RUN_ID>
```

The command refuses before generation if its whole-run estimate exceeds the cap.
Per-action admission and actual settlement still apply. Listed-price token costs
are not provider invoices; an admission estimate is not an invoice guarantee.
No second-domain robustness result is established by the mixed mock fixture.

The [charter](CHARTER.md) keeps the project text-only, API-cost focused, and limited
to black-box signals. Multimodal routing, local vLLM/GPU-second/energy accounting,
and white-box attention signals are out of scope. The trace experiments sit beside
the production router. Historical charter invoice wording does not convert
listed-price estimates into billed costs.

## Verification and evidence

```powershell
python -m pytest -q
python -m pyflakes app/ scripts/ tests/
```

`tests/test_end_to_end.py` exercises the assembled pipeline from a cold process,
checks provenance and refusals, preserves committed receipts, and compares router
decisions with tracing enabled and disabled. See [RESEARCH_STRATEGY.md](RESEARCH_STRATEGY.md)
for Checkpoint 8 evidence and limitations, and [the novelty strategy](docs/research/novelty-strategy.md)
for the forbidden claims and candidate ablations.
