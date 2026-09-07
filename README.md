# Triage

Triage is a reliability-aware text LLM router and an experimental action-outcome
trace substrate. The production router uses black-box signals from an actual
model response to choose among serving, retrieving, verifying, using a calculator,
escalating, or abstaining to PENDING_REVIEW. Its heuristic decisions are preserved.

**No performance claim is currently supported by this research program.** The approved live smoke could not obtain a real generation: both requested
NVIDIA endpoints returned HTTP 410 (retired). Its fallback outcomes are synthetic;
the full run is blocked pending an approved replacement model pair. Calibration and ablation machinery are
implemented; real-model repairability, utility gains, and sequential superiority
remain unresolved. Historical benchmark receipts are preserved for audit and are
not evidence for the new action-policy contribution.

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
retries and recorded request events. Exhausted retries remain unlabelled failures.
Artifacts under `data/traces/<run-id>/` include:

- `manifest.json`, `splits.json`, `trajectories.jsonl`, and `collect.log`;
- `preflight.json` and the frozen `corpus.json`;
- `calibration.json` with fitted model coefficients and isotonic maps when estimable;
- `report.json`, the consolidated verdicts, intervals, refusals, and cost accounting.

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

No live cap has been supplied for Checkpoint 8. After an operator supplies one,
replace `<CAP>` with that approved finite dollar value:

```powershell
python scripts/run_gate2.py --n 120 --dataset gsm8k --depth 2 --policy balanced --live --max-usd <CAP> --run-id gate2-live-20260907
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
