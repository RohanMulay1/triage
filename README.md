# Triage
# Triage

Triage is a reliability-aware text LLM router and an experimental action-outcome
trace substrate. The production router uses black-box signals from an actual
model response to choose among serving, retrieving, verifying, using a calculator,
escalating, or abstaining to PENDING_REVIEW. Its heuristic decisions are preserved.

**No performance claim is currently supported by this research program.**
A full live 120-item depth-2 trace collection on GSM8K completed under NVIDIA
(`gate2-nvidia-full-concurrent-20260909`, 2,245 trajectories, zero synthetic outcomes,
intact chains, conserved costs, `analysis_grade=true`). Gate 2 is **INCONCLUSIVE**:
11 candidate actions exhibit marginal quality variation across the sample, but none
cleared the preregistered held-out test thresholds (>= 20 test rows, >= 5 per class)
to support a trained repairability probe. Per preregistration, no controller is built.
Calibration fits succeeded for `abstain` and `retrieve -> answer`, but induced choice
difference was 0.0000; C1 and C4 refused due to missing common action support and
served-path sample size; C2 depth-1 answer-preservation is confirmed (zero gain by
construction), but depth-2 paired contrasts lacked >= 20 complete test pairs.
See [Checkpoint 11 evidence](RESEARCH_STRATEGY.md).

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
  prior work. Yin & Zhang (*Knowledge-Based Systems*, Aug 2026, DOI 10.1016/j.knosys.2026.116685,
  "Failure-mode-aware uncertainty intervention routing for large language models") remains
  a confirmed closed-access journal publication without an open preprint, and is a disclosed
  overlap risk. Broad novelty, sequential routing priority, and calibrated stopping claims remain barred.
- **Gate 2: INCONCLUSIVE on live data.** Evaluated on 120 live GSM8K items under NVIDIA
  (`gate2-nvidia-full-concurrent-20260909`). While 11 candidate actions showed marginal quality
  variation across the sample, positive instances on the held-out test split were sparse (<5 per class),
  so probe fitting refused under the preregistered minimum test thresholds. Per preregistration,
  this precludes a GO verdict and no controller was built.
- **Calibration:** `abstain` (calibrated test ECE 0.0468, 95% CI [0.0177, 0.1172], n=26) and
  `retrieve -> answer` (calibrated test ECE 0.0220, 95% CI [0.0079, 0.1013], n=21) successfully fitted
  isotonic maps on disjoint calib; induced root choice change was 0.0000 (95% CI [0.0000, 0.0000], n=26).
  Other actions were refused due to incomplete test/calib coverage or lack of training variation.
- **C1:** Refused due to incomplete common action support and comparator coverage across the full action space.
- **C2:** Depth-1 answer preservation is established (mean delta 0.0000, 95% CI [0.0000, 0.0000] across all
  informational actions). Depth-2 continuation value is `C2_NOT_ESTABLISHED` because provider failures
  left complete test pairs below the required n=20 threshold (RESAMPLE n=15, SELF_CHECK n=16, RETRIEVE n=18).
- **C4:** Refused because served behavior propensities were absent on failed branches and served support was
  below the fitting threshold (~11 observations/action vs 20 required).
- **Gate 3 and learned controller:** Unbuilt and ungated; Gate 2 did not return GO.
- **Second-domain validity:** Unresolved. Groq preflight for 120-item `mixed` requires ~$29.29, exceeding
  the remaining approved cap ($1.997); NVIDIA trial listings carry no invoice guarantee. External budget
  approval is the explicit blocker.

The approved program cap is $2.00, with earlier listed-price spend accounted
against it. NVIDIA trial pricing is snapshotted at $0 listed price (not an invoice guarantee);
Groq listed-price spend remains $0.0026961. New live commands require an explicit remaining cap and fresh run id;
the pipeline never infers authorization from a configured credential:

```powershell
python scripts/run_gate2.py --n 120 --dataset gsm8k --depth 2 --policy balanced --small nim-nemotron-lightning-30b --big nim-nemotron-ultra-550b --live --max-usd <APPROVED_REMAINING_CAP> --run-id <FRESH_RUN_ID>
```

The command refuses before generation if its whole-run estimate exceeds the cap.
Per-action admission and actual settlement still apply. Listed-price token costs
are not provider invoices; an admission estimate is not an invoice guarantee.
No second-domain robustness result is established by the single-domain GSM8K run.

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
for Checkpoint 11 evidence and limitations, and [the novelty strategy](docs/research/novelty-strategy.md)
for the forbidden claims and candidate ablations.
