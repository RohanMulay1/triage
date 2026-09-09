# Triage completion handoff — 2026-09-09 20:42 +05:30

## Repository and PR

- Upstream: https://github.com/ishaannk/triage
- Fork: https://github.com/RohanMulay1/triage
- Open PR: https://github.com/ishaannk/triage/pull/1
- Branch: `research/trace-substrate`
- Pushed source tip: `de761e3172626821ef70ba8a35dfa73d5e7aad55`
- Local checkout: `C:\Users\rohan\triage`

Do not print, commit, or rotate the NVIDIA/Groq credentials in ignored `.env`.
The approved program cap is $2.00. Previously recorded Groq listed-price usage is
$0.0026961; NVIDIA trial rows are snapshotted at zero listed price, which is not
an invoice guarantee.

## Active live run — leave it running

Detached Windows PID **23928** is collecting:

```powershell
python scripts/run_gate2.py --n 120 --dataset gsm8k --depth 2 --policy balanced --small nim-nemotron-lightning-30b --big nim-nemotron-ultra-550b --live --max-usd 1.99 --rps 0.6 --request-timeout 180 --item-concurrency 8 --run-id gate2-nvidia-full-concurrent-20260909
```

At 20:42 +05:30 it was alive with 16/120 complete items (416 trajectories), 24
items attempted, 369 request events, 340 returned attempts, 28 transient provider
failures, zero provider timeouts, and zero observed rate-limit events. The current
eight-item wave is in flight. Requests still start through one global 0.6 rps
limiter; at most eight independent items overlap. Complete batches append in item
order. Do not inspect answer labels until the completion receipt exists.

Monitor without disturbing it:

```powershell
Get-Process -Id 23928 -ErrorAction SilentlyContinue
Get-Content docs/research/checkpoint11/full-concurrent-stdout.txt -Tail 10
Get-Content docs/research/checkpoint11/full-concurrent-stderr.txt -Tail 30
Test-Path data/traces/gate2-nvidia-full-concurrent-20260909/collection-complete.json
Test-Path data/traces/gate2-nvidia-full-concurrent-20260909/report.json
```

Expected remaining wall time is roughly **4.5 hours**, based on about twenty
minutes per eight-item batch, plus final QA and documentation. Provider load can
change this. A GPT-5.6 Sol High subagent named `/root/long_run_monitor` was asked
to report only on terminal completion/failure, but the PID and files above are the
durable source of truth.

Two earlier attempts are intentionally preserved and are not evidence:

- `gate2-nvidia-full-20260909`: host sleep interrupted it after 11/120 items.
- `gate2-nvidia-full-awake-20260909`: deliberately stopped after 2/120 complete
  items because sequential runtime projected to about twenty hours.

Both validate with intact chains/cost conservation but
`collection_complete=false`, `analysis_grade=false`. Never merge their outcomes
into the active run or analyze them as a larger sample.

## What is implemented and pushed

Commits after the prior Checkpoint 10 tip:

- `f422e8f`: bounded 429/502/503/504 retries, no research mock fallback, shared
  request ledger, unknown-usage reservation, input-size refusal, raw attempt
  journals, correct C2 information-consuming VERIFY, calibration/C1/C4 honesty
  fixes, explicit Groq model pair/default override, CI and production checks.
- `f84852e`: successful-overrun and comparator-overrun evidence retention,
  independent alternate-provider pricing refusal, direct collector ledger,
  nonempty network errors, network retries, Windows host-awake guard, 180-second
  outer request timeout, collection plans/completion receipts, and incomplete-run
  analysis refusal.
- `0ca8a66`: preserved mock, NVIDIA smoke, interrupted live evidence, source
  hashes, clean-install receipts, novelty update, and replacement contract.
- `de761e3`: live zero-price-only item concurrency with task-local request IDs,
  deterministic batch append, global pacing, and timeout-inclusive operations.

The legacy router default is unchanged unless `TRIAGE_DEFAULT_MODEL` is explicitly
set. `TRIAGE_DEFAULT_MODEL=groq-gpt-oss-20b` selects a current explicit default and
escalates to `groq-gpt-oss-120b`; unknown overrides refuse. The trace executor
remains beside the production router. No deployment target exists in this repo,
so “production-ready” means tested application/configuration and research safety,
not a public deployment, capacity certification, or security audit.

## QA already completed

- Before the concurrency-only change: local full suite **374 passed**, pyflakes
  clean; fresh Python 3.11 venv from `requirements-dev.txt` also **374 passed**
  with six nonfatal framework/pydantic warnings and pyflakes clean.
- After concurrency: focused integration/safety suite **45 passed**; the separate
  concurrency/pacing suite **36 passed**; pyflakes clean.
- FastAPI cold-start `/health`, `/chat`, and `/telemetry`, legacy router tracing
  equivalence, finite-cap refusals, provider error paths, full mock pipeline,
  receipt immutability, and credential-prefix scans are covered.
- The final full suite has not been rerun after `de761e3`; do it once the live
  process exits and before the final commit. Test count must exceed 374.

Current uncommitted paths should consist only of active/interrupted live artifacts
and their redirected stdout/stderr. Preserve them; do not regenerate old fixtures
or any `data/*.json` benchmark receipt.

## Research gates and interpretation

- Gate 1 remains **provisionally NARROW**. Yin & Zhang, KBS 2026,
  DOI `10.1016/j.knosys.2026.116685`, is a disclosed overlap risk. Only the
  publisher preview was obtainable; the method remains unread. Broad novelty,
  first sequential routing, first utility selection, first calibrated stopping,
  and first abstention claims remain forbidden.
- Gate 2 is **undecided** until the active run completes and passes every gate.
- C1/calibration/C4 may refuse even on a complete run. C1 requires complete
  action fits while informational depth-1 gains are structural constants. With
  balanced root choices at n=120, C4 served-path support is likely below its fit
  minimum. Record refusals as design limitations; never invent values.
- C2 uses fixed VERIFY continuation and the matched direct-VERIFY control. Report
  RESAMPLE, SELF_CHECK, and RETRIEVE depth-1/depth-2 contrasts with paired CIs and
  n. An interval containing zero means C2 is not established.
- No controller is built in this run. Gate 3, second-domain robustness, production
  router overhead, and cross-model/corpus generalization remain unresolved.
- The 30B→550B replacement pair and GSM8K-only sample are confounds/limitations.
  Concurrency also changes provider load and is not production sequential latency.

## Simple completion PRD

**Goal:** turn the active immutable run into one honest Checkpoint 11 decision and
a reviewable PR state, without changing estimands or thresholds.

**Required behavior:**

1. Wait for PID 23928 to exit. If `collection-complete.json` or `report.json` is
   absent, classify the run as interrupted, run `store.validate_run`, report no
   Gate 2/C1/C2/C4 result, and do not reuse its run id.
2. If complete, independently run `store.validate_run`. Require 120 observed
   items, `collection_complete=true`, intact chains, conserved costs, zero
   validation errors, `force_mock=false`, zero synthetic outcomes, and
   `analysis_grade=true` before interpreting any estimate.
3. Read the consolidated report and verify support/positivity/missingness first.
   Report Gate 2 held-out response-only versus prompt-only paired AUC difference
   with 95% CI and n. Use the existing verdict exactly: GO only if its paired CI
   excludes zero; NARROW if evaluable without advantage; otherwise REFUSED or
   INCONCLUSIVE. Do not build a controller in Checkpoint 11.
4. Report calibration per supported action: train/calib/test counts, test ECE and
   CI, reliability bins, discrimination, and whether calibration changed choices.
   Keep calibration-map artifacts. A calibration change is not a performance win.
5. Report C1, C2, and C4 exactly as returned, including every refusal. Every
   numerical effect needs its paired/bootstrap 95% CI and n. “Different” is not
   “better”; no claim survives an interval containing zero.
6. Audit operational accounting: reported tokens, unknown-usage reservations,
   listed-price spend versus $1.99, wall clock, item-cluster p50/p95 latency CIs,
   429/5xx/timeouts, exhausted actions, blank responses, scoring failures, and
   no mock fallback. NVIDIA zero list price is not an invoice claim.
7. Update `README.md` and the Current-state section of `RESEARCH_STRATEGY.md`.
   Add one Checkpoint 11 block with results, failures, every gate status, the
   single-domain/model/concurrency/Yin & Zhang limitations, and one exact safe
   next command. State plainly what the run does not show.
8. Preserve all raw outputs, request journals, logs, manifests, plans, splits,
   completion receipt, analysis maps, consolidated report, and the two incomplete
   attempts. Add the active run/stdout/stderr through the existing `.gitignore`
   exceptions.
9. Run final QA: `python -m pytest -q`; `python -m pyflakes app/ scripts/ tests/`;
   `git diff --check`; validate every new run; verify no old `data/*.json` receipt
   changed; verify no credential prefix is staged; verify legacy-router equivalence.
10. Commit the final evidence/docs, push `research/trace-substrate` to remote
    `fork`, and confirm PR 1 shows the new commits. GitHub CLI normally needs
    `gh auth switch --user RohanMulay1` for the push, then switch back to
    `RohanMulayVigo`.

**Deliverable:** one final Checkpoint 11 evidence commit on PR 1, clean worktree,
all tests passing, and a report that separates established effects, negative
results, refusals, operational findings, and remaining external blockers.
