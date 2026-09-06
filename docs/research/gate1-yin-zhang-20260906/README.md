# Gate 1 access follow-up: Yin and Zhang

Date: 2026-09-06. DOI: [10.1016/j.knosys.2026.116685](https://doi.org/10.1016/j.knosys.2026.116685).
Result: **full text not obtained; NARROW remains provisional.** This is an
access checkpoint, not a completed novelty audit. No paid calls or messages to
authors were made. No controller or performance claim is licensed.

## New primary-source evidence

The search-indexed [publisher preview](https://www.sciencedirect.com/science/article/pii/S0950705126014115)
is readable despite direct requests returning 403. It describes an XGBoost
router using behavioral features to choose commit, majority-vote deliberation,
retrieval-augmented regeneration, or deferral. Its introduction distinguishes
intervention utility from answer confidence. The method preview identifies
labeling and training sections, but does not expose their procedures.

Thus learned intervention selection from response behavior is established prior
work. Whether this is calibrated, outcome-supervised, per-action marginal value
against STOP remains unknown. Classifying the best action is not sufficient
evidence of calibrated utility estimation. Missing details are not evidence
that a method lacks them.

No performance numbers from the preview are adopted: their uncertainty and
evaluation procedures could not be audited. Feature acquisition uses candidate
generation and probes, so a later comparison must align decision-time information
and charge its acquisition cost. This is a comparison requirement, not a finding
that the paper leaks labels or misaccounts costs.

## Access evidence and limits

- [source-check/receipt.json](source-check/receipt.json) records an actual
  unauthenticated Elsevier API response: HTTP 200, matching DOI, **metadata only**.
  The XML root's name includes `full-text-retrieval-response`, but there is no
  `originalText`. HTTP success and that name do not certify full text.
- [access-checks.json](access-checks.json) records URLs, timestamps, response
  hashes, status codes, and selected metadata. The explicit FULL API view
  requires authentication; the publisher PDF request returns 403. OpenAlex
  reports no repository full text; Semantic Scholar supplies no open PDF URL.
  These services can miss a copy; this does not prove none exists.
- Crossref confirms the identity and links the publisher's text-mining endpoints.
  Its issue date is October 2026; the publisher preview displays October 9.
  These are issue/cover dates, not proof of first online availability.
- The Crossref-linked public ORCID record lists this DOI, no researcher website,
  and no alternate link for this work. No contact identity was inferred from
  similarly named researchers.
- Browser discovery found no connected browser. No institutional session could
  be checked. ResearchGate's listing says full text is unavailable; its request
  button was not used.
- Searches included the exact title, DOI, title plus PDF/manuscript/calibration,
  and title restricted to arXiv, SSRN, and GitHub. No full manuscript surfaced.
  Search-indexed preview text is partial primary-source evidence, not a download
  or a substitute for reading the full methods.

Article bodies are not redistributed. The hashes identify bytes
received during these checks; a later response need not be byte-identical.

## Questions that require the manuscript

Read the complete paper, with particular attention to the method's formulation,
feature construction, intervention labeling, and router training, then record
page/section references for:

1. The supervised target: best-action class, absolute action utility, or marginal
   utility against commit/STOP; outcomes collected per item and unchosen actions.
2. Calibration: what quantity is calibrated, using which held-out data, and what
   calibration evidence is reported. A classifier probability alone is not a
   calibrated repair-value estimate.
3. Collection and identification: exhaustive versus randomized actions,
   propensities if relevant, missing/failed outcomes, and comparable parent states.
4. Feature availability: observations purchased before selection, outcome reuse,
   train/calibration/test separation, and acquisition costs.
5. Decision rule and constraints: stopping, uncertainty bounds, action costs,
   latency, and reliability; myopic versus sequential operation.

If calibrated per-action repair value learned from outcome data is already
present, apply the handoff's **STOP** rule. Do not add extra requirements to evade
it. Otherwise explain the remaining distinction and update the provisional
NARROW assessment. Nothing in this access attempt decides either outcome.

## Access request draft (not sent)

Dear authors / library access team,

Could you provide a lawful full-text copy or accepted manuscript of Shiyuan Yin
and Ruizhi Zhang, “Failure-mode-aware uncertainty intervention routing for large
language models,” Knowledge-Based Systems, DOI 10.1016/j.knosys.2026.116685?
I am auditing closely related work before proceeding with a research project.
The publisher preview is accessible, but I need the complete methods, including
the intervention labeling and router training sections, to make an accurate
comparison. Any supplementary methods or public code link would also help.

Thank you.

## Recheck command

From the repository root, this public metadata check costs no provider dollars
and refuses to reuse its output directory:

```powershell
python scripts/check_literature_source.py --doi 10.1016/j.knosys.2026.116685 --output-dir docs/research/gate1-yin-zhang-access-recheck-01
```

Exit 2 means no article-body candidate was retrieved; see the written receipt.
Even exit 0 requires manual completeness verification and reading. This command
cannot supply credentials or resolve a novelty gate. Use a new directory for
each later attempt. The practical unblock is an authorized full-text copy.

## Repository review findings before any live pilot

Reading the requested substrate exposed discrepancies with the handoff. These
are code-inspection findings, not new empirical results. They are recorded here
without changing the production router or beginning downstream experiments.

- `collect_fanout` awaits execution before `RunBudget.charge`; `would_exceed` has
  no caller. The collector passes an unchanged budget to each item, which creates
  a fresh accumulator. The stated before-action, run-wide cap is not implemented.
  Served collection also does not enforce the supplied cap.
- `assemble`, `repairability`, and `marginal_value_table` do not call
  `assert_estimable`; `gate2_report` does not reject synthetic runs before GO.
- `predict_repair` combines calibration and test rows and falls back to splitting
  sorted item IDs. Its comparison returns AUC point estimates without paired
  confidence intervals; `gate2_report` accepts any strict point improvement.
- Fan-out passes only `task_family` as prompt features. The prompt-only probe's
  other inputs default to zero, so that control cannot test the intended claim.
- RESAMPLE, SELF_CHECK, and RETRIEVE preserve `ctx.answer`. Forced STOP therefore
  makes their depth-1 answer-repair benefit zero by construction even live.
  Repairing via voting or retrieval-conditioned regeneration requires explicitly
  defined additional actions or continuation, with their full costs.

These issues require behavioral/refusal tests and correction before the old
live command can be considered valid, even after full-text access and budget
approval. They do not invalidate the mock run's stated INCONCLUSIVE outcome or
justify a claim about real model repairability. No historical trace is rewritten.
