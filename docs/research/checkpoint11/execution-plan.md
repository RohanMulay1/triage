# Checkpoint 11 execution contract

Written before the new live outcomes. Continue the existing depth-2 protocol;
do not tune thresholds or select datasets after seeing results.

- Dataset: 120 GSM8K items, seed 0; preserve the selected items and prompts.
- NVIDIA pair: explicitly identified Lightning 30B -> Ultra 550B replacements.
  Run a three-item smoke first, then the full run if structurally analysis-grade.
  Retired Llama aliases are not silently reassigned. Failures remain unlabelled.
- Remaining approved cap: $1.9973039 after $0.0026961 previously recorded
  listed-price usage. NVIDIA snapshot prices are zero. No invoice claim.
- Retry 502/503/504/429 at most three attempts, honour Retry-After, share pacing
  at 0.6 requests/second. Each attempt reserves its own request cost; unknown
  usage retains the reservation. Never substitute mock in research.
- Groq fallback is a separate model-pair run. Its optional enforced 8192-byte
  request ceiling produces a $2.899008 full-run planning bound with the existing
  2048-token completion cap, above the approval. Do not silently shorten outputs
  or reduce n to call a partial experiment complete. Groq also publishes daily
  and token quotas; response headers will be retained for operational diagnosis.
- Scoring failures cannot select heuristic actions. Under balanced collection,
  a failed auxiliary comparator assessment is recorded but does not discard a
  valid draft or prevent independent fan-out observations.
- Executor v2 makes VERIFY consume prior resample/self-check observations marked
  fallible; root VERIFY and legacy production routing remain unchanged.
- Gate 2 retains class/sample thresholds and paired multiplicity-aware interval.
  Calibration uses separate train/calib/test roles. C1 requires complete fits;
  C2 uses fixed VERIFY continuation and matched direct-VERIFY controls; C4 uses
  true selected-action propensities, not exhaustive inclusion probabilities.
- Unsupported fits and missing outcomes refuse. Underpowered evidence remains
  inconclusive. No controller, superiority claim, or broad novelty claim.
- One domain, replacement model pair, conditional-on-observed-outcome estimates,
  fixed-fit bootstrap uncertainty, and unread Yin & Zhang remain limitations.

Groq limits reference: https://console.groq.com/docs/rate-limits

## Pre-evaluation design limitations found in integration

The full run does not guarantee all ablations are statistically estimable. With
eight root actions and balanced selection, 120 complete items yield about fifteen
served observations per action across all splits, fewer than the twenty training
observations required by `fit_gain`. C4 therefore cannot be established by this
served projection at the planned sample size. Its refusal is a design limitation,
not evidence of zero disagreement. A larger prespecified collection is required.

C1 requires fits for the complete feasible set, while the existing estimator
refuses constant training targets. Root informational actions have exactly
constant zero quality increments by construction. The experiment therefore may
have no complete supported policy comparison even when answer-changing actions
can be calibrated. Do not bypass this refusal with invented/default gains. A
separate explicit treatment of structural constants would require a method change
and its tests before a new C1 experiment; it is not retrospectively introduced.

Runtime integration also bounds a retry wait at sixty seconds. A longer provider
Retry-After causes a recorded refusal and a clean stop, never an earlier retry.
This is a runtime refusal, not evidence that the action has no value. Malformed
nonfinite retry headers cannot create an infinite sleep.
