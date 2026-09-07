# Checkpoint 7: comparator and information-action ablation

The full [base paper](https://arxiv.org/html/2603.19896v1) and its released
`policies.py` were read. `source-record.json` pins the code revision and hash.
The implementation here is an equation-level Triage adaptation, not a reproduction
of the original experiment. The paper presents heterogeneous selection, while
released `UtilityPolicy` implements repeated search versus termination.
Source defaults are cost/uncertainty/redundancy weights 0.3/0.2/0.2.

Triage estimates gain and uncertainty for every feasible canonical action key in
one JSON-producing LLM call. Scores are uncalibrated and finite values are clipped
to [0,1]. This extends the source's search-only scoring prompt. The normalized
step proxy is (prior interventions + 1)/3; STOP costs zero steps. Redundancy is
exact action-key repetition, replacing the source's exact search-query repetition.
STOP participates in the requested argmax; deterministic ties favor STOP, then
canonical key order. These action mappings, prompt, step accounting and stopping
semantics must be held fixed across any future gain-estimator substitution.
They do not establish equivalence with the published experimental policy.

The scorer is a forced assessment in the shared prefix. Admission precedes its
provider call and actual reported tokens/cost/latency are recorded once. Malformed
JSON, incomplete/nonfinite scores, and synthetic fallback never receive default
gains. Unknown usage stops the entire collection. Fan-out inclusion propensity
remains one; the deterministic comparator's choice is conditional on recorded LLM
scores, not a claim about its marginal probability over stochastic LLM outputs.
The collector designates a root served branch; it does not evaluate a sequential
comparator rollout. No policy performance comparison has been performed.

The ordinary mock provider cannot emit score JSON. Its fresh comparator run
therefore contains only unlabelled prefixes with recorded scoring failures.
Scripted-client tests check successful scoring, utility selection, overhead,
finite-cap admission, stale-state refusal, fallback, and unknown usage. Those
scripted outputs are unit-test fixtures, not real LLM self-estimates or evidence.

## C2 protocol and result

VERIFY was fixed as the continuation before inspecting outcomes. For the same
item, let y0 be root STOP, ya informational STOP, yab information then VERIFY,
and yb root VERIFY. Report:

- Depth 1: ya - y0. Answer preservation requires this to be exactly zero.
- Depth 2: yab - y0. Its paired difference from depth 1 is continuation value.
- Matched information contrast: yab - yb. This controls for VERIFY alone.

There is no maximization over realized continuation outcomes. Paired percentile
bootstrap resamples item pairs. Real evidence uses test rows, at least 20 complete
pairs, and a multiplicity-adjusted positive matched interval as well as a positive
depth contrast. Missing outcomes or inconsistent parents refuse; synthetic runs
refuse unless explicitly requested as forced-mock diagnostics. Costs are reported
separately; this is a quality ablation, not an equal-budget policy comparison.

For each of RESAMPLE, SELF_CHECK and RETRIEVE, the mock paired depth difference
is 0.000, 95% CI [0.000, 0.000], n=8. The matched information contrast has the same
point and interval, n=8 each. The intervals include zero. Depth 2 did not produce
a nonzero effect in this harness. C2_NOT_ESTABLISHED is the recorded verdict.
These figures are synthetic diagnostics and cannot establish real-model benefit
or its absence. One stochastic execution per branch also cannot isolate provider
sampling noise; replication and a frozen external corpus remain necessary.

## Move to C4

The C4 support diagnostic projects the designated served branches from the same
fan-out. `assert_estimable` refuses that projection because calculator outcomes
are unsupported, while exhaustive traces pass under explicit synthetic-diagnostic
thresholds. Neither population is scientific evidence. No fitted policy comparison
was run: unsupported served-path estimates must not be fabricated to produce a
contrast. C4 is now the next candidate, not an established artifact contribution.
Its next experiment needs a prespecified estimand, adequately supported behavior
policies, disjoint training/calibration/test roles, and held-out policy evaluation.
A refusal-versus-supported diagnostic alone does not establish policy superiority.

Gate 1 remains provisionally NARROW. Yin & Zhang, Knowledge-Based Systems,
DOI [10.1016/j.knosys.2026.116685](https://doi.org/10.1016/j.knosys.2026.116685),
remains a disclosed overlap risk: its method section has not been obtained.
No novelty, calibrated stopping, sequential-routing priority, or performance claim
is made. No live budget is approved and no paid provider call was made.

Exact next command:

```bash
python -m app.trace.information_value --run checkpoint7-c2-mock --diagnostic-mock --continuation verify
```
