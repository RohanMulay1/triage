"""C2 paired depth ablation with a fixed, outcome-independent continuation.

Depth-2 total quality gain alone conflates information with its continuation.
Also report the matched continuation contrast to isolate the information term.
No oracle maximization over realized branch labels is permitted.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .analysis import bootstrap_ci, paired_bootstrap_diff
from .contract import ActionKind, OBSERVED_STATUSES
from .store import read_run, validate_run
from .support import SupportError, SupportThresholds, assert_estimable

INFO = (ActionKind.RESAMPLE, ActionKind.SELF_CHECK, ActionKind.RETRIEVE)
CONTINUATIONS = ("verify", "answer", "stronger_model", "tool")


def _observed(t):
    return t is not None and t.terminal.label is not None and all(
        s.outcome.status in OBSERVED_STATUSES for s in t.steps)


def c4_support_diagnostic(traces, thresholds):
    """Project the designated served path; refuse fitting if support is absent.

    This is a collection-protocol check, not evidence of policy superiority.
    """
    served = [t for t in traces if t.branch_id == "prefix" or
              (t.steps and t.steps[0].decision.rationale.get("served"))]
    result = {"not_evidence": True, "fitting_status": "NOT_RUN",
              "reason": "support check only; held-out policy experiment remains required"}
    for name, population in (("counterfactual", traces), ("served_projection", served)):
        try:
            support = assert_estimable(population, thresholds)
            result[name] = {"status": "SUPPORTED", "trajectories": len(population),
                            "actions_with_support": support["actions_with_support"]}
        except SupportError as exc:
            result[name] = {"status": "REFUSED", "support_error": str(exc)}
    return result


def c2_report(run_id, continuation="verify", *, diagnostic=False, seed=0):
    if continuation not in CONTINUATIONS:
        raise ValueError("continuation must be fixed before observing outcomes")
    validation = validate_run(run_id)
    meta = {"run_id": run_id, "analysis_grade": validation["analysis_grade"],
            "not_evidence": not validation["analysis_grade"], "validation": validation,
            "continuation": continuation, "continuation_selection": "fixed_not_oracle",
            "estimand": "quality_only; costs reported separately, not equal-budget superiority"}
    if diagnostic and not validation["force_mock"]:
        return {**meta, "verdict": "REFUSED", "reason": "diagnostic flag requires forced mock run"}
    thresholds = SupportThresholds(allow_synthetic=diagnostic)
    meta["support_thresholds"] = asdict(thresholds)
    if not validation["analysis_grade"] and not (
            diagnostic and validation["force_mock"] and validation["chain_ok"]
            and validation["cost_conservation_ok"] and not validation["errors"]):
        return {**meta, "verdict": "REFUSED", "reason": "non-analysis-grade input"}
    traces = read_run(run_id)
    try:
        assert_estimable(traces, thresholds)
    except SupportError as exc:
        return {**meta, "verdict": "REFUSED", "support_error": str(exc)}
    # Evidence scores only held-out test items; mock diagnostics explicitly use all.
    prefixes = [t for t in traces if t.branch_id == "prefix" and
                (diagnostic or t.split == "test")]
    results = {}
    positive = []
    for kind in INFO:
        pairs, missing = [], []
        for prefix in prefixes:
            item = [t for t in traces if t.item_id == prefix.item_id]
            root = next((t for t in item if t.branch_id == "stop"), None)
            parent = next((t for t in item if t.decision_depth == 1 and t.branch_id != "prefix"
                           and t.steps[0].decision.chosen.kind == kind), None)
            direct = next((t for t in item if t.decision_depth == 1 and t.branch_id != "prefix"
                           and t.steps[0].decision.chosen.kind.value == continuation), None)
            child = next((t for t in item if t.decision_depth == 2 and parent is not None
                          and t.baseline_branch_id == parent.branch_id
                          and t.steps[0].decision.chosen.kind.value == continuation), None)
            if not all(_observed(t) for t in (root, parent, direct, child)):
                missing.append(prefix.item_id)
                continue
            if (parent.root_state_id != root.root_state_id or
                    direct.root_state_id != root.root_state_id or
                    child.parent_trajectory_id != parent.trajectory_id or
                    child.root_state_id != parent.steps[0].outcome.result_state_id):
                return {**meta, "verdict": "REFUSED", "reason": "unmatched branch parents"}
            if parent.terminal.served_answer_hash != root.terminal.served_answer_hash:
                return {**meta, "verdict": "REFUSED", "reason": "informational action changed answer"}
            d1 = parent.terminal.label - root.terminal.label
            if d1 != 0:
                return {**meta, "verdict": "REFUSED", "reason": "same answer has inconsistent labels"}
            pairs.append({"item_id": prefix.item_id, "depth1": d1,
                          "depth2": child.terminal.label - root.terminal.label,
                          "direct": direct.terminal.label - root.terminal.label,
                          "extra_usd": child.total_cost.est_cost_usd,
                          "extra_calls": child.total_cost.llm_calls,
                          "information_usd": parent.total_cost.est_cost_usd})
        if not pairs:
            results[kind.value] = {"status": "REFUSED_INCOMPLETE_PAIRS", "n": len(pairs),
                                   "missing_items": missing}
            continue
        d1, d2, direct = ([p[k] for p in pairs] for k in ("depth1", "depth2", "direct"))
        interval = paired_bootstrap_diff(d2, d1, seed=seed)
        matched = paired_bootstrap_diff(d2, direct, seed=seed)
        adjusted = paired_bootstrap_diff(d2, direct, seed=seed, confidence=1-.05/len(INFO))
        enough = len(pairs) >= 20
        established = enough and interval["lo"] > 0 and adjusted["lo"] > 0
        positive.append(established)
        results[kind.value] = {"status": "DIAGNOSTIC" if diagnostic else
                              ("EVALUABLE" if enough else "INSUFFICIENT_ROWS"),
                              "not_evidence": meta["not_evidence"], "n": len(pairs),
                              "depth1": bootstrap_ci(d1, seed=seed),
                              "depth2": bootstrap_ci(d2, seed=seed),
                              "paired_depth2_minus_depth1_ci95": interval,
                              "matched_information_ci95": matched,
                              "matched_selection_interval": adjusted,
                              "extra_continuation_usd": bootstrap_ci([p["extra_usd"] for p in pairs]),
                              "extra_continuation_calls": bootstrap_ci([p["extra_calls"] for p in pairs]),
                              "information_usd": bootstrap_ci([p["information_usd"] for p in pairs]),
                              "pairs": pairs, "missing_items": missing}
    established = not diagnostic and validation["analysis_grade"] and any(positive)
    return {**meta, "verdict": "C2_SIGNAL" if established else "C2_NOT_ESTABLISHED",
            "per_action": results, "next_candidate": "C2_REPLICATION" if established else "C4",
            "c4": None if established else c4_support_diagnostic(traces, thresholds),
            "limitations": ["mock contrasts are harness diagnostics only",
                            "one stochastic execution per branch; no equal-cost policy comparison",
                            "response state and continuation mechanisms differ; no novelty demonstrated"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--continuation", choices=CONTINUATIONS, default="verify")
    ap.add_argument("--diagnostic-mock", action="store_true")
    args = ap.parse_args()
    print(json.dumps(c2_report(args.run, args.continuation, diagnostic=args.diagnostic_mock), indent=2))


if __name__ == "__main__":
    main()
