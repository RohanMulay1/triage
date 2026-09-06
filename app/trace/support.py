"""Action support, positivity, and the refusal gate for off-policy estimates.

Off-policy value estimation needs every action it wants to score to have been
*executed* in states where it was *feasible*, with a known non-vanishing
probability. An action that was always available and never taken has zero
support: no estimator can say what it would have done, and any model that appears
to predict its value is extrapolating.

This module measures that rather than asserting it, and `assert_estimable`
refuses to let an estimate run when the measurement fails. The refusal is the
point. A support report that is only advisory gets ignored precisely when it
matters.

    python -m app.trace.support --run <run_id>
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Optional

from .contract import OBSERVED_STATUSES, OutcomeStatus, Trajectory
from .store import read_run, validate_run


class SupportError(RuntimeError):
    """Support diagnostics failed; an off-policy estimate would be extrapolation."""


@dataclass(frozen=True)
class SupportThresholds:
    """Gates an estimate must clear. Deliberately explicit, never implicit."""

    min_coverage: float = 0.0        # fraction of feasible occasions the action was taken
    min_observations: int = 1        # executed outcomes per action
    min_propensity: float = 1e-3     # positivity floor
    min_ess_ratio: float = 0.1       # effective sample size / observations
    allow_synthetic: bool = False    # permit mock-provider outcomes in the estimate


def action_support(trajectories: list[Trajectory]) -> dict[str, Any]:
    """Per-action feasibility, execution counts, propensity spread and ESS."""
    feasible: dict[str, int] = {}
    taken: dict[str, int] = {}
    observed: dict[str, int] = {}
    props: dict[str, list[float]] = {}
    status_counts: dict[str, dict[str, int]] = {}
    providers: dict[str, dict[str, int]] = {}
    synthetic: dict[str, int] = {}
    n_decisions = 0
    policies: dict[str, int] = {}
    definitions: set[str] = set()

    for t in trajectories:
        for step in t.steps:
            n_decisions += 1
            d, o = step.decision, step.outcome
            policies[d.policy_id] = policies.get(d.policy_id, 0) + 1
            definitions.add(d.feasible_set_definition)
            for a in d.feasible:
                feasible[a.key] = feasible.get(a.key, 0) + 1
            k = d.chosen.key
            taken[k] = taken.get(k, 0) + 1
            props.setdefault(k, []).append(d.propensity)
            status_counts.setdefault(k, {})
            status_counts[k][o.status.value] = status_counts[k].get(o.status.value, 0) + 1
            providers.setdefault(k, {})
            providers[k][o.provider_label] = providers[k].get(o.provider_label, 0) + 1
            if o.status in OBSERVED_STATUSES:
                observed[k] = observed.get(k, 0) + 1
            if o.is_synthetic:
                synthetic[k] = synthetic.get(k, 0) + 1

    per_action: dict[str, Any] = {}
    for key in sorted(set(feasible) | set(taken)):
        n_f, n_t = feasible.get(key, 0), taken.get(key, 0)
        p = props.get(key, [])
        weights = [1.0 / x for x in p if x > 0]
        ess = (sum(weights) ** 2 / sum(w * w for w in weights)) if weights else 0.0
        per_action[key] = {
            "n_feasible": n_f,
            "n_taken": n_t,
            "n_observed": observed.get(key, 0),
            "coverage": round(n_t / n_f, 4) if n_f else 0.0,
            "min_propensity": round(min(p), 6) if p else None,
            "mean_propensity": round(sum(p) / len(p), 6) if p else None,
            "effective_sample_size": round(ess, 3),
            "ess_ratio": round(ess / n_t, 4) if n_t else 0.0,
            "status_counts": status_counts.get(key, {}),
            "provider_counts": providers.get(key, {}),
            "synthetic_outcomes": synthetic.get(key, 0),
        }

    unsupported = [k for k, v in per_action.items() if v["n_observed"] == 0]
    return {
        "trajectories": len(trajectories),
        "decisions": n_decisions,
        "behavior_policies": policies,
        "feasible_set_definitions": sorted(definitions),
        "per_action": per_action,
        "unsupported_actions": unsupported,
        "actions_with_support": len(per_action) - len(unsupported),
        "off_policy_ready": bool(per_action) and not unsupported,
        # Mixing action-space definitions makes coverage denominators
        # incomparable, so it is a hard defect rather than a warning.
        "mixed_feasible_set_definitions": len(definitions) > 1,
    }


def positivity_violations(
    support: dict[str, Any], thresholds: Optional[SupportThresholds] = None
) -> list[str]:
    """Every reason this run cannot carry an off-policy estimate."""
    th = thresholds or SupportThresholds()
    out: list[str] = []
    if support["mixed_feasible_set_definitions"]:
        out.append(
            "run mixes feasible-set definitions "
            f"{support['feasible_set_definitions']}; coverage denominators are not comparable"
        )
    for key, s in support["per_action"].items():
        if s["n_observed"] < th.min_observations:
            out.append(
                f"{key}: {s['n_observed']} executed outcome(s) in {s['n_feasible']} "
                f"feasible occasions (need >= {th.min_observations})"
            )
            continue
        if s["coverage"] < th.min_coverage:
            out.append(f"{key}: coverage {s['coverage']} < {th.min_coverage}")
        mp = s["min_propensity"]
        if mp is not None and mp < th.min_propensity:
            out.append(f"{key}: min propensity {mp} < positivity floor {th.min_propensity}")
        if s["ess_ratio"] < th.min_ess_ratio:
            out.append(
                f"{key}: effective sample size ratio {s['ess_ratio']} < {th.min_ess_ratio} "
                "(importance weights are dominated by a few rows)"
            )
        if s["synthetic_outcomes"] and not th.allow_synthetic:
            out.append(
                f"{key}: {s['synthetic_outcomes']} mock-provider outcome(s); "
                "synthetic text is not evidence about model behaviour"
            )
    return out


def assert_estimable(
    trajectories: list[Trajectory], thresholds: Optional[SupportThresholds] = None
) -> dict[str, Any]:
    """Return the support report, or raise if an estimate would be inadmissible.

    Every off-policy or causal estimate in this project must route through here.
    A diagnostic that callers may ignore is a diagnostic that gets ignored.
    """
    support = action_support(trajectories)
    violations = positivity_violations(support, thresholds)
    if violations:
        raise SupportError(
            "support diagnostics failed; refusing to produce an off-policy estimate:\n  - "
            + "\n  - ".join(violations[:20])
        )
    return support


def missingness(trajectories: list[Trajectory]) -> dict[str, int]:
    """Counts by outcome status across the whole run."""
    counts: dict[str, int] = {s.value: 0 for s in OutcomeStatus}
    for t in trajectories:
        for step in t.steps:
            counts[step.outcome.status.value] += 1
    return counts


def marginal_value_table(trajectories: list[Trajectory]) -> dict[str, Any]:
    """Δ(a) = label(branch a) − label(STOP branch), per item, from fan-out runs.

    Only defined where an item has a labelled STOP branch to difference against;
    items without one are reported as skipped rather than silently dropped.
    """
    by_item: dict[str, dict[str, float]] = {}
    for t in trajectories:
        if t.terminal.label is None or t.branch_id == "prefix":
            continue
        by_item.setdefault(t.item_id, {})[t.branch_id] = t.terminal.label

    stop_key = "stop"
    deltas: dict[str, list[float]] = {}
    skipped = 0
    for item, branches in by_item.items():
        if stop_key not in branches:
            skipped += 1
            continue
        base = branches[stop_key]
        for key, label in branches.items():
            if key == stop_key:
                continue
            deltas.setdefault(key, []).append(label - base)

    summary = {}
    for key, values in sorted(deltas.items()):
        n = len(values)
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
        se = math.sqrt(var / n) if n else 0.0
        summary[key] = {
            "n": n,
            "mean_delta": round(mean, 4),
            "se": round(se, 4),
            "ci95": [round(mean - 1.96 * se, 4), round(mean + 1.96 * se, 4)],
            "n_positive": sum(1 for v in values if v > 0),
            "n_negative": sum(1 for v in values if v < 0),
            "n_zero": sum(1 for v in values if v == 0),
        }
    return {"items_with_baseline": len(by_item) - skipped,
            "items_skipped_no_stop_branch": skipped,
            "per_action": summary}


def report(run_id: str) -> dict[str, Any]:
    trajectories = read_run(run_id)
    return {
        "validation": validate_run(run_id),
        "support": action_support(trajectories),
        "missingness": missingness(trajectories),
        "positivity_violations": positivity_violations(action_support(trajectories)),
        "marginal_value": marginal_value_table(trajectories),
    }


def _main() -> None:
    ap = argparse.ArgumentParser(description="Action support diagnostics for a trace run")
    ap.add_argument("--run", required=True, help="run id under the trace directory")
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = ap.parse_args()
    out = report(args.run)
    if args.json:
        print(json.dumps(out, indent=2))

    sup, val = out["support"], out["validation"]
    print(f"run                 : {args.run}")
    print(f"trajectories        : {sup['trajectories']}  decisions: {sup['decisions']}")
    print(f"behaviour policies  : {sup['behavior_policies']}")
    print(f"actions with support: {sup['actions_with_support']}/{len(sup['per_action'])}")
    print("\nper action:")
    for key, s in sup["per_action"].items():
        print(f"  {key:44s} taken={s['n_taken']:4d} observed={s['n_observed']:4d} "
              f"feasible={s['n_feasible']:4d} cov={s['coverage']:<7} "
              f"minP={s['min_propensity']} ESS={s['effective_sample_size']}")
    print(f"\nmissingness         : {out['missingness']}")
    print(f"unsupported actions : {sup['unsupported_actions']}")
    print(f"off-policy ready    : {sup['off_policy_ready']}")
    print(f"analysis grade      : {val['analysis_grade']}")
    violations = out["positivity_violations"]
    print(f"\npositivity violations ({len(violations)}):")
    for v in violations[:15]:
        print(f"  - {v}")

    mv = out["marginal_value"]
    if mv["per_action"]:
        print(f"\nmarginal value vs STOP (items with baseline: {mv['items_with_baseline']}):")
        for key, s in mv["per_action"].items():
            print(f"  {key:44s} n={s['n']:4d} mean delta={s['mean_delta']:+.4f} "
                  f"95% CI [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
                  f"(+{s['n_positive']}/-{s['n_negative']}/={s['n_zero']})")


if __name__ == "__main__":  # pragma: no cover - CLI
    _main()
