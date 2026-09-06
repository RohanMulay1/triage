"""Action support and missingness diagnostics.

Off-policy value estimation needs every action it wants to score to have been
*taken* sometimes in states where it was *feasible*. An action that was always
available and never chosen has zero support: no estimator can say what it would
have done, and any model that appears to predict its value is extrapolating.

Run over traces from the deterministic heuristic router, this reports coverage
near 1.0 for whatever the fixed program chose and 0.0 for every other feasible
action. That number is the argument for randomized branch collection: it makes
the gap a measurement rather than an assertion.

    python -m app.trace.support --run <run_id>
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from .contract import Trajectory
from .store import read_run, validate_run


def action_support(trajectories: list[Trajectory]) -> dict[str, Any]:
    """Per-action feasibility, selection counts, coverage and minimum propensity."""
    feasible: dict[str, int] = {}
    taken: dict[str, int] = {}
    min_prop: dict[str, float] = {}
    n_decisions = 0

    for t in trajectories:
        for step in t.steps:
            n_decisions += 1
            d = step.decision
            for a in d.feasible:
                feasible[a.key] = feasible.get(a.key, 0) + 1
            k = d.chosen.key
            taken[k] = taken.get(k, 0) + 1
            min_prop[k] = min(min_prop.get(k, 1.0), d.propensity)

    per_action = {}
    for key in sorted(feasible):
        n_f = feasible[key]
        n_t = taken.get(key, 0)
        per_action[key] = {
            "n_feasible": n_f,
            "n_taken": n_t,
            "coverage": round(n_t / n_f, 4) if n_f else 0.0,
            "min_propensity": round(min_prop[key], 4) if key in min_prop else None,
        }

    unsupported = [k for k, v in per_action.items() if v["n_taken"] == 0]
    return {
        "trajectories": len(trajectories),
        "decisions": n_decisions,
        "per_action": per_action,
        "unsupported_actions": unsupported,
        "actions_with_support": len(per_action) - len(unsupported),
        # Off-policy learning over this set is only defensible when every action
        # the policy may choose has been observed in states where it was feasible.
        "off_policy_ready": bool(per_action) and not unsupported,
    }


def report(run_id: str) -> dict[str, Any]:
    trajectories = read_run(run_id)
    return {
        "validation": validate_run(run_id),
        "support": action_support(trajectories),
    }


def _main() -> None:
    ap = argparse.ArgumentParser(description="Action support diagnostics for a trace run")
    ap.add_argument("--run", required=True, help="run id under the trace directory")
    args = ap.parse_args()
    out = report(args.run)
    print(json.dumps(out, indent=2))

    sup = out["support"]
    val = out["validation"]
    print("\n--- summary ---")
    print(f"trajectories        : {sup['trajectories']}  decisions: {sup['decisions']}")
    print(f"actions with support: {sup['actions_with_support']}/{len(sup['per_action'])}")
    print(f"unsupported actions : {sup['unsupported_actions']}")
    print(f"off-policy ready    : {sup['off_policy_ready']}")
    print(f"analysis grade      : {val['analysis_grade']}"
          f" (synthetic trajectories: {val['synthetic_trajectories']})")


if __name__ == "__main__":  # pragma: no cover - CLI
    _main()
