"""Signal Gate analysis: does observed response state predict repairability?

Decision Gate 2 asks two questions that are easy to conflate and must not be:

  1. Does the observed state predict whether the answer is **wrong**?
  2. Does it predict whether a given action would **fix** it?

Correctness and repairability are different latent variables. A cascade that
answers (1) and assumes (2) follows is the failure this project already recorded
on GSM8K: an answer estimated as risky while no available intervention has
positive expected value.

Everything here reports a confidence interval, and every model comparison is fit
on one split and scored on another. Where the data cannot support a claim, the
report says so instead of returning a number.

    python -m app.trace.analysis --run <run_id>
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ..router.abstain import abstain_risk
from ..config import load_router_config
from .contract import OutcomeStatus, Trajectory
from .store import read_run

STOP_BRANCH = "stop"

#: A probe scored on a handful of rows will happily report AUC 1.0, which reads
#: like a finding and is noise. These are the minimum sizes below which
#: `predict_repair` refuses to return a number at all.
MIN_TEST_ROWS = 20
MIN_TEST_PER_CLASS = 5

#: Feature sets the gate compares. If response-only does not beat prompt-only,
#: observing the answer bought nothing and the sequential story collapses.
FEATURE_SETS = {
    "prompt_only": ("message_chars", "predicted_difficulty", "approx_prompt_tokens"),
    "scalar_uncertainty": ("uncertainty",),
    "triage_risk": ("triage_risk",),
    "response_only": ("uncertainty", "instability", "contradiction",
                      "retrieval_disagreement", "evidence_sufficiency",
                      "answer_len", "answer_has_digit"),
    "prompt_plus_response": ("message_chars", "predicted_difficulty", "uncertainty",
                             "instability", "contradiction", "retrieval_disagreement",
                             "evidence_sufficiency", "answer_len", "answer_has_digit"),
}


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """Mann-Whitney AUC. None when one class is absent, rather than a fake 0.5."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], Optional[float]] = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Optional[float]]:
    """Percentile bootstrap. Returns nulls rather than a point estimate on no data."""
    values = list(values)
    if not values:
        return {"point": None, "lo": None, "hi": None, "n": 0}
    statistic = statistic or (lambda v: sum(v) / len(v))
    rng = random.Random(seed)
    point = statistic(values)
    draws = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        stat = statistic(sample)
        if stat is not None:
            draws.append(stat)
    draws.sort()
    if not draws:
        return {"point": point, "lo": None, "hi": None, "n": len(values)}
    lo = draws[int(0.025 * len(draws))]
    hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
    return {"point": round(point, 4) if point is not None else None,
            "lo": round(lo, 4), "hi": round(hi, 4), "n": len(values)}


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float],
                          n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """CI on the paired difference a - b, which is what a fair comparison needs."""
    if len(a) != len(b) or not a:
        return {"point": None, "lo": None, "hi": None, "n": 0}
    diffs = [x - y for x, y in zip(a, b)]
    out = bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
    out["excludes_zero"] = (out["lo"] is not None and
                            (out["lo"] > 0 or out["hi"] < 0))
    return out


class LogisticProbe:
    """A small L2 logistic regression, fit by gradient descent.

    Deliberately minimal. The gate question is whether a feature set carries
    signal at all, and a heavier model would make a negative result harder to
    trust rather than easier.
    """

    def __init__(self, l2: float = 1.0, steps: int = 800, lr: float = 0.2) -> None:
        self.l2, self.steps, self.lr = l2, steps, lr
        self.w: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticProbe":
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        self.sigma[self.sigma == 0] = 1.0
        Z = np.hstack([(X - self.mu) / self.sigma, np.ones((len(X), 1))])
        self.w = np.zeros(Z.shape[1])
        for _ in range(self.steps):
            p = 1.0 / (1.0 + np.exp(-Z @ self.w))
            grad = Z.T @ (p - y) / len(y)
            grad[:-1] += self.l2 * self.w[:-1] / len(y)
            self.w -= self.lr * grad
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = np.hstack([(X - self.mu) / self.sigma, np.ones((len(X), 1))])
        return 1.0 / (1.0 + np.exp(-Z @ self.w))


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def branch_features(prefix: Trajectory, cfg: dict) -> dict[str, float]:
    """State features observed at the fan-out branch point, before any action."""
    state = prefix.states[-1]
    s = state.signals
    risk, _ = abstain_risk(s, cfg["abstain"])
    pf = state.prompt_features or {}
    answer = state.answer or ""
    return {
        "uncertainty": s.uncertainty,
        "instability": s.instability,
        "contradiction": s.contradiction,
        "retrieval_disagreement": s.retrieval_disagreement,
        "evidence_sufficiency": s.evidence_sufficiency,
        "triage_risk": risk,
        "answer_len": float(len(answer)),
        "answer_has_digit": float(any(c.isdigit() for c in answer)),
        "message_chars": float(pf.get("message_chars") or 0.0),
        "predicted_difficulty": float(pf.get("predicted_difficulty") or 0.0),
        "approx_prompt_tokens": float(pf.get("approx_prompt_tokens") or 0.0),
    }


def assemble(run_id: str) -> list[dict[str, Any]]:
    """One row per item: branch-point features, per-action labels and deltas."""
    cfg = load_router_config()
    trajectories = read_run(run_id)
    prefixes = {t.item_id: t for t in trajectories if t.branch_id == "prefix"}
    rows: list[dict[str, Any]] = []

    for item_id, prefix in prefixes.items():
        branches = {t.branch_id: t for t in trajectories
                    if t.item_id == item_id and t.branch_id != "prefix"}
        stop = branches.get(STOP_BRANCH)
        if stop is None or stop.terminal.label is None:
            continue
        labels, deltas, costs = {}, {}, {}
        for key, t in branches.items():
            if t.terminal.label is None:
                continue
            if not t.steps or t.steps[0].outcome.status not in (
                    OutcomeStatus.CHOSEN, OutcomeStatus.COUNTERFACTUAL):
                continue
            labels[key] = t.terminal.label
            deltas[key] = t.terminal.label - stop.terminal.label
            costs[key] = t.steps[0].outcome.cost.est_cost_usd
        rows.append({
            "item_id": item_id,
            "split": prefix.split,
            "task_family": (prefix.states[-1].prompt_features or {}).get("task_family",
                                                                         "unknown"),
            "features": branch_features(prefix, cfg),
            "stop_label": stop.terminal.label,
            "labels": labels,
            "deltas": deltas,
            "action_cost_usd": costs,
        })
    return rows


# --------------------------------------------------------------------------- #
# Gate analyses
# --------------------------------------------------------------------------- #
def correctness_signal(rows: list[dict], feature: str = "triage_risk") -> dict[str, Any]:
    """Question 1: does the state predict that the served answer is wrong?"""
    scores = [r["features"][feature] for r in rows]
    wrong = [int(r["stop_label"] == 0) for r in rows]
    value = auc(scores, wrong)
    paired = list(zip(scores, wrong))
    ci = bootstrap_ci(
        list(range(len(paired))),
        statistic=lambda idx: auc([paired[i][0] for i in idx], [paired[i][1] for i in idx]),
    )
    return {"feature": feature, "auc_wrong": value, "ci": ci,
            "n": len(rows), "n_wrong": sum(wrong)}


def repairability(rows: list[dict]) -> dict[str, Any]:
    """Question 2: per action, is the marginal benefit over STOP nonzero?"""
    per_action: dict[str, Any] = {}
    keys = sorted({k for r in rows for k in r["deltas"]})
    for key in keys:
        if key == STOP_BRANCH:
            continue
        deltas = [r["deltas"][key] for r in rows if key in r["deltas"]]
        if not deltas:
            continue
        ci = bootstrap_ci(deltas)
        per_action[key] = {
            **ci,
            "mean_delta": ci["point"],
            "n_positive": sum(1 for d in deltas if d > 0),
            "n_negative": sum(1 for d in deltas if d < 0),
            "n_zero": sum(1 for d in deltas if d == 0),
            "negative_rate": round(sum(1 for d in deltas if d < 0) / len(deltas), 4),
            "any_variation": len(set(deltas)) > 1,
            "mean_cost_usd": round(
                sum(r["action_cost_usd"].get(key, 0.0) for r in rows if key in r["deltas"])
                / len(deltas), 8),
        }
    return per_action


def predict_repair(
    rows: list[dict], action_key: str, feature_set: str, seed: int = 0
) -> dict[str, Any]:
    """Can `feature_set` predict whether `action_key` helps, on a held-out split?

    Returns a refusal rather than a number whenever the target has no variation
    or a split is empty. A 0.5 AUC printed from a degenerate target looks like a
    finding and is not one.
    """
    feats = FEATURE_SETS[feature_set]
    usable = [r for r in rows if action_key in r["deltas"]]
    targets = [1 if r["deltas"][action_key] > 0 else 0 for r in usable]
    if not usable:
        return {"status": "no_rows"}
    if len(set(targets)) < 2:
        return {"status": "no_variation_in_target",
                "n": len(usable), "n_positive": sum(targets)}

    train = [i for i, r in enumerate(usable) if r["split"] in ("train", "pilot")]
    test = [i for i, r in enumerate(usable) if r["split"] in ("test", "calib")]
    if not train or not test:
        # Fall back to a deterministic half split so the refusal is about data
        # volume rather than about how the run happened to be labelled.
        order = sorted(range(len(usable)), key=lambda i: usable[i]["item_id"])
        half = len(order) // 2
        train, test = order[:half], order[half:]
    if not train or not test:
        return {"status": "insufficient_rows", "n": len(usable)}
    if len({targets[i] for i in train}) < 2 or len({targets[i] for i in test}) < 2:
        return {"status": "split_lacks_both_classes",
                "n_train": len(train), "n_test": len(test)}
    n_pos_test = sum(targets[i] for i in test)
    if (len(test) < MIN_TEST_ROWS or n_pos_test < MIN_TEST_PER_CLASS
            or len(test) - n_pos_test < MIN_TEST_PER_CLASS):
        # Refusing is the point. An AUC computed on a handful of rows is not a
        # weaker version of the answer, it is a different and misleading number.
        return {"status": "insufficient_test_rows", "n_train": len(train),
                "n_test": len(test), "n_positive_test": n_pos_test,
                "required": {"rows": MIN_TEST_ROWS, "per_class": MIN_TEST_PER_CLASS}}

    X = np.array([[usable[i]["features"][f] for f in feats] for i in range(len(usable))])
    y = np.array(targets, dtype=float)
    probe = LogisticProbe().fit(X[train], y[train])
    scores = probe.score(X[test])
    value = auc(list(scores), [targets[i] for i in test])
    return {"status": "ok", "feature_set": feature_set, "auc": value,
            "n_train": len(train), "n_test": len(test),
            "base_rate": round(sum(targets) / len(targets), 4)}


def gate2_report(run_id: str, seed: int = 0) -> dict[str, Any]:
    rows = assemble(run_id)
    if not rows:
        return {"run_id": run_id, "status": "no_labelled_fanout_rows",
                "verdict": "INCONCLUSIVE"}

    repair = repairability(rows)
    informative = {k: v for k, v in repair.items() if v["any_variation"]}

    comparisons: dict[str, Any] = {}
    for key in informative:
        comparisons[key] = {name: predict_repair(rows, key, name, seed)
                            for name in FEATURE_SETS}

    correctness = {f: correctness_signal(rows, f)
                   for f in ("triage_risk", "uncertainty", "predicted_difficulty")}

    # A gate can only pass if some action's benefit actually varies across items.
    # Without variation there is nothing to predict and no signal to find.
    verdict = "INCONCLUSIVE"
    reasons = []
    if not informative:
        reasons.append(
            "no action's marginal benefit varies across items, so there is no "
            "target to predict; this is the expected result on mock-provider data"
        )
    else:
        usable = [c for c in comparisons.values()
                  if c["response_only"].get("status") == "ok"
                  and c["prompt_only"].get("status") == "ok"]
        if not usable:
            reasons.append(
                "no action had enough labelled variation on a held-out split to fit "
                "and score a probe at the required sample size "
                f"(>= {MIN_TEST_ROWS} test rows, >= {MIN_TEST_PER_CLASS} per class)"
            )
        else:
            beats = [c for c in usable
                     if (c["response_only"]["auc"] or 0) > (c["prompt_only"]["auc"] or 0)]
            if beats:
                verdict = "GO"
                reasons.append(
                    f"{len(beats)}/{len(usable)} evaluable action(s) had response-only "
                    "features beating prompt-only at predicting repair"
                )
            else:
                verdict = "NARROW"
                reasons.append(
                    f"{len(usable)} action(s) were evaluable but response-only features "
                    "did not beat prompt-only; observing the answer bought nothing"
                )

    return {
        "run_id": run_id,
        "rows": len(rows),
        "splits": {s: sum(1 for r in rows if r["split"] == s) for s in
                   sorted({r["split"] for r in rows})},
        "task_families": {f: sum(1 for r in rows if r["task_family"] == f)
                          for f in sorted({r["task_family"] for r in rows})},
        "correctness_signal": correctness,
        "repairability": repair,
        "actions_with_variation": sorted(informative),
        "feature_set_comparison": comparisons,
        "verdict": verdict,
        "reasons": reasons,
    }


def _main() -> None:
    ap = argparse.ArgumentParser(description="Decision Gate 2 signal/repairability pilot")
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    report = gate2_report(args.run, args.seed)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return

    print(f"run: {report['run_id']}   rows: {report.get('rows')}")
    print(f"splits: {report.get('splits')}   task families: {report.get('task_families')}")
    print("\n--- Q1: does the state predict a WRONG answer? ---")
    for feature, s in (report.get("correctness_signal") or {}).items():
        ci = s["ci"]
        print(f"  {feature:22s} AUC(wrong)={s['auc_wrong']} "
              f"CI[{ci['lo']}, {ci['hi']}] n={s['n']} wrong={s['n_wrong']}")

    print("\n--- Q2: marginal benefit of each action vs STOP ---")
    for key, s in (report.get("repairability") or {}).items():
        print(f"  {key:44s} mean={s['mean_delta']:+.4f} CI[{s['lo']}, {s['hi']}] "
              f"n={s['n']} (+{s['n_positive']}/-{s['n_negative']}/={s['n_zero']}) "
              f"neg_rate={s['negative_rate']} varies={s['any_variation']}")

    print(f"\nactions with any variation: {report.get('actions_with_variation')}")
    for key, sets in (report.get("feature_set_comparison") or {}).items():
        print(f"\n  predicting repair by {key}:")
        for name, res in sets.items():
            print(f"    {name:22s} {res}")

    print(f"\nVERDICT: {report['verdict']}")
    for r in report.get("reasons", []):
        print(f"  - {r}")


if __name__ == "__main__":  # pragma: no cover - CLI
    _main()
