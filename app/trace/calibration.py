"""Outcome-supervised marginal gain and disjoint isotonic calibration.

ECE is measured on (delta+1)/2, not a claimed probability of correctness.
Low-level numerical routines are used by gated run-level entry points.
"""
from __future__ import annotations

import numpy as np

from .analysis import FEATURE_SETS, assemble, bootstrap_ci, paired_bootstrap_diff
from .store import validate_run
from .support import SupportError, SupportThresholds


def isotonic_fit(scores, targets):
    if len(scores) != len(targets) or not len(scores):
        raise ValueError("nonempty paired calibration data required")
    if not np.isfinite(scores).all() or not np.isfinite(targets).all():
        raise ValueError("finite calibration data required")
    unique = sorted(set(scores))
    blocks = []
    for score in unique:
        ys = [y for x, y in zip(scores, targets) if x == score]
        blocks.append([score, score, float(sum(ys)), len(ys)])
        while len(blocks) > 1 and blocks[-2][2]/blocks[-2][3] > blocks[-1][2]/blocks[-1][3]:
            right, left = blocks.pop(), blocks.pop()
            blocks.append([left[0], right[1], left[2]+right[2], left[3]+right[3]])
    x, y = [], []
    for lo, hi, total, count in blocks:
        x.append(float(lo)); y.append(total/count)
        if hi != lo:
            x.append(float(hi)); y.append(total/count)
    return {"x": x, "y": y, "interpolation": "linear", "outside": "clip"}


def isotonic_predict(mapping, scores):
    return np.interp(scores, mapping["x"], mapping["y"]).tolist()


def calibration_metrics(scores, targets, seed=0, bins=5):
    scores, targets = list(scores), list(targets)
    if not scores or len(scores) != len(targets):
        return {"status": "REFUSED", "ece": bootstrap_ci([]), "bins": []}
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in scores+targets):
        raise ValueError("normalized gains must be finite in [0,1]")
    def ece(indices):
        error = 0.0
        for b in range(bins):
            ix = [i for i in indices if min(bins-1, int(scores[i]*bins)) == b]
            if ix:
                error += abs(sum(scores[i]-targets[i] for i in ix))/len(indices)
        return error
    reliability = []
    for b in range(bins):
        ix = [i for i in range(len(scores)) if min(bins-1, int(scores[i]*bins)) == b]
        reliability.append({"lower": b/bins, "upper": (b+1)/bins, "n": len(ix),
                            "predicted": bootstrap_ci([scores[i] for i in ix], seed=seed),
                            "observed": bootstrap_ci([targets[i] for i in ix], seed=seed)})
    return {"status": "OK", "n": len(scores), "ece": bootstrap_ci(
        list(range(len(scores))), statistic=ece, seed=seed), "bins": reliability,
        "ci_scope": "test resampling conditional on fixed training and calibration fits"}


def fit_gain(rows, action, feature_set="response_only"):
    usable = [r for r in rows if action in r["deltas"]]
    roles = {role: [r for r in usable if r["split"] == role] for role in ("train", "calib", "test")}
    counts = {role: len(group) for role, group in roles.items()}
    ids = {role: {r["item_id"] for r in group} for role, group in roles.items()}
    if any(ids[a] & ids[b] for a,b in (("train","calib"),("train","test"),("calib","test"))):
        return {"status": "REFUSED_SPLIT_LEAKAGE", "counts": counts}
    if any(counts[r] < n for r,n in (("train",20),("calib",10),("test",20))):
        return {"status": "REFUSED_SPLIT_COVERAGE", "counts": counts}
    if len({r["deltas"][action] for r in roles["train"]}) < 2:
        return {"status": "REFUSED_NO_TRAIN_VARIATION", "counts": counts}
    feats = FEATURE_SETS[feature_set]
    def matrix(group):
        return np.array([[r["features"][f] for f in feats] for r in group], dtype=float)
    train = matrix(roles["train"])
    mu, scale = train.mean(axis=0), train.std(axis=0)
    scale[scale == 0] = 1
    def design(group):
        x = (matrix(group)-mu)/scale
        return np.column_stack([x, np.ones(len(x))])
    y = np.array([(r["deltas"][action]+1)/2 for r in roles["train"]])
    x = design(roles["train"])
    penalty = np.eye(x.shape[1]); penalty[-1,-1] = 0
    weights = np.linalg.solve(x.T@x+penalty, x.T@y)
    raw_calib = np.clip(design(roles["calib"])@weights, 0, 1).tolist()
    mapping = isotonic_fit(raw_calib, [(r["deltas"][action]+1)/2 for r in roles["calib"]])
    raw = np.clip(design(roles["test"])@weights,0,1).tolist()
    calibrated = isotonic_predict(mapping, raw)
    targets = [(r["deltas"][action]+1)/2 for r in roles["test"]]
    return {"status": "OK", "counts": counts, "roles": {k: sorted(v) for k,v in ids.items()},
            "model": {"features": list(feats), "mean": mu.tolist(), "scale": scale.tolist(),
                      "weights": weights.tolist(), "ridge": 1.0}, "map": mapping,
            "test_item_ids": [r["item_id"] for r in roles["test"]],
            "raw_gain": [2*s-1 for s in raw], "calibrated_gain": [2*s-1 for s in calibrated],
            "raw": calibration_metrics(raw, targets),
            "calibrated": calibration_metrics(calibrated, targets)}


def calibration_block(rows):
    actions = sorted({a for r in rows for a in r["deltas"] if a != "stop"})
    fits = {a: fit_gain(rows, a) for a in actions}
    return {"status": "COMPLETE", "n_fitted": sum(f["status"] == "OK" for f in fits.values()), "target": "normalized marginal quality gain (delta+1)/2",
            "roles": "train=ridge fit; calib=isotonic map; test=ECE and reliability",
            "per_action": fits, "induced_choice_change": calibration_choice_change(fits)}


def calibration_report(run_id, diagnostic=False):
    validation = validate_run(run_id)
    meta = {"analysis_grade": validation["analysis_grade"], "not_evidence": not validation["analysis_grade"]}
    if not validation["analysis_grade"] and not (diagnostic and validation["force_mock"]
            and not validation["errors"] and validation["chain_ok"] and validation["cost_conservation_ok"]):
        return {**meta, "status": "REFUSED", "reason": "non-analysis-grade run", "per_action": {}}
    try:
        rows = assemble(run_id, SupportThresholds(allow_synthetic=diagnostic))
    except SupportError as exc:
        return {**meta, "status": "REFUSED", "reason": str(exc), "per_action": {}}
    return {**meta, **calibration_block(rows)}


def calibration_choice_change(fits):
    """Choice changes among fitted root actions plus STOP, not a full policy.

    Unfitted actions are disclosed, never assigned default gains. Depth-2 fits
    describe other decision states and cannot be ranked as root actions.
    """
    root = {a: f for a, f in fits.items() if f.get("status") == "OK" and " -> " not in a}
    meta = {"scope": "fitted_root_action_subset_plus_STOP; not full-policy evaluation",
            "actions": sorted(root), "excluded_actions": sorted(set(fits)-set(root)),
            "ci_scope": "paired test-item resampling conditional on fixed fits"}
    if not root:
        return {**meta, "status": "REFUSED", "reason": "no fitted root actions",
                "ci95": paired_bootstrap_diff([], [])}
    common = set.intersection(*(set(f["test_item_ids"]) for f in root.values()))
    if len(common) < 20:
        return {**meta, "status": "REFUSED", "reason": "fewer than 20 common test items",
                "n_common": len(common), "ci95": paired_bootstrap_diff([], [])}
    predictions = {a: {item: (raw, cal) for item, raw, cal in zip(
        f["test_item_ids"], f["raw_gain"], f["calibrated_gain"])} for a, f in root.items()}
    changes = []
    for item in sorted(common):
        def choose(index):
            scores = {a: values[item][index] for a, values in predictions.items()}
            scores["stop"] = 0.0
            return max(scores, key=lambda a: (scores[a], a == "stop", a))
        changes.append(float(choose(0) != choose(1)))
    return {**meta, "status": "OK", "n": len(common),
            "ci95": paired_bootstrap_diff(changes, [0.0]*len(changes)),
            "test_item_ids": sorted(common), "performance_claim": False}
