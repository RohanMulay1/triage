"""Gated C1 gain replacement and C4 collection-protocol head-to-head.

Unavailable estimands carry null intervals with n=0, never invented zeros.
"""
from .analysis import assemble, bootstrap_ci, paired_bootstrap_diff
from .calibration import calibration_report, fit_gain
from .store import read_run, validate_run
from .support import SupportError, SupportThresholds, assert_estimable


def refused(reason):
    return {"status": "REFUSED", "reason": reason, "ci95": bootstrap_ci([])}


def compare_fits(left, right, rows):
    """Same held-out items; fits carry calibrated per-action expected gains."""
    per_action = {}
    for action in sorted(set(left) | set(right)):
        a, b = left.get(action, {}), right.get(action, {})
        if a.get("status") != "OK" or b.get("status") != "OK":
            per_action[action] = refused("one estimator lacks supported split coverage/variation")
            continue
        if a["test_item_ids"] != b["test_item_ids"]:
            per_action[action] = refused("unpaired held-out items")
            continue
        per_action[action] = {"status": "OK", "ci95": paired_bootstrap_diff(
            a["calibrated_gain"], b["calibrated_gain"]),
            "n": len(a["test_item_ids"])}
    if not per_action or any(v["status"] != "OK" for v in per_action.values()):
        return {**refused("incomplete common action support"), "per_action": per_action,
                "policy_disagreement": refused("incomplete common action support"),
                "established": False}
    predictions = []
    for fits in (left, right):
        predictions.append({action: dict(zip(f["test_item_ids"], f["calibrated_gain"]))
                            for action,f in fits.items()})
    disagreements, quality_left, quality_right, rank_disagreements = [], [], [], []
    for row in rows:
        if row["split"] != "test":
            continue
        actions = sorted(a for a in row["deltas"] if a != "stop")
        if not actions or any(a not in predictions[0] or a not in predictions[1] or
                              row["item_id"] not in predictions[0][a] or
                              row["item_id"] not in predictions[1][a] for a in actions):
            continue
        rankings = []
        for pred in predictions:
            values = {a: pred[a][row["item_id"]] for a in actions}
            values['stop'] = 0.0
            rankings.append(sorted(values, key=lambda a: (values[a], a == 'stop', a), reverse=True))
        x,y = rankings
        disagreements.append(float(x[0] != y[0]))
        rank_disagreements.append(float(x != y))
        quality_left.append(row["deltas"].get(x[0], 0))
        quality_right.append(row["deltas"].get(y[0], 0))
    ci = paired_bootstrap_diff(disagreements, [0.0]*len(disagreements))
    return {"status": "OK" if disagreements else "REFUSED", "per_action": per_action,
            "ci95": ci,
            "policy_disagreement": {"ci95": ci},
            "ranking_disagreement_ci95": paired_bootstrap_diff(rank_disagreements, [0.0]*len(rank_disagreements)),
            "policy_quality_difference_ci95": paired_bootstrap_diff(quality_left, quality_right),
            "established": ci["lo"] is not None and ci["lo"] > 0,
            "interpretation": "different is not better; quality-only selection, not equal-cost superiority"}


def c4_report(run_id, diagnostic=False):
    validation = validate_run(run_id)
    meta = {"not_evidence": not validation["analysis_grade"], "analysis_grade": validation["analysis_grade"],
            "design": "matched served projection: observed draft baseline, chosen treatment only; common held-out fanout evaluation"}
    full = calibration_report(run_id, diagnostic)
    if full["status"] != "COMPLETE":
        return {**meta, **refused(full.get("reason", "calibration refused")), "established": False}
    traces = read_run(run_id)
    projected = [t for t in traces if t.branch_id == 'prefix' or
                 (t.decision_depth == 1 and t.steps[0].decision.rationale.get('served'))]
    # Inclusion propensity is one for exhaustive fan-out. It is not the
    # probability of the selected served action. Preserve the latter separately.
    for trajectory in projected:
        if trajectory.branch_id == 'prefix':
            continue
        propensity = trajectory.steps[0].decision.rationale.get('behavior_propensity')
        if propensity is None:
            return {**meta, **refused('served behavior propensity not recorded'), 'established': False}
        step = trajectory.steps[0]
        trajectory.steps[0] = step.model_copy(update={'decision': step.decision.model_copy(
            update={'propensity': propensity})})
    rows = [r for r in assemble(run_id, SupportThresholds(allow_synthetic=diagnostic))
            if r.get('decision_depth',1) == 1]
    actions = sorted({a for r in rows for a in r['deltas'] if a != 'stop'})
    try:
        assert_estimable(projected, SupportThresholds(allow_synthetic=diagnostic))
    except SupportError as exc:
        return {**meta, **refused(str(exc)), "per_action": {a: refused('served action support failed') for a in actions},
                "established": False, "full_fits": {a: full['per_action'].get(a) for a in actions},
                "served_fits": {}, "limitation": "legacy served trajectories cannot identify unchosen action values"}
    choices = {t.item_id: t.branch_id for t in projected if t.branch_id != 'prefix'}
    served_rows = [{**r, 'deltas': {a:d for a,d in r['deltas'].items() if
                    r['split']=='test' or a == choices.get(r['item_id'])}} for r in rows]
    served_fits = {a: fit_gain(served_rows, a) for a in actions}
    full_fits = {a: fit_gain(rows, a) for a in actions}
    result = compare_fits(served_fits, full_fits, rows)
    return {**meta, **result, "established": result['established'] and validation['analysis_grade'],
            "served_fits": served_fits, "full_fits": full_fits}


def c1_report(run_id, diagnostic=False):
    calibrated = calibration_report(run_id, diagnostic)
    meta = {"not_evidence": calibrated['not_evidence'], "established": False,
            "design": "replace only gain; fixed recorded penalties and common feasible set"}
    if calibrated['status'] != 'COMPLETE':
        return {**meta, **refused(calibrated.get('reason','calibration refused'))}
    rows = [r for r in assemble(run_id, SupportThresholds(allow_synthetic=diagnostic))
            if r.get('decision_depth',1) == 1 and r['split']=='test']
    prefixes = {t.item_id:t for t in read_run(run_id) if t.branch_id=='prefix'}
    old, new, changes = [], [], []
    reasons = []
    for row in rows:
        details = [s.outcome.detail for s in prefixes[row['item_id']].steps
                   if s.outcome.detail.get('utility_choice')]
        if not details:
            reasons.append('missing recorded heuristic utility scores'); continue
        scores = details[-1]['utility_choice']['scores']
        if set(scores) != set(row['deltas']):
            reasons.append('missing outcomes for feasible actions'); continue
        values, base = {}, {}
        for action in row['deltas']:
            if action not in scores:
                reasons.append('missing heuristic action'); break
            score = scores[action]
            if action == 'stop':
                gain = 0.0
            else:
                fit = calibrated['per_action'].get(action, {})
                if fit.get('status') != 'OK' or row['item_id'] not in fit['test_item_ids']:
                    reasons.append('gain estimator refused for feasible action'); break
                gain = fit['calibrated_gain'][fit['test_item_ids'].index(row['item_id'])]
            base[action] = score['utility']
            values[action] = score['utility']-score['gain']+gain
        else:
            choose = lambda v: max(v, key=lambda a:(v[a], a=='stop', a))
            a,b = choose(base),choose(values)
            old.append(row['deltas'][a]); new.append(row['deltas'][b]); changes.append(float(a!=b))
    if reasons or not old:
        return {**meta, **refused('; '.join(sorted(set(reasons))) or 'no held-out pairs'),
                "n_complete": len(old)}
    ci = paired_bootstrap_diff(new, old)
    return {**meta, "status": "OK", "ci95": ci,
            "choice_disagreement_ci95": paired_bootstrap_diff(changes,[0.0]*len(changes)),
            "established": not meta['not_evidence'] and len(old)>=20 and ci['lo']>0,
            "limitation": "quality contrast only; equal realized-cost performance not established"}
