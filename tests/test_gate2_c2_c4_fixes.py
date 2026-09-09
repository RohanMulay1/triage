"""Tests for Gate 2, C2, and C4 substrate fixes."""
import random

from app.config import load_router_config
from app.trace import store
from app.trace.contract import OutcomeStatus
from app.trace.calibration import fit_gain_pooled
from app.trace.policies import ConcentratedExplorationPolicy
from app.trace.actions import InterventionContext, InterventionExecutor
from test_trace_measurement_repairs import collect


def test_propensity_persisted_in_rationale():
    """Verify that behavior propensity is stored in rationale and survives."""
    cfg = load_router_config()
    policy = ConcentratedExplorationPolicy(["stop", "stronger_model", "resample"])
    ctx = InterventionContext(question="What is 2+2?", answer="4", cfg=cfg, small_id="llama-3.1-8b", big_id="gpt-4o", seed=42)
    executor = InterventionExecutor(cfg)
    choice = policy.choose(ctx, executor, [], random.Random(42))
    assert choice.propensity > 0.0
    assert any(act in choice.action.key for act in ["stop", "stronger_model", "resample"])
    assert "concentrated" in choice.rationale


def test_fit_gain_pooled_computes_joint_regression():
    """Verify fit_gain_pooled fits joint model across multiple served actions."""
    rows = []
    actions = ["stronger_model", "resample", "verify"]
    # Provide sufficient per-action rows: 30 train, 15 calib, 25 test (>= 20)
    idx = 0
    for act in actions:
        splits = ["train"] * 30 + ["calib"] * 15 + ["test"] * 25
        for i, split in enumerate(splits):
            feat_val = float((idx + i) % 5)
            # Alternate positive and negative deltas so both classes exist
            delta_val = 1.0 if (i % 2 == 0) else -1.0
            rows.append({
                "item_id": f"item-{idx:04d}",
                "split": split,
                "decision_depth": 1,
                "features": {
                    "uncertainty": 0.1 * (i % 3),
                    "instability": feat_val * 0.1,
                    "contradiction": 0.1,
                    "retrieval_disagreement": 0.2,
                    "evidence_sufficiency": 0.8,
                    "answer_len": 100.0 + (i % 10),
                    "answer_has_digit": 1.0,
                },
                "deltas": {act: delta_val},
                "stop_label": 0.0,
            })
            idx += 1

    res = fit_gain_pooled(rows, actions, feature_set="response_only")
    assert res["status"] == "OK"
    for a in actions:
        assert a in res["per_action"]
        fit = res["per_action"][a]
        assert fit["status"] == "OK"
        assert len(fit["calibrated_gain"]) == len(fit["test_item_ids"])
        assert "raw" in fit and "calibrated" in fit


def test_c2_incomplete_pairs_does_not_disqualify_complete_pairs(monkeypatch):
    """Verify that when some items have dropped branches, complete pairs are retained."""
    from app.trace import information_value as iv
    run_id = store.new_run_id("c2-pairs-test")
    manifest = store.build_manifest(run_id, dataset="unit")
    store.create_run(manifest)
    traces = []
    # Create 25 complete test items and 3 incomplete test items
    for i in range(28):
        item_id = f"unit-{i:04d}"
        item_traces = [t.model_copy(update={"split": "test", "item_id": item_id})
                       for t in collect(run_id=run_id, depth=2, item_id=item_id, seed=i)]
        if i >= 25:
            # Drop depth-2 continuations for the last 3 items to simulate provider drops
            item_traces = [t for t in item_traces if t.decision_depth != 2]
        traces.extend(item_traces)

    monkeypatch.setattr(iv, "read_run", lambda _: traces)
    monkeypatch.setattr(iv, "validate_run", lambda _: {"analysis_grade": True, "force_mock": False,
                                                       "cost_conservation_ok": True, "chain_ok": True, "errors": []})
    monkeypatch.setattr(iv, "assert_estimable", lambda *args: {"actions_with_support": ["retrieve", "verify", "resample"]})

    report = iv.c2_report(run_id, continuation="verify", diagnostic=False)
    retrieve_res = report["per_action"]["retrieve"]
    # 25 complete pairs >= 20, so it should be EVALUABLE, not REFUSED_INCOMPLETE_PAIRS
    assert retrieve_res["status"] == "EVALUABLE"
    assert retrieve_res["n"] == 25
    assert len(retrieve_res["missing_items"]) == 3


def test_synthetic_live_contamination_raises_error():
    """Verify that store.validate_run detects synthetic data in live run."""
    run_id = store.new_run_id("live-contam-test")
    manifest = store.build_manifest(run_id, dataset="unit")
    # Live run manifest has force_mock = False
    manifest = manifest.model_copy(update={"force_mock": False})
    store.create_run(manifest)
    traj = collect(run_id=run_id, depth=1, item_id="item-0")[0]
    # Inject a mock outcome
    step = traj.steps[0]
    mock_outcome = step.outcome.model_copy(update={"status": OutcomeStatus.SYNTHETIC_FALLBACK, "provider_label": "mock"})
    traj = traj.model_copy(update={"steps": [step.model_copy(update={"outcome": mock_outcome})]})
    store.append(traj)

    val = store.validate_run(run_id)
    assert not val["analysis_grade"]
    assert any("synthetic" in e.lower() or "mock" in e.lower() for e in val["errors"])


def test_feature_variance_audit_in_analysis(monkeypatch):
    """Verify that feature variance is audited and reported in gate2_report."""
    from app.trace import analysis
    run_id = store.new_run_id("feat-var-test")
    manifest = store.build_manifest(run_id, dataset="unit")
    store.create_run(manifest)
    for i in range(4):
        for t in collect(run_id=run_id, depth=2, item_id=f"item-{i}", seed=i):
            store.append(t)
    monkeypatch.setattr(analysis, "validate_run", lambda _: {"analysis_grade": True, "force_mock": False})
    monkeypatch.setattr(analysis, "assert_estimable", lambda *a: {}, raising=False)
    report = analysis.gate2_report(run_id)
    assert "feature_variance_audit" in report
    fva = report["feature_variance_audit"]
    assert "prompt_only" in fva
    assert "response_only" in fva
    assert "n_features" in fva["prompt_only"]
    assert "non_zero_variance" in fva["prompt_only"]
