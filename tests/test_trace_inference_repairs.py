"""Gate 2 refuses invalid evidence and tests a paired ranking difference."""
from pathlib import Path

import pytest

from app.trace import analysis, store
from app.trace.contract import Action, ActionKind
from app.trace.support import SupportThresholds, marginal_value_table, SupportError
from test_trace_analysis import _row
from test_trace_branching import _fanout


def rows(split, n=40):
    return [_row(f"{split}-{i}", split, float(i % 2), {"repair": float(i % 2)})
            for i in range(n)]


def test_d4_unsupported_action_refuses_before_feature_comparison(monkeypatch):
    trajectories = _fanout()
    # A feasible action with no observed branch, independently of mock status.
    first = trajectories[0].steps[0]
    missing = Action(kind=ActionKind.VERIFY, model_id="never_observed")
    decision = first.decision.model_copy(update={"feasible": first.decision.feasible + [missing]})
    trajectories[0] = trajectories[0].model_copy(update={
        "steps": [first.model_copy(update={"decision": decision})]})
    monkeypatch.setattr(analysis, "read_run", lambda _: trajectories)
    monkeypatch.setattr(analysis, "validate_run", lambda _: {"analysis_grade": True}, raising=False)
    monkeypatch.setattr(analysis, "assemble", lambda *a, **kw: pytest.fail("estimated before support gate"))
    result = analysis.gate2_report("unsupported")
    assert result["verdict"] == "REFUSED"
    assert "never_observed" in result["support_error"]
    assert "feature_set_comparison" not in result


def test_d5_committed_mock_is_not_analysis_grade_and_cannot_go(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "data" / "traces"
    monkeypatch.setattr(store, "trace_root", lambda: root)
    result = analysis.gate2_report("gate2-pilot-mock")
    assert result["analysis_grade"] is False
    assert result["not_evidence"] is True
    assert result["verdict"] == "REFUSED"
    assert "feature_set_comparison" not in result


def test_explicit_synthetic_support_override_still_cannot_authorize_go(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "data" / "traces"
    monkeypatch.setattr(store, "trace_root", lambda: root)
    result = analysis.gate2_report("gate2-pilot-mock", thresholds=SupportThresholds(
        min_observations=0, allow_synthetic=True))
    assert result["verdict"] == "REFUSED"
    assert result["support_thresholds"]["allow_synthetic"] is True


def test_d6_train_only_never_invents_an_item_id_test_split():
    result = analysis.predict_repair(rows("train", 80), "repair", "response_only")
    assert result["status"] == "insufficient_split_coverage"
    assert "auc" not in result


def test_d10_calibration_rows_never_enter_scored_indices():
    data = rows("train") + rows("calib") + rows("test")
    result = analysis.predict_repair(data, "repair", "response_only")
    assert result["n_train"] == result["n_test"] == 40
    assert all(data[i]["split"] == "test" for i in result["test_indices"])
    assert set(result["test_item_ids"]) == {r["item_id"] for r in rows("test")}


def comparison_report(monkeypatch, response_scores):
    labels = [0] * 20 + [1] * 20
    prompt_scores = [0.5] * 40
    monkeypatch.setattr(analysis, "read_run", lambda _: [])
    monkeypatch.setattr(analysis, "assert_estimable", lambda *a: {}, raising=False)
    monkeypatch.setattr(analysis, "validate_run", lambda _: {"analysis_grade": True}, raising=False)
    monkeypatch.setattr(analysis, "assemble", lambda *a: rows("train") + rows("test"))

    def probe(data, action, name, seed):
        scores = response_scores if name == "response_only" else prompt_scores
        return {"status": "ok", "auc": analysis.auc(scores, labels),
                "test_item_ids": list(range(40)), "test_labels": labels,
                "test_scores": scores, "n_test": 40}

    monkeypatch.setattr(analysis, "predict_repair", probe)
    return analysis.gate2_report("controlled-ranking-test")


def test_d9_small_point_auc_improvement_with_paired_ci_touching_zero_is_not_go(monkeypatch):
    scores = [0.5] * 40
    scores[20] = 0.6
    result = comparison_report(monkeypatch, scores)
    assert result["verdict"] == "NARROW"
    interval = result["feature_set_comparison"]["repair"]["paired_auc_difference"]
    assert interval["point"] > 0
    assert interval["lo"] <= 0 <= interval["hi"]
    assert interval["n"] == 40


def test_paired_auc_interval_can_detect_a_large_ranking_difference(monkeypatch):
    result = comparison_report(monkeypatch, [0.1] * 20 + [0.9] * 20)
    assert result["verdict"] == "GO"
    interval = result["feature_set_comparison"]["repair"]["paired_auc_difference"]
    assert interval["lo"] > 0


def test_direct_marginal_estimate_also_requires_support():
    with pytest.raises(SupportError):
        marginal_value_table(_fanout())


def test_empty_support_is_not_an_estimable_population():
    with pytest.raises(SupportError):
        marginal_value_table([])


def test_unavailable_inventory_does_not_invent_an_unsupported_estimand():
    from app.trace.support import assert_estimable

    result = assert_estimable(_fanout(), SupportThresholds(allow_synthetic=True))
    unavailable = result["per_action"]["specialist_model"]
    assert unavailable["n_feasible"] == 0
    assert unavailable["status_counts"]["unavailable"] == 1
    assert "specialist_model" not in result["unsupported_actions"]
