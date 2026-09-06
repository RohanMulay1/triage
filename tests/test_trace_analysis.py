"""Signal Gate analysis (app/trace/analysis.py).

The dangerous failure mode here is a number that looks like a finding. A probe
scored on a handful of rows reports AUC 1.0 cheerfully; a degenerate target
reports 0.5. Both read as results. These tests pin the refusals.
"""
import numpy as np
import pytest

from app.trace.analysis import (
    FEATURE_SETS,
    MIN_TEST_PER_CLASS,
    MIN_TEST_ROWS,
    LogisticProbe,
    auc,
    bootstrap_ci,
    paired_bootstrap_diff,
    predict_repair,
    repairability,
)


def _row(item_id, split, u, deltas, stop_label=0.0):
    return {
        "item_id": item_id,
        "split": split,
        "task_family": "unit",
        "features": {
            "uncertainty": u, "instability": u, "contradiction": 0.0,
            "retrieval_disagreement": 0.0, "evidence_sufficiency": 1.0,
            "triage_risk": u, "answer_len": 10.0, "answer_has_digit": 1.0,
            "message_chars": 20.0, "predicted_difficulty": 0.5,
            "approx_prompt_tokens": 5.0,
        },
        "stop_label": stop_label,
        "labels": {},
        "deltas": deltas,
        "action_cost_usd": {k: 0.0 for k in deltas},
    }


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def test_auc_is_one_for_a_perfect_ranking_and_zero_for_a_reversed_one():
    assert auc([0.1, 0.2, 0.9, 0.8], [0, 0, 1, 1]) == 1.0
    assert auc([0.9, 0.8, 0.1, 0.2], [0, 0, 1, 1]) == 0.0


def test_auc_is_a_half_when_scores_are_uninformative():
    assert auc([0.5] * 6, [0, 1, 0, 1, 0, 1]) == 0.5


def test_auc_returns_none_rather_than_a_fake_half_when_a_class_is_missing():
    """0.5 from a single-class sample would read as 'no signal' rather than 'no test'."""
    assert auc([0.1, 0.9], [1, 1]) is None
    assert auc([0.1, 0.9], [0, 0]) is None


def test_bootstrap_ci_brackets_the_mean_and_reports_n():
    ci = bootstrap_ci([1.0] * 10 + [0.0] * 10, seed=1)
    assert ci["n"] == 20
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["point"] == pytest.approx(0.5, abs=0.01)


def test_bootstrap_ci_on_a_constant_has_zero_width():
    ci = bootstrap_ci([0.0] * 12)
    assert (ci["lo"], ci["point"], ci["hi"]) == (0.0, 0.0, 0.0)


def test_bootstrap_ci_on_no_data_returns_nulls_not_a_number():
    ci = bootstrap_ci([])
    assert ci == {"point": None, "lo": None, "hi": None, "n": 0}


def test_paired_bootstrap_flags_a_difference_that_excludes_zero():
    better = [1.0] * 20
    worse = [0.0] * 20
    assert paired_bootstrap_diff(better, worse)["excludes_zero"] is True
    assert paired_bootstrap_diff(better, better)["excludes_zero"] is False


def test_logistic_probe_learns_a_separable_target():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 0.4, (60, 1)), rng.normal(2, 0.4, (60, 1))])
    y = np.array([0.0] * 60 + [1.0] * 60)
    probe = LogisticProbe().fit(X, y)
    assert auc(list(probe.score(X)), list(y.astype(int))) > 0.95


def test_logistic_probe_survives_a_constant_feature():
    """A zero-variance column must not divide by zero."""
    X = np.hstack([np.ones((20, 1)), np.arange(20).reshape(-1, 1).astype(float)])
    y = np.array([0.0] * 10 + [1.0] * 10)
    scores = LogisticProbe().fit(X, y).score(X)
    assert np.all(np.isfinite(scores))


# --------------------------------------------------------------------------- #
# Repairability
# --------------------------------------------------------------------------- #
def test_repairability_reports_variation_and_negative_rate():
    rows = [_row(f"i{i}", "train", 0.5, {"a": 1.0}) for i in range(5)]
    rows += [_row(f"j{i}", "train", 0.5, {"a": -1.0}) for i in range(5)]
    table = repairability(rows)["a"]
    assert table["mean_delta"] == pytest.approx(0.0)
    assert table["n_positive"] == 5
    assert table["n_negative"] == 5
    assert table["negative_rate"] == 0.5
    assert table["any_variation"] is True


def test_repairability_marks_a_constant_action_as_having_no_variation():
    """The mock-provider result: an action that never changes anything."""
    rows = [_row(f"i{i}", "train", 0.5, {"a": 0.0}) for i in range(20)]
    table = repairability(rows)["a"]
    assert table["any_variation"] is False
    assert table["mean_delta"] == 0.0


def test_the_stop_baseline_is_not_reported_as_a_treatment():
    rows = [_row("i0", "train", 0.5, {"stop": 0.0, "a": 1.0})]
    assert "stop" not in repairability(rows)


# --------------------------------------------------------------------------- #
# The probe refuses rather than reporting noise
# --------------------------------------------------------------------------- #
def test_probe_refuses_a_target_with_no_variation():
    rows = [_row(f"i{i}", "train", 0.5, {"a": 0.0}) for i in range(60)]
    assert predict_repair(rows, "a", "response_only")["status"] == "no_variation_in_target"


def test_probe_refuses_a_test_split_that_is_too_small():
    """AUC 1.0 on three rows is not a weaker answer, it is a different number."""
    rows = ([_row(f"t{i}", "train", 0.9, {"a": 1.0}) for i in range(6)] +
            [_row(f"u{i}", "train", 0.1, {"a": -1.0}) for i in range(6)] +
            [_row("v0", "test", 0.9, {"a": 1.0}), _row("v1", "test", 0.1, {"a": -1.0})])
    result = predict_repair(rows, "a", "response_only")
    assert result["status"] == "insufficient_test_rows"
    assert result["required"] == {"rows": MIN_TEST_ROWS, "per_class": MIN_TEST_PER_CLASS}


def test_probe_refuses_when_a_split_holds_only_one_class():
    rows = ([_row(f"t{i}", "train", 0.9, {"a": 1.0}) for i in range(30)] +
            [_row(f"u{i}", "train", 0.1, {"a": -1.0}) for i in range(30)] +
            [_row(f"v{i}", "test", 0.9, {"a": 1.0}) for i in range(30)])
    assert predict_repair(rows, "a", "response_only")["status"] == \
        "split_lacks_both_classes"


def test_probe_reports_an_auc_once_the_data_is_large_enough():
    rows = ([_row(f"t{i}", "train", 0.9, {"a": 1.0}) for i in range(30)] +
            [_row(f"u{i}", "train", 0.1, {"a": -1.0}) for i in range(30)] +
            [_row(f"v{i}", "test", 0.9, {"a": 1.0}) for i in range(15)] +
            [_row(f"w{i}", "test", 0.1, {"a": -1.0}) for i in range(15)])
    result = predict_repair(rows, "a", "response_only")
    assert result["status"] == "ok"
    assert result["n_test"] == 30
    assert result["auc"] == 1.0


def test_prompt_only_features_cannot_see_a_response_only_signal():
    """The control that makes a positive result mean something."""
    rows = ([_row(f"t{i}", "train", 0.9, {"a": 1.0}) for i in range(30)] +
            [_row(f"u{i}", "train", 0.1, {"a": -1.0}) for i in range(30)] +
            [_row(f"v{i}", "test", 0.9, {"a": 1.0}) for i in range(15)] +
            [_row(f"w{i}", "test", 0.1, {"a": -1.0}) for i in range(15)])
    prompt_only = predict_repair(rows, "a", "prompt_only")
    response_only = predict_repair(rows, "a", "response_only")
    assert prompt_only["auc"] == 0.5, "prompt features are identical across rows here"
    assert response_only["auc"] > prompt_only["auc"]


@pytest.mark.parametrize("name", sorted(FEATURE_SETS))
def test_every_feature_set_names_features_the_extractor_produces(name):
    available = set(_row("i", "train", 0.5, {})["features"])
    assert set(FEATURE_SETS[name]) <= available, name
