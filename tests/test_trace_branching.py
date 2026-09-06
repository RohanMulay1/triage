"""Counterfactual fan-out and the support refusal gate.

This is the machinery the Signal Gate depends on. Its correctness conditions are
unusual: what must be true is not "the policy did well" but "the data can support
a claim at all" -- complete support at the branch point, honest statuses for
everything that did not really execute, and cost that is marginal where a
marginal quantity is what the analysis differences.
"""
import asyncio

import pytest

from app.config import load_router_config
from app.trace import store
from app.trace.branching import BudgetExceeded, RunBudget, collect_fanout
from app.trace.contract import ActionCost, ActionKind, Budget, OutcomeStatus
from app.trace.policies import ResponseRiskThresholdPolicy
from app.trace.support import (
    SupportError,
    SupportThresholds,
    action_support,
    assert_estimable,
    marginal_value_table,
    missingness,
)

CONFIDENT_Q = "What is the capital of France?"


def _cfg():
    return load_router_config()


def _labeler(answer, abstained):
    return 0.0 if abstained else (1.0 if "Paris" in answer else 0.0)


def _fanout(run_id="branch-test", item_id="item-0", question=CONFIDENT_Q,
            budget=None, policy=None):
    return asyncio.run(collect_fanout(
        item_id=item_id, question=question, cfg=_cfg(), small_id="llama-3.1-8b",
        big_id="gpt-4o", run_id=run_id, dataset="unit", split="pilot", seed=0,
        labeler=_labeler, label_source="unit",
        served_policy=policy or ResponseRiskThresholdPolicy(0.55),
        budget=budget, prompt_features={"task_family": "unit"},
    ))


# --------------------------------------------------------------------------- #
# Shape of the fan-out
# --------------------------------------------------------------------------- #
def test_fanout_emits_a_prefix_plus_one_branch_per_action_in_the_space():
    trajectories = _fanout()
    prefix = [t for t in trajectories if t.branch_id == "prefix"]
    assert len(prefix) == 1
    assert prefix[0].terminal.status == "prefix"
    assert prefix[0].terminal.label is None, "the prefix is not a served episode"
    assert len(trajectories) > 5


def test_every_branch_hangs_off_the_same_parent_state():
    """Complete support means the same state followed by different actions."""
    trajectories = _fanout()
    prefix = next(t for t in trajectories if t.branch_id == "prefix")
    shared = prefix.states[-1].state_id
    branches = [t for t in trajectories if t.branch_id != "prefix"]
    assert branches
    for t in branches:
        assert t.root_state_id == shared
        assert t.steps[0].outcome.parent_state_id == shared
        assert t.parent_trajectory_id == prefix.trajectory_id


def test_exactly_one_branch_is_marked_chosen_and_the_rest_counterfactual():
    trajectories = _fanout()
    executed = [t for t in trajectories if t.branch_id != "prefix"
                and t.steps[0].outcome.status in
                (OutcomeStatus.CHOSEN, OutcomeStatus.COUNTERFACTUAL)]
    chosen = [t for t in executed if t.steps[0].outcome.status is OutcomeStatus.CHOSEN]
    counterfactual = [t for t in executed
                      if t.steps[0].outcome.status is OutcomeStatus.COUNTERFACTUAL]
    assert len(chosen) == 1
    assert counterfactual
    assert all(s.outcome.is_counterfactual for t in counterfactual for s in t.steps)


def test_unavailable_actions_are_recorded_free_and_unlabelled():
    """Dropping them would claim availability nothing could have executed."""
    trajectories = _fanout()
    unavailable = [t for t in trajectories
                   if t.steps and t.steps[0].outcome.status is OutcomeStatus.UNAVAILABLE]
    assert unavailable, "specialist_model has no integration and must appear"
    for t in unavailable:
        assert t.total_cost.llm_calls == 0
        assert t.terminal.label is None
        assert t.steps[0].outcome.error


def test_specialist_model_branch_is_present_and_unavailable():
    trajectories = _fanout()
    branch = next(t for t in trajectories if t.branch_id == "specialist_model")
    assert branch.steps[0].outcome.status is OutcomeStatus.UNAVAILABLE


def test_executed_branches_reach_a_terminal_and_carry_a_label():
    trajectories = _fanout()
    for t in trajectories:
        first = t.steps[0].outcome if t.steps else None
        if first is None or first.status not in (OutcomeStatus.CHOSEN,
                                                 OutcomeStatus.COUNTERFACTUAL):
            continue
        if t.branch_id == "prefix":
            continue
        assert t.terminal.label is not None
        assert t.steps[-1].outcome.action.kind in (ActionKind.STOP, ActionKind.ABSTAIN)


def test_forced_steps_do_not_inflate_the_support_denominator():
    """The prefix and the terminal STOP were not free choices.

    Listing the whole action space as feasible there would count occasions on
    which nothing could have been chosen differently, understating coverage for
    every action.
    """
    trajectories = _fanout()
    prefix = next(t for t in trajectories if t.branch_id == "prefix")
    assert len(prefix.steps[0].decision.feasible) == 1
    assert prefix.steps[0].decision.rationale.get("forced") is True

    multi = next(t for t in trajectories if t.branch_id != "prefix" and len(t.steps) == 2)
    assert len(multi.steps[-1].decision.feasible) == 1
    assert multi.steps[-1].decision.rationale.get("forced") is True


# --------------------------------------------------------------------------- #
# Cost accounting
# --------------------------------------------------------------------------- #
def test_cumulative_cost_is_the_running_total_and_cost_is_the_increment():
    trajectories = _fanout()
    for t in trajectories:
        running = ActionCost.zero()
        for step in t.steps:
            running = running + step.outcome.cost
            assert step.outcome.cumulative_cost.tokens_in >= step.outcome.cost.tokens_in
            assert step.outcome.cumulative_cost.llm_calls >= step.outcome.cost.llm_calls
        assert t.total_cost.llm_calls == running.llm_calls


def test_stronger_model_cost_is_the_marginal_pass_not_the_whole_trajectory():
    """Trading up costs the big-model pass; the small pass was already paid."""
    trajectories = _fanout()
    prefix = next(t for t in trajectories if t.branch_id == "prefix")
    branch = next(t for t in trajectories if t.branch_id.startswith("stronger_model"))
    escalation = branch.steps[0].outcome
    assert escalation.cost.llm_calls == 1
    # The prefix's tokens are not re-charged to the escalation.
    assert escalation.cost.tokens_in < prefix.total_cost.tokens_in + escalation.cost.tokens_in


def test_branch_costs_do_not_double_count_the_shared_prefix():
    trajectories = _fanout()
    prefix = next(t for t in trajectories if t.branch_id == "prefix")
    for t in trajectories:
        if t.branch_id == "prefix":
            continue
        assert all(s.outcome.action.key != prefix.steps[0].outcome.action.key
                   or s.outcome.parent_state_id != prefix.root_state_id
                   for s in t.steps)


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #
def test_run_budget_accumulates_and_refuses_before_the_action_runs():
    budget = RunBudget(Budget(max_llm_calls=2))
    one = ActionCost(llm_calls=1)
    budget.charge(one)
    budget.charge(one)
    assert budget.state.spent_llm_calls == 2
    with pytest.raises(BudgetExceeded):
        budget.charge(one)


def test_fanout_stops_when_the_run_budget_is_exhausted():
    with pytest.raises(BudgetExceeded):
        _fanout(budget=Budget(max_llm_calls=1))


def test_an_unbounded_budget_never_blocks():
    trajectories = _fanout(budget=Budget())
    assert len(trajectories) > 5


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_counterfactual_branches_survive_a_write_and_read():
    run_id = store.new_run_id("cf-persist")
    store.create_run(store.build_manifest(run_id, dataset="unit"))
    for t in _fanout(run_id=run_id):
        store.append(t)
    reloaded = store.read_run(run_id)

    assert len(reloaded) == len(_fanout(run_id=run_id + "-x"))
    assert any(s.outcome.is_counterfactual for t in reloaded for s in t.steps)
    assert any(t.parent_trajectory_id for t in reloaded)
    assert store.validate_run(run_id)["cost_conservation_ok"] is True


# --------------------------------------------------------------------------- #
# Support gate
# --------------------------------------------------------------------------- #
def test_fanout_gives_every_executed_action_support():
    support = action_support(_fanout())
    executed = {k: v for k, v in support["per_action"].items() if v["n_observed"]}
    assert executed
    for key, s in executed.items():
        assert s["n_taken"] >= 1
        assert s["min_propensity"] == 1.0


def test_missingness_separates_unavailable_from_executed():
    counts = missingness(_fanout())
    assert counts["unavailable"] >= 1
    assert counts["counterfactual"] >= 1
    assert counts["chosen"] >= 1


def test_support_gate_refuses_when_an_action_has_no_observation():
    """The refusal is the feature; an advisory diagnostic gets ignored."""
    with pytest.raises(SupportError) as excinfo:
        assert_estimable(_fanout(), SupportThresholds(allow_synthetic=True))
    assert "specialist_model" in str(excinfo.value)


def test_support_gate_refuses_mock_outcomes_by_default():
    with pytest.raises(SupportError) as excinfo:
        assert_estimable(_fanout())
    assert "mock-provider outcome" in str(excinfo.value)


def test_support_gate_refuses_a_run_mixing_action_space_definitions():
    from app.trace.support import positivity_violations

    support = action_support(_fanout())
    support["mixed_feasible_set_definitions"] = True
    support["feasible_set_definitions"] = ["executor_v1", "other_v2"]
    violations = positivity_violations(support)
    assert any("not comparable" in v for v in violations)


# --------------------------------------------------------------------------- #
# Marginal value
# --------------------------------------------------------------------------- #
def test_marginal_value_differences_each_branch_against_the_stop_branch():
    table = marginal_value_table(_fanout())
    assert table["items_with_baseline"] == 1
    assert "stop" not in table["per_action"], "STOP is the baseline, not a treatment"
    abstain = table["per_action"]["abstain"]
    # Abstaining on a question the model answers correctly is a real loss.
    assert abstain["mean_delta"] <= 0.0


def test_items_without_a_stop_branch_are_reported_not_silently_dropped():
    trajectories = [t for t in _fanout() if t.branch_id != "stop"]
    table = marginal_value_table(trajectories)
    assert table["items_skipped_no_stop_branch"] == 1
    assert table["items_with_baseline"] == 0
