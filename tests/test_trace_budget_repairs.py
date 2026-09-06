"""Admission tests assert executions and retained spend, not only exceptions."""
import asyncio

import pytest

from app.config import load_router_config
from app.llm import LLMClient
from app.trace.actions import InterventionExecutor
from app.trace.adapter import answer_action, stop_action
from app.trace.branching import BudgetExceeded, RunBudget, collect_fanout
from app.trace.contract import ActionCost, ActionKind, Budget


def collect(budget, item="i"):
    return asyncio.run(collect_fanout(
        item_id=item, question="Who wrote Hamlet?", cfg=load_router_config(),
        small_id="llama-3.1-8b", big_id="gpt-4o", run_id="budget-unit",
        dataset="unit", split="train", seed=0, labeler=lambda a, b: 0.0,
        label_source="unit", budget=budget))


def test_d1_one_call_cap_never_executes_a_second_model_call(monkeypatch):
    calls = []
    original = LLMClient.generate

    async def counted(self, *args, **kwargs):
        calls.append(self.model_id)
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(LLMClient, "generate", counted)
    with pytest.raises(BudgetExceeded):
        collect(Budget(max_llm_calls=1))
    assert len(calls) == 1


def test_d2_single_accumulator_retains_spend_into_second_item(monkeypatch):
    # Controlled priced stand-in, never a live provider. Each item costs 2 units:
    # its shared answer plus one re-answer. The cap covers 1.5 items.
    def feasible(self, ctx, taken=None):
        return [(stop_action(), True, ""), (answer_action(ctx.small_id, ctx.cfg), True, "")]

    original = InterventionExecutor.execute
    calls = []

    async def priced(self, action, ctx):
        result = await original(self, action, ctx)
        if action.kind == ActionKind.ANSWER:
            calls.append(ctx.question)
            result.cost = ActionCost(llm_calls=1, est_cost_usd=1, source="listed_price")
        return result

    def estimate(action, ctx, prices):
        return (ActionCost(llm_calls=1, est_cost_usd=1, source="listed_price")
                if action.kind == ActionKind.ANSWER else ActionCost())

    monkeypatch.setattr(InterventionExecutor, "feasible", feasible)
    monkeypatch.setattr(InterventionExecutor, "execute", priced)
    monkeypatch.setattr("app.trace.branching.estimate_action_cost", estimate, raising=False)
    budget = RunBudget(Budget(max_usd=3))
    first = collect(budget, "first")
    assert budget.state.spent_usd == 2
    with pytest.raises(BudgetExceeded) as exc:
        collect(budget, "second")
    assert budget.state.spent_usd == 3
    assert budget.state.spent_llm_calls == len(calls) == 3
    assert sum(t.total_cost.est_cost_usd for t in first + exc.value.trajectories) == 3


def test_admission_estimate_is_not_charged_and_snapshot_is_frozen():
    snapshot = {"model": {"cost_in": 1, "cost_out": 2}}
    budget = RunBudget(Budget(max_usd=2), snapshot)
    snapshot["model"]["cost_out"] = 999
    budget.admit(ActionCost(est_cost_usd=2, source="listed_price"))
    assert budget.state.spent_usd == 0
    budget.settle(ActionCost(est_cost_usd=1, source="listed_price"))
    assert budget.state.spent_usd == 1
    assert budget.model_prices["model"]["cost_out"] == 2


def test_actual_overrun_is_retained_and_stops_further_actions():
    budget = RunBudget(Budget(max_usd=1))
    budget.settle(ActionCost(est_cost_usd=2, source="listed_price"))
    with pytest.raises(BudgetExceeded):
        budget.check()
    assert budget.state.spent_usd == 2


def test_resample_admission_counts_all_capped_completions(monkeypatch):
    from app.trace.actions import InterventionContext
    from app.trace.adapter import resample_action
    from app.trace.budgeting import estimate_action_cost
    from types import SimpleNamespace

    monkeypatch.setattr("app.trace.budgeting.get_settings",
                        lambda: SimpleNamespace(force_mock=False))
    cfg = load_router_config()
    ctx = InterventionContext(question="q", cfg=cfg, small_id="gpt-4o")
    action = resample_action(cfg, "gpt-4o")
    prices = {"gpt-4o": {"cost_in": 1.0, "cost_out": 2.0}}
    estimate = estimate_action_cost(action, ctx, prices)
    assert estimate.llm_calls == cfg["proxy"]["resamples"]
    assert estimate.tokens_out == estimate.llm_calls * cfg["proxy"]["max_tokens"]
    assert estimate.est_cost_usd == (estimate.tokens_in + 2 * estimate.tokens_out) / 1e6
