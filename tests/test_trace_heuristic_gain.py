import asyncio
import json
import random
from types import SimpleNamespace

import pytest

from app.providers.base import GenResult
from app.schemas import CostMetrics
from app.trace import heuristic_gain as hg
from app.trace.branching import RunBudget, BudgetExceeded
from app.trace.contract import Budget, ActionKind, OutcomeStatus
from test_trace_policies import _ctx, _ex
from test_trace_measurement_repairs import collect


class ScriptedClient:
    calls = 0
    mode = "valid"

    def __init__(self, model):
        self.cost = CostMetrics()
        self.provider_label = "mock"

    async def generate(self, messages, **kwargs):
        type(self).calls += 1
        self.cost.llm_calls += 1
        self.cost.tokens_in += 25
        self.cost.tokens_out += 15
        state = json.loads(messages[-1]["content"])
        payload = {a["key"]: {"expected_gain": 0.9 if a["kind"] == "verify" else 0.1,
                              "uncertainty": 0.2} for a in state["actions"]}
        if self.mode == "missing":
            payload.pop(next(iter(payload)))
        if self.mode == "nan":
            next(iter(payload.values()))["expected_gain"] = float("nan")
        text = "not json" if self.mode == "malformed" else json.dumps(payload)
        return GenResult(text=text, provider="mock", model="unit", latency_ms=7)


@pytest.fixture
def scorer(monkeypatch):
    ScriptedClient.calls, ScriptedClient.mode = 0, "valid"
    monkeypatch.setattr(hg, "LLMClient", ScriptedClient)
    return ScriptedClient


def test_gain_equation_and_conditional_propensity(scorer):
    policy, ctx, executor = hg.HeuristicGainPolicy(), _ctx(), _ex()
    budget = RunBudget(Budget(max_llm_calls=1))
    action, result = asyncio.run(policy.prepare(ctx, executor, [], budget))
    choice = policy.choose(ctx, executor, [], random.Random(0))
    assert choice.action.kind == ActionKind.VERIFY
    row = choice.rationale['scores'][choice.action.key]
    assert row['utility'] == pytest.approx(.9 - .3/3 - .2*.2)
    assert choice.propensity == 1
    assert result.cost.llm_calls == budget.state.spent_llm_calls == 1
    assert result.cost.latency_ms == 7
    assert action.params['purpose'] == 'heuristic_gain_scoring'
    ctx.answer = 'a different state'
    with pytest.raises(ValueError, match='exact state'):
        policy.choose(ctx, executor, [], random.Random(0))


@pytest.mark.parametrize('mode', ['missing', 'nan', 'malformed'])
def test_bad_scores_refuse_without_default_gain(scorer, mode):
    scorer.mode = mode
    policy, ctx, executor = hg.HeuristicGainPolicy(), _ctx(), _ex()
    budget = RunBudget()
    _, result = asyncio.run(policy.prepare(ctx, executor, [], budget))
    assert result.status == OutcomeStatus.ATTEMPTED
    assert budget.state.spent_llm_calls == 1
    with pytest.raises(ValueError):
        policy.choose(ctx, executor, [], random.Random(0))


def test_scoring_admission_precedes_provider_call(scorer):
    budget = RunBudget(Budget(max_llm_calls=1))
    policy, ctx, executor = hg.HeuristicGainPolicy(), _ctx(), _ex()
    asyncio.run(policy.prepare(ctx, executor, [], budget))
    with pytest.raises(BudgetExceeded):
        asyncio.run(policy.prepare(ctx, executor, [], budget))
    assert scorer.calls == 1


def test_mock_fallback_scores_cannot_be_used(scorer, monkeypatch):
    monkeypatch.setattr(hg, 'get_settings', lambda: SimpleNamespace(force_mock=False))
    _, result = asyncio.run(hg.HeuristicGainPolicy().prepare(_ctx(), _ex(), [], RunBudget()))
    assert result.status == OutcomeStatus.SYNTHETIC_FALLBACK


def test_fanout_records_overhead_once_and_selects_comparator_branch(scorer):
    budget = RunBudget()
    traces = collect(served_policy=hg.HeuristicGainPolicy(), budget=budget, depth=2)
    prefix = traces[0]
    assert len(prefix.steps) == 2
    assert prefix.steps[-1].decision.rationale['policy_overhead']
    assert sum(t.total_cost.llm_calls for t in traces) == budget.state.spent_llm_calls
    chosen = [t for t in traces[1:] if t.steps[0].outcome.status == OutcomeStatus.CHOSEN]
    assert len(chosen) == 1 and chosen[0].steps[0].decision.chosen.kind == ActionKind.VERIFY
    assert chosen[0].steps[0].decision.rationale['behavior_choice']['uncalibrated']


def test_bad_scoring_stops_with_unlabelled_costed_prefix(scorer):
    scorer.mode = 'malformed'
    traces = collect(served_policy=hg.HeuristicGainPolicy())
    assert len(traces) == 1 and traces[0].terminal.label is None
    assert traces[0].total_cost.llm_calls == 2


@pytest.mark.parametrize('weights', [{'lambda_cost': float('inf')}, {'max_steps': 0}])
def test_invalid_utility_configuration_refuses(weights):
    with pytest.raises(ValueError):
        hg.HeuristicGainPolicy(**weights)


def test_exhausted_step_budget_forces_stop_after_scoring(scorer):
    ctx, executor = _ctx(), _ex()
    taken = ['initial_answer', 'a', 'b', 'c']
    policy = hg.HeuristicGainPolicy(max_steps=3)
    asyncio.run(policy.prepare(ctx, executor, taken, RunBudget()))
    assert policy.choose(ctx, executor, taken, random.Random(0)).action.kind == ActionKind.STOP


def test_unknown_scoring_usage_stops_entire_collection(scorer, monkeypatch):
    async def broken(*args, **kwargs):
        raise RuntimeError('transport closed before usage')
    monkeypatch.setattr(ScriptedClient, 'generate', broken)
    with pytest.raises(BudgetExceeded, match='usage unknown') as exc:
        collect(served_policy=hg.HeuristicGainPolicy())
    assert len(exc.value.trajectories) == 1
    assert exc.value.trajectories[0].terminal.label is None
    assert exc.value.trajectories[0].steps[-1].outcome.detail['unknown_usage']
