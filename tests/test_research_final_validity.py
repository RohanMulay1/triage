import asyncio

from app.trace.actions import verification_question
from app.trace.calibration import calibration_choice_change
from app.trace.contract import ActionKind
from app.trace.policies import RandomFeasiblePolicy
from test_trace_policies import _ctx, _ex
from test_trace_measurement_repairs import collect


def test_verify_consumes_prior_information_without_changing_root_prompt(monkeypatch):
    from app.trace import actions
    ctx = _ctx()
    assert verification_question(ctx) == ctx.question
    ctx.samples = ["A sampled draft"]
    ctx.signals.detail["self_check"] = {"probe": "CONTRADICTORY: arithmetic error"}
    seen = []
    async def verify(client, question, candidate, evidence):
        seen.append(question)
        return {"pass": True, "revised": candidate, "grounded": False, "reason": "test"}
    monkeypatch.setattr(actions, "verify", verify)
    executor = _ex()
    action = next(a for a in executor.available_actions(ctx, []) if a.kind == ActionKind.VERIFY)
    asyncio.run(executor.execute(action, ctx))
    assert "A sampled draft" in seen[0] and "arithmetic error" in seen[0]
    assert "not ground truth" in seen[0]


def test_self_check_observation_is_retained_for_continuation():
    ctx, executor = _ctx(), _ex()
    action = next(a for a in executor.available_actions(ctx, []) if a.kind == ActionKind.SELF_CHECK)
    result = asyncio.run(executor.execute(action, ctx))
    assert "self_check" in result.signals.detail
    assert "self_check" in verification_question(result.next_context(ctx))


def test_served_probability_is_separate_from_exhaustive_inclusion():
    traces = collect(served_policy=RandomFeasiblePolicy())
    served = next(t for t in traces if t.branch_id != "prefix" and
                  t.steps[0].decision.rationale.get("served"))
    decision = served.steps[0].decision
    assert decision.propensity == 1.0
    assert 0 < decision.rationale["behavior_propensity"] < 1.0


def _fit(raw, calibrated, n=20):
    return {"status": "OK", "test_item_ids": [str(i) for i in range(n)],
            "raw_gain": [raw]*n, "calibrated_gain": [calibrated]*n}


def test_calibration_change_reports_scope_and_paired_interval():
    result = calibration_choice_change({"verify": _fit(.4, -.1),
        "retrieve -> verify": _fit(.8, .9), "answer": {"status": "REFUSED"}})
    assert result["status"] == "OK" and result["n"] == 20
    assert result["ci95"]["point"] == 1.0
    assert result["actions"] == ["verify"]
    assert set(result["excluded_actions"]) == {"answer", "retrieve -> verify"}
    assert not result["performance_claim"]


def test_calibration_change_refuses_small_or_unfitted_sets():
    for fits in ({}, {"verify": _fit(.4, -.1, n=3)}, {"retrieve -> verify": _fit(.4, -.1)}):
        result = calibration_choice_change(fits)
        assert result["status"] == "REFUSED"
        assert result["ci95"]["point"] is None


def test_failed_auxiliary_comparator_does_not_discard_balanced_fanout(monkeypatch):
    from app.trace import heuristic_gain as hg
    from test_trace_heuristic_gain import ScriptedClient
    monkeypatch.setattr(hg, 'LLMClient', ScriptedClient)
    monkeypatch.setattr(ScriptedClient, 'mode', 'malformed')
    traces = collect(served_policy=RandomFeasiblePolicy(), scoring_policy=hg.HeuristicGainPolicy())
    assert traces[0].steps[-1].outcome.error
    assert traces[0].terminal.label is None
    assert len(traces) > 1
    assert any(t.terminal.label is not None for t in traces[1:])


def test_request_budget_refusal_retains_prefix_and_failed_branch(monkeypatch):
    import pytest
    from app.trace.actions import InterventionExecutor
    from app.trace.budgeting import BudgetExceeded
    from app.trace.contract import OutcomeStatus
    original = InterventionExecutor.execute
    async def execute(self, action, ctx):
        result = await original(self, action, ctx)
        if action.kind == ActionKind.RESAMPLE:
            result.status = OutcomeStatus.FAILED
            result.detail['budget_refused'] = True
        return result
    monkeypatch.setattr(InterventionExecutor, 'execute', execute)
    with pytest.raises(BudgetExceeded) as caught:
        collect(depth=2)
    traces = caught.value.trajectories
    assert traces[0].branch_id == 'prefix'
    assert traces[-1].steps[0].outcome.detail['budget_refused']
    assert traces[-1].terminal.label is None
