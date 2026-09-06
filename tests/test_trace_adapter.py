"""The legacy-router bridge (app/trace/adapter.py) on the mock provider.

The load-bearing test here is the regression guard: routing must be identical
with and without a recorder. The committed receipts in data/ were produced by
the untraced behaviour, so tracing that changed a decision would silently
invalidate every number the project has published.
"""
import asyncio

import pytest

from app.config import load_router_config
from app.providers.registry import get_registry
from app.router.router import route_and_answer
from app.schemas import ChatRequest
from app.trace import store
from app.trace.adapter import TraceRecorder, cost_delta, feasible_actions, snapshot_cost
from app.trace.contract import ActionKind
from app.trace.support import action_support

CONFIDENT_Q = "What is the capital of France?"
UNCERTAIN_Q = "What is the definitive, objective meaning of life?"
MATH_Q = "what is 17 * 23 + 5?"


def _cfg():
    return load_router_config()


def _traced(message: str, item_id: str = "item-0", run_id: str = "adapter-test", **req_kwargs):
    """Route one request with a recorder attached and return (response, trajectory)."""
    rec = TraceRecorder(run_id, item_id, _cfg(), dataset="unit", split="pilot")
    resp, _telem = asyncio.run(route_and_answer(ChatRequest(message=message, **req_kwargs),
                                                recorder=rec))
    return resp, rec.finish(resp)


def _untraced(message: str, **req_kwargs):
    resp, _telem = asyncio.run(route_and_answer(ChatRequest(message=message, **req_kwargs)))
    return resp


# --------------------------------------------------------------------------- #
# Regression guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question", [CONFIDENT_Q, UNCERTAIN_Q, MATH_Q])
def test_tracing_does_not_change_the_routing_decision(question):
    """recorder=None and recorder=<recorder> must route identically."""
    traced, _traj = _traced(question)
    plain = _untraced(question)
    assert (traced.route.tier, traced.route.tier_name) == (plain.route.tier, plain.route.tier_name)
    assert traced.route.escalated == plain.route.escalated
    assert traced.route.abstained == plain.route.abstained
    assert traced.route.retrieved == plain.route.retrieved
    assert traced.route.verified == plain.route.verified
    assert traced.status == plain.status
    assert traced.model == plain.model
    assert traced.cost.llm_calls == plain.cost.llm_calls


# --------------------------------------------------------------------------- #
# Shape of the recorded program
# --------------------------------------------------------------------------- #
def test_calculator_route_is_one_free_tool_action_then_stop():
    resp, traj = _traced(MATH_Q)
    kinds = [s.outcome.action.kind for s in traj.steps]
    assert kinds == [ActionKind.TOOL, ActionKind.STOP]
    assert traj.summed_cost().est_cost_usd == 0.0
    assert traj.summed_cost().llm_calls == 0
    assert resp.route.tier_name == "tool"


def test_single_pass_records_an_answer_then_stop():
    _resp, traj = _traced(CONFIDENT_Q)
    kinds = [s.outcome.action.kind for s in traj.steps]
    assert kinds[0] == ActionKind.ANSWER
    assert kinds[-1] == ActionKind.STOP


def test_deep_signal_path_records_resample_and_self_check_separately():
    """The self-check probe is its own billed call, so it is its own action.

    Folding it into RESAMPLE or VERIFY would attribute its cost to an
    intervention that did not incur it.
    """
    _resp, traj = _traced(UNCERTAIN_Q)
    kinds = [s.outcome.action.kind for s in traj.steps]
    assert ActionKind.RESAMPLE in kinds
    assert ActionKind.SELF_CHECK in kinds
    resample = next(s for s in traj.steps if s.outcome.action.kind == ActionKind.RESAMPLE)
    self_check = next(s for s in traj.steps if s.outcome.action.kind == ActionKind.SELF_CHECK)
    assert resample.outcome.cost.llm_calls >= 1
    assert self_check.outcome.cost.llm_calls == 1


def test_retrieval_is_recorded_as_a_free_non_model_action():
    _resp, traj = _traced(UNCERTAIN_Q)
    retrieves = [s for s in traj.steps if s.outcome.action.kind == ActionKind.RETRIEVE]
    assert retrieves, "the uncertain question should have triggered retrieval"
    for s in retrieves:
        assert s.outcome.cost.llm_calls == 0
        assert s.outcome.provider_label == "retrieval"


def test_state_chain_is_contiguous():
    _resp, traj = _traced(UNCERTAIN_Q)
    previous = traj.root_state_id
    for step in traj.steps:
        assert step.outcome.parent_state_id == previous
        previous = step.outcome.result_state_id
    assert traj.states[0].state_id == traj.root_state_id
    assert len(traj.states) == len(traj.steps) + 1


def test_every_referenced_state_is_stored_with_its_features():
    _resp, traj = _traced(UNCERTAIN_Q)
    for step in traj.steps:
        assert traj.state(step.outcome.parent_state_id) is not None
        assert traj.state(step.outcome.result_state_id) is not None
    # The state features a value model would condition on are actually present.
    assert "prefilter_route" in traj.states[-1].prompt_features


# --------------------------------------------------------------------------- #
# Cost accounting
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question", [CONFIDENT_Q, UNCERTAIN_Q, MATH_Q])
def test_per_step_costs_sum_to_the_responses_own_accounting(question):
    """No token is lost between actions, and none is counted twice."""
    resp, traj = _traced(question)
    summed = traj.summed_cost()
    assert summed.tokens_in == resp.cost.tokens_in
    assert summed.tokens_out == resp.cost.tokens_out
    assert summed.llm_calls == resp.cost.llm_calls


def test_trajectory_total_cost_matches_its_steps():
    _resp, traj = _traced(UNCERTAIN_Q)
    assert traj.total_cost == traj.summed_cost()


def test_mock_outcomes_carry_no_dollars():
    """The price table would happily invoice synthetic tokens; it must not."""
    _resp, traj = _traced(CONFIDENT_Q)
    model_steps = [s for s in traj.steps if s.outcome.provider_label == "mock"]
    assert model_steps, "conftest forces the mock provider"
    for s in model_steps:
        assert s.outcome.cost.est_cost_usd == 0.0
        assert s.outcome.cost.source == "free"


def test_cost_delta_labels_live_provider_dollars_as_listed_price_not_an_invoice():
    """The router prices tokens from config/models.yaml, it does not read a bill."""
    from app.schemas import CostMetrics

    before = CostMetrics(tokens_in=10, tokens_out=5, llm_calls=1, est_cost_usd=0.001)
    after = CostMetrics(tokens_in=30, tokens_out=15, llm_calls=2, est_cost_usd=0.004)
    delta = cost_delta(before, after, "openai")
    assert (delta.tokens_in, delta.tokens_out, delta.llm_calls) == (20, 10, 1)
    assert delta.est_cost_usd == pytest.approx(0.003)
    assert delta.source == "listed_price"
    assert delta.source != "billed_api"
    assert delta.compute_units == pytest.approx(20 + 10 * 10)


def test_snapshot_cost_is_a_copy_not_a_live_view():
    class _FakeClient:
        def __init__(self):
            from app.schemas import CostMetrics
            self.cost = CostMetrics()

    c = _FakeClient()
    mark = snapshot_cost(c)
    c.cost.tokens_in += 42
    assert mark.tokens_in == 0


# --------------------------------------------------------------------------- #
# Escalation
# --------------------------------------------------------------------------- #
def test_escalation_records_one_stronger_model_action_priced_at_the_whole_pass(monkeypatch):
    """Force a live-looking provider so the escalation branch is reachable.

    Under the plain mock setup the router refuses to escalate into a mock target
    (see tests/test_router_integration.py), so the STRONGER_MODEL path cannot be
    exercised without a stand-in provider label.
    """
    registry = get_registry()
    real_resolve = registry.resolve

    def fake_resolve(model_id):
        adapter, provider_model, _label = real_resolve(model_id)
        return adapter, provider_model, "fake-live"

    monkeypatch.setattr(registry, "resolve", fake_resolve)
    # Drop the learned escalation bar to zero so the branch fires on any input:
    # this test is about how escalation is *recorded*, not about when the
    # heuristic decides to escalate.
    monkeypatch.setattr("app.router.router.online.current_risk_high",
                        lambda *_a, **_k: 0.0)

    resp, traj = _traced(UNCERTAIN_Q, item_id="esc-0", model="llama-3.1-8b",
                         escalate_to="gpt-4o")
    kinds = [s.outcome.action.kind for s in traj.steps]
    assert resp.route.escalated is True
    assert ActionKind.STRONGER_MODEL in kinds
    assert kinds.index(ActionKind.ANSWER) < kinds.index(ActionKind.STRONGER_MODEL)
    esc = next(s for s in traj.steps if s.outcome.action.kind == ActionKind.STRONGER_MODEL)
    assert esc.outcome.action.model_id == "gpt-4o"
    assert esc.outcome.cost.llm_calls >= 1
    # Cost conservation must still hold across two clients.
    assert traj.summed_cost().tokens_in == resp.cost.tokens_in
    assert traj.summed_cost().llm_calls == resp.cost.llm_calls


# --------------------------------------------------------------------------- #
# Feasible sets
# --------------------------------------------------------------------------- #
def test_feasible_set_names_the_acting_model_so_keys_match_what_is_recorded():
    """A feasible action must produce the same key as the recorded chosen one.

    RESAMPLE, SELF_CHECK and VERIFY run on the model holding the answer and
    carry it in their key. If the feasible set omitted it, the same intervention
    would split into two action keys and every coverage number would be wrong.
    """
    cfg = _cfg()
    keys = {a.key for a in feasible_actions(cfg, stage="post", small_id="llama-3.1-8b",
                                            acting_model="llama-3.1-8b")}
    assert "resample:llama-3.1-8b:n=3,t=0.7" in keys
    assert "self_check:llama-3.1-8b" in keys
    assert "verify:llama-3.1-8b:max_rounds=1" in keys


def test_every_recorded_action_was_listed_as_feasible():
    _resp, traj = _traced(UNCERTAIN_Q)
    for step in traj.steps:
        assert step.decision.chosen.key in {a.key for a in step.decision.feasible}


def test_heuristic_traces_have_no_counterfactual_support():
    """The checkpoint's honest finding, asserted end to end.

    The deterministic router leaves feasible actions it never chose, so a policy
    fitted on these traces would be extrapolating on those actions.
    """
    run_id = store.new_run_id("heuristic-support")
    trajectories = []
    for i, q in enumerate([CONFIDENT_Q, UNCERTAIN_Q, MATH_Q]):
        _resp, traj = _traced(q, item_id=f"item-{i}", run_id=run_id)
        trajectories.append(traj)
    sup = action_support(trajectories)

    assert sup["unsupported_actions"], "expected feasible-but-never-chosen actions"
    assert sup["off_policy_ready"] is False
    assert all(v["min_propensity"] == 1.0
               for v in sup["per_action"].values() if v["n_taken"])
