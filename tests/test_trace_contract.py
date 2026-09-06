"""The trace contract's invariants (app/trace/contract.py).

These guard the properties the research plan depends on: canonical action keys
(so per-action value heads can group outcomes), content-addressed states, an
immutable budget, honest cost provenance, and validators that refuse a
structurally impossible trace rather than storing it.
"""
import json

import pytest
from pydantic import ValidationError

from app.schemas import SignalSet
from app.trace.contract import (
    Action,
    ActionCost,
    ActionKind,
    ActionOutcome,
    ActionScore,
    Budget,
    PolicyDecision,
    SchemaVersionError,
    StateSnapshot,
    TerminalRecord,
    TraceStep,
    Trajectory,
)

ANSWER = Action(kind=ActionKind.ANSWER, model_id="small-1")
STOP = Action(kind=ActionKind.STOP)


def _trajectory(**overrides):
    """A minimal two-state, one-step trajectory that satisfies every validator."""
    root = StateSnapshot.build("item-1", 0, "", SignalSet(), "small-1", [])
    result = StateSnapshot.build("item-1", 1, "Paris.", SignalSet(uncertainty=0.1),
                                 "small-1", [ANSWER.key])
    step = TraceStep(
        decision=PolicyDecision(state_id=root.state_id, feasible=[ANSWER, STOP],
                                chosen=ANSWER, policy_id="test"),
        outcome=ActionOutcome(action=ANSWER, parent_state_id=root.state_id,
                              result_state_id=result.state_id,
                              cost=ActionCost(tokens_in=5, tokens_out=7, llm_calls=1),
                              provider_label="mock"),
    )
    kwargs = dict(
        trajectory_id="t1", run_id="r1", item_id="item-1",
        root_state_id=root.state_id, states=[root, result], steps=[step],
        terminal=TerminalRecord(served_answer="Paris.", served_answer_hash=result.answer_hash),
        total_cost=ActionCost(tokens_in=5, tokens_out=7, llm_calls=1),
    )
    kwargs.update(overrides)
    return Trajectory(**kwargs)


# --------------------------------------------------------------------------- #
# Action
# --------------------------------------------------------------------------- #
def test_action_key_is_stable_across_param_ordering():
    a = Action(kind=ActionKind.RESAMPLE, model_id="m", params={"n": 3, "t": 0.7})
    b = Action(kind=ActionKind.RESAMPLE, model_id="m", params={"t": 0.7, "n": 3})
    assert a.key == b.key == "resample:m:n=3,t=0.7"


def test_action_key_separates_different_models():
    small = Action(kind=ActionKind.ANSWER, model_id="small-1")
    big = Action(kind=ActionKind.ANSWER, model_id="big-1")
    assert small.key != big.key


def test_action_is_frozen():
    with pytest.raises(ValidationError):
        ANSWER.model_id = "something-else"


# --------------------------------------------------------------------------- #
# StateSnapshot
# --------------------------------------------------------------------------- #
def test_state_id_is_deterministic_for_identical_content():
    a = StateSnapshot.build("i", 1, "hello", SignalSet(uncertainty=0.3), "m", ["answer:m"])
    b = StateSnapshot.build("i", 1, "hello", SignalSet(uncertainty=0.3), "m", ["answer:m"])
    assert a.state_id == b.state_id


@pytest.mark.parametrize("field,value", [
    ("answer", "different"),
    ("step", 2),
    ("model_id", "other"),
])
def test_state_id_changes_when_content_changes(field, value):
    base = dict(item_id="i", step=1, answer="hello", signals=SignalSet(),
                model_id="m", actions_taken=[])
    a = StateSnapshot.build(**base)
    b = StateSnapshot.build(**{**base, field: value})
    assert a.state_id != b.state_id


def test_state_id_changes_when_signals_change():
    base = dict(item_id="i", step=1, answer="hello", model_id="m", actions_taken=[])
    a = StateSnapshot.build(signals=SignalSet(uncertainty=0.1), **base)
    b = StateSnapshot.build(signals=SignalSet(uncertainty=0.9), **base)
    assert a.state_id != b.state_id


def test_state_copies_signals_so_later_router_mutation_cannot_rewrite_it():
    """The router mutates one SignalSet in place across a request.

    If the snapshot kept the reference, a signal computed three actions later
    would retroactively appear in a state recorded before it existed.
    """
    signals = SignalSet()
    state = StateSnapshot.build("i", 1, "a", signals, "m", [])
    signals.instability = 0.9
    assert state.signals.instability == 0.0


# --------------------------------------------------------------------------- #
# ActionCost
# --------------------------------------------------------------------------- #
def test_proxy_cost_may_not_carry_dollars():
    with pytest.raises(ValidationError):
        ActionCost(est_cost_usd=0.01, source="token_proxy")


def test_free_cost_may_not_carry_dollars():
    with pytest.raises(ValidationError):
        ActionCost(est_cost_usd=0.01, source="free")


@pytest.mark.parametrize("source", ["billed_api", "listed_price", "local_gpu", "mixed"])
def test_sources_that_can_carry_dollars_do(source):
    assert ActionCost(est_cost_usd=0.01, source=source).est_cost_usd == 0.01


def test_summing_different_provenance_yields_mixed_and_keeps_the_dollars():
    """A total spanning a priced call and a proxy-costed one is neither.

    Labelling it with the weaker source used to make the sum carry dollars under
    source="token_proxy", which the validator rejects -- so adding two
    legitimately costed actions raised instead of producing a total.
    """
    priced = ActionCost(tokens_in=10, est_cost_usd=0.5, source="listed_price")
    proxy = ActionCost(tokens_in=5, compute_units=100.0, source="token_proxy")
    total = priced + proxy
    assert total.tokens_in == 15
    assert total.est_cost_usd == 0.5
    assert total.source == "mixed"


def test_summing_with_free_keeps_the_real_source():
    priced = ActionCost(tokens_in=10, est_cost_usd=0.5, source="listed_price")
    assert (priced + ActionCost.zero()).source == "listed_price"
    assert (ActionCost.zero() + priced).source == "listed_price"


def test_summing_like_with_like_keeps_that_source():
    a = ActionCost(tokens_in=1, est_cost_usd=0.1, source="listed_price")
    assert (a + a).source == "listed_price"


def test_a_price_table_figure_is_not_called_an_invoice():
    """`listed_price` and `billed_api` are different claims and stay different.

    Everything the router computes is tokens x a configured price, which the
    run manifest snapshots. Only a figure read back from a provider billing
    record earns `billed_api`.
    """
    from app.trace.contract import DOLLAR_SOURCES

    assert "listed_price" in DOLLAR_SOURCES
    assert "billed_api" in DOLLAR_SOURCES
    assert "token_proxy" not in DOLLAR_SOURCES
    assert "free" not in DOLLAR_SOURCES


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #
def test_charge_returns_a_new_budget_and_leaves_the_original_untouched():
    b = Budget(max_usd=1.0)
    charged = b.charge(ActionCost(est_cost_usd=0.25, source="listed_price"))
    assert charged is not b
    assert charged.spent_usd == 0.25
    assert b.spent_usd == 0.0


@pytest.mark.parametrize("cap,cost_kwargs", [
    ({"max_usd": 0.10}, {"est_cost_usd": 0.20, "source": "listed_price"}),
    ({"max_llm_calls": 2}, {"llm_calls": 3}),
    ({"max_wall_ms": 100.0}, {"latency_ms": 250.0}),
    ({"max_compute_units": 50.0}, {"compute_units": 80.0}),
])
def test_can_afford_is_false_when_any_single_dimension_is_exceeded(cap, cost_kwargs):
    assert Budget(**cap).can_afford(ActionCost(**cost_kwargs)) is False


def test_can_afford_counts_escalations_separately_from_spend():
    b = Budget(max_escalations=1, escalations_used=1)
    assert b.can_afford(ActionCost(), escalation=True) is False
    assert b.can_afford(ActionCost(), escalation=False) is True


def test_uncapped_dimensions_never_block():
    assert Budget().can_afford(ActionCost(tokens_in=10**9, llm_calls=10**6)) is True


def test_remaining_reports_none_for_uncapped_dimensions():
    assert Budget(max_llm_calls=5).remaining()["usd"] is None


# --------------------------------------------------------------------------- #
# PolicyDecision
# --------------------------------------------------------------------------- #
def test_chosen_action_must_be_in_the_feasible_set():
    with pytest.raises(ValidationError):
        PolicyDecision(state_id="s", feasible=[STOP], chosen=ANSWER, policy_id="p")


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_propensity_must_lie_in_zero_to_one_exclusive_of_zero(bad):
    with pytest.raises(ValidationError):
        PolicyDecision(state_id="s", feasible=[ANSWER], chosen=ANSWER,
                       policy_id="p", propensity=bad)


def test_deterministic_policy_records_propensity_one():
    d = PolicyDecision(state_id="s", feasible=[ANSWER, STOP], chosen=ANSWER, policy_id="p")
    assert d.propensity == 1.0


def test_action_score_rejects_negative_sigma():
    with pytest.raises(ValidationError):
        ActionScore(delta_hat=0.1, sigma_hat=-1.0, lcb=0.0)


def test_trace_step_rejects_a_decision_about_a_different_state():
    root = StateSnapshot.build("i", 0, "", SignalSet(), "m", [])
    with pytest.raises(ValidationError):
        TraceStep(
            decision=PolicyDecision(state_id="somewhere-else", feasible=[ANSWER],
                                    chosen=ANSWER, policy_id="p"),
            outcome=ActionOutcome(action=ANSWER, parent_state_id=root.state_id,
                                  result_state_id="s2"),
        )


def test_trace_step_rejects_an_outcome_for_a_different_action():
    root = StateSnapshot.build("i", 0, "", SignalSet(), "m", [])
    with pytest.raises(ValidationError):
        TraceStep(
            decision=PolicyDecision(state_id=root.state_id, feasible=[ANSWER, STOP],
                                    chosen=ANSWER, policy_id="p"),
            outcome=ActionOutcome(action=STOP, parent_state_id=root.state_id,
                                  result_state_id="s2"),
        )


# --------------------------------------------------------------------------- #
# Trajectory
# --------------------------------------------------------------------------- #
def test_trajectory_json_round_trip_is_lossless():
    t = _trajectory()
    restored = Trajectory.model_validate_json(t.model_dump_json())
    assert restored == t


def test_unknown_schema_version_is_rejected():
    raw = json.loads(_trajectory().model_dump_json())
    raw["schema_version"] = "9.9.9"
    with pytest.raises((SchemaVersionError, ValidationError)):
        Trajectory.model_validate(raw)


def test_trajectory_rejects_a_step_hanging_off_an_unknown_state():
    raw = json.loads(_trajectory().model_dump_json())
    raw["steps"][0]["outcome"]["parent_state_id"] = "not-a-real-state"
    raw["steps"][0]["decision"]["state_id"] = "not-a-real-state"
    with pytest.raises(ValidationError):
        Trajectory.model_validate(raw)


def test_trajectory_rejects_referencing_a_state_it_does_not_store():
    raw = json.loads(_trajectory().model_dump_json())
    raw["states"] = raw["states"][:1]        # drop the result state
    with pytest.raises(ValidationError):
        Trajectory.model_validate(raw)


def test_summed_cost_adds_every_step():
    t = _trajectory()
    assert t.summed_cost().tokens_in == 5
    assert t.summed_cost().llm_calls == 1


def test_mock_outcomes_are_flagged_as_synthetic():
    assert _trajectory().has_synthetic_outcomes is True


def test_redaction_drops_answers_but_keeps_hashes():
    t = _trajectory().redacted()
    assert all(s.answer == "" for s in t.states)
    assert t.terminal.served_answer == ""
    # The hashes survive, so a redacted trace stays joinable and verifiable.
    assert t.terminal.served_answer_hash
    assert any(s.answer_hash for s in t.states)
