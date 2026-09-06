"""Behaviour policies (app/trace/policies.py).

What matters here is the propensity. A policy that reports the wrong probability
for the action it took produces data that looks usable and yields biased
estimates, which is worse than data that is visibly unusable.
"""
import random

import pytest

from app.config import load_router_config
from app.schemas import SignalSet
from app.trace.actions import InterventionContext, InterventionExecutor
from app.trace.contract import ActionKind
from app.trace.policies import (
    POLICIES,
    BalancedExplorationPolicy,
    EpsilonGreedyPolicy,
    FixedCascadePolicy,
    PromptOnlyPolicy,
    RandomFeasiblePolicy,
    ResponseRiskThresholdPolicy,
    build_policy,
)


def _cfg():
    return load_router_config()


def _ctx(answer="Paris.", signals=None, question="What is the capital of France?"):
    return InterventionContext(
        question=question, cfg=_cfg(), small_id="llama-3.1-8b", big_id="gpt-4o",
        model_id="llama-3.1-8b", answer=answer, signals=signals or SignalSet(),
    )


def _ex():
    return InterventionExecutor(_cfg())


# --------------------------------------------------------------------------- #
# Propensities
# --------------------------------------------------------------------------- #
def test_random_policy_reports_uniform_propensity_over_available_actions():
    ex, ctx = _ex(), _ctx()
    choice = RandomFeasiblePolicy().choose(ctx, ex, [], random.Random(0))
    assert choice.propensity == pytest.approx(1.0 / len(choice.feasible))
    assert choice.action.key in {a.key for a in choice.feasible}


def test_random_policy_gives_every_available_action_positive_probability():
    """Positivity by construction is the whole reason this policy exists."""
    ex, ctx = _ex(), _ctx()
    seen = set()
    for seed in range(200):
        seen.add(RandomFeasiblePolicy().choose(ctx, ex, [], random.Random(seed)).action.key)
    assert seen == {a.key for a in ex.available_actions(ctx, [])}


def test_epsilon_greedy_propensity_is_the_mixture_not_the_base_policy():
    """eps/|A| for an explored action, eps/|A| + (1-eps) for the greedy one."""
    ex, ctx = _ex(), _ctx()
    base = ResponseRiskThresholdPolicy(0.55)
    policy = EpsilonGreedyPolicy(base, epsilon=0.5)
    n = len(ex.available_actions(ctx, []))
    greedy = base.choose(ctx, ex, [], random.Random(0)).action.key

    for seed in range(60):
        choice = policy.choose(ctx, ex, [], random.Random(seed))
        expected = 0.5 / n + (0.5 if choice.action.key == greedy else 0.0)
        assert choice.propensity == pytest.approx(expected)


def test_epsilon_must_be_a_probability():
    with pytest.raises(ValueError):
        EpsilonGreedyPolicy(RandomFeasiblePolicy(), epsilon=0.0)
    with pytest.raises(ValueError):
        EpsilonGreedyPolicy(RandomFeasiblePolicy(), epsilon=1.5)


def test_deterministic_policies_report_propensity_one():
    """Honest, and exactly why they cannot support off-policy estimation."""
    ex, ctx = _ex(), _ctx()
    for policy in (FixedCascadePolicy(), PromptOnlyPolicy(), ResponseRiskThresholdPolicy()):
        choice = policy.choose(ctx, ex, [], random.Random(0))
        assert choice.propensity == 1.0


def test_balanced_exploration_equalises_action_counts():
    ex, ctx = _ex(), _ctx()
    policy = BalancedExplorationPolicy()
    for seed in range(60):
        policy.choose(ctx, ex, [], random.Random(seed))
    counts = list(policy.counts.values())
    # Every available action should have been visited a comparable number of
    # times; uniform sampling alone does not guarantee this.
    assert max(counts) - min(counts) <= 1
    assert len(policy.counts) == len(ex.available_actions(ctx, []))


# --------------------------------------------------------------------------- #
# Deterministic replay
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(POLICIES))
def test_same_seed_replays_the_same_choice(name):
    """A recorded exploration_seed must reproduce the recorded action."""
    ex, ctx = _ex(), _ctx()
    first = build_policy(name, _cfg()).choose(ctx, ex, [], random.Random(7))
    second = build_policy(name, _cfg()).choose(ctx, ex, [], random.Random(7))
    assert first.action.key == second.action.key
    assert first.propensity == second.propensity


def test_unknown_policy_name_is_refused():
    with pytest.raises(KeyError):
        build_policy("not_a_policy", _cfg())


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #
def test_policies_answer_first_when_there_is_no_candidate():
    ex, ctx = _ex(), _ctx(answer="")
    for policy in (FixedCascadePolicy(), PromptOnlyPolicy(), ResponseRiskThresholdPolicy()):
        assert policy.choose(ctx, ex, [], random.Random(0)).action.kind is ActionKind.ANSWER


def test_response_risk_policy_escalates_only_above_its_threshold():
    ex = _ex()
    calm = _ctx(signals=SignalSet())
    alarmed = _ctx(signals=SignalSet(uncertainty=1.0, instability=1.0, contradiction=1.0,
                                     retrieval_disagreement=1.0, evidence_sufficiency=0.0))
    policy = ResponseRiskThresholdPolicy(0.55)
    assert policy.choose(calm, ex, [], random.Random(0)).action.kind is ActionKind.STOP
    assert policy.choose(alarmed, ex, [], random.Random(0)).action.kind is \
        ActionKind.STRONGER_MODEL


def test_prompt_only_policy_ignores_the_response_entirely():
    """The RouteLLM-shaped control: identical prompt, opposite response state."""
    ex = _ex()
    policy = PromptOnlyPolicy()
    calm = _ctx(answer="", signals=SignalSet())
    alarmed = _ctx(answer="", signals=SignalSet(uncertainty=1.0, instability=1.0,
                                                contradiction=1.0))
    a = policy.choose(calm, ex, [], random.Random(0)).action.key
    b = policy.choose(alarmed, ex, [], random.Random(0)).action.key
    assert a == b


def test_every_registered_policy_returns_an_available_action():
    ex, ctx = _ex(), _ctx()
    available = {a.key for a in ex.available_actions(ctx, [])}
    for name in POLICIES:
        choice = build_policy(name, _cfg()).choose(ctx, ex, [], random.Random(3))
        assert choice.action.key in available, name
