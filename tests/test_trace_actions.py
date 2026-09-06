"""The typed intervention executor (app/trace/actions.py).

The executor exists so a collector can run any feasible action from any state,
which the fixed router cannot do. What these tests protect is the honesty of its
reporting: an action with no integration must say so, a failure must not look
like a successful no-op, and availability must be recomputed from the state
rather than assumed.
"""
import asyncio

from app.config import load_router_config
from app.trace.actions import (
    DEFAULT_SYSTEM,
    InterventionContext,
    InterventionExecutor,
)
from app.trace.contract import Action, ActionKind, OutcomeStatus

CONFIDENT_Q = "What is the capital of France?"
MATH_Q = "what is 17 * 23 + 5?"


def _cfg():
    return load_router_config()


def _ctx(question=CONFIDENT_Q, **kw):
    cfg = _cfg()
    base = dict(question=question, cfg=cfg, small_id="llama-3.1-8b",
                big_id="gpt-4o", model_id="llama-3.1-8b")
    base.update(kw)
    return InterventionContext(**base)


def _run(coro):
    return asyncio.run(coro)


def _of(kind, ctx, executor):
    return next(a for a, _, _ in executor.feasible(ctx) if a.kind == kind)


# --------------------------------------------------------------------------- #
# Feasibility is computed, not assumed
# --------------------------------------------------------------------------- #
def test_specialist_model_is_always_unavailable_here():
    """There is no specialist integration in this repository.

    Listing it as feasible would put an action in every support denominator that
    nothing could ever execute, deflating coverage for real actions.
    """
    ex = InterventionExecutor(_cfg())
    entry = next((a, ok, why) for a, ok, why in ex.feasible(_ctx())
                 if a.kind == ActionKind.SPECIALIST_MODEL)
    _action, ok, why = entry
    assert ok is False
    assert "no specialist-model integration" in why


def test_executing_specialist_model_reports_unavailable_not_failure():
    ex = InterventionExecutor(_cfg())
    res = _run(ex.execute(Action(kind=ActionKind.SPECIALIST_MODEL), _ctx()))
    assert res.status is OutcomeStatus.UNAVAILABLE
    assert res.cost.llm_calls == 0


def test_tool_is_available_only_when_the_calculator_can_parse_the_question():
    ex = InterventionExecutor(_cfg())
    _a, math_ok, _ = next((a, ok, w) for a, ok, w in ex.feasible(_ctx(MATH_Q))
                          if a.kind == ActionKind.TOOL)
    _a, prose_ok, why = next((a, ok, w) for a, ok, w in ex.feasible(_ctx(CONFIDENT_Q))
                             if a.kind == ActionKind.TOOL)
    assert math_ok is True
    assert prose_ok is False
    assert "cannot parse" in why


def test_actions_needing_a_candidate_are_unavailable_before_one_exists():
    """RESAMPLE/SELF_CHECK/VERIFY/STOP all operate on an existing answer."""
    ex = InterventionExecutor(_cfg())
    at_root = {a.kind: ok for a, ok, _ in ex.feasible(_ctx(answer=""))}
    for kind in (ActionKind.RESAMPLE, ActionKind.SELF_CHECK,
                 ActionKind.VERIFY, ActionKind.STOP):
        assert at_root[kind] is False, f"{kind} should need a candidate answer"
    assert at_root[ActionKind.ANSWER] is True


def test_already_taken_actions_drop_out_of_the_feasible_set():
    ex = InterventionExecutor(_cfg())
    ctx = _ctx(answer="Paris.")
    retrieve = _of(ActionKind.RETRIEVE, ctx, ex)
    after = {a.key: ok for a, ok, _ in ex.feasible(ctx, taken=[retrieve.key])}
    assert after[retrieve.key] is False


def test_stronger_model_unavailable_without_a_distinct_target():
    ex = InterventionExecutor(_cfg())
    same = {a.kind: (ok, why) for a, ok, why in
            ex.feasible(_ctx(answer="x", big_id="llama-3.1-8b"))}
    ok, why = same[ActionKind.STRONGER_MODEL]
    assert ok is False
    assert "same model" in why

    none = {a.kind: (ok, why) for a, ok, why in ex.feasible(_ctx(answer="x", big_id=None))}
    assert none[ActionKind.STRONGER_MODEL][0] is False


# --------------------------------------------------------------------------- #
# Execution semantics
# --------------------------------------------------------------------------- #
def test_answer_produces_a_candidate_and_an_uncertainty_signal():
    ex = InterventionExecutor(_cfg())
    ctx = _ctx()
    res = _run(ex.execute(_of(ActionKind.ANSWER, ctx, ex), ctx))
    assert res.status is OutcomeStatus.CHOSEN
    assert res.answer.strip()
    assert res.cost.llm_calls == 1
    assert res.detail["prompt_hash"]


def test_retrieval_is_free_and_touches_no_model():
    ex = InterventionExecutor(_cfg())
    ctx = _ctx(answer="Paris.")
    res = _run(ex.execute(_of(ActionKind.RETRIEVE, ctx, ex), ctx))
    assert res.cost.llm_calls == 0
    assert res.cost.est_cost_usd == 0.0
    assert res.provider_label == "retrieval"


def test_self_check_is_billed_separately_from_resample():
    """Their costs must not be conflated; each is its own LLM call."""
    ex = InterventionExecutor(_cfg())
    ctx = _ctx(answer="Paris.")
    check = _run(ex.execute(_of(ActionKind.SELF_CHECK, ctx, ex), ctx))
    resample = _run(ex.execute(_of(ActionKind.RESAMPLE, ctx, ex), ctx))
    assert check.cost.llm_calls == 1
    assert resample.cost.llm_calls == _cfg()["proxy"]["resamples"]


def test_terminal_actions_are_free_and_marked_terminal():
    ex = InterventionExecutor(_cfg())
    ctx = _ctx(answer="Paris.")
    for kind in (ActionKind.STOP, ActionKind.ABSTAIN):
        res = _run(ex.execute(_of(kind, ctx, ex), ctx))
        assert res.terminal is True
        assert res.cost.llm_calls == 0


def test_abstain_records_the_hash_of_the_answer_it_withheld():
    """The withheld candidate must stay recoverable, or repair analysis loses it."""
    ex = InterventionExecutor(_cfg())
    ctx = _ctx(answer="Paris.")
    res = _run(ex.execute(_of(ActionKind.ABSTAIN, ctx, ex), ctx))
    assert res.detail["withheld_answer_hash"]


def test_next_context_carries_the_result_forward():
    ex = InterventionExecutor(_cfg())
    ctx = _ctx()
    res = _run(ex.execute(_of(ActionKind.ANSWER, ctx, ex), ctx))
    nxt = res.next_context(ctx)
    assert nxt.answer == res.answer
    assert nxt.question == ctx.question
    assert nxt.signals.uncertainty == res.signals.uncertainty


def test_executor_does_not_mutate_the_context_it_was_given():
    """Branches share a parent context; mutating it would corrupt the siblings."""
    ex = InterventionExecutor(_cfg())
    ctx = _ctx(answer="Paris.")
    before_answer, before_u = ctx.answer, ctx.signals.uncertainty
    _run(ex.execute(_of(ActionKind.RETRIEVE, ctx, ex), ctx))
    _run(ex.execute(_of(ActionKind.SELF_CHECK, ctx, ex), ctx))
    assert ctx.answer == before_answer
    assert ctx.signals.uncertainty == before_u
    assert ctx.evidence == []


def test_default_system_prompt_is_used_when_none_is_supplied():
    assert _ctx().messages()[0]["content"] == DEFAULT_SYSTEM
