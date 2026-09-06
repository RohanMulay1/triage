"""Legacy Triage adapter: turn a heuristic ``route_and_answer`` run into a Trajectory.

The existing router is a fixed program, not a policy over actions. This adapter
re-describes that program in the trace vocabulary without changing a single
routing decision: the router threads an optional ``recorder`` through, the
recorder observes, and with ``recorder=None`` the code path is byte-identical to
before. That matters because the committed receipts in ``data/`` were produced
by the current behaviour and must stay reproducible.

The mapping:

    main pass                    -> ANSWER (or TOOL when the calculator solved it)
    N resamples                  -> one RESAMPLE action (the intervention, not
                                    its realisation - the realised sample count
                                    goes in the decision rationale)
    contradiction probe          -> SELF_CHECK
    evidence lookup              -> RETRIEVE
    grounding pass               -> VERIFY
    trade up to the big model    -> STRONGER_MODEL
    serve / PENDING_REVIEW       -> STOP / ABSTAIN

One deliberate coarsening: STRONGER_MODEL folds the big model's whole assessment
(its own main pass, and any retrieve/verify it runs) into a single action with a
single cost. The intervention being valued is "trade up", so its price is the
whole big-model pass; splitting it would make the sub-actions look free.

Every decision records ``propensity = 1.0``, because the heuristic is
deterministic. That is honest, and it is exactly what ``app/trace/support.py``
uses to show these traces carry no information about unchosen actions.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from ..schemas import ChatResponse, CostMetrics, SignalSet
from .contract import (
    Action,
    ActionCost,
    ActionKind,
    ActionOutcome,
    Budget,
    PolicyDecision,
    StateSnapshot,
    TerminalRecord,
    TraceStep,
    Trajectory,
)

_SIGNAL_FIELDS = (
    "uncertainty", "instability", "contradiction",
    "retrieval_disagreement", "evidence_sufficiency",
)


# --------------------------------------------------------------------------- #
# Action builders - the single source of an action's canonical key
# --------------------------------------------------------------------------- #
# The action recorded as *chosen* and the same action listed as *feasible* must
# produce identical keys, or PolicyDecision rejects the pair. Building both
# through these helpers is what guarantees that.
def answer_action(model_id: str, cfg: dict) -> Action:
    return Action(kind=ActionKind.ANSWER, model_id=model_id,
                  params={"t": cfg["proxy"]["base_temperature"]})


def resample_action(cfg: dict, model_id: Optional[str] = None) -> Action:
    p = cfg["proxy"]
    return Action(kind=ActionKind.RESAMPLE, model_id=model_id,
                  params={"n": p["resamples"], "t": p["resample_temperature"]})


def self_check_action(model_id: Optional[str] = None) -> Action:
    return Action(kind=ActionKind.SELF_CHECK, model_id=model_id)


def retrieve_action(cfg: dict) -> Action:
    rc = cfg["retrieval"]
    return Action(kind=ActionKind.RETRIEVE,
                  params={"top_k": rc["top_k"], "min_score": rc["min_score"]})


def verify_action(cfg: dict, model_id: Optional[str] = None) -> Action:
    vcfg = cfg.get("verify", {}) or {}
    return Action(kind=ActionKind.VERIFY, model_id=model_id,
                  params={"max_rounds": vcfg.get("max_rounds", 1)})


def tool_action(name: str = "calculator") -> Action:
    return Action(kind=ActionKind.TOOL, params={"name": name})


def stronger_model_action(model_id: str) -> Action:
    return Action(kind=ActionKind.STRONGER_MODEL, model_id=model_id)


def abstain_action() -> Action:
    return Action(kind=ActionKind.ABSTAIN)


def stop_action() -> Action:
    return Action(kind=ActionKind.STOP)


def feasible_actions(
    cfg: dict,
    *,
    stage: str,
    small_id: str,
    big_id: Optional[str] = None,
    big_valid: bool = False,
    escalation_on: bool = False,
    acting_model: Optional[str] = None,
    taken: Optional[list[str]] = None,
) -> list[Action]:
    """The actions the current architecture could have taken at this stage.

    This is the denominator of the support diagnostic. It is deliberately the
    set the *code* could execute, not the set the heuristic happens to reach, so
    an action that is always available and never chosen shows up as coverage 0.

    `acting_model` is the model currently holding the answer. RESAMPLE,
    SELF_CHECK and VERIFY all run on that model and carry it in their key, so
    the feasible set must name it too -- otherwise the same intervention splits
    into two action keys and the coverage numbers below are meaningless.
    """
    taken = taken or []
    acting = acting_model or small_id
    if stage == "root":
        acts = [answer_action(small_id, cfg)]
        if cfg.get("tools", {}).get("calculator", True):
            acts.insert(0, tool_action())
        if big_valid and big_id:
            acts.append(answer_action(big_id, cfg))
        return acts

    acts = [stop_action(), abstain_action()]
    once = [
        resample_action(cfg, acting), self_check_action(acting),
        retrieve_action(cfg), verify_action(cfg, acting),
        answer_action(small_id, cfg),
    ]
    if big_valid and big_id:
        # ANSWER and STRONGER_MODEL on the same model are genuinely different
        # interventions in a sequential framing: ANSWER re-answers from scratch
        # and discards the observed state (what the pre-filter's hard_direct
        # route does), STRONGER_MODEL trades up carrying that state forward.
        once.append(answer_action(big_id, cfg))
        if escalation_on:
            once.append(stronger_model_action(big_id))
    acts.extend(a for a in once if a.key not in taken)
    return acts


# --------------------------------------------------------------------------- #
# Cost measurement
# --------------------------------------------------------------------------- #
def snapshot_cost(client: Any) -> CostMetrics:
    """Freeze an LLMClient's running cost so a later diff isolates one action."""
    return client.cost.model_copy()


def cost_delta(before: CostMetrics, after: CostMetrics, provider_label: str) -> ActionCost:
    """Isolate one action's cost by differencing an LLMClient's running totals.

    The dollar figure is `LLMClient.estimate_cost`: measured token counts times
    the price in config/models.yaml, which the run manifest snapshots. That is
    reproducible but it is NOT an invoice, so it is labelled `listed_price`
    rather than `billed_api`. Only a figure read back from a provider billing
    record earns `billed_api`.
    """
    tin = after.tokens_in - before.tokens_in
    tout = after.tokens_out - before.tokens_out
    usd = after.est_cost_usd - before.est_cost_usd
    if provider_label == "mock":
        # No cost was incurred at all. Reporting the price table's figure for
        # synthetic tokens would put fictional dollars in a cost table.
        usd, source = 0.0, "free"
    else:
        source = "listed_price"
    return ActionCost(
        tokens_in=tin, tokens_out=tout,
        llm_calls=after.llm_calls - before.llm_calls,
        est_cost_usd=usd,
        compute_units=tin + 10.0 * tout,
        source=source,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# Recorder
# --------------------------------------------------------------------------- #
class TraceRecorder:
    """Accumulates one Trajectory while the legacy router runs.

    Pure observation: nothing here feeds back into a routing decision.
    """

    def __init__(
        self,
        run_id: str,
        item_id: str,
        cfg: dict,
        *,
        dataset: str = "unknown",
        split: str = "pilot",
        policy_id: str = "heuristic_triage",
        budget: Optional[Budget] = None,
        seed: int = 0,
        redact_answers: bool = False,
        branch_id: str = "served",
        parent_trajectory_id: Optional[str] = None,
    ) -> None:
        self.run_id = run_id
        self.item_id = item_id
        self.cfg = cfg
        self.dataset = dataset
        self.split = split
        self.policy_id = policy_id
        self.budget = budget or Budget()
        self.seed = seed
        self.redact_answers = redact_answers
        self.branch_id = branch_id
        self.parent_trajectory_id = parent_trajectory_id

        self.steps: list[TraceStep] = []
        self.states: list[StateSnapshot] = []
        self.prompt_features: dict[str, Any] = {}
        self._current: Optional[StateSnapshot] = None
        self._root_id: str = ""

        # Routing context, so the recorder can derive feasible sets itself and
        # callers never hand-build one that fails to match the chosen action.
        self.small_id: str = ""
        self.big_id: Optional[str] = None
        self.big_valid: bool = False
        self.escalation_on: bool = False

    def set_context(
        self,
        small_id: str,
        big_id: Optional[str] = None,
        big_valid: bool = False,
        escalation_on: bool = False,
    ) -> None:
        self.small_id = small_id
        self.big_id = big_id
        self.big_valid = big_valid
        self.escalation_on = escalation_on

    def feasible(self, stage: Optional[str] = None) -> list[Action]:
        stage = stage or ("root" if not self.steps else "post")
        return feasible_actions(
            self.cfg, stage=stage, small_id=self.small_id or "unknown",
            big_id=self.big_id, big_valid=self.big_valid,
            escalation_on=self.escalation_on,
            acting_model=self._acting_model(),
            taken=self.current.actions_taken,
        )

    def _acting_model(self) -> str:
        """The model a follow-up intervention would run on.

        After the calculator tool answers, ``current.model_id`` is
        ``calculator-tool``. Resampling or verifying *on the calculator* is not
        a real action, so a non-model holder falls back to the small model --
        which is what the router would actually use next.
        """
        from ..config import get_model

        held = self.current.model_id
        return held if held and get_model(held) else self.small_id

    def update_features(self, features: dict[str, Any]) -> None:
        self.prompt_features = {**self.prompt_features, **features}

    # -- lifecycle --------------------------------------------------------- #
    def begin(self, model_id: str = "", prompt_features: Optional[dict] = None) -> StateSnapshot:
        self.prompt_features = prompt_features or {}
        root = StateSnapshot.build(
            item_id=self.item_id, step=0, answer="", signals=SignalSet(),
            model_id=model_id, actions_taken=[], spent=ActionCost(),
            prompt_features=self.prompt_features,
        )
        self._current = root
        self._root_id = root.state_id
        self.states = [root]
        return root

    @property
    def current(self) -> StateSnapshot:
        if self._current is None:
            return self.begin()
        return self._current

    def record(
        self,
        action: Action,
        *,
        answer: str,
        signals: SignalSet,
        model_id: str,
        cost: Optional[ActionCost] = None,
        provider_label: str = "unknown",
        rationale: Optional[dict] = None,
        propensity: float = 1.0,
        evidence_count: int = 0,
        feasible: Optional[list[Action]] = None,
        error: Optional[str] = None,
        is_counterfactual: bool = False,
    ) -> StateSnapshot:
        parent = self.current
        cost = cost or ActionCost()
        feas = list(feasible) if feasible is not None else self.feasible()
        if action.key not in {a.key for a in feas}:
            # The action space model missed something the router actually did.
            # Recording it anyway keeps the trace faithful; the support
            # diagnostic would otherwise silently under-count the denominator.
            feas = feas + [action]
        actions_taken = parent.actions_taken + [action.key]
        result = StateSnapshot.build(
            item_id=self.item_id, step=parent.step + 1, answer=answer, signals=signals,
            model_id=model_id, actions_taken=actions_taken,
            spent=parent.spent + cost, evidence_count=evidence_count,
            prompt_features=self.prompt_features,
        )
        changed = result.answer_hash != parent.answer_hash
        outcome = ActionOutcome(
            action=action,
            parent_state_id=parent.state_id,
            result_state_id=result.state_id,
            cost=cost,
            answer_changed=changed,
            answer_similarity=self._similarity(parent.answer, answer),
            signal_delta=_signal_delta(parent.signals, signals),
            provider_label=provider_label,
            error=error,
            is_counterfactual=is_counterfactual,
        )
        decision = PolicyDecision(
            state_id=parent.state_id, feasible=feas, chosen=action,
            policy_id=self.policy_id, propensity=propensity,
            rationale=rationale or {},
        )
        self.steps.append(TraceStep(decision=decision, outcome=outcome))
        self.states.append(result)
        self._current = result
        return result

    def finish(
        self,
        resp: ChatResponse,
        *,
        feasible: Optional[list[Action]] = None,
        label: Optional[float] = None,
        label_source: Optional[str] = None,
        rationale: Optional[dict] = None,
    ) -> Trajectory:
        """Record the terminal STOP/ABSTAIN step and seal the trajectory."""
        terminal_action = abstain_action() if resp.route.abstained else stop_action()
        self.record(
            terminal_action, feasible=feasible, answer=resp.answer,
            signals=resp.signals, model_id=resp.model,
            cost=ActionCost(), provider_label=resp.provider,
            rationale=rationale or {"status": resp.status, "tier": resp.route.tier},
        )

        total = ActionCost.zero()
        for s in self.steps:
            total = total + s.outcome.cost
        traj = Trajectory(
            trajectory_id=uuid.uuid4().hex[:16],
            run_id=self.run_id,
            item_id=self.item_id,
            dataset=self.dataset,
            split=self.split,  # type: ignore[arg-type]
            branch_id=self.branch_id,
            parent_trajectory_id=self.parent_trajectory_id,
            root_state_id=self._root_id,
            states=self.states,
            steps=self.steps,
            terminal=TerminalRecord(
                served_answer=resp.answer,
                served_answer_hash=self.current.answer_hash,
                status=resp.status,
                abstained=resp.route.abstained,
                tier=resp.route.tier,
                models_used=list(resp.route.models_used),
                label=label,
                label_source=label_source,
            ),
            total_cost=total,
            budget=self.budget.charge(total),
            seed=self.seed,
            created_ts=time.time(),
        )
        return traj.redacted() if self.redact_answers else traj

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _similarity(before: str, after: str) -> Optional[float]:
        if not before or not after:
            return None
        from ..signals.proxy import text_similarity

        return round(text_similarity(before, after), 4)


def _signal_delta(before: SignalSet, after: SignalSet) -> dict[str, float]:
    return {
        f: round(getattr(after, f) - getattr(before, f), 4)
        for f in _SIGNAL_FIELDS
        if getattr(after, f) != getattr(before, f)
    }
