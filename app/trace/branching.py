"""Counterfactual branch collection.

The Signal Gate asks a question the served path cannot answer: what is the
*incremental* benefit of each action relative to stopping? Answering it needs the
same state to be followed by different actions, which is exactly what a
deterministic router never produces.

`collect_fanout` does the smallest thing that answers it honestly. For one item:

    root --ANSWER--> s1 ┬── STOP            -> label
                        ├── RETRIEVE  -> STOP -> label
                        ├── VERIFY    -> STOP -> label
                        ├── RESAMPLE  -> STOP -> label
                        ├── SELF_CHECK-> STOP -> label
                        └── STRONGER_MODEL -> STOP -> label

Every available action is executed exactly once from the *same* parent state, so
support is complete by construction and Δ(a) = label(branch a) − label(branch
STOP) is measured, not imputed. The STOP branch is the baseline every other
branch is differenced against.

The shared prefix is emitted as its own trajectory and each branch begins at s1
with `parent_trajectory_id` pointing at it. Nothing is duplicated, so summing
costs across a run gives what the run actually spent.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from ..schemas import SignalSet
from .actions import (
    FEASIBLE_SET_DEFINITION,
    ExecutionResult,
    InterventionContext,
    InterventionExecutor,
)
from .adapter import answer_action, stop_action
from .budgeting import BudgetExceeded, RunBudget, estimate_action_cost
from .contract import (
    Action,
    ActionCost,
    ActionKind,
    ActionOutcome,
    Budget,
    OutcomeStatus,
    PolicyDecision,
    StateSnapshot,
    TerminalRecord,
    TraceStep,
    Trajectory,
)

#: (answer, abstained) -> label in [0, 1], or None when no label is available.
Labeler = Callable[[str, bool], Optional[float]]


@dataclass
class _Builder:
    """Accumulates one branch. Kept private: branches are built here, not by callers."""

    item_id: str
    run_id: str
    dataset: str
    split: str
    seed: int
    branch_id: str
    parent_trajectory_id: Optional[str]
    policy_id: str
    policy_version: str
    prompt_features: dict

    def __post_init__(self) -> None:
        self.states: list[StateSnapshot] = []
        self.steps: list[TraceStep] = []
        self._current: Optional[StateSnapshot] = None
        self._root_id: str = ""

    def start_at(self, state: StateSnapshot) -> None:
        self._current = state
        self._root_id = state.state_id
        self.states = [state]

    def start_root(self, model_id: str) -> StateSnapshot:
        root = StateSnapshot.build(
            item_id=self.item_id, step=0, answer="", signals=SignalSet(),
            model_id=model_id, actions_taken=[], spent=ActionCost(),
            prompt_features=self.prompt_features,
        )
        self.start_at(root)
        return root

    @property
    def current(self) -> StateSnapshot:
        assert self._current is not None, "call start_root/start_at first"
        return self._current

    def add(
        self,
        action: Action,
        result: ExecutionResult,
        *,
        feasible: list[Action],
        propensity: float,
        status: OutcomeStatus,
        rationale: dict,
        exploration_seed: Optional[int],
    ) -> StateSnapshot:
        parent = self.current
        cumulative = parent.spent + result.cost
        nxt = StateSnapshot.build(
            item_id=self.item_id, step=parent.step + 1, answer=result.answer,
            signals=result.signals,
            model_id=result.detail.get("model_id", parent.model_id),
            actions_taken=parent.actions_taken + [action.key],
            spent=cumulative, evidence_count=len(result.evidence),
            prompt_features=self.prompt_features,
        )
        feas = list(feasible)
        if action.key not in {a.key for a in feas}:
            feas.append(action)
        self.steps.append(TraceStep(
            decision=PolicyDecision(
                state_id=parent.state_id, feasible=feas, chosen=action,
                policy_id=self.policy_id, policy_version=self.policy_version,
                feasible_set_definition=FEASIBLE_SET_DEFINITION,
                exploration_seed=exploration_seed, propensity=propensity,
                rationale=rationale,
            ),
            outcome=ActionOutcome(
                action=action, parent_state_id=parent.state_id,
                result_state_id=nxt.state_id, status=status,
                cost=result.cost, cumulative_cost=cumulative,
                answer_changed=nxt.answer_hash != parent.answer_hash,
                signal_delta=_signal_delta(parent.signals, result.signals),
                provider_label=result.provider_label, error=result.error,
                is_counterfactual=status == OutcomeStatus.COUNTERFACTUAL,
                seed=exploration_seed, detail=result.detail,
            ),
        ))
        self.states.append(nxt)
        self._current = nxt
        return nxt

    def build(self, *, status: str, abstained: bool, label: Optional[float],
              label_source: Optional[str], budget: Budget) -> Trajectory:
        total = ActionCost.zero()
        for s in self.steps:
            total = total + s.outcome.cost
        return Trajectory(
            trajectory_id=uuid.uuid4().hex[:16], run_id=self.run_id, item_id=self.item_id,
            dataset=self.dataset, split=self.split,  # type: ignore[arg-type]
            branch_id=self.branch_id, parent_trajectory_id=self.parent_trajectory_id,
            root_state_id=self._root_id, states=self.states, steps=self.steps,
            terminal=TerminalRecord(
                served_answer=self.current.answer,
                served_answer_hash=self.current.answer_hash,
                status=status, abstained=abstained,
                models_used=[self.current.model_id] if self.current.model_id else [],
                label=label, label_source=label_source,
            ),
            total_cost=total, budget=budget, seed=self.seed,
            created_ts=time.time(),
        )


_SIGNAL_FIELDS = ("uncertainty", "instability", "contradiction",
                  "retrieval_disagreement", "evidence_sufficiency")


def _signal_delta(before: SignalSet, after: SignalSet) -> dict[str, float]:
    return {f: round(getattr(after, f) - getattr(before, f), 4)
            for f in _SIGNAL_FIELDS
            if getattr(after, f) != getattr(before, f)}


async def collect_fanout(
    *,
    item_id: str,
    question: str,
    cfg: dict,
    small_id: str,
    big_id: Optional[str],
    run_id: str,
    dataset: str,
    split: str,
    seed: int,
    labeler: Labeler,
    label_source: str,
    served_policy=None,
    budget: Optional[Budget | RunBudget] = None,
    prompt_features: Optional[dict] = None,
) -> list[Trajectory]:
    """One shared ANSWER prefix, then every available action once from that state.

    Returns [prefix, branch_STOP, branch_a1, ...]. Exactly one branch is marked
    CHOSEN (the one `served_policy` would have taken); the rest are
    COUNTERFACTUAL. Unavailable actions are recorded too, as zero-cost
    UNAVAILABLE outcomes, so the support denominator stays honest.
    """
    import random

    executor = InterventionExecutor(cfg)
    run_budget = budget if isinstance(budget, RunBudget) else RunBudget(budget)
    rng = random.Random(seed)
    features = dict(prompt_features or {})

    ctx = InterventionContext(question=question, cfg=cfg, small_id=small_id,
                              big_id=big_id, model_id=small_id, seed=seed)

    # ---- shared prefix: one ANSWER on the small model --------------------- #
    prefix = _Builder(item_id, run_id, dataset, split, seed, "prefix", None,
                      "fanout_prefix", "1", features)
    prefix.start_root(small_id)
    answer = answer_action(small_id, cfg)
    run_budget.admit(estimate_action_cost(answer, ctx, run_budget.model_prices))
    result = await executor.execute(answer, ctx)
    run_budget.settle(result.cost)
    # The prefix is forced, not chosen: the feasible set is the single action the
    # collector was always going to take. Recording the full root action space
    # here would claim support this design does not provide.
    s1 = prefix.add(answer, result, feasible=[answer], propensity=1.0,
                    status=result.status,
                    rationale={"role": "shared prefix", "forced": True},
                    exploration_seed=seed)
    prefix_traj = prefix.build(status="prefix", abstained=False, label=None,
                               label_source=None, budget=run_budget.state)

    ctx1 = result.next_context(ctx)
    taken = [answer.key]
    trajectories = [prefix_traj]

    def check_budget():
        try:
            run_budget.check()
        except BudgetExceeded as exc:
            exc.trajectories = trajectories
            raise

    check_budget()

    if result.status not in (OutcomeStatus.CHOSEN, OutcomeStatus.COUNTERFACTUAL):
        # The prefix itself failed; there is no state worth branching from and
        # recording branches would attribute the failure to each action.
        return trajectories

    # ---- which branch would the served policy have taken? ----------------- #
    served_key = None
    if served_policy is not None:
        served_key = served_policy.choose(ctx1, executor, taken, rng).action.key

    branchable = executor.feasible(ctx1, taken)
    available = [a for a, ok, _ in branchable if ok]
    if served_key is None:
        served_key = stop_action().key

    for action, ok, reason in branchable:
        builder = _Builder(item_id, run_id, dataset, split, seed, action.key,
                           prefix_traj.trajectory_id,
                           getattr(served_policy, "policy_id", "exhaustive_fanout"),
                           getattr(served_policy, "policy_version", "1"), features)
        builder.start_at(s1)
        is_served = action.key == served_key

        if not ok:
            # Record unavailability as a real, distinguishable outcome. Dropping
            # it would leave the support denominator claiming an action was
            # available when nothing could have executed it.
            unavailable = ExecutionResult(
                status=OutcomeStatus.UNAVAILABLE, answer=ctx1.answer,
                signals=ctx1.signals, evidence=ctx1.evidence, cost=ActionCost(),
                error=reason, detail={"reason": reason},
            )
            builder.add(action, unavailable, feasible=available, propensity=1.0,
                        status=OutcomeStatus.UNAVAILABLE,
                        rationale={"unavailable": reason}, exploration_seed=seed)
            trajectories.append(builder.build(
                status="unavailable", abstained=False, label=None,
                label_source=None, budget=run_budget.state))
            check_budget()
            continue

        try:
            run_budget.admit(estimate_action_cost(action, ctx1, run_budget.model_prices))
        except BudgetExceeded as exc:
            exc.trajectories = trajectories
            raise
        res = await executor.execute(action, ctx1)
        run_budget.settle(res.cost)

        if res.status is not OutcomeStatus.CHOSEN:
            # The action ran but produced nothing usable, errored, or was served
            # by the mock standing in for a failed live call. None of those is an
            # observation of what this action is worth, so the branch is recorded
            # with its real status and is NOT carried to a terminal or labelled.
            # Labelling it would enter a provider failure into the dataset as
            # evidence that the action does not help.
            builder.add(action, res, feasible=available, propensity=1.0,
                        status=res.status,
                        rationale={"fanout": True, "served": is_served,
                                   "not_observed": res.status.value},
                        exploration_seed=seed)
            trajectories.append(builder.build(
                status=res.status.value, abstained=False, label=None,
                label_source=None, budget=run_budget.state))
            check_budget()
            continue

        # `is_counterfactual` on an outcome means "an observed execution of an
        # action the served policy did not take". Which branch a trajectory is
        # remains recorded on `branch_id`/`parent_trajectory_id` regardless.
        status = OutcomeStatus.CHOSEN if is_served else OutcomeStatus.COUNTERFACTUAL
        builder.add(action, res, feasible=available, propensity=1.0, status=status,
                    rationale={"fanout": True, "served": is_served},
                    exploration_seed=seed)

        ctx2 = res.next_context(ctx1)
        abstained = action.kind == ActionKind.ABSTAIN
        if action.kind not in executor.TERMINAL:
            stop = stop_action()
            stop_res = await executor.execute(stop, ctx2)
            # Also forced: the branch ends here by construction. Its feasible set
            # is itself, so it contributes no phantom support for the actions the
            # fan-out never explored at depth 2.
            builder.add(stop, stop_res, feasible=[stop], propensity=1.0,
                        status=status,
                        rationale={"terminal": True, "forced": True},
                        exploration_seed=seed)

        label = labeler(builder.current.answer, abstained)
        trajectories.append(builder.build(
            status="PENDING_REVIEW" if abstained else "OK", abstained=abstained,
            label=label, label_source=label_source, budget=run_budget.state))
        check_budget()

    return trajectories
