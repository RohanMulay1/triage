"""Behaviour policies for trace collection.

A behaviour policy decides which action to execute next *while collecting data*.
It is not the policy under study; it is the thing that determines what the data
can later support. Every choice returns a **propensity** — the probability this
policy assigned to the action it took — because without it no importance-weighted
or doubly-robust estimate is admissible.

The deterministic policies honestly report propensity 1.0. That is not a
formality: it is the record that they provide zero support for the actions they
did not take, which is what `app/trace/support.py` refuses to run OPE against.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Protocol

from ..router.abstain import abstain_risk
from .actions import FEASIBLE_SET_DEFINITION, InterventionContext, InterventionExecutor
from .adapter import answer_action, stop_action, stronger_model_action
from .contract import Action, ActionKind


@dataclass(frozen=True)
class PolicyChoice:
    action: Action
    propensity: float
    feasible: list[Action]
    rationale: dict


class BehaviorPolicy(Protocol):
    policy_id: str
    policy_version: str

    def choose(
        self,
        ctx: InterventionContext,
        executor: InterventionExecutor,
        taken: list[str],
        rng: random.Random,
    ) -> PolicyChoice: ...


def _uniform(actions: list[Action], rng: random.Random, rationale: dict) -> PolicyChoice:
    chosen = actions[rng.randrange(len(actions))]
    return PolicyChoice(chosen, 1.0 / len(actions), actions, rationale)


class RandomFeasiblePolicy:
    """Uniform over available actions. Maximum exploration, no exploitation.

    This is the policy that makes off-policy estimation possible at all: every
    available action has propensity 1/|A| > 0, so positivity holds by
    construction.
    """

    policy_id = "random_feasible"
    policy_version = "1"

    def choose(self, ctx, executor, taken, rng) -> PolicyChoice:
        actions = executor.available_actions(ctx, taken)
        return _uniform(actions, rng, {"n_feasible": len(actions)})


class EpsilonGreedyPolicy:
    """Explore uniformly with probability epsilon, otherwise follow `base`.

    The propensity is the true mixture probability, not the base policy's, which
    is the whole point of recording it: `eps/|A| + (1-eps)*1{a == greedy}`.
    """

    policy_version = "1"

    def __init__(self, base: BehaviorPolicy, epsilon: float = 0.3) -> None:
        if not 0.0 < epsilon <= 1.0:
            raise ValueError("epsilon must lie in (0, 1]")
        self.base = base
        self.epsilon = epsilon
        self.policy_id = f"epsilon_greedy({base.policy_id},eps={epsilon})"

    def choose(self, ctx, executor, taken, rng) -> PolicyChoice:
        actions = executor.available_actions(ctx, taken)
        greedy = self.base.choose(ctx, executor, taken, rng).action
        explore = rng.random() < self.epsilon
        chosen = actions[rng.randrange(len(actions))] if explore else greedy
        n = len(actions)
        propensity = self.epsilon / n + (1 - self.epsilon) * (1.0 if chosen.key == greedy.key else 0.0)
        return PolicyChoice(chosen, propensity, actions,
                            {"explored": explore, "greedy": greedy.key, "epsilon": self.epsilon})


class BalancedExplorationPolicy:
    """Prefer the action taken least often so far in this run.

    Uniform exploration still leaves rare-but-available actions thinly covered
    when episodes differ in length. Balancing counts equalises support across the
    action space, which is what makes per-action value estimates comparable.
    """

    policy_id = "balanced_exploration"
    policy_version = "1"

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def choose(self, ctx, executor, taken, rng) -> PolicyChoice:
        actions = executor.available_actions(ctx, taken)
        lowest = min(self.counts.get(a.key, 0) for a in actions)
        tied = [a for a in actions if self.counts.get(a.key, 0) == lowest]
        chosen = tied[rng.randrange(len(tied))]
        self.counts[chosen.key] = self.counts.get(chosen.key, 0) + 1
        return PolicyChoice(chosen, 1.0 / len(tied), actions,
                            {"tied": [a.key for a in tied], "min_count": lowest})


class FixedCascadePolicy:
    """Answer small, escalate once if risk clears a fixed bar, then stop.

    The classic cascade baseline, expressed over the same action space as
    everything else so its coverage is measured on the same footing.
    """

    policy_id = "fixed_cascade"
    policy_version = "1"

    def __init__(self, risk_high: float = 0.55) -> None:
        self.risk_high = risk_high

    def choose(self, ctx, executor, taken, rng) -> PolicyChoice:
        actions = executor.available_actions(ctx, taken)
        by_kind = {a.kind: a for a in actions}
        if not ctx.answer and ActionKind.ANSWER in by_kind:
            chosen = by_kind[ActionKind.ANSWER]
            reason = "no answer yet"
        else:
            risk, _ = abstain_risk(ctx.signals, ctx.cfg["abstain"])
            escalate = risk >= self.risk_high and ActionKind.STRONGER_MODEL in by_kind
            chosen = by_kind[ActionKind.STRONGER_MODEL] if escalate else by_kind.get(
                ActionKind.STOP, actions[0])
            reason = f"risk={risk:.4f} vs {self.risk_high}"
        return PolicyChoice(chosen, 1.0, actions, {"reason": reason})


class PromptOnlyPolicy:
    """Route on prompt features alone, never on the observed response.

    The RouteLLM-shaped control. If response state carries no information beyond
    the prompt, this policy is not beaten and the Signal Gate fails.
    """

    policy_id = "prompt_only"
    policy_version = "1"

    def __init__(self, hard_direct: Optional[float] = None) -> None:
        self.hard_direct = hard_direct

    def choose(self, ctx, executor, taken, rng) -> PolicyChoice:
        from ..router import prefilter

        actions = executor.available_actions(ctx, taken)
        by_key = {a.key: a for a in actions}
        pfcfg = ctx.cfg.get("prefilter", {}) or {}
        bar = self.hard_direct if self.hard_direct is not None else pfcfg.get("hard_direct", 0.70)
        pf = prefilter.predict(ctx.question, {"hit": False, "n": 0}, pfcfg)
        difficulty = pf.get("difficulty") or 0.0

        if not ctx.answer:
            hard = difficulty >= bar and ctx.big_id
            want = (answer_action(ctx.big_id, ctx.cfg) if hard
                    else answer_action(ctx.small_id, ctx.cfg))
            chosen = by_key.get(want.key) or actions[0]
        else:
            chosen = by_key.get(stop_action().key, actions[0])
        return PolicyChoice(chosen, 1.0, actions,
                            {"difficulty": difficulty, "hard_direct": bar})


class ResponseRiskThresholdPolicy:
    """Escalate when the multi-signal abstain risk of the observed answer is high.

    This is the current Triage rule, isolated from the rest of its fixed program
    so it can be compared as a policy rather than as a codebase.
    """

    policy_id = "response_risk_threshold"
    policy_version = "1"

    def __init__(self, risk_high: float = 0.55) -> None:
        self.risk_high = risk_high

    def choose(self, ctx, executor, taken, rng) -> PolicyChoice:
        actions = executor.available_actions(ctx, taken)
        by_kind = {a.kind: a for a in actions}
        if not ctx.answer:
            return PolicyChoice(by_kind.get(ActionKind.ANSWER, actions[0]), 1.0, actions,
                                {"reason": "no answer yet"})
        risk, components = abstain_risk(ctx.signals, ctx.cfg["abstain"])
        if risk >= self.risk_high and ActionKind.STRONGER_MODEL in by_kind:
            chosen = by_kind[ActionKind.STRONGER_MODEL]
        else:
            chosen = by_kind.get(ActionKind.STOP, actions[0])
        return PolicyChoice(chosen, 1.0, actions,
                            {"risk": risk, "risk_high": self.risk_high,
                             "components": {k: round(v, 4) for k, v in components.items()}})


#: Registry for `scripts/collect_traces.py --policy`.
POLICIES: dict[str, callable] = {
    "random": lambda cfg: RandomFeasiblePolicy(),
    "balanced": lambda cfg: BalancedExplorationPolicy(),
    "epsilon_greedy": lambda cfg: EpsilonGreedyPolicy(
        ResponseRiskThresholdPolicy(
            (cfg.get("escalation") or {}).get("risk_high", 0.55)), epsilon=0.3),
    "fixed_cascade": lambda cfg: FixedCascadePolicy(
        (cfg.get("escalation") or {}).get("risk_high", 0.55)),
    "prompt_only": lambda cfg: PromptOnlyPolicy(),
    "response_risk": lambda cfg: ResponseRiskThresholdPolicy(
        (cfg.get("escalation") or {}).get("risk_high", 0.55)),
}


def build_policy(name: str, cfg: dict) -> BehaviorPolicy:
    if name not in POLICIES:
        raise KeyError(f"unknown behaviour policy {name!r}; known: {sorted(POLICIES)}")
    return POLICIES[name](cfg)


FEASIBLE_SET = FEASIBLE_SET_DEFINITION
