"""Run-wide admission estimates and actual-cost settlement for trace collection."""
from __future__ import annotations

import math
from copy import deepcopy

from ..config import get_settings
from .contract import ActionCost, ActionKind, Budget


class BudgetExceeded(RuntimeError):
    """Collection stopped; completed trajectories remain available for persistence."""

    def __init__(self, message: str):
        super().__init__(message)
        self.trajectories = []


class RunBudget:
    """One caller-owned accumulator per run; Budget itself remains immutable.

    Admission uses an estimate and never charges it. Settlement records actual
    reported cost, including an overrun, then check() stops further execution.
    Token estimates and listed prices are not an invoice guarantee.
    """

    def __init__(self, budget: Budget | None = None, model_prices: dict | None = None):
        self.state = budget or Budget()
        if self.state.max_usd is not None and (
                not math.isfinite(self.state.max_usd) or self.state.max_usd <= 0):
            raise ValueError("max_usd must be finite and strictly positive")
        if model_prices is None:
            from .store import model_snapshot
            model_prices = model_snapshot()
        self.model_prices = deepcopy(model_prices)

    def would_exceed(self, cost: ActionCost) -> bool:
        return not self.state.can_afford(cost)

    def admit(self, estimate: ActionCost) -> None:
        if self.would_exceed(estimate):
            raise BudgetExceeded(
                f"action estimate {estimate.model_dump()} exceeds remaining run "
                f"budget {self.state.remaining()}; refused before execution")

    def settle(self, actual: ActionCost) -> None:
        self.state = self.state.charge(actual)

    def check(self) -> None:
        if self.would_exceed(ActionCost()):
            raise BudgetExceeded("actual reported cost exceeded admission estimate; "
                                 "cost retained, further execution stopped")

    def charge(self, cost: ActionCost) -> None:
        """Compatibility helper for callers admitting an already-known cost."""
        self.admit(cost)
        self.settle(cost)


def estimate_action_cost(action, ctx, prices: dict) -> ActionCost:
    """Conservative input-byte allowance + capped completions for every call.

    UTF-8 bytes are intentionally more conservative than len(prompt)/4. Wrapper
    allowance covers the fixed verification/self-check templates and chat framing.
    Count all resamples. Prices are the run-manifest snapshot, not mutable config.
    No future latency estimate is asserted. Unknown/invalid prices fail closed.
    """
    if action.kind in (ActionKind.STOP, ActionKind.ABSTAIN, ActionKind.RETRIEVE,
                       ActionKind.TOOL, ActionKind.SPECIALIST_MODEL):
        return ActionCost()
    n = int(action.params.get("n", ctx.cfg["proxy"]["resamples"])) \
        if action.kind == ActionKind.RESAMPLE else 1
    max_tokens = (80 if action.kind == ActionKind.SELF_CHECK else
                  400 if action.kind == ActionKind.VERIFY else
                  int(ctx.cfg["proxy"]["max_tokens"]))
    if n <= 0 or max_tokens <= 0:
        raise ValueError("call count and completion cap must be positive")
    prompt = ctx.system + ctx.question
    if action.kind in (ActionKind.SELF_CHECK, ActionKind.VERIFY):
        prompt += ctx.answer
    if action.kind == ActionKind.VERIFY:
        prompt += "".join(h.text for h in ctx.evidence)
    tin = n * (len(prompt.encode("utf-8")) + 1024 + 8 * len(ctx.evidence))
    tout = n * max_tokens
    model = action.model_id or ctx.model_id or ctx.small_id
    price = prices.get(model)
    if price is None:
        raise ValueError(f"no snapshotted price for model {model!r}")
    rates = [float(price[k]) for k in ("cost_in", "cost_out")]
    if any(not math.isfinite(v) or v < 0 for v in rates):
        raise ValueError("snapshotted prices must be finite and nonnegative")
    mock = get_settings().force_mock
    dollars = 0.0 if mock else (tin * rates[0] + tout * rates[1]) / 1_000_000
    return ActionCost(tokens_in=tin, tokens_out=tout, llm_calls=n,
                      est_cost_usd=dollars, compute_units=tin + 10 * tout,
                      source="free" if mock else "listed_price")
