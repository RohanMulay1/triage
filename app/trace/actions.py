"""Typed intervention executor.

This is the action architecture the research program needs, and it deliberately
does **not** replace the router. `app/router/router.py` keeps its own fixed
program and its own control flow; this module is what lets a *collector* execute
an arbitrary feasible action from an arbitrary state, which is what counterfactual
branch collection requires and what the legacy router structurally cannot do.

Two rules the rest of the pipeline depends on:

  * Failure is an outcome, never a silently successful action. A provider error,
    an empty answer, or a mock answer standing in for a failed live call each get
    their own `OutcomeStatus`, so a value estimate can exclude them instead of
    reading them as "this action did not help".
  * An action with no real integration here is reported UNAVAILABLE rather than
    quietly omitted. `SPECIALIST_MODEL` has no integration in this repository and
    always reports UNAVAILABLE; pretending otherwise would put a phantom action
    in the feasible set and deflate every coverage number computed against it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Optional

from ..config import get_model, get_settings
from ..llm import LLMClient, ProviderFailureError, RateLimitError
from ..retrieval.store import get_store
from ..schemas import SignalSet
from ..signals import proxy
from ..tools import calculator
from ..verify.verify import verify
from .adapter import (
    abstain_action,
    answer_action,
    cost_delta,
    resample_action,
    retrieve_action,
    self_check_action,
    snapshot_cost,
    stop_action,
    stronger_model_action,
    tool_action,
)
from .contract import Action, ActionCost, ActionKind, OutcomeStatus, text_hash
from .budgeting import BudgetExceeded

#: Action space definition id. Support denominators computed under one definition
#: must never be compared with another; this string is recorded on every decision.
FEASIBLE_SET_DEFINITION = "executor_v2_information_consumer"

DEFAULT_SYSTEM = (
    "You are a helpful, accurate assistant. Think step by step when the question "
    "requires reasoning, then state the final answer."
)


def _degenerate(answer: str) -> bool:
    a = (answer or "").strip()
    return a == "" or a == "(no answer)" or len(a) < 2


@dataclass
class InterventionContext:
    """Everything an action needs to run, and everything it may change."""

    question: str
    cfg: dict
    small_id: str
    big_id: Optional[str] = None
    system: str = DEFAULT_SYSTEM
    answer: str = ""
    signals: SignalSet = field(default_factory=SignalSet)
    evidence: list = field(default_factory=list)
    model_id: str = ""
    samples: list[str] = field(default_factory=list)
    seed: Optional[int] = None

    def messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system},
                {"role": "user", "content": self.question}]


@dataclass
class ExecutionResult:
    """The observable consequence of one action."""

    status: OutcomeStatus
    answer: str
    signals: SignalSet
    evidence: list
    cost: ActionCost
    provider_label: str = "none"
    error: Optional[str] = None
    detail: dict[str, Any] = field(default_factory=dict)
    samples: list[str] = field(default_factory=list)
    terminal: bool = False

    def next_context(self, ctx: InterventionContext) -> InterventionContext:
        return replace(
            ctx, answer=self.answer, signals=self.signals, evidence=self.evidence,
            samples=self.samples or ctx.samples,
            model_id=self.detail.get("model_id", ctx.model_id),
        )


def _unavailable(ctx: InterventionContext, reason: str) -> ExecutionResult:
    return ExecutionResult(
        status=OutcomeStatus.UNAVAILABLE, answer=ctx.answer, signals=ctx.signals,
        evidence=ctx.evidence, cost=ActionCost(), error=reason,
        detail={"reason": reason},
    )


def _model_status(client: LLMClient, answer: str) -> OutcomeStatus:
    """Classify a model call's outcome.

    Under a forced mock provider everything is mock by design and the manifest
    records that. A mock answer in a run that expected a live provider is a
    different animal: the live call failed and synthetic text was substituted.
    """
    if client.provider_label == "mock" and not get_settings().force_mock:
        return OutcomeStatus.SYNTHETIC_FALLBACK
    if _degenerate(answer):
        return OutcomeStatus.ATTEMPTED
    return OutcomeStatus.CHOSEN


def verification_question(ctx):
    """Expose only previously observed fallible information to the continuation.

    The legacy verify function and ordinary root VERIFY prompt are unchanged.
    These observations are not retrieval evidence or correctness labels.
    """
    observations = {}
    if ctx.samples:
        observations["resampled_answers"] = ctx.samples
    if "self_check" in ctx.signals.detail:
        observations["self_check"] = ctx.signals.detail["self_check"]
    if not observations:
        return ctx.question
    return (ctx.question + "\n\nPRIOR INTERVENTION OBSERVATIONS "
            "(fallible model diagnostics, not ground truth; assess rather than trust):\n"
            + json.dumps(observations, ensure_ascii=True, sort_keys=True))


def _provider_failed(client, mark, ctx, error):
    provider = getattr(error, "provider", client.provider_label)
    unknown_usage = client.unknown_usage or (
        isinstance(error, ProviderFailureError) and not error.usage_known
    )
    budget_refused = isinstance(error, BudgetExceeded)
    return ExecutionResult(status=OutcomeStatus.FAILED, answer=ctx.answer,
        signals=ctx.signals, evidence=ctx.evidence,
        cost=cost_delta(mark, client.cost, provider), provider_label=provider,
        error=str(error), detail={
            "provider_failure": not budget_refused,
            "rate_limited": isinstance(error, RateLimitError),
            "retryable_exhausted": bool(getattr(error, "retryable", False)),
            "retry_after": getattr(error, "retry_after", None),
            "http_status": getattr(error, "status_code", None),
            "usage_known": not unknown_usage,
            "unknown_usage": unknown_usage,
              "budget_refused": budget_refused,
              "runtime_refused": getattr(error, 'runtime_refused', False),
            "model_id": client.model_id,
        })


class InterventionExecutor:
    """Executes one typed action against a context. Stateless between calls."""

    #: Actions that end an episode and can never be followed by another.
    TERMINAL = frozenset({ActionKind.STOP, ActionKind.ABSTAIN})

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    # -- feasibility -------------------------------------------------------- #
    def feasible(
        self, ctx: InterventionContext, taken: Optional[list[str]] = None
    ) -> list[tuple[Action, bool, str]]:
        """Every action in the space, with whether it is available and why not.

        Returns (action, available, reason). Callers that want the feasible set
        take the available ones; the unavailable ones are still returned so a
        collector can record UNAVAILABLE outcomes rather than dropping them.
        """
        taken = taken or []
        cfg = self.cfg
        acting = ctx.model_id if (ctx.model_id and get_model(ctx.model_id)) else ctx.small_id
        answered = bool(ctx.answer)
        out: list[tuple[Action, bool, str]] = [
            (stop_action(), answered, "" if answered else "no candidate answer yet"),
            (abstain_action(), True, ""),
            (answer_action(ctx.small_id, cfg), True, ""),
        ]

        tool_ok = bool(cfg.get("tools", {}).get("calculator", True)) and \
            calculator.solve(ctx.question)["handled"]
        out.append((tool_action(), tool_ok, "" if tool_ok
                    else "calculator cannot parse this question"))

        for act in (resample_action(cfg, acting), self_check_action(acting),
                    verify_action_for(cfg, acting)):
            ok = answered and act.key not in taken
            reason = "" if ok else ("no candidate answer yet" if not answered else "already taken")
            out.append((act, ok, reason))

        ret = retrieve_action(cfg)
        out.append((ret, ret.key not in taken, "" if ret.key not in taken else "already taken"))

        big_ok, big_reason = self._big_model_available(ctx)
        big_id = ctx.big_id or ctx.small_id
        out.append((stronger_model_action(big_id),
                    big_ok and stronger_model_action(big_id).key not in taken,
                    big_reason))

        # No specialist-model integration exists in this repository. Reporting it
        # as feasible would put an action in the support denominator that nothing
        # could ever execute.
        out.append((Action(kind=ActionKind.SPECIALIST_MODEL), False,
                    "no specialist-model integration in this repository"))
        return out

    def available_actions(
        self, ctx: InterventionContext, taken: Optional[list[str]] = None
    ) -> list[Action]:
        return [a for a, ok, _ in self.feasible(ctx, taken) if ok]

    def _big_model_available(self, ctx: InterventionContext) -> tuple[bool, str]:
        if not ctx.big_id:
            return False, "no escalation target configured"
        if ctx.big_id == ctx.small_id:
            return False, "escalation target is the same model"
        if not get_model(ctx.big_id):
            return False, f"unknown model {ctx.big_id!r}"
        from ..providers.registry import get_registry

        if get_registry().resolve(ctx.big_id)[2] == "mock" and not get_settings().force_mock:
            return False, "escalation target resolves to the mock provider"
        return True, ""

    # -- execution ---------------------------------------------------------- #
    async def execute(self, action: Action, ctx: InterventionContext) -> ExecutionResult:
        handler = {
            ActionKind.STOP: self._stop,
            ActionKind.ABSTAIN: self._abstain,
            ActionKind.ANSWER: self._answer,
            ActionKind.STRONGER_MODEL: self._stronger_model,
            ActionKind.RESAMPLE: self._resample,
            ActionKind.SELF_CHECK: self._self_check,
            ActionKind.RETRIEVE: self._retrieve,
            ActionKind.VERIFY: self._verify,
            ActionKind.TOOL: self._tool,
            ActionKind.SPECIALIST_MODEL: self._specialist,
        }.get(action.kind)
        if handler is None:  # pragma: no cover - the enum is exhaustive
            return _unavailable(ctx, f"no executor for {action.kind}")
        return await handler(action, ctx)

    # -- terminal ----------------------------------------------------------- #
    async def _stop(self, action: Action, ctx: InterventionContext) -> ExecutionResult:
        return ExecutionResult(
            status=OutcomeStatus.CHOSEN, answer=ctx.answer, signals=ctx.signals,
            evidence=ctx.evidence, cost=ActionCost(), terminal=True,
            detail={"served": True},
        )

    async def _abstain(self, action: Action, ctx: InterventionContext) -> ExecutionResult:
        return ExecutionResult(
            status=OutcomeStatus.CHOSEN, answer=ctx.answer, signals=ctx.signals,
            evidence=ctx.evidence, cost=ActionCost(), terminal=True,
            detail={"abstained": True, "withheld_answer_hash": text_hash(ctx.answer)},
        )

    # -- generation --------------------------------------------------------- #
    async def _generate_on(self, model_id: str, ctx: InterventionContext) -> ExecutionResult:
        pcfg = self.cfg["proxy"]
        client = LLMClient(model_id)
        mark = snapshot_cost(client)
        try:
            res = await client.generate(
                ctx.messages(), temperature=pcfg["base_temperature"],
                max_tokens=pcfg["max_tokens"], want_logprobs=True,
            )
        except (ProviderFailureError, BudgetExceeded) as e:
            return _provider_failed(client, mark, ctx, e)
        answer = res.text or "(no answer)"
        signals = ctx.signals.model_copy(deep=True)
        signals.uncertainty, signals.detail["uncertainty"] = proxy.uncertainty(res, [])
        return ExecutionResult(
            status=_model_status(client, answer), answer=answer, signals=signals,
            evidence=ctx.evidence,
            cost=cost_delta(mark, client.cost, client.provider_label),
            provider_label=client.provider_label, error=res.error,
            samples=[],
            detail={"model_id": model_id, "had_logprobs": bool(res.logprobs),
                    "prompt_hash": text_hash(ctx.question), "finish": res.finish_reason},
        )

    async def _answer(self, action: Action, ctx: InterventionContext) -> ExecutionResult:
        return await self._generate_on(action.model_id or ctx.small_id, ctx)

    async def _stronger_model(self, action: Action, ctx: InterventionContext) -> ExecutionResult:
        ok, reason = self._big_model_available(ctx)
        if not ok:
            return _unavailable(ctx, reason)
        # The cost recorded here is the big-model pass alone. The small-model
        # trajectory was already charged to earlier steps and is not re-charged,
        # so this IS the marginal increment of trading up.
        return await self._generate_on(action.model_id or ctx.big_id, ctx)

    # -- signal-producing interventions ------------------------------------- #
    async def _resample(self, action: Action, ctx: InterventionContext) -> ExecutionResult:
        pcfg = self.cfg["proxy"]
        n = int(action.params.get("n", pcfg["resamples"]))
        temp = float(action.params.get("t", pcfg["resample_temperature"]))
        model_id = action.model_id or ctx.model_id or ctx.small_id
        client = LLMClient(model_id)
        mark = snapshot_cost(client)
        samples: list[str] = []
        try:
            for _ in range(n):
                r = await client.generate(ctx.messages(), temperature=temp,
                                          max_tokens=pcfg["max_tokens"])
                if r.text:
                    samples.append(r.text)
        except (ProviderFailureError, BudgetExceeded) as e:
            return _provider_failed(client, mark, ctx, e)
        signals = ctx.signals.model_copy(deep=True)
        if samples:
            signals.instability, signals.detail["instability"] = proxy.instability(
                ctx.answer, samples)
        status = OutcomeStatus.CHOSEN if samples else OutcomeStatus.ATTEMPTED
        if client.provider_label == "mock" and not get_settings().force_mock:
            status = OutcomeStatus.SYNTHETIC_FALLBACK
        return ExecutionResult(
            status=status, answer=ctx.answer, signals=signals, evidence=ctx.evidence,
            cost=cost_delta(mark, client.cost, client.provider_label),
            provider_label=client.provider_label,
            samples=ctx.samples + samples,
            detail={"n_requested": n, "n_realised": len(samples), "temperature": temp},
        )

    async def _self_check(self, action: Action, ctx: InterventionContext) -> ExecutionResult:
        model_id = action.model_id or ctx.model_id or ctx.small_id
        client = LLMClient(model_id)
        mark = snapshot_cost(client)
        try:
            score, detail = await proxy.contradiction_probe(client, ctx.question, ctx.answer)
        except (ProviderFailureError, BudgetExceeded) as e:
            return _provider_failed(client, mark, ctx, e)
        signals = ctx.signals.model_copy(deep=True)
        signals.contradiction, signals.detail["contradiction"] = score, detail
        signals.detail["self_check"] = detail
        status = OutcomeStatus.CHOSEN
        if client.provider_label == "mock" and not get_settings().force_mock:
            status = OutcomeStatus.SYNTHETIC_FALLBACK
        return ExecutionResult(
            status=status, answer=ctx.answer, signals=signals, evidence=ctx.evidence,
            cost=cost_delta(mark, client.cost, client.provider_label),
            provider_label=client.provider_label, detail={"contradiction": score},
        )

    async def _retrieve(self, action: Action, ctx: InterventionContext) -> ExecutionResult:
        rc = self.cfg["retrieval"]
        top_k = int(action.params.get("top_k", rc["top_k"]))
        min_score = float(action.params.get("min_score", rc["min_score"]))
        hits = get_store().search(ctx.question, top_k=top_k, min_score=min_score)
        signals = ctx.signals.model_copy(deep=True)
        signals.evidence_sufficiency, signals.detail["evidence_sufficiency"] = (
            proxy.evidence_sufficiency(hits, min_score))
        if ctx.answer:
            signals.retrieval_disagreement, signals.detail["retrieval_disagreement"] = (
                proxy.retrieval_disagreement(ctx.answer, hits))
        signals.retrieval_support = round(
            signals.evidence_sufficiency * (1.0 - signals.retrieval_disagreement), 4)
        return ExecutionResult(
            status=OutcomeStatus.CHOSEN, answer=ctx.answer, signals=signals,
            evidence=hits, cost=ActionCost(), provider_label="retrieval",
            detail={"hits": len(hits), "top_k": top_k, "min_score": min_score},
        )

    async def _verify(self, action: Action, ctx: InterventionContext) -> ExecutionResult:
        model_id = action.model_id or ctx.model_id or ctx.small_id
        client = LLMClient(model_id)
        mark = snapshot_cost(client)
        try:
            v = await verify(client, verification_question(ctx), ctx.answer, ctx.evidence)
        except (ProviderFailureError, BudgetExceeded) as e:
            return _provider_failed(client, mark, ctx, e)
        signals = ctx.signals.model_copy(deep=True)
        signals.detail["verify"] = {"pass": v["pass"], "reason": v["reason"]}
        answer = v["revised"] if v["pass"] else ctx.answer
        status = OutcomeStatus.CHOSEN
        if client.provider_label == "mock" and not get_settings().force_mock:
            status = OutcomeStatus.SYNTHETIC_FALLBACK
        return ExecutionResult(
            status=status, answer=answer, signals=signals, evidence=ctx.evidence,
            cost=cost_delta(mark, client.cost, client.provider_label),
            provider_label=client.provider_label,
            detail={"pass": v["pass"], "grounded": v["grounded"], "reason": v["reason"][:160]},
        )

    async def _tool(self, action: Action, ctx: InterventionContext) -> ExecutionResult:
        sol = calculator.solve(ctx.question)
        if not sol["handled"]:
            return _unavailable(ctx, "calculator cannot parse this question")
        signals = ctx.signals.model_copy(deep=True)
        signals.detail["tool"] = {"name": "calculator", "expression": sol["expression"]}
        return ExecutionResult(
            status=OutcomeStatus.CHOSEN, answer=str(sol["answer"]), signals=signals,
            evidence=ctx.evidence, cost=ActionCost(), provider_label="tool",
            detail={"expression": sol["expression"], "model_id": "calculator-tool"},
        )

    async def _specialist(self, action: Action, ctx: InterventionContext) -> ExecutionResult:
        return _unavailable(ctx, "no specialist-model integration in this repository")


def verify_action_for(cfg: dict, model_id: Optional[str] = None) -> Action:
    """Local alias so this module owns one import surface for action builders."""
    from .adapter import verify_action

    return verify_action(cfg, model_id)
