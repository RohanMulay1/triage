"""Thin LLM client used by the router, signals and verify. Wraps the provider
registry, tracks cost/latency across every call made while serving one request."""
from __future__ import annotations

import re

from .config import get_model
from .providers.base import GenResult, Message
from .providers.registry import get_registry
from .schemas import CostMetrics


class ProviderFailureError(Exception):
    """A live-provider failure surfaced without a synthetic substitution."""

    def __init__(self, provider: str, result: GenResult, *, retryable: bool = False) -> None:
        self.provider = provider
        self.result = result
        self.status_code = result.http_status
        self.retry_after = result.retry_after
        self.usage_known = result.usage_known
        self.retryable = retryable
        super().__init__(result.error or f"{provider} provider failure")


class RetryableProviderError(ProviderFailureError):
    """A transient provider response that may succeed after bounded backoff."""

    def __init__(self, provider: str, result: GenResult) -> None:
        super().__init__(provider, result, retryable=True)


class RateLimitError(ProviderFailureError):
    """The live provider returned HTTP 429 — surface it to the user instead of
    silently serving the mock fallback."""

    def __init__(self, provider: str, retry_after: float | None,
                 result: GenResult | None = None) -> None:
        result = result or GenResult(
            text="", provider=provider, model="", usage_known=True,
            http_status=429, retry_after=retry_after,
            error=f"{provider} rate-limited (retry_after={retry_after})",
        )
        super().__init__(provider, result, retryable=True)


def estimate_cost(model_id: str, tokens_in: int, tokens_out: int) -> float:
    m = get_model(model_id)
    if not m:
        return 0.0
    return (tokens_in / 1_000_000) * m.get("cost_in", 0.0) + (
        tokens_out / 1_000_000
    ) * m.get("cost_out", 0.0)


class LLMClient:
    """One instance per served request; accumulates cost across all LLM calls
    (main pass + resamples + self-check + verify)."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.registry = get_registry()
        self.cost = CostMetrics()
        self.provider_label = "mock"
        self.unknown_usage = False

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.2,
        max_tokens: int = 512,
        want_logprobs: bool = False,
    ) -> GenResult:
        from .trace.pacing import active_control
        control = active_control.get()
        operation = lambda: self._generate_once(messages, temperature, max_tokens, want_logprobs)
        return await control.call(self, operation) if control is not None else await operation()

    async def _generate_once(self, messages, temperature, max_tokens, want_logprobs):
        from .trace.pacing import active_control
        from .trace.request_budget import active_ledger
        adapter, provider_model, label = self.registry.resolve(self.model_id)
        self.provider_label = label
        ledger = active_ledger.get()
        ticket = ledger.admit(self.model_id, label, provider_model, messages, max_tokens) if ledger else None
        res = await adapter.generate(
            provider_model, messages, temperature, max_tokens, want_logprobs
        )
        if ledger is not None:
            from .trace.budgeting import BudgetExceeded
            try:
                ledger.settle(ticket, res)
            except BudgetExceeded:
                self.cost.tokens_in += res.tokens_in
                self.cost.tokens_out += res.tokens_out
                self.cost.est_cost_usd += estimate_cost(self.model_id, res.tokens_in, res.tokens_out)
                self.cost.llm_calls += 1
                raise
        # Rate limits are user-facing: tell them when to come back, don't mock.
        if res.error and res.error.startswith("RATE_LIMIT") and label != "mock":
            m = re.search(r"retry_after=([0-9]+(?:\.[0-9]+)?)", res.error)
            retry_after = res.retry_after
            if retry_after is None and m:
                retry_after = float(m.group(1))
            raise RateLimitError(label, retry_after, res)
        # Research collection must never turn a provider failure into synthetic
        # text. Retry transient gateway failures through the run-wide control;
        # surface every other failure immediately. Legacy routing keeps its
        # historical mock fallback when no research control is active.
        if res.error and label != "mock" and active_control.get() is not None:
            if res.http_status in (502, 503, 504):
                raise RetryableProviderError(label, res)
            raise ProviderFailureError(label, res)
        # Fall back to mock on a live-provider error so a request never hard-fails.
        if res.error and label != "mock":
            provider_error = res.error
            res = await self.registry.mock.generate(
                provider_model, messages, temperature, max_tokens, want_logprobs
            )
            res.raw["fell_back_from"] = label
            res.raw["fallback_error"] = provider_error
            self.provider_label = "mock"
        self.cost.tokens_in += res.tokens_in
        self.cost.tokens_out += res.tokens_out
        self.cost.est_cost_usd += estimate_cost(self.model_id, res.tokens_in, res.tokens_out)
        self.cost.llm_calls += 1
        return res
