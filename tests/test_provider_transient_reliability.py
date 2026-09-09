import asyncio
from types import SimpleNamespace

import httpx
import pytest

from app.llm import LLMClient, ProviderFailureError, RetryableProviderError
from app.providers.base import GenResult
from app.providers.mock import MockProvider
from app.providers.openai_compat import OpenAICompatAdapter, _retry_after_seconds
from app.trace.pacing import RequestControl, active_control
from app.trace.budgeting import BudgetExceeded
from app.trace.contract import ActionKind, OutcomeStatus
from test_trace_policies import _ctx, _ex


class SequenceAdapter:
    name = "nvidia"

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    async def generate(self, *args, **kwargs):
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return result


def failed(status=503, retry_after=2.5):
    return GenResult(
        text="", provider="nvidia", model="provider-model",
        error=f"HTTP {status}: temporarily overloaded", usage_known=False,
        http_status=status, retry_after=retry_after,
    )


def client_for(adapter):
    client = LLMClient("unit-model")
    client.registry = SimpleNamespace(
        resolve=lambda model: (adapter, "provider-model", "nvidia"),
        mock=MockProvider(),
    )
    return client


def test_adapter_preserves_503_and_retry_after(monkeypatch):
    async def post(self, url, **kwargs):
        return httpx.Response(
            503, headers={"Retry-After": "4.5"},
            json={"error": {"message": "temporarily overloaded"}},
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    adapter = OpenAICompatAdapter("nvidia", "test", "https://example.invalid")
    result = asyncio.run(adapter.generate("model", [{"role": "user", "content": "Hi"}]))
    assert result.http_status == 503
    assert result.retry_after == 4.5
    assert result.usage_known is False
    assert result.tokens_in == result.tokens_out == 0


def test_transient_503_retries_with_shared_control_without_mock_fallback():
    now = [0.0]
    sleeps = []

    async def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    adapter = SequenceAdapter([
        failed(),
        GenResult(text="real", provider="nvidia", model="provider-model",
                  tokens_in=5, tokens_out=1),
    ])
    client = client_for(adapter)
    control = RequestControl(rps=1, attempts=3, sleep=sleep, clock=lambda: now[0])

    async def run():
        token = active_control.set(control)
        try:
            return await client.generate([{"role": "user", "content": "Hi"}])
        finally:
            active_control.reset(token)

    result = asyncio.run(run())
    assert result.text == "real" and client.provider_label == "nvidia"
    assert adapter.calls == 2 and 2.5 in sleeps
    assert client.cost.llm_calls == 1
    assert client.unknown_usage is True
    assert result.raw["unknown_usage_before_success"] is True
    assert [event["status"] for event in control.events] == [
        "provider_failure", "returned"
    ]


def test_transient_503_exhaustion_is_bounded_and_never_uses_mock():
    now = [0.0]

    async def sleep(delay):
        now[0] += delay

    adapter = SequenceAdapter([failed(retry_after=0)])
    client = client_for(adapter)
    control = RequestControl(rps=2, attempts=3, sleep=sleep, clock=lambda: now[0])

    async def run():
        token = active_control.set(control)
        try:
            with pytest.raises(RetryableProviderError) as raised:
                await client.generate([{"role": "user", "content": "Hi"}])
            return raised.value
        finally:
            active_control.reset(token)

    error = asyncio.run(run())
    assert adapter.calls == 3
    assert error.status_code == 503 and error.usage_known is False
    assert client.provider_label == "nvidia" and client.cost.llm_calls == 0
    assert client.unknown_usage is True
    assert len(control.events) == 3
    assert control.events[-1]["backoff_seconds"] == 0


def test_nonretryable_research_failure_fails_closed_once():
    adapter = SequenceAdapter([failed(status=410)])
    client = client_for(adapter)
    control = RequestControl(rps=1000, attempts=3)

    async def run():
        token = active_control.set(control)
        try:
            with pytest.raises(ProviderFailureError) as raised:
                await client.generate([{"role": "user", "content": "Hi"}])
            return raised.value
        finally:
            active_control.reset(token)

    error = asyncio.run(run())
    assert type(error) is ProviderFailureError
    assert adapter.calls == 1 and error.status_code == 410
    assert control.events[0]["retryable"] is False


def test_legacy_call_still_uses_historical_mock_fallback_for_503():
    adapter = SequenceAdapter([failed()])
    client = client_for(adapter)
    result = asyncio.run(client.generate([{"role": "user", "content": "Hi"}]))
    assert adapter.calls == 1
    assert client.provider_label == "mock"
    assert result.raw["fallback_error"].startswith("HTTP 503")


@pytest.mark.parametrize(
    "kind",
    [ActionKind.ANSWER, ActionKind.RESAMPLE, ActionKind.SELF_CHECK, ActionKind.VERIFY],
)
def test_all_generation_actions_record_exhausted_503_as_failed(monkeypatch, kind):
    async def unavailable(self, *args, **kwargs):
        self.provider_label = "nvidia"
        self.unknown_usage = True
        raise RetryableProviderError("nvidia", failed())

    monkeypatch.setattr(LLMClient, "generate", unavailable)
    ctx, executor = _ctx(), _ex()
    action = next(a for a in executor.available_actions(ctx, []) if a.kind == kind)
    result = asyncio.run(executor.execute(action, ctx))
    assert result.status == OutcomeStatus.FAILED
    assert result.provider_label == "nvidia"
    assert result.detail["http_status"] == 503
    assert result.detail["retryable_exhausted"] is True
    assert result.detail["unknown_usage"] is True


@pytest.mark.parametrize(
    "kind",
    [ActionKind.ANSWER, ActionKind.RESAMPLE, ActionKind.SELF_CHECK, ActionKind.VERIFY],
)
def test_all_generation_actions_record_request_budget_refusal(monkeypatch, kind):
    async def refused(self, *args, **kwargs):
        self.provider_label = "groq"
        raise BudgetExceeded("request reservation exceeds remaining run-wide cap")

    monkeypatch.setattr(LLMClient, "generate", refused)
    ctx, executor = _ctx(), _ex()
    action = next(a for a in executor.available_actions(ctx, []) if a.kind == kind)
    result = asyncio.run(executor.execute(action, ctx))
    assert result.status == OutcomeStatus.FAILED
    assert result.provider_label == "groq"
    assert result.detail["budget_refused"] is True
    assert result.detail["unknown_usage"] is False


def test_partial_resample_cost_survives_exhausted_503(monkeypatch):
    count = 0

    async def partial(self, *args, **kwargs):
        nonlocal count
        count += 1
        self.provider_label = "nvidia"
        if count == 1:
            self.cost.llm_calls += 1
            self.cost.tokens_in += 7
            self.cost.tokens_out += 2
            return GenResult(text="sample", provider="nvidia", model=self.model_id,
                             tokens_in=7, tokens_out=2)
        self.unknown_usage = True
        raise RetryableProviderError("nvidia", failed())

    monkeypatch.setattr(LLMClient, "generate", partial)
    ctx, executor = _ctx(), _ex()
    action = next(a for a in executor.available_actions(ctx, [])
                  if a.kind == ActionKind.RESAMPLE)
    result = asyncio.run(executor.execute(action, ctx))
    assert result.status == OutcomeStatus.FAILED
    assert result.cost.llm_calls == 1
    assert result.cost.tokens_in == 7 and result.cost.tokens_out == 2
    assert result.detail["unknown_usage"] is True


@pytest.mark.parametrize("value, expected", [("3.25", 3.25), ("bad", None), (None, None)])
def test_retry_after_parser(value, expected):
    assert _retry_after_seconds(value) == expected
