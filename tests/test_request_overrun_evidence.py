import asyncio
from types import SimpleNamespace

import pytest

from app.llm import LLMClient
from app.providers.base import GenResult
from app.providers.mock import MockProvider
from app.trace.budgeting import BudgetExceeded
from app.trace.pacing import RequestControl, active_control
from app.trace.request_budget import RequestLedger, active_ledger, input_allowance


MODEL_ID = "gpt-4o-mini"
PROVIDER_MODEL = "provider-model"
MESSAGES = [{"role": "user", "content": "Hi"}]
PRICES = {
    MODEL_ID: {
        "live_provider": "nvidia",
        "live_model": PROVIDER_MODEL,
        "cost_in": 0.15,
        "cost_out": 0.60,
    }
}


class SuccessfulAdapter:
    name = "nvidia"

    def __init__(self, response):
        self.response = response
        self.calls = 0

    async def generate(self, *args, **kwargs):
        self.calls += 1
        return self.response


def test_successful_response_overrun_retains_cost_and_request_evidence():
    tokens_in = input_allowance(MESSAGES) + 1_000
    tokens_out = 2
    reported_usd = (
        tokens_in * PRICES[MODEL_ID]["cost_in"]
        + tokens_out * PRICES[MODEL_ID]["cost_out"]
    ) / 1_000_000
    reserved_usd = (
        input_allowance(MESSAGES) * PRICES[MODEL_ID]["cost_in"]
        + PRICES[MODEL_ID]["cost_out"]
    ) / 1_000_000
    raw_response = {
        "id": "successful-response-over-admission-bound",
        "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
    }
    response = GenResult(
        text="provider returned this answer",
        provider="nvidia",
        model=PROVIDER_MODEL,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        raw=raw_response,
        usage_known=True,
        http_status=200,
    )
    adapter = SuccessfulAdapter(response)
    client = LLMClient(MODEL_ID)
    client.registry = SimpleNamespace(
        resolve=lambda model_id: (adapter, PROVIDER_MODEL, "nvidia"),
        mock=MockProvider(),
    )
    ledger = RequestLedger(PRICES, (reserved_usd + reported_usd) / 2)
    control = RequestControl(rps=1_000, attempts=1)

    async def run():
        ledger_token = active_ledger.set(ledger)
        control_token = active_control.set(control)
        try:
            with pytest.raises(BudgetExceeded, match="cost retained") as raised:
                await client.generate(MESSAGES, max_tokens=1)
            return raised.value
        finally:
            active_control.reset(control_token)
            active_ledger.reset(ledger_token)

    error = asyncio.run(run())

    assert type(error) is BudgetExceeded
    assert adapter.calls == 1
    assert ledger.reported_usd == pytest.approx(reported_usd)
    assert ledger.upper_usd == pytest.approx(reported_usd)
    assert client.cost.tokens_in == tokens_in
    assert client.cost.tokens_out == tokens_out
    assert client.cost.llm_calls == 1
    assert client.cost.est_cost_usd == pytest.approx(reported_usd)

    assert len(control.events) == 1
    event = control.events[0]
    assert event["tokens_in"] == tokens_in
    assert event["tokens_out"] == tokens_out
    assert event["usage_known"] is True
    assert event["raw_response"] == raw_response


@pytest.mark.parametrize('ceiling', [None, 8192])
def test_alternate_provider_cannot_inherit_free_primary_price(ceiling):
    from app.config import load_router_config
    from app.trace.pipeline import preflight
    prices = {'small': {'live_provider': 'groq', 'live_model': 'paid-model',
        'price_provider': 'nvidia', 'price_model': 'free-model',
        'cost_in': 0.0, 'cost_out': 0.0}}
    with pytest.raises(ValueError, match='independent declared price'):
        preflight(['question'], load_router_config(), prices, 'small', None, True, ceiling)
    ledger = RequestLedger(prices, 1.0)
    with pytest.raises(BudgetExceeded, match='independent declared price'):
        ledger.admit('small', 'groq', 'paid-model', MESSAGES, 10)
    assert ledger.records == []


def test_model_snapshot_separates_price_identity_from_resolved_identity(monkeypatch):
    from app.trace import store
    from app.providers import registry
    monkeypatch.setattr(store, 'load_models', lambda: [dict(id='unit', provider='nvidia',
        provider_model='free-model', cost_in=0, cost_out=0)])
    monkeypatch.setattr(registry, 'get_registry', lambda: SimpleNamespace(
        resolve=lambda _: (None, 'paid-model', 'groq')))
    snapshot = store.model_snapshot()['unit']
    assert snapshot['live_provider'] == 'groq'
    assert snapshot['price_provider'] == 'nvidia'
    assert snapshot['live_model'] != snapshot['price_model']


def test_direct_collector_shares_request_ledger_across_items(monkeypatch):
    from scripts import collect_traces as collector
    from app.trace import store
    run_id = store.new_run_id('direct-ledger')
    monkeypatch.setattr('sys.argv', ['collect_traces.py', '--live', '--max-usd', '1',
        '--dataset', 'mixed', '--n', '2', '--policy', 'balanced', '--run-id', run_id])
    args = collector._parse_args()
    seen = []
    async def collect_item(**kwargs):
        ledger = active_ledger.get()
        assert ledger is not None and ledger.max_usd == 1
        seen.append(ledger)
        return []
    monkeypatch.setattr(collector, 'collect_fanout', collect_item)
    asyncio.run(collector.collect(args))
    assert len(seen) == 2 and seen[0] is seen[1]
    assert active_ledger.get() is None
    assert (store.run_dir(run_id)/'request-budget.json').exists()


def test_scoring_overrun_retains_costed_failed_prefix(monkeypatch):
    from app.schemas import CostMetrics
    from app.trace import heuristic_gain as hg
    from test_trace_measurement_repairs import collect
    class OverrunClient:
        def __init__(self, model):
            self.cost = CostMetrics()
            self.provider_label = 'mock'
        async def generate(self, *args, **kwargs):
            self.cost.tokens_in = 10000
            self.cost.tokens_out = 1500
            self.cost.llm_calls = 1
            error = BudgetExceeded('reported provider overrun; cost retained')
            error.result = GenResult(text='raw overrun response', provider='mock', model='unit')
            raise error
    monkeypatch.setattr(hg, 'LLMClient', OverrunClient)
    with pytest.raises(BudgetExceeded) as caught:
        collect(scoring_policy=hg.HeuristicGainPolicy())
    prefix = caught.value.trajectories[0]
    assert prefix.total_cost.llm_calls == 2
    assert prefix.total_cost.tokens_in >= 10000
    assert prefix.terminal.label is None
    assert prefix.steps[-1].outcome.detail['budget_refused']
    assert prefix.steps[-1].outcome.detail['raw_response'] == 'raw overrun response'
