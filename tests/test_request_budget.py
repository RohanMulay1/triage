import asyncio

import pytest

from app.providers.base import GenResult
from app.trace.budgeting import BudgetExceeded
from app.trace.request_budget import RequestLedger, active_ledger, input_allowance
from app.trace.pacing import RequestControl, active_control
from test_provider_transient_reliability import SequenceAdapter, client_for, failed


PRICES = {'unit-model': {'live_provider': 'nvidia', 'live_model': 'provider-model',
                         'cost_in': 1.0, 'cost_out': 1.0}}
MESSAGES = [{'role': 'user', 'content': 'Hi'}]


def test_unknown_request_reserves_cost_and_refuses_retry_before_call():
    reserve = (input_allowance(MESSAGES)+10)/1e6
    ledger = RequestLedger(PRICES, reserve*1.5)
    adapter = SequenceAdapter([failed()])
    client = client_for(adapter)
    async def sleep(seconds):
        pass
    async def run():
        a = active_ledger.set(ledger)
        b = active_control.set(RequestControl(sleep=sleep))
        try:
            with pytest.raises(BudgetExceeded):
                await client.generate(MESSAGES, max_tokens=10)
        finally:
            active_ledger.reset(a)
            active_control.reset(b)
    asyncio.run(run())
    assert adapter.calls == 1
    assert ledger.upper_usd == reserve
    assert ledger.unknown_requests == 1
    assert ledger.reported_usd == 0


def test_request_settlement_releases_only_known_unused_reservation():
    ledger = RequestLedger(PRICES, 1.0)
    ticket = ledger.admit('unit-model', 'nvidia', 'provider-model', MESSAGES, 10)
    ledger.settle(ticket, GenResult(text='ok', provider='nvidia', model='provider-model', tokens_in=3, tokens_out=2))
    assert ledger.upper_usd == pytest.approx(5/1e6)
    assert ledger.reported_usd == pytest.approx(5/1e6)
    with pytest.raises(ValueError):
        ledger.settle(ticket, None)


def test_byte_ceiling_and_provider_pin_refuse_without_admission():
    ledger = RequestLedger(PRICES, 1.0, max_input_bytes=1100)
    with pytest.raises(BudgetExceeded, match='byte ceiling'):
        ledger.admit('unit-model', 'nvidia', 'provider-model', [{'content': 'x'*200}], 10)
    with pytest.raises(BudgetExceeded, match='frozen price'):
        ledger.admit('unit-model', 'groq', 'provider-model', MESSAGES, 10)
    assert not ledger.records and ledger.upper_usd == 0


def test_bounded_full_groq_plan_retains_actual_completion_caps():
    from app.config import load_router_config
    from app.trace.pipeline import preflight
    prices = {'small': dict(cost_in=.075, cost_out=.30, live_provider='groq'),
              'big': dict(cost_in=.15, cost_out=.60, live_provider='groq')}
    report = preflight(['question']*120, load_router_config(), prices, 'small', 'big', True, 8192)
    assert report['n_items'] == 120 and report['max_calls'] == 2160
    # The configured 2048-token completion cap cannot be silently replaced by
    # 256 to manufacture an affordable plan.
    assert report['admission_estimate_usd'] == pytest.approx(2.899008)
    assert not report['retry_completion_guarantee']


def test_daily_quota_delay_refuses_instead_of_waiting_or_retrying_early():
    from app.llm import ProviderFailureError
    adapter = SequenceAdapter([failed(status=503, retry_after=3600)])
    client = client_for(adapter)
    sleeps = []
    async def sleep(seconds):
        sleeps.append(seconds)
    control = RequestControl(sleep=sleep, max_retry_wait=60)
    async def run():
        token = active_control.set(control)
        try:
            with pytest.raises(ProviderFailureError) as caught:
                await client.generate(MESSAGES)
            assert caught.value.runtime_refused
        finally:
            active_control.reset(token)
    asyncio.run(run())
    assert adapter.calls == 1 and not sleeps
    assert control.events[0]['runtime_refused']
    assert control.events[0]['backoff_seconds'] == 0


def test_nonfinite_retry_headers_cannot_create_unbounded_sleep():
    from app.providers.openai_compat import _retry_after_seconds
    assert _retry_after_seconds('inf') is None
    assert _retry_after_seconds('nan') is None


def test_request_journal_persists_reservation_before_response(tmp_path):
    import json
    path = tmp_path/'requests.jsonl'
    ledger = RequestLedger(PRICES, 1, journal_path=path)
    ledger.admit('unit-model', 'nvidia', 'provider-model', MESSAGES, 10)
    entry = json.loads(path.read_text())
    assert entry['event'] == 'admitted' and entry['upper_usd'] > 0
    assert ledger.report()['unknown_usage_requests'] == 1


def test_operations_latencies_use_item_clusters_and_no_router_claim():
    from app.trace.operations import operational_report
    events = [dict(item_id='a',status='returned',latency_ms=10,tokens_in=3,tokens_out=2),
              dict(item_id='a',status='returned',latency_ms=20,tokens_in=4,tokens_out=1)]
    report = operational_report(events, 1, .1, [], None)
    assert report['latency']['p50_ms']['point'] == 15
    assert report['latency']['p50_ms']['n'] == 1
    assert report['reported_tokens_in'] == 7
    assert report['legacy_router_overhead']['status'] == 'NOT_MEASURED'
