"""Operational accounting; latency intervals resample complete item clusters."""
from collections import Counter, defaultdict

from .analysis import bootstrap_ci


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values)-1)*q
    lo = int(index)
    return values[lo]+(values[min(lo+1, len(values)-1)]-values[lo])*(index-lo)


def operational_report(events, wall_seconds, throttle_seconds, traces, ledger):
    requests = [e for e in events if e.get('status') in
                ('returned', 'rate_limited', 'provider_failure', 'provider_timeout')]
    groups = defaultdict(list)
    for i, event in enumerate(requests):
        groups[event.get('item_id') or f'ungrouped-{i}'].append(event['latency_ms'])
    latency = {}
    for name, q in [('p50_ms', .5), ('p95_ms', .95)]:
        latency[name] = bootstrap_ci(list(groups.values()),
            statistic=lambda sample, q=q: quantile([v for group in sample for v in group], q))
    overhead = [sum(s.outcome.cost.latency_ms for s in t.steps
                    if s.decision.rationale.get('policy_overhead'))
                for t in traces if t.branch_id == 'prefix']
    return {'n_items': len({t.item_id for t in traces}), 'n_request_attempts': len(requests),
        'wall_seconds': wall_seconds, 'throttle_seconds': throttle_seconds,
        'request_status_counts': dict(Counter(e['status'] for e in requests)),
        'http_status_counts': dict(Counter(str(e['http_status']) for e in requests if e.get('http_status'))),
        'reported_tokens_in': sum(e.get('tokens_in', 0) for e in requests),
        'reported_tokens_out': sum(e.get('tokens_out', 0) for e in requests),
        'latency': latency, 'latency_ci_unit': 'item clusters; descriptive, not an availability guarantee',
        'scoring_overhead_ms_per_item': bootstrap_ci(overhead),
        'legacy_router_overhead': {'status': 'NOT_MEASURED',
            'reason': 'fan-out executor does not execute the production router'},
        'request_budget': ledger,
        'accounting_scope': 'counts/tokens/time are observed accounting totals, not statistical estimates; unknown token usage is not zero'}
