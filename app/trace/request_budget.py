"""Live request admission, including retries and unknown-usage reservations.

An explicit input-byte ceiling makes a finite planning bound enforceable without
assuming that generated tokens always retokenize identically. Inputs are never
truncated: an oversized request refuses before contacting the provider.
"""
from contextvars import ContextVar
import json
import math

from .budgeting import BudgetExceeded


active_ledger = ContextVar("triage_live_request_ledger", default=None)
CHAT_ALLOWANCE = 1024


def input_allowance(messages):
    return CHAT_ALLOWANCE + sum(len(str(m.get("content", "")).encode("utf-8"))
                                + len(str(m.get("role", "")).encode("utf-8"))
                                for m in messages)


class RequestLedger:
    def __init__(self, prices, max_usd, max_input_bytes=None, journal_path=None):
        if not math.isfinite(max_usd) or max_usd <= 0:
            raise ValueError("finite positive request budget required")
        if max_input_bytes is not None and (
                not isinstance(max_input_bytes, int) or max_input_bytes <= CHAT_ALLOWANCE):
            raise ValueError("input-byte ceiling must exceed chat allowance")
        self.prices = prices
        self.max_usd = max_usd
        self.max_input_bytes = max_input_bytes
        self.upper_usd = 0.0
        self.reported_usd = 0.0
        self.unknown_requests = 0
        self.records = []
        self.journal_path = journal_path
        if journal_path is not None:
            with journal_path.open('x', encoding='utf-8'):
                pass

    def _journal(self, event, ticket):
        if self.journal_path is not None:
            with self.journal_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'event': event, 'ticket': ticket,
                    'upper_usd': self.upper_usd}, allow_nan=False)+'\n')

    def admit(self, model_id, provider, provider_model, messages, max_tokens):
        price = self.prices[model_id]
        if ('price_provider' in price and (price['price_provider'], price['price_model'])
                != (provider, provider_model)):
            raise BudgetExceeded('resolved alternate provider lacks an independent declared price')
        if (price["live_provider"], price["live_model"]) != (provider, provider_model):
            raise BudgetExceeded("provider/model differs from frozen price snapshot")
        tokens = input_allowance(messages)
        if self.max_input_bytes is not None and tokens > self.max_input_bytes:
            raise BudgetExceeded("request exceeds declared input-byte ceiling; no truncation or provider call")
        rates = [float(price[k]) for k in ("cost_in", "cost_out")]
        if any(not math.isfinite(v) or v < 0 for v in rates) or max_tokens <= 0:
            raise BudgetExceeded("invalid snapshotted request price or completion limit")
        reserve = (tokens*rates[0]+max_tokens*rates[1])/1e6
        if self.upper_usd + reserve > self.max_usd:
            raise BudgetExceeded("request/retry reservation exceeds remaining run-wide cap")
        self.upper_usd += reserve
        ticket = {"model_id": model_id, "provider": provider, "input_allowance": tokens,
                  "max_tokens": max_tokens, "messages": messages,
                  "reserved_usd": reserve, "settled": False}
        self.records.append(ticket)
        self._journal('admitted', ticket)
        return ticket

    def settle(self, ticket, response):
        if ticket["settled"]:
            raise ValueError("request reservation already settled")
        ticket["settled"] = True
        known = response.usage_known
        ticket["usage_known"] = known
        ticket["http_status"] = response.http_status
        if known:
            price = self.prices[ticket["model_id"]]
            reported = (response.tokens_in*price["cost_in"]+
                        response.tokens_out*price["cost_out"])/1e6
            self.upper_usd += reported-ticket["reserved_usd"]
            self.reported_usd += reported
            ticket["reported_usd"] = reported
            if self.upper_usd > self.max_usd:
                self._journal('settled_overrun', ticket)
                raise BudgetExceeded("reported usage exceeded admission bound; stopped with cost retained")
        else:
            self.unknown_requests += 1
            ticket["reported_usd"] = None
        self._journal('settled', ticket)

    def report(self):
        return {"max_usd": self.max_usd, "max_input_bytes": self.max_input_bytes,
                "reported_listed_usd": self.reported_usd,
                "reported_plus_reserved_upper_usd": self.upper_usd,
                "unknown_usage_requests": self.unknown_requests + sum(not t['settled'] for t in self.records),
                "accounting": "unknown usage retains full reservation; not an invoice",
                "requests": self.records}
