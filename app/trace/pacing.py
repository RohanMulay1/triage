"""Run-scoped pacing and bounded provider retries; inactive on legacy routing."""
import asyncio
import json
from contextvars import ContextVar
import math
import time

active_control = ContextVar("triage_run_request_control", default=None)


class EventJournal(list):
    """Persist each completed attempt before the next call can start."""
    def __init__(self, path):
        super().__init__()
        self.path = path
        with path.open('x', encoding='utf-8'):
            pass

    def append(self, event):
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(event, allow_nan=False)+'\n')
        super().append(event)


class RequestControl:
    def __init__(self, rps=.6, attempts=3, sleep=asyncio.sleep, clock=time.monotonic,
                 max_retry_wait=60.0, request_timeout=180.0):
        if not math.isfinite(rps) or rps <= 0 or attempts < 1:
            raise ValueError("finite positive rps and positive attempts required")
        self.rps, self.attempts = rps, attempts
        if not math.isfinite(max_retry_wait) or max_retry_wait <= 0:
            raise ValueError('finite positive retry-wait bound required')
        if not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError('finite positive request-timeout bound required')
        self.max_retry_wait = max_retry_wait
        self.request_timeout = request_timeout
        self.sleep, self.clock = sleep, clock
        self.next_start = 0.0
        self.lock = asyncio.Lock()
        self.events = []
        self.throttle_seconds = 0.0
        self.item_id = None

    async def acquire(self):
        async with self.lock:
            delay = max(0.0, self.next_start-self.clock())
            if delay:
                await self.sleep(delay)
                self.throttle_seconds += delay
            self.next_start = self.clock()+1/self.rps

    async def call(self, client, operation):
        from ..llm import ProviderFailureError, RateLimitError
        from .budgeting import BudgetExceeded
        for attempt in range(1,self.attempts+1):
            await self.acquire()
            start = self.clock()
            try:
                result = await asyncio.wait_for(operation(), timeout=self.request_timeout)
            except asyncio.TimeoutError:
                from ..providers.base import GenResult
                from ..llm import RetryableProviderError
                exc = RetryableProviderError(client.provider_label, GenResult(
                    text="", provider=client.provider_label, model=client.model_id,
                    latency_ms=1000*(self.clock()-start), usage_known=False,
                    error=f"request exceeded {self.request_timeout:g}s runtime bound",
                ))
                client.unknown_usage = True
                wait = min(30.0, 2.0**(attempt-1))
                self.events.append({"item_id":self.item_id,"model":client.model_id,
                    "provider":exc.provider,"attempt":attempt,"status":"provider_timeout",
                    "latency_ms":1000*(self.clock()-start),"http_status":None,
                    "retryable":True,"usage_known":False,"error":str(exc),
                    "retry_after":None,"runtime_refused":False,
                    "backoff_seconds":wait if attempt<self.attempts else 0})
                if attempt == self.attempts:
                    raise exc
                await self.sleep(wait)
            except BudgetExceeded as exc:
                result = getattr(exc, 'result', None)
                if result is not None:
                    self.events.append({'item_id': self.item_id, 'model': client.model_id,
                        'provider': client.provider_label, 'attempt': attempt,
                        'status': 'returned', 'budget_overrun': True,
                        'latency_ms': 1000*(self.clock()-start),
                        'tokens_in': result.tokens_in, 'tokens_out': result.tokens_out,
                        'usage_known': result.usage_known, 'http_status': result.http_status,
                        'raw_response': result.raw, 'response_text': result.text,
                        'error': str(exc)})
                raise
            except ProviderFailureError as exc:
                if not exc.usage_known:
                    client.unknown_usage = True
                status = "rate_limited" if isinstance(exc, RateLimitError) else "provider_failure"
                wait = max(float(exc.retry_after or 0), min(30.0, 2.0**(attempt-1)))
                runtime_refused = not math.isfinite(wait) or wait > self.max_retry_wait
                exc.runtime_refused = runtime_refused
                self.events.append({"item_id":self.item_id,"model":client.model_id,"provider":exc.provider,
                    "attempt":attempt,"status":status,"latency_ms":1000*(self.clock()-start),
                    "http_status":exc.status_code,"retryable":exc.retryable,
                    "usage_known":exc.usage_known,"error":str(exc),
                    "retry_after":exc.retry_after,
                    "runtime_refused": runtime_refused,
                    "backoff_seconds":wait if exc.retryable and attempt<self.attempts and not runtime_refused else 0})
                if runtime_refused or not exc.retryable or attempt == self.attempts:
                    raise
                await self.sleep(wait)
            else:
                self.events.append({"item_id":self.item_id,"model":client.model_id,"provider":client.provider_label,
                    "attempt":attempt,"status":"returned","latency_ms":1000*(self.clock()-start),
                    "tokens_in":result.tokens_in,"tokens_out":result.tokens_out,
                    "usage_known":result.usage_known,
                    "unknown_usage_before_success":client.unknown_usage,
                    "raw_response":result.raw, "response_text":result.text,
                    "error":result.error,"fallback_error":result.raw.get("fallback_error")})
                if client.unknown_usage:
                    result.raw["unknown_usage_before_success"] = True
                return result
