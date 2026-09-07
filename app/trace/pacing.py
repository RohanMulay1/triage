"""Run-scoped pacing and bounded 429 retries; inactive on the legacy router."""
import asyncio
from contextvars import ContextVar
import math
import time

active_control = ContextVar("triage_run_request_control", default=None)


class RequestControl:
    def __init__(self, rps=.6, attempts=3, sleep=asyncio.sleep, clock=time.monotonic):
        if not math.isfinite(rps) or rps <= 0 or attempts < 1:
            raise ValueError("finite positive rps and positive attempts required")
        self.rps, self.attempts = rps, attempts
        self.sleep, self.clock = sleep, clock
        self.next_start = 0.0
        self.lock = asyncio.Lock()
        self.events = []
        self.throttle_seconds = 0.0

    async def acquire(self):
        async with self.lock:
            delay = max(0.0, self.next_start-self.clock())
            if delay:
                await self.sleep(delay)
                self.throttle_seconds += delay
            self.next_start = self.clock()+1/self.rps

    async def call(self, client, operation):
        from ..llm import RateLimitError
        for attempt in range(1,self.attempts+1):
            await self.acquire()
            start = self.clock()
            try:
                result = await operation()
            except RateLimitError as exc:
                wait = max(float(exc.retry_after or 0), min(30.0, 2.0**(attempt-1)))
                self.events.append({"model":client.model_id,"provider":exc.provider,
                    "attempt":attempt,"status":"rate_limited","latency_ms":1000*(self.clock()-start),
                    "retry_after":exc.retry_after,"backoff_seconds":wait if attempt<self.attempts else 0})
                if attempt == self.attempts:
                    raise
                await self.sleep(wait)
            else:
                self.events.append({"model":client.model_id,"provider":client.provider_label,
                    "attempt":attempt,"status":"returned","latency_ms":1000*(self.clock()-start),
                    "tokens_in":result.tokens_in,"tokens_out":result.tokens_out,
                    "error":result.error,"fallback_error":result.raw.get("fallback_error")})
                return result
