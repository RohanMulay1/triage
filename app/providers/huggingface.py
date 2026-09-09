"""HuggingFace Inference adapter. Uses the OpenAI-compatible router endpoint
(`/models/{model}/v1/chat/completions`) when possible. No reliable per-token
logprobs => proxy uncertainty uses resample entropy."""
from __future__ import annotations

import time
from typing import Any

import httpx

from .base import GenResult, Message, ProviderAdapter
from .openai_compat import _retry_after_seconds


class HuggingFaceAdapter(ProviderAdapter):
    name = "huggingface"

    async def generate(
        self,
        model: str,
        messages: list[Message],
        temperature: float = 0.2,
        max_tokens: int = 512,
        want_logprobs: bool = False,
    ) -> GenResult:
        # HF's serverless inference exposes an OpenAI-compatible chat route per model.
        url = f"{self.base_url}/models/{model}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": max(temperature, 0.01),
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
            latency = (time.perf_counter() - t0) * 1000
            if resp.status_code >= 400:
                retry_after = _retry_after_seconds(resp.headers.get("retry-after"))
                prefix = f"RATE_LIMIT retry_after={retry_after}: " if resp.status_code == 429 else ""
                return GenResult(text="", provider=self.name, model=model, latency_ms=latency,
                                 error=f"{prefix}HTTP {resp.status_code}: {resp.text[:300]}",
                                 usage_known=resp.status_code == 429,
                                 http_status=resp.status_code, retry_after=retry_after)
            data = resp.json()
        except Exception as exc:
            return GenResult(text="", provider=self.name, model=model,
                             latency_ms=(time.perf_counter() - t0) * 1000, error=str(exc),
                             usage_known=False)

        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content", "") or ""
        usage = data.get("usage", {}) or {}
        return GenResult(
            text=text.strip(), provider=self.name, model=model,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            ttft_ms=latency, latency_ms=latency, logprobs=None, raw=data,
        )
