"""Explicit synthetic scoring provider; never an automatic fallback."""
import hashlib
import json

from .base import GenResult
from .mock import MockProvider


class DiagnosticScoringProvider(MockProvider):
    def __init__(self, seed):
        super().__init__()
        self.seed = int(seed)

    async def generate(self, model, messages, temperature=0.0, max_tokens=1024,
                       want_logprobs=False):
        state = json.loads(messages[-1]["content"])
        scores = {}
        for action in state["actions"]:
            digest = hashlib.sha256((str(self.seed) + json.dumps(state, sort_keys=True)
                                     + action["key"]).encode()).digest()
            scores[action["key"]] = {"expected_gain": int.from_bytes(digest[:4], "big") / 2**32,
                                      "uncertainty": int.from_bytes(digest[4:8], "big") / 2**32}
        text = json.dumps(scores, sort_keys=True)
        return GenResult(text=text, provider="mock", model=model,
                         tokens_in=sum(len(m["content"].encode()) for m in messages),
                         tokens_out=max(1, len(text.encode())//4),
                         raw={"mock": True, "diagnostic_scores": True, "seed": self.seed})
