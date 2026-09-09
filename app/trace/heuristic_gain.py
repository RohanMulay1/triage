"""Equation-level adaptation of arXiv:2603.19896, not an exact code reproduction.

Source code is search/stop only; this adapter scores Triage's feasible actions.
Scoring is asynchronous, budgeted and recorded before synchronous selection.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import replace

from ..config import get_settings
from ..llm import LLMClient, ProviderFailureError
from .actions import ExecutionResult
from .adapter import cost_delta, snapshot_cost
from .budgeting import BudgetExceeded, estimate_action_cost
from .contract import Action, ActionKind, OutcomeStatus


class HeuristicGainPolicy:
    policy_id = "heuristic_gain_arxiv2603_19896_adapted"
    policy_version = "2"

    def __init__(self, lambda_cost=0.3, lambda_uncertainty=0.2,
                 lambda_redundancy=0.2, max_steps=3, diagnostic_seed=None):
        self.weights = [lambda_cost, lambda_uncertainty, lambda_redundancy]
        if any(not math.isfinite(x) or x < 0 for x in self.weights):
            raise ValueError("utility weights must be finite and nonnegative")
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        self.max_steps = max_steps
        self.diagnostic_seed = diagnostic_seed
        self._prepared = None

    def _request(self, ctx, executor, taken):
        actions = executor.available_actions(ctx, taken)
        state = {"question": ctx.question, "draft": ctx.answer,
                 "evidence": [h.text for h in ctx.evidence], "samples": ctx.samples,
                 "signals": ctx.signals.model_dump(), "history": taken,
                 "actions": [{"key": a.key, "kind": a.kind.value} for a in actions]}
        system = ("Estimate the marginal answer-quality gain and uncertainty for each action. "
                  "These are uncalibrated self-estimates, not observed outcomes. "
                  "Return only a JSON object keyed by each exact action key; each value has "
                  "numeric expected_gain and uncertainty in [0,1]. Do not answer the question. "
                  "RESAMPLE, SELF_CHECK and RETRIEVE preserve the draft and acquire information; "
                  "estimate their potential benefit if consumed by a later action.")
        return actions, system, json.dumps(state, sort_keys=True)

    async def prepare(self, ctx, executor, taken, budget):
        """Returns a forced assessment action/result for the collector to record.

        No scores survive a failed assessment. No mock fallback or malformed JSON
        gets a default gain; actual reported cost remains visible on failure.
        """
        self._prepared = None
        if self.diagnostic_seed is not None and not get_settings().force_mock:
            raise ValueError("diagnostic scores require forced mock; forbidden live")
        actions, system, prompt = self._request(ctx, executor, taken)
        action = Action(kind=ActionKind.SELF_CHECK, model_id=ctx.small_id,
                        params={"purpose": "heuristic_gain_scoring", "version": 2})
        cfg = {**ctx.cfg, "proxy": {**ctx.cfg["proxy"], "max_tokens": 1024}}
        estimate_ctx = replace(ctx, question=prompt, system=system, cfg=cfg)
        estimate_action = Action(kind=ActionKind.ANSWER, model_id=ctx.small_id)
        budget.admit(estimate_action_cost(estimate_action, estimate_ctx, budget.model_prices))
        client = LLMClient(ctx.small_id)
        if self.diagnostic_seed is not None:
            from types import SimpleNamespace
            from ..providers.diagnostic_scoring import DiagnosticScoringProvider
            provider = DiagnosticScoringProvider(self.diagnostic_seed)
            client.registry = SimpleNamespace(resolve=lambda model: (provider, model, "mock"),
                                              mock=provider)
        before = snapshot_cost(client)
        payload, error, latency = None, None, 0.0
        raw_text = None
        runtime_refused = False
        status = OutcomeStatus.CHOSEN
        try:
            response = await client.generate(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=1024)
            raw_text = response.text
            latency = response.latency_ms
            if client.provider_label == "mock" and not get_settings().force_mock:
                status = OutcomeStatus.SYNTHETIC_FALLBACK
                raise ValueError("synthetic fallback during policy scoring")
            if response.error:
                raise ValueError(response.error)
            payload = decode_score_json(response.text)
            if not isinstance(payload, dict) or set(payload) != {a.key for a in actions}:
                raise ValueError("scores must cover exactly the feasible action keys")
            scores = {}
            for a in actions:
                values = [payload[a.key][k] for k in ("expected_gain", "uncertainty")]
                if any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                    raise ValueError("nonfinite, out-of-range or nonnumeric self-estimate")
                scores[a.key] = [min(1.0, max(0.0, v)) for v in values]
            self._prepared = (prompt, scores)
        except BudgetExceeded:
            raise
        except ProviderFailureError as exc:
            error = str(exc)
            status = OutcomeStatus.FAILED
            runtime_refused = getattr(exc, 'runtime_refused', False)
        except Exception as exc:
            error = str(exc)
            if status == OutcomeStatus.CHOSEN:
                status = OutcomeStatus.ATTEMPTED
        cost = cost_delta(before, client.cost, client.provider_label).model_copy(
            update={"latency_ms": latency})
        budget.settle(cost)
        return action, ExecutionResult(
            status=status, answer=ctx.answer, signals=ctx.signals, evidence=ctx.evidence,
            samples=ctx.samples, cost=cost, provider_label=client.provider_label, error=error,
            detail={"purpose": "policy_overhead", "scores": payload if error is None else None,
                    "raw_response": raw_text,
                    "runtime_refused": runtime_refused,
                    "unknown_usage": getattr(client, 'unknown_usage', False) or (error is not None and client.cost.llm_calls == 0 and status != OutcomeStatus.FAILED),
                    "weights": self.weights, "max_steps": self.max_steps,
                    "adaptation": "heterogeneous_actions_exact_key_redundancy",
                    "uncalibrated": True})

    def choose(self, ctx, executor, taken, rng):
        from .policies import PolicyChoice
        actions, _, prompt = self._request(ctx, executor, taken)
        if self._prepared is None or self._prepared[0] != prompt:
            raise ValueError("budgeted scoring must prepare this exact state first")
        scores = self._prepared[1]
        rows = {}
        used = max(0, len(taken) - 1)  # shared initial draft is not an intervention
        for a in actions:
            gain, uncertainty = scores[a.key]
            stop = a.kind == ActionKind.STOP
            step = 0.0 if stop else (used + 1) / self.max_steps
            redundancy = float(a.key in taken)
            utility = gain - self.weights[0]*step - self.weights[1]*uncertainty - self.weights[2]*redundancy
            rows[a.key] = dict(gain=gain, uncertainty=uncertainty, step_cost=step,
                               redundancy=redundancy, utility=utility)
        candidates = [a for a in actions if used < self.max_steps or a.kind == ActionKind.STOP]
        if not candidates:
            raise ValueError("step budget exhausted without feasible STOP")
        chosen = max(candidates, key=lambda a: (rows[a.key]["utility"], a.kind == ActionKind.STOP, a.key))
        return PolicyChoice(chosen, 1.0, actions,
                            {"scores": rows, "weights": self.weights,
                             "propensity_condition": "conditional_on_recorded_llm_scores",
                            "uncalibrated": True, "adapted_from": "arxiv:2603.19896v1"})


def decode_score_json(text):
    """Accept raw JSON or one complete JSON fence, never extract from prose.

    This is transport-envelope handling only. All action coverage and numeric
    validation still happens in prepare, with no repaired/default score values.
    """
    fenced = re.fullmatch(r"\s*```json\s*\n(.*?)\n```\s*", text, re.DOTALL)
    return json.loads(fenced.group(1) if fenced else text)
