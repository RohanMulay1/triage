import asyncio
import json

import httpx
import pytest

from app.config import default_small_model, get_model
from app.providers.openai_compat import OpenAICompatAdapter
from app.providers.registry import ProviderRegistry
from app.trace.store import model_snapshot


def test_groq_capability_and_explicit_lane_provenance(monkeypatch):
    monkeypatch.setenv('TRIAGE_FORCE_MOCK', '0')
    monkeypatch.setenv('GROQ_API_KEY', 'test-key')
    from app.config import get_settings
    get_settings.cache_clear()
    registry = ProviderRegistry()
    adapter, model, label = registry.resolve('groq-gpt-oss-20b')
    assert (model, label) == ('openai/gpt-oss-20b', 'groq')
    assert not adapter.supports_logprobs
    assert adapter.request_options == {'reasoning_effort': 'low'}
    get_settings.cache_clear()


@pytest.mark.parametrize('options', [
    {'reasoning_effort': 'low'},
    {'stream': False, 'chat_template_kwargs': {'enable_thinking': False}},
])
def test_documented_payload_keeps_cap_and_excludes_logprobs(monkeypatch, options):
    captured = []
    async def post(self, url, **kwargs):
        captured.append(kwargs['json'])
        return httpx.Response(200, json={
            'choices': [{'message': {'content': 'OK'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 12, 'completion_tokens': 4},
        })
    monkeypatch.setattr(httpx.AsyncClient, 'post', post)
    adapter = OpenAICompatAdapter('groq', 'test', 'https://example.invalid',
                                 supports_logprobs=False, request_options=options)
    result = asyncio.run(adapter.generate('model', [{'role': 'user', 'content': 'Hi'}],
                                          max_tokens=64, want_logprobs=True))
    payload = captured[0]
    assert payload['max_tokens'] == 64 and payload['model'] == 'model'
    assert 'logprobs' not in payload and 'top_logprobs' not in payload
    assert all(payload[k] == v for k, v in options.items())
    assert result.text == 'OK' and result.tokens_out == 4


@pytest.mark.parametrize('options', [{'max_tokens': 99999}, {'model': 'other'}, {'stream': True}])
def test_extension_cannot_override_identity_budget_or_response_contract(options):
    adapter = OpenAICompatAdapter('nvidia', 'test', 'https://example.invalid',
                                 request_options=options)
    with pytest.raises(ValueError):
        asyncio.run(adapter.generate('model', []))


def test_research_models_do_not_remap_legacy_default():
    assert default_small_model()['id'] == 'llama-3.1-8b'
    assert get_model('llama-3.1-8b')['provider_model'] == 'meta/llama-3.1-8b-instruct'
    assert get_model('groq-gpt-oss-120b')['cost_out'] == .60
    snap = model_snapshot()
    assert snap['nim-nemotron-super-120b']['request_options']['stream'] is False
    assert snap['groq-gpt-oss-20b']['supports_logprobs'] is False
    json.dumps(snap, allow_nan=False)


def test_scoring_json_envelope_preserves_values():
    from app.trace.heuristic_gain import decode_score_json
    payload = {'stop': {'expected_gain': 0.1, 'uncertainty': 0.2}}
    assert decode_score_json('```json\n'+json.dumps(payload)+'\n```') == payload


@pytest.mark.parametrize('text', [
    'Here are scores: ```json\n{}\n```',
    '```json\n{}\n``` trailing explanation',
    '```json\n{bad json}\n```',
    '```json\n{}\n```\n```json\n{}\n```',
])
def test_scoring_envelope_does_not_repair_malformed_output(text):
    from app.trace.heuristic_gain import decode_score_json
    with pytest.raises(ValueError):
        decode_score_json(text)


def test_fenced_out_of_range_scores_still_refuse(monkeypatch):
    from app.trace import heuristic_gain as hg
    from app.trace.branching import RunBudget
    from app.trace.contract import OutcomeStatus
    from test_trace_heuristic_gain import ScriptedClient
    from test_trace_policies import _ctx, _ex
    class FencedClient(ScriptedClient):
        async def generate(self, messages, **kwargs):
            result = await super().generate(messages, **kwargs)
            payload = json.loads(result.text)
            next(iter(payload.values()))['expected_gain'] = 1.1
            result.text = '```json\n'+json.dumps(payload)+'\n```'
            return result
    FencedClient.mode = 'valid'
    monkeypatch.setattr(hg, 'LLMClient', FencedClient)
    _, result = asyncio.run(hg.HeuristicGainPolicy().prepare(_ctx(), _ex(), [], RunBudget()))
    assert result.status == OutcomeStatus.ATTEMPTED
    assert result.detail['scores'] is None
    assert 'out-of-range' in result.error
