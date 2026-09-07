import asyncio
from types import SimpleNamespace

import pytest

from app.llm import LLMClient, RateLimitError
from app.providers.base import GenResult
from app.trace.actions import InterventionExecutor
from app.trace.contract import ActionKind, OutcomeStatus
from app.trace.pacing import RequestControl, active_control
from test_trace_policies import _ctx, _ex
from test_trace_measurement_repairs import collect


@pytest.mark.parametrize('kind',[ActionKind.RESAMPLE,ActionKind.SELF_CHECK,ActionKind.VERIFY])
def test_rate_limited_action_is_failed_and_collection_continues(monkeypatch,kind):
    original=LLMClient.generate
    async def rate_limited(self,*args,**kwargs):
        raise RateLimitError('nvidia',0)
    method={ActionKind.RESAMPLE:'_resample',ActionKind.SELF_CHECK:'_self_check',ActionKind.VERIFY:'_verify'}[kind]
    original_action=getattr(InterventionExecutor,method)
    async def isolated(self,action,ctx):
        with monkeypatch.context() as patch:
            patch.setattr(LLMClient,'generate',rate_limited)
            return await original_action(self,action,ctx)
    monkeypatch.setattr(InterventionExecutor,method,isolated)
    traces=collect(depth=2)
    failed=[t for t in traces if t.steps[0].decision.chosen.kind==kind]
    assert failed and all(t.steps[0].outcome.status==OutcomeStatus.FAILED for t in failed)
    assert all(t.terminal.label is None for t in failed)
    assert all(t.steps[0].outcome.provider_label=='nvidia' for t in failed)
    assert any(t.terminal.label is not None for t in traces)
    assert LLMClient.generate is original


def test_retry_honors_header_and_resolves_without_failed_action(monkeypatch):
    sleeps=[]
    async def sleep(delay): sleeps.append(delay)
    count=0
    async def once(self,*args):
        nonlocal count
        count+=1
        self.provider_label='nvidia'
        if count==1: raise RateLimitError('nvidia',7)
        return GenResult(text='No contradiction found.',provider='nvidia',model=self.model_id)
    monkeypatch.setattr(LLMClient,'_generate_once',once)
    async def run():
        control=RequestControl(.6,sleep=sleep)
        token=active_control.set(control)
        try:
            ctx=_ctx(); executor=_ex()
            action=next(a for a in executor.available_actions(ctx,[]) if a.kind==ActionKind.SELF_CHECK)
            result=await executor.execute(action,ctx)
            return result,control
        finally: active_control.reset(token)
    result,control=asyncio.run(run())
    assert result.status==OutcomeStatus.CHOSEN
    assert count==2 and 7 in sleeps
    assert [e['status'] for e in control.events]==['rate_limited','returned']


def test_retry_is_bounded_and_throttle_is_shared():
    now=[0.0]
    waits=[]
    async def sleep(delay): waits.append(delay); now[0]+=delay
    control=RequestControl(.5,attempts=3,sleep=sleep,clock=lambda:now[0])
    client=SimpleNamespace(model_id='unit',provider_label='nvidia')
    async def limited(): raise RateLimitError('nvidia',0)
    async def run():
        with pytest.raises(RateLimitError): await control.call(client,limited)
        await control.acquire()
    asyncio.run(run())
    assert len(control.events)==3
    assert all(delay>=0 for delay in waits)
    assert control.next_start>=6


@pytest.mark.parametrize('rps',[0,-1,float('inf'),float('nan')])
def test_invalid_throttle_refuses(rps):
    with pytest.raises(ValueError): RequestControl(rps)


def test_partial_resample_cost_is_preserved_on_exhaustion(monkeypatch):
    async def partly_successful(self,*args,**kwargs):
        self.provider_label='nvidia'
        if self.cost.llm_calls:
            raise RateLimitError('nvidia',2)
        self.cost.llm_calls=1
        self.cost.tokens_in=10
        self.cost.tokens_out=3
        return GenResult(text='one sample',provider='nvidia',model=self.model_id)
    monkeypatch.setattr(LLMClient,'generate',partly_successful)
    ctx,executor=_ctx(),_ex()
    action=next(a for a in executor.available_actions(ctx,[]) if a.kind==ActionKind.RESAMPLE)
    result=asyncio.run(executor.execute(action,ctx))
    assert result.status==OutcomeStatus.FAILED
    assert result.cost.llm_calls==1 and result.cost.tokens_in==10


def test_original_provider_error_survives_mock_fallback():
    from app.providers.mock import MockProvider
    class Gone:
        async def generate(self,*args):
            return GenResult(text='',provider='nvidia',model='unit',error='HTTP 410: retired')
    client=LLMClient('llama-3.1-8b')
    client.registry=SimpleNamespace(resolve=lambda model:(Gone(),model,'nvidia'),mock=MockProvider())
    result=asyncio.run(client.generate([{'role':'user','content':'Reply OK.'}]))
    assert client.provider_label=='mock'
    assert result.raw['fallback_error']=='HTTP 410: retired'
