import asyncio
import json
import random

import pytest

from app.providers.diagnostic_scoring import DiagnosticScoringProvider
from app.trace.heuristic_gain import HeuristicGainPolicy
from app.trace.branching import RunBudget
from app.trace.contract import OutcomeStatus
from test_trace_policies import _ctx, _ex


def test_explicit_diagnostic_provider_is_seeded_and_exercises_real_client():
    ctx,ex=_ctx(),_ex()
    policy=HeuristicGainPolicy(diagnostic_seed=3)
    _,result=asyncio.run(policy.prepare(ctx,ex,[],RunBudget()))
    assert result.provider_label=='mock' and result.status==OutcomeStatus.CHOSEN
    choice=policy.choose(ctx,ex,[],random.Random(0))
    assert choice.rationale['scores']
    _,again=asyncio.run(policy.prepare(ctx,ex,[],RunBudget()))
    assert result.detail['scores']==again.detail['scores']


@pytest.mark.parametrize('bad',[-.01,1.01,'malformed'])
def test_out_of_range_and_malformed_scores_refuse(monkeypatch,bad):
    original=DiagnosticScoringProvider.generate
    async def corrupted(self,*args,**kwargs):
        response=await original(self,*args,**kwargs)
        scores=json.loads(response.text)
        next(iter(scores.values()))['expected_gain']=bad
        response.text='malformed' if bad=='malformed' else json.dumps(scores)
        return response
    monkeypatch.setattr(DiagnosticScoringProvider,'generate',corrupted)
    policy=HeuristicGainPolicy(diagnostic_seed=1)
    _,result=asyncio.run(policy.prepare(_ctx(),_ex(),[],RunBudget()))
    assert result.status==OutcomeStatus.ATTEMPTED
    with pytest.raises(ValueError):
        policy.choose(_ctx(),_ex(),[],random.Random(0))
