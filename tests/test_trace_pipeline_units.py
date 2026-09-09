import pytest
from app.config import load_router_config
from app.trace.pipeline import (PipelineOptions, preflight, validate_options,
                                validate_item_concurrency)
from app.trace.ablations import compare_fits
from test_trace_calibration import rows
from app.trace.calibration import fit_gain


@pytest.mark.parametrize('cap',[None,0,-1,float('inf'),float('nan')])
def test_pipeline_rejects_unapproved_or_nonfinite_live_cap(cap):
    with pytest.raises(ValueError):
        validate_options(PipelineOptions('unit',live=True,max_usd=cap))


def test_pipeline_refuses_live_diagnostic_provider():
    with pytest.raises(ValueError):
        validate_options(PipelineOptions('unit',live=True,max_usd=1,diagnostic_scores=True))


@pytest.mark.parametrize('timeout',[0,-1,float('inf'),float('nan')])
def test_pipeline_rejects_invalid_request_timeout(timeout):
    with pytest.raises(ValueError):
        validate_options(PipelineOptions('unit',request_timeout=timeout))


@pytest.mark.parametrize('value',[0,-1])
def test_pipeline_rejects_invalid_item_concurrency(value):
    with pytest.raises(ValueError):
        validate_options(PipelineOptions('unit',item_concurrency=value))


def test_item_concurrency_is_live_and_zero_price_only():
    zero={m:{'cost_in':0,'cost_out':0} for m in ('s','b')}
    validate_item_concurrency(PipelineOptions('unit',live=True,max_usd=1,
                              item_concurrency=4),zero,'s','b')
    with pytest.raises(ValueError,match='zero-price'):
        validate_item_concurrency(PipelineOptions('unit',item_concurrency=4),
                                  zero,'s','b')
    priced={**zero,'b':{'cost_in':0,'cost_out':.1}}
    with pytest.raises(ValueError,match='priced'):
        validate_item_concurrency(PipelineOptions('unit',live=True,max_usd=1,
                                  item_concurrency=4),priced,'s','b')


def test_whole_run_estimate_counts_all_calls():
    cfg=load_router_config()
    prices={m:{'cost_in':1,'cost_out':1,'live_provider':'unit'} for m in ('s','b')}
    estimate=preflight(['a','b'],cfg,prices,'s','b',True)
    assert estimate['max_calls']==2*(cfg['proxy']['resamples']+15)
    assert estimate['admission_estimate_usd']>0


def test_supported_head_to_head_has_paired_intervals_and_zero_disagreement():
    data=rows()
    fit={'verify':fit_gain(data,'verify')}
    report=compare_fits(fit,fit,data)
    assert report['status']=='OK'
    ci=report['policy_disagreement']['ci95']
    assert ci['n']==20 and ci['lo']==ci['hi']==0
    assert not report['established']


def test_unsupported_fit_does_not_invent_interval():
    result=compare_fits({'verify':{'status':'REFUSED'}},{},rows())
    assert result['status']=='REFUSED'
    assert result['per_action']['verify']['ci95']['point'] is None


def test_whole_run_cap_refuses_before_any_generation(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from app.trace import pipeline
    from app.trace.branching import BudgetExceeded
    monkeypatch.setattr(pipeline,'get_settings',lambda:SimpleNamespace(force_mock=False))
    prices={m:{'cost_in':100,'cost_out':100,'live_provider':'unit'} for m in ('s','b')}
    monkeypatch.setattr(pipeline.store,'model_snapshot',lambda:prices)
    async def forbidden(**kwargs):
        pytest.fail('generation occurred before whole-run cap refusal')
    monkeypatch.setattr(pipeline,'collect_fanout',forbidden)
    with pytest.raises(BudgetExceeded,match='no calls made'):
        asyncio.run(pipeline.run_pipeline(PipelineOptions('never-created',live=True,
            max_usd=.001,small='s',big='b')))


def test_head_to_head_can_measure_nonzero_disagreement_when_supported():
    data=rows()
    fit=fit_gain(data,'verify')
    left={'verify':{**fit,'calibrated_gain':[.5]*20}}
    right={'verify':{**fit,'calibrated_gain':[-.5]*20}}
    result=compare_fits(left,right,data)
    assert result['policy_disagreement']['ci95']['lo']==1
    assert result['policy_disagreement']['ci95']['n']==20
