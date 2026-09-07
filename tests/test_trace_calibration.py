import pytest

from app.trace.calibration import calibration_metrics, fit_gain, isotonic_fit, isotonic_predict
from app.trace.analysis import FEATURE_SETS


def rows():
    out=[]
    for role,n in [('train',40),('calib',20),('test',20)]:
        for i in range(n):
            features={f: float(i%2) for fs in FEATURE_SETS.values() for f in fs}
            out.append({'item_id':f'{role}-{i}','split':role,'features':features,
                        'deltas':{'verify':float(i%2)}})
    return out


def test_isotonic_changes_ece_on_same_test_data():
    mapping=isotonic_fit([.1,.2,.8,.9],[0,0,1,1])
    raw=[.15,.85]*10
    labels=[0,1]*10
    calibrated=isotonic_predict(mapping,raw)
    before=calibration_metrics(raw,labels)
    after=calibration_metrics(calibrated,labels)
    assert before['ece']['point'] != after['ece']['point']
    assert before['ece']['n']==after['ece']['n']==20
    assert all('predicted' in b and 'observed' in b for b in after['bins'])


def test_gain_fit_keeps_roles_disjoint_and_serializes_map():
    fit=fit_gain(rows(),'verify')
    assert fit['status']=='OK'
    assert fit['counts']=={'train':40,'calib':20,'test':20}
    assert set(fit['roles']['calib']).isdisjoint(fit['test_item_ids'])
    assert fit['map']['x'] and fit['model']['weights']
    assert fit['calibrated']['ece']['n']==20


@pytest.mark.parametrize('mode',['one_split','no_variation','leakage'])
def test_calibration_refuses_degenerate_or_leaking_data(mode):
    data=rows()
    if mode=='one_split':
        for r in data: r['split']='train'
    elif mode=='no_variation':
        for r in data: r['deltas']['verify']=0.0
    else:
        data[-1]['item_id']=data[0]['item_id']
    assert fit_gain(data,'verify')['status'].startswith('REFUSED')


def test_isotonic_monotone_with_ties():
    mapping=isotonic_fit([0,0,1,2],[1,0,0,1])
    y=isotonic_predict(mapping,[-1,0,.5,1,2,3])
    assert y==sorted(y)
