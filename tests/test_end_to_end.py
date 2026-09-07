"""Cold-start assembled research pipeline; synthetic outcomes never become evidence."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from app.trace import store
from app.trace.support import SupportError, assert_estimable
from test_trace_adapter import _traced, _untraced, CONFIDENT_Q, UNCERTAIN_Q, MATH_Q


def test_whole_pipeline_from_cold_start(tmp_path, monkeypatch):
    repo=Path(__file__).resolve().parents[1]
    protected=[repo/p for p in subprocess.check_output(
        ['git','ls-files','data'],cwd=repo,text=True).splitlines() if (repo/p).is_file()]
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    trace_root=tmp_path/'traces'
    env={**os.environ,'TRIAGE_FORCE_MOCK':'1','TRIAGE_TRACE_DIR':str(trace_root),
         'TELEMETRY_DB':str(tmp_path/'telemetry.db')}
    result=subprocess.run([sys.executable,'scripts/run_gate2.py','--n','8','--dataset','mixed',
        '--depth','2','--policy','balanced','--diagnostic-scores','--run-id','e2e-cold'],
        cwd=repo,env=env,capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    assert 'Preflight estimate' in result.stdout
    root=trace_root/'e2e-cold'
    report=json.loads((root/'report.json').read_text())
    assert report['validation']['chain_ok'] and report['validation']['cost_conservation_ok']
    assert report['validation']['errors']==[]
    assert report['not_evidence'] and not report['analysis_grade']
    monkeypatch.setattr(store,'trace_root',lambda:trace_root)
    traces=store.read_run('e2e-cold')
    assert any(t.decision_depth==2 for t in traces)
    assert any(s.outcome.detail.get('utility_choice',{}).get('scores') for t in traces for s in t.steps)
    with pytest.raises(SupportError):
        assert_estimable(traces)
    assert report['evidence_support']['status']=='REFUSED'
    assert report['positivity'] and report['missingness']
    assert report['gate2']['verdict']!='GO'
    assert report['gate2']['calibration']==report['calibration']
    assert json.loads((root/'calibration.json').read_text())==report['calibration']
    assert report['calibration']['per_action']
    # Undefined intervals must be explicit; treating missing support as zero is a failure.
    for candidate in ('C1','C4'):
        block=report['ablations'][candidate]
        assert 'ci95' in block
        if block['status']=='REFUSED':
            assert block['ci95']['n']==0 and block['ci95']['point'] is None
        assert not block['established']
    for action in ('resample','self_check','retrieve'):
        ci=report['ablations']['C2']['per_action'][action]['paired_depth2_minus_depth1_ci95']
        assert ci['n']==8 and ci['lo']<=ci['point']<=ci['hi']
    assert sum(t.total_cost.llm_calls for t in traces)==report['budget']['spent_llm_calls']
    for filename in ('manifest.json','splits.json','trajectories.jsonl','collect.log','corpus.json'):
        assert (root/filename).is_file()
    for question in (CONFIDENT_Q,UNCERTAIN_Q,MATH_Q):
        traced,_=_traced(question)
        plain=_untraced(question)
        for field in ('tier','tier_name','escalated','abstained','retrieved','verified'):
            assert getattr(traced.route,field)==getattr(plain.route,field)
        assert (traced.status,traced.model,traced.cost.llm_calls)==(plain.status,plain.model,plain.cost.llm_calls)
    assert before=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
