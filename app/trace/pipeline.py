"""One run-wide collector and gated consolidated analysis; no live authorization implicit."""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import default_small_model, get_settings, load_router_config
from ..retrieval import store as retrieval
from ..retrieval.seed import SEED_CORPUS
from . import store
from .ablations import c1_report, c4_report
from .analysis import gate2_report
from .branching import BudgetExceeded, RunBudget, collect_fanout
from .calibration import calibration_report
from .contract import Budget
from .heuristic_gain import HeuristicGainPolicy
from .information_value import c2_report
from .pacing import RequestControl, active_control
from .policies import build_policy
from .splits import GroupKey, SplitManifest, prompt_fingerprint
from .support import SupportError, action_support, assert_estimable, missingness, positivity_violations


@dataclass
class PipelineOptions:
    run_id: str
    n: int = 8
    dataset: str = "mixed"
    depth: int = 2
    policy: str = "balanced"
    live: bool = False
    max_usd: float | None = None
    diagnostic_scores: bool = False
    seed: int = 0
    small: str | None = None
    big: str | None = None
    rps: float = .6


def validate_options(args):
    if not math.isfinite(args.rps) or args.rps <= 0:
        raise ValueError("rps must be finite and strictly positive")
    if not isinstance(args.n, int) or args.n <= 0:
        raise ValueError("n must be positive integer")
    if args.depth != 2 or args.policy != 'balanced':
        raise ValueError("assembled pipeline requires depth 2 and balanced behavior")
    if (args.live and args.max_usd is None) or (args.max_usd is not None and
            (not math.isfinite(args.max_usd) or args.max_usd <= 0)):
        raise ValueError("live requires an explicit finite positive max-usd cap")
    if args.live and args.diagnostic_scores:
        raise ValueError("diagnostic scoring is forbidden live")


def preflight(prompts, cfg, prices, small, big, live):
    """Conservative planning envelope, separate from per-action admission.

    Includes all depth-2 branches, resamples and one scoring call per item.
    Input allowance assumes 64 UTF-8 bytes per generated token plus frozen
    retrieval and JSON overhead. This is an estimate, not an invoice guarantee.
    """
    max_tokens = max(1024, int(cfg['proxy']['max_tokens']))
    corpus = sorted((len(t.encode()) for t in SEED_CORPUS), reverse=True)
    evidence = sum(corpus[:int(cfg['retrieval']['top_k'])])
    small_calls = int(cfg['proxy']['resamples']) + 11  # prefix, score, answer/check/verify, six continuations
    big_calls = 4 if big else 0
    total = 0.0
    for prompt in prompts:
        input_tokens = len(prompt.encode()) + 64*max_tokens + evidence + 8192
        for model, calls in ((small,small_calls),(big,big_calls)):
            if not calls:
                continue
            price = prices[model]
            rates = [float(price[k]) for k in ('cost_in','cost_out')]
            if any(not math.isfinite(r) or r < 0 for r in rates):
                raise ValueError('invalid snapshotted model price')
            if live and price['live_provider'] == 'mock':
                raise ValueError('live model resolves to mock; configure a real provider first')
            total += calls*(input_tokens*rates[0]+max_tokens*rates[1])/1e6
    return {'listed_price_planning_usd': total, 'admission_estimate_usd': total if live else 0.0,
            'max_calls': len(prompts)*(small_calls+big_calls), 'n_items':len(prompts),
            'assumptions':'64 bytes/generated token; frozen corpus; all actions; one scoring call',
            'not_invoice_guarantee': True}


def write_json(path: Path, value):
    with path.open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2,allow_nan=False)


async def run_pipeline(args: PipelineOptions):
    validate_options(args)
    if not args.live and not get_settings().force_mock:
        raise ValueError('non-live pipeline requires TRIAGE_FORCE_MOCK=1 before app import')
    if args.live and get_settings().force_mock:
        raise ValueError('live pipeline cannot use forced mock')
    from scripts.collect_traces import load_items, build_prompt, make_labeler
    cfg = load_router_config()
    small = args.small or default_small_model()['id']
    big = args.big or cfg['escalation'].get('default_target')
    items = load_items(args.dataset,args.n)
    prompts = [build_prompt(args.dataset,item) for item in items]
    prices = store.model_snapshot()
    estimate = preflight(prompts,cfg,prices,small,big,args.live)
    print('Preflight estimate (before any generation): '+json.dumps(estimate),flush=True)
    if args.max_usd is not None and estimate['admission_estimate_usd'] > args.max_usd:
        raise BudgetExceeded('whole-run planning estimate exceeds supplied cap; no calls made')
    corpus_hash = hashlib.sha256(json.dumps(SEED_CORPUS,sort_keys=True).encode()).hexdigest()
    manifest = store.build_manifest(args.run_id,dataset=args.dataset,split='mixed',seeds=[args.seed],
        command='scripts/run_gate2.py '+json.dumps(vars(args),sort_keys=True),
        notes='consolidated depth2; frozen seed corpus '+corpus_hash+
              '; diagnostic_scores='+str(args.diagnostic_scores)+'; rps='+str(args.rps)+'; max_attempts=3')
    store.create_run(manifest)
    root = store.run_dir(args.run_id)
    write_json(root/'preflight.json',estimate)
    write_json(root/'items.json',{'dataset':args.dataset,'items':items,'prompts':prompts})
    write_json(root/'corpus.json',{'sha256':corpus_hash,'documents':SEED_CORPUS})
    splits = SplitManifest(args.run_id)
    budget = RunBudget(Budget(max_usd=args.max_usd),prices)
    policy = build_policy(args.policy,cfg)
    scorer = HeuristicGainPolicy(diagnostic_seed=args.seed if args.diagnostic_scores else None)
    old_store = retrieval._store
    frozen = retrieval.LocalStore()
    frozen.add(SEED_CORPUS,source='frozen:'+corpus_hash)
    retrieval._store = frozen
    stopped = None
    control = RequestControl(args.rps)
    control_token = active_control.set(control) if args.live else None
    wall_start = time.monotonic()
    try:
        with (root/'collect.log').open('x',encoding='utf-8') as log:
            log.write(json.dumps({'options':vars(args),'preflight':estimate})+'\n')
            for i,(item,prompt) in enumerate(zip(items,prompts)):
                item_id = f'{args.dataset}-{i:04d}'
                group = GroupKey(item_id,args.dataset,task_family=item.get('kind',args.dataset),
                    model_pair=f'{small}->{big}',retriever_version=corpus_hash,
                    prompt_template=args.dataset,prompt_fingerprint=prompt_fingerprint(prompt))
                split = splits.add(group)
                try:
                    traces = await collect_fanout(item_id=item_id,question=prompt,cfg=cfg,
                        small_id=small,big_id=big,run_id=args.run_id,dataset=args.dataset,
                        split=split,seed=args.seed+i,labeler=make_labeler(args.dataset,item),
                        label_source='rigor.score',served_policy=policy,scoring_policy=scorer,
                        budget=budget,depth=args.depth,prompt_features={'task_family':item.get('kind',args.dataset)})
                except BudgetExceeded as exc:
                    traces,stopped = exc.trajectories,str(exc)
                for trace in traces:
                    store.append(trace)
                log.write(json.dumps({'item_id':item_id,'trajectories':len(traces),'stop':stopped})+'\n')
                log.flush()
                print(f"Collected {i+1}/{len(items)}: {len(traces)} trajectories; "
                      f"calls={budget.state.spent_llm_calls}; usd={budget.state.spent_usd:.6f}", flush=True)
                if stopped:
                    break
    finally:
        retrieval._store = old_store
        if control_token is not None:
            active_control.reset(control_token)
        write_json(root/'request-events.json', {'rps':args.rps,'max_attempts':3,
            'wall_seconds':time.monotonic()-wall_start,'throttle_seconds':control.throttle_seconds,
            'events':control.events})
    splits.write(root)
    traces = store.read_run(args.run_id)
    validation = store.validate_run(args.run_id)
    support = action_support(traces)
    try:
        assert_estimable(traces)
        evidence_support = {'status':'SUPPORTED'}
    except SupportError as exc:
        evidence_support = {'status':'REFUSED','reason':str(exc)}
    gate2 = gate2_report(args.run_id)
    calibration = calibration_report(args.run_id,diagnostic=not args.live)
    gate2['calibration'] = calibration
    ablations = {'C1':c1_report(args.run_id,diagnostic=not args.live),
                 'C2':c2_report(args.run_id,diagnostic=not args.live),
                 'C4':c4_report(args.run_id,diagnostic=not args.live)}
    report = {'run_id':args.run_id,'analysis_grade':validation['analysis_grade'],
        'not_evidence':not validation['analysis_grade'],'validation':validation,
        'support':support,'positivity':positivity_violations(support),
        'missingness':missingness(traces),'evidence_support':evidence_support,
        'preflight':estimate,'budget':budget.state.model_dump(),'collection_stopped':stopped,
        'gate2':gate2,'calibration':calibration,'ablations':ablations,
        'gate1':'NARROW_PROVISIONAL','gate3':'UNRESOLVED','c2_real_data_status':'UNRESOLVED',
        'live_boundary':not args.live}
    write_json(root/'calibration.json',calibration)
    write_json(root/'report.json',report)
    return report
