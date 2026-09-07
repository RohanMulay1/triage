"""Minimal capped provider diagnostic; prints no credentials."""
import argparse
import asyncio
import json
import math
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--max-usd',type=float,required=True)
    parser.add_argument('--small', default='llama-3.1-8b')
    parser.add_argument('--big', default='llama-3.3-70b')
    parser.add_argument('--output', type=Path)
    args=parser.parse_args()
    if not args.live or not math.isfinite(args.max_usd) or args.max_usd<=0:
        parser.error('explicit --live and finite positive --max-usd required')
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
    from app.config import get_settings
    from app.providers.registry import get_registry
    from app.trace.store import model_snapshot
    if get_settings().force_mock:
        raise ValueError('forced mock is enabled')
    async def probe():
        prices=model_snapshot()
        models=(args.small,args.big)
        planned=sum((100*prices[m]['cost_in']+128*prices[m]['cost_out'])/1e6
                    for m in models)
        if planned>args.max_usd: raise ValueError('estimate exceeds cap')
        if args.output and args.output.exists():
            raise FileExistsError(args.output)
        records=[]
        for model in models:
            adapter,name,label=get_registry().resolve(model)
            if label=='mock': raise ValueError('model resolves to mock')
            result=await adapter.generate(name,[{'role':'user','content':'Reply with OK.'}],
                                          0.0,128,True)
            record={'model':model,'provider':label,'provider_model':name,
                'error':result.error,'tokens_in':result.tokens_in,'tokens_out':result.tokens_out,
                'text':result.text,'latency_ms':result.latency_ms,'raw':result.raw,
                'model_snapshot':prices[model]}
            records.append(record)
            print(json.dumps({k:v for k,v in record.items() if k!='raw'}),flush=True)
            await asyncio.sleep(1/.6)
        if args.output:
            with args.output.open('x',encoding='utf-8') as handle:
                json.dump({'planning_usd':planned,'max_usd':args.max_usd,'responses':records},handle,indent=2)
        if any(r['error'] or not r['text'] or r['tokens_out']<=0 for r in records):
            raise SystemExit('lane probe failed: no valid completion from every model')
    asyncio.run(probe())


if __name__=='__main__': main()
