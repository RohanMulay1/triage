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
        planned=sum((100*prices[m]['cost_in']+8*prices[m]['cost_out'])/1e6
                    for m in ('llama-3.1-8b','llama-3.3-70b'))
        if planned>args.max_usd: raise ValueError('estimate exceeds cap')
        for model in ('llama-3.1-8b','llama-3.3-70b'):
            adapter,name,label=get_registry().resolve(model)
            if label=='mock': raise ValueError('model resolves to mock')
            result=await adapter.generate(name,[{'role':'user','content':'Reply with OK.'}],
                                          0.0,8,False)
            print(json.dumps({'model':model,'provider':label,'provider_model':name,
                'error':result.error,'tokens_in':result.tokens_in,'tokens_out':result.tokens_out,
                'text':result.text,'latency_ms':result.latency_ms}),flush=True)
            await asyncio.sleep(1/.6)
    asyncio.run(probe())


if __name__=='__main__': main()
