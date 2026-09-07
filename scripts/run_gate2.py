"""Collect, calibrate and evaluate all research gates under one run budget."""
import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--n',type=int,default=8)
    parser.add_argument('--dataset',choices=['mixed','traps','gsm8k','mmlu'],default='mixed')
    parser.add_argument('--depth',type=int,choices=[2],default=2)
    parser.add_argument('--policy',choices=['balanced'],default='balanced')
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--max-usd',type=float)
    parser.add_argument('--diagnostic-scores',action='store_true')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--small')
    parser.add_argument('--big')
    parser.add_argument('--rps',type=float,default=.6)
    args = parser.parse_args()
    if not math.isfinite(args.rps) or args.rps <= 0:
        parser.error("--rps must be finite and strictly positive")
    if args.n <= 0:
        parser.error('n must be strictly positive')
    if (args.live and args.max_usd is None) or (args.max_usd is not None and
            (not math.isfinite(args.max_usd) or args.max_usd <= 0)):
        parser.error('--live requires an explicit finite positive --max-usd')
    if args.live and args.diagnostic_scores:
        parser.error('diagnostic scores are forbidden live')
    if not args.live:
        os.environ['TRIAGE_FORCE_MOCK']='1'
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
    from app.trace.pipeline import PipelineOptions, run_pipeline
    result = asyncio.run(run_pipeline(PipelineOptions(**vars(args))))
    print(json.dumps({'run_id':result['run_id'],'gate2':result['gate2']['verdict'],
                      'analysis_grade':result['analysis_grade'],'report':'report.json'}))


if __name__ == '__main__':
    main()
