"""Collect action-level research traces from the heuristic router.

Checkpoint 1 of RESEARCH_STRATEGY.md. Defaults to the offline mock provider, so
the default invocation spends nothing:

    python scripts/collect_traces.py --n 20 --dataset traps --split pilot
    python scripts/collect_traces.py --n 50 --dataset gsm8k --split train

Live providers require --live explicitly. That gate is deliberate: the strategy
says the schemas and their tests must pass before any paid trace collection, and
a run that quietly spent money would also be a run whose manifest claims
force_mock=False without anyone having decided that.

Writes <trace_dir>/<run_id>/{manifest.json,trajectories.jsonl} and prints the
action-support report.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="number of items")
    ap.add_argument("--dataset", default="traps", choices=["traps", "gsm8k", "mmlu"])
    ap.add_argument("--split", default="pilot", choices=["train", "calib", "test", "pilot"])
    ap.add_argument("--small", default=None, help="small model id (auto-pick if omitted)")
    ap.add_argument("--big", default=None, help="escalation target model id")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--live", action="store_true",
                    help="allow live providers and real spend (default: mock only)")
    return ap.parse_args()


ARGS = _parse_args()
# Must be set before importing app.config, whose Settings are lru_cached.
if not ARGS.live:
    os.environ["TRIAGE_FORCE_MOCK"] = "1"

from app.benchmark import rigor  # noqa: E402
from app.config import load_router_config  # noqa: E402
from app.router import memory  # noqa: E402
from app.router.router import route_and_answer  # noqa: E402
from app.schemas import ChatRequest  # noqa: E402
from app.telemetry import db as telemetry_db  # noqa: E402
from app.trace import store  # noqa: E402
from app.trace.adapter import TraceRecorder  # noqa: E402
from app.trace.support import action_support  # noqa: E402


def load_items(dataset: str, n: int) -> list[dict]:
    if dataset == "traps":
        return rigor.load_traps(n)
    if dataset == "gsm8k":
        return rigor.load_gsm8k(n)
    return rigor.load_mmlu(n)


async def collect(args: argparse.Namespace) -> str:
    random.seed(args.seed)
    telemetry_db.init_db()
    memory.init_db()

    cfg = load_router_config()
    run_id = args.run_id or store.new_run_id(f"{args.dataset}-{args.split}")
    # create_run refuses a run id that already holds traces, so a re-run can
    # never silently append a second collection behind one manifest.
    store.create_run(store.build_manifest(
        run_id, dataset=args.dataset, split=args.split, seeds=[args.seed],
        command=" ".join(sys.argv), notes=args.notes,
    ))

    items = load_items(args.dataset, args.n)
    trajectories = []
    for i, item in enumerate(items):
        item_id = f"{args.dataset}-{i:04d}"
        recorder = TraceRecorder(run_id, item_id, cfg,
                                 dataset=args.dataset, split=args.split, seed=args.seed)
        req = ChatRequest(message=rigor._prompt(item), model=args.small,
                          escalate_to=args.big, max_signals=True)
        resp, telem = await route_and_answer(req, recorder=recorder)
        telemetry_db.log_request(telem)

        label = rigor.score(item, resp.answer, resp.route.abstained)
        traj = recorder.finish(
            resp, label=float(label),
            label_source=f"rigor.score:{item.get('kind', args.dataset)}",
        )
        store.append(traj)
        trajectories.append(traj)
        print(f"  [{i + 1}/{len(items)}] {item_id} tier={resp.route.tier} "
              f"escalated={resp.route.escalated} label={label} "
              f"actions={len(traj.steps)}", flush=True)

    return run_id


def main() -> None:
    args = ARGS
    if args.live:
        print("[collect_traces] LIVE MODE: this run may incur real provider charges.")
    else:
        print("[collect_traces] mock provider forced: this run spends nothing.")

    run_id = asyncio.run(collect(args))
    trajectories = store.read_run(run_id)

    validation = store.validate_run(run_id)
    support = action_support(trajectories)
    print(f"\nrun_id: {run_id}")
    print(f"written to: {store.run_dir(run_id)}")
    print("\n--- validation ---")
    print(json.dumps(validation, indent=2))
    print("\n--- action support ---")
    for key, s in support["per_action"].items():
        print(f"  {key:48s} taken={s['n_taken']:4d} feasible={s['n_feasible']:4d} "
              f"coverage={s['coverage']}")
    print(f"\n  unsupported actions : {support['unsupported_actions']}")
    print(f"  off-policy ready    : {support['off_policy_ready']}")
    print(f"  analysis grade      : {validation['analysis_grade']}")
    if not support["off_policy_ready"]:
        print("\n  Feasible actions were never chosen, so this run carries no evidence\n"
              "  about what they would have done. Off-policy value estimation over it\n"
              "  would be extrapolation. Randomized branch collection is required.")


if __name__ == "__main__":
    main()
