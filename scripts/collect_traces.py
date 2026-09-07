"""Collect action-level research traces.

Two modes:

  --mode served   run the legacy heuristic router and record what it did. This
                  reproduces production routing exactly and is what Checkpoint 1
                  measured; it produces no counterfactual support.
  --mode fanout   execute a shared ANSWER prefix, then every available action
                  once from that same state, each carried to a terminal and
                  labelled. This is what the Signal Gate needs: Δ(a) versus STOP
                  is measured rather than imputed.

Defaults to the offline mock provider, so the default invocation spends nothing:

    python scripts/collect_traces.py --n 20 --dataset traps --mode fanout
    python scripts/collect_traces.py --n 50 --dataset gsm8k --mode fanout --policy response_risk

A live run requires BOTH `--live` and an explicit finite positive `--max-usd`.
The caller owns one run-wide accumulator. Admission estimates include every
call's completion cap; actual reported costs are settled after execution.
Live served mode is refused because the legacy router has no budget control.

Writes <trace_dir>/<run_id>/{manifest.json,trajectories.jsonl,splits.json}.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="number of items")
    ap.add_argument("--dataset", default="traps", choices=["mixed", "traps", "gsm8k", "mmlu"])
    ap.add_argument("--mode", default="fanout", choices=["served", "fanout"])
    ap.add_argument("--depth", type=int, choices=[1, 2], default=1,
                    help="fan-out decision depth; depth 2 multiplies calls within the same run cap")
    ap.add_argument("--policy", default="response_risk",
                    help="behaviour policy that designates the served branch in fanout mode")
    ap.add_argument("--split", default=None,
                    help="force every item into this split; default assigns by group hash")
    ap.add_argument("--split-salt", default="", help="salt for split assignment")
    ap.add_argument("--small", default=None, help="small model id (auto-pick if omitted)")
    ap.add_argument("--big", default=None, help="escalation target model id")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--live", action="store_true",
                    help="allow live providers and real spend (requires --max-usd)")
    ap.add_argument("--max-usd", type=float, default=None,
                    help="finite positive run-wide admission cap in listed-price dollars")
    args = ap.parse_args()
    if (args.live and args.max_usd is None) or (args.max_usd is not None and (
            not math.isfinite(args.max_usd) or args.max_usd <= 0)):
        ap.error("--live requires an explicit positive --max-usd cap that is finite")
    if args.n <= 0:
        ap.error("--n must be a strictly positive integer")
    if args.live and args.mode == "served":
        ap.error("--mode served --live refused: served mode has no cost control")
    if args.depth != 1 and args.mode == "served":
        ap.error("--depth 2 requires --mode fanout")
    return args


ARGS = _parse_args()
# Must be set before importing app.config, whose Settings are lru_cached.
if not ARGS.live:
    os.environ["TRIAGE_FORCE_MOCK"] = "1"

from app.benchmark import rigor  # noqa: E402
from app.config import default_small_model, load_router_config  # noqa: E402
from app.router import memory  # noqa: E402
from app.router.router import route_and_answer  # noqa: E402
from app.schemas import ChatRequest  # noqa: E402
from app.telemetry import db as telemetry_db  # noqa: E402
from app.trace import store  # noqa: E402
from app.trace.adapter import TraceRecorder  # noqa: E402
from app.trace.branching import BudgetExceeded, RunBudget, collect_fanout  # noqa: E402
from app.trace.contract import Budget  # noqa: E402
from app.trace.policies import build_policy  # noqa: E402
from app.trace.splits import GroupKey, SplitManifest, prompt_fingerprint  # noqa: E402
from app.trace.support import (  # noqa: E402
    SupportError, action_support, marginal_value_table, missingness,
)


#: A deliberately heterogeneous set for fixtures and smoke runs. One item per
#: shape the action space can take, so a fixture built from it exercises TOOL
#: available and unavailable, retrieval-relevant and retrieval-irrelevant
#: questions, and an unanswerable trap where abstaining is the correct terminal.
_MIXED: list[dict] = [
    {"q": "what is 17 * 23 + 5?", "gold": "396", "kind": "arith"},
    {"q": "what is the square root of 144?", "gold": "12", "kind": "arith"},
    {"q": "What is the capital of France?", "gold": "Paris", "kind": "factual"},
    {"q": "Who wrote Hamlet?", "gold": "Shakespeare", "kind": "factual"},
    {"q": "At what temperature does water boil at sea level?", "gold": "100",
     "kind": "factual"},
    {"q": "What are next week's winning lottery numbers?", "gold": None, "kind": "trap"},
    {"q": "What did I eat for breakfast this morning?", "gold": None, "kind": "trap"},
    {"q": "What is the exact current population of the planet Mars right now?",
     "gold": None, "kind": "trap"},
]


def load_items(dataset: str, n: int) -> list[dict]:
    if dataset == "mixed":
        return (_MIXED * ((n // len(_MIXED)) + 1))[:n]
    if dataset == "traps":
        return rigor.load_traps(n)
    if dataset == "gsm8k":
        return rigor.load_gsm8k(n)
    return rigor.load_mmlu(n)


def _mixed_prompt(item: dict) -> str:
    """Ask the arithmetic items plainly.

    Appending "Give the final numeric answer." (as `rigor._prompt` does) leaves
    `calculator.solve` unable to parse the request, which silently drops TOOL
    from the feasible set for exactly the items it exists to serve.
    """
    return item["q"]


def _mixed_score(item: dict, answer: str, abstained: bool) -> int:
    """Scorer for the mixed fixture set.

    `rigor.score` matches numerically for everything that is not MMLU or a trap,
    so it would mark every factual item wrong for having no digits in it. These
    items are heterogeneous by design, so they need a scorer that is too.
    """
    if item["kind"] == "trap":
        return int(abstained)
    if abstained:
        return 0
    if item["kind"] == "factual":
        return int(str(item["gold"]).lower() in (answer or "").lower())
    return rigor.score(item, answer, abstained)


def build_prompt(dataset: str, item: dict) -> str:
    return _mixed_prompt(item) if dataset == "mixed" else rigor._prompt(item)


def score_item(dataset: str, item: dict, answer: str, abstained: bool) -> int:
    if dataset == "mixed":
        return _mixed_score(item, answer, abstained)
    return rigor.score(item, answer, abstained)


def make_labeler(dataset: str, item: dict):
    """Terminal correctness for this item."""
    def label(answer: str, abstained: bool):
        return float(score_item(dataset, item, answer, abstained))
    return label


async def collect(args: argparse.Namespace) -> str:
    if args.live and args.mode == "served":
        raise ValueError("served mode has no cost control; live collection refused")
    random.seed(args.seed)
    telemetry_db.init_db()
    memory.init_db()

    cfg = load_router_config()
    small_id = args.small or default_small_model()["id"]
    big_id = args.big or (cfg.get("escalation") or {}).get("default_target")
    run_id = args.run_id or store.new_run_id(f"{args.dataset}-{args.mode}")

    manifest = store.build_manifest(
        run_id, dataset=args.dataset, split=args.split or "mixed", seeds=[args.seed],
        command=" ".join(sys.argv),
        notes=(args.notes or "") + f" | mode={args.mode} policy={args.policy} "
                                   f"small={small_id} big={big_id} depth={args.depth} "
                                   "prompt_features=cold_memory",
    )
    store.create_run(manifest)

    splits = SplitManifest(run_id=run_id, salt=args.split_salt)
    items = load_items(args.dataset, args.n)
    policy = build_policy(args.policy, cfg)
    budget = RunBudget(Budget(max_usd=args.max_usd), manifest.model_snapshot)
    n_written = 0

    for i, item in enumerate(items):
        item_id = f"{args.dataset}-{i:04d}"
        prompt = build_prompt(args.dataset, item)
        key = GroupKey(
            item_id=item_id, dataset=args.dataset,
            task_family=item.get("kind", args.dataset),
            model_pair=f"{small_id}->{big_id}",
            retriever_version=str(cfg["retrieval"]["top_k"]),
            prompt_template=args.dataset,
            prompt_fingerprint=prompt_fingerprint(prompt),
        )
        split = args.split or splits.add(key)
        if args.split:
            splits.add(key)

        try:
            if args.mode == "fanout":
                trajectories = await collect_fanout(
                    item_id=item_id, question=prompt, cfg=cfg, small_id=small_id,
                    big_id=big_id, run_id=run_id, dataset=args.dataset, split=split,
                    seed=args.seed, labeler=make_labeler(args.dataset, item),
                    label_source=f"rigor.score:{item.get('kind', args.dataset)}",
                    served_policy=policy, budget=budget,
                    depth=args.depth,
                    prompt_features={"task_family": item.get("kind", args.dataset)},
                )
            else:
                recorder = TraceRecorder(run_id, item_id, cfg, dataset=args.dataset,
                                         split=split, seed=args.seed)
                req = ChatRequest(message=prompt, model=args.small,
                                  escalate_to=args.big, max_signals=True)
                resp, telem = await route_and_answer(req, recorder=recorder)
                telemetry_db.log_request(telem)
                trajectories = [recorder.finish(
                    resp, label=float(score_item(args.dataset, item, resp.answer,
                                                 resp.route.abstained)),
                    label_source=f"rigor.score:{item.get('kind', args.dataset)}")]
        except BudgetExceeded as e:
            for trajectory in e.trajectories:
                store.append(trajectory)
            print(f"\n[collect_traces] STOPPED at item {i}: {e}")
            print("[collect_traces] traces written so far are intact and readable.")
            break

        for t in trajectories:
            store.append(t)
        n_written += len(trajectories)
        print(f"  [{i + 1}/{len(items)}] {item_id} split={split} "
              f"branches={len(trajectories)} written={n_written}", flush=True)

    splits.write(store.run_dir(run_id))
    return run_id


def main() -> None:
    args = ARGS
    if args.live:
        print(f"[collect_traces] LIVE MODE: real provider charges, capped at ${args.max_usd}")
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
    print("\n--- missingness ---")
    print(json.dumps(missingness(trajectories), indent=2))
    print("\n--- action support ---")
    for key, s in support["per_action"].items():
        print(f"  {key:46s} taken={s['n_taken']:4d} observed={s['n_observed']:4d} "
              f"feasible={s['n_feasible']:4d} coverage={s['coverage']}")
    print(f"\n  unsupported actions : {support['unsupported_actions']}")
    print(f"  off-policy ready    : {support['off_policy_ready']}")
    print(f"  analysis grade      : {validation['analysis_grade']}")

    try:
        mv = marginal_value_table(trajectories)
    except SupportError as exc:
        mv = {"per_action": {}}
        print(f"Marginal-value estimate REFUSED: {exc}")
    if mv["per_action"]:
        print(f"\n--- marginal value vs STOP "
              f"(items with baseline: {mv['items_with_baseline']}) ---")
        for key, s in mv["per_action"].items():
            print(f"  {key:46s} n={s['n']:4d} mean delta={s['mean_delta']:+.4f} "
                  f"95% CI [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] "
                  f"(+{s['n_positive']}/-{s['n_negative']}/={s['n_zero']})")
    print(f"\nnext: python -m app.trace.support --run {run_id}")


if __name__ == "__main__":
    main()
