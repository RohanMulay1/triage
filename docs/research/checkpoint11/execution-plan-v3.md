# Checkpoint 11 bounded-concurrency execution contract

Written before inspecting any quality labels from a complete live sample. The
scientific design remains fixed: 120 GSM8K items, seed 0, depth 2, balanced
behavior, and NVIDIA Lightning 30B to Ultra 550B. The sequential awake run showed
that the four approximately 60--90 second Ultra calls per item imply about twenty
hours on this workstation, recreating the operational exposure that interrupted
the first run.

The replacement overlaps eight independent items. Request admission and start
pacing remain global at 0.6 requests/second, so concurrency does not raise the
prespecified request-start rate. Every event uses a task-local item identity and
completed batches append in input order. This option refuses non-live runs and
any model with a nonzero snapshotted input or output price, because overlapping
pre-action admissions are not used for paid lanes. The $1.99 run-wide request
ledger, three-attempt retry bound, 60-second maximum retry wait, 180-second
per-attempt runtime bound, failure non-labelling, completion receipt, and Windows
host-awake guard remain unchanged.

Concurrency changes provider load and may change failure incidence. Request
failures, retry counts, and latency are therefore part of the result, and the run
does not establish behavior under a sequential production workload.

```powershell
python scripts/run_gate2.py --n 120 --dataset gsm8k --depth 2 --policy balanced --small nim-nemotron-lightning-30b --big nim-nemotron-ultra-550b --live --max-usd 1.99 --rps 0.6 --request-timeout 180 --item-concurrency 8 --run-id gate2-nvidia-full-concurrent-20260909
```
