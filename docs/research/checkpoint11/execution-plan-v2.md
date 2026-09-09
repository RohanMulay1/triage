# Checkpoint 11 replacement execution contract

Written after `gate2-nvidia-full-20260909` was interrupted and before inspecting
any quality labels from a complete live sample. The scientific design remains the
frozen Checkpoint 11 contract: 120 GSM8K items, seed 0, depth 2, balanced behavior,
and the explicitly identified NVIDIA Lightning 30B to Ultra 550B pair.

The replacement run changes runtime safeguards only. Windows host-awake state is
held during collection. Each provider attempt has a 180-second outer wall-runtime
bound in addition to the HTTP timeout. Empty-message network exceptions are
explicit failures and network failures receive the same bounded three-attempt
retry treatment as 502/503/504. Unknown usage retains its admission reservation.
The pipeline writes an immutable collection plan and writes a completion receipt
only after every planned item finishes. Missing or inconsistent completion makes
`analysis_grade` false and all evidence reports refuse.

The interrupted run remains immutable. It contains 11 complete items, 286
trajectories, 223 request events over 12 attempted items, and no consolidated
scientific report. One request recorded 32,047,672 ms while the Windows host was
asleep. Its empty exception string exposed the false-success defect repaired for
the replacement. No label or partial-sample statistic was inspected to select
this runtime change.

The approved program cap remains $2.00. Previously recorded Groq listed-price
usage is $0.0026961. Both NVIDIA rows remain snapshotted at zero listed price;
that is not an invoice guarantee. The replacement command retains a $1.99
run-wide cap and uses a fresh run id.

```powershell
python scripts/run_gate2.py --n 120 --dataset gsm8k --depth 2 --policy balanced --small nim-nemotron-lightning-30b --big nim-nemotron-ultra-550b --live --max-usd 1.99 --rps 0.6 --request-timeout 180 --run-id gate2-nvidia-full-awake-20260909
```
