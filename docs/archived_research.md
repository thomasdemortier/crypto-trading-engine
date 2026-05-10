# Archived research

Brief, hand-maintained log of research branches that were finalised
with a non-PASS verdict. Each entry points at the branch and the
commit / tag that pin the verdict — none of these branches are
merged into `main` and none should be.

For the canonical machine-readable list see
`src/research_dashboard.ARCHIVED_BRANCHES`. The Streamlit Research
Dashboard reads from there and renders an Archived timeline table
with `merge_allowed=False` for every row.

## Why archived branches stay archived

* **Decision evidence.** Future maintainers (and future-you) need to
  see *why* a strategy class was tried and rejected, otherwise the
  same idea cycles back. Reports under `reports/*.md` on each
  branch are the evidence.
* **Reproducibility.** The CSV outputs (re-run from the branch)
  regenerate the verdict; nothing about the conclusion is hidden.
* **Discipline.** Merging a failed strategy into `main` would
  contaminate the production state. Tagging the failed commit lets
  us address it without copying its code into the trunk.

## Recent archive events

### 2026-05-10 — Portfolio rebalancing strategy v1 — FAIL

| | |
| --- | --- |
| Branch | `research/portfolio-rebalancing-strategy-v1` |
| Tag | `fail-portfolio-rebalancing-strategy-v1` |
| Commit | `19b1268` |
| Type | strategy |
| Verdict | **FAIL** |
| Gates passed | 3 of 5 |
| `merge_allowed` | False |
| `paper_trading_allowed` | False |
| `live_trading_allowed` | False |

Locked configuration tested: BTC/USDT 0.60 + ETH/USDT 0.30 + cash 0.10,
monthly rebalance, long-only, no leverage.

The strategy passed the Sharpe-within-0.10 gate (gap 0.023), beat the
placebo median drawdown by ~1 pp, and clocked 56 rebalances over 14
walk-forward windows. It **failed** the ≥ 15 pp drawdown improvement
gate (delivered only +10.78 pp vs BTC) and **failed** the placebo
median return gate (45th percentile — random allocations that
under-weighted ETH outperformed the locked 0.60/0.30 mix because ETH
lost 32.5 % over the 4-year window).

The locked PASS criteria stand. Parameters were not retuned. The
branch is archived; the strategy code, generated CSVs, and report
remain only on the archived branch and will not be merged into `main`
as a live strategy. Inline report viewing requires
`git checkout research/portfolio-rebalancing-strategy-v1`.

## How to read this list

Run:

```bash
python -c 'from src import research_dashboard as rd; \
print(rd.archived_timeline_dataframe().to_string(index=False))'
```

Or open the Streamlit dashboard — Research Dashboard → Archived
timeline tab.

## What NOT to do

* Do not merge any archived research branch into `main`. Every entry
  in `ARCHIVED_BRANCHES` carries `merge_allowed=False`; a unit test
  enforces it.
* Do not copy the strategy implementation files into `main`.
* Do not copy generated CSVs (`results/*.csv`) — they are gitignored
  and reproducible from the archived branch.
* Do not interpret a FAIL verdict as a recommendation to retune the
  thresholds. Locked criteria are locked.
