# Portfolio rebalancing strategy

A locked-weight, monthly-rebalanced long-only allocator. **First**
strategy research family after the universe-selection branch
concluded that portfolio rebalancing was the only candidate that fit
the existing engine without leverage, shorts, paid data, or broker
integration.

## What the strategy does

Holds a fixed weight vector across BTC, ETH, and a cash bucket and
rebalances monthly back to those weights. Locked configuration
(in `src/strategies/portfolio_rebalancing_allocator.py`):

* BTC/USDT: **0.60**
* ETH/USDT: **0.30**
* Cash bucket: **0.10** (returned as unallocated weight; the
  backtester treats it as cash)
* Rebalance frequency: **monthly**
* Long-only, no leverage, no shorting, lookahead-free

Missing-asset handling: if BTC or ETH is missing at the rebalance
bar, the absent leg's risk weight redistributes onto the surviving
risk leg; the cash bucket is preserved. If both are missing, the
strategy returns 100 % cash.

## What it does NOT do

* Does not optimise weights against historical performance.
* Does not fit a model.
* Does not use any signal (price, funding, basis, regime, or
  otherwise).
* Does not place orders. Does not connect to brokers. Does not
  enable paper trading or live trading. Source-level tests enforce.
* Does not use lookahead — fixed weights are by definition
  lookahead-free, and a unit test asserts that truncating future
  history does not change past target_weights output.

## Why it is NOT judged by raw return vs BTC

Eight prior long-only allocator branches on this project failed
under a "beat BTC outright" scorecard. The honest finding was that
in a 4-year BTC-favourable sample, defensive allocators give up
too much upside to clear that bar.

This branch's locked PASS criteria are **risk-adjusted instead**:

| Gate | Threshold |
| --- | --- |
| Sharpe within 0.10 of BTC b&h | `|strat_sharpe − btc_sharpe| ≤ 0.10` |
| Max drawdown ≥ 15 pp tighter than BTC | `strat_dd − btc_dd ≥ 15.0` |
| Beats placebo MEDIAN return | strategy return > median of 20 random fixed-weight placebos |
| Beats placebo MEDIAN drawdown | strategy DD better than median of 20 random fixed-weight placebos |
| ≥ 24 rebalances | total rebalance count across walk-forward windows |

PASS requires **all five** gates. FAIL otherwise. INCONCLUSIVE only
if data is insufficient (< 5 OOS windows or required CSVs missing).

## How to run

```bash
# Single-window backtest vs benchmarks:
python main.py portfolio_rebalancing_backtest

# 14-window walk-forward:
python main.py portfolio_rebalancing_walk_forward

# 20-seed random fixed-weight placebo:
python main.py portfolio_rebalancing_placebo

# Build the scorecard from saved CSVs:
python main.py portfolio_rebalancing_scorecard

# End-to-end pipeline:
python main.py research_all_portfolio_rebalancing
```

Generated CSVs live in `results/` and are gitignored:

```
results/portfolio_rebalancing_comparison.csv
results/portfolio_rebalancing_walk_forward.csv
results/portfolio_rebalancing_placebo.csv
results/portfolio_rebalancing_scorecard.csv
```

## What the PASS criteria mean

* **Sharpe within 0.10**: comparable risk-adjusted return — neither
  meaningfully better nor worse than holding BTC straight.
* **Drawdown ≥ 15 pp tighter**: the diversification benefit of
  adding 10 % cash and rebalancing must produce a *materially*
  smaller drawdown than BTC. 5 pp tighter does not count.
* **Beats placebo on both axes**: a fixed-weight strategy must do
  *better* than a randomly drawn fixed weight allocation, in both
  return and drawdown. Otherwise the locked weight choice has no
  evidence of being better than picking blindly.
* **≥ 24 rebalances**: enough rebalance events to make the result
  representative of the rebalancing mechanic, not a single lucky /
  unlucky moment.

## Why execution remains locked

Even on a PASS verdict — which this branch did NOT achieve — paper
trading and live trading remain blocked at two independent layers:

1. The strategy registry entry carries `paper_trading_allowed =
   False` and `live_trading_allowed = False`. Test
   `test_no_strategy_is_paper_or_live_allowed` enforces.
2. The safety lock has no path that releases without an explicit
   code change AND an independent review step (see
   `docs/unlock_procedure.md`).

A PASS verdict would be the first signal that a strategy is worth
discussing further — not the trigger to start trading.

## The verdict on this branch

**FAIL.** Sharpe gate cleared (0.50 vs 0.52), placebo drawdown
cleared, rebalance count cleared. But max drawdown was only +10.78
pp tighter than BTC (the gate requires ≥ 15 pp), and the strategy
lost the placebo return contest (45th percentile — random
allocations that happened to under-weight ETH outperformed the
fixed 0.60/0.30 mix). See
[`reports/portfolio_rebalancing_report.md`](../reports/portfolio_rebalancing_report.md)
for the full verdict, walk-forward table, and exact next step.

## Safety rules

* No broker SDK imports anywhere in the strategy or research
  modules.
* No API key reads.
* No order placement strings.
* No paper / live trading enablement.
* No optimiser, no model fitting, no historical-performance-driven
  weights.
* Generated CSVs gitignored at `results/*.csv`.
* Source-level unit tests enforce every invariant.

The safety lock continues to be locked.
