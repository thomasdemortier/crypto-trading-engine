# FX EOD Trend strategy — v1

The `research/fx-eod-trend-strategy-v1` branch tests a single, locked,
end-of-day trend rule on EUR/USD using the validated v1 FX dataset.

> Research only. No execution. No broker. No API keys. No paper
> trading. No live trading.

## What this strategy does

A vectorised, no-lookahead long-or-cash trend rule:

```text
asset            = EUR/USD
source filter    = ecb_sdmx (ECB EUR-quoted reference rate, daily)
timeframe        = 1d
lookback_days    = 200
position[t+1]    = 1 if close[t] > SMA200(close)[t] else 0
strategy_return  = position * raw_return  (one-day lagged signal)
initial_cash     = 1.0
```

The position is binary `{0, 1}`. There is no leverage, no shorting,
no fractional sizing, no carry, no spread, no slippage, and no fees.
Every input row must come from `data/fx/fx_daily_v1.parquet` and
have `data_quality_status == "ok"`. The strategy module never opens
a network socket; the orchestrator never imports a broker SDK.

## What this strategy does NOT do

- It does **not** trade. There is no broker integration, no API
  keys, no order placement, no execution module, and no paper-
  trading switch on this branch.
- It does **not** assume realistic trading costs. ECB EUR/USD is a
  daily reference fix, not a tradable broker quote, so the
  "strategy returns" in this study are pre-spread, pre-slippage,
  pre-fees, and pre-carry. They are a research signal, not a P&L
  forecast.
- It does **not** sweep parameters. The 200-day SMA is the only
  lookback tested. There is no optimiser, no grid search, and no
  parameter chosen after seeing results.
- It does **not** rebalance intraday or take partial positions.
  Signals are evaluated at the daily close and applied as a binary
  long-or-cash position to the next day's return.

## Why ECB reference data is not executable broker data

The ECB EUR/USD time series is a daily *reference fix*: a single
quote published each TARGET business day, computed from a snapshot
of contributing-bank quotes around 14:15 CET. It has no spread, no
intraday tick history, no order-book depth, and no associated
volume. Real execution at a retail or institutional broker:

- carries a bid/ask spread (≈ 0.5-1 bp interbank; several bp
  retail);
- experiences slippage on size, order type, and market state;
- pays or earns the EUR-USD overnight rate differential (carry),
  which has been *negative* for most of 1999-2024 and is omitted
  here;
- may be subject to weekend / holiday gaps that the reference rate
  smooths over.

A research result on ECB reference rates can rule a hypothesis
*out* — if a strategy cannot beat buy-and-hold even with zero
costs, costed execution is unlikely to save it. It cannot rule a
hypothesis *in*. A PASS verdict on this branch is necessary, not
sufficient, for any future tradability claim.

## How to run

```bash
# Prerequisites (must succeed before research is allowed):
python main.py build_fx_dataset           # produces data/fx/fx_daily_v1.parquet
python main.py check_fx_data_quality      # verdict must be PASS or WARNING

# Individual stages:
python main.py fx_eod_trend_backtest      # full-window backtest
python main.py fx_eod_trend_walk_forward  # 5-window OOS stability
python main.py fx_eod_trend_placebo       # 20-seed matched-exposure placebo
python main.py fx_eod_trend_scorecard     # combined verdict

# End-to-end:
python main.py research_all_fx_eod_trend
```

Each command exits **0** on PASS / no-error, **1** on FAIL, and
**2** on INCONCLUSIVE (dataset missing or quality verdict
`FAIL`/`INCONCLUSIVE`).

## PASS criteria (locked, never tuned by results)

A scorecard verdict of `PASS` requires *all nine* checks to hold:

1. `pass_positive_return` — total_return > 0.
2. `pass_sharpe_beats_benchmark` — annualised Sharpe > EUR/USD
   buy-and-hold Sharpe.
3. `pass_drawdown_tighter` — strategy max-drawdown is at least 5
   percentage points tighter than buy-and-hold.
4. `pass_beats_placebo_return` — strategy total_return strictly
   greater than the placebo median.
5. `pass_beats_placebo_drawdown` — strategy max_drawdown is
   strictly tighter than the placebo median (less negative).
6. `pass_min_trade_count` — at least 20 signal flips over the
   full window.
7. `no_leverage` — position ≤ 1 everywhere.
8. `no_shorts` — position ≥ 0 everywhere.
9. `no_lookahead` — every SMA value at index `t` uses only data
   with index ≤ `t` (asserted on every run by
   `fx_eod_trend.assert_no_lookahead`).

If any one check fails, the verdict is `FAIL`. If the dataset is
missing or the data-quality verdict is `FAIL` / `INCONCLUSIVE`, the
verdict is `INCONCLUSIVE`.

## Why execution remains locked

`config.LIVE_TRADING_ENABLED` is hard-coded `False` in v1 and there
is no execution module. The safety lock therefore reports
`safety_lock_status: locked` with `execution_allowed=False`,
`paper_trading_allowed=False`, and `kraken_connection_allowed=False`
on every status read. The strategy registry's `fx_eod_trend` entry
is wired through the lock — even on a `PASS` verdict, both
`paper_trading_allowed` and `live_trading_allowed` resolve to
`False`. A unit test (`test_no_strategy_in_registry_is_paper_or_
live_allowed`) breaks the build if anyone flips either flag.

To lift the lock, a future branch must:

1. add a separate, audited execution module (carry, spreads,
   slippage, fees, order-routing safety);
2. re-run the full FX EOD trend battery with realistic costs and
   confirm a verdict survives;
3. obtain independent review of the resulting scorecard;
4. follow the documented procedure in `docs/unlock_procedure.md`.

None of those steps are part of this branch. This branch ends at
the scorecard.

## Files

| Path                                            | Purpose                                                |
|:------------------------------------------------|:-------------------------------------------------------|
| `src/strategies/fx_eod_trend.py`                | Strategy class + frozen config + pure metric helpers   |
| `src/fx_eod_trend_research.py`                  | Quality guard + backtest / walk-forward / placebo / scorecard orchestrator |
| `tests/test_fx_eod_trend_strategy.py`           | Strategy + invariant tests                             |
| `tests/test_fx_eod_trend_research.py`           | Orchestrator + scorecard logic + safety invariant tests |
| `reports/fx_eod_trend_report.md`                | Honest research report and verdict                     |
| `results/fx_eod_trend_backtest.csv`             | Full-window backtest output (gitignored)               |
| `results/fx_eod_trend_walk_forward.csv`         | Walk-forward output (gitignored)                       |
| `results/fx_eod_trend_placebo.csv`              | 20-seed matched-exposure placebo (gitignored)          |
| `results/fx_eod_trend_scorecard.csv`            | Final scorecard / verdict (gitignored)                 |
