# FX end-of-day trend — research report v1

Branch: `research/fx-eod-trend-strategy-v1`
Base tag: `v0.8-fx-data-quality-locked`
Strategy module: `src/strategies/fx_eod_trend.py`
Research module: `src/fx_eod_trend_research.py`

> Research only. No execution. No broker. No API keys. No paper
> trading. No live trading. Even if the verdict is PASS, paper
> trading remains blocked at the safety lock until separate approval
> and independent review.

## Objective

Test whether a single, locked, end-of-day trend rule on EUR/USD
materially improves on simple buy-and-hold of the same series, on
the validated v1 FX dataset. The rule was fixed before any results
were seen; there is no parameter sweep, no optimiser, no
multi-asset extension, no carry, no spread / slippage / fee model,
no leverage, and no shorting.

## Why FX EOD trend is being tested

The audit at `v0.6-fx-crypto-source-audit-locked` proved that
multi-decade FX reference-rate data is reachable without API keys.
The dataset at `v0.7-fx-research-dataset-locked` packaged it. The
quality checks at `v0.8-fx-data-quality-locked` validated it
(verdict WARNING, acceptable for research). EOD trend is the
simplest, oldest FX hypothesis worth testing first — if it cannot
clear a fair, locked PASS bar, no more elaborate FX strategy
deserves the bench time.

## Dataset used

`data/fx/fx_daily_v1.parquet` — 63,600 rows, 8 assets, 1968-04-01
→ 2026-05-08. The strategy reads only EUR/USD rows where
`source == "ecb_sdmx"` and `data_quality_status == "ok"`.

## Data quality verdict (guard)

`WARNING` — acceptable for research. The orchestrator's quality
guard (`fx_eod_trend_research.load_and_guard`) refuses to run on
`FAIL` or `INCONCLUSIVE`. The 18 historical extreme-return events
in the dataset (SNB cap removal, Brexit, GFC, 1970s gold rally) are
real signal, not contamination, and are not filtered out — the
strategy has to survive them.

## Strategy rule (locked before any results were observed)

```text
asset           = EUR/USD
source filter   = ecb_sdmx (ECB EUR-quoted reference rate)
timeframe       = 1d
lookback_days   = 200
mode            = long_or_cash
position[t+1]   = 1 if close[t] > SMA200(close)[t] else 0   (lag = 1)
strategy_return[t+1] = position[t+1] * raw_return[t+1]
initial_cash    = 1.0
no leverage, no shorts, no spreads, no slippage, no fees, no carry
```

`assert_long_cash_only` and `assert_no_lookahead` are invoked by
the orchestrator on every backtest call.

## Benchmarks

- **EUR/USD buy-and-hold** — same series, same first active day,
  same initial cash.
- **Cash benchmark** — held at `initial_cash`; total return = 0.
- **Matched-exposure random binary placebo, 20 deterministic seeds.**
  Each placebo permutes a fixed-count {0,1} vector with mean equal
  to the strategy's realised exposure on the same backtest period,
  then computes the same metrics. This removes the trivial
  "more-time-in-market beats less-time-in-market" effect.

## Full-window result (1999-10-11 → 2026-05-08, 6,801 active days)

| Metric                | Strategy   | Buy-and-hold | Cash |
|:----------------------|-----------:|-------------:|-----:|
| Total return          | **+52.27%** | +10.36%     | 0.00% |
| Annualised Sharpe (rf=0) | **0.280** | 0.086       | n/a   |
| Max drawdown          | **-16.48%** | -40.18%     | 0.00% |
| Drawdown improvement  | **+23.70 pp** vs buy-and-hold | — | — |
| Trade count           | 177 (≈6.5/yr signal flips) | 0 | 0 |
| Exposure              | 53.87% time-in-market | 100% | 0% |

**Honest reading.** CAGR is small in absolute terms — roughly 1.6%
strategy vs 0.4% buy-and-hold over ~26 years. EUR/USD as a
reference rate is not a meaningful return source. The strategy's
edge is *almost entirely on the drawdown side*, not on
compounding. Sharpe of 0.28 is weak in absolute terms; it just
happens to clear a buy-and-hold benchmark whose Sharpe is even
weaker.

## Walk-forward result (5 contiguous, equal-length OOS windows)

The rule is locked, so this is an OOS *stability* report — not a
parameter search.

| Window | Dates                  | Strat ret | B&H ret | Strat Sharpe | B&H Sharpe | Strat DD | B&H DD | DD improvement |
|:-------|:-----------------------|----------:|--------:|-------------:|-----------:|---------:|-------:|---------------:|
| 0      | 1999-10-11 → 2005-02-02 | +15.22%  | +22.56% | 0.372 | 0.401 | -16.01% | -24.08% | +8.07 pp |
| 1      | 2005-02-03 → 2010-05-31 | +24.06%  | -5.77%  | 0.640 | -0.052 | -8.12% | -23.56% | +15.44 pp |
| 2      | 2010-06-01 → 2015-09-18 | -2.20%   | -7.22%  | -0.037 | -0.102 | -10.33% | -29.10% | +18.76 pp |
| 3      | 2015-09-21 → 2021-01-13 | +9.35%   | +6.54%  | 0.344 | 0.193 | -6.83% | -14.30% | +7.47 pp |
| 4      | 2021-01-14 → 2026-05-08 | -0.40%   | -3.33%  | 0.011 | -0.045 | -11.10% | -22.01% | +10.91 pp |

- **Strategy beats buy-and-hold return** in 4/5 windows.
- **Strategy beats buy-and-hold Sharpe** in 4/5 windows.
- **Strategy improves drawdown** in 5/5 windows by at least 7.47 pp;
  this is the most stable signal across windows.
- **Window 0 (1999-2005, the early-EUR period) loses to buy-and-hold.**
  The strategy was less invested while the EUR was rising sharply
  off its 0.85 trough; buy-and-hold rode the appreciation. This is
  a real weakness: the rule does badly when a strong directional
  trend is already underway from the very first observation.

## Placebo result (20 deterministic seeds, matched exposure)

20 placebos, each a permuted {0,1} position vector with mean
exposure ≈ 53.87% (matched to the strategy on the same window).

| Metric         | Strategy | Placebo min | Placebo median | Placebo max |
|:---------------|---------:|------------:|---------------:|------------:|
| Total return   | +52.27%  | -26.51%     | +16.96%        | +61.89%     |
| Max drawdown   | -16.48%  | -50.21%     | -29.53%        | -18.86%     |

- **Return percentile vs placebo:** 90th (strategy beats 18 of 20
  placebos on return).
- **Drawdown percentile vs placebo:** 100th (strategy beats every
  placebo on drawdown — i.e. the strategy's drawdown is tighter
  than every random matched-exposure shuffle).

The placebo design is conservative: matched-exposure random shuffle
already absorbs the "in market more" effect, so beating its median
return is not free. Beating every placebo on drawdown suggests the
SMA-200 filter is, in fact, doing its job of avoiding the worst
trough periods rather than just sampling them randomly.

## Scorecard

```text
strategy_name         : fx_eod_trend_v1
asset                 : EUR/USD
verdict               : PASS
total_return          : +52.27%
benchmark_total_return: +10.36%
cash_total_return     :  0.00%
sharpe                : 0.280
benchmark_sharpe      : 0.086
max_drawdown          : -16.48%
benchmark_max_drawdown: -40.18%
drawdown_improvement_pp: +0.2370 (= 23.70 pp tighter than buy-and-hold)
placebo_return_percentile : 90.0
placebo_drawdown_percentile: 100.0
placebo_median_return : +16.96%
placebo_median_drawdown: -29.53%
trade_count           : 177
exposure_percent      : 53.87%
checks_passed         : 9 / 9
data_quality_verdict  : WARNING
```

All 9 locked checks pass:

1. total_return > 0 — `pass_positive_return`
2. sharpe > benchmark_sharpe — `pass_sharpe_beats_benchmark`
3. drawdown ≥ 5pp tighter than benchmark — `pass_drawdown_tighter`
4. total_return > placebo median — `pass_beats_placebo_return`
5. drawdown tighter than placebo median — `pass_beats_placebo_drawdown`
6. trade_count ≥ 20 — `pass_min_trade_count`
7. position ≤ 1 everywhere — `no_leverage`
8. position ≥ 0 everywhere — `no_shorts`
9. SMA at index t uses only close ≤ t — `no_lookahead` (asserted by
   the orchestrator on every run)

## Verdict

**PASS** on the locked criteria.

### Exact reasons for verdict

- Total return is positive (+52.27%) over the full window.
- Annualised Sharpe (0.280) exceeds the buy-and-hold benchmark
  (0.086).
- Max drawdown (-16.48%) is 23.70 pp tighter than buy-and-hold
  (-40.18%) — well above the 5 pp bar.
- Strategy total return beats the placebo median return (+52.27%
  vs +16.96%), at the 90th percentile of placebos.
- Strategy drawdown is tighter than the placebo median drawdown
  (-16.48% vs -29.53%), at the 100th percentile.
- Trade count (177) exceeds the 20-flip minimum.
- Position is binary {0, 1}; no leverage and no shorts (asserted).
- No lookahead in the SMA computation (asserted on every run).

## Limitations (be honest)

1. **Reference rate, not a tradable quote.** ECB EUR/USD is a daily
   fix, not a broker price. Real execution carries a spread (≈
   0.5-1 bp interbank, several bp at retail), slippage, and an
   interest-rate carry differential that this study ignores.
   Applying spreads + plausible slippage at every signal flip would
   cut the strategy's net edge measurably; whether it would survive
   is an open question this branch does not answer.
2. **Carry not modelled.** A real long EUR/USD position earns the
   EUR-USD overnight rate differential, which has been negative for
   most of the sample period. A long-only strategy ignoring carry
   overstates net return; in the post-2008 negative-rate years this
   could be a 1-2% per annum drag. Cash benchmark assumes 0% rather
   than a real-world money-market rate, which is also generous.
3. **Single asset, single rule, single threshold.** No external
   validation on other FX pairs and no robustness sweep on the
   200-day lookback. The 200-day SMA was chosen because it is the
   canonical EOD trend benchmark — not because it was tuned. Even
   so, the result on a single (asset, rule, threshold) triplet
   should not be over-extrapolated.
4. **Modest absolute Sharpe.** 0.28 annualised is unimpressive by
   itself; the strategy clears the bar mostly because the buy-and-
   hold benchmark Sharpe is essentially zero. This is not a story
   about a high-quality return stream — it is a story about
   drawdown control on a near-zero-trend series.
5. **Window 0 underperforms buy-and-hold.** The strategy is less
   invested during a directional rise off the 1999-2001 EUR low;
   buy-and-hold rides it. The rule is naturally weaker when a
   strong move is already under way at the start of the window.
6. **No transaction costs assumed.** 177 flips over 26 years ≈ 7
   round-turn pairs per year. Even at 1 bp per round-turn the
   gross-to-net cost is small (~0.07% per year), but it is real and
   not accounted for.
7. **Multiple-testing correction not applicable here** because the
   rule was locked before results — but a future researcher should
   not interpret PASS as a license to sweep parameters and
   re-evaluate.

## Is further research justified?

Yes, modestly. The drawdown control is consistent across all 5
walk-forward windows and across every matched-exposure placebo,
which is the kind of stability worth following up. A reasonable
next step is **not** to enable execution; it is to extend the
research with a quality-controlled robustness audit:

- Robustness on the lookback parameter (e.g. {100, 150, 200, 250,
  300}-day SMA) — *purely diagnostic*, with the 200-day result
  already locked in this report so no parameter is re-chosen after
  seeing results.
- Apply the same locked rule to USD/JPY, GBP/USD, EUR/JPY,
  EUR/GBP, EUR/CHF, USD/CHF, XAU/USD on the same dataset, again
  diagnostically, to see whether the drawdown-control story
  generalises beyond EUR/USD.
- A simple cost-sensitivity sweep: at what assumed bp/round-turn
  does the strategy stop beating buy-and-hold on Sharpe?

## Is paper trading justified?

**No.** Even with a PASS verdict, paper trading is not justified
on this branch:

- The rule was tested on a reference-rate series, not a tradable
  quote.
- Carry, spread, slippage, and fees are not modelled.
- A single (asset, rule, threshold) triplet has been examined.
- The Sharpe is modest in absolute terms.

Per the project's safety contract, paper trading and live trading
remain **blocked at the safety lock** regardless of any strategy's
verdict. The registry entry for `fx_eod_trend` carries
`paper_trading_allowed=False` and `live_trading_allowed=False`, and
the `test_no_strategy_in_registry_is_paper_or_live_allowed` test
will fail if either flag is ever flipped.

If a future branch wants to lift the paper-trading block, it must:

1. add a separate, audited execution module (none exists in v1);
2. assume realistic spread, slippage, and carry, and re-run the
   full battery;
3. provide independent review of the resulting scorecard;
4. release the safety lock through the documented unlock procedure
   in `docs/unlock_procedure.md`.

None of those steps are part of this branch. This branch ends at
the scorecard.

## Files generated (gitignored)

```text
results/fx_eod_trend_backtest.csv      # 6,801 rows, full daily history
results/fx_eod_trend_walk_forward.csv  # 5 OOS-style stability windows
results/fx_eod_trend_placebo.csv       # 20 deterministic placebo seeds
results/fx_eod_trend_scorecard.csv     # one-row verdict
```

All four are excluded by the existing `results/*.csv` gitignore
rule. None are committed.
