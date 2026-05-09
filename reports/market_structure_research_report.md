# Market-structure research — honest verdict

This is the closure document for branch
**`research/strategy-2-market-structure`**, the third strategy family
tested after v1 (price-based TA) and the failed funding/derivatives
branch (`research/fail-1-funding-derivatives`). The hypothesis was that
on-chain liquidity flows (DefiLlama TVL + stablecoin supply) plus pure
BTC market-structure metrics (Blockchain.com hash rate, transactions,
market cap) plus alt-basket breadth could decide whether a portfolio
should hold BTC, hold an alt basket, or hold cash — better than simple
momentum or buy-and-hold.

> **Verdict: FAIL.** 6 of 9 PASS checks satisfied. The allocator beat
> simple momentum (57 % of OOS windows), beat the random state-picker
> placebo, and held drawdown 11 pp tighter than BTC's. But it **lost to
> BTC in 11 of 14 windows, lost to the equal-weight basket in 9 of 14,
> and registered 0 % stability** against the 60 % bar. It is too
> conservative: it sits in cash whenever BTC is below its 200d MA, and
> the on-chain liquidity rules don't pick up risk-on regimes fast
> enough to capture the rebound rallies.

## Files added on this branch

| File | Role |
| --- | --- |
| [`src/market_structure_data_collector.py`](../src/market_structure_data_collector.py) | DefiLlama + Blockchain.com collector. No keys, no paid endpoints. |
| [`src/market_structure_data_audit.py`](../src/market_structure_data_audit.py) (already on branch) | Free-tier source audit. |
| [`src/market_structure_signals.py`](../src/market_structure_signals.py) | 28-column signal table, lookahead-free, partial-vs-full regression test. |
| [`src/strategies/market_structure_allocator.py`](../src/strategies/market_structure_allocator.py) | State-driven BTC / alt-basket / cash allocator. |
| [`src/market_structure_research.py`](../src/market_structure_research.py) | Walk-forward + state-picker placebo + 9-check scorecard. |
| [`tests/test_market_structure_data_collector.py`](../tests/test_market_structure_data_collector.py) | 13 unit tests. |
| [`tests/test_market_structure_signals.py`](../tests/test_market_structure_signals.py) | 12 unit tests, including a strict partial-vs-full lookahead check. |
| [`tests/test_market_structure_allocator.py`](../tests/test_market_structure_allocator.py) | 10 unit tests. |
| [`tests/test_market_structure_research.py`](../tests/test_market_structure_research.py) | 10 unit tests, including the "beating placebo only ≠ PASS" check. |
| `main.py` (+7 commands) | `download_market_structure_data`, `audit_market_structure_data`, `market_structure_signals`, `market_structure_allocator`, `market_structure_walk_forward`, `market_structure_placebo`, `market_structure_scorecard`, `research_all_market_structure`. |
| `streamlit_app.py` (Market Structure card) | 8-tab section: Coverage, State over time, Now, Trends, Equity vs benchmarks, Walk-forward, Placebo, Scorecard. |

Combined test suite: **245 / 245 passing** on this branch.

## Market-structure data coverage

| Source | Dataset | Rows | Coverage |
| --- | --- | --- | --- |
| DefiLlama | Total TVL (all chains) | 3 147 | 3 146 days (since 2017-09) |
| DefiLlama | Stablecoin supply (total) | 3 084 | 3 083 days (since 2017-11) |
| Blockchain.com | BTC market cap (USD) | 1 503 | 1 824 days (~5 years) |
| Blockchain.com | BTC hash rate | 1 822 | 1 824 days |
| Blockchain.com | BTC transactions / day | 1 822 | 1 824 days |
| Binance (cached) | Daily OHLCV × 10 symbols | 15 000 | 1 499 days |

All six clear the 4-year (1 460-day) threshold. The Blockchain.com BTC
market-cap series has 145 small day-gaps (largest 2.2 days) — they are
declared in the coverage CSV but do not invalidate the 30/90-day return
features used downstream.

## Signal engineering

`market_structure_signals.py` produces a daily 28-column table whose
states are a pure function of the row's features:

* **Returns** — `btc_return_30d`, `btc_return_90d`, `alt_basket_return_30d`,
  `alt_basket_return_90d`. Backward-only (`close[t] / close[t-N] - 1`).
* **Trend** — `btc_above_200d_ma` (rolling 200-day mean, `min_periods=200`).
* **Liquidity** — `total_tvl_return_30d/90d`, `stablecoin_supply_return_30d/90d`.
* **On-chain** — `btc_market_cap_return_30d`, `btc_hash_rate_return_30d`,
  `btc_transactions_return_30d`.
* **Cross-section** — `alt_basket_above_200d_ma_pct` (breadth),
  `alt_basket_vs_btc_30d/90d`.
* **Composite scores** in [-1, 1] / [0, 1] —
  `liquidity_score`, `onchain_health_score`, `alt_risk_score`,
  `defensive_score`.
* **`market_structure_state`** ∈
  {`alt_risk_on`, `btc_leadership`, `defensive`, `neutral`, `unknown`}.

A `partial_vs_full` test verifies that recomputing the signal table on
the first 180 days only produces identical values to those rows in the
full-history run — proves no future row leaks into a past feature.

## Allocator strategy

`MarketStructureAllocatorStrategy` is a portfolio strategy keyed only
on the latest signal row at or before the rebalance bar. Allocation by
state:

| State | Allocation |
| --- | --- |
| `alt_risk_on` | Equal-weight Top-5 alts by 90d momentum (BTC excluded) |
| `btc_leadership` | 100 % BTC/USDT |
| `defensive` | 100 % cash (`return {}`) |
| `neutral` | 100 % BTC/USDT |
| `unknown` | 100 % cash |

Long-only, no leverage, no shorting, no margin. Weekly rebalance
through the existing portfolio backtester (next-bar-open execution,
fees, slippage). Assets without 90 days of history are excluded from
any selection.

## Market-structure state distribution (1 500 rows over 4 years)

```
defensive       838  (55.9 %)
neutral         280  (18.7 %)
alt_risk_on     258  (17.2 %)
unknown          90  ( 6.0 %)
btc_leadership   34  ( 2.3 %)
```

The defensive state dominates because BTC was below its 200d MA for
large portions of 2022, mid-2024, and parts of 2025 — and the
defensive trigger is the first one evaluated. This is the single
biggest reason the strategy underperforms BTC: it hands away BTC
exposure during the early innings of every recovery rally.

## Benchmark result (full 4-year window)

```
market_structure_allocator     return = +0.75 %    max DD = −54.79 %   sharpe = 0.22
BTC_buy_and_hold               return = +74.84 %   max DD = −66.12 %   sharpe = 0.52
ETH_buy_and_hold               return = −32.55 %   max DD = −71.74 %   sharpe = 0.21
equal_weight_basket            return = −15.32 %   max DD = −66.84 %   sharpe = 0.26
simple_momentum                return = +208.81 %  max DD = −58.23 %   sharpe = 0.76
```

The allocator's headline drawdown is 11.3 pp better than BTC's — the
"defensive" state genuinely sidesteps deep crashes. But it also
sidesteps the recoveries.

## Walk-forward result (14 disjoint OOS windows × 90 days)

```
                                     allocator   BTC    basket   simple
window 1   2022-09 → 2022-12             0.00   −12.84   −15.45   −4.58
window 2   2022-12 → 2023-03            28.25    67.41    40.12    4.10
window 3   2023-03 → 2023-06            −8.10    12.54    −7.30   −2.67
window 4   2023-06 → 2023-09             0.00   −12.75    −9.34  −14.21
window 5   2023-09 → 2023-12            36.65    65.04   129.01  226.81
window 6   2023-12 → 2024-03            29.43    54.29    42.73   17.04
window 7   2024-03 → 2024-06           −10.39    −0.50   −19.19  −12.49
window 8   2024-06 → 2024-09           −20.25   −10.40   −12.81  −31.91
window 9   2024-09 → 2024-12            62.62    73.20   147.37  198.15
window 10  2024-12 → 2025-03           −35.22   −20.48   −34.69  −45.68
window 11  2025-03 → 2025-06             2.40    28.45     8.65   −5.30
window 12  2025-06 → 2025-09            −2.30     9.55    48.86   30.76
window 13  2025-09 → 2025-12           −28.27   −20.70   −35.38  −25.62
window 14  2025-12 → 2026-03             0.00   −24.39   −32.91    0.00

beats_btc:              3/14 (21.4 %)
beats_basket:           5/14 (35.7 %)
beats_simple_momentum:  8/14 (57.1 %)
profitable:             5/14 (35.7 %)
stability score:        0/14 ( 0.0 %)   profitable AND beats_btc AND beats_basket AND beats_simple
```

Stability is `0` — there is no single window where the allocator was
profitable AND beat all three benchmarks. Several windows where it
was profitable (2, 5, 6, 9) are exactly the windows where BTC and the
basket were even more profitable.

## Placebo result (20 seeds, random BTC/alt/cash state-picker)

```
strategy full-window return       +0.75 %
placebo median return             −1.47 %     ← strategy beats
strategy full-window max DD       −54.79 %
placebo median max DD             −64.77 %    ← strategy beats
n_seeds                           20
```

The state-picker placebo (BTC, alt-basket, or cash with equal
probability at each rebalance) is a properly hard baseline — it is
exposed to the universe and rebalances at the same cadence. The
allocator beats it by ≈2.2 pp on return and 10 pp on drawdown.

## Scorecard result (9 checks, FAIL)

| # | Check | Pass? | Value |
| - | --- | --- | --- |
| 1 | positive_return | ✅ | +0.75 % |
| 2 | beats_btc_oos (>50 %) | ❌ | 21.4 % |
| 3 | beats_basket_oos (>50 %) | ❌ | 35.7 % |
| 4 | beats_simple_momentum_oos (>50 %) | ✅ | 57.1 % |
| 5 | beats_placebo_median | ✅ | +0.75 % vs −1.47 % |
| 6 | oos_stability_above_60 | ❌ | 0.0 % |
| 7 | at_least_10_rebalances | ✅ | 190 |
| 8 | dd_within_btc_gap_20pp | ✅ | gap = +11.3 pp (better) |
| 9 | enough_market_structure_coverage | ✅ | ok |

**checks_passed = 6 / 9 → verdict = FAIL.**

## Streamlit changes

A new `Market Structure` card is added to the dashboard, with eight
internal tabs: Coverage, State over time, Now, Trends, Equity vs
benchmarks, Walk-forward, Placebo, Scorecard. The Coverage tab raises a
clear warning when any dataset falls below the 4-year threshold —
preventing a silent INCONCLUSIVE verdict downstream.

## CLI commands added

```
python main.py download_market_structure_data
python main.py audit_market_structure_data
python main.py market_structure_signals
python main.py market_structure_allocator
python main.py market_structure_walk_forward
python main.py market_structure_placebo
python main.py market_structure_scorecard
python main.py research_all_market_structure
```

Every command is research-only. None call broker APIs, none accept
keys, none touch live execution.

## Tests added

* 13 collector tests — parsers (BC.com + DefiLlama), normaliser,
  gap stats, coverage row builder, schema lock, network-failure
  handling, no-API-keys invariant, no-order-plumbing invariant.
* 12 signal tests — schema lock, lookahead on `btc_return_30d` and
  `btc_above_200d_ma`, alt-universe excludes BTC, breadth correctness,
  stablecoin/TVL return correctness, defensive trigger, alt_risk_on
  trigger, missing-data → unknown, and a strict partial-vs-full no-
  lookahead check. Includes a regression test that locks `timestamp`
  to milliseconds since epoch (an earlier version stored kiloseconds,
  which silently caused the allocator to never match a signal row).
* 10 allocator tests — every state's allocation, no-leverage,
  short-history exclusion, lookahead invariance.
* 10 research tests — OOS-window mechanics, placebo determinism,
  placebo does not consult signals, scorecard FAIL/INCONCLUSIVE/PASS
  paths, "beating placebo only is not PASS", coverage gating.

**Test results: 245 / 245 passing** (191 v1 + 9 audit + 35 market
structure).

## The questions, answered without softening

* **Does it beat BTC?** No. Beat BTC in only 3 of 14 OOS windows (21 %).
  Full-window return: +0.75 % vs BTC +74.84 %.
* **Does it beat the equal-weight basket?** No. 5 of 14 (36 %).
* **Does it beat simple momentum?** **Yes**, 8 of 14 (57 %). This is
  the first time on this project that a non-BTC strategy has cleared
  the simple-momentum bar — but only marginally, and not in a way that
  rescues the BTC and basket failures.
* **Does it beat the placebo?** Yes, on both return (+0.75 % vs
  −1.47 %) and drawdown (−54.79 % vs −64.77 %).
* **Does it deserve paper testing?** **No.** Two of the four primary
  benchmarks (BTC, basket) are not beaten in the majority of windows,
  and stability is 0. The placebo and simple-momentum checks alone are
  not enough — they were not enough on the funding branch either.

## Limitations

* The defensive state — defined as BTC below its 200d MA, OR
  TVL+stables both contracting, OR alt breadth < 35 % — fires on
  55.9 % of all rows. This is the dominant reason the strategy
  underperforms: it spends most of the period in cash. The state rules
  were fixed by spec and were NOT softened to chase a verdict.
* True historical BTC dominance is unavailable on free APIs; the alt
  basket here is a synthetic equal-weight index of 9 large-cap
  USDT pairs, not a true total-altcoin basket.
* The Blockchain.com BTC market-cap series has 145 day-gaps (largest
  2.2 days). For 30/90-day return features this is harmless, but a
  finer-grained on-chain study would want a denser source.
* Walk-forward windows were inherited from the v1 portfolio scorecard
  (180 IS / 90 OOS / 90 step). They were not changed for this branch.
* The 60 % stability bar is the same conservative one v1 and the
  funding branch failed against. **It was not lowered.**

## What this does NOT change about v1

* **No strategy has passed.** Market-structure joins the failed list.
* **Do not paper trade.**
* **Do not connect Kraken** or any other broker.
* **Do not add API keys.**
* **BTC buy-and-hold remains the strongest practical baseline.**
* No threshold was lowered. No parameter was tuned to chase a verdict.
  The score weights, the breadth thresholds (50 % / 35 %), and the 200d
  MA are all the values listed in the spec.

## Exact next step

Pick exactly one of the following for the next session:

1. **Sentiment / Fear & Greed historical** (alternative.me public
   feed has multi-year daily history, no key). The hypothesis is that
   sentiment regime turns *lead* price + on-chain regime turns by 1–4
   weeks. If it does, layering it on top of THIS allocator (use
   sentiment as the regime-switch signal instead of the static
   liquidity-and-breadth rules) might lift the BTC win-rate from 21 %.
2. **Volatility-targeted allocator on the same signal class** —
   replace the all-or-nothing defensive rule with a continuous BTC
   weight that scales down as drawdown deepens, instead of cutting to
   0 %. This is closer to the v2 candidates list in the v1 README.
3. **Re-run THIS allocator with a paid OI feed** to combine market-
   structure liquidity with the funding+OI signals from
   `research/fail-1-funding-derivatives` (which was INCONCLUSIVE on
   data length, not a real strategy failure).
4. **Accept BTC buy-and-hold as the baseline.** Still on the table,
   still the only direction that does not require more research.

The rule from v1 stands. PASS requires beating BTC, the basket, the
placebo, AND simple momentum, with > 60 % stability across ≥ 5 OOS
windows AND a drawdown not worse than BTC's by > 20 pp AND adequate
data coverage. Anything weaker is FAIL or INCONCLUSIVE.

## Final rule

If the next experiment also fails, say it fails. Do not paper trade.
Do not connect Kraken. Do not lower thresholds. The goal is robust
edge discovery — the engine is a falsification tool, not a backtest
beautifier.
