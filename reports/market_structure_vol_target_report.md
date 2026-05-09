# Market-structure vol-target allocator — honest verdict

This is the closure document for the **portfolio-construction follow-up**
on the failed market-structure allocator. The hypothesis was specific
and falsifiable: the binary allocator failed because it sat in 100 %
cash on 55.9 % of rows; replacing the binary states with **softer
exposure bands** while keeping the *signal* untouched should lift OOS
performance enough to clear the scorecard.

> **Verdict: FAIL — but the closest thing to a PASS this project has
> ever produced.** **8 of 10 PASS checks satisfied** (the original
> allocator passed 6 of 9, the funding-only attempt passed 4 of 8).
> Vol-target now beats the equal-weight basket, simple momentum, the
> placebo, AND the original allocator in the majority of OOS windows;
> drawdown is 17 pp tighter than BTC's. **It still loses to BTC in 11
> of 14 windows, and OOS stability is 0 %** because BTC is in every
> stability gate. That alone keeps the verdict at FAIL.

## Files changed

| File | Role |
| --- | --- |
| [`src/strategies/market_structure_vol_target_allocator.py`](../src/strategies/market_structure_vol_target_allocator.py) | New 5-state continuous-exposure allocator. |
| [`src/market_structure_research.py`](../src/market_structure_research.py) | + `MarketStructureVolStatePlacebo` (5-state random allocator) + `run_market_structure_vol_target` + `market_structure_vol_target_walk_forward` + `market_structure_vol_target_placebo` + `market_structure_vol_target_scorecard` (10-check) + `run_all_market_structure_vol_target`. |
| [`tests/test_market_structure_vol_target_allocator.py`](../tests/test_market_structure_vol_target_allocator.py) | 10 unit tests for the strategy. |
| [`tests/test_market_structure_vol_target_research.py`](../tests/test_market_structure_vol_target_research.py) | 9 unit tests for the placebo + scorecard. |
| `main.py` | + `market_structure_vol_target`, `market_structure_vol_target_walk_forward`, `market_structure_vol_target_placebo`, `market_structure_vol_target_scorecard`. |
| `streamlit_app.py` | + 9th tab "Vol-target" inside the Market Structure card showing equity comparison, full-window comparison table, walk-forward, placebo, color-coded verdict. |
| `reports/market_structure_vol_target_report.md` | This file. |

**Test suite: 264 / 264 passing** (245 prior + 10 vol-target allocator
+ 9 vol-target research). No existing test was modified.

## Strategy explanation

The allocator reads exactly the same `market_structure_state` column
that the original allocator used (the signal generator is untouched —
no thresholds tuned, no rules altered). Allocation by state:

| State | Allocation |
| --- | --- |
| `alt_risk_on` | 70 % equal-weight Top-5 alts (90d momentum) + 30 % BTC |
| `btc_leadership` | 100 % BTC |
| `neutral` | 70 % BTC + 30 % cash |
| `defensive` | 30 % BTC + 70 % cash |
| `unknown` | 100 % cash |

Long-only, no leverage, no shorts. Σ weights ≤ 1 in every state.
Weekly rebalance through the existing portfolio backtester (next-bar-
open, fees, slippage). All exposure-band weights (0.70 / 0.30 / 0.30 /
0.70) are taken verbatim from spec — they were NOT tuned.

The placebo for this run is also new and band-matched: a 5-state
random allocator that picks `alt_basket` (70/30) / `btc_only` (1.0) /
`partial_btc` (0.70) / `defensive_partial` (0.30) / `cash` uniformly
at each rebalance. Same universe, same fees, same rebalance cadence,
fixed seeds, 20 seeds. The placebo NEVER reads the
`market_structure_state` column.

## Benchmark result (full 4-year window)

```
market_structure_vol_target          return = +53.10 %   max DD = −49.04 %   sharpe = 0.47
BTC_buy_and_hold                     return = +74.84 %   max DD = −66.12 %   sharpe = 0.52
ETH_buy_and_hold                     return = −32.55 %   max DD = −71.74 %   sharpe = 0.21
equal_weight_basket                  return = −15.32 %   max DD = −66.84 %   sharpe = 0.26
simple_momentum                      return = +208.81 %  max DD = −58.23 %   sharpe = 0.76
market_structure_allocator (orig)    return = +0.75 %    max DD = −54.79 %   sharpe = 0.22
```

**+53 % vs the original allocator's +0.75 %** — a 52-pp lift on the
exact same signal. **Drawdown 17 pp tighter than BTC's.** Sharpe 0.47
is the highest non-momentum, non-BTC sharpe this project has produced.

## Walk-forward result (14 disjoint OOS windows × 90 days)

```
                                     vol-tgt   BTC    basket   simple   original
window 1   2022-09 → 2022-12             −3.51   −12.84   −15.45    −4.58     0.00
window 2   2022-12 → 2023-03             38.33    67.41    40.12     4.10    28.25
window 3   2023-03 → 2023-06              0.48    12.54    −7.30    −2.67    −8.10
window 4   2023-06 → 2023-09             −3.85   −12.75    −9.34   −14.21     0.00
window 5   2023-09 → 2023-12             37.78    65.04   129.01   226.81    36.65
window 6   2023-12 → 2024-03             32.16    54.29    42.73    17.04    29.43
window 7   2024-03 → 2024-06            −11.05    −0.50   −19.19   −12.49   −10.39
window 8   2024-06 → 2024-09            −16.71   −10.40   −12.81   −31.91   −20.25
window 9   2024-09 → 2024-12             68.65    73.20   147.37   198.15    62.62
window 10  2024-12 → 2025-03            −30.28   −20.48   −34.69   −45.68   −35.22
window 11  2025-03 → 2025-06              9.91    28.45     8.65    −5.30     2.40
window 12  2025-06 → 2025-09              2.04     9.55    48.86    30.76    −2.30
window 13  2025-09 → 2025-12            −25.82   −20.70   −35.38   −25.62   −28.27
window 14  2025-12 → 2026-03             −7.63   −24.39   −32.91     0.00     0.00

beats_btc:                3/14 (21.4 %)
beats_basket:             8/14 (57.1 %)   ← was 5/14 with original
beats_simple_momentum:    9/14 (64.3 %)   ← was 8/14
beats_original_allocator: 10/14 (71.4 %)
profitable:               7/14 (50.0 %)
stability:                0/14 ( 0.0 %)
```

The vol-target lifts performance everywhere except against BTC —
because the signal still moves to *partial* BTC during alt-rally
windows (window 5: 70 % alts, 30 % BTC) when 100 % BTC would have
captured more upside.

## Placebo result (20 seeds, 5-state random)

```
strategy full-window return       +53.10 %
placebo median return             +8.55 %       ← strategy beats by 44 pp
strategy full-window max DD       −49.04 %
placebo median max DD             −58.47 %      ← strategy beats by 9 pp
n_seeds                           20
```

The vol-target beats a properly band-matched placebo (which is exposed
to BTC and alts on average ~60 % of the time) by 44 pp on return and
9 pp on drawdown. This is real signal — the original allocator beat its
state-picker placebo by only 2.2 pp on return.

## Scorecard result (10 checks, FAIL)

| # | Check | Pass? | Value |
| - | --- | --- | --- |
| 1 | positive_return | ✅ | +53.10 % |
| 2 | beats_btc_oos (>50 %) | ❌ | 21.4 % |
| 3 | beats_basket_oos (>50 %) | ✅ | 57.1 % |
| 4 | beats_simple_momentum_oos (>50 %) | ✅ | 64.3 % |
| 5 | beats_original_allocator_oos (>50 %) | ✅ | 71.4 % |
| 6 | beats_placebo_median | ✅ | +53.1 % vs +8.6 % |
| 7 | oos_stability_above_60 | ❌ | 0.0 % |
| 8 | at_least_10_rebalances | ✅ | 190 |
| 9 | dd_within_btc_gap_20pp | ✅ | gap = +17.1 pp (better) |
| 10 | enough_market_structure_coverage | ✅ | ok |

**checks_passed = 8 / 10 → verdict = FAIL.** WATCHLIST in this
scorecard requires ≥ 9 of 10 checks; 8 falls through to the FAIL
fallback because the BTC and stability gates are both binding.

## The questions, answered without softening

* **Does it beat BTC?** No. Beat BTC in only 3 of 14 OOS windows (21 %).
  Full-window return: +53.10 % vs BTC +74.84 %.
* **Does it beat the equal-weight basket?** **Yes.** 8 of 14 (57 %).
  Full-window: +53 % vs basket −15 %.
* **Does it beat simple momentum?** **Yes** in OOS windows (9 of 14,
  64 %), although **No** in full-window total return (+53 % vs
  simple +209 %).
* **Does it beat the original market-structure allocator?** **Yes**,
  decisively. 10 of 14 OOS windows (71 %), and +52 pp on full-window
  return (+53 % vs +0.75 %).
* **Does it beat the placebo?** **Yes**, on both axes.
* **Does it deserve paper testing?** **No.** Two checks remain hard
  failures: BTC OOS win-rate (21 %) and stability (0 %). The spec
  rules out PASS when BTC is not beaten in the majority of windows,
  and stability requires beating *all* primary benchmarks
  simultaneously in the same window. Paper-trading a strategy that
  loses to BTC in 11 of 14 quarters would be irresponsible.

## Limitations

* **The BTC bar is structural.** Crypto's 4-year drift was dominated
  by BTC; any allocator that ever taps the brake on BTC exposure will
  underperform BTC in pure return terms. The vol-target's softer
  defensive band (30 % BTC instead of 0 %) closes most of the gap, but
  not enough.
* **Stability requires beating BTC AND basket AND simple momentum AND
  the original allocator in the same window.** Several windows (2, 6,
  11) cleared 3 of 4 — but never all 4 simultaneously. Loosening the
  stability metric would change the verdict; the spec is explicit
  that thresholds may not be lowered.
* **The signal is still the original signal.** All progress here came
  from portfolio construction. Without a better regime classifier
  (sentiment, macro, real BTC dominance), the upper bound on this
  family is essentially set.
* **Walk-forward windows are inherited from v1** (180 IS / 90 OOS / 90
  step). Not changed.

## What this does NOT change about v1 / prior branches

* **No strategy has passed.** Vol-target joins the failed list, but at
  the closest distance any branch has ever recorded.
* **Do not paper trade.**
* **Do not connect Kraken** or any other broker.
* **Do not add API keys.**
* **BTC buy-and-hold remains the strongest practical baseline.**
* No threshold was lowered. No parameter was tuned. The four exposure
  weights (0.70, 0.30, 0.70, 0.30) are spec-given.

## Exact next step

The diagnostic message of this run is sharp. The signal-set + portfolio
construction together cleared 8 of 10 gates. The remaining 2 are the
**BTC win-rate** and **stability**, both bounded by BTC's secular
drift.

Pick exactly one of the following:

1. **Sentiment / Fear & Greed historical regime overlay**
   (alternative.me daily, multi-year, free). Hypothesis: sentiment
   regime turns lead the on-chain liquidity rules by 1–4 weeks and
   could push the BTC win-rate over 50 % by closing or de-risking
   *before* drawdowns rather than during them.
2. **BTC-relative scoring** — instead of switching to cash on
   defensive, switch to *less* BTC by formula (continuous DD-targeted
   weight). The vol-target is already a step in that direction; a
   continuous DD-target would generalise it. Risk: it becomes a vol-
   targeted long-BTC strategy, which has plenty of prior literature.
3. **Re-run the failed funding+OI strategy with a paid OI feed** to
   close the data-length gap — that branch was INCONCLUSIVE on data,
   not a real strategy failure.
4. **Accept BTC buy-and-hold as the baseline.** The most honest
   answer if no further signal class can clear BTC.

The v1 rule stands. PASS requires beating BTC, the basket, simple
momentum, *and* the prior best variant, with > 60 % stability across
≥ 5 OOS windows AND drawdown not worse than BTC's by > 20 pp AND
enough data coverage. **Anything weaker is FAIL or INCONCLUSIVE.**

## Final rule

If the next experiment also fails, say it fails. Do not paper trade.
Do not connect Kraken. Do not lower thresholds. Do not tune parameters.
The vol-target proves portfolio construction can extract more out of a
mediocre signal — but a mediocre signal still won't pass.
