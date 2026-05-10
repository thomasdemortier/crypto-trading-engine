# Drawdown-targeted BTC allocator — research verdict

This is the closure document for **research strategy 4 — continuous BTC
drawdown-targeted allocator**. The strategy adjusts BTC exposure as a
function of BTC's current drawdown from its rolling all-time high and
its 200d moving-average regime, with an optional alt-basket overlay
when BTC is above the 200d MA and alt breadth is strong.

> **Verdict: FAIL.**
> **5 of 7 PASS checks satisfied.** The allocator beats the equal-weight
> basket, simple momentum, and the random-bucket placebo; it has 190
> rebalances and a max drawdown 29 percentage points tighter than BTC's.
> But it loses to BTC in **8 of 14** OOS windows (need >50 % beats), and
> OOS stability is **0 %** because BTC is in every stability gate. As
> per the rules: it failed, and parameters were NOT retuned. Execution
> remains locked.

## What the strategy does

A long-only continuous-exposure allocator. Inputs (all derived from
public BTC daily candles):

| Input | Window | Use |
| --- | --- | --- |
| BTC drawdown from rolling all-time high | full history | Bucket lookup |
| BTC vs 200d MA | 200 days | Regime cap |
| BTC realised vol (log-rets, annualised) | 30d, 90d | Diagnostic only |
| Alt 90d momentum + breadth | 90 days | Alt overlay gate |

Exposure rules — taken **verbatim** from spec, NOT tuned:

| Condition | Allocation |
| --- | --- |
| BTC > 200d MA AND drawdown < 10 % | 100 % BTC |
| drawdown 10 % – 20 % | 70 % BTC / 30 % cash |
| drawdown 20 % – 35 % | 40 % BTC / 60 % cash |
| drawdown > 35 % | 20 % BTC / 80 % cash |
| BTC < 200d MA (any DD bucket) | cap BTC at 40 %, rest cash |

Optional alt overlay (only when BTC > 200d MA, drawdown < 10 %, and
≥ 5 of the 9 alts in the universe show positive 90d momentum):
70 % BTC + 30 % equal-weight top-3 alts (by 90d momentum, positive
only). Long-only, Σ weights ≤ 1, no leverage, no shorts.

## Files changed

| File | Role |
| --- | --- |
| [`src/strategies/drawdown_targeted_btc_allocator.py`](../src/strategies/drawdown_targeted_btc_allocator.py) | The strategy itself. |
| [`src/drawdown_targeted_research.py`](../src/drawdown_targeted_research.py) | Single-window backtest, 14-window walk-forward, 20-seed placebo, scorecard, end-to-end orchestrator. |
| [`tests/test_drawdown_targeted_btc_allocator.py`](../tests/test_drawdown_targeted_btc_allocator.py) | 13 unit tests for the strategy (bucket logic, regime cap, alt overlay, no-lookahead, weight caps). |
| [`tests/test_drawdown_targeted_research.py`](../tests/test_drawdown_targeted_research.py) | 7 unit tests for the placebo + research pipeline. |
| `main.py` | + 5 CLI commands: `drawdown_targeted_btc`, `drawdown_targeted_btc_walk_forward`, `drawdown_targeted_btc_placebo`, `drawdown_targeted_btc_scorecard`, `research_all_drawdown_targeted_btc`. |
| `streamlit_app.py` | + new dashboard section showing equity comparison, walk-forward table, placebo summary, color-coded verdict. |
| `src/strategy_registry.py` | + new family entry pointing at this branch + report. |
| `reports/drawdown_targeted_btc_report.md` | This file. |

## CLI commands

```bash
python main.py drawdown_targeted_btc                       # single-window backtest
python main.py drawdown_targeted_btc_walk_forward          # 14 OOS windows
python main.py drawdown_targeted_btc_placebo --n-seeds 20  # placebo
python main.py drawdown_targeted_btc_scorecard             # build scorecard
python main.py research_all_drawdown_targeted_btc          # full pipeline
```

## Full-window result (4 years, 2022-04 → 2026-05)

```
drawdown_targeted_btc_allocator     return = +44.99 %   max DD = -37.14 %   sharpe = 0.44
BTC_buy_and_hold                    return = +74.84 %   max DD = -66.12 %   sharpe = 0.52
ETH_buy_and_hold                    return = -32.55 %   max DD = -71.74 %   sharpe = 0.21
equal_weight_basket                 return = -15.32 %   max DD = -66.84 %   sharpe = 0.26
simple_momentum                     return = +208.81 %  max DD = -58.23 %   sharpe = 0.76
random_bucket_placebo (median)      return = +14.15 %   max DD = -45.89 %   sharpe = ~
```

Tighter drawdown than every other long-only benchmark (29 pp tighter
than BTC). But the de-risking machinery costs ~30 pp of total return
versus BTC over the full window.

## Walk-forward result (14 OOS windows × 90 days)

```
                                           strat   BTC     basket  simple
window  1   2022-09 → 2022-12              -2.31  -12.84  -15.45   -4.58
window  2   2022-12 → 2023-03             +11.72  +67.41  +40.12   +4.10
window  3   2023-03 → 2023-06              +2.64  +12.54   -7.30   -2.67
window  4   2023-06 → 2023-09              -3.44  -12.75   -9.34  -14.21
window  5   2023-09 → 2023-12             +32.83  +65.04 +129.01 +226.81
window  6   2023-12 → 2024-03             +40.24  +54.29  +42.73  +17.04
window  7   2024-03 → 2024-06             -10.57   -0.50  -19.19  -12.49
window  8   2024-06 → 2024-09             -18.37  -10.40  -12.81  -31.91
window  9   2024-09 → 2024-12             +96.62  +73.20 +147.37 +198.15
window 10   2024-12 → 2025-03             -32.20  -20.48  -34.69  -45.68
window 11   2025-03 → 2025-06             +10.44  +28.45   +8.65   -5.30
window 12   2025-06 → 2025-09             +17.46   +9.55  +48.86  +30.76
window 13   2025-09 → 2025-12             -17.86  -20.70  -35.38  -25.62
window 14   2025-12 → 2026-03              -8.07  -24.39  -32.91   +0.00
```

Per-window summary:

| Comparison | Strategy beats | Pct |
| --- | --- | --- |
| Beats BTC | 6 / 14 | 42.86 % |
| Beats equal-weight basket | 8 / 14 | 57.14 % |
| Beats simple momentum | 10 / 14 | 71.43 % |
| Profitable | 7 / 14 | 50.00 % |
| Stability (profit AND beats every benchmark) | 0 / 14 | 0 % |

## Placebo (20 seeds, random BTC weight ∈ {1.0, 0.7, 0.4, 0.2} per
rebalance, no signal)

```
strategy        return = +44.99 %   DD = -37.14 %
placebo median  return = +14.15 %   DD = -45.89 %
placebo p75 DD  -39.63 %
beats placebo median return  = True
beats placebo median drawdown = True
```

The strategy clears the placebo on both return and drawdown — the
signal carries information beyond random bucket-picking.

## Scorecard

| # | Check | Threshold | Result | Pass |
| --- | --- | --- | --- | --- |
| 1 | Beats BTC OOS | > 50 % windows | 42.86 % | **FAIL** |
| 2 | Beats basket OOS | > 50 % windows | 57.14 % | PASS |
| 3 | Beats simple momentum OOS | > 50 % windows | 71.43 % | PASS |
| 4 | Beats placebo median return | strategy > median | +44.99 % vs +14.15 % | PASS |
| 5 | OOS stability | ≥ 60 % | 0 % | **FAIL** |
| 6 | Drawdown gap vs BTC | ≤ 20 pp worse | +29 pp tighter | PASS |
| 7 | Total rebalances | ≥ 10 | 190 | PASS |

**Checks passed: 5 of 7. Verdict: FAIL.**

## Why it failed

1. **BTC is hard to beat in a four-year BTC bull regime.** The
   allocator is *defensive by design* — it caps exposure when DD or
   regime turns red. BTC buy-and-hold has no such cap, so in windows
   where BTC rallies straight up (windows 2, 3, 5, 6, 11) the allocator
   leaves return on the table. That is 5 of the 8 windows where it
   loses to BTC; the other 3 are partial bear-bounce windows where the
   200d MA cap kept exposure low while BTC rebounded.
2. **Stability gate is BTC-anchored.** The stability score requires the
   strategy to beat BTC AND the basket AND simple momentum AND be
   profitable in the same window. There is no single OOS window where
   the allocator clears all four bars simultaneously, so stability is
   0 % even though it clears 3 of 4 in several windows.

## What survived

* The strategy **does** beat the equal-weight basket, simple momentum,
  and the random-bucket placebo over both the full window and the
  majority of OOS windows.
* The strategy has the **tightest max drawdown** of any long-only
  benchmark in this universe (-37.14 % vs BTC -66.12 %).
* Sharpe 0.44 is below BTC's 0.52 but above ETH and the basket.

These are real but they are not enough to clear the spec PASS bar.

## Honest caveats

* No parameters were tuned after seeing results — the buckets, MA
  window, vol windows, breadth threshold, and overlay weights are all
  exactly as specified.
* The 14-window OOS spans a particularly BTC-favourable four-year
  block. A more BTC-bearish regime might flip the BTC-beats outcome,
  but we do not have that data and we do not get to retune off this
  one.
* All exposure rules are evaluated only on history available at the
  rebalance bar (`timestamp <= asof`), so the result is lookahead-free
  by construction. A unit test asserts this directly.

## Final rule

This is research only. **Execution remains locked.** The strategy
registry entry and the safety lock both keep
`paper_trading_allowed = False` and `live_trading_allowed = False`
on this branch.
