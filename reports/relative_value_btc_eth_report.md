# BTC/ETH relative-value allocator — research verdict

This is the closure document for **research strategy 7 — long-only
BTC/ETH relative-value allocator** on branch
`research/strategy-7-relative-value-btc-eth`.

## 1. Executive summary

> **Verdict: FAIL.**
> **4 of 9 PASS checks satisfied.** The strategy clears positive return
> (+33.63 %), beats the random-bucket placebo on both return and
> drawdown, and pulls drawdown to the edge of the BTC-gap rule
> (-45.56 % vs BTC -66.12 %, exactly 20 pp tighter). But it loses to
> BTC in **10 of 14** OOS windows, to the equal-weight basket in
> **10 of 14**, to ETH in **8 of 14**, and stability is **0 %**.
> Per the rules: parameters were NOT retuned. Execution remains
> locked.
>
> **Same structural pattern as strategies 4 and 6.** Switching from
> "BTC-only with caps" to "BTC vs ETH rotation" did not change the
> outcome — defensive long-only on this universe gives up too much
> BTC upside.

## 2. Hypothesis

The earlier branches (`drawdown_targeted_btc`, `funding_basis_carry`,
the market-structure vol-target allocator) all produced "tighter
drawdown but loses to BTC" because they re-shape **BTC exposure**
without adding a way to profit from the **ETH leg of the universe**.

Hypothesis: rotating between BTC, ETH, and cash based on the ETH/BTC
ratio's trend and z-score might extract dispersion alpha — i.e. profit
from ETH's outperformance windows when they exist, and revert to BTC
otherwise.

If true, the allocator should beat BTC out-of-sample more than half
the time. It does not.

## 3. Data coverage

The strategy uses only the project's existing local 1d cache for
**BTC/USDT** and **ETH/USDT** spot OHLCV — both 1500 days deep
(2022-04-01 → 2026-05-09). No external endpoints, no API keys, no
broker integration. Data coverage is therefore **PASS** by definition.

## 4. Signal construction

Daily, lookahead-free. Every roller is `min_periods = window` and
backward-only. A unit test asserts that truncating future inputs
leaves past signal rows unchanged (`test_no_lookahead_truncating_future_doesnt_change_past`).

Features (locked in `RelativeValueSignalConfig` defaults):

* `eth_btc_ratio = ETH.close / BTC.close`.
* `ratio_30d_return`, `ratio_90d_return`, `ratio_ma_200`,
  `ratio_above_ma_200`, `ratio_z90` (90d z-score).
* Per-asset 30d / 90d returns, 200d MA flag, and 30d annualised
  realised vol.
* `relative_momentum_score`, `relative_trend_score` — linear blends
  of the ratio signals; centred so 0 = no edge.
* Six regime states:
  `eth_leadership`, `btc_leadership`, `defensive`,
  `unstable_rotation`, `neutral`, `unknown`.

Regime mix on the full 2022-04 → 2026-05 window:

| Regime | % of bars |
| --- | ---: |
| btc_leadership | 39.3 % |
| defensive | 31.3 % |
| unstable_rotation | 13.0 % |
| unknown | 6.0 % |
| neutral | 5.3 % |
| eth_leadership | 5.0 % |

The first observation: **`eth_leadership` only fires 5 % of the
time** in this 4-year window. ETH has not been a sustained relative
winner versus BTC (ratio still well below 2021 highs at end-of-window),
so the rotation signal almost never points at ETH.

## 5. Strategy rules

Verbatim from spec (locked in `RelativeValueAllocatorConfig`
defaults):

| Regime | Allocation |
| --- | --- |
| `eth_leadership` | 100 % ETH if ETH > 200d MA, else 50 % ETH + 50 % cash |
| `btc_leadership` | 100 % BTC if BTC > 200d MA, else 50 % BTC + 50 % cash |
| `defensive` | 100 % cash |
| `unstable_rotation` | 50 % BTC + 50 % cash if BTC > 200d MA, else cash |
| `neutral` | 50 % BTC + 50 % ETH if both > 200d MA, else 100 % BTC if BTC > 200d MA, else cash |
| `unknown` | 100 % cash |

Long-only, Σ weights ≤ 1, weekly rebalance, fees + slippage from the
existing portfolio backtester, next-bar-open execution. **No
shorting.** The current backtester does not support short legs and
the spec forbids faking it.

## 6. Full-window benchmark result (2022-04 → 2026-05)

```
relative_value_btc_eth_allocator   return = +33.63 %   max DD = -45.56 %   sharpe = 0.38
BTC_buy_and_hold                   return = +74.84 %   max DD = -66.12 %   sharpe = 0.52
ETH_buy_and_hold                   return = -32.55 %   max DD = -71.74 %   sharpe = 0.21
equal_weight_basket                return = +21.15 %   max DD = -67.34 %   sharpe = 0.36
simple_momentum                    return =  +0.00 %   max DD =   0.00 %*
random_bucket_placebo (median)     return = -17.80 %   max DD = -59.44 %
```

\* `MomentumRotationConfig` defaults to `min_assets_required = 5`. On
this 2-asset universe simple momentum never trades and stays in cash.
The "beats_simple_momentum" gate is therefore measuring "did the
strategy avoid a loss this window" — see §10.

## 7. Walk-forward result (14 OOS × 90 days)

```
window  oos_window                strat   BTC     ETH     basket
  1     2022-09 → 2022-12          0.00  -12.84   -8.16  -10.50
  2     2022-12 → 2023-03         24.62  +67.41  +46.41  +56.91
  3     2023-03 → 2023-06          5.48  +12.54   +9.31  +10.93
  4     2023-06 → 2023-09        -15.44  -12.75  -16.10  -14.42
  5     2023-09 → 2023-12         38.99  +65.04  +40.51  +52.78
  6     2023-12 → 2024-03         28.81  +54.29  +51.28  +52.79
  7     2024-03 → 2024-06         -3.91   -0.50   -0.27   -0.38
  8     2024-06 → 2024-09        -30.78  -10.40  -35.52  -22.96
  9     2024-09 → 2024-12         39.14  +73.20  +69.06  +71.13
 10     2024-12 → 2025-03        -24.15  -20.48  -51.40  -35.94
 11     2025-03 → 2025-06          4.85  +28.45  +36.68  +32.57
 12     2025-06 → 2025-09         14.06   +9.55  +76.19  +42.87
 13     2025-09 → 2025-12        -20.01  -20.70  -29.46  -25.08
 14     2025-12 → 2026-03          0.00  -24.39  -37.09  -30.74
```

Per-window summary:

| Comparison | Strategy beats | Pct |
| --- | --- | --- |
| Beats BTC | 4 / 14 | 28.6 % |
| Beats ETH | 6 / 14 | 42.9 % |
| Beats basket | 4 / 14 | 28.6 % |
| Beats simple momentum | 7 / 14 | 50.0 % |
| Profitable | 7 / 14 | 50.0 % |
| Stability (profit AND beats every benchmark) | 0 / 14 | 0 % |

## 8. Placebo result (20 seeds)

The placebo picks one of six long-only weight buckets uniformly at
random each rebalance:
`{BTC=1}, {ETH=1}, {BTC=0.5,ETH=0.5}, {BTC=0.5}, {ETH=0.5}, cash`.
It never reads regime state.

```
strategy        return = +33.63 %   DD = -45.56 %
placebo median  return = -17.80 %   DD = -59.44 %
placebo p75 DD            -57.03 %
beats placebo median return    = True (+51 pp)
beats placebo median drawdown  = True (+14 pp)
```

The signal carries information beyond random bucket-picking — the
strategy is statistically distinguishable from noise. That gate
clears.

## 9. Scorecard verdict

| # | Check | Threshold | Result | Pass |
| --- | --- | --- | --- | --- |
| 1 | Positive full-window return | strategy > 0 | +33.63 % | PASS |
| 2 | Beats BTC OOS | > 50 % windows | 28.6 % | **FAIL** |
| 3 | Beats ETH OOS | > 50 % windows | 42.9 % | **FAIL** |
| 4 | Beats basket OOS | > 50 % windows | 28.6 % | **FAIL** |
| 5 | Beats simple momentum OOS | > 50 % windows | 50.0 % | **FAIL** |
| 6 | Beats placebo median return | strategy > median | +33.63 vs -17.80 | PASS |
| 7 | OOS stability | ≥ 60 % | 0 % | **FAIL** |
| 8 | Drawdown gap vs BTC | ≤ 20 pp worse | exactly +20 pp tighter | PASS |
| 9 | Total rebalances | ≥ 10 | 190 | PASS |

**Checks passed: 4 of 9. Verdict: FAIL.**

## 10. Why it failed

1. **`eth_leadership` is too rare to do work.** Only 5 % of bars.
   Across the four-year sample ETH did not sustain relative
   strength against BTC, so the only state where the strategy can
   "win the dispersion bet" almost never fires. The strategy is
   effectively `btc_leadership` (39 %) plus `defensive` (31 %) plus
   half-exposure `unstable_rotation` (13 %).

2. **`defensive` (31 % of bars) eats the BTC bull regime.** When BTC
   is in drawdown the strategy goes to cash — but BTC is *also*
   the right thing to hold during the recovery, and the strategy
   re-enters slowly via the 200d MA. Windows 2, 5, 6, 9, 11 all show
   BTC ripping while the strategy was either in cash, in 50 % BTC,
   or rotating into ETH at the wrong moment.

3. **The 200d MA filter is BTC-anchored.** Cash → BTC re-entry
   triggers when BTC reclaims its 200d MA, which lags the trough
   significantly. The strategy reliably misses the first ~30 % of
   the recovery rally each cycle.

4. **No short leg means no genuine dispersion alpha.** With only
   long-only legs, "rotating to ETH when ETH is leading" is just
   "swap one beta for a slightly higher-beta beta". The rotation
   has no way to PROFIT from BTC-vs-ETH dispersion — it can only
   pick which beta to ride. A market-neutral pair (long ETH/short
   BTC or vice versa) could potentially do that, but the existing
   portfolio backtester does not support short legs and the spec
   prohibits faking it.

5. **Stability gate is BTC-anchored.** Every stability check requires
   beating BTC AND ETH AND the basket AND being profitable. There
   is exactly **zero** windows where the strategy clears all four,
   so stability is 0 %.

6. **Simple-momentum benchmark is degenerate at 2-asset universe.**
   Same issue documented in the funding+basis report — the gate
   measures "avoided a loss" rather than "outperformed real
   momentum". Even on that lenient reading the strategy hit exactly
   50 %, which does not strictly clear the `> 50 %` gate.

## 11. Limitations

* **Long-only constraint.** A real relative-value strategy needs the
  ability to express dispersion as a market-neutral spread. The
  existing portfolio backtester does not support short legs and we
  did not add it (per spec).
* **2-asset universe.** Spec restricts BTC + ETH only.
* **`eth_leadership` rarity is sample-specific.** A future window
  with sustained ETH outperformance might shift the regime mix,
  but we cannot measure that off the available data — and we cannot
  retune for it.
* **Window 1 and window 14 land in 0 % return.** Window 1 because
  the warm-up period coincides with the first OOS bars (200d MA
  needs 200 days of history starting 2022-04-01); window 14 because
  the strategy was in `defensive` cash for almost the entire window
  (BTC and ETH both below their 200d MAs).
* **The strategy's `unstable_rotation` state at 13 % of bars does
  fire sensibly** — it caps exposure during high-vol or extreme-z
  windows. But that's a defensive feature, not a source of alpha.

## 12. Whether it deserves paper testing

**No.** Verdict is FAIL on the locked scorecard. **Execution remains
locked.** The strategy registry entry carries
`paper_trading_allowed = False` and `live_trading_allowed = False`,
and the safety lock independently enforces both regardless of the
scorecard. A scorecard PASS would still require independent review
before any trading — but we do not have a PASS to review.

## 13. Exact next step

Three branches in a row (drawdown-targeted, funding+basis, relative
value) have produced the same verdict: tighter drawdown, beats placebo,
loses to BTC. The pattern is now structural to the long-only
single-asset-or-rotation formulation on a 4-year BTC bull regime.

**Stop building long-only allocators on BTC + ETH.** The next serious
direction must change at least one of the constraints:

1. **Add a true market-neutral pair leg.** Build a separate
   `pair_backtester` module that supports long-A / short-B with
   borrow cost + funding cost — then re-attempt the BTC/ETH spread
   strategy as a market-neutral z-score reversion. This requires a
   real shorting framework (not a hack). It also crosses out of
   "long-only research only" territory and needs explicit spec
   approval before building.

2. **Buy paid positioning data.** A CoinGlass / CryptoQuant
   developer plan unlocks historical OI, liquidations, and
   long-short ratios — the inputs that the strategy-5 audit found
   are FAIL on public endpoints. With those inputs a long-only
   allocator might still clear the bar by timing BTC entries
   differently from "200d MA crosses".

3. **Accept BTC buy-and-hold as the baseline.** Use the engine for
   risk reporting, OOS falsification, and registry hygiene — not
   for alpha generation. Stop opening new long-only branches
   against the same universe and same window.

This is research only. Execution remains locked.
