# Funding + basis carry allocator — research verdict

This is the closure document for **research strategy 6 — funding +
basis carry / crowding allocator** on branch
`research/strategy-6-funding-basis-carry`.

## 1. Executive summary

> **Verdict: FAIL.**
> **5 of 9 PASS checks satisfied.** The strategy clears positive
> return, beats the random-bucket placebo on both return and drawdown,
> rebalanced 190 times, has a max drawdown 12 percentage points
> tighter than BTC, and runs on data with full PASS coverage. But it
> loses to BTC in **11 of 14** OOS windows, to the equal-weight basket
> in **9 of 14**, and stability is **7.1 %**. Per the rules:
> parameters were NOT retuned. Execution remains locked.

## 2. Hypothesis

Funding alone failed in the prior `derivatives_funding` branch. The
spec hypothesis was that **funding + futures basis read together**
might better identify crowded longs (high funding *and* stretched
basis), underpriced risk (low funding in healthy trend), and stress
regimes (negative funding *with* weak trend), turning what was a
single weak signal into a positioning filter.

If true, the allocator should beat BTC out-of-sample more than half
the time. It does not.

## 3. Data sources and coverage

Public endpoints only. No API keys. All ten required data streams
clear the **PASS** bar (≥ 1460 days):

| Source | Asset | Dataset | Coverage (days) |
| --- | --- | --- | ---: |
| binance_futures | BTC/USDT | funding_rate_history | 1500 |
| binance_futures | BTC/USDT | mark_price_klines_1d | 1499 |
| binance_futures | BTC/USDT | index_price_klines_1d | 1499 |
| binance_futures | ETH/USDT | funding_rate_history | 1500 |
| binance_futures | ETH/USDT | mark_price_klines_1d | 1499 |
| binance_futures | ETH/USDT | index_price_klines_1d | 1499 |
| bybit | BTC/USDT | funding_rate_history | 1500 |
| bybit | ETH/USDT | funding_rate_history | 1500 |
| deribit | BTC/USDT | perpetual_funding_rate_history | 1500 |
| deribit | ETH/USDT | perpetual_funding_rate_history | 1500 |

OI / liquidations / long-short ratios were **not** used — the prior
audit found public endpoints capped at 30–500 days, well short of
the 1460-day bar.

## 4. Signal construction

Daily-resampled per asset. Every roller is backward-only with
`min_periods = window`. Daily funding aggregates are timestamped at
the **end** of their day so a backward `merge_asof` onto a 00:00 UTC
spot bar cannot peek at later same-day funding ticks. A unit test
asserts that truncating future inputs leaves past signal rows
unchanged.

Features per asset (all derivable from the locked
`FundingBasisSignalConfig` defaults):

* `funding_1d_avg`, `funding_7d_avg`, `funding_30d_avg`.
* `funding_z90` (90d z-score), `funding_pct_rank_365`,
  `funding_trend_pos`.
* `basis = (mark.close - index.close) / index.close`.
* `basis_7d_avg`, `basis_30d_avg`, `basis_z90`.
* `price_return_30d`, `price_return_90d`, `above_200dma`.
* `realised_vol_30d` (annualised log-return std).
* `crowding_score` and `carry_attractiveness` (linear blends of the
  above with locked weights).
* `regime_state` ∈
  {`crowded_long`, `neutral_risk_on`, `cheap_bullish`,
  `stress_negative_funding`, `defensive`, `unknown`}.

Regime mix on the full 2022-04 → 2026-05 window, averaged across
BTC + ETH:

| Regime | % of bars |
| --- | ---: |
| neutral_risk_on | 33.6 % |
| defensive | 29.9 % |
| cheap_bullish | 14.5 % |
| stress_negative_funding | 12.0 % |
| unknown | 6.0 % |
| crowded_long | 4.1 % |

## 5. Strategy rules

Per-asset state drives a per-asset weight cap (locked in
`FundingBasisCarryConfig`):

| State | Weight cap |
| --- | ---: |
| cheap_bullish | 0.80 |
| neutral_risk_on | 0.70 |
| crowded_long | 0.30 |
| stress_negative_funding | 0.00 |
| defensive | 0.00 |
| unknown | 0.00 |

Allocation (verbatim from spec):

* If 2+ tradable (state in {`neutral_risk_on`, `cheap_bullish`}):
  rank by `carry_attractiveness`, top → 0.70, second → 0.30,
  each capped by its per-asset cap.
* If exactly 1 tradable: 0.80, capped by the per-asset cap.
* If 0 tradable but BTC/ETH is `crowded_long`: that asset at 0.30.
* Else: 100 % cash.
* Long-only, Σ weights ≤ 1, weekly rebalance, fees + slippage from
  the existing portfolio backtester, next-bar-open execution.

## 6. Full-window benchmark result (2022-04 → 2026-05)

```
funding_basis_carry_allocator   return = +46.09 %   max DD = -53.72 %   sharpe = 0.44
BTC_buy_and_hold                return = +74.84 %   max DD = -66.12 %   sharpe = 0.52
ETH_buy_and_hold                return = -32.55 %   max DD = -71.74 %   sharpe = 0.21
equal_weight_basket             return = +21.15 %   max DD = -67.34 %   sharpe = 0.36
simple_momentum                 return =  +0.00 %   max DD =   0.00 %   sharpe = 0.00*
random_bucket_placebo (median)  return = -10.78 %   max DD = -63.46 %
```

\* The simple-momentum benchmark requires `min_assets_required = 5`
in its default config and we only run BTC + ETH, so it never trades
and returns 0 % over the whole window. Beats are reported but the
benchmark is degenerate at this universe size — see §10.

## 7. Walk-forward result (14 OOS windows × 90 days)

```
window  oos_window                strat   BTC     ETH     basket
  1     2022-09 → 2022-12          0.00  -12.84   -8.16  -10.50
  2     2022-12 → 2023-03         24.48  +67.41  +46.41  +56.91
  3     2023-03 → 2023-06          6.57  +12.54   +9.31  +10.93
  4     2023-06 → 2023-09        -14.57  -12.75  -16.10  -14.42
  5     2023-09 → 2023-12         34.82  +65.04  +40.51  +52.78
  6     2023-12 → 2024-03         53.61  +54.29  +51.28  +52.79
  7     2024-03 → 2024-06         -6.30   -0.50   -0.27   -0.38
  8     2024-06 → 2024-09        -37.50  -10.40  -35.52  -22.96
  9     2024-09 → 2024-12         38.07  +73.20  +69.06  +71.13
 10     2024-12 → 2025-03        -32.98  -20.48  -51.40  -35.94
 11     2025-03 → 2025-06          1.60  +28.45  +36.68  +32.57
 12     2025-06 → 2025-09         51.26   +9.55  +76.19  +42.87
 13     2025-09 → 2025-12        -26.71  -20.70  -29.46  -25.08
 14     2025-12 → 2026-03          0.00  -24.39  -37.09  -30.74
```

Per-window summary:

| Comparison | Strategy beats | Pct |
| --- | --- | --- |
| Beats BTC | 3 / 14 | 21.4 % |
| Beats ETH | 6 / 14 | 42.9 % |
| Beats equal-weight basket | 5 / 14 | 35.7 % |
| Beats simple momentum | 7 / 14 | 50.0 % |
| Profitable | 7 / 14 | 50.0 % |
| Stability (profit AND beats every benchmark) | 1 / 14 | 7.1 % |

## 8. Placebo result (20 seeds)

The placebo picks per-asset weights uniformly at random from the same
bucket set the strategy uses (0.80 / 0.70 / 0.30 / 0.00) at every
weekly rebalance, normalised when needed. It never reads regime
state, funding, basis, or trend.

```
strategy        return = +46.09 %   DD = -53.72 %
placebo median  return = -10.78 %   DD = -63.46 %
placebo p75 DD            -55.90 %
beats placebo median return     = True (+57 pp)
beats placebo median drawdown   = True (+10 pp)
```

The signal carries information beyond random bucket-picking — the
strategy is statistically distinguishable from noise. That gate
clears.

## 9. Scorecard verdict

| # | Check | Threshold | Result | Pass |
| --- | --- | --- | --- | --- |
| 1 | Positive full-window return | strategy > 0 | +46.09 % | PASS |
| 2 | Beats BTC OOS | > 50 % windows | 21.4 % | **FAIL** |
| 3 | Beats basket OOS | > 50 % windows | 35.7 % | **FAIL** |
| 4 | Beats simple momentum OOS | > 50 % windows | 50.0 % | **FAIL** |
| 5 | Beats placebo median return | strategy > median | +46.09 vs -10.78 | PASS |
| 6 | OOS stability | ≥ 60 % | 7.1 % | **FAIL** |
| 7 | Drawdown gap vs BTC | ≤ 20 pp worse | +12 pp tighter | PASS |
| 8 | Total rebalances | ≥ 10 | 190 | PASS |
| 9 | Data coverage adequate | every required source PASS | ok | PASS |

**Checks passed: 5 of 9. Verdict: FAIL.**

## 10. Why it failed

1. **The defensive caps eat too much of the BTC bull regime.**
   `defensive` (29.9 % of bars) and `stress_negative_funding`
   (12.0 %) together are 42 % of bars where BTC weight is **0**. In
   the 2023 and 2025 BTC rallies, the strategy was disproportionately
   in cash precisely when buy-and-hold compounded. Funding alone said
   "carry is cheap", but the trend filter said "below 200d MA, sit
   out". Window 1 (warm-up) and window 14 (last quarter) ended at
   0 % return for exactly this reason.

2. **`crowded_long` is too rare to do work.** Only 4.1 % of bars.
   The signal class barely matters because it almost never fires.
   Most of the time the allocator is choosing between
   `neutral_risk_on`, `cheap_bullish`, `defensive` — and `defensive`
   is the modal off-state during BTC drawdowns.

3. **Stability gate is BTC-anchored.** Even on windows where the
   strategy was profitable AND beat the basket AND beat simple
   momentum (e.g. window 6, 12), BTC returned more, so the window
   does not count toward stability.

4. **Simple-momentum benchmark is degenerate at 2-asset universe.**
   `MomentumRotationConfig.min_assets_required = 5`, so simple
   momentum never trades on BTC/ETH alone and returns 0 % every
   window. The "beats simple momentum" gate measures "did the
   strategy avoid a loss this window", not "did it beat a real
   momentum baseline". On a richer universe the comparison would be
   meaningful — at this universe size it is essentially a profitable
   /not-profitable gate. Even on that lenient reading the result
   (50 %) does not strictly clear `> 50 %`.

## 11. Limitations

* **2-asset universe.** Spec restricts BTC + ETH only. SOL was
  optional and the audit did not verify SOL funding/basis depth on
  this branch; we did not force a 10-asset universe.
* **No OI / liquidations / long-short / taker data.** Those are FAIL
  on public endpoints (≤ 30–500 days). The strategy is funding +
  basis only, which is the cleanest free positioning signal but
  excludes one half of the original hypothesis.
* **Trend filter is a hard 200d MA.** Below the MA the asset is
  excluded outright (cap = 0). A softer cap might keep more of the
  BTC bull window — but **changing it now would be retuning** and
  is out of bounds.
* **Window boundaries hit warm-up.** The 200d MA needs ~200 days of
  daily history; the first OOS window therefore overlaps the warm-up
  period and shows 0 % return because most rebalances landed in
  `unknown`. This is a real cost of the 4-year data window, not a
  bug.

## 12. Whether it deserves paper testing

**No.** Verdict is FAIL on the locked scorecard. **Execution remains
locked.** The strategy registry entry for this family carries
`paper_trading_allowed = False` and `live_trading_allowed = False`,
and the safety lock independently enforces both regardless of the
scorecard. A scorecard PASS would still require independent review
before any trading — but we do not have a PASS to review.

## 13. Exact next step

1. **Archive this branch as a FAIL.** The signal does carry
   information (beats placebo, tightens drawdown), but it does not
   beat BTC reliably out of sample. Same fate as the
   drawdown-targeted allocator: risk control, not alpha.

2. **Stop building rule-based long-only allocators on BTC + ETH.**
   Three branches in a row (drawdown-targeted, vol-target, funding +
   basis) have produced "tighter drawdown but loses to BTC". The
   pattern is structural — the four-year sample is BTC-favourable,
   so any defensive allocator gives up too much upside.

3. **The next serious direction is one of:**
   * Buy a paid CoinGlass / CryptoQuant developer plan and re-audit
     the OI / liquidation / long-short stack. Funding + basis alone
     is not enough.
   * Introduce **a long-short or relative-value formulation** so the
     strategy can profit from BTC/ETH dispersion rather than always
     taking BTC beta.
   * Accept BTC buy-and-hold as the baseline and use the engine for
     risk reporting + research falsification, not alpha generation.

This is research only. Execution remains locked.
