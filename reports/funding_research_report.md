# Funding-only signal research — honest verdict

This is the closure document for the **funding-only** half of the
`research/new-signal-class` branch. It is the second derivatives-style
experiment proposed at the end of the v1 closure
([`reports/final_crypto_research_report.md`](final_crypto_research_report.md))
and the first one whose data window is long enough to actually evaluate
through the conservative scorecard. The first experiment, the
funding+OI joint signal, was inconclusive on data length
([`reports/derivatives_research_report.md`](derivatives_research_report.md)).
This one is not.

> **Verdict: FAIL.** 4 of 8 PASS checks satisfied. Funding-only rotation
> earned a positive return and beat the random placebo over the full
> 4-year window, but it **lost to BTC, lost to the equal-weight basket,
> and lost to plain momentum rotation in the majority of OOS windows.**
> Beating the placebo alone — which the spec explicitly calls out — is
> not enough.

## Files added on this turn

| File | Role |
| --- | --- |
| [`src/funding_signals.py`](../src/funding_signals.py) | Funding-only feature engineering (21-column documented schema, lookahead-free). |
| [`src/strategies/funding_rotation.py`](../src/strategies/funding_rotation.py) | `FundingRotationStrategy` — long-only, top-3, weekly, excludes `crowded_long`, allows `capitulation` only after stabilization. |
| [`src/funding_research.py`](../src/funding_research.py) | Walk-forward + multi-seed placebo + 8-check scorecard. |
| [`tests/test_funding_signals.py`](../tests/test_funding_signals.py) | 10 unit tests (z-score lookahead, state classification, missing inputs). |
| [`tests/test_funding_rotation.py`](../tests/test_funding_rotation.py) | 7 unit tests (state filter, capitulation gating, cash filter, lookahead). |
| [`tests/test_funding_research.py`](../tests/test_funding_research.py) | 7 unit tests (DD-gap check, "beating placebo only ≠ PASS", every-check requirement). |
| `main.py` (+6 commands) | `funding_signals`, `funding_rotation`, `funding_walk_forward`, `funding_placebo`, `funding_scorecard`, `research_all_funding`. |
| `streamlit_app.py` (Funding signals card) | Coverage / signal table / state-by-asset / equity / walk-forward / placebo / scorecard tabs. |

Combined test suite: **256 / 256 passing** (191 v1 + 41 funding+OI + 24
funding-only) on this branch.

## Funding data coverage

The funding-only research uses ONLY the funding-rate feed; no OI is
read by any module here. The `download_futures_data` CLI from the
previous experiment already populated the cache.

| Symbol | Rows | Coverage | Note |
| --- | --- | --- | --- |
| BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, DOGEUSDT, ADAUSDT, LINKUSDT, AVAXUSDT, LTCUSDT | 4 380 each | **1 459 days** | Clean — every 8h funding event present. |
| SOLUSDT | 4 455 | 1 459 days | Same. |

That is the **full ~4 years** advertised in the spec.

## Funding signal explanation

Daily resampled funding mean is fed into a bank of backward-rolling
features (no `center=True`, no forward fill, every rolling stat resolves
on data with timestamp ≤ t):

* `funding_7d_mean`, `funding_30d_mean` — funding regime measured over
  short and long horizons.
* `funding_90d_zscore` — z-score over a 90-day backward window. NaN
  until 90 valid funding observations exist; the warm-up tests verify
  that no row before the warm-up has a finite z-score.
* `extreme_positive_funding`, `extreme_negative_funding` — booleans at
  the |z| > 1.5 threshold. **Fixed by spec.**
* `funding_trend = funding_7d_mean − funding_30d_mean`. Positive trend
  means funding is rising (bad for longs); negative means it is falling
  (good for longs).
* `funding_normalization` — how close current funding is to its 90d mean,
  in [0, 1] where 1 means within ±1σ.
* `funding_attractiveness` — negative 30d funding mapped to a positive
  longs-friendly score.
* `funding_improvement` — negative trend mapped to a positive score.
* `stabilization_score` — 1 minus normalised |7d return|.
* `crowding_penalty` — positive z mapped to a [0, 1] crowding tax.

State classification is a pure function of those features:

| State | Trigger |
| --- | --- |
| **crowded_long** | z(funding) > +1.5 AND 30d return > 0 |
| **capitulation** | z(funding) < −1.5 AND 30d return < 0 |
| **recovering** | 30d funding mean negative, 7d mean improving above 30d, 7d return within ±5 % |
| **neutral** | nothing extreme fired |
| **unknown** | warm-up rows where the rule's inputs are NaN |

Empirical state distribution over the 14 600 row table (10 symbols ×
~1 460 days each):

```
neutral       11 857   (81.2 %)
recovering       958   ( 6.6 %)
capitulation     777   ( 5.3 %)
crowded_long     708   ( 4.8 %)
unknown          300   ( 2.1 %)
```

Crowded-long fires often enough to actually exclude assets, capitulation
and recovering fire less often, and the warm-up share is small. This is
a far cry from the funding+OI experiment where 100 % of rows were
`unknown`.

## Strategy explanation

`FundingRotationStrategy` is a portfolio rotation in the same family as
v1's `MomentumRotationStrategy` — same `target_weights(asof_ts_ms,
asset_frames, timeframe)` contract, same long-only / no-leverage /
no-margin / no-shorting invariants, weekly rebalance frequency.

At every rebalance bar `t`:

1. **Cash filter.** If BTC/USDT close at `t` is below its 200-day
   trailing MA, return `{}` (all cash). Same filter the v1 momentum
   rotation uses.
2. Drop assets whose latest signal row ≤ t is `unknown` or `crowded_long`.
3. `capitulation` is admissible only when the asset's 7-day return
   is no worse than −5 % (price has stabilised).
4. Score every survivor:

   ```
   score = 0.4 · return_30d
         + 0.3 · funding_attractiveness
         + 0.2 · funding_improvement
         + 0.1 · stabilization_score
         − 0.5 · crowding_penalty
   ```

   These weights are fixed in spec — they were not tuned to chase
   PASS.
5. Top-3 by score, equal-weight. If fewer than 3 admissible assets,
   stay in cash.

## Benchmark result (full 4-year window, single backtest)

```
funding_rotation       return = +37.27 %   max DD = −55.62 %   sharpe = 0.43
BTC_buy_and_hold       return = +175.84 %  max DD = −50.37 %   sharpe = 0.76
ETH_buy_and_hold       return = +10.51 %   max DD = −63.75 %   sharpe = 0.38
equal_weight_basket    return = +59.66 %   max DD = −60.91 %   sharpe = 0.50
simple_momentum        return = +143.92 %  max DD = −58.23 %   sharpe = 0.67
```

Funding rotation underperforms BTC by 138 pp, the basket by 22 pp, and
plain momentum rotation by 107 pp. DD is 5.2 pp deeper than BTC's,
which is well within the 20 pp gap budget the scorecard allows — the
DD ceiling is not where this loses.

## Walk-forward result (14 disjoint OOS windows × 90 days)

```
                                     funding   BTC    basket   simple
window 1   2022-11 → 2023-02          −11.8    13.3    1.6     2.8
window 2   2023-02 → 2023-05           −4.5    28.7    3.4    −5.7
window 3   2023-05 → 2023-08            0.9     1.2   −2.9    −3.5
window 4   2023-08 → 2023-11           24.8    21.7   13.8    30.1
window 5   2023-11 → 2024-01           33.4    22.9   53.3   104.7
window 6   2024-01 → 2024-04            7.5    50.0   32.9    31.3
window 7   2024-04 → 2024-07          −15.2    12.5    5.2    −7.8
window 8   2024-07 → 2024-10          −24.9     0.5   −8.6   −22.4
window 9   2024-10 → 2025-01          142.2    54.2  120.7   177.0
window 10  2025-01 → 2025-04          −44.5   −10.3  −32.4   −49.0
window 11  2025-04 → 2025-07           14.7    25.5   33.7    19.7
window 12  2025-07 → 2025-10           −8.2    −8.5   −6.2    −3.8
window 13  2025-10 → 2026-01          −16.2   −13.9  −24.0   −14.2
window 14  2026-01 → 2026-04            0.0   −16.5  −25.6     0.0

beats_btc:              5/14 (35.7 %)
beats_basket:           5/14 (35.7 %)
beats_simple_momentum:  3/14 (21.4 %)
profitable:             6/14 (42.9 %)
stability score:        2/14 (14.3 %)   profitable AND beats_btc AND beats_basket
```

The strategy is not capturing the bullish quarters cleanly enough. In
window 5 (a +53 % basket quarter) it returned only +33 %; in window 6
(+50 % BTC) it returned only +7 %. In window 9 the funding signals
finally lined up and it produced +142 %, but a single great quarter
does not rescue the rest.

## Placebo result (20 seeds, random Top-3 rotation, same universe)

```
strategy full-window return       +37.27 %
placebo median return             +22.34 %     ← strategy beats placebo
strategy full-window max DD       −55.62 %
placebo median max DD             −68.71 %     ← strategy beats placebo
n_seeds                           20
```

Funding-rotation **does** beat the random Top-3 placebo on both return
and drawdown over the full window. That is real signal — but the spec
warns this is the lowest possible bar.

## Scorecard result (8 checks, FAIL)

| # | Check | Pass? | Value |
| - | --- | --- | --- |
| 1 | positive_return | ✅ | +37.27 % |
| 2 | beats_btc_oos (>50 %) | ❌ | 35.7 % |
| 3 | beats_basket_oos (>50 %) | ❌ | 35.7 % |
| 4 | beats_simple_momentum_oos (>50 %) | ❌ | 21.4 % |
| 5 | beats_placebo_median | ✅ | +37.3 % vs +22.3 % |
| 6 | oos_stability_above_60 | ❌ | 14.3 % |
| 7 | at_least_10_rebalances | ✅ | 190 |
| 8 | dd_within_btc_gap_20pp | ✅ | gap = −5.2 pp |

**checks_passed = 4 / 8 → verdict = FAIL.**

The reason chain captures it cleanly: `positive_return=True;
beats_btc_oos=False; beats_basket_oos=False; beats_simple_momentum_oos=False;
beats_placebo_median=True; oos_stability_above_60=False;
at_least_10_rebalances=True; dd_within_btc_gap_20pp=True`.

## The questions, answered without softening

* **Does it beat BTC?** No. −138 pp full-window, beats BTC in only 5 of
  14 OOS windows.
* **Does it beat the equal-weight basket?** No. −22 pp full-window,
  beats basket in 5 of 14 windows.
* **Does it beat simple momentum rotation?** No, and this one is the
  most damning. −107 pp full-window, beats simple momentum in only 3
  of 14 windows. Funding signals are *worse* than the price-only
  rotation we already had in v1.
* **Does it beat the placebo?** Yes — by ≈15 pp on return and by
  ≈13 pp on drawdown.
* **Does it deserve paper testing?** **No.** The spec is explicit:
  beating placebo alone does not warrant paper testing. The strategy
  has to clear BTC, the basket, and (here) plain momentum. It clears
  none of those.

## Limitations

* The score weights (0.4/0.3/0.2/0.1, crowding penalty 0.5) are fixed
  by spec. They were not tuned. We did not search for a parameter
  combination that would force PASS — the v1 closure was explicit that
  parameter tuning is not permitted on this branch.
* Universe is fixed at 10 large-cap pairs that overlap heavily with
  BTC's beta. The funding signal therefore competes with a very strong
  baseline.
* Funding is a single number per asset per 8h. It is one channel of
  information; expecting it to dominate price-only momentum is
  optimistic in hindsight.
* Walk-forward windows are 90 days OOS / 180 days IS / 90-day step.
  These are inherited from the v1 portfolio scorecard and were not
  changed for this experiment.
* The scorecard's 60 % stability threshold is a *conservative* bar.
  Even loosening it would not rescue this run — the stability score is
  14 % vs the threshold of 60 %.

## What this does NOT change about v1

* **No strategy has passed.** v1 closure stands. Funding-only joins
  the failed list.
* **Do not paper trade.**
* **Do not connect Kraken** or any other broker.
* **Do not add API keys.**
* **BTC buy-and-hold remains the strongest practical baseline.**
* No threshold was lowered to chase a verdict.

## Exact next step

**Pick exactly one of the items below for the next session — not all at
once. Each is a separate experiment with its own scorecard run:**

1. **On-chain flow signals** (active addresses, exchange netflow,
   stablecoin supply changes, miner balances). Public sources exist
   (Glassnode free tier, blockchain.com, mempool.space, Santiment free
   tier). The hypothesis is that on-chain divergences from price lead
   regime changes more cleanly than funding does.
2. **Real BTC dominance vs USDT-quoted dominance.** v1 noted the
   distinction but did not test it. A regime selector that flips on
   true BTC.D crossing its long MA might shift behavior in a way the
   funding signal could not.
3. **Sentiment / Fear & Greed historical**. Cleanly licensed and
   historically downloadable.
4. **Re-run THIS funding strategy with a paid OI feed.** That would
   close the data gap from
   [the funding+OI experiment](derivatives_research_report.md) and
   answer "does the joint signal class beat funding-only?". This is
   the only item that does NOT need a new strategy class — same
   `derivatives_rotation.py` runs unchanged once the OI history is
   long enough.
5. **Accept BTC buy-and-hold as the baseline.** Still on the table, and
   still the only item that does not require more research.

The rule from v1 is unchanged: PASS requires beating BTC, the basket,
the placebo, simple momentum, with > 60 % stability across ≥ 5 OOS
windows AND a drawdown not worse than BTC's by > 20 pp. **Anything
weaker is FAIL or INCONCLUSIVE.**

## Final rule

If the next experiment also fails, say it fails. Do not paper trade.
Do not connect Kraken. Do not lower thresholds. Do not use open
interest unless a proper historical OI source is added — the public
Binance OI cap will not change just because a strategy needs more
data.
