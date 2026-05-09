# Derivatives signal research — honest verdict

This is the closure document for the **`research/new-signal-class`**
branch experiment that built the futures-funding + open-interest signal
class proposed at the end of the v1 research closure
([`reports/final_crypto_research_report.md`](final_crypto_research_report.md)).

The v1 closure ended with the recommendation to "test new signal classes
— funding rates, on-chain flows, sentiment, real BTC dominance — rather
than tweak more price-based TA on the same universe." This branch
implemented the funding-rate + open-interest part of that proposal end
to end, ran it through the same conservative scorecard that every v1
strategy was held to, and is being reported here without softening the
result.

> **Verdict: INCONCLUSIVE on the joint funding+OI signal class.**
> Mechanically the scorecard reads **FAIL** (1 of 6 checks passed),
> but the FAIL is an artefact of the strategy never trading: the
> signal_state classifier produced `'unknown'` on **100 % of rows**
> because `open_interest_30d_change_pct` is undefined when there are
> fewer than 31 OI rows in history, and the public Binance OI endpoint
> is capped at ≈30 rows total.

## What was built

| Module | Purpose |
| --- | --- |
| [`src/futures_data_collector.py`](../src/futures_data_collector.py) | Public Binance Futures funding + OI collector. No API keys. Same stuck-pagination defence as the v1 Kraken fix. |
| [`src/derivatives_signals.py`](../src/derivatives_signals.py) | Per-symbol + cross-asset signal table. Backward-rolling z-scores, no lookahead. Documented schema: 24 columns, 6 signal states. |
| [`src/strategies/derivatives_rotation.py`](../src/strategies/derivatives_rotation.py) | Long-only portfolio strategy that excludes `crowded_long` and rewards `healthy_trend`. Same `target_weights(asof_ts_ms, asset_frames, timeframe)` contract as the v1 momentum rotation. |
| [`src/derivatives_research.py`](../src/derivatives_research.py) | Walk-forward + multi-seed placebo + scorecard. **Same conservative verdict thresholds as the v1 portfolio scorecard** — no softer bar for the new signal class. |
| `streamlit_app.py` (Derivatives signals card) | Coverage / signals / equity / walk-forward / placebo / scorecard tabs. |
| 7 new CLI commands | `download_futures_data`, `derivatives_signals`, `derivatives_rotation`, `derivatives_walk_forward`, `derivatives_placebo`, `derivatives_scorecard`, `research_all_derivatives`. |

Every module ships with tests. The combined suite is **232 / 232 passing**
(191 v1 + 41 new for the derivatives layer), all green on this branch.

## Data we actually got

| Dataset | Source | Symbols | Coverage |
| --- | --- | --- | --- |
| Spot OHLCV (daily) | Binance public OHLCV (cached from v1) | 10 | **1459 days** |
| Funding rate (8h cadence) | `https://fapi.binance.com/fapi/v1/fundingRate` | 10 | **1459 days · ~4 380 events / symbol** |
| Open interest (1d) | `https://fapi.binance.com/futures/data/openInterestHist` | 10 | **29 days · 30 rows / symbol** |

The OI cap was anticipated before any code was written and is documented
prominently in the collector
([`OI_HISTORY_PUBLIC_DAYS_CAP = 30`](../src/futures_data_collector.py)).
The endpoint returns HTTP 400 when `startTime` reaches into the past
beyond that cap. There is no public Binance alternative that extends the
window. Paid data feeds (Coinglass, Velo, Amberdata, etc.) would close
the gap, but the v1 invariant is "no API keys, no paid feeds". So the
gap is real and structural, not a code bug.

## What the run actually produced

```
=== derivatives rotation vs benchmarks (full window) ===
  derivatives_rotation           return=+0.00%   dd=  0.00%   sharpe= 0.00
  BTC_buy_and_hold               return=+175.84% dd=-50.37%   sharpe= 0.76
  ETH_buy_and_hold               return=+10.51%  dd=-63.75%   sharpe= 0.38
  equal_weight_basket            return=+59.66%  dd=-60.91%   sharpe= 0.50
```

```
=== derivatives walk-forward (14 OOS windows × 90 days) ===
  every window: oos_return_pct = 0.00, n_rebalances ≈ 14, n_trades = 0
  beats_btc:    5 / 14 (35.7 %, only because cash beat negative BTC quarters)
  beats_basket: 6 / 14 (42.9 %, same reason)
  profitable:   0 / 14
```

```
=== derivatives placebo (20 seeds, random Top-3 rotation) ===
  strategy_return_pct                 0.00
  placebo_median_return_pct          22.34
  strategy_max_drawdown_pct           0.00
  placebo_median_drawdown_pct       -68.71
  strategy_beats_median_return       False
  strategy_beats_median_drawdown      True   (only because the strategy never opened a position)
```

```
=== derivatives scorecard ===
  verdict                FAIL  ← mechanical
  checks_passed          1 / 6
  reason                 positive_return=False; beats_btc_oos=False;
                         beats_basket_oos=False; beats_placebo_median=False;
                         oos_stability_above_60=False; at_least_10_rebalances=True
```

The "FAIL" here is mechanical — the conservative scorecard counts a
strategy that never enters any position as "did not beat BTC" in most
windows, because cash returns 0 % while BTC returns positive in 9 of 14
quarters. But this is not a real strategy failure; it is the absence of
any signal output to act on.

## Why every signal classified as `unknown`

`signal_state` is set in
[`src/derivatives_signals.py::_classify_state`](../src/derivatives_signals.py).
The opening guard:

```python
if pd.isna(oi30) or pd.isna(r30) or pd.isna(r7):
    return "unknown"
```

`oi30` is `open_interest_30d_change_pct` — `pct_change(30)` on the OI
series. With ≤ 30 rows of OI per symbol, that quantity is **NaN at
every row**. The guard is therefore satisfied on every row, and every
row falls through to `"unknown"`. The strategy then drops every symbol
with `signal_state == "unknown"`, fails the `min_assets_required ≥ 3`
check, and returns `{}` (= all cash) on every rebalance.

This is the correct, lookahead-free behavior. The fix is not "loosen
the warm-up rule". The fix is more OI history, which only paid feeds
can supply.

## Honest verdict

**Joint funding-rate + open-interest signal class — INCONCLUSIVE on
data length.** The full strategy as designed cannot be evaluated on
public Binance data alone. The mechanical scorecard reads FAIL but the
underlying interpretation is INCONCLUSIVE: the `crowded_long`,
`capitulation`, `healthy_trend`, and `deleveraging` rules **never had a
chance to fire** because each requires `open_interest_30d_change_pct` or
the joint price/OI quadrant flags, all of which depend on more than 30
days of OI history.

This is also a separate, real finding for the project:

* Public Binance OI history is not enough to walk-forward an
  OI-dependent strategy through any reasonable OOS regime.
* Paid OI feeds would change this. Funding-only feeds would not.
* Funding rate alone has 4 years of history and is testable; it was
  **not** evaluated as a standalone signal class on this branch — that
  would be a separate experiment, with its own scorecard run, and the
  scope of this branch was the joint signal class as specified.

## What this does NOT change about v1

* **No strategy has passed.** The v1 closure stands.
* **Do not paper trade.**
* **Do not connect Kraken or any other broker.**
* **BTC buy-and-hold remains the strongest practical baseline.**
* The scorecard's conservative thresholds are unchanged. No threshold
  was lowered to chase a verdict on this branch.

## Credible next directions (unchanged from v1, plus one)

The v1 closure listed funding, on-chain flows, sentiment, and real BTC
dominance as candidate new signal classes. After this branch, the list
should be **refined**, not abandoned:

1. **Funding-rate-only rotation** (≈ 4 years of public history): a
   genuine standalone test, distinct from this experiment. Build it as
   a new strategy class with its own scorecard run. Will it pass? On
   v1 priors, *probably not* — but funding has theoretical edges
   (basis arb, leverage capitulation) that price-only TA does not.
2. **OI signals via a paid feed.** This is what the FAIL in this report
   is actually waiting on. Coinglass / Velo / Amberdata serve multi-year
   OI; with that data the full strategy here can be re-run unchanged.
3. **On-chain flows, sentiment, real BTC dominance** — still untested.
4. **Accept BTC buy-and-hold as the baseline** — also still on the
   table, and it is the only thing on this list that does not require
   more research.

The next session should pick exactly one of (1)–(3) and run it through
the same scorecard. Repeat the rule from v1: PASS only when the
strategy beats BTC, beats the basket, beats the placebo median, has
≥ 60 % OOS stability, and runs at least 10 rebalances over ≥ 5 OOS
windows. **Anything weaker than that is FAIL or INCONCLUSIVE.**
