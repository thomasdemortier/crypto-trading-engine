# Sentiment / Fear & Greed overlay — honest verdict

This is the closure document for branch
**`research/strategy-3-sentiment-fear-greed`**, the third strategy
family tested after v1 (price-based TA), the failed funding/derivatives
branch, and the failed market-structure branch (whose `vol-target`
variant became the prior-best comparison benchmark).

The hypothesis was specific and falsifiable: the prior-best vol-target
allocator failed because it lost to BTC in 79 % of OOS windows;
**Fear & Greed regime turns lead price + on-chain regime turns by
1–4 weeks**, so an overlay that boosts BTC on extreme fear, cuts alts
on extreme greed, and de-risks on deteriorating sentiment should lift
the BTC OOS win-rate over 50 % without sacrificing the placebo gap.

> **Verdict: FAIL — and worse than expected.** **6 of 10 PASS checks
> satisfied.** The sentiment overlay **destroys value** versus the same
> overlay actions applied at random timing: real sentiment full-window
> return was **+25 %**; the random-overlay placebo's median was
> **+55 %**. The overlay also **underperformed the unmodified vol-target
> in 8 of 14 OOS windows** (43 % beat-rate). It still beats basket and
> simple momentum in OOS, and held drawdown 20 pp tighter than BTC, but
> the placebo loss is the kind of result that says *sentiment as
> currently structured is anti-signal here*, not signal.

## Files changed

| File | Role |
| --- | --- |
| [`src/sentiment_data_audit.py`](../src/sentiment_data_audit.py) | (already on branch) audit of free sentiment sources. |
| [`src/sentiment_data_collector.py`](../src/sentiment_data_collector.py) | NEW — alternative.me Fear & Greed downloader. No keys. |
| [`src/sentiment_signals.py`](../src/sentiment_signals.py) | NEW — 17-column signal table (0–100 value, 7d/30d/90d stats, 7-state classifier). Lookahead-free. |
| [`src/strategies/sentiment_market_structure_allocator.py`](../src/strategies/sentiment_market_structure_allocator.py) | NEW — overlay that wraps the prior-best vol-target allocator. |
| [`src/sentiment_research.py`](../src/sentiment_research.py) | NEW — `SentimentRandomOverlayPlacebo`, single-window, walk-forward, placebo, 10-check scorecard, `run_all_sentiment`. |
| `tests/test_sentiment_data_collector.py` | NEW — 11 tests. |
| `tests/test_sentiment_signals.py` | NEW — 11 tests, including a strict partial-vs-full lookahead check. |
| `tests/test_sentiment_market_structure_allocator.py` | NEW — 13 tests covering every (market_structure × sentiment) state combination. |
| `tests/test_sentiment_research.py` | NEW — 7 tests for placebo + scorecard logic. |
| Imported from prior branch | All 7 `market_structure_*` modules + their 65 tests so the prior best vol-target serves as the apples-to-apples benchmark. |
| `main.py` (+8 commands) | `download_sentiment_data`, `sentiment_signals`, `sentiment_allocator`, `sentiment_walk_forward`, `sentiment_placebo`, `sentiment_scorecard`, `research_all_sentiment` (plus the import). |
| `streamlit_app.py` (Sentiment / Fear & Greed card) | NEW 8-tab section: Coverage, F&G history, State over time, Now, Equity, Walk-forward, Placebo, Scorecard. |

**Combined test suite: 317 / 317 passing on this branch.**

## Sentiment data coverage

| Source | Dataset | Rows | Coverage |
| --- | --- | --- | --- |
| alternative.me | Fear & Greed daily | 3 016 | **3 019 days (~8.3 years)**, since 2018-02-01 |

Two ~4-day gaps observed in the 8-year history; both flagged in the
coverage CSV `notes` column. Easily clears the 4-year threshold.

## Sentiment signal explanation

`sentiment_signals.py` produces a daily 17-column table with strict
backward-only rolling stats (`min_periods=window`, no `center=True`):

* `fear_greed_value` (0–100), `fear_greed_classification` (string)
* `fg_7d_change`, `fg_30d_change`
* `fg_7d_mean`, `fg_30d_mean`, `fg_90d_zscore`
* Boolean band flags: `extreme_fear` (≤ 25), `fear`, `neutral`, `greed`,
  `extreme_greed` (≥ 75)
* `sentiment_recovering` — past-14d minimum ≤ 30 AND 7d change > 0 (and
  not currently in an extreme regime)
* `sentiment_deteriorating` — 7d change < 0 AND prior 30d mean ≥ 50
  (sentiment rolling over from a positive regime)
* **`sentiment_state`** ∈ {`extreme_fear`, `fear_recovery`, `neutral`,
  `extreme_greed`, `deteriorating`, `unknown`}.

A `partial_vs_full` regression test recomputes signals on the first
120 days only and verifies every column at every row matches the
full-history equivalent — locking out lookahead.

State distribution over 3 016 daily rows:

```
neutral         932  (30.9 %)
extreme_fear    687  (22.8 %)
fear_recovery   532  (17.6 %)
deteriorating   510  (16.9 %)
extreme_greed   330  (10.9 %)
unknown          25  ( 0.8 %)
```

## Allocator explanation

`SentimentMarketStructureAllocatorStrategy` reads target weights from
the prior-best `MarketStructureVolTargetAllocatorStrategy` (no signal
rules touched, no thresholds tuned), then applies one of the spec's
overlay actions per the latest sentiment row:

| Sentiment state | Overlay action |
| --- | --- |
| `extreme_fear` (and BTC > 200d MA) | + 20 pp BTC, funded from cash |
| `fear_recovery` | base allocation passes through |
| `extreme_greed` | − 20 pp from alts (proportional) → BTC if base had BTC, else cash |
| `deteriorating` | − 20 pp risky exposure (alts first, then BTC) → cash |
| `neutral` / `unknown` / other | base allocation passes through |

Long-only, no leverage, no shorts. Σ weights ≤ 1. Lookahead-free.

## Benchmark result (full ~4-year window)

```
sentiment_market_structure_allocator    return = +24.93 %   max DD = −46.19 %   sharpe = 0.33
BTC_buy_and_hold                        return = +74.84 %   max DD = −66.12 %   sharpe = 0.52
ETH_buy_and_hold                        return = −32.55 %   max DD = −71.74 %   sharpe = 0.21
equal_weight_basket                     return = −15.32 %   max DD = −66.84 %   sharpe = 0.26
simple_momentum                         return = +208.81 %  max DD = −58.23 %   sharpe = 0.76
market_structure_vol_target             return = +53.10 %   max DD = −49.04 %   sharpe = 0.47   ← prior best
```

The overlay **shaves 28 pp off the prior best vol-target's return** in
exchange for ~3 pp tighter drawdown. That is a bad risk-adjusted trade
in absolute terms; a **WORSE** trade than just doing nothing to the
vol-target allocator.

## Walk-forward result (14 disjoint OOS windows × 90 days)

```
                                     overlay   BTC    basket   simple   vol-tgt
window 1   2022-09 → 2022-12             −3.51   −12.84   −15.45    −4.58    −3.51
window 2   2022-12 → 2023-03             34.07    67.41    40.12     4.10    38.33
window 3   2023-03 → 2023-06             −2.46    12.54    −7.30    −2.67     0.48
window 4   2023-06 → 2023-09             −3.30   −12.75    −9.34   −14.21    −3.85
window 5   2023-09 → 2023-12             31.37    65.04   129.01   226.81    37.78
window 6   2023-12 → 2024-03             31.51    54.29    42.73    17.04    32.16
window 7   2024-03 → 2024-06            −10.76    −0.50   −19.19   −12.49   −11.05
window 8   2024-06 → 2024-09            −20.12   −10.40   −12.81   −31.91   −16.71
window 9   2024-09 → 2024-12             51.39    73.20   147.37   198.15    68.65
window 10  2024-12 → 2025-03            −27.88   −20.48   −34.69   −45.68   −30.28
window 11  2025-03 → 2025-06             10.54    28.45     8.65    −5.30     9.91
window 12  2025-06 → 2025-09              2.21     9.55    48.86    30.76     2.04
window 13  2025-09 → 2025-12            −24.44   −20.70   −35.38   −25.62   −25.82
window 14  2025-12 → 2026-03             −7.63   −24.39   −32.91     0.00    −7.63

beats_btc:                3/14 (21.4 %)   ← unchanged from vol-target
beats_basket:             8/14 (57.1 %)   ← unchanged
beats_simple_momentum:   10/14 (71.4 %)   ← +1 vs vol-target's 9/14
beats_vol_target:         6/14 (42.9 %)   ← OVERLAY UNDERPERFORMS BASE in 8 windows
profitable:               6/14 (42.9 %)
stability:                0/14 ( 0.0 %)
```

Per-window the overlay rarely produces a meaningfully different
result than the vol-target alone. When it does, it is **more often
worse** (8 windows) than better (6 windows). The largest negative
deltas come from windows 9 and 5 — alt-rally quarters where the
overlay cut alt exposure (extreme_greed firing) and missed the
upside.

## Placebo result (20 seeds, random sentiment-overlay)

```
strategy full-window return       +24.93 %
placebo median return             +54.91 %    ← STRATEGY LOSES BY 30 pp
strategy full-window max DD       −46.19 %
placebo median max DD             −47.01 %    ← roughly equal
n_seeds                           20
```

This is the most damning data point in the run. The placebo applies
**the same overlay actions** (extreme_fear → +20 pp BTC, extreme_greed
→ −20 pp alts, etc.) at the **same empirical frequency** as the real
sentiment classifier — but at randomly-timed rebalance bars instead of
the actual F&G bars. The placebo's median return is more than 2× the
real strategy's return.

The interpretation is unambiguous: **the actual timing of F&G regime
turns, as encoded in this signal generator, is value-destructive.**
Random firing of the same overlay actions at the same rate produces a
better return.

## Scorecard result (10 checks, FAIL)

| # | Check | Pass? | Value |
| - | --- | --- | --- |
| 1 | positive_return | ✅ | +24.93 % |
| 2 | beats_btc_oos (>50 %) | ❌ | 21.4 % |
| 3 | beats_basket_oos (>50 %) | ✅ | 57.1 % |
| 4 | beats_simple_momentum_oos (>50 %) | ✅ | 71.4 % |
| 5 | beats_market_structure_vol_target_oos (>50 %) | ❌ | 42.9 % |
| 6 | beats_placebo_median | ❌ | +24.9 % vs +54.9 % |
| 7 | oos_stability_above_60 | ❌ | 0.0 % |
| 8 | at_least_10_rebalances | ✅ | 190 |
| 9 | dd_within_btc_gap_20pp | ✅ | gap = +19.9 pp (tighter than BTC) |
| 10 | enough_sentiment_data_coverage | ✅ | ok |

**checks_passed = 6 / 10 → verdict = FAIL.**

This is **strictly worse** than the prior-best vol-target's 8 / 10. The
overlay regressed the project, not advanced it.

## The questions, answered without softening

* **Does it beat BTC?** No. 21 % of OOS windows, identical to the
  vol-target. The overlay didn't move the needle on the BTC bar.
* **Does it beat the equal-weight basket?** Yes — 57 % of windows,
  identical to the vol-target. Same.
* **Does it beat simple momentum?** Yes in OOS — 71 % (+1 vs the
  vol-target's 64 %). Marginal improvement, but full-window total
  return is way below simple momentum (+25 % vs +209 %).
* **Does it beat the market-structure vol-target?** **No.** The overlay
  **underperforms the base allocator in 8 of 14 windows** (43 % beat
  rate). Full-window return drops 28 pp.
* **Does it beat the placebo?** **No, decisively.** Placebo median was
  +55 % vs strategy +25 %. This is the kind of placebo-loss that
  signals the source signal isn't doing what the overlay rules assume
  it is.
* **Does it deserve paper testing?** **No.** Failing the BTC bar,
  failing the prior-best variant, and failing the placebo all at once
  is a triple-block. Paper-trading a strategy whose own random-timing
  variant beats it would be irresponsible.

## Limitations

* **The sentiment signal at face value isn't anticipatory.** F&G
  reaches extreme fear at the *bottom of a drawdown*, not before it.
  The overlay's "+20 pp BTC on extreme fear" rule therefore adds BTC
  exposure exactly when BTC has already fallen — and reduces alt
  exposure on extreme greed when alts often have weeks of run left.
  This is the textbook "sentiment is a coincident, not leading,
  indicator" failure mode, and it explains the placebo loss.
* **Single source.** alternative.me is the only free sentiment feed
  with multi-year daily history. There is no second source to
  cross-corroborate the signal.
* **No parameter tuning was permitted on this branch.** The 20 pp
  overlay sizes are spec-given. Tuning them down to 5–10 pp might
  reduce the magnitude of the underperformance, but cannot turn a
  signal that loses to its own placebo into a passing strategy.

## What this does NOT change about v1 / prior branches

* **No strategy has passed.** Sentiment overlay joins the failed list.
* **Do not paper trade.**
* **Do not connect Kraken** or any other broker.
* **Do not add API keys.**
* **BTC buy-and-hold remains the strongest practical baseline.**
* **The prior best is still the market-structure vol-target allocator**
  (8 / 10 checks passed, FAIL). This branch is a regression.

## Exact next step

The placebo loss is diagnostic. Sentiment-as-overlay does not work in
the form tested. Pick exactly one of:

1. **Invert the overlay direction (contrarian variant)** — extreme
   greed would *boost* BTC, extreme fear would *cut* it. This is the
   simplest test of the "sentiment is coincident" hypothesis. If a
   contrarian variant beats both placebo AND the vol-target, the
   diagnosis is confirmed. If it doesn't, sentiment is genuinely
   noise on this universe and we should retire it.
2. **Continuous DD-targeted BTC weight** (item 2 from the prior
   branch's "next step" list) — replace the discrete state-band
   structure entirely with a continuous function of realised BTC
   drawdown. This generalises the vol-target without adding a new
   signal class; it is the conservative path forward.
3. **Re-run the failed funding+OI strategy with a paid OI feed** (item
   3) — closes the data-length gap that left funding+OI INCONCLUSIVE
   rather than truly failed.
4. **Accept BTC buy-and-hold as the baseline.** The vol-target was
   8 / 10 — closer than any other research result on this project.
   Sentiment did not help. After three failed strategy families, this
   is the most honest position.

The v1 rule stands. PASS requires beating BTC, the basket, simple
momentum, AND the prior-best variant, with > 60 % stability across
≥ 5 OOS windows AND drawdown not worse than BTC's by > 20 pp AND
adequate data coverage. **Anything weaker is FAIL or INCONCLUSIVE.**

## Final rule

If the next experiment also fails, say it fails. Do not paper trade.
Do not connect Kraken. Do not lower thresholds. Do not tune parameters.
Sentiment as a free-tier overlay didn't just fail to help — it made
the prior-best result strictly worse. That is the strongest possible
hypothesis-rejecting outcome.
