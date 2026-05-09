# crypto_trading_engine — Final Research Report (v1)

> **Closure document. No paper trading. No live trading. No Kraken execution. No API keys.**
>
> This report summarises every strategy tested, every result obtained, and the next research directions that remain credible.
> It is brutally honest. It is not a sales document.

---

## 1. Executive summary

After building a credible research engine and testing **ten** rule-based strategies on a properly-paginated 1,459-day BTC + ETH (+ 8 alts) dataset:

- **0 strategies reached PASS on the conservative scorecard.**
- **0 strategies deserve paper trading.**
- The strongest practical baseline remains **buy-and-hold BTC**, which the research process never produced an honest improvement on.
- The simple portfolio momentum strategy beat a random placebo by a clear margin, but it did not beat BTC or the equal-weight basket consistently — beating placebo is not evidence of a tradable edge against real benchmarks.
- The regime-aware variant *lost ~46%* of starting capital on the full window while BTC buy-and-hold returned *+176%* — adding regime gating destroyed value rather than creating it.

**Hard rules going forward:**

- **Do not paper trade any strategy in this repo.**
- **Do not connect Kraken (or any other broker) for execution.**
- **Do not lower the scorecard thresholds to manufacture a PASS.**
- The research engine itself is sound and reusable. The signal class tested in v1 is exhausted; future work needs a genuinely different signal source (see §11).

---

## 2. Project scope

This is a research-only project. The following are **prohibited by design** and enforced by the code at multiple levels:

| Constraint | How it is enforced |
|---|---|
| No live trading | `LIVE_TRADING_ENABLED = False` in `src/config.py`; every privileged function calls `utils.assert_paper_only()` which refuses to proceed when the flag is True. There is no execution module to flip on. |
| No leverage / margin | Risk engine caps each position at `max_position_pct` of equity (5%). Cash never goes negative — verified by `tests/test_portfolio_audit.py::test_audit_rebalance_logic`. |
| No short selling | Every strategy interface only emits `BUY` / `SELL` / `HOLD` / `SKIP`. No `SHORT` action exists. |
| No broker execution | No order-placement code path. Forbidden tokens (`create_order`, `apiKey=`, `secret=`, etc.) are scanned for and asserted absent in `tests/test_strategies.py::test_no_strategy_imports_live_trading_or_keys` and similar tests across the codebase. |
| No API keys | The data collector only calls the public OHLCV endpoint (`fetch_ohlcv`). No exchange constructor receives `apiKey` / `secret`. |
| No automated paper-trading loops | The `paper_trader` module exists for one-off ticks only and is never called on a schedule. |

---

## 3. Research engine built

The engine was built incrementally over the project. It is the *deliverable*, regardless of strategy outcomes.

| Component | Module(s) | Purpose |
|---|---|---|
| Data collector | `src/data_collector.py` | Public OHLCV download via `ccxt`. Binance primary (proper backwards pagination); Kraken fallback (capped at most-recent ~720 bars — detected and abandoned). Supports `--lookback-days` and merges sources when one is short. |
| Single-asset backtester | `src/backtester.py` | Per-bar per-asset strategy loop with next-bar-open fills, fees, slippage, stop-loss. Window slicing for walk-forward. |
| Portfolio backtester | `src/portfolio_backtester.py` | Multi-asset rebalancing simulator. Long-only, no leverage. Fees + slippage on every fill. Weekly or monthly rebalance. |
| Risk engine | `src/risk_engine.py` | Position cap, daily loss circuit breaker, fees / slippage, stop-loss. Sole authority for trade approval. |
| Regime detection | `src/regime.py` | Bull / bear / sideways + low / high volatility per bar. Lookahead-free. |
| Cross-asset signals | `src/crypto_regime_signals.py` | ETH/BTC ratio, alt basket vs BTC, breadth (% above 100d/200d MA), relative strength. Four-state risk regime label per bar. |
| Walk-forward | `src/research.py` (`walk_forward`, `walk_forward_by_strategy`); `src/portfolio_research.py` | Disjoint IS / OOS windows. Strategy-specific rows (no proxy reuse). |
| Robustness | `src/research.py` (`robustness`, `robustness_by_strategy`) | Per-family parameter sweeps. Fragility detection only — never used to crown a winner. |
| Placebo | `src/strategies/placebo_random.py`, `src/strategies/regime_aware_momentum_rotation.py` (`RegimeAwareRandomPlacebo`), `src/portfolio_research.py` (`portfolio_placebo`, `regime_aware_portfolio_placebo`) | Fixed-seed random control. 20-seed multi-run aggregate for portfolio strategies. |
| OOS audit | `src/oos_audit.py`, `src/portfolio_audit.py` | Inspects saved walk-forward / weights / equity / trades CSVs for window mechanics, gap detection, leverage prevention, cash never negative. |
| Scorecard | `src/scorecard.py`, `src/portfolio_research.py` (`portfolio_scorecard`, `regime_aware_portfolio_scorecard`) | PASS / WATCHLIST / FAIL / INCONCLUSIVE. PLACEBO and BENCHMARK verdicts excluded from tradable PASS counts by construction. |
| Streamlit Research Lab | `streamlit_app.py` | 11-tab dark dashboard surfacing every result. Includes resumable stage runner with run-state tracking. |
| Stage runner | `src/research.py` (`run_stages`) | Resumable, named stages; flushes the FINAL verdict files at the start of any partial run so an interrupted pipeline cannot leave a stale verdict in place. |

Every CSV output is gitignored. Every public function calls `assert_paper_only()` before doing anything stateful.

---

## 4. Data coverage

**The single most important methodological fix in v1.**

### The original problem

The earliest research runs used Kraken as the primary data source. Kraken's public OHLC endpoint returns at most ~720 of the most-recent candles regardless of the `since` parameter. After dedup, this gave the project:

- **30 days** of 1h data (721 candles)
- **120 days** of 4h data (721 candles)
- **720 days** of 1d data (721 candles)

This was uncovered by the OOS audit — at 90/30/30 walk-forward parameters, the 1h timeframe produced **zero** valid OOS windows; 4h produced only one. Strategies were being declared PASS or FAIL on samples too thin to support any conclusion.

### The fix

`src/data_collector.py` was rewritten to:

1. Default **Binance** as the primary source (Binance supports proper `startTime` paging at 1000 candles per call).
2. Add **stuck-pagination detection** — if the first timestamp of a new batch is not strictly greater than the previous batch's first timestamp, the loop bails out so the caller can fall back to a different exchange.
3. Add an explicit `--lookback-days` CLI flag.
4. **Merge** sources when the primary returns less than 50% of the requested span.
5. Validate gaps post-download (`validate_gaps`) and surface them in `data_coverage.csv`.

### Coverage after the fix

```
asset      timeframe   candles   coverage_days   gaps   enough_for_WF
BTC/USDT   1h          35,039    1459.96         1*     True
BTC/USDT   4h           8,760    1459.83         0      True
BTC/USDT   1d           1,460    1459.00         0      True
ETH/USDT   1h          35,039    1459.96         1*     True
ETH/USDT   4h           8,760    1459.83         0      True
ETH/USDT   1d           1,460    1459.00         0      True
[+ 8 expanded-universe assets at 1d, all 1,460 candles each]
```
*One trivial 2-bar gap on 1h, almost certainly a Binance maintenance window. Documented but not material.*

### Binance vs Kraken — known limitation

- **Binance** is the source of truth for historical research in this repo.
- **Kraken** is reserved for *execution* (no execution code is in v1 — the constraint exists for any future work). Its public OHLC limits make it unfit for walk-forward sample sizes.
- Per-symbol price differences between exchanges are real (Kraken often quotes USD instead of USDT and bridges via `KRAKEN_USDT_TO_USD_FALLBACK`). When `download_symbol` merges fallback rows into a primary-short result, small price jumps where the two feeds meet are possible. This is why single-source downloads are preferred and why `data_coverage.csv` records actual provenance.

### Why this matters

The earliest "promising" results — small samples on Kraken-truncated data — partly **vanished** when the same strategies were re-tested on the full Binance history. For example, simple momentum's "3/6 cells beat B&H" on the truncated dataset became "0/54 cells beat B&H" on the full one. That is the data fix doing exactly its job: separating real edge from sampling noise. **Any verdict in this report is conditional on the post-fix dataset.**

---

## 5. Single-asset strategy results

Seven single-asset families were tested through the per-asset risk engine, identical fees + slippage + next-bar-open execution, on BTC + ETH × 1h + 4h + 1d:

| family | what was tested |
|---|---|
| `rsi_ma_atr` | RSI mean reversion in confirmed uptrends with ATR volatility filter |
| `ma_cross` | 50/200 moving-average crossover |
| `breakout` | Donchian-style breakout (lookahead-free shifted rolling high/low), with optional volume + trend + ATR filters |
| `trend_following` | RSI-recovery entries inside confirmed uptrends |
| `pullback_continuation` | Pullbacks to MA20/MA50 with RSI band, green-candle confirmation |
| `sideways_mean_reversion` | Bollinger-band lower entry only when regime detector labels the bar `sideways` |
| `regime_filtered` | Wrapper that blocks BUYs in `bear_trend` |
| `regime_selector` | Wrapper that routes to trend-following in bull/low-vol, sideways MR in sideways/low-vol, cash otherwise |

### Results across the post-fix dataset

- **0 / 54** tradable cells (strategy × asset × timeframe) beat buy-and-hold.
- **0 / 42** beat the random placebo on both stability and mean OOS return.
- **0 PASS, 0 WATCHLIST**, 40 FAIL, 14 INCONCLUSIVE.
- The "best" tradable cell underperforms B&H by **−13.6%**.
- Walk-forward stability (% of OOS windows that are profitable AND beat B&H) — averaged across strategies — is in the **single digits** for every cell with enough trades to score.

### Why they failed

1. **OOS instability is the dominant blocker.** Even when a strategy produced a positive total-window return, that return was concentrated in 1-2 fortunate windows; the rest were neutral or negative. The conservative scorecard requires ≥70% of windows to be both profitable and beat B&H — none came close.
2. **Trade count too low** on most cells (mean 5.5 round-trips per (asset, timeframe) at the original 4h-only cut). Even with proper data (1d) the trade count remained <3 per OOS window in many cases — fundamentally insufficient to distinguish signal from noise.
3. **No improvement vs placebo.** Random entries with the same risk engine produced statistically indistinguishable stability scores.
4. **Regime selector did not help.** Routing among the sub-strategies tied the best always-on tradable on its single WATCHLIST cell and was worse everywhere else; OOS stability did not improve.

---

## 6. Portfolio momentum results

`src/strategies/momentum_rotation.py` + `src/portfolio_backtester.py`. Long-only equal-weight Top-N rotation across the **expanded 10-asset crypto universe** (BTC, ETH, SOL, AVAX, LINK, XRP, DOGE, ADA, LTC, BNB), 1d candles, 1459 days, weekly rebalance, BTC 200d-MA cash filter.

### Single-window result

| | strategy | BTC B&H | ETH B&H | equal-weight basket |
|---|---|---|---|---|
| Total return | **+143.9%** | +175.8% | +10.5% | +59.7% |
| Max drawdown | -58.2% | -50.4% | -63.7% | -60.9% |
| Sharpe | 0.67 | 0.76 | 0.38 | 0.50 |

### Walk-forward (14 OOS × 90 days)

```
Avg OOS return:                +18.5 %
% windows profitable:           42.9 %
% windows beat BTC:             35.7 %
% windows beat basket:          50.0 %
Stability (profitable AND beats BOTH):  21.4 %
Total rebalances:               190
```

### Placebo (20 seeds, regime-naive random Top-N)

```
Strategy total return:    +143.9 %
Placebo median return:    -12.8 %
Beats placebo median:     YES
```

### Scorecard verdict: **FAIL** (3 / 6 checks satisfied)

```
positive_return         = True   ✓
beats_btc_oos           = False  ✗  (35.7%)
beats_basket_oos        = False  ✗  (50.0% — at threshold, not above)
beats_placebo_median    = True   ✓
oos_stability_above_60  = False  ✗  (21.4%)
at_least_10_rebalances  = True   ✓  (190)
```

### Why beating placebo was not enough

The strategy beat random selection by ~157pp in total return, which is a real signal. But:

- It only beat BTC in 36% of OOS windows.
- It only beat the equal-weight basket in 50% of windows (literal coin-flip).
- Stability is 21% — well below the 60% PASS bar.

In short: the rotation rules contain *some* information vs random, but not enough to beat the much simpler decision of "just hold BTC" or "just hold the basket." The placebo bar is informative but it is **not** the bar a tradable strategy must clear.

The full mechanical audit (`src/portfolio_audit.py`) confirmed the cash filter fired correctly on 115/115 of bearish-BTC rebalance dates, benchmarks were aligned to identical OOS timestamps, and rebalance logic respected the no-leverage / cash-never-negative invariants. The FAIL is a real strategy failure, not a measurement bug.

---

## 7. Regime-aware portfolio results

`src/strategies/regime_aware_momentum_rotation.py`. Routes the rebalance based on the cross-asset risk state from `src/crypto_regime_signals.py`:

| risk_state | route |
|---|---|
| `alt_risk_on` | top 3 by composite momentum + relative-strength score |
| `btc_leadership` | 100% BTC |
| `defensive` | 100% cash |
| `mixed` / `unknown` | 100% BTC (conservative default) |

### Risk-state distribution (1459 days)

```
defensive       41.4 %
btc_leadership  29.5 %
mixed           23.3 %
alt_risk_on      5.8 %
```

### Result: dramatic FAIL

| | regime-aware | simple momentum | BTC B&H | basket |
|---|---|---|---|---|
| Total return | **−46.4%** | +143.9% | +175.8% | +59.7% |
| Max drawdown | −61.0% | −58.2% | −50.4% | −60.9% |
| Sharpe | −0.14 | 0.67 | 0.76 | 0.50 |

### Walk-forward (14 OOS × 90 days)

```
Avg OOS return:                  +0.10 %  (essentially flat)
% windows beat BTC:               21.4 %
% windows beat basket:            42.9 %
% windows beat simple momentum:   50.0 %  (coin flip)
Stability:                         0.0 %  (no window beat BOTH benchmarks)
```

### Placebo

```
Strategy:          -46.4 %
Placebo median:    -49.5 %  (placebo also lost half)
Beats placebo:     YES (by 3pp — meaningless when both lose)
```

### Why the regime gating destroyed value

The `defensive` regime fires on **41% of bars** (any time BTC < 200d MA OR breadth < 40%). Combined with `btc_leadership` (29.5%, 100% BTC) and `mixed` (23.3%, 100% BTC), the strategy is "BTC-with-cash" 95% of the time. The 6% of bars in `alt_risk_on` are too rare to drive meaningful alt rotation.

Worse: when BTC is below its 200d MA but recovering (the typical bottom of a bear-to-bull transition), the strategy is in cash precisely while alts begin to move. WF windows 4, 5, 6, 9 in the saved CSV all show BTC at +20% to +50% while the regime-aware strategy was flat or down.

### Why Phase B (allocation optimization) was not justified

The user's own gate read: *"Do not move to Phase B unless the result is at least better than simple momentum and placebo."*

The regime-aware strategy is **worse than simple momentum** on every metric except the total-window placebo comparison (where the placebo also lost half its capital — not a meaningful win). Allocation tweaks (equal-weight vs inverse-volatility vs top-2) cannot rescue a signal that itself does not beat real benchmarks. Phase B is **deferred indefinitely**.

---

## 8. What was learned

1. **Simple TA does not beat buy-and-hold here.** Long-only RSI / MA / Bollinger / breakout rules — alone or in combination — do not produce a tradable edge over BTC spot at 1h, 4h, or 1d after fees and slippage on this 4-year dataset.
2. **Regime filters can reduce activity but do not create edge.** Both the `regime_filtered` wrapper (single-asset bear-trend block) and the cross-asset `regime_selector` reduced trade count without improving any OOS metric. Reducing risk by sitting in cash is not the same as identifying when to trade.
3. **Momentum rotation has *some* signal vs random but not enough vs real benchmarks.** Beating a random Top-N selection by 157pp in total return is real, but BTC buy-and-hold beat the rotation strategy in the same window by 32pp.
4. **Strict benchmarks are necessary.** Without BTC and the equal-weight basket as comparators, the strategy's nominal +144% looks attractive. With them, it loses on the comparison that matters.
5. **Placebo testing is essential.** Beating a placebo distinguishes "any signal at all" from "pure noise." Failing to beat real benchmarks distinguishes "signal exists" from "signal is tradable." The two tests are complementary; neither alone is sufficient.
6. **OOS stability is the single most informative metric.** A strategy can produce a large total-window return on 1-2 lucky windows. Stability — the fraction of windows that are *both* profitable *and* beat benchmark — collapsed every otherwise-attractive total-window result tested in this project.
7. **Execution integration is premature.** Connecting Kraken (or any broker) to a strategy that has not passed the scorecard would only stress-test execution mechanics. The audit run on the portfolio backtester already confirmed those mechanics are clean. There is nothing further to learn from paper-trading a FAIL strategy.

---

## 9. Current final verdict

| Strategy family | Verdict | Main reason |
|---|---|---|
| Single-asset RSI / MA / ATR | **FAIL** | 0/54 cells beat B&H, 0/42 beat placebo, OOS stability in single digits |
| Single-asset MA crossover | **FAIL** | Same, with even fewer trades |
| Single-asset breakout (filtered) | **FAIL** | Trend + volume + ATR filters reduce false signals but never produce a positive OOS edge |
| Single-asset trend following | **FAIL** | RSI-recovery entries plus MA50 exits behave like a high-friction long-only buy-and-hold |
| Single-asset pullback continuation | **FAIL** | Pullback-to-MA + RSI band entries did not survive walk-forward |
| Single-asset sideways mean reversion | **FAIL** | Best on ETH 1h on the truncated data (WATCHLIST score 4) but vanished after the data fix; still no PASS |
| Regime-filtered variants (`+regime`) | **FAIL** | Reduced trade count without improving stability; no PASS on any cell |
| Regime selector (cross-strategy router) | **FAIL** | Tied the best always-on tradable on its single WATCHLIST cell; never improved OOS |
| Portfolio momentum rotation (simple) | **FAIL** | Beat placebo (+157pp) but beat BTC in only 36% of OOS windows; stability 21% |
| Regime-aware portfolio momentum | **FAIL (dramatically)** | Lost 46% of capital while BTC +176%; stability 0%; defensive gating fires 41% of bars |
| **BTC buy-and-hold** | **BENCHMARK** (not a tradable candidate) | Strongest practical baseline; the research process never produced an honest improvement on it |

---

## 10. Paper-trading decision

- **No paper trading.**
- **No Kraken execution.**
- **No API keys.**
- **No live deployment.**
- **No automated paper-trading loops.**

Paper-trading any of the strategies in this repo would test execution mechanics that the audit module has already confirmed work correctly. It would not produce evidence of an edge. The project's own conservative rule applies in full force: "Do not paper-trade unless a tradable strategy beats buy-and-hold AND beats placebo AND passes OOS stability." Zero strategies meet this bar.

---

## 11. Credible next directions

The signal class tested in v1 — price-based long-only TA on a fixed crypto universe — is **exhausted**. Two more iterations on this class would not produce different results. The credible next directions are *different signals*, not different rules on the same signal:

1. **Real BTC dominance data.** Currently approximated as `1 - (alt_basket_value / (btc_value + alt_basket_value))`. CoinGecko or CoinMarketCap publish actual dominance series. A real dominance time-series would let you test the "altcoin season" hypothesis directly rather than via a noisy proxy.
2. **Total crypto market cap / altcoin market cap.** Same providers. Allows trend-on-the-sector rather than trend-on-individual-assets.
3. **Volume expansion + liquidity signals.** OBV-style cumulative volume, volume-weighted momentum, change in 7d/30d average dollar volume. Some research suggests volume regime changes lead price regime changes at 4h-1d horizons.
4. **Funding rates** (perp futures). Available from public exchange APIs without keys. Crowded-long detection: strategy flips defensive when funding stays elevated.
5. **Open interest.** Same source. Combined with funding, lets you detect aggressive long-buildup that historically precedes squeezes.
6. **On-chain exchange flows.** Net deposits/withdrawals from major exchange addresses (Glassnode, CryptoQuant, Nansen). Free tiers exist; some research suggests this leads price.
7. **Sentiment / news signals.** Fear-and-greed index (free from alternative.me), or NLP scores over crypto news headlines.
8. **Long/short or pair trading on existing data** — only if/when the project's "no shorts" constraint is intentionally relaxed. ETH/BTC ratio is the obvious first pair. This is the single biggest *available-from-this-data* direction; it is currently disabled by the safety rules of v1.
9. **Accept BTC buy-and-hold as the baseline** and stop trying to beat it. This is also a valid research conclusion. The project's deliverable becomes the honest scorecard pipeline that documents *which* rules don't add edge — which is itself useful.

Each of (1)-(8) requires a *new data source*. The research engine itself is reusable: data ingestion → indicator computation → backtester (single or portfolio) → walk-forward → placebo → scorecard. Adding a new signal means adding a new column to the per-bar feature set, not redesigning the pipeline.

---

## 12. What not to do next

These actions would represent motion, not progress, and should be explicitly avoided:

- **Do not tweak the failed strategies' parameters.** The robustness sweep already tested 100+ parameter variants per family — none produced PASS. Manual tuning beyond the sweep would either produce in-sample fits that fail OOS, or it would be redoing work already done.
- **Do not lower scorecard thresholds.** The 60% stability bar and the requirement to beat BOTH benchmark AND placebo are conservative on purpose. Loosening them would produce a paper PASS without the underlying signal having improved.
- **Do not add allocation optimization on a failed signal.** Inverse-volatility weighting, top-2 instead of top-3, etc., cannot rescue a signal that doesn't beat benchmarks. Phase B was correctly deferred.
- **Do not paper-trade.** The execution mechanics audit already passes. Paper-trading would only add operational risk (manual monitoring, stale state) without producing edge evidence.
- **Do not connect Kraken** or any other broker. There is no edge to execute against.
- **Do not mistake "beats placebo" for "beats benchmark."** The portfolio momentum strategy did the former and is still FAIL. This distinction must be preserved in any future research.

---

## 13. Recommended next action

**Freeze the current research engine as v1.** Concretely:

1. Tag or commit the current state in git as `v1-research-closure`.
2. Keep the Streamlit Research Lab as a standing reference dashboard. The 178-test suite documents its expected behaviour.
3. Open a new research branch *only* if introducing a genuinely new signal source from §11 (funding rates, on-chain flows, sentiment, real BTC dominance, etc.).
4. If that new branch produces results that pass the existing scorecard, *then* — and only then — discuss whether paper-trading is justified.
5. If no new signal is introduced, **accept BTC buy-and-hold as the supported conclusion of the project** and treat the engine as a tool for falsifying future strategy ideas rather than discovering them.

The honest framing: this project successfully built the research apparatus required to *not* deploy a bad strategy. That is a valuable outcome. The absence of a PASS verdict is information.

---

## 14. Appendix

### A. Key CLI commands used

```bash
# Data
python main.py download --timeframes 1h 4h 1d --lookback-days 1460
python main.py download --assets BTC/USDT ETH/USDT SOL/USDT AVAX/USDT \
    LINK/USDT XRP/USDT DOGE/USDT ADA/USDT LTC/USDT BNB/USDT \
    --timeframes 1d --lookback-days 1460

# Single-asset research
python main.py research_all --timeframes 1h 4h 1d
python main.py audit_oos
python main.py placebo_audit
python main.py scorecard

# Portfolio research
python main.py portfolio_momentum
python main.py portfolio_walk_forward
python main.py portfolio_placebo
python main.py portfolio_scorecard
python main.py audit_portfolio

# Cross-asset regime + regime-aware portfolio
python main.py crypto_regime_signals
python main.py regime_aware_portfolio
python main.py regime_aware_portfolio_walk_forward
python main.py regime_aware_portfolio_placebo
python main.py regime_aware_portfolio_scorecard

# Aggregate
python main.py research_all_portfolio
python main.py research_all --skip-robustness  # quick iteration
```

### B. Key output files (all gitignored)

```
results/
├── data_coverage.csv
├── crypto_regime_signals.csv
├── strategy_comparison.csv
├── walk_forward_by_strategy.csv
├── robustness_by_strategy.csv
├── strategy_scorecard.csv
├── oos_audit.csv
├── oos_audit_summary.csv
├── placebo_comparison.csv
├── portfolio_universe_availability.csv
├── portfolio_momentum_equity.csv
├── portfolio_momentum_trades.csv
├── portfolio_momentum_weights.csv
├── portfolio_walk_forward.csv
├── portfolio_placebo_comparison.csv
├── portfolio_scorecard.csv
├── portfolio_cash_filter_audit.csv
├── portfolio_benchmark_alignment_audit.csv
├── portfolio_rebalance_audit.csv
├── regime_aware_portfolio_comparison.csv
├── regime_aware_portfolio_equity.csv
├── regime_aware_portfolio_walk_forward.csv
├── regime_aware_portfolio_placebo.csv
├── regime_aware_portfolio_scorecard.csv
├── research_run_state.json
└── research_summary.csv
```

### C. Test count

**178 tests** at the time of this report. All pass.

### D. Important limitations

- **No survivorship-bias correction.** All 10 universe assets are still actively traded; no test against a delisted coin. Inflates absolute returns for both strategy and placebo (the relative comparison is mostly fine).
- **Universe is fixed.** The portfolio strategies cannot rotate into a coin that didn't exist at backtest start.
- **High-volatility regime never fires** in the cached window (`atr_pct > 5%` threshold), so the high-vol branch of the regime selector / regime-aware strategy was untested in practice.
- **Spot only.** No futures, no funding-rate data, no perpetual mechanics modelled.
- **Daily-only for portfolio strategies.** Momentum windows of 30/90 days and weekly rebalancing are only meaningful at 1d. The portfolio backtester technically supports 1h/4h but it is untested at those frequencies.
- **Signal class is price-based throughout.** Every strategy in this repo derives its signal from price, volume (sometimes), and one another. Signals from outside that data — funding, on-chain, sentiment — are absent and would represent the next genuine direction.

### E. How to rerun safely

The stage runner writes `results/research_run_state.json` after each completed stage. If a run is interrupted, the file ends with `interrupted: true` — and any partial run that *included* `summary` or `scorecard` flushes the previous final-verdict files at the start, so a half-finished run cannot leave a misleading PASS/FAIL in place.

Recommended recipes:

```bash
# Quick smoke (no robustness, no Monte Carlo, no slow stages):
python main.py research_all --skip-robustness --stage data_coverage \
    strategy_comparison walk_forward scorecard summary

# Full clean run (multi-hour at 1h):
python main.py download --timeframes 1h 4h 1d --lookback-days 1460 --refresh
python main.py research_all
python main.py research_all_portfolio

# Inspect a finished run:
cat results/research_run_state.json | python -m json.tool
```

---

*End of report. v1 research closure.*
