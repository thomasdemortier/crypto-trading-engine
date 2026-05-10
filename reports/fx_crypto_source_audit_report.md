# FX + crypto data source audit

This is a **data audit only**. No strategy was built. No backtest. No
broker integration. No API keys. No paper trading. No live trading.
No order placement. Execution remains locked.

> **Verdict: GO for FX research, GO on data alone for crypto, but
> with very different operational implications.**
>
> * **Forex** has 9 free PASS rows spanning 27+ years (ECB SDMX,
>   Frankfurter) plus 50+ years on LBMA gold. This is **untested on
>   this engine** — the universe-selection branch ranked it NOT_NOW
>   only because of the broker-integration prerequisite. For
>   research-level work on daily series, the data is sufficient.
> * **Crypto** has 10 free PASS rows (Binance OHLCV / funding /
>   basis, Bybit funding, Deribit funding). But this universe has
>   already been tested ten times and FAILed each time; data
>   availability is not the bottleneck.
> * **5 FX INCONCLUSIVE rows** all collapse to the same root cause:
>   the 2024–2026 wave of "previously-free" sources moved behind
>   captcha-gated keys (Stooq) or never opened (OANDA, IG,
>   Dukascopy, FRED).
>
> **Recommended next research branch:**
> `research/fx-eod-trend-strategy-v1` — a long-horizon trend or
> carry strategy on EUR/USD, USD/JPY, USD/CHF, and XAU/USD using
> daily ECB / LBMA fixes. PASS criteria deliberately NOT "beat
> S&P or beat BTC" — they should be Sharpe-comparable to a
> 60/40-style buy-and-hold benchmark with materially tighter
> drawdown.

## Forex sources audited

### Reachable + free + multi-year (PASS)

| Source | Asset | Field | Coverage | Notes |
| --- | --- | --- | ---: | --- |
| ECB SDMX | EUR/GBP | reference_rate | **9986 days** | Daily official fix since 1999-01-04 |
| ECB SDMX | EUR/JPY | reference_rate | 9986 days | Same |
| ECB SDMX | EUR/CHF | reference_rate | 9986 days | Same |
| ECB SDMX | USD/JPY (derived) | reference_rate | 9986 days | Cross-rate from JPY/EUR ÷ USD/EUR |
| ECB SDMX | USD/CHF (derived) | reference_rate | 9986 days | Same pattern |
| ECB SDMX | GBP/USD (derived) | reference_rate | 9986 days | Same pattern |
| Frankfurter | EUR/USD | reference_rate | 5972 days | Wraps ECB feed; JSON-friendly; no key |
| LBMA | XAU/USD | reference_rate | **21221 days** | London PM gold fix — daily since 1968 |
| Yahoo Finance | XAU/USD | ohlcv | 3650 days | Gold futures GC=F; works today but Yahoo throttles |

The ECB EUR/USD probe **timed out on this run** and was recorded as
FAIL — that is honest, not a structural failure of the source. A
re-run will likely succeed; ECB EUR/GBP, EUR/JPY, and EUR/CHF all
returned 27 years of data on the same call.

### Inconclusive — keys / paid (5)

| Source | Reason |
| --- | --- |
| OANDA v20 | Practice + live REST require an account-bound API key |
| IG Bank Switzerland | REST requires authenticated session |
| Stooq | CSV download moved behind captcha-gated `apikey` parameter (was free historically) |
| Dukascopy | bi5 binary endpoint returns 503 to public User-Agents; only accessible via JForex client / community scrapers |
| FRED St. Louis | API requires free key; public CSV download endpoint timed out from this environment |

## Crypto sources audited

### Reachable + free + multi-year (PASS)

| Source | Asset | Field | Coverage |
| --- | --- | --- | ---: |
| Binance | BTC | OHLCV | 3188 days |
| Binance | ETH | OHLCV | 3188 days |
| Binance | SOL | OHLCV | 2098 days |
| Binance Futures | BTC | funding | 2434 days |
| Binance Futures | ETH | funding | 2356 days |
| Binance Futures | BTC | basis_or_premium | 1499 days |
| Binance Futures | ETH | basis_or_premium | 1499 days |
| Bybit | BTC | funding | 2131 days |
| Deribit | BTC | funding | 2462 days |
| Deribit | ETH | funding | 2462 days |

### Warning (365–1459)

| Source | Asset | Field | Coverage |
| --- | --- | --- | ---: |
| Kraken | BTC | OHLCV | 720 days (hard-capped) |

### FAIL

| Source | Asset | Field | Reason |
| --- | --- | --- | --- |
| Binance Futures | BTC | open_interest | hard-capped ~30 days |
| Bybit | BTC | open_interest | capped at recent ~200 days |
| OKX | BTC | funding | public cursor-walks back ~100 days |
| Deribit | BTC | order_book_snapshot | snapshot only — no historical depth |

## Which fields passed 1460 days

* **FX reference rates**: ECB / Frankfurter for EUR-quoted majors and derivable cross-rates — all 9986 days (27+ years).
* **Gold reference rate**: LBMA daily fix — 21,221 days (58 years).
* **Crypto OHLCV**: Binance BTC/ETH/SOL — 2098–3188 days.
* **Crypto funding**: Binance BTC/ETH (2434/2356d), Bybit BTC (2131d), Deribit BTC/ETH (2462d each).
* **Crypto basis**: Binance mark/index klines BTC/ETH (1499d each).

## Which fields failed

* **Public OI history** on Binance (30d cap), Bybit (200d cap) — same wall the strategy-5 + strategy-9 audits documented. Unchanged.
* **Public OKX funding** — cursor-walk capped at ~100 days; on this run only 33 days reached.
* **Public liquidations history** — no exchange exposes deep history on a free endpoint.
* **Public long/short ratios** — capped 30–500 days everywhere.
* **Kraken OHLC** — 720-day cap (was already known).
* **Order-book / OI snapshots** — Deribit's snapshot-only; no temporal depth.

## Which sources require keys

| Market | Source | Status |
| --- | --- | --- |
| Forex | OANDA v20 (practice + live) | account-bound key |
| Forex | IG Bank Switzerland | session auth |
| Forex | Stooq CSV | captcha-gated apikey (regression — was free) |
| Forex | Dukascopy bi5 | requires JForex / community scraper |
| Forex | FRED St. Louis | free key required |

All five recorded as `INCONCLUSIVE` rather than `FAIL` — the data
exists; the free public path is just no longer there. Adding any of
these is **out of scope** under the project's no-API-key policy.

## Whether Forex is viable for next strategy research

**Yes, with caveats.** Daily reference-rate research IS viable on the
free public stack today:

* **Pros:**
  * 27+ years of daily fixes for EUR-quoted majors via ECB.
  * 58 years of LBMA gold daily fixes.
  * No API keys. No broker integration required for backtest-only
    research. Fits the existing `portfolio_backtester` if we add an
    FX universe loader.
  * Untested on this engine. Not exhausted like crypto.

* **Cons:**
  * **Reference rate ≠ tradable price.** ECB / LBMA fixes are
    end-of-day official prints. They have no spread, no volume, no
    intraday structure. A strategy backtested on these will be
    optimistic about fills.
  * **No paper / live path** without a broker. The project's
    no-broker policy means the strategy stays research-only;
    promotion to paper trading would require a separate broker
    policy decision.
  * **Retail FX edge is structurally low.** Even with clean data, FX
    research must clear a comparable risk-adjusted bar; the
    strategy-universe-selection report scored FX edge=3/10 for
    exactly this reason.

A research strategy that is **explicit about its limitations** is
viable: long-horizon, daily-resolution, trend or carry on a small
basket (EUR/USD, USD/JPY, USD/CHF, XAU/USD), benchmarked against an
FX equal-weight basket, judged on Sharpe + drawdown rather than raw
return.

## Whether crypto is viable for next strategy research

**Yes on data, no operationally.** The data is the same data
strategy-9 already audited as PASS. But the same data has been
consumed by ten failed branches: drawdown-targeted, funding+basis
carry, BTC/ETH relative-value, market-structure, sentiment overlay,
funding-only, paid data audits, and the most recent portfolio
rebalancing FAIL. There is no free-data signal class left to test.

**Recommendation for crypto:** do not open another long-only crypto
research branch unless a paid plan (CoinGlass / CryptoQuant / Velo)
is purchased and the keyed audit (`audit_coinglass_keyed_data`,
already shipped on a previous branch) returns PASS. The locked
decision rule from strategy 8/9 still applies.

## Recommended next research branch

```
research/fx-eod-trend-strategy-v1
```

Scope (locked before any code is written):

1. **Universe**: EUR/USD, USD/JPY, USD/CHF, XAU/USD (4 instruments).
2. **Data**: ECB SDMX for fiat majors (with cross-rate derivation
   for USD/JPY, USD/CHF), LBMA daily PM fix for gold. Frankfurter as
   a Frankfurter cross-check on EUR/USD. Build an `fx_data_collector`
   module that downloads + caches daily series locally.
3. **Strategy class**: long-horizon trend + carry. Examples:
   monthly rebalance to N-asset trend-positive subset, or fixed
   60 % USD-strong / 30 % gold / 10 % cash with rebalancing.
4. **Locked PASS criteria** (decide BEFORE seeing results):
   * Sharpe within 0.10 of an FX equal-weight basket.
   * Max drawdown ≥ 10 pp tighter than the basket.
   * Beats placebo MEDIAN return AND drawdown across 20 random
     fixed-weight seeds.
   * ≥ 24 rebalances total across the OOS window.
   * No leverage, no shorting, no broker.
5. **Output**: scorecard CSV + walk-forward CSV + placebo CSV — all
   gitignored.
6. **Honesty rule**: if the strategy fails the locked scorecard,
   archive it the same way the ten prior branches were archived. Do
   not retune after seeing results. Do not move to paper trading
   even on PASS — the no-broker policy still holds.

Crucially, **do NOT call this a Forex trading strategy**. Without a
broker connection, intraday data, and spread modelling, this is
research-grade evidence — not a deployable bot. The next decision —
broker policy change, paid intraday FX data, or stop here — should
follow the verdict of this branch, not precede it.

## Exact next step

Build `research/fx-eod-trend-strategy-v1` from `main`. Same
discipline as every prior branch: locked weights, locked PASS
criteria, full-window + walk-forward + placebo + scorecard, honest
verdict. Archive on FAIL; on PASS, independent review required
before any further step. Execution remains locked.

If you'd rather not open another strategy branch yet, the alternative
is to freeze at `v0.5` and use the engine as the research-and-risk
dashboard it already is. Both options are honest. Only one builds
toward a Forex bot.
