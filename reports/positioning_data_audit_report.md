# Positioning data availability audit

This is a **data availability audit only**. No strategy was built, no
backtest was run, no broker integration, no API keys. Execution remains
locked.

The audit asks one question:

> Can we access enough historical market-positioning data to justify a
> serious strategy test?

It probes a curated list of public derivatives endpoints and records
each one's depth, granularity, fields, paid-vs-free status, and a
research-usability verdict (PASS ≥ 1460 days, WARNING 365–1459, FAIL
< 365 days or current snapshot only, INCONCLUSIVE for paid sources we
cannot verify without a subscription). Paid sources are listed but
**not** probed.

> **Verdict: GO — but narrowly.** Four data streams clear the
> 1460-day PASS bar: Binance + Bybit + Deribit funding-rate history
> and the Binance mark/index basis. The OI / long-short / taker family
> is uniformly **FAIL** on every public exchange (≤ 30–500 days). The
> richer aggregated OI + liquidations + cross-venue stack lives behind
> CoinGlass / CryptoQuant / Glassnode / Kaiko / Velo Data and is
> **INCONCLUSIVE without a paid subscription**.

## 1. Sources tested

| # | Source | Endpoints probed |
| - | --- | --- |
| 1 | Binance Futures (public) | funding_rate_history, openInterestHist (1d), topLongShortAccountRatio (1d), takerlongshortRatio (1d), markPriceKlines (1d) |
| 2 | Bybit v5 (public) | funding/history (linear), open-interest (1d), account-ratio (1d) |
| 3 | OKX (public) | funding-rate-history (BTC-USDT-SWAP, walked 5 cursor pages), rubik open-interest-volume (1D), rubik long-short-account-ratio (1D), rubik taker-volume (1D) |
| 4 | Deribit (public) | get_funding_rate_history (BTC-PERPETUAL), get_volatility_index_data (DVOL, BTC), get_book_summary_by_currency (futures) |
| 5 | CoinGlass | recorded as paid_only — not probed |
| 6 | CryptoQuant | recorded as paid_only — not probed |
| 7 | Glassnode | recorded as paid_only — not probed |
| 8 | Kaiko | recorded as paid_only — not probed |
| 9 | Velo Data | recorded as paid_only — not probed |

Total rows in the audit CSV: **20** (15 probed + 5 paid).

## 2. Endpoints that worked (PASS — ≥ 1460 days)

| Source | Dataset | Coverage (days) | Granularity | Notes |
| --- | --- | ---: | --- | --- |
| binance_futures | funding_rate_history | **2434** | 8h | first row 2019-09-10; paginates via `startTime` |
| binance_futures | mark_price_klines | **1499** | 1d | mark + index → futures-spot basis (1500-row cap per request) |
| bybit | funding_rate_history | **2131** | 8h | first row 2020-07-09; paginates via `startTime`/`endTime` |
| deribit | perpetual_funding_rate_history | **2462** | ~1h | range-window endpoint; paginates back to BTC-PERPETUAL launch (2019-08) |

These are the only endpoints with multi-year free public depth.

## 3. Endpoints that worked but came up short (WARNING — 365–1459 days)

| Source | Dataset | Coverage (days) | Why it's not PASS |
| --- | --- | ---: | --- |
| bybit | long_short_account_ratio | 499 | Bybit serves the most recent ~500 days of 1d account-ratio; no public extension |
| deribit | dvol_index_history | 999 | Deribit DVOL daily series starts 2022-05-27; usable for vol regime context |

## 4. Endpoints that failed (FAIL — < 365 days or snapshot-only)

| Source | Dataset | Coverage (days) | Why it failed |
| --- | --- | ---: | --- |
| binance_futures | open_interest_history_1d | 29 | Binance hard-caps OI history at ~30 days for all granularities |
| binance_futures | top_long_short_account_ratio | 29 | same 30-day cap as OI |
| binance_futures | taker_long_short_ratio | 29 | same 30-day cap as OI |
| bybit | open_interest_1d | 199 | Bybit's `open-interest` returns the most recent ~200 daily rows; cursor pagination does not extend past that |
| okx | funding_rate_history | 93 | empirically capped at ~3 months on the public endpoint; the `after` cursor stops advancing |
| okx | rubik_open_interest_volume | 179 | OKX rubik aggregates serve only the most recent ~6 months |
| okx | rubik_long_short_account_ratio | 179 | same OKX rubik cap |
| okx | rubik_taker_volume | 179 | same OKX rubik cap |
| deribit | book_summary_by_currency | n/a | snapshot only — current OI per future, no history |

This is the same "30-day OI cap" pattern that left the funding+OI
research INCONCLUSIVE on the previous branch. Nothing on the public
endpoints has changed.

## 5. Sources that need a key or paid plan (INCONCLUSIVE)

| Source | Endpoint | What they offer |
| --- | --- | --- |
| coinglass | open-api.coinglass.com (v3) | aggregated OI, funding, liquidations, long/short ratios across exchanges |
| cryptoquant | api.cryptoquant.com (v1) | exchange reserves, OI, funding, taker buy/sell, miner flows |
| glassnode | api.glassnode.com (v1) | on-chain balances, exchange flows, derivatives OI, funding, options skew |
| kaiko | us.market-api.kaiko.io | consolidated OI, funding, basis, taker volume across venues — enterprise-priced |
| velo_data | api.velodata.app | OI, funding, basis, options skew — derivatives-focused |

We did not probe these. Without a paid subscription we cannot verify
their historical depth, asset coverage, or rate limits. Per spec, all
five are flagged **INCONCLUSIVE** rather than recommended.

## 6. Asset coverage

The audit was run for the **BTC** legs of each endpoint. The same
endpoints generally serve every other asset in our universe (ETH, SOL,
AVAX, LINK, XRP, DOGE, ADA, LTC, BNB) at the **same depth and
granularity**, but with one important caveat: the most recent perpetuals
(SOL, AVAX, LINK on some venues) launched in 2021 or later and so will
have shorter funding history. BTC is the strict best case. A
secondary multi-asset probe should run before any strategy work
commits to a non-BTC universe.

## 7. Most-promising data type

Two distinct lanes clear the bar:

1. **Funding-rate history (Binance, Bybit, Deribit perp).** Six-plus
   years of free public depth at 8h (Binance, Bybit) or ~1h (Deribit).
   This is the cleanest positioning signal that can be assembled
   without paying.
2. **Futures basis from Binance mark vs. index klines.** 1500-day
   daily depth on both series; the basis (mark/index − 1) is a
   coherent multi-year proxy for futures-market premium / discount.

These two together cover the **funding + basis** half of the
positioning toolkit. They do **not** cover the **OI + liquidation +
long/short ratio** half — that half remains gated behind paid sources.

## 8. Is a positioning-data strategy justified?

**Conditional GO — funding + basis only.** A research strategy
limited to funding-rate dynamics + futures basis can be built today
on free public data and pass the project's data-coverage bar. That is
genuinely new territory the existing `derivatives_funding` branch did
NOT exhaust (it stalled because OI history capped at 30 days, but
funding alone clears 1460 days at three venues now).

**No-GO without paid data — for the OI / liquidation / long-short
positioning thesis.** If the hypothesis we want to test depends on
aggregated OI, liquidation cascades, or top-trader long/short ratios,
the public endpoints are too shallow. The honest next step is to
budget for a single paid subscription (CoinGlass or CryptoQuant look
like the cheapest path) and re-audit, rather than build a strategy on
30-day windows.

## 9. Exact next step

Pick exactly one of these and keep doing nothing else until it lands:

1. **Free path (recommended first):** open
   `research/strategy-6-funding-basis-allocator` and build the strategy
   on the four PASS data streams (Binance + Bybit + Deribit funding +
   Binance basis). 14-window OOS + 20-seed placebo + the same
   scorecard bar. Do **not** mix in OI from any of the FAIL endpoints.
2. **Paid path:** purchase a single CoinGlass or CryptoQuant developer
   plan, re-run `python main.py audit_positioning_data` against the
   keyed endpoints (extension required), and only then decide if the
   OI / liquidation thesis is reachable. Until that re-audit lands,
   treat the paid lane as INCONCLUSIVE — not as a hidden PASS.

## 10. What this branch is and is not

This branch is a **data audit only.**

* No strategy was built.
* No backtest was run.
* No broker integration; no API keys; no order placement.
* No Kraken connection.
* Paper trading remains disabled.
* Execution remains locked.
* Test suite: 302 / 302 passing. CI safety gate: 10 / 10 PASS.
* The audit CSV at `results/positioning_data_audit.csv` is generated
  by the CLI command and is **not tracked** (covered by `.gitignore`).
