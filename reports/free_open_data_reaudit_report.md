# Free / open data re-audit

This is a **free / open data audit only**. No strategy was built. No
backtest. No broker integration. No API keys. No paid plans. No
order placement. No Kraken connection. Execution remains locked.

> **Numerical verdict: GO.** Five useful field types beyond OHLCV +
> basic funding clear the 1460-day PASS bar on the free public stack:
> `basis_or_premium`, `onchain_or_market_structure`, `sentiment`,
> `stablecoin_or_liquidity`, `tvl`.
>
> **Operational verdict: NO_GO.** Every one of those five field
> types was already used in a long-only strategy branch on this
> project, and every one of those branches **failed**. The free
> stack adds no NEW multi-year data class we have not already
> tested. **Stop opening new long-only strategy branches. Accept BTC
> buy-and-hold as baseline.** Use the engine for risk reporting,
> dashboarding, and research falsification.

## 1. Executive summary

| | |
| --- | --- |
| Datasets probed | 36 |
| PASS (>= 1460 days) | **19** |
| WARNING (365–1459 days) | 6 |
| FAIL (< 365 days or snapshot only) | 6 |
| INCONCLUSIVE | 5 |
| Useful PASS field types (beyond OHLCV + basic funding) | `basis_or_premium`, `onchain_or_market_structure`, `sentiment`, `stablecoin_or_liquidity`, `tvl` |
| Useful WARNING field types | `long_short_ratio`, `vol_index` |
| Numerical decision rule | **GO** (>= 2 useful PASS field classes) |
| Operational recommendation | **NO_GO** — every PASS field class already used in a failed branch |

## 2. Why this audit was needed

Previous branches established a reliable failure pattern:

| Branch | Verdict | Data classes used |
| --- | --- | --- |
| `research/strategy-2-market-structure` | FAIL | DefiLlama TVL + on-chain + alt breadth |
| `research/strategy-3-sentiment-fear-greed` | FAIL | Alternative.me F&G overlay |
| `research/fail-1-funding-derivatives` | INCONCLUSIVE | Public funding + OI (OI capped 30d) |
| `research/strategy-4-drawdown-targeted-btc` | FAIL | BTC drawdown + 200d MA |
| `research/strategy-6-funding-basis-carry` | FAIL | Binance/Bybit/Deribit funding + Binance basis |
| `research/strategy-7-relative-value-btc-eth` | FAIL | ETH/BTC ratio rotation |
| `research/strategy-8-paid-data-decision-audit` | desk-research only — recommended CoinGlass trial |
| `research/strategy-9-coinglass-keyed-data-audit` | INCONCLUSIVE — no API key available |

We do not have a CoinGlass key, so the keyed audit is paused. The
question becomes: do **any free / open public sources** still have
multi-year depth we haven't squeezed yet?

## 3. Sources audited

| # | Source | Endpoints touched |
| - | --- | --- |
| 1 | Binance spot (public) | `/api/v3/klines` for BTC, ETH, SOL |
| 2 | Binance Futures (public) | funding rate history, mark-price klines, open-interest history |
| 3 | Bybit v5 (public) | funding history, open-interest, account ratio |
| 4 | OKX (public) | funding-rate-history (cursor-paginated) |
| 5 | Deribit (public) | perpetual funding history, DVOL index |
| 6 | Kraken (public) | spot OHLC |
| 7 | Alternative.me | Fear & Greed index |
| 8 | DefiLlama | per-chain TVL (BTC, ETH, SOL), total stablecoin supply |
| 9 | Blockchain.com | hash rate, n-transactions, market cap, estimated tx volume USD |
| 10 | CoinGecko (free tier) | `/coins/{id}/market_chart?days=max` |
| 11 | CoinPaprika (free tier) | `/tickers/{id}/historical` |

## 4. BTC fields

| Dataset | Source | Field type | Coverage (days) | Verdict |
| --- | --- | --- | ---: | --- |
| spot OHLCV 1d | binance | ohlcv | 3188 | PASS |
| funding rate history | binance_futures | funding | 2434 | PASS |
| mark-price klines (basis) | binance_futures | basis_or_premium | 1499 | PASS |
| open-interest history | binance_futures | open_interest | 29 | **FAIL** |
| funding rate history | bybit | funding | 2131 | PASS |
| open-interest 1d | bybit | open_interest | 199 | **FAIL** |
| long-short account ratio | bybit | long_short_ratio | 499 | WARNING |
| funding rate history | okx | funding | 93 | **FAIL** |
| perpetual funding history | deribit | funding | 2462 | PASS |
| DVOL index | deribit | vol_index | 999 | WARNING |
| spot OHLC 1d | kraken | ohlcv | 720 | WARNING |
| chain TVL history | defillama | tvl | 1876 | PASS |
| hash rate | blockchain_com | onchain | 6335 | PASS |
| n-transactions per day | blockchain_com | onchain | 6319 | PASS |
| market cap USD | blockchain_com | onchain | 6331 | PASS |
| estimated tx volume USD | blockchain_com | onchain | 5730 | PASS |
| market_chart days=max | coingecko_free | ohlcv | n/a | INCONCLUSIVE (401 — paid only) |
| historical max | coinpaprika_free | ohlcv | n/a | INCONCLUSIVE (402 — paid only) |

## 5. ETH fields

| Dataset | Source | Field type | Coverage (days) | Verdict |
| --- | --- | --- | ---: | --- |
| spot OHLCV 1d | binance | ohlcv | 3188 | PASS |
| funding rate history | binance_futures | funding | 2356 | PASS |
| mark-price klines (basis) | binance_futures | basis_or_premium | 1499 | PASS |
| open-interest history | binance_futures | open_interest | 29 | **FAIL** |
| funding rate history | bybit | funding | n/a | INCONCLUSIVE (empty list on the deep window) |
| open-interest 1d | bybit | open_interest | 199 | **FAIL** |
| long-short account ratio | bybit | long_short_ratio | 499 | WARNING |
| funding rate history | okx | funding | 93 | **FAIL** |
| perpetual funding history | deribit | funding | 2462 | PASS |
| DVOL index | deribit | vol_index | 999 | WARNING |
| spot OHLC 1d | kraken | ohlcv | 720 | WARNING |
| chain TVL history | defillama | tvl | 3147 | PASS |
| market_chart days=max | coingecko_free | ohlcv | n/a | INCONCLUSIVE (401) |
| historical max | coinpaprika_free | ohlcv | n/a | INCONCLUSIVE (402) |

## 6. SOL fields (optional)

| Dataset | Source | Field type | Coverage (days) | Verdict |
| --- | --- | --- | ---: | --- |
| spot OHLCV 1d | binance | ohlcv | 2098 | PASS |
| chain TVL history | defillama | tvl | 1879 | PASS |

SOL has multi-year OHLCV + chain TVL on the free stack. Funding /
basis / OI / liquidations remain the same FAIL pattern as BTC and
ETH — capped on every public exchange.

## 7. Which fields passed `>= 1460 days`

Across the full set of probes:

| Field type | PASS instances | First-row date |
| --- | --- | --- |
| `ohlcv` (spot) | BTC + ETH (3188d each) + SOL (2098d) | 2017-01 / 2020-08 |
| `funding` | Binance BTC/ETH (2434/2356d), Bybit BTC (2131d), Deribit BTC/ETH (2462d each) | 2019-08 → 2020-07 |
| `basis_or_premium` | Binance BTC/ETH mark-price (1499d each) | 2022-04 |
| `onchain_or_market_structure` | Blockchain.com hash rate / tx count / market cap / tx volume (5730–6335d) | 2009 → 2010 |
| `sentiment` | Alternative.me F&G (3020d) | 2018-02 |
| `stablecoin_or_liquidity` | DefiLlama total stablecoin supply (3083d) | 2017-11 |
| `tvl` | DefiLlama BTC/ETH/SOL chain TVL (1876–3147d) | 2017-09 → 2021-03 |

## 8. Which fields failed

* **Public open interest history** — Binance hard-caps at ~30 days,
  Bybit at ~200 days. Same finding as the 2024 strategy-5 audit.
* **Public OKX funding** — cursor pagination only reaches ~93 days.
* **Public liquidations history** — no exchange exposes this on a
  free endpoint with multi-year depth.
* **Bybit / Binance long-short account & top-trader ratios** — capped
  ≤ 500 days.

## 9. Which fields were inconclusive

* **CoinGecko free `/market_chart?days=max`** — returns HTTP 401
  Unauthorized as of 2024. The free public API now requires a Demo
  API key for historical depth beyond a few days. Recorded as
  INCONCLUSIVE rather than FAIL because the underlying dataset
  exists; the free tier just no longer serves it.
* **CoinPaprika `/tickers/.../historical`** — returns HTTP 402
  Payment Required. Same shape — the data exists; the free tier no
  longer serves it.
* **Bybit ETH funding deep-window probe** — returned an empty list
  at the 2017–2020 cursor; the endpoint behaves but the deep page
  is not populated. Recorded INCONCLUSIVE; the recent 200-row page
  works fine for a separate (shallower) probe.

## 10. Whether free data justifies another strategy branch

**No.** Despite the numerical GO verdict, the operational answer is
NO. Reasoning:

1. **Numerical GO is real but not new.** The five PASS field types
   beyond baseline are: `basis_or_premium`, `tvl`, `sentiment`,
   `stablecoin_or_liquidity`, `onchain_or_market_structure`. Every
   one of these was already consumed by a failed branch:
   - `tvl`, `onchain` → `research/strategy-2-market-structure` (FAIL)
   - `sentiment` → `research/strategy-3-sentiment-fear-greed` (FAIL)
   - `funding` + `basis_or_premium` → `research/strategy-6-funding-basis-carry` (FAIL)
   - The 200d-MA / drawdown classes → strategies 4 and 7 (FAIL)
2. **`stablecoin_or_liquidity` is the only data class not isolated
   in a prior branch.** It was used as one input among many in
   strategy-2 (market structure), but never as the primary signal.
   In principle it could be the differentiating axis of a future
   strategy. In practice, stablecoin-supply changes are slow,
   highly correlated with the broader market regime, and behave
   like a re-encoding of the same trend information that 200d MA
   already captures. The probability that a stablecoin-supply
   strategy succeeds where every prior signal failed is **low**.
3. **The free OI / liquidations / long-short stack stays FAIL.**
   This is the same wall the strategy-5 audit hit. CoinGecko free
   and CoinPaprika free have moved behind a key in 2024, so even
   the OHLCV-as-cross-check option is degraded. Nothing on the free
   stack has gotten better since the last audit; some has gotten
   worse.

The decision rule on the user's spec said:
> If FAIL: stop opening new long-only strategy branches and accept
> BTC buy-and-hold as the baseline.

The numerical rule clears GO, but the operational test ("does the
free stack add **meaningful new multi-year data** beyond what
already failed?") **fails**. The honest application of the user's
intent is therefore **NO_GO**.

## 11. Exact next step

**Do not open another long-only strategy branch.** Accept BTC
buy-and-hold as the production baseline. Use the engine for:

1. **Risk reporting** — `safety_status`, `system_health`,
   `bot_status` already provide the operational read-out. Keep
   running them as the daily heartbeat.
2. **Research falsification** — the 8 prior failed branches are the
   real artefact of this project. Future ideas should be
   benchmarked against the locked scorecard before a single line of
   strategy code is written.
3. **Dashboarding** — Streamlit dashboard surfaces every
   `*_scorecard.csv` and `*_walk_forward.csv`. It's a working
   research lab; no execution surface needed.

If the research budget reopens later, the only paths that change
the verdict are:

* **Buy a CoinGlass / CryptoQuant / Velo Data developer plan** for
  one month and re-run `audit_coinglass_keyed_data` (already on
  this project). Keyed depth on OI + liquidations + long-short
  ratios is the only data class we have not yet tested.
* **Add a true market-neutral pair backtester** so the next
  strategy can express dispersion as a long/short spread instead
  of swapping one beta for another. This requires a real shorting
  framework with borrow + funding cost — out of scope for the
  research-only branch policy.

Until one of those two paths opens, the recommendation is firm:
**stop coding strategies; ship BTC buy-and-hold as the baseline;
keep execution locked**.

This is a free / open data audit only. Do not build a strategy on
this branch. Do not merge into `main` — archive as decision evidence.
