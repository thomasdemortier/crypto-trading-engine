# crypto_trading_engine

A research-only backtesting and paper-trading engine for **BTC/USDT** and **ETH/USDT** spot, with a Streamlit dashboard.

> **No live trading. No financial advice. Educational use only.**
> Version 1 has no order-placement code path. The hard-coded safety lock
> `LIVE_TRADING_ENABLED = False` in `src/config.py` is enforced at runtime by
> `utils.assert_paper_only()`. Flipping the flag will not enable trading — it
> will refuse to run, because no execution module exists in v1.

## Current research status

- **No strategy has passed.**
- **Do not paper trade.**
- **Do not connect Kraken** (or any other broker).
- The research engine is usable as a tool for falsifying future strategy ideas.
- **BTC buy-and-hold remains the strongest practical baseline.**
- Future work should use **new signal classes** (funding, on-chain flows, sentiment, real BTC dominance), not further tweaking of price-based TA on the same universe.

For the full evidence see [`reports/final_crypto_research_report.md`](reports/final_crypto_research_report.md).

**Latest measured results** (1459 days of BTC + ETH 1h / 4h / 1d, plus 8 alts at 1d):

- **0 / 54** single-asset cells beat buy-and-hold.
- **0 / 42** beat the random placebo on both OOS stability and mean return.
- Portfolio momentum (Top-3 weekly rebalance): beat placebo by 157pp in total return, but only beat BTC in 36% of OOS windows; **FAIL** at 21% stability.
- Regime-aware portfolio momentum: lost 46% of capital while BTC returned +176%; **FAIL** at 0% stability.
- Scorecard verdicts across all variants: 40 FAIL · 14 INCONCLUSIVE · 6 BENCHMARK · 6 PLACEBO · **0 PASS · 0 WATCHLIST**.

**Data sources:**

- **Binance** is the primary source for historical research (proper backwards pagination, 1000 candles/call, multi-year depth on BTC/ETH/USDT).
- **Kraken** is reserved for live/execution use only — its public OHLC API caps at the most-recent ~720 candles regardless of the `since` parameter, which is insufficient for walk-forward validation. The collector detects this "stuck pagination" and falls back to Binance.

---

## What it does

1. Downloads public OHLCV candles via `ccxt` (Kraken first, Binance fallback).
2. Computes indicators (RSI, SMA50, SMA200, ATR%, volume MA) without lookahead.
3. Runs a deterministic, **long-only** strategy: buy below RSI 35 in an uptrend,
   exit on RSI 65, MA50 break, or stop loss.
4. Routes every signal through a **risk engine** that enforces position caps,
   per-trade risk, daily loss limits, fees, and slippage.
5. Simulates fills at the **next bar's open** (no same-bar lookahead).
6. Saves equity curve, trade log, decision log, and summary metrics.
7. Exposes everything in a Streamlit dashboard.

## Project layout

```
crypto_trading_engine/
├── data/{raw,processed}/        # CSV cache (gitignored)
├── logs/                        # trades.csv, decisions.csv (gitignored)
├── results/                     # equity_curve.csv, summary_metrics.csv,
│                                # paper_state.json (gitignored)
├── src/
│   ├── config.py                # SINGLE SOURCE OF TRUTH for parameters
│   ├── data_collector.py        # ccxt fetch_ohlcv (public only)
│   ├── indicators.py            # RSI, SMA, ATR — shift-aware, no leakage
│   ├── strategy.py              # long-only signal generator
│   ├── risk_engine.py           # the only path to a simulated fill
│   ├── backtester.py            # next-bar fill, fees+slippage
│   ├── performance.py           # metrics & buy-and-hold benchmark
│   ├── paper_trader.py          # one-tick paper trader (simulated only)
│   ├── plotting.py              # Plotly chart builders
│   └── utils.py                 # paths, logging, paper-only guard
├── tests/                       # pytest suite
├── main.py                      # CLI
├── streamlit_app.py             # dashboard (entrypoint at repo root)
├── requirements.txt
└── .gitignore
```

## Local setup (macOS / Linux)

```bash
cd "/Users/thomasd/Desktop/DATA TD/CV AND WORK/Trading Bots"
mkdir -p crypto_trading_engine
cd crypto_trading_engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py download
python main.py backtest
streamlit run streamlit_app.py
```

## CLI reference

| Command | Description |
| --- | --- |
| `python main.py download`            | Download all configured assets/timeframes (cached). |
| `python main.py download --refresh`  | Force re-download. |
| `python main.py backtest`            | Run a backtest with the defaults from `config.py`. |
| `python main.py backtest --timeframe 1d --assets BTC/USDT` | Override defaults. |
| `python main.py paper`               | Run one paper-trade evaluation tick. |
| `python main.py clean_logs -y`       | Wipe `logs/` and `results/`. |
| `python main.py status`              | Show config and cache state. |

## How to read the results

* `results/equity_curve.csv` — portfolio value, cash, exposure per bar.
* `logs/trades.csv` — every executed simulated fill, including fee and slippage.
* `logs/decisions.csv` — every BUY/SELL/HOLD/SKIP/REJECT decision with reason.
* `results/summary_metrics.csv` — one-row summary of all performance metrics.

The dashboard reads these artifacts; you can also load them in a notebook.

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repository (the repo root must contain
   `streamlit_app.py` and `requirements.txt`).
2. Visit https://share.streamlit.io and **New app** → pick the repo and
   `streamlit_app.py` as the entrypoint.
3. **No secrets are required** in v1.
4. The cloud will install `requirements.txt` and run the app.

If you ever add features that need API keys, define them in
`.streamlit/secrets.toml` locally (gitignored) and in the **Secrets** UI on
Streamlit Cloud.

## Known limitations (be honest about these)

* **Stop-loss fill model.** When a bar's open is below the stop (gap-down),
  we fill at the bar's open (the realistic outcome). Otherwise, when the bar
  trades through the stop intra-bar, we fill at the stop price. Real fills
  on illiquid markets may still be a touch worse.
* **No order book.** We assume fills at the next-bar open with a configurable
  slippage. Real fills depend on liquidity at the moment of execution.
* **One position per asset.** No averaging down, no scaling in or out.
* **No funding rates, no tax, no withdrawals.**
* **Buy-and-hold benchmark is equal-weighted across the selected assets.**
* **Strategy is intentionally simple.** It is not optimised; the engine's
  purpose is to test whether *any* result survives fees, slippage and
  drawdowns.

## Future improvements (v2 candidates)

1. Walk-forward and out-of-sample testing.
2. Multiple strategies (mean-reversion, breakout, momentum) with regime
   classification.
3. Monte Carlo bootstrapping over the equity curve / trade order.
4. Better risk analytics: Ulcer index, tail VaR, beta to BTC.
5. Kronos signal comparison as an additional feature.
6. Paper trading scheduler (cron) writing to a database.
7. SQLite/Postgres storage replacing CSV.
8. Authentication on the dashboard for multi-user deployment.
9. Deployment hardening: rate-limit guards, retries with backoff,
   per-exchange symbol sanity checks.

## Recommended local Python version

**Python 3.12** is recommended for local development. Streamlit Community
Cloud uses Python 3.12 and the project has been validated against it.
macOS system Python 3.9 still works but produces a harmless `urllib3` /
`LibreSSL` warning at startup. This is **not** a blocker.

## Optional Kronos research

Kronos is an optional, **local-only** ML confirmation layer for the
rule-based strategy signals. It is experimental, opt-in, and **not** part
of the deployed Streamlit Cloud app.

* Kronos is optional — the dashboard, backtester and Research Lab work
  without it.
* It is experimental — treat its forecasts as confirmation hints, not
  truth.
* It is not live trading — Kronos never places orders.
* It does not replace the risk engine — every signal still goes through
  fees, slippage, position caps, and stop-loss.
* It confirms / rejects existing rule-based signals — the wrapper strategy
  keeps a base BUY only if Kronos agrees, etc.
* It requires local setup — model weights are pulled from Hugging Face on
  first use; do not commit them.

### Setup (local only)

```bash
# 1. Optional ML deps (NOT in main requirements.txt)
pip install -r requirements-ml.txt

# 2. Clone the official Kronos repo outside this project
mkdir -p external
git clone https://github.com/shiyu-coder/Kronos.git external/Kronos
# or set:  export KRONOS_REPO_PATH="/path/to/Kronos"

# 3. Verify
python main.py kronos_status
```

### Running Kronos research

```bash
python main.py kronos_forecast --asset BTC/USDT --timeframe 4h --model Kronos-mini
python main.py kronos_evaluate --asset BTC/USDT --timeframe 4h --max-windows 10
python main.py kronos_confirm  --asset BTC/USDT --timeframe 4h
python main.py kronos_compare  --asset BTC/USDT --timeframe 4h
```

The Streamlit Research Lab also exposes a "Kronos confirmation" tab when
the optional dependencies are installed locally. **Kronos should be tested
locally first because model loading may be heavy** — do not enable it on
Streamlit Cloud.

The `external/`, `*.safetensors`, `*.bin`, `*.pt`, `models/`, `hf_cache/`
and `results/kronos_*` paths are all gitignored — model weights and the
upstream Kronos repo are never committed.

## Data source limitations

* **Primary source: Binance public OHLCV** (`startTime` paging, 1000 candles
  per call — the only one of the two that actually serves multi-year
  history for 1h / 4h / 1d on BTC/USDT and ETH/USDT).
* **Fallback source: Kraken public OHLCV** — caps at the most-recent ~720
  candles regardless of the `since` parameter. The collector detects this
  "stuck pagination" and bails out so it can fall back to Binance.
* **Per-symbol price differences between exchanges are real.** Binance and
  Kraken trade BTC/USDT and ETH/USDT against slightly different spot
  books (Kraken often uses USD, not USDT, with `KRAKEN_USDT_TO_USD_FALLBACK`
  bridging the gap). When `download_symbol` merges fallback rows into a
  primary-short result, you may see small price jumps where the two
  feeds meet. These are real exchange-level differences, not a bug, and
  they can affect backtest results. Prefer single-source downloads
  (Binance is sufficient for our timeframes) when running serious
  research.
* `data_coverage.csv` records the actual provenance: `actual_start`,
  `actual_end`, `candle_count`, `expected_bars`, `gap_count`,
  `largest_gap_bars`, `coverage_days`, `enough_for_walk_forward`. Inspect
  it before trusting any walk-forward verdict.

## Current project state

* **BTC buy-and-hold is the production baseline.**
* **No strategy has passed the conservative scorecard.** Nine archived
  research branches are kept as decision evidence — none are merged
  into `main`.
* The engine is now used as a **research and risk dashboard**:
  scorecard surfacing, archived-verdict tracking, baseline metrics,
  and decision-journal review.
* **Execution remains locked.** Kraken is not connected. Paper
  trading and live trading are both disabled and have no entry point
  in the codebase.
* The Streamlit app exposes a `Research Dashboard` section
  (executive state, strategy verdicts, archived timeline, baseline,
  risk dashboard, safety + governance, next allowed actions). It is
  read-only.
* Archived research branches: `research/fail-1-funding-derivatives`,
  `research/strategy-2-market-structure`,
  `research/strategy-3-sentiment-fear-greed`,
  `research/strategy-4-drawdown-targeted-btc`,
  `research/strategy-5-paid-positioning-data-audit`,
  `research/strategy-6-funding-basis-carry`,
  `research/strategy-7-relative-value-btc-eth`,
  `research/strategy-8-paid-data-decision-audit`,
  `research/strategy-9-free-open-data-reaudit`. See
  [`docs/research_dashboard.md`](docs/research_dashboard.md) for what
  the dashboard shows and what NOT to do with it.

## Portfolio risk dashboard

The engine can now be used to review portfolio exposure and drawdown
scenario risk from a local **gitignored** CSV file. This is read-only.
It does not place trades. It does not connect to brokers. It does not
produce strategy signals.

* Drop a local file at `data/portfolio_holdings.csv` (the path is
  gitignored — never commit user financial data). Required columns:
  `asset, quantity, average_cost, currency, current_price,
  price_source, notes`.
* Run `streamlit run streamlit_app.py` and scroll to the
  **Portfolio Risk** section below the Research Dashboard.
* The dashboard shows: schema status, position table, portfolio
  summary, drawdown scenarios (-10 % / -20 % / -30 % / -50 %), BTC
  baseline comparison, locked risk classification (LOW / MODERATE /
  HIGH / EXTREME / UNKNOWN), and a non-trading recommendation
  (`hold risk steady` / `review concentration` / `reduce
  concentration` / `data missing` / `do nothing until data is
  complete`).
* The recommendation language is locked. The dashboard never says
  "buy", "sell", "open position", "place order", or "connect broker"
  — a unit test enforces this.

See [`docs/portfolio_risk_dashboard.md`](docs/portfolio_risk_dashboard.md)
for the CSV format, the locked thresholds, and the safety rules.

## Health snapshots

The engine can append a single row of bot state to a gitignored CSV
on demand. Each row captures: safety lock status, execution / paper /
Kraken flags, system-health pass / warning / fail counts, strategy
registry verdict counts, and (if a portfolio CSV is present)
portfolio total market value, risk classification, and
recommendation. No trading. No execution. No broker. No API keys.

Run from the CLI:

```bash
python main.py write_health_snapshot
```

Output is appended to `results/health_snapshots.csv` (gitignored —
covered by the existing `results/*.csv` rule). The Streamlit
**Health Timeline** section reads the file and renders the most
recent 50 rows, plus a per-snapshot summary of locked / unlocked
state and warnings. The dashboard intentionally does not expose a
"write snapshot" button — snapshots are written from the CLI only.

See [`docs/health_snapshots.md`](docs/health_snapshots.md) for the
locked schema, what the snapshot tracks, and what it does not do.

## Safety reminders

* `LIVE_TRADING_ENABLED` must remain `False` in v1.
* No code path imports private ccxt methods or accepts API keys.
* Kronos modules use lazy imports — `torch` / `transformers` / Hugging
  Face are NEVER imported at app startup, so the dashboard still loads
  on Streamlit Cloud without `requirements-ml.txt`.
* If you need live execution later, write a **separate**, audited
  execution module — do not graft it into the existing engine.
