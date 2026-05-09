# crypto_trading_engine

A research-only backtesting and paper-trading engine for **BTC/USDT** and **ETH/USDT** spot, with a Streamlit dashboard.

> **No live trading. No financial advice. Educational use only.**
> Version 1 has no order-placement code path. The hard-coded safety lock
> `LIVE_TRADING_ENABLED = False` in `src/config.py` is enforced at runtime by
> `utils.assert_paper_only()`. Flipping the flag will not enable trading — it
> will refuse to run, because no execution module exists in v1.

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

## Safety reminders

* `LIVE_TRADING_ENABLED` must remain `False` in v1.
* No code path imports private ccxt methods or accepts API keys.
* If you need live execution later, write a **separate**, audited
  execution module — do not graft it into the existing engine.
