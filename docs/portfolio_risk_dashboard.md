# Portfolio Risk Dashboard

A read-only review surface for personal crypto portfolio exposure
and drawdown scenario risk. It loads a local gitignored CSV, computes
position metrics + scenario shocks + a risk classification, and emits
one of five locked non-trading recommendations.

It exists because the engine's research track produced a clear
verdict — **no strategy passed, BTC buy-and-hold is the production
baseline** — and the next useful thing to build is a tool that helps
the user see their actual concentration and drawdown exposure
honestly. Not a strategy. Not a signal generator. Not a trader.

## Purpose

Help answer ten questions:

1. What is my current crypto exposure?
2. How much BTC and ETH risk am I carrying?
3. What is my drawdown exposure?
4. What happens if BTC drops 10 / 20 / 30 / 50 %?
5. How does my portfolio compare to BTC buy-and-hold?
6. Is the portfolio too concentrated?
7. What is the current risk state?
8. What is the recommended action category — hold, reduce risk,
   rebalance, or do nothing?
9. What data is missing before making decisions?
10. Is the bot still locked and safe?

The dashboard does **not** answer "should I buy", "should I sell",
"is now a good time", or "which asset will go up". Those are
deliberately out of scope.

## CSV format

Drop a file at `data/portfolio_holdings.csv`. The path is gitignored —
**never commit your real holdings**. Required columns:

| Column | Type | Notes |
| --- | --- | --- |
| `asset` | str | Ticker. e.g. `BTC`, `ETH`, `SOL`, `USDC`, `USD` |
| `quantity` | float | Units held |
| `average_cost` | float | Average cost per unit (in `currency`) |
| `currency` | str | Quote currency for `average_cost` and `current_price`. Typically `USD` |
| `current_price` | float | Current price per unit |
| `price_source` | str | Free-text source label (e.g. `coinbase`, `manual`, `coingecko`) |
| `notes` | str | Free-text notes |

Stablecoins (`USDT`, `USDC`, `BUSD`, `DAI`, `FRAX`, `TUSD`, `GUSD`,
`PYUSD`, `LUSD`, `USDD`) and cash tickers (`USD`, `EUR`, `GBP`, `JPY`,
`CHF`, `CAD`, `CASH`) are recognised automatically — they are NOT
shocked in drawdown scenarios.

### Example (do not commit your real version)

```csv
asset,quantity,average_cost,currency,current_price,price_source,notes
BTC,0.5,30000,USD,80000,manual,
ETH,5,2000,USD,3000,manual,
SOL,50,40,USD,140,manual,
USDC,5000,1.0,USD,1.0,manual,treasury
USD,2000,1.0,USD,1.0,manual,checking
```

## How to run

```bash
streamlit run streamlit_app.py
```

The Portfolio Risk section sits below the Research Dashboard. Six
tabs:

1. **Status** — schema validation, row count, current risk class,
   warnings.
2. **Positions** — your enriched positions with `position_value`,
   `cost_basis`, `unrealized_pnl`, `unrealized_pnl_percent`,
   `portfolio_weight`.
3. **Summary** — total market value, cost basis, PnL, largest
   position + weight, crypto exposure, stablecoin exposure, asset
   count.
4. **Drawdown scenarios** — what happens to portfolio value under
   -10 % / -20 % / -30 % / -50 % crypto shocks. Stablecoins/cash
   held flat.
5. **BTC baseline** — BTC weight, non-BTC weight, concentration
   description vs BTC buy-and-hold baseline.
6. **Risk + recommendation** — locked risk class + locked
   recommendation phrase + safe next actions.

## What the risk categories mean

The thresholds are **locked** in `src/portfolio_risk.py`. They
were chosen before any user data was loaded; they will not be
tuned to make a portfolio look better.

| Class | Trigger |
| --- | --- |
| `UNKNOWN` | No data, or zero market value |
| `EXTREME` | Any single position > 80 % |
| `HIGH` | Any single position > 60 %, OR crypto exposure > 90 % |
| `MODERATE` | Any single position > 40 % |
| `LOW` | Otherwise |

## Recommendation vocabulary (locked)

The dashboard emits **only one** of these phrases per render:

* `hold risk steady` — risk class is LOW or MODERATE
* `review concentration` — risk class is HIGH
* `reduce concentration` — risk class is EXTREME
* `data missing` (category)
* `do nothing until data is complete` (action)

A unit test enforces that no phrase contains "buy", "sell", "trade",
"open position", "submit order", "place order", "connect broker", or
"go long/short". The dashboard's job is to describe exposure, not to
tell you what to trade.

## What the dashboard does NOT do

* Does not place orders.
* Does not connect to any broker (Kraken or otherwise).
* Does not enter your portfolio data into any external service.
* Does not run paper trading.
* Does not run live trading.
* Does not generate strategy signals.
* Does not write your portfolio file (you supply it; it stays local
  and gitignored).
* Does not call any external API. All computations are local.
* Does not change the safety lock.

## Safety rules

* **Your portfolio CSV must never be committed.** `.gitignore` lists
  `data/portfolio_holdings.csv` and `data/portfolio_holdings_*.csv`
  explicitly. If you accidentally stage it, `ci_safety_check.py`
  already blocks generated/personal files in tracked source.
* **Do not paste API keys into the dashboard.** No input field
  accepts them and the helper module reads no environment variables.
* **Do not interpret the dashboard as financial advice.** It is a
  concentration / scenario calculator. Real decisions need
  independent judgement.

## Programmatic access

For notebooks or unit tests, the helper functions are read-only:

```python
from src import portfolio_risk as pr

pr.load_portfolio_holdings()                # (DataFrame, warning|None)
pr.validate_portfolio_schema(df)            # SchemaStatus dataclass
pr.calculate_position_values(df)            # enriched DataFrame
pr.calculate_portfolio_summary(df)          # dict of aggregates
pr.calculate_drawdown_scenarios(df)         # DataFrame of scenarios
pr.compare_to_btc_baseline(df)              # dict
pr.classify_portfolio_risk(summary)         # str ∈ RISK_CLASSES
pr.generate_risk_recommendation(summary, scenarios)  # dict
pr.get_portfolio_risk_dashboard_state()     # one-call dashboard state
```

All helpers fail soft: missing files return empty DataFrames and a
warning string; the dashboard never crashes.
