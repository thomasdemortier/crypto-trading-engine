# Strategy universe selection — research decision

This is a **research decision** branch — not a strategy build, not a
backtest, not an execution surface. It picks the next universe the
project should pursue **as research**, given everything we now know.

> **Recommended next research branch:**
> `research/portfolio-rebalancing-strategy-v1` — fixed-weight monthly
> rebalancing on BTC + ETH + USDC, benchmarked against BTC buy-and-hold
> and an equal-weight basket. PASS criteria deliberately NOT "beat
> BTC" — they are "comparable risk-adjusted return with materially
> lower drawdown". This is a **research branch**, not a trading
> branch.

## Ranked table

| Rank | Universe | Score | Decision |
| ---: | --- | ---: | --- |
| 1 | Portfolio rebalancing / risk allocation | **7.8** | **RECOMMENDED** |
| 2 | Crypto spot | 6.4 | **REJECTED** |
| 3 | Forex | 4.0 | NOT_NOW |
| 4 | Crypto derivatives / perps | 4.0 | NOT_NOW |

Score is the equal-weighted average of five integer 0–10 axes:
data, execution, edge, complexity, risk. Weights are locked at
equal — they are not tuned per universe. Tied scores break by
decision-status priority (RECOMMENDED > WATCHLIST > NOT_NOW >
REJECTED) so the actionable choice surfaces first.

## Per-universe assessment

### 1. Portfolio rebalancing / risk allocation — RECOMMENDED (7.8)

| Axis | Score | Why |
| --- | ---: | --- |
| Data | 9 | Uses the existing local 1d cache — no new data; every input has multi-year coverage. |
| Execution | 8 | Fits the existing `portfolio_backtester` directly; no broker required for research. |
| Edge | 4 | Not a high-return alpha class, but a real, measurable drawdown / Sharpe improvement vs unrebalanced BTC b&h. |
| Complexity | 9 | Lowest complexity of the four; reuses every existing module; smallest new test surface. |
| Risk | 9 | No leverage, no shorting, fixed weights — the lowest overfitting and blow-up risk of the four. |

**PASS criteria:** Sharpe within 0.10 of BTC b&h AND max drawdown
≥ 15 pp tighter than BTC AND beats placebo on both axes AND ≥ 24
rebalances over the OOS window. **Crucially, the bar is NOT "beat
BTC outright on return"** — that bar is what failed every prior
long-only branch. Rebalance research is judged on risk-adjusted
metrics by design.

**FAIL criteria:** Cannot tighten drawdown materially OR fails the
placebo OR introduces concentration that violates the locked
`portfolio_risk` thresholds (largest position > 80 %).

**Recommended next action:** open
`research/portfolio-rebalancing-strategy-v1`. Reuse
`portfolio_backtester`, `portfolio_risk` (already merged), and the
locked scorecard pipeline.

### 2. Crypto spot — REJECTED (6.4)

| Axis | Score | Why |
| --- | ---: | --- |
| Data | 9 | Binance public OHLCV / funding / mark / index all PASS at 1500+ days. |
| Execution | 7 | Fits `portfolio_backtester`; no new infrastructure. |
| Edge | 2 | **EXHAUSTED.** Eight branches failed against BTC b&h on this exact universe. |
| Complexity | 8 | Lowest build cost — every signal stack already exists in archived branches. |
| Risk | 6 | Long-only, no leverage; risk is opportunity cost vs BTC b&h, not blow-up. |

The score looks high because data and complexity are favourable,
but the **edge axis is dispositive**. Eight long-only branches
already failed — `drawdown_targeted_btc`, `funding_basis_carry`,
`relative_value_btc_eth`, `market_structure_vol_target`,
`sentiment_overlay`, `funding_only`, plus the original v1
single-asset and portfolio-momentum families. The free-data
re-audit (strategy 9) confirmed there is no untested signal class
on free public endpoints.

**Recommended next action:** **closed.** Do not open another
long-only research branch on this universe. Future research is
only justified if a paid positioning data plan is purchased AND a
strategy passes the locked scorecard against BTC b&h.

### 3. Forex — NOT_NOW (4.0)

| Axis | Score | Why |
| --- | ---: | --- |
| Data | 6 | Free multi-year OHLC available, but tick-level depth + verified spread modelling typically need paid data. |
| Execution | 3 | Would require a broker / demo integration; current engine policy forbids any broker. |
| Edge | 3 | Retail FX is one of the most-arbitraged retail markets; edge is structurally low. |
| Complexity | 4 | Moderate: new asset class, new pricing model, spread / swap accounting, new universe loader. |
| Risk | 4 | Leverage cultures + tight spreads + 24×5 sessions = elevated blow-up risk. |

**Recommended next action:** **defer**. The next research branch
on this universe would be `research/forex-data-audit-v1` — but
only if a broker integration policy change is approved separately.
**Out of scope under the current no-broker policy.**

### 4. Crypto derivatives / perps — NOT_NOW (4.0)

| Axis | Score | Why |
| --- | ---: | --- |
| Data | 4 | Public OI / liquidations / long-short ratios FAIL on every free venue (capped 30–500 days). |
| Execution | 3 | Requires a perp + margin backtester (not yet built) plus a future broker for any paper / live phase. |
| Edge | 5 | Best signal-class potential of the four — crowded longs + liquidation cascades are real patterns — but data-gated. |
| Complexity | 4 | Two prerequisite branches before a strategy: keyed data audit, then perp backtester; not cheap. |
| Risk | 4 | Leverage, funding, liquidation mechanics — the highest blow-up risk if executed badly. |

**Recommended next action:** defer until a paid positioning data
trial is purchased. The on-deck research branch is
`research/strategy-9-coinglass-keyed-data-audit` (already shipped,
currently INCONCLUSIVE pending `COINGLASS_API_KEY`) — or a
keyed re-audit on CryptoQuant / Velo Data.

## Why this ranking and not another

* **The edge axis dominates.** Crypto spot has high data and
  complexity scores but the strategy track record on this exact
  universe is conclusive: it does not beat BTC b&h with the data
  available. That makes it REJECTED regardless of composite score.
* **Portfolio rebalancing is the only universe where the engine
  can deliver a real, measurable improvement TODAY without changing
  scope.** No new data, no new backtester, no broker, no API keys,
  no policy change.
* **Both NOT_NOW universes need a prerequisite.** Forex needs a
  broker integration policy change. Derivatives needs paid
  positioning data plus a perp backtester. Either can move out of
  NOT_NOW if those prerequisites are met — but they are not the
  next branch.

## What "RECOMMENDED" means here

The recommendation is to **open a research branch**. It is not:

* not a recommendation to start trading
* not a recommendation to connect a broker
* not a recommendation to enable paper trading
* not a recommendation to buy or sell any asset
* not a recommendation to place any order

The locked recommendation vocabulary in
`strategy_universe_selection.FORBIDDEN_RECOMMENDATION_TOKENS` and a
unit test (`test_no_action_contains_forbidden_tokens`) enforce this
at the source level. A future edit that introduces "buy", "sell",
"place order", or "connect broker" into any
`recommended_next_action` will fail CI.

## Exact next step

```bash
git checkout main
git pull origin main
git checkout -b research/portfolio-rebalancing-strategy-v1
```

Inside that branch, the work is:

1. Define the rebalance policy (fixed weights, monthly cadence,
   slippage + fee model from `PortfolioBacktestConfig` defaults).
2. Build a strategy class with a `target_weights(asof, frames, tf)`
   contract that returns the locked weight vector.
3. Add a research orchestrator: full-window backtest + walk-forward
   + placebo (random allocation within the same weight buckets) +
   scorecard.
4. Use the **rebalancing-specific** PASS criteria — Sharpe within
   0.10 of BTC b&h AND max drawdown ≥ 15 pp tighter than BTC AND
   beats placebo on both axes AND ≥ 24 rebalances. Do NOT reuse
   the failed-allocator scorecard verbatim, because that scorecard
   demanded "beat BTC outright" which is the wrong bar for
   rebalancing.
5. If it PASSes, add the family to `strategy_registry` with
   `paper_trading_allowed = False` and `live_trading_allowed =
   False` like every other entry. Independent review required
   before any trading discussion. Execution remains locked.

If it fails, archive it with the same honesty applied to the
previous nine branches.

This is research-decision infrastructure only. No strategy. No
execution. Safety lock remains locked.
