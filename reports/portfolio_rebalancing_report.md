# Portfolio rebalancing strategy — research verdict

This is the closure document for **research strategy 10 — locked-weight,
monthly portfolio rebalancing allocator**, the project's first strategy
research family after the universe-selection branch concluded that
portfolio rebalancing was the only universe that fit the existing
engine without leverage, shorting, paid data, or broker integration.

> **Verdict: FAIL.** **3 of 5 locked rebalancing-specific gates
> passed.** Sharpe is 0.50 vs BTC's 0.52 (well within the locked
> 0.10 gap, PASS). Drawdown is -55.34 % vs BTC's -66.12 % — only
> **+10.78 pp** tighter, where the locked gate requires **≥ 15 pp**
> tighter (FAIL). The strategy beats the random fixed-weight placebo
> on drawdown (PASS) but **loses on return** — landing in the
> **45th percentile** of placebos, with the median random allocation
> returning +72.7 % (FAIL). 56 rebalances over 14 OOS windows clears
> the ≥ 24 gate (PASS). Per the rules: parameters were NOT retuned;
> the locked weights and locked PASS criteria stand. Execution
> remains locked.

## 1. Objective

Test whether a fixed-weight, monthly-rebalanced long-only portfolio
(60 % BTC + 30 % ETH + 10 % cash) delivers a measurable risk-adjusted
improvement over BTC buy-and-hold — defined explicitly as comparable
Sharpe AND materially tighter drawdown — without using leverage,
shorts, paid data, or any broker integration.

## 2. Why this strategy universe was chosen

The strategy-universe-selection branch (merged at
`v0.4-universe-selection-locked`) ranked four candidate universes:
Forex, Crypto spot, Crypto derivatives / perps, and Portfolio
rebalancing / risk allocation. Portfolio rebalancing was
**RECOMMENDED** because:

* It fits the existing `portfolio_backtester` end-to-end with no
  new data, no new broker, no API keys.
* The locked PASS criteria explicitly NOT-"beat BTC outright on
  return" — that bar is what exhausted every prior long-only branch.
* Lowest overfitting risk of the four universes (fixed weights,
  no optimiser, no fitting to historical performance).

The other three universes were NOT_NOW (Forex needs broker policy
change; derivatives need paid data + perp backtester) or REJECTED
(Crypto spot — eight prior failed branches).

## 3. Strategy config (LOCKED — never tuned)

| Parameter | Locked value |
| --- | --- |
| BTC/USDT weight | 0.60 |
| ETH/USDT weight | 0.30 |
| Cash bucket | 0.10 (unallocated; the backtester treats it as cash) |
| Rebalance frequency | monthly |
| Long-only | yes |
| Leverage | none |
| Shorting | none |
| Optimiser | none — fixed weights |
| Lookahead | none — fixed weights are by definition lookahead-free |

Missing-asset handling: if BTC OR ETH is absent at the rebalance
bar, the missing leg's risk weight redistributes onto the surviving
risk leg; the cash bucket is preserved. If both are absent, the
strategy returns 100 % cash. (Documented in
`src/strategies/portfolio_rebalancing_allocator.py` and asserted by
unit tests.)

## 4. Assets used

* BTC/USDT (Binance public 1d cache, 2022-04-01 → 2026-05-09).
* ETH/USDT (Binance public 1d cache, 2022-04-01 → 2026-05-09).
* No new data sources. No paid APIs. No API keys.

## 5. Data window

Full window: **2022-04-01 → 2026-05-09** (~1500 days). Fees and
slippage from the existing `PortfolioBacktestConfig` defaults
(0.10 % fee per side, 0.05 % slippage per side, monthly rebalance,
next-bar-open execution).

## 6. Benchmarks

BTC buy-and-hold, ETH buy-and-hold, equal-weight BTC/ETH basket,
plus a 20-seed random fixed-weight placebo (each placebo seed picks
`(w_btc, w_eth)` from `Uniform(0, 1)` with `w_eth ≤ 1 - w_btc`).
The placebo shares the strategy's monthly cadence and long-only
constraint; the only difference is the choice of weight vector.

## 7. Full-window result

```
portfolio_rebalancing_allocator   return = +66.69 %   max DD = -55.34 %   sharpe = 0.499
BTC_buy_and_hold                  return = +74.84 %   max DD = -66.12 %   sharpe = 0.522
ETH_buy_and_hold                  return = -32.55 %   max DD = -71.74 %   sharpe = 0.207
equal_weight_basket               return = +21.15 %   max DD = -67.34 %   sharpe = 0.360
```

* Strategy lagged BTC b&h on return by 8 pp.
* Strategy beat the equal-weight basket by 45 pp.
* Sharpe is **0.023 below** BTC's — well within the locked 0.10 gap.
* Drawdown is 10.78 pp tighter than BTC's — below the locked
  15 pp gate.

## 8. Walk-forward result (14 OOS × 90 days)

```
window  oos_window               strat   BTC    Δsharpe  DD_15pp_tighter
  1     2022-09 → 2022-12        -10.00  -12.84  -0.27         no
  2     2022-12 → 2023-03        +53.58  +67.41  -0.25         no
  3     2023-03 → 2023-06        +10.37  +12.54  -0.09         no
  4     2023-06 → 2023-09        -12.59  -12.75  -0.19         no
  5     2023-09 → 2023-12        +50.26  +65.04  -0.37         no
  6     2023-12 → 2024-03        +47.63  +54.29  +0.04         no
  7     2024-03 → 2024-06         -0.05   -0.50  +0.03         no
  8     2024-06 → 2024-09        -17.61  -10.40  -0.77         no
  9     2024-09 → 2024-12        +63.73  +73.20  -0.22         no
 10     2024-12 → 2025-03        -29.23  -20.48  -0.93         no
 11     2025-03 → 2025-06        +28.67  +28.45  -0.23         no
 12     2025-06 → 2025-09        +27.21   +9.55  +1.44         no
 13     2025-09 → 2025-12        -21.17  -20.70  +0.07         no
 14     2025-12 → 2026-03        -26.16  -24.39  -0.25         no
```

* Total rebalances across 14 windows: **56** (clears the ≥ 24 gate).
* `sharpe_within_010` was True in 5 of 14 windows.
* `drawdown_15pp_tighter` was **True in 0 of 14 windows** — the
  rebalancer mostly tracks BTC's drawdown rather than tightening
  it.

## 9. Placebo result (20 seeds)

```
strategy        return = +66.69 %   DD = -55.34 %
placebo median  return = +72.72 %   DD = -56.69 %
return percentile     45th (strategy lost the median return contest)
drawdown percentile   55th (strategy beat the median drawdown by 1.3 pp)
beats placebo return    : False
beats placebo drawdown  : True
```

The placebo distribution explains the FAIL. Random fixed-weight
allocations skewed heavier-on-BTC (e.g. seed 4 drew `w_btc=0.94,
w_eth=0.03` → return +105.3 %; seed 10 drew `w_btc=0.96, w_eth=0.01`
→ return +107.6 %) consistently outperformed the locked 0.60/0.30
mix on raw return. ETH dragged the strategy down — ETH lost 32.5 %
over the window, so the 0.30 ETH allocation actively reduced
return. Random allocations that randomly under-weighted ETH won.

The drawdown axis is closer: the strategy's -55.34 % DD beat 11 of
the 20 placebos, so 55th percentile.

## 10. Scorecard verdict

| # | Gate | Threshold | Result | Pass |
| - | --- | --- | --- | --- |
| 1 | Sharpe within 0.10 of BTC | \|Δ\| ≤ 0.10 | -0.023 | PASS |
| 2 | Max drawdown ≥ 15 pp tighter than BTC | strat - btc ≥ 15 | +10.78 pp | **FAIL** |
| 3 | Beats placebo MEDIAN return | strat > median | +66.7 % vs +72.7 % | **FAIL** |
| 4 | Beats placebo MEDIAN drawdown | strat > median | -55.3 % vs -56.7 % | PASS |
| 5 | ≥ 24 rebalances total | total ≥ 24 | 56 | PASS |

**Checks passed: 3 of 5. Verdict: FAIL.**

## 11. What passed

* **Risk-adjusted return is comparable to BTC.** Sharpe 0.499 vs
  BTC's 0.522 — gap of 0.023, well inside the locked 0.10
  threshold.
* **Mechanics are clean.** 56 rebalances over 14 OOS windows; no
  lookahead; no leverage; no shorting.
* **Beat the placebo on drawdown.** The locked 60/30/10 mix
  delivers a tighter max drawdown than the median random allocation
  drew. Modest evidence that the locked weights chose a
  better-than-random risk profile.

## 12. What failed

* **Drawdown improvement insufficient.** The locked criterion
  asks for ≥ 15 pp tighter than BTC; the strategy delivered +10.78 pp.
  Adding 30 % ETH (which had a -71.7 % drawdown) cancelled most of
  the diversification benefit the 10 % cash bucket provided. The
  strategy doesn't get points for "tighter than BTC" — only for
  "tighter than BTC by at least 15 pp".
* **Lost the placebo return contest.** ETH was a structural drag in
  this 4-year window (-32.5 % over the period), so any allocation
  that happened to under-weight ETH won. The random allocator,
  being uniformly distributed in `(w_btc, w_eth)` space, sampled
  high-BTC / low-ETH portfolios about half the time and they
  consistently outperformed the 0.60/0.30 mix.

## 13. Limitations

* **Single sample window.** 4 years (2022-04 → 2026-05) is one
  realisation of a regime that happened to be BTC-favourable AND
  ETH-unfavourable simultaneously. A different period might flip
  the placebo verdict — but we do not get to retune to that period.
* **Locked weights are not optimised.** A 60/30/10 mix was specified
  in the universe-selection report; we did not test 70/20/10 or
  80/10/10 because that would be retuning. (Doing so would also
  defeat the point — the next branch would be just one of an
  infinite family of weight choices.)
* **No lookahead, no fit.** The strategy is fixed-weight by design.
  The placebo failure is therefore a statement about ETH's
  performance in this window, not about overfitting.

## 14. Whether this justifies further research

**No** — at least not as more weight-mix experiments on this
universe.

The honest finding is that **a fixed 60/30/10 mix did not deliver
the locked risk-adjusted improvement on the available 4-year sample
because ETH was a structural drag**. Two reasonable readings:

1. **The locked PASS criteria were too tight for this asset mix.**
   The 15 pp DD gate was chosen before seeing data; in practice
   adding 30 % ETH only tightens drawdown by ~10 pp. We do not get
   to widen the gate after seeing results.
2. **The asset mix would need to change** to clear the gates — but
   that is not "rebalancing research", it's weight optimisation, and
   it would be retuning.

Either reading is honest evidence that **the rebalancing universe
under these locked criteria does not pass on this sample**.

## 15. Whether this justifies paper trading

**No.** Paper trading remains blocked at the safety_lock and
strategy_registry layers regardless of this verdict. Even if the
verdict had been PASS, the rule on this project is that paper
trading remains disabled until a separate, explicit approval step
plus independent review — neither of which has happened.

The strategy registry entry for `portfolio_rebalancing` carries
`paper_trading_allowed = False` and `live_trading_allowed = False`,
and the test `test_no_strategy_is_paper_or_live_allowed` enforces
that they remain false on every branch.

## 16. Exact next step

1. **Archive this branch as a FAIL.** Do not merge it into `main`
   as a live strategy. The registry already points the family at
   this branch and at this report; that is sufficient evidence.
2. **Do not open a "weight tuning" research branch.** That would
   be retuning the locked rules, exactly the failure mode every
   prior branch report warned against. The locked weights came
   from the universe-selection report; changing them is out of
   scope.
3. **Re-evaluate from the universe-selection table.** With
   portfolio rebalancing now FAIL on this sample, the remaining
   universes are the NOT_NOW pair (Forex, Crypto derivatives) —
   each of which needs a prerequisite that the user has explicitly
   excluded (broker integration, paid data). The honest conclusion
   is therefore:
4. **Stop opening new strategy branches.** The engine's value is
   now its honesty about what doesn't work. Use it for risk
   reporting, archived-verdict tracking, drift snapshots, and the
   portfolio-risk dashboard — all of which are merged on `main`.

This is the first strategy research branch after universe selection,
and it has FAILED on the locked rebalancing-specific scorecard. The
verdict stands. Execution remains locked.
