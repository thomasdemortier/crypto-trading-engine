# Paid data vendor decision audit

This is a **vendor / data decision audit only**. No strategy was built.
No backtest was run. No broker integration. No API keys. No paid data
was downloaded. Execution remains locked.

> **Recommendation: do not code another strategy yet.** Three of the
> 11 vendors reviewed clear the PASS_CANDIDATE bar (CoinGlass,
> CryptoQuant, Velo Data) but **none of those clearances has been
> verified by us with real API access**. The next concrete step is
> to spend ~$29–$99 on **one** monthly trial — CoinGlass is the
> cheapest verifiable shortlist entry — and re-run the existing
> `positioning_data_audit` against the keyed endpoints. Until that
> re-audit lands, treat the paid lane as INCONCLUSIVE in operational
> terms even though the desk research flags it as a PASS_CANDIDATE.

## 1. Executive summary

| | |
| --- | --- |
| Vendors reviewed | 11 |
| `PASS_CANDIDATE` | 3 — CoinGlass, CryptoQuant, Velo Data |
| `WATCHLIST` | 3 — Glassnode, Coinalyze, Santiment |
| `SALES_REQUIRED` | 3 — Kaiko, Laevitas, Amberdata |
| `DOCS_GATED` | 0 |
| `INCONCLUSIVE` | 0 |
| `FAIL` | 2 — TokenTerminal, TheTie |
| Top recommendation | **CoinGlass** — lowest visible plan ($29/mo), broadest field set, most-cited public docs |

Key call: `PASS_CANDIDATE` here means "the vendor's marketing pages
say they sell what we need at a price we can see". It does **not**
mean we have verified historical depth via their actual API. That
verification is the explicit next step before any further strategy
code is written.

## 2. Why this audit was needed

The bar to research was as follows. Three branches in a row produced
the same failure pattern:

| Branch | Result | Failure |
| --- | --- | --- |
| `research/strategy-4-drawdown-targeted-btc` | FAIL (5/7 checks) | beats placebo, tighter DD; loses to BTC in 8/14 windows |
| `research/strategy-6-funding-basis-carry` | FAIL (5/9 checks) | beats placebo, tighter DD; loses to BTC in 11/14, basket in 9/14 |
| `research/strategy-7-relative-value-btc-eth` | FAIL (4/9 checks) | beats placebo, tighter DD; loses to BTC in 10/14, basket in 10/14 |

All three are **defensive long-only allocators on the BTC + ETH
universe**. They reduce drawdown 12–29 pp tighter than BTC and clear
the random-bucket placebo, but they cannot beat BTC in a 4-year
BTC-favourable sample. The pattern is now structural.

The `research/strategy-5-paid-positioning-data-audit` branch found
that **public** OI / liquidations / long-short ratios are all `FAIL`
on free endpoints (≤ 30–500 days). That was the gap: every long-only
strategy we have built relies on price + funding signals because
that's all the public data offered with multi-year depth. Without
positioning information we cannot test the hypotheses that might
actually beat BTC.

The honest question now is: **is paying for that data justified?**

## 3. What failed before — and why long-only allocation research is exhausted

Long-only allocators on BTC + ETH all face the same constraint: any
defensive cap reduces BTC exposure during the bull regime, and the
4-year sample is BTC-favourable. The strategies that survived to
research could only express two positions ("more BTC" or "less BTC")
and neither beats BTC consistently. The next idea would have to:

* test a market-neutral pair (long ETH / short BTC or vice versa) —
  but the existing backtester does not support short legs and the
  spec forbids faking it; or
* use **positioning data** (OI buildup, liquidation cascades,
  long/short ratio extremes, funding+basis joint signals on
  multi-year depth) to time BTC entries differently from "200d MA
  cross" — but the public endpoints do not have that depth.

The second path is cheaper and bounded — buy a one-month trial of
the cheapest vendor that ships the right field set, re-audit, then
decide whether to write the strategy. This audit identifies that
vendor.

## 4. Vendors reviewed

The full ledger is in
[`results/paid_data_vendor_audit.csv`](../results/paid_data_vendor_audit.csv)
(generated, not tracked). 11 vendors:

| # | Vendor | Category |
| - | --- | --- |
| 1 | CoinGlass | derivatives_aggregator |
| 2 | CryptoQuant | onchain_and_derivatives |
| 3 | Velo Data | derivatives_focused |
| 4 | Glassnode | onchain_and_derivatives |
| 5 | Kaiko | institutional_data_platform |
| 6 | Laevitas | derivatives_options_focused |
| 7 | Amberdata | institutional_data_platform |
| 8 | Coinalyze | derivatives_aggregator |
| 9 | TokenTerminal | protocol_financials |
| 10 | Santiment | sentiment_and_onchain |
| 11 | TheTie | sentiment_for_institutions |

## 5. Data types compared

The audit ledger records six priority fields per vendor:
`historical_open_interest`, `historical_liquidations`,
`historical_long_short_ratios`, `historical_funding`,
`historical_basis`, `exchange_flows_reserves`. A vendor needs `>= 2`
of these as `yes` to clear the `PASS_CANDIDATE` bar (the locked rule
in `paid_data_vendor_audit.classify_decision`).

| Vendor | OI | Liq | L/S | Fund | Basis | Flows |
| --- | :-: | :-: | :-: | :-: | :-: | :-: |
| CoinGlass | yes | yes | yes | yes | yes | no |
| CryptoQuant | yes | yes | limited | yes | limited | yes |
| Velo Data | yes | yes | yes | yes | yes | no |
| Glassnode | limited | limited | no | limited | no | yes |
| Kaiko | yes | yes | yes | yes | yes | limited |
| Laevitas | yes | limited | limited | yes | yes | no |
| Amberdata | yes | yes | limited | yes | yes | limited |
| Coinalyze | yes | yes | yes | yes | limited | no |
| TokenTerminal | no | no | no | no | no | no |
| Santiment | limited | no | no | no | no | yes |
| TheTie | unknown | unknown | unknown | unknown | unknown | unknown |

CoinGlass, Velo Data, Coinalyze, Kaiko, Amberdata, Laevitas all claim
the OI / liquidation / funding / basis stack we need. The split is
not "who has it" but "who has it at a price we can verify on a
non-enterprise plan".

## 6. PASS_CANDIDATE shortlist

| Vendor | Lowest visible plan | Confidence | Why it cleared |
| --- | ---: | --- | --- |
| **CoinGlass** | **$29 / mo** | medium | Public pricing, public docs, claims OI + liquidations + L/S + funding + basis at multi-year depth on Standard+ tier |
| CryptoQuant | $29 / mo | medium | Public pricing + docs; strongest on exchange flows + on-chain combined with OI / funding |
| Velo Data | $99 / mo | medium | Smaller / newer but explicitly derivatives-first with minute-level history; trial available |

All three have public pricing pages and public API docs. None have
been verified against their actual API for >= 1460-day depth — that
is what a paid trial buys us.

## 7. SALES_REQUIRED vendors

| Vendor | Why |
| --- | --- |
| Kaiko | enterprise contracts only, no public pricing |
| Amberdata | enterprise contracts only, no public pricing |
| Laevitas | API pricing not publicly listed; demo on request |

These are skipped unless the project has explicit institutional
budget. The strategy registry stays research-only and the safety
lock keeps `paper_trading_allowed = False` and
`live_trading_allowed = False` on every branch regardless of any
vendor decision.

## 8. WATCHLIST and FAIL

WATCHLIST (vendors with some useful data but pricing or fields
incomplete relative to a derivatives-positioning strategy):

* **Glassnode** ($29/mo Standard) — exposes a small subset on the
  cheapest tier; granular derivatives lives on Advanced+. The
  on-chain story is excellent but the prior strategies showed
  on-chain alone is not the gap.
* **Coinalyze** (free tier) — public free API exists but history
  depth on the free tier is shallow. Cheap to verify but the depth
  question is the open one.
* **Santiment** ($44/mo Pro) — strong on sentiment + on-chain
  flows, weak on OI / liquidations. The earlier
  `sentiment_overlay` branch already failed on this project.

FAIL:

* **TokenTerminal** — protocol financials (TVL / fees / revenue),
  not derivatives positioning. Wrong category.
* **TheTie** — institutional sentiment, docs gated, sentiment is not
  the bottleneck on this project.

## 9. Which data fields matter most for the next strategy

If the next strategy is going to be different from the three we just
failed, it needs to test a hypothesis that BTC price + 200d MA can't
already test. Best candidates, in priority order:

1. **Open interest changes through liquidation events.** "OI builds,
   funding turns sharply, then a cascade liquidation flushes the
   crowded side" is a classic crypto-derivatives pattern that
   requires multi-year OI + liquidation data to test. Public
   endpoints cap at 30 days — paid is the only way.
2. **Long/short ratio extremes as fade signals.** Same depth
   problem on public endpoints (≤ 30 days at Binance, ≤ 500 days at
   Bybit's account-ratio endpoint).
3. **Cross-venue OI dispersion.** Aggregated OI across exchanges is
   only available from CoinGlass-class vendors; single-venue OI
   does not test the same hypothesis.

A "funding alone" signal has already been tested twice on this
project (the original `derivatives_funding` branch and the
`funding_basis_carry` branch). It's not enough.

## 10. Recommended vendor shortlist

In recommended order:

1. **CoinGlass — $29/mo Standard tier as the first verification
   spend.** Highest field coverage at the lowest visible plan price.
   Run for one month; pull BTC + ETH OI history, liquidation
   history, long/short ratios, funding rates; re-run
   `positioning_data_audit` with the keyed endpoints; only THEN
   decide whether the field depth is sufficient.
2. **Velo Data — trial first.** If the CoinGlass data ships shorter
   history than claimed, Velo is the second pick. Their derivatives
   focus and minute-level history may be a better technical fit
   even at the higher price.
3. **CryptoQuant — Advanced tier $29/mo if the strategy needs
   exchange-flow data.** Their differentiator is exchange flows +
   reserves combined with funding/OI; that combination is not on
   CoinGlass.

Skip Kaiko / Amberdata / Laevitas unless institutional budget is
approved. They are excellent but out of scope for a research-only
project.

## 11. Estimated decision path

| Action | Cost | Owner check |
| --- | ---: | --- |
| Buy CoinGlass Standard | $29 / 1 mo | If after one month the data does not clear `>= 1460` days for BTC OI + liquidations, **cancel and stop**. |
| Optional: Velo Data trial | trial / $99 if extended | Only if CoinGlass falls short. |
| Optional: CryptoQuant Advanced | $29 / 1 mo | Only if the next strategy idea is exchange-flow centred. |
| Re-run `positioning_data_audit` against keyed endpoints | $0 | Already built on `research/strategy-5-paid-positioning-data-audit`. Add a `--key-source coinglass` mode. |
| Re-evaluate strategy work | $0 | If audit clears `PASS` on multi-year OI + liquidations, open the next strategy branch. Otherwise close the loop. |

Maximum committed spend before we know whether positioning data
can actually clear the bar: **$29**, one month. Anything beyond that
is conditional on the re-audit verdict.

## 12. Whether strategy work is justified now

**No.** Not until paid-data field depth is verified. Coding another
strategy on the same public data we already audited would just
repeat the failure pattern. Three strategies have already shown
that. The cheapest way to break the pattern is to spend $29,
verify, and then decide.

If the user is unwilling to spend that, the alternative is:

> **Accept BTC buy-and-hold as the baseline.** Use the engine for
> risk reporting, OOS falsification, and registry hygiene rather
> than alpha generation. Stop opening new long-only branches against
> the same universe and same window.

Either path is honest. The wrong path is to write strategy 9 against
the same public data and pretend the verdict will be different.

## 13. Exact next step

Pick exactly one and do nothing else until it lands.

1. **Recommended:** purchase **CoinGlass Standard** for one month
   (~$29 USD). Pull BTC + ETH OI history, liquidation history,
   long/short ratios, funding, and basis. Extend
   `src/positioning_data_audit.py` with a keyed-source path. Run
   the audit. **If the audit shows `PASS` (>= 1460 days for at
   least two priority fields), then — and only then — open the
   next strategy branch.**
2. **If unwilling to spend:** archive this branch as the closure
   document. State that further strategy research on this project
   is paused. Use the engine for risk reporting only. Treat BTC
   buy-and-hold as the production baseline.

This is research only. Execution remains locked.
