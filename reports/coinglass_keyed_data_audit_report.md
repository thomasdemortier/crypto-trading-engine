# CoinGlass keyed data-depth audit

This is the closure document for **research strategy 9 — keyed
CoinGlass data-depth verification**. It is a **data-depth audit
only**: no strategy was built, no backtest, no broker integration,
no order placement, no Kraken connection. Execution remains locked.

## 1. Executive summary

> **Verdict: INCONCLUSIVE** in the environment this branch was first
> shipped from — `COINGLASS_API_KEY` was not set, so all 16 probes
> recorded `AUTH_FAILED` and the strategy decision rule could not be
> evaluated. The audit pipeline itself works (315/315 tests passing,
> CSV well-formed, no key-leak risk). To get a GO / NO-GO verdict
> the user must set their CoinGlass API key in the shell and re-run:
>
> ```bash
> export COINGLASS_API_KEY=...
> python main.py audit_coinglass_keyed_data
> ```
>
> The decision rule is locked: **GO** if at least 2 priority field
> classes return `>= 1460 days` for **both** BTC and ETH; otherwise
> **NO-GO**. No retuning.

## 2. Why this audit was needed

The prior `research/strategy-8-paid-data-decision-audit` branch named
**CoinGlass** as the cheapest realistic first paid-data test ($29/mo
Standard tier, broadest field coverage on the public marketing page).
But that audit was **desk research only** — it could not verify the
historical depth CoinGlass actually serves through their paid API.

Three long-only allocator branches in a row already failed on the
same pattern (tighter drawdown, beats placebo, loses to BTC). The
public-data audit found the OI / liquidation / long-short stack is
all `FAIL` on free endpoints (≤ 30–500 days). Without keyed
verification of CoinGlass's actual depth, opening another strategy
branch would just repeat the failure pattern. This audit is the
last gate before any further alpha research.

## 3. API key handling and safety

The module is built around a single rule: **the key value is read
from `COINGLASS_API_KEY` at runtime and never leaves the request
header.** Specifically:

* The env-var name is split across constants
  (`_API_KEY_ENV_NAME = "COINGLASS" + "_" + "API_KEY"`) so no
  literal API-key string sits next to a value assignment in source.
* The reader function `_read_api_key()` returns the raw value to
  the audit orchestrator only; the public predicate
  `api_key_present()` returns a `bool`, never the value.
* The HTTP helper takes the key as a positional argument and uses
  it ONLY to populate the `CG-API-KEY` request header. The key is
  not in the URL, not in logs, not in error messages.
* Every audit row has an `api_key_present` boolean column —
  truthy or falsy — but the key value itself is never written to
  any column.
* `_assert_key_not_in_frame()` runs a final cell-by-cell scan over
  the audit DataFrame before the CSV is written; if the key value
  appears anywhere, it raises rather than persisting.
* Three offline tests verify the key never reaches CSV / log / stdout
  even when injected via `monkeypatch.setenv` with a sentinel value:
  `test_api_key_value_never_in_csv`,
  `test_api_key_value_never_logged`,
  `test_api_key_value_never_printed`.
* The CI safety gate (10/10 PASS) is unchanged.

## 4. Datasets tested

The audit hits **16 probes** on CoinGlass v3 — 8 dataset shapes per
asset, BTC and ETH. Endpoint paths reflect the v3 documentation as of
the audit date; if any have been renamed in v4 the audit records
`ENDPOINT_NOT_AVAILABLE` rather than crashing.

| # | Dataset | Asset | Path |
| - | --- | --- | --- |
| 1, 9 | open_interest_binance | BTC, ETH | `/api/futures/openInterest/ohlc-history` |
| 2, 10 | open_interest_aggregated | BTC, ETH | `/api/futures/openInterest/ohlc-aggregated-history` |
| 3, 11 | liquidations | BTC, ETH | `/api/futures/liquidation/coin-history` |
| 4, 12 | long_short_ratio_global | BTC, ETH | `/api/futures/global-long-short-account-ratio/history` |
| 5, 13 | top_long_short_position_ratio | BTC, ETH | `/api/futures/top-long-short-position-ratio/history` |
| 6, 14 | funding_binance | BTC, ETH | `/api/futures/funding-rate/exchange-history` |
| 7, 15 | funding_oi_weighted | BTC, ETH | `/api/futures/funding-rate/oi-weight-history` |
| 8, 16 | basis_binance | BTC, ETH | `/api/futures/basis/history` |

Probe collapse rule into priority field classes (the GO gate uses
classes, not individual probes):

| Field class | Probes per asset |
| --- | --- |
| `open_interest` | 2 (Binance + aggregated) |
| `liquidations` | 1 |
| `long_short_ratios` | 2 (global account + top-trader position) |
| `funding` | 2 (Binance + OI-weighted) |
| `basis_or_premium` | 1 (Binance) |

A field class counts as PASS if **any** probe under it returns
`>= 1460 days`.

## 5. Coverage results — present run

| | Count |
| --- | ---: |
| API key present | **no** (this branch's first run) |
| datasets tested | 16 |
| `PASS` | 0 |
| `WARNING` | 0 |
| `FAIL` | 0 |
| `INCONCLUSIVE` | 0 |
| `AUTH_FAILED` | 16 |
| `RATE_LIMITED` | 0 |
| `ENDPOINT_NOT_AVAILABLE` | 0 |
| BTC PASS field classes | 0 — `[]` |
| ETH PASS field classes | 0 — `[]` |

Every probe correctly recorded `AUTH_FAILED` because the
`COINGLASS_API_KEY` env var is not set in the current shell. The CSV
shape is correct, the rate-limiting between probes works, and no
key-leak occurred (no key to leak). The audit pipeline is shipped
green; the only thing missing is a real key.

## 6. Which fields passed `>= 1460 days`

**Cannot answer yet.** The user must set their CoinGlass API key and
re-run. When they do, this section becomes a real table.

## 7. Which fields failed

**Cannot answer yet.** Same reason.

## 8. Which fields were inconclusive

In the current run, **all 16** probes are `AUTH_FAILED` →
`INCONCLUSIVE` for the strategy decision rule. The verdict will
collapse to GO / NO-GO once the key is set.

## 9. Whether CoinGlass is worth keeping beyond one month

**Decision deferred to the keyed re-run.** The structure is:

1. The user sets `COINGLASS_API_KEY` (Standard plan, $29/mo) and runs
   `python main.py audit_coinglass_keyed_data`.
2. The audit produces a real per-field decision table.
3. If the keyed verdict is **GO** → a one-month subscription is
   justified; the next strategy branch can use the verified fields.
4. If **NO-GO** → cancel after one month, archive this branch as a
   FAIL outcome, accept BTC buy-and-hold as baseline.
5. If **INCONCLUSIVE on a real key** (e.g. tier-gated endpoints
   return 401/403 / empty arrays) → the wrong tier was bought.
   Either upgrade, or accept the answer.

## 10. Whether strategy work is justified

**Not yet.** Justified only after a keyed re-run produces `GO`.

> **If GO**: strategy work is justified, but execution remains
> locked. The next branch should be strategy-only research using
> the verified fields. Independent review is required before any
> trading.
>
> **If NO-GO**: stop buying data. Accept BTC buy-and-hold as
> baseline. Use the engine for risk reporting and research hygiene.
>
> **If INCONCLUSIVE on real key**: rerun with a different tier or
> different asset codes; if it stays INCONCLUSIVE, treat as NO-GO.

## 11. Exact next step

```bash
export COINGLASS_API_KEY=<your-key>      # never commit this value
python main.py audit_coinglass_keyed_data
```

Read the `BTC PASS field classes` and `ETH PASS field classes` lines
in the CLI output. Those two counts drive the verdict:

* **Both ≥ 2**: open the next branch and start a strategy on the
  verified fields. Locked scorecard, no retuning, same PASS bar as
  every prior branch.
* **Either < 2**: archive this branch as the closure document.
  Cancel the CoinGlass subscription before the next billing cycle.
  Accept BTC buy-and-hold as the production baseline.

The decision rule is **already encoded** in
`coinglass_keyed_data_audit.summarise()` and printed by the CLI —
there is no human judgement step beyond reading the verdict line.

This is a CoinGlass keyed data-depth audit only. Do not build a
strategy on this branch. Keep execution locked. Do not merge this
branch into `main` — archive it as decision evidence, the same way
the failed research branches are archived.
