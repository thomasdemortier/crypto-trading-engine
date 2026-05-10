# FX + crypto data source audit

A read-only, deterministic audit that probes free public Forex and
crypto data sources, records what is actually reachable today, and
classifies each source under a locked coverage threshold. The point
is to decide — **before** opening a new research branch — whether
the underlying data lane is strong enough to support strategy work.

## What it is

`src/fx_crypto_source_audit.py` is a pure-Python probe + ledger. It
issues fail-soft HTTP probes to public endpoints (ECB SDMX,
Frankfurter, LBMA, Yahoo Finance, Binance, Bybit, OKX, Deribit,
Kraken) and records, per source, the actual coverage window the
public path returns today. Sources that historically were free but
now require a key (OANDA, IG, Stooq, Dukascopy, FRED) are recorded
as `INCONCLUSIVE` with the key requirement noted.

The audit emits one of four locked decision statuses per row:

* **PASS** — coverage ≥ 1460 days (4 years), free, reachable.
* **WARNING** — coverage 365–1459 days; usable but limited.
* **FAIL** — coverage < 365 days, snapshot-only, or endpoint
  unreachable on this run.
* **INCONCLUSIVE** — source exists but the free public path is no
  longer available (key required, captcha-gated, broker-bound).

## What it is NOT

* Not a strategy.
* Not a backtest.
* Not a broker integration.
* Not an API-key consumer. The audit issues unauthenticated GETs
  only; a unit test enforces that the source contains no API-key
  reads, no broker SDK imports, no Kraken private endpoints, no
  order-placement strings, and no paper / live enablement.
* Not a recommendation to trade. The next-branch recommendation
  output is research-only; promotion to paper or live trading would
  require a separate broker policy decision that this branch does
  not make.

## How to use it

CLI:

```
python main.py audit_fx_crypto_sources
```

This writes `results/fx_crypto_source_audit.csv` (gitignored) with
the locked 14-column schema, prints a per-market summary, and
returns viability flags.

Programmatic:

```python
from src import fx_crypto_source_audit as audit

df = audit.run_audit()                  # DataFrame, locked schema
summary = audit.summarise(df)           # dict: fx_pass, crypto_pass, ...
status = audit.classify_coverage(...)   # PASS / WARNING / FAIL / INCONCLUSIVE
```

`run_audit(http_get=...)` accepts a custom HTTP function for tests,
so the test-suite version uses synthetic payloads and never hits
the network.

## Locked schema

Output CSV columns (in order):

```
market, source, asset, field_type, endpoint_or_source,
requires_api_key, free_access, actual_start, actual_end,
coverage_days, granularity, usable_for_research,
decision_status, notes
```

* `market` ∈ {`forex`, `crypto`}.
* `field_type` ∈ {`reference_rate`, `ohlcv`, `funding`,
  `basis_or_premium`, `open_interest`, `order_book_snapshot`}.
* `coverage_days` is the integer count of days actually returned by
  the public endpoint on this run. Zero for INCONCLUSIVE rows.
* `usable_for_research` is a boolean derived from
  `decision_status in {PASS, WARNING}`.
* `decision_status` is one of the four locked statuses.

## Locked thresholds

```python
PASS_DAYS    = 1460   # 4 years
WARNING_DAYS = 365    # 1 year
```

A schema test and a threshold test in
`tests/test_fx_crypto_source_audit.py` enforce that neither moves
without an explicit edit + test update.

## Safety invariants (enforced by tests)

`tests/test_fx_crypto_source_audit.py` regex-greps the source file
for forbidden patterns and fails the suite if any appear:

* No broker SDK imports (`ccxt`, `binance.client`, `oandapyV20`,
  `python-iglib`, `alpaca`, etc.).
* No API-key environment reads (`os.getenv` patterns referencing
  `*_KEY`, `*_SECRET`, `*_TOKEN`).
* No Kraken private endpoints (`/0/private/`).
* No order-placement strings (`place_order`, `create_order`,
  `submit_order`, `new_order`, `Buy`, `Sell` as enum values).
* No paper / live trading enablement (`enable_live`,
  `enable_paper`, `start_trading`, `start_paper`).
* No imports of strategy modules — the audit never invokes
  strategy code.

Plus a `.gitignore` sanity check: `results/*.csv` must be ignored
so the per-run CSV never gets committed.

## Decision rule for next research branch

The audit produces a viability verdict per market:

* **Forex viable** = at least one PASS row covering ≥ 1460 days for
  a fiat pair OR for gold (XAU/USD).
* **Crypto viable** = at least one PASS row covering ≥ 1460 days
  for a major (BTC or ETH) on OHLCV.

If both are viable on data, the recommendation prefers whichever
universe has **not** already been exhaustively tested by prior
branches. Crypto has been tested ten times and FAILed each time on
this engine, so on a tie the recommendation goes to FX.

The recommended-next-branch sentence in the report is locked to be
research-only — no broker, no paper, no live language. An update to
the recommendation requires an explicit edit; the report is the
single source of truth.

## Outputs

* `results/fx_crypto_source_audit.csv` — gitignored, regenerated
  each run.
* `reports/fx_crypto_source_audit_report.md` — human verdict,
  PASS / WARNING / FAIL tables, viability analysis, recommended
  next research branch with locked PASS criteria. Hand-written;
  not regenerated by the tool.
* This doc — explains what the module is and how to use it.

## Honesty rule

If a re-run shifts a source from PASS to FAIL (e.g. an endpoint
goes down), the report verdict is not auto-rewritten. The CSV
reflects the current run; the report reflects the verdict at the
time it was authored. A material change in source availability
warrants a new audit branch with a fresh report — not a silent
edit of the existing one.
