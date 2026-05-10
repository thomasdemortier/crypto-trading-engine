# FX research dataset — v1

The `data/fx-research-dataset-v1` branch builds the first reproducible
FX research dataset for this repo. It is **data infrastructure only**.

> No strategy. No backtest. No broker. No API keys. No paper trading.
> No live trading. No order placement. No execution.

## Why this branch exists

The previous branch (`data/fx-crypto-source-audit-v1`, merged at tag
`v0.6-fx-crypto-source-audit-locked`) proved that free, public, no-key
FX data exists with multi-decade history. This branch is the next step
in the locked roadmap:

1. ~~Audit sources — done at `v0.6-fx-crypto-source-audit-locked`.~~
2. **Build clean FX dataset — this branch.**
3. Run data quality checks — next branch.
4. Build first FX strategy — only after the dataset is stable.

We do **not** jump to a strategy branch yet. A strategy on top of a
messy dataset wastes time fighting the data instead of testing the
hypothesis.

## Sources

Only sources that passed the audit at PASS-tier without keys:

| Source           | Pairs                      | Coverage      | Notes |
|:-----------------|:---------------------------|:--------------|:------|
| `ecb_sdmx`       | EUR/USD, EUR/GBP, EUR/JPY, EUR/CHF | 1999-→ (~9,986 days) | ECB official EUR-quoted reference rate; daily fix only, no intraday, no spreads, no volume. CSV at `https://data-api.ecb.europa.eu/service/data/EXR/D.{CCY}.EUR.SP00.A?format=csvdata`. |
| `lbma`           | XAU/USD                    | 1968-→ (~21k days)   | LBMA London PM gold fix (USD); daily, free JSON at `https://prices.lbma.org.uk/json/gold_pm.json`. We take the USD column (`v[0]`). |
| `derived`        | USD/JPY, USD/CHF, GBP/USD  | overlap of legs | Cross-derived from ECB EUR-quoted legs (see below). |

`frankfurter_app` is acknowledged as a documented fallback (it wraps
the same ECB feed) but is not used in v1; the SDMX endpoint is the
primary.

Sources that require a key are **not** integrated: OANDA, IG Bank,
Stooq (recently key-gated), Dukascopy, FRED. The audit recorded them
as `INCONCLUSIVE` and we leave them there.

## Derived crosses

Both legs must share the same date — no silent forward-fill. Missing-
date pairs are dropped.

```text
USD/JPY  =  EUR/JPY  /  EUR/USD
USD/CHF  =  EUR/CHF  /  EUR/USD
GBP/USD  =  EUR/USD  /  EUR/GBP
```

Each derived row carries `is_derived=True` and a `source_pair` label
like `"EUR/JPY÷EUR/USD"` so a future researcher can always see how the
quote was constructed.

## Schema (locked)

```text
date                 datetime64[ns], UTC, daily
asset                str, e.g. "EUR/USD", "XAU/USD"
source               str: "ecb_sdmx" | "lbma" | "derived"
base                 str (e.g. "EUR", "XAU")
quote                str (e.g. "USD")
close                float — daily reference rate / fix
return_1d            float — simple daily return; NaN at series start
log_return_1d        float — natural-log daily return; NaN at start
is_derived           bool
source_pair          str — "EUR/JPY÷EUR/USD" for derived, else ""
data_quality_status  str: "ok" | "missing" | "stale" | "warning"
notes                str — free-form provenance / quality note
```

## Outputs (gitignored)

```text
data/fx/fx_daily_v1.parquet      # primary
data/fx/fx_daily_v1.csv          # companion for inspection
data/fx/fx_daily_v1_summary.json # summary dump from the CLI script
```

All three are excluded by `.gitignore` under the `data/fx/*` rules.
`data/fx/.gitkeep` is the only tracked file in the directory.

The writer (`fx_research_dataset.write_fx_dataset`) refuses to write
anywhere outside `data/fx/`. Schema validation runs **before** the
write so a bad frame never lands on disk.

## Failure model

Every fetch is fail-soft. A network error or unparseable payload
records a single `data_quality_status="missing"` placeholder row for
the affected asset, plus a human-readable warning. The build never
crashes; the summary surfaces gaps explicitly.

ECB returns `"."` for TARGET-holiday days. We **drop** those rows
(no forward-fill) and emit a counted warning. If a future module
needs a forward-filled view it must do that explicitly and document
it — silent imputation is forbidden in v1.

## Usage

```bash
# Inside the repo's venv:
python main.py build_fx_dataset
# or:
python scripts/build_fx_dataset.py
```

Both paths call `fx_research_dataset.build_and_write()`.

## Safety invariants

The dataset module is checked by the standard `ci_safety_check`. In
addition, `tests/test_fx_research_dataset.py` asserts:

- no broker imports (ccxt / alpaca / kraken / ig_bank / oanda);
- no `API_KEY` / `API_SECRET` reads, no `os.environ` / `os.getenv`;
- no order-placement tokens (`create_order`, `place_order`, `AddOrder`,
  …);
- no `LIVE_TRADING_ENABLED = True`, no `ENABLE_LIVE` / `UNLOCK_TRADING`
  / `FORCE_TRADE` env reads;
- no strategy registration (`strategy_registry`, `scorecard`,
  `backtester`, `paper_trader`, `Strategy(`, …);
- the safety lock remains `locked` after import;
- the `.gitignore` rules excluding `data/fx/*.parquet` and
  `data/fx/*.csv` are present.

## Next branch (recommended)

`data/fx-data-quality-checks-v1` — run automated quality checks on the
v1 dataset (gap analysis, holiday calendar reconciliation, leg-cross
sanity, return-distribution sanity, look-ahead-free invariants) and
emit a quality report. Still no strategy, still no broker.
