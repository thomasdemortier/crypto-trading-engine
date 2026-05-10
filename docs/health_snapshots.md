# Health Snapshots

Read-only timeline of bot state — safety lock, system health,
strategy registry verdicts, and portfolio risk. One row per CLI
invocation. The engine writes a single row to a gitignored CSV;
the Streamlit dashboard reads it back and shows a Health Timeline.

## Purpose

Convert the engine from "tells you the current state" to "tells you
how the state has drifted over time". Specifically:

* **Did the safety lock ever unlock unexpectedly?** Each row records
  `safety_lock_status`, `execution_allowed`,
  `paper_trading_allowed`, and `kraken_connection_allowed`. If any
  of these flips, the timeline shows it.
* **Is the system health trending?** Pass / warning / fail counts
  per snapshot.
* **Has any strategy passed?** Strategy registry counts (PASS / FAIL
  / INCONCLUSIVE).
* **What was the portfolio risk class on the day?** If a portfolio
  CSV is present at snapshot time, total market value plus risk
  classification plus recommendation are recorded.
* **Did anything go wrong reading these inputs?** The `notes` column
  records any module that raised — the snapshot is fail-soft.

## Schema (locked)

The CSV schema is locked in `src/health_snapshot.SNAPSHOT_COLUMNS`
and a unit test asserts it. Columns:

| Column | Type | Source |
| --- | --- | --- |
| `snapshot_timestamp` | ISO 8601 UTC | `datetime.now(timezone.utc)` |
| `safety_lock_status` | `locked` / `unlocked` / `unknown` | `safety_lock.safety_lock_status()` |
| `execution_allowed` | bool | `safety_lock.is_execution_allowed()` |
| `paper_trading_allowed` | bool | `safety_lock.is_paper_trading_allowed()` |
| `kraken_connection_allowed` | bool | `safety_lock.is_kraken_connection_allowed()` |
| `blocked_reason_count` | int | `len(safety_lock.reasons_blocked())` |
| `system_health_pass_count` | int | `system_health.run_health_checks()` rows where `status == PASS` |
| `system_health_warning_count` | int | rows where `status == WARNING` |
| `system_health_fail_count` | int | rows where `status == FAIL` |
| `strategy_total_count` | int | `len(strategy_registry.build_registry())` |
| `strategy_pass_count` | int | rows where `verdict == PASS` |
| `strategy_fail_count` | int | rows where `verdict == FAIL` |
| `strategy_inconclusive_count` | int | rows where `verdict ∈ {INCONCLUSIVE, UNKNOWN}` |
| `portfolio_file_present` | bool | `data/portfolio_holdings.csv` exists at snapshot time |
| `portfolio_schema_valid` | bool | `portfolio_risk.validate_portfolio_schema(...).ok` |
| `portfolio_total_market_value` | float | sum of `quantity * current_price` |
| `portfolio_risk_classification` | `LOW` / `MODERATE` / `HIGH` / `EXTREME` / `UNKNOWN` | `portfolio_risk.classify_portfolio_risk(...)` |
| `portfolio_recommendation` | one of the locked phrases | `portfolio_risk.generate_risk_recommendation(...).action` |
| `notes` | str | any module read errors recorded as `module_error:ExceptionType` |

The `notes` column is the canary: any non-empty value means a
downstream module raised; everything else in that row used safe
defaults. Read it.

## How to run

```bash
python main.py write_health_snapshot
```

The CLI:

1. Calls `collect_health_snapshot()` to read live state from the
   four upstream modules.
2. Calls `append_health_snapshot()` to add one row to
   `results/health_snapshots.csv` (creates the file with a header
   the first time).
3. Prints a one-line summary of the new row to stdout.

Output is appended only — the writer never overwrites prior rows.
A defensive `_assert_inside_results()` guard refuses to write any
path outside a `results/` directory; a unit test enforces this.

## What this snapshot tracks

The four read paths run in this order, each guarded by `try/except`
so a single broken upstream never crashes the snapshot:

1. **safety_lock** — execution / paper / Kraken booleans + reasons
   blocked count.
2. **system_health** — `run_health_checks()` rows summarised by
   status.
3. **strategy_registry** — `build_registry()` rows summarised by
   verdict.
4. **portfolio_risk** — `get_portfolio_risk_dashboard_state(path)`
   only if the user's portfolio CSV exists at the resolved path.

If a step raises, that row's metric for the failed module falls back
to the safe default in `_safe_defaults()` and the exception type is
recorded in `notes`.

## What this snapshot does NOT do

* **Does not place orders.** No broker imports, no order placement
  strings, no execution surface — unit tests enforce this at the
  source level.
* **Does not connect to any broker** (Kraken, Binance, Bybit,
  Alpaca, etc.).
* **Does not enable paper trading or live trading.** It only READS
  the safety lock state — it cannot flip any flag.
* **Does not call any external API.** No network calls; everything
  is computed from local module state.
* **Does not write the user's portfolio CSV** — it reads it if
  present and records `portfolio_file_present=False` if absent.
* **Does not auto-schedule.** This is on-demand only. A future
  branch could add a cron / launchd hook, but that is out of scope
  here. The dashboard does not even expose a "write snapshot"
  button to keep the policy clean.

## Safety rules

* The output file `results/health_snapshots.csv` is gitignored via
  the existing `results/*.csv` rule — a unit test asserts the rule
  is in `.gitignore`.
* Writes are confined to `results/` — `_assert_inside_results()`
  raises `ValueError` on any other path.
* The schema is locked. Adding a column requires editing
  `SNAPSHOT_COLUMNS` AND the schema test in
  `tests/test_health_snapshot.py`. This prevents drift.
* The module imports nothing that could initiate trading (no `ccxt`,
  no `kraken`, no `binance.client`, no `alpaca`). Tests assert.

## Programmatic access

For notebooks or other tooling:

```python
from src import health_snapshot as hs

snap = hs.collect_health_snapshot()              # one dict
hs.append_health_snapshot(snapshot=snap)         # writes one row
df, warning = hs.load_health_snapshots()         # read all rows
summary = hs.summarize_health_timeline(df)       # latest-row summary
```

`collect_health_snapshot()` accepts an optional `portfolio_path`
override for tests; in production the default
`portfolio_risk.DEFAULT_PORTFOLIO_PATH` is used.

`load_health_snapshots()` returns `(DataFrame, warning|None)` — the
DataFrame is empty and the warning is set if the file is missing or
malformed. The dashboard renders both.
