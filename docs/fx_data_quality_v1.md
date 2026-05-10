# FX research dataset — data quality v1

The `data/fx-data-quality-checks-v1` branch adds a fixed battery of
**read-only** quality checks against the v1 FX dataset built in
`data/fx-research-dataset-v1`. It validates the dataset before any
strategy research begins.

> No strategy. No backtest. No broker. No API keys. No paper trading.
> No live trading. No order placement. No execution. No network.

## Why this branch exists

A strategy on top of an unvalidated dataset is garbage in, garbage
out. The locked roadmap is:

1. ~~Audit sources — `v0.6-fx-crypto-source-audit-locked`.~~
2. ~~Build clean FX dataset — `data/fx-research-dataset-v1`.~~
3. **Run data quality checks — this branch.**
4. Build first FX strategy — only after the dataset earns a
   non-FAIL verdict here.

Until the quality verdict is acceptable, the next branch
(`research/fx-eod-trend-strategy-v1`) cannot start.

## What the checks do

| # | Check                  | What it asserts                                                                                              |
|:-:|:-----------------------|:-------------------------------------------------------------------------------------------------------------|
| 1 | `schema`               | The locked 12-column schema is present and exclusive.                                                        |
| 2 | `asset_coverage`       | All 8 expected pairs are present (EUR/USD, EUR/GBP, EUR/JPY, EUR/CHF, USD/JPY, USD/CHF, GBP/USD, XAU/USD).   |
| 3 | `date_monotonicity`    | Every asset's dates are strictly ascending.                                                                  |
| 4 | `duplicate_rows`       | No duplicate `(asset, date)` rows.                                                                           |
| 5 | `missing_close`        | No `NaN` close on rows whose `data_quality_status="ok"`.                                                     |
| 6 | `return_consistency`   | `return_1d` and `log_return_1d` recompute from close within `1e-9`.                                          |
| 7 | `derived_pair_sanity`  | `USD/JPY = EUR/JPY ÷ EUR/USD`; `USD/CHF = EUR/CHF ÷ EUR/USD`; `GBP/USD = EUR/USD ÷ EUR/GBP`. Same-date join. |
| 8 | `extreme_returns`      | Flags `|return_1d|` > 5% (FX) and > 10% (XAU/USD). **Flagged, not deleted.**                                 |
| 9 | `coverage_gaps`        | Per-asset calendar gaps > 7 days. Weekend gaps are expected and not flagged.                                 |

## Verdict combinator

| Verdict        | Trigger                                                                                                          |
|:---------------|:-----------------------------------------------------------------------------------------------------------------|
| `PASS`         | Every check is `PASS`.                                                                                           |
| `WARNING`      | At least one `WARNING` (e.g. extreme returns or large gaps), but no `FAIL`.                                      |
| `FAIL`         | Schema missing, asset missing, duplicates, missing close, derived formula breaks, or non-monotonic dates.        |
| `INCONCLUSIVE` | Dataset file missing or unreadable, or fewer than 10 rows.                                                       |

The CLI exits `0` on PASS / WARNING, `1` on FAIL, and `2` on
INCONCLUSIVE so a CI step can distinguish "dataset not built yet"
from "dataset is broken".

## Outputs (gitignored)

```text
results/fx_data_quality_report.csv   # one row per check + overall verdict
results/fx_data_quality_report.json  # same content, structured
```

Both are excluded by the existing `results/*.csv` and `results/*.json`
rules. The writer (`fx_data_quality.write_fx_data_quality_report`)
refuses to write outside `results/`.

## Failure model

- The runner is fail-soft on a missing dataset: it returns an
  `INCONCLUSIVE` report with the message
  `"FX dataset not found. Run python main.py build_fx_dataset first."`
  rather than crashing.
- No row is deleted, no value is imputed, and no holiday gap is
  filled. Findings are recorded in the report; the caller decides.
- Tolerances are locked: `1e-9` for return recomputation, `1e-9`
  relative for derived ratios, 5% / 10% for extreme returns, 7
  calendar days for coverage gaps.

## Usage

```bash
# Build the dataset first if you haven't:
python main.py build_fx_dataset

# Run the quality checks:
python main.py check_fx_data_quality
```

Both commands call the modules directly — no network.

## Current verdict on the v1 dataset

Running the full battery against the v1 dataset (`data/fx/fx_daily_v1.parquet`,
63,600 rows, 8 assets, 1968-04-01 → 2026-05-08):

```text
verdict: WARNING

[PASS]    schema                — 12 required columns present
[PASS]    asset_coverage        — all 8 expected assets present
[PASS]    date_monotonicity     — strictly ascending per asset
[PASS]    duplicate_rows        — none
[PASS]    missing_close         — none on OK rows
[PASS]    return_consistency    — max residual: simple=0.00e+00, log=0.00e+00
[PASS]    derived_pair_sanity   — max relative residual = 0.00e+00 across all 3 derived pairs
[WARNING] extreme_returns       — 18 daily moves flagged (real shocks, not data errors)
[PASS]    coverage_gaps         — no per-asset gap > 7 days
```

The 18 flagged extreme returns are well-known historical FX shocks,
not data errors:

- 2015-01-15 EUR/CHF -14.4% (SNB cap removal); USD/CHF -13.9%.
- 2011-09-06 EUR/CHF +8.3% (SNB cap announcement); USD/CHF +8.5%.
- 2016-06-24 EUR/GBP +5.4% / GBP/USD -7.8% (Brexit referendum).
- Several 2000-2008 EUR/JPY moves around major BOJ / GFC episodes.
- Seven XAU/USD moves above 10% in the late-1970s gold rally and the
  early-1980s reversal.

These are real signal, not contamination, so the dataset is
**acceptable for research** under a WARNING verdict. The next branch
should reference these dates explicitly when reasoning about
strategy robustness.

## Safety invariants

The module is checked by the standard `ci_safety_check`. In addition,
`tests/test_fx_data_quality.py` asserts:

- no broker imports (`ccxt`, `alpaca`, `kraken`, `ig_bank`, `oanda`);
- no `API_KEY` / `API_SECRET` / `os.environ` / `os.getenv` reads;
- no order-placement tokens (`create_order`, `place_order`,
  `AddOrder`, …);
- no `LIVE_TRADING_ENABLED = True`, no `ENABLE_LIVE` / `UNLOCK_TRADING`
  / `FORCE_TRADE`;
- no strategy registration (`strategy_registry`, `scorecard`,
  `backtester`, `register_strategy`, `paper_trader`);
- the safety lock remains `locked` after import;
- the `.gitignore` rules excluding `results/*.csv` and
  `results/*.json` are present.

## Next branch (recommended)

`research/fx-eod-trend-strategy-v1` — branch from this commit and
build the first end-of-day FX trend strategy on the validated
dataset, with the existing strategy / backtester / scorecard
plumbing. **Require** that `python main.py check_fx_data_quality`
returns PASS or WARNING (never FAIL or INCONCLUSIVE) as a prereq in
the strategy module's `assert_paper_only`-equivalent guard. Same hard
rules: no broker, no keys, no execution, no paper, no live; the
safety lock and CI gate must remain green.
