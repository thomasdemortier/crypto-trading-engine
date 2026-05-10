# Research Dashboard

The `Research Dashboard` section of the Streamlit app is the project's
honest read-out of where research stands today. It exists because nine
research branches have been archived without merging, and the project
needs to be **operationally honest** about that — not optimistic.

## What the dashboard shows

The dashboard sits below the existing **Bot Control Center**. It has
seven tabs:

1. **Executive state** — project mode, active strategy, production
   baseline, safety lock status, execution / paper / live / Kraken
   flags, and the next allowed action. Every value comes from
   `safety_lock` live, not a cached config — the dashboard cannot
   silently lie about whether trading is locked.
2. **Strategy verdicts** — wraps `src/strategy_registry.build_registry()`.
   Failed and inconclusive rows are shaded red; PASS would be green
   (we have none). The dashboard surfaces a hard banner if any row
   has paper or live trading allowed.
3. **Archived timeline** — the curated list of nine archived research
   branches with kind, verdict, one-line reason, report path, and
   `merge_allowed=False`. Source of truth lives in
   `src/research_dashboard.ARCHIVED_BRANCHES`.
4. **Baseline** — BTC buy-and-hold metrics computed off
   `results/equity_curve.csv` if present (starting capital, final
   value, total return, max drawdown, annualised vol, bar count).
   Degrades gracefully when the file is missing.
5. **Risk dashboard** — inventories the expected results files
   (`equity_curve.csv`, `summary_metrics.csv`, scorecards) and
   reports freshness in days. No equity reconstruction is done —
   the dashboard only surfaces what the prior pipeline produced.
6. **Safety + governance** — safety status, decision journal latest
   row, latest 10 alerts, plus an excerpt of `docs/unlock_procedure.md`.
7. **Next allowed actions** — explicitly enumerates what is OK
   (research / risk reporting / dashboard improvements / paid-data
   audits when a key is supplied) and explicitly forbids paper trading,
   live trading, Kraken connection, API key entry, and order placement.

## Why failed strategies are retained

The nine archived branches are kept on `origin/research/...` rather
than merged or deleted. Three reasons:

* **Decision evidence.** Future maintainers (or future-you) need to
  see *why* a strategy class was tried and rejected, otherwise the
  same idea cycles back. The reports under `reports/*.md` are the
  evidence; the dashboard makes them browsable.
* **Falsification track record.** "BTC is hard to beat in this
  4-year sample" is only credible after you've shown nine attempts
  that failed under a locked scorecard. Deleting them weakens the
  argument.
* **Reproducibility.** The CSV outputs (when re-run from the
  branches) regenerate the verdict; nothing about the conclusion is
  hidden.

## Why BTC buy-and-hold is the baseline

The 4-year sample (2022-04 → 2026-05) is BTC-favourable. Every
long-only allocator we tried — drawdown-targeted, funding+basis,
BTC/ETH relative-value, market-structure, sentiment overlay,
funding-only — produced the same pattern: tighter drawdown than BTC,
beats the random-bucket placebo, but loses to BTC out-of-sample in
more than 50 % of windows. That is risk control, not alpha. Until a
**new untested data class** is available (paid OI / liquidations /
long-short ratios from CoinGlass / CryptoQuant / Velo Data, or a
real long/short pair backtester), there is no honest case for
opening another long-only branch on the existing free-data stack.

## How to use the dashboard

```bash
streamlit run streamlit_app.py
```

The Research Dashboard appears below the Bot Control Center.
Everything is read-only; no inputs trigger trading.

For programmatic access, the helper functions are in
`src/research_dashboard.py`:

```python
from src import research_dashboard as rd

rd.executive_state()             # dict: lock + flags + next action
rd.strategy_verdict_board()      # DataFrame matching registry schema
rd.archived_timeline_dataframe() # DataFrame of archived branches
rd.baseline_metrics()            # dict of BTC equity-curve metrics
rd.latest_results_state()        # dict: which results files exist
rd.decision_journal_latest_row() # dict or None
rd.alert_history_latest_rows()   # DataFrame
```

All helpers fail soft: missing files return empty DataFrames or
typed defaults, never exceptions.

## What NOT to do

* **Do not enable paper trading.** The codebase has no execution
  module; flipping a flag will not make one appear.
* **Do not connect Kraken** (or any other broker). The safety lock
  is wired to refuse Kraken connection until every gate is released.
* **Do not enter API keys into the dashboard.** No input field
  accepts them and no helper reads them.
* **Do not merge an archived research branch into `main`.** Every
  archived entry has `merge_allowed=False`; a unit test enforces it.
* **Do not write a new strategy on the existing free-data stack.**
  Three branches in a row failed on the same pattern. The
  free-open-data re-audit confirmed there is no untested signal
  class on free public endpoints.
* **Do not soften the scorecard.** The PASS bar is locked and was
  derived from spec, not tuned after seeing results. Lowering it
  would invalidate every prior verdict.

If the user reopens strategy research later, the dashboard's
"Next allowed actions" tab lists the only two paths that could
change the verdict: a paid-data audit with a real API key, or a
true long/short pair backtester (with explicit spec approval).
