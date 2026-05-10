# Strategy universe selection

A read-only, deterministic ranking of candidate universes the project
*could* pursue next. The point of this module is to make the next
strategic decision **before** writing any strategy code, so the
choice of universe is documented and falsifiable rather than ad-hoc.

## What it is

`src/strategy_universe_selection.py` is a pure-Python ledger plus
a ranker. It scores four universes on five axes (data, execution,
edge, complexity, risk), takes their equal-weighted average, and
emits one of four locked decision statuses per universe:

* **RECOMMENDED** — open the next research branch on this universe.
* **WATCHLIST** — useful but not the next branch.
* **NOT_NOW** — defer; a prerequisite is missing.
* **REJECTED** — track record is conclusive; do not reopen.

## What it is NOT

* Not a strategy.
* Not a backtest.
* Not an execution surface.
* Not a recommendation to buy, sell, place orders, paper-trade, go
  live, or connect a broker. The recommendation vocabulary is
  locked in `FORBIDDEN_RECOMMENDATION_TOKENS` and a unit test
  enforces that no `recommended_next_action` contains any of those
  tokens.

## How to use it

```python
from src import strategy_universe_selection as sus

table = sus.rank_universes()           # DataFrame, locked schema
top = sus.top_recommendation()         # dict for the dashboard
ok = sus.recommendation_is_clean(s)    # bool — must mention "research"
                                          # and contain no forbidden
                                          # trade-action language
```

The Streamlit dashboard renders `rank_universes()` directly under
the Research Dashboard. There is no CLI command — this is a static
research artefact, not an operational tool.

## Locked schema

Output table columns (in order):

```
universe, score, rank,
data_score, execution_score, edge_score, complexity_score, risk_score,
recommended_next_action, decision_status, notes
```

* `score` is the equal-weighted mean of the five `*_score` axes.
* `rank` is 1-indexed; ties break by `decision_status` priority
  (RECOMMENDED → WATCHLIST → NOT_NOW → REJECTED).
* `decision_status` is one of the four locked statuses above.
* `recommended_next_action` MUST contain "research" and MUST NOT
  contain any forbidden trade-action token.

## When to update it

Each per-universe assessment is locked in a frozen dataclass. To
change a score, edit the dataclass — there is no tuning function.
Any edit must be accompanied by a written rationale in the report
(`reports/strategy_universe_selection_report.md`); a unit test
asserts every assessment has both `pass_criteria` and `fail_criteria`
populated.

If a new universe should be added (e.g. equities, options), add a
new `UniverseAssessment` to the locked tuple AND extend the
`UNIVERSES` constant AND extend the test suite to cover it. The
existing tests will catch any drift.

## Safety rules

* No network calls.
* No broker SDK imports.
* No API key reads.
* No order placement strings.
* No paper / live trading enablement.
* No file writes from the module.
* The recommendation vocabulary is locked at the source level.

The safety lock continues to be locked; no path through this module
can flip any execution flag.
