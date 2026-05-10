# Unlock procedure

> **The bot is currently locked.** This document describes the
> **only** way execution can ever be enabled. There is no shortcut, no
> environment variable, no config flag, no hidden override.

## Current status (this branch)

| Field | Value |
| --- | --- |
| Bot mode | `research_only` |
| Execution enabled | **False** |
| Paper trading enabled | **False** |
| Kraken connected | **False** |
| API keys loaded | **False** |
| Active strategy | none — **no strategy has reached PASS** |
| Safety lock status | **locked** |

The lock is enforced by [`src/safety_lock.py`](../src/safety_lock.py).
Every check that contributes to the lock state lives in that single
module; every other module asks the lock — they do not decide on their
own.

### What is currently blocking execution

Every one of the following conditions is currently active and is
sufficient on its own to keep the lock locked:

1. `config.LIVE_TRADING_ENABLED = False` (the v1 hard-coded invariant).
2. No scorecard CSV in `results/` reports `verdict = PASS`.
3. No Kraken / broker execution module exists in `src/`.
4. Paper trading is disabled by spec on this branch.

Removing **any one** of those four reasons leaves the other three in
place, and the safety lock continues to return `execution_allowed = False`.

## What `unlock` does NOT mean

* No environment variable releases any gate. Setting things like
  `UNLOCK_TRADING=1`, `ENABLE_LIVE=true`, `KRAKEN_API_KEY=xxx` has
  zero effect on the safety lock. There is a unit test that proves
  this (`test_safety_lock_no_env_var_can_unlock`).
* No CLI flag releases any gate. The CLI commands in `main.py` are
  read-only by design — `bot_status`, `strategy_registry`, `bot_alerts`,
  `dry_run_plan`, `safety_status`. Adding a `--live` flag to any of
  them would not change the lock; it would simply be ignored.
* Connecting an API key in `.streamlit/secrets.toml` is **not**
  unlocking. The engine currently never reads broker keys from
  secrets, environment, or anywhere else.
* Editing `config.LIVE_TRADING_ENABLED` to `True` is **not** sufficient
  on its own — the safety lock has three other independent reasons it
  blocks. All four must be addressed.

## The 10 gates that must be cleared before any trading consideration

These are listed in the order any future maintainer would need to
satisfy them. **All ten** are required. Skipping any one is a defect.

### Strategy / research gates (1–6)

These are research gates. They prove the engine has actually found an
edge, not just an in-sample artefact.

1. **Strategy reaches PASS.** A `*_scorecard.csv` row in `results/`
   has `verdict == "PASS"`. PASS is defined per branch by the
   conservative scorecard; verdict alone is not enough on its own —
   gates 2–5 must also clear.

2. **Strategy beats BTC OOS** in > 50 % of out-of-sample windows.
   The scorecard's `pct_windows_beat_btc` field must exceed 50.0.

3. **Strategy beats the equal-weight basket OOS** in > 50 % of
   windows (`pct_windows_beat_basket > 50.0`).

4. **Strategy beats the placebo median.**
   `placebo.placebo_median_return_pct < strategy_full_return_pct`. A
   strategy that only beats the placebo on drawdown is **not**
   eligible — return must be beaten too.

5. **OOS stability ≥ 60 %.** "Stability" is the share of OOS windows
   that simultaneously beat every primary benchmark (BTC, basket,
   simple momentum, prior best variant) AND are profitable. This is
   the single hardest gate; on this project's prior strategy families
   it has been the binding constraint.

6. **Dry-run produces a sane plan.** `python main.py dry_run_plan`
   reports the strategy's theoretical positions cleanly with `mode =
   DRY_RUN_ONLY` and `execution_status = BLOCKED_NO_PASS_STRATEGY`.
   The plan must round-trip parse and produce a non-trivial allocation
   in current state.

### Infrastructure gates (7–10)

These are platform gates. They are entirely separate from research and
each requires explicit code to be added — no unlock happens by
implication.

7. **Paper-trading module reviewed separately.** The current
   `src/paper_trader.py` is a per-tick simulator that writes JSON
   state. Before any paper-trading activation, the module must be
   audited for: idempotency, state-file safety on crash, and the
   guarantee that no order ever leaves the process. The safety lock's
   `_paper_trader_blocked()` reason must be explicitly removed in code,
   not via config.

8. **Read-only Kraken connector first.** Any Kraken integration must
   begin with a public-endpoint-only client (OHLCV, order book) that
   carries no API key. This connector lives in a clearly-named module
   such as `src/kraken_public.py` and never imports `kraken.private`.
   The safety lock's `_kraken_blocked()` check must be updated to
   reflect that a connector exists, but `is_kraken_connection_allowed`
   continues to require gate 9 below.

9. **Trading key with withdrawals disabled.** If — much later — a
   private Kraken client is added, it must be authenticated against a
   key that explicitly has *trade* permission only and *withdraw*
   permission revoked. The verification of withdrawal-disabled status
   is part of the unlock checklist; the key file path must be in
   `.streamlit/secrets.toml` (gitignored), never in environment vars
   visible to other processes.

10. **Explicit code change required.** Even when 1–9 are satisfied,
    actually flipping the lock requires editing `src/safety_lock.py`
    to remove the reason strings, AND adding the broker call site
    behind an additional `assert_execution_blocked()` check that fires
    if the lock has not been intentionally released.  No "auto-unlock
    when conditions are met" path is permitted. The author of the
    code change owns the trade decision; the lock is the audit trail.

## Hard line

> **No config flag, environment variable, secrets entry, CLI argument,
> or hidden override can unlock trading.**
> **Every gate above is required.**
> **Even after every gate clears, an explicit code change is the only
> way to release the lock.**

If you find yourself adding a config option that conditionally enables
execution, you are doing it wrong. Failing tests are the second line of
defence — `test_safety_lock_no_env_var_can_unlock` and
`test_no_strategy_is_paper_or_live_allowed` are designed to break loudly
if any module starts pretending the gates have moved.

## Where to look in the code

| Concern | File |
| --- | --- |
| Single source of truth for "is execution allowed?" | [`src/safety_lock.py`](../src/safety_lock.py) |
| Per-family verdict + trading-allowed gates | [`src/strategy_registry.py`](../src/strategy_registry.py) |
| Status snapshot consumed by the dashboard | [`src/bot_status.py`](../src/bot_status.py) |
| Read-only theoretical position planner | [`src/dry_run_planner.py`](../src/dry_run_planner.py) |
| Append-only decision log | [`src/decision_journal.py`](../src/decision_journal.py) |
| Operational sanity checks | [`src/system_health.py`](../src/system_health.py) |
| The hard-coded v1 invariant | `LIVE_TRADING_ENABLED` in [`src/config.py`](../src/config.py) |

## What to do next

1. Run `python main.py safety_status` and confirm every reason listed
   matches the four reasons in this document.
2. Run `python main.py system_health` and confirm every check returns
   PASS.
3. Continue research. Add another strategy family or refine an
   existing one. Run it through the scorecard.
4. Do **not** edit anything under the "Hard line" section until every
   one of gates 1–10 has been independently verified.
