"""
Decision journal — append-only record of "what would the bot do, and why,
if it were allowed to do anything?"

This module is the single auditable record of every dry-run reasoning
pass. It does not place orders, does not call brokers, does not track
paper-trading state. The `execution_status` is **always** `"BLOCKED"`
on this branch by construction (the safety lock controls it; if the
lock ever releases, this module logs that explicitly but still does
not act).

Inputs (all read-only):
    bot_status.compute_status()
    strategy_registry.build_registry()
    alert_engine.build_alerts()
    dry_run_planner.build_plan()

Output: results/decision_journal.csv (append-only, one row per call).

Allowed `decision` values:
    RESEARCH_ONLY      Lock locked, no PASS strategy, no urgent issue.
    DRY_RUN_ONLY       Lock locked but a PASS scorecard was found —
                        we'd theoretically be allowed to dry-run, but
                        we still do not execute.
    EXECUTION_BLOCKED  General "blocked by safety lock" rollup.
    NO_VALID_STRATEGY  No PASS scorecard exists.
    DATA_STALE         Latest data is older than the freshness window.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import (alert_engine, bot_status, config, dry_run_planner,
                safety_lock, strategy_registry, utils)

logger = utils.get_logger("cte.decision_journal")


JOURNAL_COLUMNS: List[str] = [
    "timestamp",
    "bot_mode",
    "active_strategy",
    "strategy_verdict",
    "decision",
    "reason",
    "theoretical_action_summary",
    "execution_status",
    "safety_lock_status",
]

EXECUTION_STATUS_BLOCKED = "BLOCKED"

# Allowed decisions — any other value would be a coding error.
DECISION_RESEARCH_ONLY = "RESEARCH_ONLY"
DECISION_DRY_RUN_ONLY = "DRY_RUN_ONLY"
DECISION_EXECUTION_BLOCKED = "EXECUTION_BLOCKED"
DECISION_NO_VALID_STRATEGY = "NO_VALID_STRATEGY"
DECISION_DATA_STALE = "DATA_STALE"

ALLOWED_DECISIONS = (
    DECISION_RESEARCH_ONLY, DECISION_DRY_RUN_ONLY,
    DECISION_EXECUTION_BLOCKED, DECISION_NO_VALID_STRATEGY,
    DECISION_DATA_STALE,
)

OUTPUT_FILENAME = "decision_journal.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _output_path() -> Path:
    return config.RESULTS_DIR / OUTPUT_FILENAME


def _summarise_plan(plan_rows: List[Dict[str, Any]]) -> str:
    if not plan_rows:
        return "no_plan_available"
    firsts = []
    for r in plan_rows[:5]:
        firsts.append(
            f"{r.get('strategy_name', '?')}:{r.get('asset', '?')}"
            f"@{float(r.get('target_weight', 0)):.2f}"
        )
    suffix = "" if len(plan_rows) <= 5 else f" (+{len(plan_rows) - 5} more)"
    return "; ".join(firsts) + suffix


def _decide(snap: bot_status.BotStatus,
              registry: List[Dict[str, Any]],
              alerts: List[Dict[str, Any]]) -> Dict[str, str]:
    """Pure function — compute (decision, reason) from the inputs.
    Order of priority — first match wins:
        1. NO_VALID_STRATEGY  (no PASS verdict anywhere)
        2. DATA_STALE         (cache is older than the freshness window)
        3. EXECUTION_BLOCKED  (lock locked AND a PASS strategy exists,
                                 i.e. the only block left is mechanical)
        4. DRY_RUN_ONLY       (lock unlocked AND a PASS strategy exists
                                 — we still never execute on this branch)
        5. RESEARCH_ONLY      (default)
    """
    pass_count = sum(1 for r in registry if r.get("verdict") == "PASS")
    has_critical = any(a.get("severity") == "critical" for a in alerts)

    if pass_count == 0:
        return {
            "decision": DECISION_NO_VALID_STRATEGY,
            "reason": "no strategy has reached PASS — research only",
        }
    if snap.stale_data_warning:
        return {
            "decision": DECISION_DATA_STALE,
            "reason": (f"latest cache update {snap.latest_data_update}; "
                          "refresh before drawing conclusions"),
        }
    if not snap.execution_enabled:
        # PASS exists but the lock still blocks (Kraken / paper / live flag).
        return {
            "decision": DECISION_EXECUTION_BLOCKED,
            "reason": ("PASS strategy present but safety lock still "
                          "blocks: " + snap.reason_execution_blocked),
        }
    # Lock unlocked + PASS — still NEVER auto-execute on this branch.
    if has_critical:
        return {
            "decision": DECISION_EXECUTION_BLOCKED,
            "reason": "active CRITICAL alert prevents any further action",
        }
    return {
        "decision": DECISION_DRY_RUN_ONLY,
        "reason": ("PASS strategy + lock unlocked — dry-run only "
                    "(this module never executes)"),
    }


def _load_existing() -> pd.DataFrame:
    p = _output_path()
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    try:
        df = pd.read_csv(p)
    except Exception as e:  # noqa: BLE001
        logger.warning("decision journal unreadable; starting fresh (%s)", e)
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    for col in JOURNAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[JOURNAL_COLUMNS]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def record_decision(save: bool = True) -> pd.DataFrame:
    """Append one decision row to the journal. Returns the full journal
    (existing + new) frame."""
    snap = bot_status.compute_status()
    registry = strategy_registry.build_registry()
    alerts = alert_engine.build_alerts()
    plan = dry_run_planner.build_plan()
    decision = _decide(snap, registry, alerts)

    new_row: Dict[str, Any] = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "bot_mode": snap.bot_mode,
        "active_strategy": snap.active_strategy,
        "strategy_verdict": snap.active_strategy_verdict,
        "decision": decision["decision"],
        "reason": decision["reason"],
        "theoretical_action_summary": _summarise_plan(plan),
        "execution_status": EXECUTION_STATUS_BLOCKED,
        "safety_lock_status": snap.safety_lock_status,
    }
    # Defence in depth — assert at write time that the chosen decision
    # is in the documented set and that execution_status is BLOCKED.
    if new_row["decision"] not in ALLOWED_DECISIONS:
        raise ValueError(
            f"decision_journal produced unrecognised decision: "
            f"{new_row['decision']!r}",
        )
    if new_row["execution_status"] != EXECUTION_STATUS_BLOCKED:
        raise ValueError(
            f"decision_journal must always set execution_status=BLOCKED "
            f"on this branch; got {new_row['execution_status']!r}",
        )

    existing = _load_existing()
    out = pd.concat(
        [existing, pd.DataFrame([new_row], columns=JOURNAL_COLUMNS)],
        ignore_index=True,
    )
    if save:
        utils.write_df(out, _output_path())
    return out


def load_journal() -> pd.DataFrame:
    return _load_existing()
