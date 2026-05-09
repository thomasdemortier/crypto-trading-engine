"""
Alert engine — read-only summary of the engine's research state for
the Bot Control Center.

Generates 0-N alert rows from the existing scorecards / coverage CSVs /
data freshness checks. Severity ∈ {info, warning, critical}. Recommended
actions are **always research-only** — there is no code path that
recommends trading.

Output: results/bot_alerts.csv with columns
    timestamp, severity, category, message, recommended_action

Hard rules:
    * Never returns "trade", "paper trade", "go live", or anything
      that recommends execution.
    * Always emits at least one CRITICAL alert when no PASS strategy
      exists (defence in depth — the Bot Control Center must show a
      blocked banner regardless of the rest of the file's state).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from . import bot_status, config, safety_lock, strategy_registry, utils


ALERT_COLUMNS: List[str] = [
    "timestamp", "severity", "category", "message", "recommended_action",
]
SEVERITIES = ("info", "warning", "critical")
RECOMMENDED_RESEARCH_ONLY = "continue research; do not trade"
RECOMMENDED_REFRESH_DATA = "refresh local data caches"
RECOMMENDED_RUN_RESEARCH = (
    "run the research pipeline that produced the missing CSV"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ts_now_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()


def _row(severity: str, category: str, message: str,
          recommended_action: str = RECOMMENDED_RESEARCH_ONLY) -> Dict[str, Any]:
    if severity not in SEVERITIES:
        raise ValueError(f"unknown severity {severity!r}")
    return {
        "timestamp": _ts_now_iso(),
        "severity": severity, "category": category,
        "message": message,
        "recommended_action": recommended_action,
    }


# ---------------------------------------------------------------------------
# Alert builders
# ---------------------------------------------------------------------------
def _safety_lock_alerts() -> List[Dict[str, Any]]:
    sl = safety_lock.status()
    if sl.execution_allowed:
        return [_row(
            "info", "safety_lock",
            "Safety lock is unlocked — review every check before any trade.",
            "review the unlock conditions; do not trade automatically",
        )]
    return [_row(
        "critical", "safety_lock",
        "Safety lock is LOCKED. Trading is disabled. " +
        "; ".join(sl.reasons_blocked),
        RECOMMENDED_RESEARCH_ONLY,
    )]


def _registry_alerts() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    reg = strategy_registry.build_registry()
    pass_count = sum(1 for r in reg if r["verdict"] == "PASS")
    fail_count = sum(1 for r in reg if r["verdict"] == "FAIL")
    inconc_count = sum(1 for r in reg if r["verdict"] == "INCONCLUSIVE")
    unknown_count = sum(1 for r in reg if r["verdict"] == "UNKNOWN")

    if pass_count == 0:
        rows.append(_row(
            "critical", "strategy",
            f"No strategy has reached PASS. {fail_count} family/families "
            f"FAIL, {inconc_count} INCONCLUSIVE, {unknown_count} unknown. "
            "Trading remains disabled.",
            RECOMMENDED_RESEARCH_ONLY,
        ))
    if unknown_count > 0:
        rows.append(_row(
            "warning", "strategy",
            f"{unknown_count} family/families have no scorecard CSV in "
            f"results/ — verdict unknown. Re-run the relevant research "
            f"pipeline to populate.",
            RECOMMENDED_RUN_RESEARCH,
        ))

    # Per-family informational rows describing the verdict + the most
    # important headline metric.
    for r in reg:
        if r["verdict"] in ("PASS", "WATCHLIST"):
            sev = "info"
        elif r["verdict"] in ("INCONCLUSIVE", "UNKNOWN"):
            sev = "warning"
        else:
            sev = "info"
        rows.append(_row(
            sev, "strategy",
            f"{r['strategy_family']}: {r['verdict']} — "
            f"{r['best_result_summary'][:200]}",
            RECOMMENDED_RESEARCH_ONLY,
        ))
    return rows


def _data_freshness_alerts() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    s = bot_status.compute_status()
    if s.stale_data_warning:
        msg = (f"Data is stale (latest update: {s.latest_data_update}). "
                if s.latest_data_update else
                "No daily OHLCV cache found. ")
        rows.append(_row(
            "warning", "data_freshness",
            msg + "Refresh before drawing conclusions from any rerun.",
            RECOMMENDED_REFRESH_DATA,
        ))
    if not s.generated_results_available:
        rows.append(_row(
            "warning", "results_files",
            "No generated CSVs in results/. Run a research command "
            "(e.g. `python main.py download` then a scorecard command).",
            RECOMMENDED_RUN_RESEARCH,
        ))
    return rows


def _missing_results_alerts() -> List[Dict[str, Any]]:
    """Flag specific result files the registry expected but didn't find."""
    rows: List[Dict[str, Any]] = []
    reg = strategy_registry.build_registry()
    for r in reg:
        if r["scorecard_status"] == "missing":
            rows.append(_row(
                "warning", "results_files",
                f"{r['strategy_family']}: scorecard CSV missing in "
                "results/. Cannot determine current verdict.",
                RECOMMENDED_RUN_RESEARCH,
            ))
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_alerts() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rows.extend(_safety_lock_alerts())
    rows.extend(_registry_alerts())
    rows.extend(_data_freshness_alerts())
    rows.extend(_missing_results_alerts())
    return rows


def write_alerts(save: bool = True) -> pd.DataFrame:
    rows = build_alerts()
    df = pd.DataFrame(rows, columns=ALERT_COLUMNS)
    if save:
        utils.write_df(df, config.RESULTS_DIR / "bot_alerts.csv")
    return df
