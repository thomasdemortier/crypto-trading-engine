"""
Health snapshot writer + reader.

Captures one row of bot state — safety lock, system health, strategy
registry, portfolio risk — and appends it to
`results/health_snapshots.csv` (gitignored).

Hard rules (locked):
    * No network calls.
    * No broker imports.
    * No API key reads.
    * No Kraken private endpoints.
    * No order placement strings.
    * No paper-trading or live-trading enablement.
    * Writes ONLY to `results/health_snapshots.csv` (or an explicit
      override path under `results/` for tests).
    * Reads from existing project modules: safety_lock, system_health,
      strategy_registry, portfolio_risk. If any of those raise, the
      snapshot still produces a row with safe defaults and an entry
      in the `notes` column.
    * Never creates the user's portfolio CSV; reads it if present,
      records `portfolio_file_present=False` if absent.

Output schema (locked, per spec):
    snapshot_timestamp
    safety_lock_status
    execution_allowed
    paper_trading_allowed
    kraken_connection_allowed
    blocked_reason_count
    system_health_pass_count
    system_health_warning_count
    system_health_fail_count
    strategy_total_count
    strategy_pass_count
    strategy_fail_count
    strategy_inconclusive_count
    portfolio_file_present
    portfolio_schema_valid
    portfolio_total_market_value
    portfolio_risk_classification
    portfolio_recommendation
    notes
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Locked schema.
# ---------------------------------------------------------------------------
SNAPSHOT_COLUMNS: List[str] = [
    "snapshot_timestamp",
    "safety_lock_status",
    "execution_allowed",
    "paper_trading_allowed",
    "kraken_connection_allowed",
    "blocked_reason_count",
    "system_health_pass_count",
    "system_health_warning_count",
    "system_health_fail_count",
    "strategy_total_count",
    "strategy_pass_count",
    "strategy_fail_count",
    "strategy_inconclusive_count",
    "portfolio_file_present",
    "portfolio_schema_valid",
    "portfolio_total_market_value",
    "portfolio_risk_classification",
    "portfolio_recommendation",
    "notes",
]

DEFAULT_OUTPUT_PATH: Path = (
    config.RESULTS_DIR / "health_snapshots.csv"
)


# ---------------------------------------------------------------------------
# Defensive defaults — used when an external module fails or is absent.
# ---------------------------------------------------------------------------
def _safe_defaults() -> Dict[str, Any]:
    return {
        "safety_lock_status": "unknown",
        "execution_allowed": False,
        "paper_trading_allowed": False,
        "kraken_connection_allowed": False,
        "blocked_reason_count": 0,
        "system_health_pass_count": 0,
        "system_health_warning_count": 0,
        "system_health_fail_count": 0,
        "strategy_total_count": 0,
        "strategy_pass_count": 0,
        "strategy_fail_count": 0,
        "strategy_inconclusive_count": 0,
        "portfolio_file_present": False,
        "portfolio_schema_valid": False,
        "portfolio_total_market_value": float("nan"),
        "portfolio_risk_classification": "UNKNOWN",
        "portfolio_recommendation": "data missing",
    }


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# 2) collect_health_snapshot
# ---------------------------------------------------------------------------
def collect_health_snapshot(
    portfolio_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build one snapshot dict matching `SNAPSHOT_COLUMNS`. Never raises;
    each external read is wrapped, and any failure adds an entry to
    the `notes` column instead of crashing."""
    out = _safe_defaults()
    out["snapshot_timestamp"] = _utc_iso_now()
    notes_parts: List[str] = []

    # Safety lock — read live state.
    try:
        from . import safety_lock
        out["safety_lock_status"] = safety_lock.safety_lock_status()
        out["execution_allowed"] = bool(safety_lock.is_execution_allowed())
        out["paper_trading_allowed"] = bool(
            safety_lock.is_paper_trading_allowed())
        out["kraken_connection_allowed"] = bool(
            safety_lock.is_kraken_connection_allowed())
        out["blocked_reason_count"] = int(
            len(safety_lock.reasons_blocked()))
    except Exception as exc:  # noqa: BLE001 — fail-soft per spec
        notes_parts.append(f"safety_lock_error:{type(exc).__name__}")

    # System health — sum status counts.
    try:
        from . import system_health
        rows = system_health.run_health_checks()
        df_status = [str(r.get("status", "")).upper() for r in rows]
        out["system_health_pass_count"] = sum(
            1 for s in df_status if s == "PASS")
        out["system_health_warning_count"] = sum(
            1 for s in df_status if s == "WARNING")
        out["system_health_fail_count"] = sum(
            1 for s in df_status if s == "FAIL")
    except Exception as exc:  # noqa: BLE001
        notes_parts.append(f"system_health_error:{type(exc).__name__}")

    # Strategy registry — verdict counts.
    try:
        from . import strategy_registry
        rows = strategy_registry.build_registry()
        out["strategy_total_count"] = int(len(rows))
        verdicts = [str(r.get("verdict", "")).upper() for r in rows]
        out["strategy_pass_count"] = sum(
            1 for v in verdicts if v == "PASS")
        out["strategy_fail_count"] = sum(
            1 for v in verdicts if v == "FAIL")
        out["strategy_inconclusive_count"] = sum(
            1 for v in verdicts if v in ("INCONCLUSIVE", "UNKNOWN"))
    except Exception as exc:  # noqa: BLE001
        notes_parts.append(f"strategy_registry_error:{type(exc).__name__}")

    # Portfolio risk — only if a file is actually present at the path.
    try:
        from . import portfolio_risk
        path = (portfolio_path
                  if portfolio_path is not None
                  else portfolio_risk.DEFAULT_PORTFOLIO_PATH)
        present = Path(path).exists()
        out["portfolio_file_present"] = bool(present)
        if present:
            state = portfolio_risk.get_portfolio_risk_dashboard_state(path)
            out["portfolio_schema_valid"] = bool(
                state["schema_status"].ok)
            mv = state["summary"].get("total_market_value", 0.0)
            out["portfolio_total_market_value"] = (
                float(mv) if mv is not None else float("nan"))
            out["portfolio_risk_classification"] = str(
                state.get("risk_classification", "UNKNOWN"))
            out["portfolio_recommendation"] = str(
                state["recommendation"].get(
                    "action", "do nothing until data is complete"))
    except Exception as exc:  # noqa: BLE001
        notes_parts.append(f"portfolio_risk_error:{type(exc).__name__}")

    out["notes"] = "; ".join(notes_parts)
    # Ensure exact column ordering on the way out.
    return {col: out.get(col) for col in SNAPSHOT_COLUMNS}


# ---------------------------------------------------------------------------
# Path safety — never write outside results/.
# ---------------------------------------------------------------------------
def _assert_inside_results(path: Path) -> None:
    """Defensive check: refuse to write to any path outside `results/`.
    The caller is expected to use `DEFAULT_OUTPUT_PATH` or a tmp path
    that resolves under a results directory; if a future edit tries to
    point this writer somewhere else, it raises."""
    try:
        resolved = Path(path).resolve()
    except (OSError, RuntimeError):
        return  # nothing we can do; write attempt will fail naturally
    parts = [p.lower() for p in resolved.parts]
    if "results" not in parts:
        raise ValueError(
            f"health_snapshot writer refuses to write outside a 'results' "
            f"directory: {resolved}",
        )


# ---------------------------------------------------------------------------
# 3) append_health_snapshot
# ---------------------------------------------------------------------------
def append_health_snapshot(
    path: Optional[Path] = None,
    snapshot: Optional[Dict[str, Any]] = None,
) -> Path:
    """Append a single snapshot row to the CSV. Creates the file (with
    header) if it does not exist. Never overwrites existing rows."""
    out_path = Path(path) if path is not None else DEFAULT_OUTPUT_PATH
    _assert_inside_results(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    row = (snapshot if snapshot is not None
            else collect_health_snapshot())
    # Keep the row aligned to the locked column order.
    aligned = {col: row.get(col) for col in SNAPSHOT_COLUMNS}
    df_row = pd.DataFrame([aligned], columns=SNAPSHOT_COLUMNS)

    write_header = not out_path.exists() or out_path.stat().st_size == 0
    df_row.to_csv(out_path, mode="a", header=write_header, index=False)
    return out_path


# ---------------------------------------------------------------------------
# 4) load_health_snapshots
# ---------------------------------------------------------------------------
def load_health_snapshots(
    path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Read the snapshots CSV. Returns `(df, warning|None)`. Never
    raises: missing or malformed files yield an empty DataFrame plus a
    warning string."""
    p = Path(path) if path is not None else DEFAULT_OUTPUT_PATH
    if not p.exists():
        return (
            pd.DataFrame(columns=SNAPSHOT_COLUMNS),
            f"No snapshot file at {p}. Run "
            "`python main.py write_health_snapshot` to create it.",
        )
    try:
        df = pd.read_csv(p)
    except Exception as exc:  # noqa: BLE001
        return (
            pd.DataFrame(columns=SNAPSHOT_COLUMNS),
            f"snapshot CSV at {p} unreadable: {type(exc).__name__}",
        )
    missing = [c for c in SNAPSHOT_COLUMNS if c not in df.columns]
    if missing:
        return (
            pd.DataFrame(columns=SNAPSHOT_COLUMNS),
            f"snapshot CSV at {p} missing required columns: {missing}",
        )
    return df[SNAPSHOT_COLUMNS], None


# ---------------------------------------------------------------------------
# 5) summarize_health_timeline
# ---------------------------------------------------------------------------
def summarize_health_timeline(df: pd.DataFrame) -> Dict[str, Any]:
    """Compact summary the dashboard / CLI render. Empty frames yield
    safe defaults — no exceptions."""
    if df is None or df.empty:
        return {
            "row_count": 0,
            "latest_snapshot_timestamp": None,
            "latest_safety_lock_status": "unknown",
            "latest_system_health_fail_count": 0,
            "latest_strategy_pass_count": 0,
            "latest_portfolio_risk_classification": "UNKNOWN",
            "warnings": ["no snapshots yet — run "
                          "`python main.py write_health_snapshot`"],
        }
    last = df.iloc[-1]
    warnings: List[str] = []
    notes_val = last.get("notes")
    if isinstance(notes_val, str) and notes_val.strip():
        warnings.append(f"latest snapshot notes: {notes_val.strip()}")
    if int(last.get("system_health_fail_count", 0)) > 0:
        warnings.append("latest system_health has FAIL rows")
    if str(last.get("safety_lock_status", "")).lower() != "locked":
        warnings.append("safety lock not 'locked' in latest snapshot")
    return {
        "row_count": int(len(df)),
        "latest_snapshot_timestamp": str(last.get("snapshot_timestamp")),
        "latest_safety_lock_status": str(last.get("safety_lock_status",
                                                       "unknown")),
        "latest_system_health_fail_count": int(
            last.get("system_health_fail_count", 0)),
        "latest_strategy_pass_count": int(
            last.get("strategy_pass_count", 0)),
        "latest_portfolio_risk_classification": str(
            last.get("portfolio_risk_classification", "UNKNOWN")),
        "warnings": warnings,
    }
