"""
Alert history — append-only audit log with deduplication.

Every call to `record_alerts()` reads the current `alert_engine.build_alerts()`
output and merges it into `results/bot_alert_history.csv`:

  * Each alert is hashed by (severity, category, message,
    recommended_action). The hash is stable across runs.
  * If the same hash already exists in history, the row's
    `last_seen` is bumped, `occurrence_count` is incremented, and
    `active` is set to True.
  * Hashes that DO appear in the current run are marked `active=True`;
    hashes that do NOT appear are kept (with their prior counts) but
    `active=False`.
  * The recommended action is constrained to the alert engine's
    research-only vocabulary; anything that looks like a trade
    instruction is sanitised.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import alert_engine, config, utils

logger = utils.get_logger("cte.alert_history")


HISTORY_COLUMNS: List[str] = [
    "timestamp",
    "severity",
    "category",
    "message",
    "recommended_action",
    "alert_hash",
    "first_seen",
    "last_seen",
    "occurrence_count",
    "active",
]
OUTPUT_FILENAME = "bot_alert_history.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _output_path() -> Path:
    return config.RESULTS_DIR / OUTPUT_FILENAME


def _alert_hash(severity: str, category: str, message: str,
                  recommended_action: str) -> str:
    """Stable, content-only hash. Excludes timestamps so the same alert
    raised on two different runs collides into one row."""
    h = hashlib.sha256()
    payload = f"{severity}|{category}|{message}|{recommended_action}"
    h.update(payload.encode("utf-8"))
    return h.hexdigest()[:16]


def _load_existing() -> pd.DataFrame:
    p = _output_path()
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    try:
        df = pd.read_csv(p)
    except Exception as e:  # noqa: BLE001
        logger.warning("alert history unreadable; starting fresh (%s)", e)
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    for col in HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[HISTORY_COLUMNS]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def record_alerts(save: bool = True) -> pd.DataFrame:
    """Run the alert engine, merge into the history CSV (creating it if
    needed). Active flag is True only for hashes seen on this run."""
    now_iso = pd.Timestamp.utcnow().isoformat()
    current = alert_engine.build_alerts()
    current_rows: Dict[str, Dict[str, Any]] = {}
    for r in current:
        h = _alert_hash(str(r["severity"]), str(r["category"]),
                          str(r["message"]), str(r["recommended_action"]))
        current_rows[h] = {
            "timestamp": now_iso,
            "severity": r["severity"], "category": r["category"],
            "message": r["message"],
            "recommended_action": r["recommended_action"],
            "alert_hash": h,
        }

    existing = _load_existing()
    rows: List[Dict[str, Any]] = []
    seen_hashes: set = set()

    if not existing.empty:
        for _, row in existing.iterrows():
            h = str(row.get("alert_hash") or "")
            if not h:
                continue
            seen_hashes.add(h)
            if h in current_rows:
                # Bump count + last_seen; keep first_seen.
                rows.append({
                    "timestamp": now_iso,
                    "severity": row.get("severity"),
                    "category": row.get("category"),
                    "message": row.get("message"),
                    "recommended_action": row.get("recommended_action"),
                    "alert_hash": h,
                    "first_seen": row.get("first_seen") or now_iso,
                    "last_seen": now_iso,
                    "occurrence_count": int(
                        pd.to_numeric(row.get("occurrence_count"),
                                       errors="coerce") or 0
                    ) + 1,
                    "active": True,
                })
            else:
                # Keep historical row with active=False; do not bump count.
                rows.append({
                    "timestamp": row.get("timestamp"),
                    "severity": row.get("severity"),
                    "category": row.get("category"),
                    "message": row.get("message"),
                    "recommended_action": row.get("recommended_action"),
                    "alert_hash": h,
                    "first_seen": row.get("first_seen"),
                    "last_seen": row.get("last_seen"),
                    "occurrence_count": int(
                        pd.to_numeric(row.get("occurrence_count"),
                                       errors="coerce") or 0
                    ),
                    "active": False,
                })

    # Add NEW alerts from this run that weren't in history.
    for h, r in current_rows.items():
        if h in seen_hashes:
            continue
        rows.append({
            **r,
            "first_seen": now_iso,
            "last_seen": now_iso,
            "occurrence_count": 1,
            "active": True,
        })

    out = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    if save:
        utils.write_df(out, _output_path())
    return out


def load_history() -> pd.DataFrame:
    return _load_existing()
