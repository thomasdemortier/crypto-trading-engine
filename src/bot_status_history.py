"""
Bot status history — append-only audit log of every `bot_status` snapshot.

Each call to `record_status()` reads the current `bot_status.compute_status()`
and appends one row to `results/bot_status_history.csv`. Identical-
timestamp rows (re-runs in the same second) are deduplicated; everything
else is preserved.

Rules:
    * No API keys, no secrets, no broker keys are recorded — even if
      `BotStatus.api_keys_loaded` is True, the value is the boolean
      flag, never the key value.
    * If the file does not exist it is created with the documented
      header. If it exists, only the new row(s) are appended.
    * No execution. No Kraken. No order plumbing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import bot_status, config, utils

logger = utils.get_logger("cte.bot_status_history")


HISTORY_COLUMNS: List[str] = [
    "timestamp",
    "bot_mode",
    "execution_enabled",
    "paper_trading_enabled",
    "kraken_connected",
    "api_keys_loaded",
    "active_strategy",
    "active_strategy_verdict",
    "safety_lock_status",
    "reason_execution_blocked",
    "latest_data_update",
    "stale_data_warning",
]
OUTPUT_FILENAME = "bot_status_history.csv"


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------
def _output_path() -> Path:
    return config.RESULTS_DIR / OUTPUT_FILENAME


def _row_from_status(snap: bot_status.BotStatus) -> Dict[str, Any]:
    """Project the relevant fields out of `BotStatus`. Never carries
    raw key values — `api_keys_loaded` is a boolean flag only."""
    return {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "bot_mode": snap.bot_mode,
        "execution_enabled": bool(snap.execution_enabled),
        "paper_trading_enabled": bool(snap.paper_trading_enabled),
        "kraken_connected": bool(snap.kraken_connected),
        "api_keys_loaded": bool(snap.api_keys_loaded),
        "active_strategy": snap.active_strategy,
        "active_strategy_verdict": snap.active_strategy_verdict,
        "safety_lock_status": snap.safety_lock_status,
        "reason_execution_blocked": snap.reason_execution_blocked,
        "latest_data_update": snap.latest_data_update,
        "stale_data_warning": bool(snap.stale_data_warning),
    }


def _load_existing() -> pd.DataFrame:
    p = _output_path()
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    try:
        df = pd.read_csv(p)
    except Exception as e:  # noqa: BLE001
        logger.warning("status history unreadable; starting fresh (%s)", e)
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    # Re-align to the documented schema (older runs may have a subset).
    for col in HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[HISTORY_COLUMNS]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def record_status(save: bool = True) -> pd.DataFrame:
    """Append the current bot status to the history CSV (creating it if
    needed). Returns the full history frame after the append. Identical-
    timestamp rows are deduplicated."""
    snap = bot_status.compute_status()
    new_row = _row_from_status(snap)
    existing = _load_existing()
    # Drop any prior row with the same timestamp string (shouldn't
    # happen unless the user calls this twice in the same second).
    if not existing.empty:
        existing = existing[existing["timestamp"] != new_row["timestamp"]]
    out = pd.concat([existing, pd.DataFrame([new_row], columns=HISTORY_COLUMNS)],
                     ignore_index=True)
    if save:
        utils.write_df(out, _output_path())
    return out


def load_history() -> pd.DataFrame:
    return _load_existing()
