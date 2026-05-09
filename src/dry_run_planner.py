"""
Dry-run trade planner — read-only, never produces orders.

Reads the most recent saved target weights from any strategy's
`*_weights.csv` artifact and the latest cached prices from
`data/raw/*_1d.csv`. Outputs a "what would the target rebalance look
like, IF we were trading?" CSV with the columns:

    timestamp, strategy_name, asset,
    current_weight, target_weight,
    theoretical_action, theoretical_notional,
    reason, mode, execution_status

Hard rules:
    * `mode` is always `DRY_RUN_ONLY`.
    * `execution_status` is always `BLOCKED_NO_PASS_STRATEGY` whenever
      the safety lock blocks execution. This is the case on this branch
      by construction.
    * The module never imports a broker, never opens a network
      connection, and never reads / sets API keys.
    * If no `*_weights.csv` exists, returns a single-row "no plan
      available" message (mode + status still set per spec).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from . import config, safety_lock, utils


PLAN_COLUMNS: List[str] = [
    "timestamp", "strategy_name", "asset",
    "current_weight", "target_weight",
    "theoretical_action", "theoretical_notional",
    "reason", "mode", "execution_status",
]

MODE_DRY_RUN_ONLY = "DRY_RUN_ONLY"
EXECUTION_STATUS_BLOCKED = "BLOCKED_NO_PASS_STRATEGY"

# When the safety lock is fully released AND a PASS verdict exists,
# we'd report a different `execution_status` — but on this branch
# `_resolve_execution_status` always returns the blocked constant.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ts_now_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()


def _resolve_execution_status() -> str:
    """Even if a future PASS exists, this stays blocked unless the
    safety lock independently confirms unlock. Currently always
    returns the blocked constant by construction."""
    if safety_lock.is_execution_allowed():
        return "READY_BUT_NOT_AUTOMATED"  # never reached on this branch
    return EXECUTION_STATUS_BLOCKED


def _list_weight_artifacts() -> List[Path]:
    p = config.RESULTS_DIR
    if not p.exists():
        return []
    return sorted(p.glob("*_weights*.csv"))


def _parse_weights_string(raw: str) -> Optional[Dict[str, float]]:
    """Accept either of two formats the portfolio backtester has used:

      1. `KEY=VALUE,KEY=VALUE` (current format, e.g.
         `"BTC/USDT=0.3,ETH/USDT=0.7"`).
      2. Python dict literal (e.g. `"{'BTC/USDT': 0.3}"`) for any
         future variant — parsed via `ast.literal_eval`.

    Returns None if the input doesn't match either shape."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = raw.strip()
    # Format 2: dict literal.
    if s.startswith("{"):
        try:
            import ast
            d = ast.literal_eval(s)
            if isinstance(d, dict):
                return {str(k): float(v) for k, v in d.items()}
        except Exception:  # noqa: BLE001
            return None
        return None
    # Format 1: KEY=VALUE pairs.
    out: Dict[str, float] = {}
    for piece in s.split(","):
        piece = piece.strip()
        if "=" not in piece:
            continue
        k, _, v = piece.partition("=")
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            continue
    return out or None


def _load_latest_target_weights(
    artifact: Path,
) -> Optional[Dict[str, Any]]:
    """A `*_weights.csv` from the portfolio_backtester carries the
    weights in column `weights_json` (current format) or `weights`
    (older variant). We pick the LAST row whose `filled` is True (or
    just the last row if no `filled` column) and parse the weights."""
    try:
        df = pd.read_csv(artifact)
    except Exception:  # noqa: BLE001
        return None
    if df.empty:
        return None
    if "filled" in df.columns:
        valid = df[df["filled"].astype(bool)]
        if not valid.empty:
            df = valid
    # Drop rows where the weights cell is empty.
    weights_col = next((c for c in ("weights_json", "weights")
                          if c in df.columns), None)
    if weights_col is None:
        return None
    df = df[df[weights_col].astype(str).str.strip() != ""]
    df = df[df[weights_col].notna()]
    if df.empty:
        return None
    last = df.iloc[-1]
    parsed = _parse_weights_string(str(last[weights_col]))
    if not parsed:
        return None
    ts = int(last.get("timestamp", 0)) if pd.notna(last.get("timestamp")) else 0
    return {"timestamp": ts, "weights": parsed}


def _latest_price_usd(asset: str) -> Optional[float]:
    safe = asset.replace("/", "_").replace(":", "_")
    p = config.DATA_RAW_DIR / f"{safe}_1d.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if df.empty or "close" not in df.columns:
            return None
        return float(df["close"].iloc[-1])
    except Exception:  # noqa: BLE001
        return None


def _strategy_name_from_artifact(p: Path) -> str:
    return p.stem.replace("_weights_history", "").replace("_weights", "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_plan(
    starting_capital: float = 10_000.0,
    artifacts: Optional[Sequence[Path]] = None,
) -> List[Dict[str, Any]]:
    """Build the dry-run plan from the most-recent saved target weights
    of every strategy that has a `*_weights*.csv` artifact. Always
    returns at least one row; when nothing is available, the row is a
    single 'no plan available' marker carrying the locked mode/status.
    """
    paths = list(artifacts) if artifacts else _list_weight_artifacts()
    if not paths:
        return [{
            "timestamp": _ts_now_iso(),
            "strategy_name": "(none)",
            "asset": "n/a",
            "current_weight": 0.0,
            "target_weight": 0.0,
            "theoretical_action": "no_plan_available",
            "theoretical_notional": 0.0,
            "reason": "no `*_weights.csv` artifact found in results/ — "
                      "run a portfolio backtest first",
            "mode": MODE_DRY_RUN_ONLY,
            "execution_status": _resolve_execution_status(),
        }]
    rows: List[Dict[str, Any]] = []
    status = _resolve_execution_status()
    blocked_reason = (safety_lock.reason_blocked() if status ==
                       EXECUTION_STATUS_BLOCKED else "")
    for p in paths:
        bundle = _load_latest_target_weights(p)
        strat = _strategy_name_from_artifact(p)
        if bundle is None or not bundle["weights"]:
            rows.append({
                "timestamp": _ts_now_iso(),
                "strategy_name": strat,
                "asset": "n/a",
                "current_weight": 0.0,
                "target_weight": 0.0,
                "theoretical_action": "all_cash",
                "theoretical_notional": 0.0,
                "reason": "latest target weights are empty (cash) — "
                            "no theoretical trade",
                "mode": MODE_DRY_RUN_ONLY,
                "execution_status": status,
            })
            continue
        ts_iso = _ts_now_iso()
        for asset, w in bundle["weights"].items():
            price = _latest_price_usd(asset)
            target_value = float(starting_capital) * float(w)
            current_value = 0.0  # dry-run has no real "current" position
            delta_value = target_value - current_value
            theoretical_action = (
                "BUY" if delta_value > 0 else
                ("SELL" if delta_value < 0 else "HOLD")
            )
            reason_parts = []
            if blocked_reason:
                reason_parts.append(f"safety_lock: {blocked_reason}")
            if price is None:
                reason_parts.append("no cached price for asset")
            reason = (" | ".join(reason_parts) if reason_parts
                        else "weight delta from latest backtest snapshot")
            rows.append({
                "timestamp": ts_iso,
                "strategy_name": strat,
                "asset": str(asset),
                "current_weight": 0.0,
                "target_weight": float(w),
                "theoretical_action": theoretical_action,
                "theoretical_notional": round(abs(delta_value), 2),
                "reason": reason,
                "mode": MODE_DRY_RUN_ONLY,
                "execution_status": status,
            })
    return rows


def write_plan(save: bool = True,
                starting_capital: float = 10_000.0) -> pd.DataFrame:
    rows = build_plan(starting_capital=starting_capital)
    df = pd.DataFrame(rows, columns=PLAN_COLUMNS)
    if save:
        utils.write_df(df, config.RESULTS_DIR / "dry_run_trade_plan.csv")
    return df
