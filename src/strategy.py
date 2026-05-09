"""
Deterministic, long-only strategy.

The strategy is INTENTIONALLY simple. It is NOT trying to be profitable on its
own — its job is to give us a signal stream we can put through fees, slippage,
and risk controls to see whether anything survives.

Inputs : a DataFrame with OHLCV + indicator columns (see indicators.py).
Output : a list of `Signal` objects (one per evaluated candle).

The strategy NEVER fills, NEVER touches cash, NEVER decides position size.
That is the risk engine's job.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional

import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Action labels
# ---------------------------------------------------------------------------
BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"
SKIP = "SKIP"


@dataclass
class Signal:
    asset: str
    timestamp: int            # ms epoch (UTC)
    datetime: pd.Timestamp
    action: str               # BUY / SELL / HOLD / SKIP
    price: float              # signal-bar close (NOT the fill price)
    reason: str
    rsi: float
    ma50: float
    ma200: float
    atr_pct: float
    trend_status: str         # 'up' if close > ma200 else 'down'
    volatility_status: str    # 'normal' or 'high'

    def to_dict(self) -> dict:
        d = asdict(self)
        d["datetime"] = self.datetime.isoformat() if pd.notna(self.datetime) else None
        return d


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def _classify_trend(close: float, ma200: float) -> str:
    if pd.isna(ma200):
        return "unknown"
    return "up" if close > ma200 else "down"


def _classify_volatility(atr_pct: float, threshold: float) -> str:
    if pd.isna(atr_pct):
        return "unknown"
    return "high" if atr_pct > threshold else "normal"


def _row_signal(
    asset: str,
    row: pd.Series,
    in_position: bool,
    cfg: config.StrategyConfig,
) -> Signal:
    """Decide what action this candle warrants, given current position state."""
    close = float(row["close"])
    rsi = float(row["rsi"]) if pd.notna(row["rsi"]) else float("nan")
    ma50 = float(row["ma50"]) if pd.notna(row["ma50"]) else float("nan")
    ma200 = float(row["ma200"]) if pd.notna(row["ma200"]) else float("nan")
    atr_pct = float(row["atr_pct"]) if pd.notna(row["atr_pct"]) else float("nan")

    trend_status = _classify_trend(close, ma200)
    vol_status = _classify_volatility(atr_pct, cfg.atr_pct_max)

    # Need full warmup before we look at anything.
    if pd.isna(rsi) or pd.isna(ma200) or pd.isna(ma50) or pd.isna(atr_pct):
        action, reason = SKIP, "insufficient indicator history"
    elif in_position:
        # ----- Exit logic -----
        if rsi > cfg.rsi_sell_threshold:
            action, reason = SELL, f"rsi {rsi:.1f} > sell threshold {cfg.rsi_sell_threshold}"
        elif close < ma50:
            action, reason = SELL, f"close {close:.2f} < ma50 {ma50:.2f}"
        else:
            action, reason = HOLD, "in position, no exit trigger"
    else:
        # ----- Entry logic -----
        if trend_status != "up":
            action, reason = HOLD, f"trend filter: close {close:.2f} <= ma200 {ma200:.2f}"
        elif vol_status == "high":
            action, reason = SKIP, f"volatility filter: atr_pct {atr_pct:.2f}% > {cfg.atr_pct_max}%"
        elif rsi < cfg.rsi_buy_threshold:
            action, reason = BUY, f"rsi {rsi:.1f} < buy threshold {cfg.rsi_buy_threshold} and trend up"
        else:
            action, reason = HOLD, f"no entry trigger (rsi {rsi:.1f})"

    return Signal(
        asset=asset,
        timestamp=int(row["timestamp"]) if pd.notna(row.get("timestamp")) else 0,
        datetime=row.get("datetime"),
        action=action,
        price=close,
        reason=reason,
        rsi=rsi,
        ma50=ma50,
        ma200=ma200,
        atr_pct=atr_pct,
        trend_status=trend_status,
        volatility_status=vol_status,
    )


def generate_signals(
    df_with_indicators: pd.DataFrame,
    asset: str,
    position_states: Optional[List[bool]] = None,
    cfg: config.StrategyConfig | None = None,
) -> List[Signal]:
    """Generate one Signal per candle.

    `position_states` is an optional pre-computed list of booleans aligned with
    `df_with_indicators` rows. If omitted, we assume "no open position" for
    every bar — which is fine for inspecting raw signals but not for
    backtesting. The backtester passes the live position state per-bar via
    `signal_for_row` instead.
    """
    cfg = cfg or config.STRATEGY
    if position_states is None:
        position_states = [False] * len(df_with_indicators)
    if len(position_states) != len(df_with_indicators):
        raise ValueError("position_states must align with df rows")

    out: List[Signal] = []
    for (_, row), in_pos in zip(df_with_indicators.iterrows(), position_states):
        out.append(_row_signal(asset, row, in_pos, cfg))
    return out


def signal_for_row(
    asset: str,
    row: pd.Series,
    in_position: bool,
    cfg: config.StrategyConfig | None = None,
) -> Signal:
    """Single-row variant used by the backtester loop and the paper trader."""
    return _row_signal(asset, row, in_position, cfg or config.STRATEGY)
