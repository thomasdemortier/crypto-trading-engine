"""
The incumbent RSI/MA/ATR strategy. Logic preserved verbatim from the
original `src/strategy.py` so that all existing tests, paper-trader output,
and saved backtests remain identical.

Entry rule: trend up (close > MA200) AND volatility OK (atr_pct <= max)
            AND RSI < buy threshold.
Exit rule:  in position AND (RSI > sell threshold OR close < MA50).
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from .. import config, indicators
from .base import Strategy, Signal, BUY, SELL, HOLD, SKIP


def _classify_trend(close: float, ma200: float) -> str:
    if pd.isna(ma200):
        return "unknown"
    return "up" if close > ma200 else "down"


def _classify_volatility(atr_pct: float, threshold: float) -> str:
    if pd.isna(atr_pct):
        return "unknown"
    return "high" if atr_pct > threshold else "normal"


def _row_signal(asset: str, row: pd.Series, in_position: bool,
                cfg: config.StrategyConfig) -> Signal:
    close = float(row["close"])
    rsi = float(row["rsi"]) if pd.notna(row["rsi"]) else float("nan")
    ma50 = float(row["ma50"]) if pd.notna(row["ma50"]) else float("nan")
    ma200 = float(row["ma200"]) if pd.notna(row["ma200"]) else float("nan")
    atr_pct = float(row["atr_pct"]) if pd.notna(row["atr_pct"]) else float("nan")

    trend_status = _classify_trend(close, ma200)
    vol_status = _classify_volatility(atr_pct, cfg.atr_pct_max)

    if pd.isna(rsi) or pd.isna(ma200) or pd.isna(ma50) or pd.isna(atr_pct):
        action, reason = SKIP, "insufficient indicator history"
    elif in_position:
        if rsi > cfg.rsi_sell_threshold:
            action, reason = SELL, f"rsi {rsi:.1f} > sell threshold {cfg.rsi_sell_threshold}"
        elif close < ma50:
            action, reason = SELL, f"close {close:.2f} < ma50 {ma50:.2f}"
        else:
            action, reason = HOLD, "in position, no exit trigger"
    else:
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
        action=action, price=close, reason=reason,
        rsi=rsi, ma50=ma50, ma200=ma200, atr_pct=atr_pct,
        trend_status=trend_status, volatility_status=vol_status,
    )


class RsiMaAtrStrategy(Strategy):
    name = "rsi_ma_atr"

    def __init__(self, cfg: Optional[config.StrategyConfig] = None) -> None:
        self._cfg_override = cfg

    def _resolve(self, cfg: Any) -> config.StrategyConfig:
        return cfg or self._cfg_override or config.STRATEGY

    def min_history(self, cfg: Any) -> int:
        return self._resolve(cfg).min_history_candles

    def prepare(self, df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
        return indicators.add_indicators(df, self._resolve(cfg))

    def signal_for_row(self, asset: str, row: pd.Series, in_position: bool,
                       cfg: Any) -> Signal:
        return _row_signal(asset, row, in_position, self._resolve(cfg))
