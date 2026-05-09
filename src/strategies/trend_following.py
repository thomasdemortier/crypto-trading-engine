"""
Trend-following with RSI-recovery entries (long-only).

Entry conditions (ALL must hold):
  * close above slow MA (default MA200)
  * mid MA above slow MA (default MA50 > MA200)
  * slow MA slope is positive
  * close above fast MA (default MA20)
  * RSI crosses upward through `rsi_recovery_level` (default 40)
    after being below it on the prior bar — looks for momentum recovery,
    not just oversold weakness

Exit conditions (ANY triggers):
  * close drops below mid MA (default MA50)
  * RSI drops below 35 after entry
  * trailing ATR stop (handled by the risk engine's stop-loss; the
    strategy emits SELL on close < mid_ma which is the primary trail)

This strategy uses the standard risk engine — no leverage, no shorting,
no bypass. Position sizing, fees, slippage and stop-loss are all unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from .. import indicators
from .base import Strategy, Signal, BUY, SELL, HOLD, SKIP


@dataclass(frozen=True)
class TrendFollowingConfig:
    fast_ma: int = 20
    mid_ma: int = 50
    slow_ma: int = 200
    rsi_period: int = 14
    rsi_recovery_level: float = 40.0
    rsi_exit_level: float = 35.0
    atr_period: int = 14
    slope_window: int = 20  # bars used to estimate slow_ma slope


class TrendFollowingStrategy(Strategy):
    name = "trend_following"

    def __init__(self, cfg: Optional[TrendFollowingConfig] = None) -> None:
        self.cfg = cfg or TrendFollowingConfig()

    def min_history(self, cfg: Any) -> int:
        return self.cfg.slow_ma + self.cfg.slope_window + 5

    def prepare(self, df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
        c = self.cfg
        out = df.copy()
        out["tf_fast"] = indicators.sma(out["close"], c.fast_ma)
        out["tf_mid"] = indicators.sma(out["close"], c.mid_ma)
        out["tf_slow"] = indicators.sma(out["close"], c.slow_ma)
        out["tf_slow_prev"] = out["tf_slow"].shift(c.slope_window)
        out["tf_rsi"] = indicators.rsi(out["close"], c.rsi_period)
        out["tf_rsi_prev"] = out["tf_rsi"].shift(1)
        out["tf_atr"] = indicators.atr(
            out["high"], out["low"], out["close"], c.atr_period,
        )
        return out

    def signal_for_row(
        self, asset: str, row: pd.Series, in_position: bool, cfg: Any,
    ) -> Signal:
        c = self.cfg
        ts = int(row["timestamp"]) if pd.notna(row.get("timestamp")) else 0
        close = float(row["close"])
        fast = row.get("tf_fast")
        mid = row.get("tf_mid")
        slow = row.get("tf_slow")
        slow_prev = row.get("tf_slow_prev")
        rsi_now = row.get("tf_rsi")
        rsi_prev = row.get("tf_rsi_prev")

        if any(pd.isna(x) for x in (fast, mid, slow, slow_prev, rsi_now, rsi_prev)):
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=SKIP, price=close,
                reason="insufficient indicator history",
            )

        slow_slope = float(slow) - float(slow_prev)

        if in_position:
            if close < float(mid):
                return Signal(
                    asset=asset, timestamp=ts, datetime=row.get("datetime"),
                    action=SELL, price=close,
                    reason=(f"close {close:.2f} < ma{c.mid_ma} {float(mid):.2f}"),
                    rsi=float(rsi_now), ma50=float(mid), ma200=float(slow),
                )
            if float(rsi_now) < c.rsi_exit_level:
                return Signal(
                    asset=asset, timestamp=ts, datetime=row.get("datetime"),
                    action=SELL, price=close,
                    reason=(f"rsi {float(rsi_now):.1f} < exit "
                            f"{c.rsi_exit_level}"),
                    rsi=float(rsi_now), ma50=float(mid), ma200=float(slow),
                )
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=HOLD, price=close, reason="trend ok, no exit",
                rsi=float(rsi_now), ma50=float(mid), ma200=float(slow),
            )

        # Entry filters
        in_uptrend = (close > float(slow)) and (float(mid) > float(slow)) \
            and (slow_slope > 0)
        if not in_uptrend:
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=HOLD, price=close, reason="no confirmed uptrend",
                rsi=float(rsi_now), ma50=float(mid), ma200=float(slow),
            )
        above_fast = close > float(fast)
        rsi_recovery = (
            float(rsi_prev) < c.rsi_recovery_level
            and float(rsi_now) >= c.rsi_recovery_level
        )
        if above_fast and rsi_recovery:
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=BUY, price=close,
                reason=(f"rsi recovery {float(rsi_prev):.1f}->{float(rsi_now):.1f} "
                        f"in confirmed uptrend"),
                rsi=float(rsi_now), ma50=float(mid), ma200=float(slow),
            )
        return Signal(
            asset=asset, timestamp=ts, datetime=row.get("datetime"),
            action=HOLD, price=close,
            reason="uptrend ok, waiting for rsi recovery + close>fast_ma",
            rsi=float(rsi_now), ma50=float(mid), ma200=float(slow),
        )
