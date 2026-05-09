"""
Pullback continuation (long-only).

Buy controlled pullbacks INSIDE confirmed uptrends, not random dips.

Entry conditions (ALL must hold):
  * close > slow MA (confirmed uptrend, e.g. > MA200)
  * mid MA > slow MA (e.g. MA50 > MA200)
  * price has pulled back to within `pullback_band_pct` of `pullback_ma`
    (default: 1.5% of MA20) — i.e. close to the moving average from above
  * RSI in [`rsi_min`, `rsi_max`] band (default 35-55) — neither
    over-bought nor capitulating
  * current candle closes green (close > open)
  * ATR% below `max_atr_pct` (default 5%) — skip volatile dumps

Exit conditions (ANY triggers):
  * close < mid MA
  * RSI overextended (> 70) AND turns down (rsi_now < rsi_prev)
  * stop-loss handled by risk engine
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from .. import indicators
from .base import Strategy, Signal, BUY, SELL, HOLD, SKIP


@dataclass(frozen=True)
class PullbackContinuationConfig:
    pullback_ma: int = 20
    mid_ma: int = 50
    slow_ma: int = 200
    rsi_period: int = 14
    rsi_min: float = 35.0
    rsi_max: float = 55.0
    max_atr_pct: float = 5.0
    atr_period: int = 14
    pullback_band_pct: float = 1.5  # how close to pullback_ma we must be


class PullbackContinuationStrategy(Strategy):
    name = "pullback_continuation"

    def __init__(self, cfg: Optional[PullbackContinuationConfig] = None) -> None:
        self.cfg = cfg or PullbackContinuationConfig()

    def min_history(self, cfg: Any) -> int:
        return self.cfg.slow_ma + 5

    def prepare(self, df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
        c = self.cfg
        out = df.copy()
        out["pb_pull_ma"] = indicators.sma(out["close"], c.pullback_ma)
        out["pb_mid"] = indicators.sma(out["close"], c.mid_ma)
        out["pb_slow"] = indicators.sma(out["close"], c.slow_ma)
        out["pb_rsi"] = indicators.rsi(out["close"], c.rsi_period)
        out["pb_rsi_prev"] = out["pb_rsi"].shift(1)
        out["pb_atr"] = indicators.atr(
            out["high"], out["low"], out["close"], c.atr_period,
        )
        out["pb_atr_pct"] = (out["pb_atr"] / out["close"]) * 100.0
        return out

    def signal_for_row(
        self, asset: str, row: pd.Series, in_position: bool, cfg: Any,
    ) -> Signal:
        c = self.cfg
        ts = int(row["timestamp"]) if pd.notna(row.get("timestamp")) else 0
        close = float(row["close"])
        open_ = float(row["open"])
        pull_ma = row.get("pb_pull_ma")
        mid = row.get("pb_mid")
        slow = row.get("pb_slow")
        rsi_now = row.get("pb_rsi")
        rsi_prev = row.get("pb_rsi_prev")
        atr_pct = row.get("pb_atr_pct")

        if any(pd.isna(x) for x in (pull_ma, mid, slow, rsi_now, rsi_prev, atr_pct)):
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=SKIP, price=close,
                reason="insufficient indicator history",
            )

        if in_position:
            if close < float(mid):
                return Signal(
                    asset=asset, timestamp=ts, datetime=row.get("datetime"),
                    action=SELL, price=close,
                    reason=f"close {close:.2f} < ma{c.mid_ma} {float(mid):.2f}",
                    rsi=float(rsi_now), ma50=float(mid), ma200=float(slow),
                )
            overextended_turn = (
                float(rsi_now) > 70.0 and float(rsi_now) < float(rsi_prev)
            )
            if overextended_turn:
                return Signal(
                    asset=asset, timestamp=ts, datetime=row.get("datetime"),
                    action=SELL, price=close,
                    reason=(f"rsi turn from overextended "
                            f"{float(rsi_prev):.1f}->{float(rsi_now):.1f}"),
                    rsi=float(rsi_now), ma50=float(mid), ma200=float(slow),
                )
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=HOLD, price=close, reason="hold pullback trade",
                rsi=float(rsi_now), ma50=float(mid), ma200=float(slow),
            )

        # Entry filters
        in_uptrend = (close > float(slow)) and (float(mid) > float(slow))
        if not in_uptrend:
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=HOLD, price=close, reason="no confirmed uptrend",
                rsi=float(rsi_now), ma50=float(mid), ma200=float(slow),
            )
        if float(atr_pct) > c.max_atr_pct:
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=SKIP, price=close,
                reason=f"atr_pct {float(atr_pct):.2f}% > {c.max_atr_pct}%",
            )
        # Close to pullback MA from above (price has pulled back, not crashed).
        dist_pct = (close - float(pull_ma)) / float(pull_ma) * 100.0
        near_pullback = (0.0 <= dist_pct <= c.pullback_band_pct)
        rsi_in_band = c.rsi_min <= float(rsi_now) <= c.rsi_max
        green_candle = close > open_

        if near_pullback and rsi_in_band and green_candle:
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=BUY, price=close,
                reason=(f"pullback to ma{c.pullback_ma} "
                        f"({dist_pct:.2f}% above), rsi {float(rsi_now):.1f}"),
                rsi=float(rsi_now), ma50=float(mid), ma200=float(slow),
            )
        return Signal(
            asset=asset, timestamp=ts, datetime=row.get("datetime"),
            action=HOLD, price=close,
            reason="uptrend ok, waiting for pullback + rsi band + green close",
            rsi=float(rsi_now), ma50=float(mid), ma200=float(slow),
        )
