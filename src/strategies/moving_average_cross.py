"""
Moving-average crossover (long-only).

Entry: prior bar had `ma_fast <= ma_slow` AND current bar has
       `ma_fast > ma_slow` (a fresh up-cross).
Exit:  prior bar had `ma_fast >= ma_slow` AND current bar has
       `ma_fast < ma_slow` (a fresh down-cross).

Default: 50 / 200 — the "golden / death" pair.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .. import indicators
from .base import Strategy, Signal, BUY, SELL, HOLD, SKIP


@dataclass(frozen=True)
class MaCrossConfig:
    fast: int = 50
    slow: int = 200


class MovingAverageCrossStrategy(Strategy):
    name = "ma_cross"

    def __init__(self, fast: int = 50, slow: int = 200) -> None:
        if fast >= slow:
            raise ValueError("ma_cross: fast must be < slow")
        self.cfg = MaCrossConfig(fast=fast, slow=slow)

    def min_history(self, cfg: Any) -> int:
        return self.cfg.slow + 5

    def prepare(self, df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
        out = df.copy()
        out["ma_fast"] = indicators.sma(out["close"], self.cfg.fast)
        out["ma_slow"] = indicators.sma(out["close"], self.cfg.slow)
        out["ma_fast_prev"] = out["ma_fast"].shift(1)
        out["ma_slow_prev"] = out["ma_slow"].shift(1)
        return out

    def signal_for_row(self, asset: str, row: pd.Series, in_position: bool,
                       cfg: Any) -> Signal:
        ts = int(row["timestamp"]) if pd.notna(row.get("timestamp")) else 0
        close = float(row["close"])
        fast, slow = row.get("ma_fast"), row.get("ma_slow")
        fast_prev, slow_prev = row.get("ma_fast_prev"), row.get("ma_slow_prev")

        if any(pd.isna(x) for x in (fast, slow, fast_prev, slow_prev)):
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=SKIP, price=close,
                reason="insufficient MA history",
            )

        up_cross = (fast_prev <= slow_prev) and (fast > slow)
        down_cross = (fast_prev >= slow_prev) and (fast < slow)

        if in_position:
            if down_cross:
                return Signal(
                    asset=asset, timestamp=ts, datetime=row.get("datetime"),
                    action=SELL, price=close,
                    reason=(f"ma_cross down: ma{self.cfg.fast} {fast:.2f} "
                            f"< ma{self.cfg.slow} {slow:.2f}"),
                )
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=HOLD, price=close,
                reason="in position, no down-cross",
            )

        if up_cross:
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=BUY, price=close,
                reason=(f"ma_cross up: ma{self.cfg.fast} {fast:.2f} "
                        f"> ma{self.cfg.slow} {slow:.2f}"),
            )
        return Signal(
            asset=asset, timestamp=ts, datetime=row.get("datetime"),
            action=HOLD, price=close,
            reason="no fresh up-cross",
        )
