"""
Donchian-style breakout (long-only).

Entry: close breaks above the rolling N-period high (excluding the current
       bar — uses `high.shift(1).rolling(N).max()` so we don't peek).
Exit:  close drops below the rolling M-period low (also shifted).

Default: 20-bar entry, 10-bar exit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .base import Strategy, Signal, BUY, SELL, HOLD, SKIP


@dataclass(frozen=True)
class BreakoutConfig:
    entry_window: int = 20
    exit_window: int = 10


class BreakoutStrategy(Strategy):
    name = "breakout"

    def __init__(self, entry_window: int = 20, exit_window: int = 10) -> None:
        if entry_window < 2 or exit_window < 2:
            raise ValueError("breakout: windows must be >= 2")
        self.cfg = BreakoutConfig(entry_window=entry_window,
                                  exit_window=exit_window)

    def min_history(self, cfg: Any) -> int:
        return max(self.cfg.entry_window, self.cfg.exit_window) + 5

    def prepare(self, df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
        out = df.copy()
        # Shift by 1 so we use only data strictly before the current bar — no
        # lookahead, no "breaks above its own high".
        out["roll_high"] = (out["high"].shift(1)
                            .rolling(self.cfg.entry_window).max())
        out["roll_low"] = (out["low"].shift(1)
                           .rolling(self.cfg.exit_window).min())
        return out

    def signal_for_row(self, asset: str, row: pd.Series, in_position: bool,
                       cfg: Any) -> Signal:
        ts = int(row["timestamp"]) if pd.notna(row.get("timestamp")) else 0
        close = float(row["close"])
        roll_high = row.get("roll_high")
        roll_low = row.get("roll_low")

        if pd.isna(roll_high) or pd.isna(roll_low):
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=SKIP, price=close,
                reason="insufficient breakout-window history",
            )

        if in_position:
            if close < roll_low:
                return Signal(
                    asset=asset, timestamp=ts, datetime=row.get("datetime"),
                    action=SELL, price=close,
                    reason=(f"breakout exit: close {close:.2f} < "
                            f"{self.cfg.exit_window}-low {roll_low:.2f}"),
                )
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=HOLD, price=close, reason="in position, no exit",
            )

        if close > roll_high:
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=BUY, price=close,
                reason=(f"breakout entry: close {close:.2f} > "
                        f"{self.cfg.entry_window}-high {roll_high:.2f}"),
            )
        return Signal(
            asset=asset, timestamp=ts, datetime=row.get("datetime"),
            action=HOLD, price=close, reason="no breakout",
        )
