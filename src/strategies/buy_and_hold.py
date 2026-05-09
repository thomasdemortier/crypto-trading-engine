"""
Buy-and-hold baseline.

Buys once on the first eligible bar (after a tiny warmup) and never sells.
Useful as a research baseline against which any active strategy must
genuinely justify its trading and fee load.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .base import Strategy, Signal, BUY, HOLD


class BuyAndHoldStrategy(Strategy):
    name = "buy_and_hold"

    def min_history(self, cfg: Any) -> int:
        return 2  # nothing to compute; just need 2 bars to have an "open" next bar

    def prepare(self, df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
        # No indicators required.
        return df.copy()

    def signal_for_row(self, asset: str, row: pd.Series, in_position: bool,
                       cfg: Any) -> Signal:
        ts = int(row["timestamp"]) if pd.notna(row.get("timestamp")) else 0
        if in_position:
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=HOLD, price=float(row["close"]),
                reason="buy-and-hold: holding indefinitely",
            )
        return Signal(
            asset=asset, timestamp=ts, datetime=row.get("datetime"),
            action=BUY, price=float(row["close"]),
            reason="buy-and-hold: initial entry",
        )
