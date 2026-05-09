"""
Strategy base class + shared `Signal` dataclass.

A `Strategy` is a stateless signal factory. The backtester drives it bar by
bar; every emitted Signal is routed through the risk engine for execution.
Strategies never touch cash, never decide position size, never bypass risk
controls.

To add a new strategy, subclass `Strategy` and implement:
  * `name` — short identifier used in CSV outputs
  * `min_history(cfg)` — how many warmup bars are needed before signals fire
  * `prepare(df, cfg)` — return a copy of `df` with the indicator columns
                         this strategy needs
  * `signal_for_row(asset, row, in_position, cfg)` — return a `Signal`
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd


# Action labels — kept identical to the legacy `src/strategy.py` constants.
BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"
SKIP = "SKIP"


@dataclass
class Signal:
    """Strategy output for one bar. Identical schema to the legacy module."""
    asset: str
    timestamp: int
    datetime: pd.Timestamp
    action: str
    price: float
    reason: str
    rsi: float = float("nan")
    ma50: float = float("nan")
    ma200: float = float("nan")
    atr_pct: float = float("nan")
    trend_status: str = "unknown"
    volatility_status: str = "unknown"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["datetime"] = (
            self.datetime.isoformat() if pd.notna(self.datetime) else None
        )
        return d


class Strategy:
    """Abstract strategy. Subclasses MUST override the four methods below."""

    name: str = "abstract"

    def min_history(self, cfg: Any) -> int:
        raise NotImplementedError

    def prepare(self, df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
        """Return a copy of df with indicator columns this strategy needs."""
        raise NotImplementedError

    def signal_for_row(
        self,
        asset: str,
        row: pd.Series,
        in_position: bool,
        cfg: Any,
    ) -> Signal:
        raise NotImplementedError
