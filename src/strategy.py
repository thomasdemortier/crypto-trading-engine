"""
Back-compatibility shim.

The incumbent RSI/MA/ATR strategy lives at `src.strategies.rsi_ma_atr`.
This module re-exports its public surface so existing call sites
(`from src import strategy`, `strategy.signal_for_row(...)`,
`strategy.Signal(...)`) keep working unchanged.

New code should import from `src.strategies` directly.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from . import config
from .strategies.base import BUY, SELL, HOLD, SKIP, Signal
from .strategies.rsi_ma_atr import _row_signal as _legacy_row_signal
from .strategies.rsi_ma_atr import RsiMaAtrStrategy

__all__ = [
    "BUY", "SELL", "HOLD", "SKIP", "Signal",
    "signal_for_row", "generate_signals", "RsiMaAtrStrategy",
]


def signal_for_row(
    asset: str,
    row: pd.Series,
    in_position: bool,
    cfg: config.StrategyConfig | None = None,
) -> Signal:
    """Single-row signal — used by the backtester loop and the paper trader."""
    return _legacy_row_signal(asset, row, in_position, cfg or config.STRATEGY)


def generate_signals(
    df_with_indicators: pd.DataFrame,
    asset: str,
    position_states: Optional[List[bool]] = None,
    cfg: config.StrategyConfig | None = None,
) -> List[Signal]:
    """Generate one Signal per candle. Identical behaviour to the legacy
    module — kept here so notebooks and external callers keep working."""
    cfg = cfg or config.STRATEGY
    if position_states is None:
        position_states = [False] * len(df_with_indicators)
    if len(position_states) != len(df_with_indicators):
        raise ValueError("position_states must align with df rows")
    out: List[Signal] = []
    for (_, row), in_pos in zip(df_with_indicators.iterrows(), position_states):
        out.append(_legacy_row_signal(asset, row, in_pos, cfg))
    return out
