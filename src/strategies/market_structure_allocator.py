"""
Market-structure portfolio allocator.

Same `target_weights(asof_ts_ms, asset_frames, timeframe) -> dict[asset, weight]`
contract as `MomentumRotationStrategy` and `FundingRotationStrategy`.

The strategy reads the precomputed `market_structure_state` for the
latest signal row at or before `asof_ts_ms` and allocates by state:

    alt_risk_on    -> equal-weight Top-5 alts by 90d momentum
    btc_leadership -> 100 % BTC/USDT
    defensive      -> 100 % cash (return {})
    neutral        -> 100 % BTC/USDT (BTC by default in the absence of edge)
    unknown        -> 100 % cash

Hard rules:
    * Long-only. Weights ≥ 0, total weight ≤ 1.
    * No leverage, no margin, no shorting.
    * Assets without enough price history are excluded from any selection.
    * Missing signal at asof → cash.
    * Lookahead-free: only signal rows with `timestamp <= asof_ts_ms` are
      consulted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def _bars_per_day(timeframe: str) -> int:
    return {"1h": 24, "4h": 6, "1d": 1}.get(timeframe, 1)


@dataclass(frozen=True)
class MarketStructureAllocatorConfig:
    btc_asset: str = "BTC/USDT"
    alt_top_n: int = 5
    momentum_window_days: int = 90
    rebalance_frequency: str = "weekly"   # informational
    # The following alt list is the trading universe (BTC excluded).
    # Any symbol not present in `asset_frames` is silently dropped.
    alt_universe: Sequence[str] = (
        "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT",
        "XRP/USDT", "DOGE/USDT", "ADA/USDT", "LTC/USDT", "BNB/USDT",
    )
    min_history_days: int = 90  # need at least this for momentum scoring


class MarketStructureAllocatorStrategy:
    """Portfolio strategy: state-driven BTC / alt-basket / cash allocator."""

    name = "market_structure_allocator"

    def __init__(self, signals_df: pd.DataFrame,
                 cfg: Optional[MarketStructureAllocatorConfig] = None) -> None:
        self.cfg = cfg or MarketStructureAllocatorConfig()
        if signals_df is None or signals_df.empty:
            self._signals = pd.DataFrame()
        else:
            df = signals_df.copy()
            df["timestamp"] = pd.to_numeric(df["timestamp"]).astype("int64")
            df = df.sort_values("timestamp").reset_index(drop=True)
            self._signals = df

    # ---- helpers ----------------------------------------------------------
    def _signal_asof(self, asof_ts_ms: int) -> Optional[pd.Series]:
        if self._signals.empty:
            return None
        sub = self._signals[self._signals["timestamp"] <= int(asof_ts_ms)]
        if sub.empty:
            return None
        return sub.iloc[-1]

    def _has_enough_history(self, df: pd.DataFrame, asof_ts_ms: int,
                             bars_per_day: int) -> bool:
        if df is None or df.empty:
            return False
        sub = df[df["timestamp"] <= int(asof_ts_ms)]
        return len(sub) >= self.cfg.min_history_days * bars_per_day + 1

    def _alt_momentum(self, df: pd.DataFrame, asof_ts_ms: int,
                       bars_per_day: int) -> Optional[float]:
        if not self._has_enough_history(df, asof_ts_ms, bars_per_day):
            return None
        sub = df[df["timestamp"] <= int(asof_ts_ms)]
        n = self.cfg.momentum_window_days * bars_per_day
        last = float(sub["close"].iloc[-1])
        ago = float(sub["close"].iloc[-1 - n])
        if ago <= 0 or last <= 0:
            return None
        return (last / ago) - 1.0

    # ---- public API -------------------------------------------------------
    def target_weights(
        self,
        asof_ts_ms: int,
        asset_frames: Dict[str, pd.DataFrame],
        timeframe: str = "1d",
    ) -> Dict[str, float]:
        bars_per_day = _bars_per_day(timeframe)
        sig = self._signal_asof(asof_ts_ms)
        if sig is None:
            return {}
        state = sig.get("market_structure_state", "unknown")
        # `defensive` and `unknown` -> cash.
        if state in ("defensive", "unknown"):
            return {}
        # `btc_leadership` and `neutral` -> 100 % BTC.
        if state in ("btc_leadership", "neutral"):
            btc = self.cfg.btc_asset
            if btc not in asset_frames or asset_frames[btc] is None:
                return {}
            if not self._has_enough_history(asset_frames[btc],
                                              asof_ts_ms, bars_per_day):
                return {}
            return {btc: 1.0}
        # `alt_risk_on` -> equal-weight Top-N alt momentum.
        if state == "alt_risk_on":
            scores: Dict[str, float] = {}
            for alt in self.cfg.alt_universe:
                df = asset_frames.get(alt)
                m = self._alt_momentum(df, asof_ts_ms, bars_per_day)
                if m is None:
                    continue
                scores[alt] = m
            if not scores:
                return {}
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            top = ranked[: self.cfg.alt_top_n]
            n = len(top)
            if n == 0:
                return {}
            w = 1.0 / n
            return {asset: w for asset, _ in top}
        # Any other state -> cash (safe default).
        return {}
