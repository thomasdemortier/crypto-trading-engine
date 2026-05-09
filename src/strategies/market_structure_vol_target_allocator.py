"""
Vol-target market-structure allocator.

Same input as `MarketStructureAllocatorStrategy` (the precomputed
`market_structure_state` from `src.market_structure_signals`), but with
**softer exposure bands** instead of all-or-nothing cash:

    alt_risk_on    -> 70 % equal-weight Top-5 alts (by 90d momentum)
                       + 30 % BTC
    btc_leadership -> 100 % BTC
    neutral        -> 70 % BTC + 30 % cash
    defensive      -> 30 % BTC + 70 % cash
    unknown        -> 100 % cash

Hypothesis: the original binary allocator was too conservative — it
sat in cash 56 % of the time and missed the early innings of recovery
rallies. Tapering BTC exposure rather than cutting it should let the
strategy participate in BTC's positive long-run drift while still
de-risking when the signal turns red.

Hard rules carried over:
    * Long-only. Σ weights ≤ 1. No leverage, no margin, no shorts.
    * Same `target_weights(asof_ts_ms, asset_frames, timeframe)` contract
      as every other portfolio strategy in the project — fees + slippage
      come from the existing portfolio backtester.
    * Lookahead-free: `_signal_asof` slices the signal frame on
      `timestamp <= asof`.
    * State rules / signal computation are NOT touched in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import pandas as pd


def _bars_per_day(timeframe: str) -> int:
    return {"1h": 24, "4h": 6, "1d": 1}.get(timeframe, 1)


@dataclass(frozen=True)
class MarketStructureVolTargetConfig:
    btc_asset: str = "BTC/USDT"
    alt_top_n: int = 5
    momentum_window_days: int = 90
    rebalance_frequency: str = "weekly"  # informational only
    alt_universe: Sequence[str] = (
        "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT",
        "XRP/USDT", "DOGE/USDT", "ADA/USDT", "LTC/USDT", "BNB/USDT",
    )
    min_history_days: int = 90
    # Exposure bands per spec — fixed, not tuning knobs.
    alt_risk_on_alt_weight: float = 0.70
    alt_risk_on_btc_weight: float = 0.30
    neutral_btc_weight: float = 0.70
    defensive_btc_weight: float = 0.30


class MarketStructureVolTargetAllocatorStrategy:
    """Continuous-exposure variant of the market-structure allocator."""

    name = "market_structure_vol_target_allocator"

    def __init__(
        self,
        signals_df: pd.DataFrame,
        cfg: Optional[MarketStructureVolTargetConfig] = None,
    ) -> None:
        self.cfg = cfg or MarketStructureVolTargetConfig()
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

    def _btc_only(self, asset_frames: Dict[str, pd.DataFrame],
                   asof_ts_ms: int, bars_per_day: int,
                   weight: float) -> Dict[str, float]:
        df = asset_frames.get(self.cfg.btc_asset)
        if df is None or df.empty:
            return {}
        if not self._has_enough_history(df, asof_ts_ms, bars_per_day):
            return {}
        return {self.cfg.btc_asset: weight}

    def _top_alts(self, asset_frames: Dict[str, pd.DataFrame],
                   asof_ts_ms: int, bars_per_day: int,
                   total_weight: float) -> Dict[str, float]:
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
        w_each = total_weight / n
        return {asset: w_each for asset, _ in top}

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

        if state == "unknown":
            return {}

        if state == "btc_leadership":
            return self._btc_only(asset_frames, asof_ts_ms,
                                    bars_per_day, weight=1.0)

        if state == "neutral":
            return self._btc_only(asset_frames, asof_ts_ms,
                                    bars_per_day,
                                    weight=self.cfg.neutral_btc_weight)

        if state == "defensive":
            return self._btc_only(asset_frames, asof_ts_ms,
                                    bars_per_day,
                                    weight=self.cfg.defensive_btc_weight)

        if state == "alt_risk_on":
            alt_weights = self._top_alts(
                asset_frames, asof_ts_ms, bars_per_day,
                total_weight=self.cfg.alt_risk_on_alt_weight,
            )
            btc_part = self._btc_only(
                asset_frames, asof_ts_ms, bars_per_day,
                weight=self.cfg.alt_risk_on_btc_weight,
            )
            # Merge — keys are disjoint (BTC excluded from alt_universe).
            out = dict(alt_weights)
            out.update(btc_part)
            return out

        # Unrecognised state -> cash, never raise.
        return {}
