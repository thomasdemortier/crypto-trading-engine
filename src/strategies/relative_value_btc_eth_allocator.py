"""
BTC/ETH long-only relative-value allocator (research strategy 7).

Reads the precomputed signal DataFrame from
`src.relative_value_signals.compute_signals` and turns the per-bar
regime + per-asset trend into target weights at every rebalance bar.

Hard rules:
    * Long-only. No leverage. No shorting. Σ weights ≤ 1.
    * Cash is allowed.
    * Weekly rebalance (the portfolio backtester decides the bar).
    * Lookahead-free: signals sliced on `timestamp <= asof_ts_ms`.
    * Allocation thresholds are LOCKED in `RelativeValueAllocatorConfig`
      defaults — never tuned after seeing results.

Allocation rules (verbatim from spec, NOT tuned):
    eth_leadership      -> 100 % ETH if ETH > 200d MA, else 50 % ETH +
                              50 % cash
    btc_leadership      -> 100 % BTC if BTC > 200d MA, else 50 % BTC +
                              50 % cash
    defensive           -> 100 % cash
    unstable_rotation   -> 50 % BTC + 50 % cash if BTC > 200d MA, else
                              cash
    neutral             -> 50 % BTC + 50 % ETH if both > 200d MA, else
                              100 % BTC if BTC > 200d MA, else cash
    unknown             -> 100 % cash
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..relative_value_signals import (
    REGIME_STATES, STATE_BTC_LEADERSHIP, STATE_DEFENSIVE,
    STATE_ETH_LEADERSHIP, STATE_NEUTRAL, STATE_UNKNOWN,
    STATE_UNSTABLE_ROTATION,
)


# ---------------------------------------------------------------------------
# Locked configuration. Edits = retuning.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RelativeValueAllocatorConfig:
    btc_asset: str = "BTC/USDT"
    eth_asset: str = "ETH/USDT"
    rebalance_frequency: str = "weekly"  # informational only

    full_weight: float = 1.00
    half_weight: float = 0.50
    neutral_split: float = 0.50  # 50/50 BTC/ETH in the neutral state


def _empty(asset_frames: Dict[str, pd.DataFrame], asset: str,
            asof_ts_ms: int) -> bool:
    df = asset_frames.get(asset)
    if df is None or df.empty:
        return True
    sub = df[df["timestamp"] <= int(asof_ts_ms)]
    return len(sub) < 2


class RelativeValueBTCETHAllocatorStrategy:
    """Portfolio strategy. Implements the standard
    `target_weights(asof_ts_ms, asset_frames, timeframe)` contract."""

    name = "relative_value_btc_eth_allocator"

    def __init__(
        self,
        signals_df: pd.DataFrame,
        cfg: Optional[RelativeValueAllocatorConfig] = None,
    ) -> None:
        self.cfg = cfg or RelativeValueAllocatorConfig()
        if signals_df is None or signals_df.empty:
            self._signals = pd.DataFrame()
        else:
            df = signals_df.copy()
            df["timestamp"] = pd.to_numeric(df["timestamp"]).astype("int64")
            df = df.sort_values("timestamp").reset_index(drop=True)
            self._signals = df

    # ---- helpers -----------------------------------------------------------
    def _signal_asof(self, asof_ts_ms: int) -> Optional[pd.Series]:
        if self._signals.empty:
            return None
        sub = self._signals[self._signals["timestamp"]
                              <= int(asof_ts_ms)]
        if sub.empty:
            return None
        return sub.iloc[-1]

    # ---- public API --------------------------------------------------------
    def diagnostics(self, asof_ts_ms: int) -> Dict[str, object]:
        row = self._signal_asof(asof_ts_ms)
        if row is None:
            return {
                "asof_ts_ms": int(asof_ts_ms),
                "regime_state": STATE_UNKNOWN,
            }
        return {
            "asof_ts_ms": int(asof_ts_ms),
            "regime_state": str(row.get("regime_state", STATE_UNKNOWN)),
            "eth_btc_ratio": float(row.get("eth_btc_ratio", float("nan"))),
            "ratio_30d_return": float(row.get("ratio_30d_return",
                                                 float("nan"))),
            "ratio_90d_return": float(row.get("ratio_90d_return",
                                                 float("nan"))),
            "ratio_above_ma_200": float(row.get("ratio_above_ma_200",
                                                   float("nan"))),
            "ratio_z90": float(row.get("ratio_z90", float("nan"))),
            "btc_above_200dma": float(row.get("btc_above_200dma",
                                                 float("nan"))),
            "eth_above_200dma": float(row.get("eth_above_200dma",
                                                 float("nan"))),
        }

    def target_weights(
        self,
        asof_ts_ms: int,
        asset_frames: Dict[str, pd.DataFrame],
        timeframe: str = "1d",
    ) -> Dict[str, float]:
        row = self._signal_asof(asof_ts_ms)
        if row is None:
            return {}
        state = str(row.get("regime_state", STATE_UNKNOWN))

        btc_missing = _empty(asset_frames, self.cfg.btc_asset, asof_ts_ms)
        eth_missing = _empty(asset_frames, self.cfg.eth_asset, asof_ts_ms)
        btc_above = bool(row.get("btc_above_200dma", 0.0) > 0.0)
        eth_above = bool(row.get("eth_above_200dma", 0.0) > 0.0)

        weights: Dict[str, float] = {}
        if state == STATE_ETH_LEADERSHIP:
            if not eth_missing:
                weights[self.cfg.eth_asset] = (self.cfg.full_weight
                                                  if eth_above
                                                  else self.cfg.half_weight)
        elif state == STATE_BTC_LEADERSHIP:
            if not btc_missing:
                weights[self.cfg.btc_asset] = (self.cfg.full_weight
                                                  if btc_above
                                                  else self.cfg.half_weight)
        elif state == STATE_DEFENSIVE:
            return {}
        elif state == STATE_UNSTABLE_ROTATION:
            if not btc_missing and btc_above:
                weights[self.cfg.btc_asset] = self.cfg.half_weight
            else:
                return {}
        elif state == STATE_NEUTRAL:
            if not btc_missing and not eth_missing and btc_above and eth_above:
                weights[self.cfg.btc_asset] = self.cfg.neutral_split
                weights[self.cfg.eth_asset] = self.cfg.neutral_split
            elif not btc_missing and btc_above:
                weights[self.cfg.btc_asset] = self.cfg.full_weight
            else:
                return {}
        elif state == STATE_UNKNOWN:
            return {}

        # Defence in depth: ensure Σ ≤ 1.
        total = sum(weights.values())
        if total > 1.0 and total > 0.0:
            scale = 1.0 / total
            weights = {k: float(v) * scale for k, v in weights.items()}
        return weights
