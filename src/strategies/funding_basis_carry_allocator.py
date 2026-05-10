"""
Funding + basis carry / crowding allocator (long-only portfolio strategy).

Reads the precomputed signal DataFrame from
`src.funding_basis_signals.compute_signals` and turns the per-asset
regime + carry score into target weights at every rebalance bar.

Hard rules:
    * Long-only. No leverage. No shorting. Σ weights ≤ 1.
    * Cash is allowed.
    * Weekly rebalance (the portfolio backtester decides the bar).
    * Lookahead-free: signals are sliced on `timestamp <= asof_ts_ms`.
    * Allocation thresholds are LOCKED in `FundingBasisCarryConfig`
      defaults — never tuned after seeing results.

Allocation rules (verbatim from the spec, NOT tuned):
    * Per-asset state -> per-asset cap:
        cheap_bullish              -> 0.80 (single-asset cap)
        neutral_risk_on            -> 0.70
        crowded_long               -> 0.30
        stress_negative_funding    -> 0.00
        defensive                  -> 0.00
        unknown                    -> 0.00
    * Order tradable assets (cap > 0 AND state in {neutral, cheap_bullish})
      by carry_attractiveness desc.
    * If 2+ tradable: top -> 0.70, second -> 0.30, each capped by their
      per-asset cap.
    * If 1 tradable: 0.80 (capped). Rest cash.
    * If 0 tradable but BTC/ETH show crowded_long: that asset at 0.30.
    * Else: 100 % cash.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..funding_basis_signals import (
    REGIME_STATES, STATE_CHEAP_BULLISH, STATE_CROWDED_LONG, STATE_DEFENSIVE,
    STATE_NEUTRAL_RISK_ON, STATE_STRESS_NEGATIVE_FUNDING, STATE_UNKNOWN,
)


# ---------------------------------------------------------------------------
# Locked configuration. Edits = retuning. Don't.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FundingBasisCarryConfig:
    rebalance_frequency: str = "weekly"   # informational only

    # Per-asset weight caps by regime state.
    cap_cheap_bullish: float = 0.80
    cap_neutral_risk_on: float = 0.70
    cap_crowded_long: float = 0.30
    cap_stress_negative_funding: float = 0.0
    cap_defensive: float = 0.0
    cap_unknown: float = 0.0

    # Two-asset split when 2+ tradable.
    two_asset_top_weight: float = 0.70
    two_asset_second_weight: float = 0.30
    one_asset_weight: float = 0.80
    crowded_fallback_weight: float = 0.30


_TRADABLE_STATES: Tuple[str, ...] = (STATE_CHEAP_BULLISH,
                                       STATE_NEUTRAL_RISK_ON)
_FALLBACK_STATES: Tuple[str, ...] = (STATE_CROWDED_LONG,)


def _state_cap(state: str, cfg: FundingBasisCarryConfig) -> float:
    return {
        STATE_CHEAP_BULLISH: cfg.cap_cheap_bullish,
        STATE_NEUTRAL_RISK_ON: cfg.cap_neutral_risk_on,
        STATE_CROWDED_LONG: cfg.cap_crowded_long,
        STATE_STRESS_NEGATIVE_FUNDING: cfg.cap_stress_negative_funding,
        STATE_DEFENSIVE: cfg.cap_defensive,
        STATE_UNKNOWN: cfg.cap_unknown,
    }.get(state, cfg.cap_unknown)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
class FundingBasisCarryAllocatorStrategy:
    """Portfolio strategy. Implements the standard
    `target_weights(asof_ts_ms, asset_frames, timeframe)` contract."""

    name = "funding_basis_carry_allocator"

    def __init__(
        self,
        signals_df: pd.DataFrame,
        cfg: Optional[FundingBasisCarryConfig] = None,
    ) -> None:
        self.cfg = cfg or FundingBasisCarryConfig()
        if signals_df is None or signals_df.empty:
            self._signals = pd.DataFrame()
        else:
            df = signals_df.copy()
            df["timestamp"] = pd.to_numeric(df["timestamp"]).astype("int64")
            df = df.sort_values(["asset", "timestamp"]).reset_index(drop=True)
            self._signals = df

    # ---- helpers -----------------------------------------------------------
    def _asof_per_asset(
        self, asof_ts_ms: int,
    ) -> Dict[str, pd.Series]:
        """Most-recent signal row per asset with `timestamp <= asof`."""
        if self._signals.empty:
            return {}
        sub = self._signals[self._signals["timestamp"]
                              <= int(asof_ts_ms)]
        if sub.empty:
            return {}
        latest = sub.groupby("asset", as_index=False).tail(1)
        return {row["asset"]: row for _, row in latest.iterrows()}

    # ---- public API --------------------------------------------------------
    def diagnostics(self, asof_ts_ms: int) -> List[Dict[str, object]]:
        """Per-asset state + carry score at this asof. Used by the
        dashboard + report."""
        out: List[Dict[str, object]] = []
        for asset, row in self._asof_per_asset(asof_ts_ms).items():
            out.append({
                "asof_ts_ms": int(asof_ts_ms),
                "asset": str(asset),
                "regime_state": str(row.get("regime_state",
                                                  STATE_UNKNOWN)),
                "funding_z90": float(row.get("funding_z90", float("nan"))),
                "basis_z90": float(row.get("basis_z90", float("nan"))),
                "above_200dma": float(row.get("above_200dma",
                                                 float("nan"))),
                "carry_attractiveness": float(
                    row.get("carry_attractiveness", float("nan"))),
                "crowding_score": float(row.get("crowding_score",
                                                   float("nan"))),
            })
        return out

    def target_weights(
        self,
        asof_ts_ms: int,
        asset_frames: Dict[str, pd.DataFrame],
        timeframe: str = "1d",
    ) -> Dict[str, float]:
        latest = self._asof_per_asset(asof_ts_ms)
        if not latest:
            return {}

        # Build (asset, state, carry_score, cap) tuples.
        candidates: List[Tuple[str, str, float, float]] = []
        for asset, row in latest.items():
            if asset not in asset_frames:
                continue
            sub = asset_frames[asset]
            if sub is None or sub.empty:
                continue
            sub = sub[sub["timestamp"] <= int(asof_ts_ms)]
            if len(sub) < 2:
                continue
            state = str(row.get("regime_state", STATE_UNKNOWN))
            cap = _state_cap(state, self.cfg)
            score = float(row.get("carry_attractiveness", float("nan")))
            if np.isnan(score):
                score = -np.inf  # NaN scores can never win the ranking
            candidates.append((asset, state, score, cap))

        if not candidates:
            return {}

        tradable = [(a, s, sc, cap) for (a, s, sc, cap) in candidates
                     if s in _TRADABLE_STATES and cap > 0.0]
        tradable.sort(key=lambda x: x[2], reverse=True)

        weights: Dict[str, float] = {}
        if len(tradable) >= 2:
            top_a, _, _, top_cap = tradable[0]
            sec_a, _, _, sec_cap = tradable[1]
            weights[top_a] = min(self.cfg.two_asset_top_weight, top_cap)
            weights[sec_a] = min(self.cfg.two_asset_second_weight, sec_cap)
        elif len(tradable) == 1:
            only_a, _, _, only_cap = tradable[0]
            weights[only_a] = min(self.cfg.one_asset_weight, only_cap)
        else:
            # No tradable assets — fall back to the crowded-long cap if any.
            crowded = [(a, s, sc, cap) for (a, s, sc, cap) in candidates
                        if s in _FALLBACK_STATES and cap > 0.0]
            crowded.sort(key=lambda x: x[2], reverse=True)
            if crowded:
                only_a, _, _, only_cap = crowded[0]
                weights[only_a] = min(self.cfg.crowded_fallback_weight,
                                        only_cap)

        # Final safety: ensure Σw ≤ 1 (defence in depth — caps already
        # honour this, but a future config edit could break it).
        total = sum(weights.values())
        if total > 1.0 and total > 0.0:
            scale = 1.0 / total
            weights = {k: float(v) * scale for k, v in weights.items()}
        return weights
