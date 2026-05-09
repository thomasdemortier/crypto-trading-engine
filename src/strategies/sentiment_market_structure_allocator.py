"""
Sentiment-overlaid market-structure vol-target allocator.

Wraps `MarketStructureVolTargetAllocatorStrategy` (the previous best
allocator) and adjusts its target weights using the precomputed
`sentiment_state` from `src.sentiment_signals`.

Overlay rules (per spec — fixed, not tuning knobs):

    extreme_fear (and BTC > 200d MA):
        +20 pp BTC, funded from cash.
    fear_recovery:
        no change — base allocation passes through.
    extreme_greed:
        −20 pp from alts (proportionally), shifted to BTC if base
        already had BTC, otherwise to cash.
    deteriorating:
        −20 pp risky exposure (alts first, then BTC) shifted to cash.
    neutral / unknown / other:
        no change.

The base market-structure signal rules are NOT touched.

Hard rules:
    * Long-only. Σ weights ≤ 1. No leverage, no margin, no shorts.
    * Cash share = 1 − Σ weights, clamped to ≥ 0.
    * Lookahead-free: only sentiment rows with `timestamp <= asof` are
      consulted, mirroring the base allocator's `_signal_asof` slice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from .market_structure_vol_target_allocator import (
    MarketStructureVolTargetAllocatorStrategy,
    MarketStructureVolTargetConfig,
)


def _no_slash(symbol: str) -> str:
    return symbol.replace("/", "")


@dataclass(frozen=True)
class SentimentMarketStructureAllocatorConfig:
    base_cfg: MarketStructureVolTargetConfig = field(
        default_factory=MarketStructureVolTargetConfig)
    # Overlay sizes — fixed by spec.
    extreme_fear_btc_boost: float = 0.20
    extreme_greed_alt_cut: float = 0.20
    deteriorating_risk_cut: float = 0.20


class SentimentMarketStructureAllocatorStrategy:
    """Sentiment overlay on the vol-target market-structure allocator."""

    name = "sentiment_market_structure_allocator"

    def __init__(
        self,
        market_structure_signals_df: pd.DataFrame,
        sentiment_signals_df: pd.DataFrame,
        cfg: Optional[SentimentMarketStructureAllocatorConfig] = None,
    ) -> None:
        self.cfg = cfg or SentimentMarketStructureAllocatorConfig()
        self._base = MarketStructureVolTargetAllocatorStrategy(
            market_structure_signals_df, self.cfg.base_cfg,
        )
        if sentiment_signals_df is None or sentiment_signals_df.empty:
            self._sent = pd.DataFrame()
        else:
            df = sentiment_signals_df.copy()
            df["timestamp"] = pd.to_numeric(df["timestamp"]).astype("int64")
            self._sent = df.sort_values("timestamp").reset_index(drop=True)

    # ---- helpers ----------------------------------------------------------
    def _sentiment_asof(self, asof_ts_ms: int) -> Optional[pd.Series]:
        if self._sent.empty:
            return None
        sub = self._sent[self._sent["timestamp"] <= int(asof_ts_ms)]
        if sub.empty:
            return None
        return sub.iloc[-1]

    def _btc_above_200d_ma(self, asof_ts_ms: int) -> bool:
        """Read from the base allocator's market_structure signal table —
        same column the vol-target rules already use."""
        row = self._base._signal_asof(asof_ts_ms)
        if row is None:
            return False
        v = row.get("btc_above_200d_ma")
        if v is None or pd.isna(v):
            return False
        return bool(v)

    def _scale_weights(self, weights: Dict[str, float],
                        scale: float) -> Dict[str, float]:
        if scale <= 0:
            return {}
        return {k: float(v) * float(scale) for k, v in weights.items()}

    def _split_btc_alts(self, weights: Dict[str, float]
                          ) -> Dict[str, Dict[str, float]]:
        btc_key = self.cfg.base_cfg.btc_asset
        btc = {k: v for k, v in weights.items() if k == btc_key}
        alts = {k: v for k, v in weights.items() if k != btc_key}
        return {"btc": btc, "alts": alts}

    # ---- public API -------------------------------------------------------
    def target_weights(
        self,
        asof_ts_ms: int,
        asset_frames: Dict[str, pd.DataFrame],
        timeframe: str = "1d",
    ) -> Dict[str, float]:
        # Start from the base vol-target allocation.
        base_weights = self._base.target_weights(
            asof_ts_ms, asset_frames, timeframe=timeframe,
        )

        sent = self._sentiment_asof(asof_ts_ms)
        if sent is None:
            return base_weights
        state = sent.get("sentiment_state", "unknown")

        btc_key = self.cfg.base_cfg.btc_asset

        # ---- extreme_fear -------------------------------------------------
        if state == "extreme_fear" and self._btc_above_200d_ma(asof_ts_ms):
            # +20 pp BTC, funded from cash. Cap total at 1.0.
            current_total = sum(base_weights.values())
            cash_share = max(0.0, 1.0 - current_total)
            boost = min(self.cfg.extreme_fear_btc_boost, cash_share)
            if boost <= 0:
                return base_weights
            new = dict(base_weights)
            new[btc_key] = new.get(btc_key, 0.0) + boost
            return new

        # ---- extreme_greed ------------------------------------------------
        if state == "extreme_greed":
            # Cut total alt weight by `extreme_greed_alt_cut` pp,
            # proportionally across alts. Move the cut into BTC if BTC was
            # already in the base allocation, otherwise to cash.
            split = self._split_btc_alts(base_weights)
            alts = split["alts"]
            alt_total = sum(alts.values())
            cut = min(self.cfg.extreme_greed_alt_cut, alt_total)
            if cut <= 0:
                return base_weights
            scale = (alt_total - cut) / alt_total if alt_total > 0 else 0.0
            new_alts = self._scale_weights(alts, scale)
            new = dict(split["btc"])
            new.update(new_alts)
            if btc_key in new:
                new[btc_key] = new[btc_key] + cut
            # else: cut moves to cash by leaving total < 1.
            return new

        # ---- deteriorating ------------------------------------------------
        if state == "deteriorating":
            # Cut total risky exposure by `deteriorating_risk_cut` pp.
            # Cut alts first (they're the higher-vol leg), then BTC.
            target_cut = self.cfg.deteriorating_risk_cut
            split = self._split_btc_alts(base_weights)
            alts = split["alts"]
            btc = split["btc"]
            alt_total = sum(alts.values())
            btc_total = sum(btc.values())
            current_total = alt_total + btc_total
            if current_total <= 0:
                return base_weights
            actual_cut = min(target_cut, current_total)
            cut_from_alts = min(actual_cut, alt_total)
            cut_from_btc = actual_cut - cut_from_alts
            scale_alts = ((alt_total - cut_from_alts) / alt_total
                            if alt_total > 0 else 0.0)
            scale_btc = ((btc_total - cut_from_btc) / btc_total
                           if btc_total > 0 else 0.0)
            new = self._scale_weights(alts, scale_alts)
            new.update(self._scale_weights(btc, scale_btc))
            # Drop tiny residual zeros to keep the dict clean.
            new = {k: v for k, v in new.items() if v > 1e-9}
            return new

        # ---- everything else (neutral, unknown, fear_recovery, greed,
        #      fear) — pass base through.
        return base_weights
