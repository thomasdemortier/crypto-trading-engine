"""
Regime-aware momentum rotation (long-only portfolio strategy).

Routes the rebalance decision based on the cross-asset regime signal
(`src/crypto_regime_signals.py`):

  * `alt_risk_on`     -> hold top 3 by composite momentum + relative-strength
  * `btc_leadership`  -> hold BTC only
  * `defensive`       -> hold cash
  * `mixed` / unknown -> hold BTC only (conservative default)

The composite ranking score (default weights from spec):

    score = 0.4 * ret30 + 0.4 * ret90 + 0.2 * (asset_30d - btc_30d)

`target_weights()` reads the precomputed `signals_df` produced by
`crypto_regime_signals.compute_regime_signals()` — passed in once at
construction. This keeps the strategy lookahead-free at backtest time:
the regime label at row t was itself computed from data at or before
row t.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .. import crypto_regime_signals as crs


@dataclass(frozen=True)
class RegimeAwareMomentumConfig:
    top_n_alt_risk_on: int = 3
    cash_filter_asset: str = "BTC/USDT"
    momentum_short_window: int = 30
    momentum_long_window: int = 90
    short_weight: float = 0.4
    long_weight: float = 0.4
    relative_strength_weight: float = 0.2
    use_volatility_penalty: bool = False
    min_assets_required: int = 5
    rebalance_frequency: str = "weekly"


def _bars_per_day(timeframe: str) -> int:
    return {"1h": 24, "4h": 6, "1d": 1}.get(timeframe, 1)


class RegimeAwareMomentumRotationStrategy:
    """Portfolio strategy. Reads `signals_df` (one row per timestamp)
    keyed on `timestamp` to look up the regime at each rebalance bar."""

    name = "regime_aware_momentum_rotation"

    def __init__(
        self,
        signals_df: pd.DataFrame,
        cfg: Optional[RegimeAwareMomentumConfig] = None,
    ) -> None:
        if signals_df is None or signals_df.empty:
            raise ValueError(
                "RegimeAwareMomentumRotationStrategy requires a "
                "non-empty signals_df from crypto_regime_signals."
            )
        if "timestamp" not in signals_df.columns or "risk_state" not in signals_df.columns:
            raise ValueError(
                "signals_df must include 'timestamp' and 'risk_state' columns"
            )
        self._signals = signals_df.set_index("timestamp")
        self.cfg = cfg or RegimeAwareMomentumConfig()

    # ---- Helpers -----------------------------------------------------------
    def _regime_at(self, asof_ts_ms: int) -> str:
        """Return the regime label at or before `asof_ts_ms`. Never peeks
        ahead — we use `<=` slicing then take the last row."""
        sub = self._signals.loc[self._signals.index <= int(asof_ts_ms)]
        if sub.empty:
            return crs.UNKNOWN
        return str(sub["risk_state"].iloc[-1])

    def _ret_at(self, df: pd.DataFrame, asof_ts_ms: int,
                lag_bars: int) -> Optional[float]:
        sub = df[df["timestamp"] <= int(asof_ts_ms)]
        if len(sub) < lag_bars + 1:
            return None
        last = float(sub["close"].iloc[-1])
        ago = float(sub["close"].iloc[-1 - lag_bars])
        if ago <= 0 or last <= 0:
            return None
        return (last / ago) - 1.0

    def _score_one(self, df: pd.DataFrame, asof_ts_ms: int,
                   bars_per_day: int, btc_ret_short: Optional[float]) -> Optional[float]:
        ret_short = self._ret_at(
            df, asof_ts_ms, self.cfg.momentum_short_window * bars_per_day,
        )
        ret_long = self._ret_at(
            df, asof_ts_ms, self.cfg.momentum_long_window * bars_per_day,
        )
        if ret_short is None or ret_long is None:
            return None
        rs = (ret_short - btc_ret_short) if btc_ret_short is not None else 0.0
        score = (self.cfg.short_weight * ret_short
                 + self.cfg.long_weight * ret_long
                 + self.cfg.relative_strength_weight * rs)
        return float(score)

    def _btc_only(self, asset_frames: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        if self.cfg.cash_filter_asset in asset_frames:
            return {self.cfg.cash_filter_asset: 1.0}
        return {}  # BTC missing — defensive cash

    # ---- Public API --------------------------------------------------------
    def target_weights(
        self,
        asof_ts_ms: int,
        asset_frames: Dict[str, pd.DataFrame],
        timeframe: str = "1d",
    ) -> Dict[str, float]:
        regime = self._regime_at(asof_ts_ms)

        if regime in (crs.DEFENSIVE,):
            return {}  # cash
        if regime in (crs.UNKNOWN, crs.MIXED, crs.BTC_LEADERSHIP):
            return self._btc_only(asset_frames)

        # alt_risk_on: top-N by composite score.
        bpd = _bars_per_day(timeframe)
        btc_df = asset_frames.get("BTC/USDT")
        btc_ret_short = (
            self._ret_at(btc_df, asof_ts_ms,
                         self.cfg.momentum_short_window * bpd)
            if btc_df is not None else None
        )
        scores: Dict[str, float] = {}
        for asset, df in asset_frames.items():
            s = self._score_one(df, asof_ts_ms, bpd, btc_ret_short)
            if s is not None:
                scores[asset] = s
        if len(scores) < self.cfg.min_assets_required:
            return self._btc_only(asset_frames)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[: self.cfg.top_n_alt_risk_on]
        if not top:
            return self._btc_only(asset_frames)
        w = 1.0 / len(top)
        return {asset: w for asset, _ in top}


# ---------------------------------------------------------------------------
# Regime-aware random placebo
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegimeAwareRandomConfig:
    top_n_alt_risk_on: int = 3
    seed: int = 42
    cash_filter_asset: str = "BTC/USDT"
    min_assets_required: int = 5


class RegimeAwareRandomPlacebo:
    """Same regime gating as the real strategy, but RANDOM selection
    within the allowed asset set. Used to isolate the value of the
    momentum + relative-strength ranking from the value of the regime
    routing."""

    name = "regime_aware_random_placebo"

    def __init__(
        self,
        signals_df: pd.DataFrame,
        cfg: Optional[RegimeAwareRandomConfig] = None,
    ) -> None:
        if signals_df is None or signals_df.empty:
            raise ValueError("signals_df is required")
        self._signals = signals_df.set_index("timestamp")
        self.cfg = cfg or RegimeAwareRandomConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

    def reset(self, seed: Optional[int] = None) -> None:
        s = self.cfg.seed if seed is None else int(seed)
        self._rng = np.random.default_rng(s)

    def _regime_at(self, asof_ts_ms: int) -> str:
        sub = self._signals.loc[self._signals.index <= int(asof_ts_ms)]
        if sub.empty:
            return crs.UNKNOWN
        return str(sub["risk_state"].iloc[-1])

    def _btc_only(self, asset_frames: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        if self.cfg.cash_filter_asset in asset_frames:
            return {self.cfg.cash_filter_asset: 1.0}
        return {}

    def target_weights(
        self,
        asof_ts_ms: int,
        asset_frames: Dict[str, pd.DataFrame],
        timeframe: str = "1d",
    ) -> Dict[str, float]:
        regime = self._regime_at(asof_ts_ms)
        if regime == crs.DEFENSIVE:
            return {}
        if regime in (crs.UNKNOWN, crs.MIXED, crs.BTC_LEADERSHIP):
            return self._btc_only(asset_frames)
        # alt_risk_on: random N from eligible assets.
        eligible = []
        for asset, df in asset_frames.items():
            sub = df[df["timestamp"] <= int(asof_ts_ms)]
            if len(sub) >= 2:
                eligible.append(asset)
        if len(eligible) < self.cfg.min_assets_required:
            return self._btc_only(asset_frames)
        n = min(self.cfg.top_n_alt_risk_on, len(eligible))
        chosen = list(self._rng.choice(eligible, size=n, replace=False))
        return {asset: 1.0 / n for asset in chosen}
