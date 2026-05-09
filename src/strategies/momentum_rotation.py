"""
Multi-asset momentum rotation (long-only portfolio strategy).

This is intentionally NOT a `Strategy` subclass — it does not emit per-bar
per-asset BUY/SELL signals. Instead, on each rebalance date, it ranks all
available assets by a momentum score and returns target weights for the
top-N. The portfolio backtester (`src/portfolio_backtester.py`) consumes
those target weights and rebalances accordingly.

Defaults from spec:
  * top_n = 3
  * rebalance_frequency = weekly
  * momentum_short_window = 30 days
  * momentum_long_window = 90 days
  * use_volatility_adjustment = False
  * cash_filter_asset = "BTC/USDT"
  * cash_filter_ma = 200 days
  * min_assets_required = 5

Lookahead invariant:
  At rebalance bar t, the score for asset A is computed from
  `close[t] / close[t - short_window]` and the same for the long window.
  Only data with timestamp <= t is touched. The 200-day MA used for the
  cash filter is `close.rolling(200).mean().iloc[t]`, which by definition
  uses only past data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MomentumRotationConfig:
    top_n: int = 3
    rebalance_frequency: str = "weekly"   # "weekly" or "monthly"
    momentum_short_window: int = 30       # days
    momentum_long_window: int = 90        # days
    use_volatility_adjustment: bool = False
    cash_filter_asset: str = "BTC/USDT"
    cash_filter_ma: int = 200             # days
    min_assets_required: int = 5
    short_weight: float = 0.5             # weight on short-window return
    long_weight: float = 0.5              # weight on long-window return


def _bars_per_day(timeframe: str) -> int:
    return {"1h": 24, "4h": 6, "1d": 1}.get(timeframe, 1)


class MomentumRotationStrategy:
    """Portfolio strategy. NOT a single-asset signal strategy.

    Public surface:
      * `name`
      * `target_weights(asof_ts_ms, asset_frames, timeframe) -> dict[asset, weight]`
        — returns the desired allocation at this rebalance bar. Empty dict
        means "all cash".
    """

    name = "momentum_rotation"

    def __init__(self, cfg: Optional[MomentumRotationConfig] = None) -> None:
        self.cfg = cfg or MomentumRotationConfig()

    # ---- Helpers -----------------------------------------------------------
    def _score_one(self, df: pd.DataFrame, asof_ts_ms: int,
                   bars_per_day: int) -> Optional[float]:
        """Momentum score for one asset at `asof_ts_ms`, computed from
        ONLY past+current bars (no lookahead). Returns None if the asset
        has insufficient history."""
        sub = df[df["timestamp"] <= int(asof_ts_ms)]
        n_short = self.cfg.momentum_short_window * bars_per_day
        n_long = self.cfg.momentum_long_window * bars_per_day
        if len(sub) < n_long + 1:
            return None
        close = sub["close"].astype(float)
        last = float(close.iloc[-1])
        short_ago = float(close.iloc[-1 - n_short])
        long_ago = float(close.iloc[-1 - n_long])
        if short_ago <= 0 or long_ago <= 0 or last <= 0:
            return None
        ret_short = (last / short_ago) - 1.0
        ret_long = (last / long_ago) - 1.0
        score = (self.cfg.short_weight * ret_short
                 + self.cfg.long_weight * ret_long)
        if self.cfg.use_volatility_adjustment:
            rets = close.pct_change().dropna().iloc[-n_short:]
            vol = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
            if vol > 0:
                score = score / (vol * np.sqrt(bars_per_day * 365))
        return float(score)

    def _cash_filter_bearish(self, asset_frames: Dict[str, pd.DataFrame],
                             asof_ts_ms: int, bars_per_day: int) -> bool:
        """True if the cash-filter asset's close is BELOW its long MA at
        `asof_ts_ms`. Returns False if the cash filter asset is missing
        from the universe (we never block trading just because the filter
        asset is unavailable — record a warning instead)."""
        df = asset_frames.get(self.cfg.cash_filter_asset)
        if df is None or df.empty:
            return False
        sub = df[df["timestamp"] <= int(asof_ts_ms)]
        ma_bars = self.cfg.cash_filter_ma * bars_per_day
        if len(sub) < ma_bars + 1:
            return False  # warmup — do not block
        last_close = float(sub["close"].iloc[-1])
        ma = float(sub["close"].iloc[-ma_bars:].mean())
        return last_close < ma

    # ---- Public API --------------------------------------------------------
    def target_weights(
        self,
        asof_ts_ms: int,
        asset_frames: Dict[str, pd.DataFrame],
        timeframe: str = "1d",
    ) -> Dict[str, float]:
        """Return target portfolio weights at this rebalance bar."""
        bars_per_day = _bars_per_day(timeframe)
        # Cash filter first — never enter when broad-market is bearish.
        if self._cash_filter_bearish(asset_frames, asof_ts_ms, bars_per_day):
            return {}
        # Score every available asset.
        scores: Dict[str, float] = {}
        for asset, df in asset_frames.items():
            s = self._score_one(df, asof_ts_ms, bars_per_day)
            if s is not None:
                scores[asset] = s
        if len(scores) < self.cfg.min_assets_required:
            # Not enough universe — stay in cash.
            return {}
        # Top N.
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[: self.cfg.top_n]
        # Equal weight across selected assets.
        n = len(top)
        if n == 0:
            return {}
        w = 1.0 / n
        return {asset: w for asset, _ in top}


# ---------------------------------------------------------------------------
# Placebo: random Top-N rotation, fixed seed
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RandomRotationConfig:
    top_n: int = 3
    seed: int = 42
    apply_cash_filter: bool = False
    cash_filter_asset: str = "BTC/USDT"
    cash_filter_ma: int = 200
    min_assets_required: int = 5


class RandomRotationPlacebo:
    """Random Top-N selection at each rebalance with a fixed seed.
    Reproducible: same (seed, asset list, rebalance dates) -> identical
    target weights every run."""

    name = "portfolio_random"

    def __init__(self, cfg: Optional[RandomRotationConfig] = None) -> None:
        self.cfg = cfg or RandomRotationConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

    def reset(self, seed: Optional[int] = None) -> None:
        s = self.cfg.seed if seed is None else int(seed)
        self._rng = np.random.default_rng(s)

    def target_weights(
        self,
        asof_ts_ms: int,
        asset_frames: Dict[str, pd.DataFrame],
        timeframe: str = "1d",
    ) -> Dict[str, float]:
        # Optional cash filter (default off — placebos should be the
        # "naive" baseline against which the real strategy must prove value).
        if self.cfg.apply_cash_filter:
            bars_per_day = _bars_per_day(timeframe)
            df_cf = asset_frames.get(self.cfg.cash_filter_asset)
            if df_cf is not None and not df_cf.empty:
                sub = df_cf[df_cf["timestamp"] <= int(asof_ts_ms)]
                ma_bars = self.cfg.cash_filter_ma * bars_per_day
                if len(sub) > ma_bars:
                    last = float(sub["close"].iloc[-1])
                    ma = float(sub["close"].iloc[-ma_bars:].mean())
                    if last < ma:
                        return {}
        # Eligible = assets with at least one bar at or before `asof_ts_ms`.
        eligible = []
        for asset, df in asset_frames.items():
            sub = df[df["timestamp"] <= int(asof_ts_ms)]
            if len(sub) >= 2:
                eligible.append(asset)
        if len(eligible) < self.cfg.min_assets_required:
            return {}
        n = min(self.cfg.top_n, len(eligible))
        chosen = list(self._rng.choice(eligible, size=n, replace=False))
        w = 1.0 / n
        return {asset: w for asset in chosen}
