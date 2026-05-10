"""
Drawdown-targeted BTC allocator (research strategy 4).

A long-only continuous-exposure BTC allocator. The allocator adjusts BTC
exposure as a function of:

  * BTC current drawdown from its rolling all-time high (close).
  * BTC 200d moving-average regime (close vs 200d MA).
  * BTC realised volatility over 30d and 90d (log-returns,
    annualised) — diagnostic only, NOT used to scale weights.
  * Optional alt exposure when BTC > 200d MA AND the alt-basket breadth
    is "strong" (>= `min_strong_alts` of the alt universe show positive
    90d momentum at the rebalance bar).

Exposure rules (verbatim from the research spec, NOT tuned):

    BTC above 200d MA and drawdown <  10 %   -> 100 % BTC
    drawdown 10 % – 20 %                     -> 70 % BTC / 30 % cash
    drawdown 20 % – 35 %                     -> 40 % BTC / 60 % cash
    drawdown > 35 %                           -> 20 % BTC / 80 % cash
    BTC below 200d MA                        -> cap BTC exposure at 40 %

Alt overlay (only in the 100 % BTC bucket, BTC > 200d MA AND breadth
strong): replace 30 percentage points of BTC with an equal-weight basket
of the top-N alts by 90d momentum (default N=3).

Hard rules:
  * Long-only. Σ weights ≤ 1.0. No leverage, no shorts.
  * Lookahead-free: every series is sliced on `timestamp <= asof`.
  * Same `target_weights(asof_ts_ms, asset_frames, timeframe)` contract
    as every portfolio strategy in the project — fees + slippage come
    from the existing portfolio backtester. Weekly rebalance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def _bars_per_day(timeframe: str) -> int:
    return {"1h": 24, "4h": 6, "1d": 1}.get(timeframe, 1)


@dataclass(frozen=True)
class DrawdownTargetedBTCConfig:
    btc_asset: str = "BTC/USDT"
    # Drawdown buckets (positive numbers; drawdowns are |x|).
    dd_bucket_1: float = 0.10        # < 10 %
    dd_bucket_2: float = 0.20        # 10–20 %
    dd_bucket_3: float = 0.35        # 20–35 %
    btc_weight_bucket_1: float = 1.00
    btc_weight_bucket_2: float = 0.70
    btc_weight_bucket_3: float = 0.40
    btc_weight_bucket_4: float = 0.20
    below_ma_cap: float = 0.40

    # Regime / vol windows.
    ma_window_days: int = 200
    realised_vol_short_days: int = 30
    realised_vol_long_days: int = 90
    rebalance_frequency: str = "weekly"  # informational only

    # Alt overlay.
    enable_alt_overlay: bool = True
    alt_universe: Sequence[str] = (
        "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT",
        "XRP/USDT", "DOGE/USDT", "ADA/USDT", "LTC/USDT", "BNB/USDT",
    )
    alt_top_n: int = 3
    alt_momentum_window_days: int = 90
    alt_overlay_btc_weight: float = 0.70
    alt_overlay_alt_weight: float = 0.30
    min_strong_alts: int = 5      # >= this many alts with positive 90d mom


def _slice(df: pd.DataFrame, asof_ts_ms: int) -> pd.DataFrame:
    return df[df["timestamp"] <= int(asof_ts_ms)]


def _btc_drawdown(close: pd.Series) -> float:
    if close.empty:
        return float("nan")
    running_max = close.cummax()
    last = float(close.iloc[-1])
    peak = float(running_max.iloc[-1])
    if peak <= 0:
        return float("nan")
    return float((last / peak) - 1.0)  # negative number, e.g. -0.23


def _btc_above_ma(close: pd.Series, ma_days: int, bars_per_day: int) -> Optional[bool]:
    n = ma_days * bars_per_day
    if len(close) < n + 1:
        return None
    ma = float(close.iloc[-n:].mean())
    last = float(close.iloc[-1])
    return last > ma


def _realised_vol(close: pd.Series, window_days: int,
                   bars_per_day: int) -> Optional[float]:
    n = window_days * bars_per_day
    if len(close) < n + 2:
        return None
    rets = np.log(close).diff().dropna().iloc[-n:]
    if rets.empty:
        return None
    daily_std = float(rets.std(ddof=1))
    return daily_std * float(np.sqrt(365.0 * bars_per_day))


def _btc_weight_for_drawdown(dd_abs: float,
                              cfg: DrawdownTargetedBTCConfig) -> float:
    if dd_abs < cfg.dd_bucket_1:
        return cfg.btc_weight_bucket_1
    if dd_abs < cfg.dd_bucket_2:
        return cfg.btc_weight_bucket_2
    if dd_abs < cfg.dd_bucket_3:
        return cfg.btc_weight_bucket_3
    return cfg.btc_weight_bucket_4


class DrawdownTargetedBTCAllocatorStrategy:
    """Continuous-exposure BTC allocator driven by BTC drawdown + 200d MA."""

    name = "drawdown_targeted_btc_allocator"

    def __init__(
        self,
        cfg: Optional[DrawdownTargetedBTCConfig] = None,
    ) -> None:
        self.cfg = cfg or DrawdownTargetedBTCConfig()

    # ---- internals ---------------------------------------------------------
    def _alt_momentum(
        self, df: pd.DataFrame, asof_ts_ms: int, bars_per_day: int,
    ) -> Optional[float]:
        n = self.cfg.alt_momentum_window_days * bars_per_day
        sub = _slice(df, asof_ts_ms)
        if len(sub) < n + 1:
            return None
        close = sub["close"].astype(float)
        past = float(close.iloc[-(n + 1)])
        last = float(close.iloc[-1])
        if past <= 0:
            return None
        return last / past - 1.0

    def _alt_weights(
        self, asset_frames: Dict[str, pd.DataFrame],
        asof_ts_ms: int, bars_per_day: int, total_weight: float,
    ) -> Dict[str, float]:
        scored = []
        for alt in self.cfg.alt_universe:
            df = asset_frames.get(alt)
            if df is None or df.empty:
                continue
            mom = self._alt_momentum(df, asof_ts_ms, bars_per_day)
            if mom is None:
                continue
            scored.append((alt, mom))
        if not scored:
            return {}
        scored.sort(key=lambda x: x[1], reverse=True)
        chosen = [a for a, m in scored[: self.cfg.alt_top_n] if m > 0]
        if not chosen:
            return {}
        w = total_weight / len(chosen)
        return {a: w for a in chosen}

    def _alt_breadth_strong(
        self, asset_frames: Dict[str, pd.DataFrame],
        asof_ts_ms: int, bars_per_day: int,
    ) -> bool:
        positive = 0
        for alt in self.cfg.alt_universe:
            df = asset_frames.get(alt)
            if df is None or df.empty:
                continue
            mom = self._alt_momentum(df, asof_ts_ms, bars_per_day)
            if mom is not None and mom > 0:
                positive += 1
        return positive >= self.cfg.min_strong_alts

    # ---- public API --------------------------------------------------------
    def diagnostics(
        self, asof_ts_ms: int, asset_frames: Dict[str, pd.DataFrame],
        timeframe: str = "1d",
    ) -> Dict[str, float]:
        """Return the inputs used by `target_weights`. Diagnostic only."""
        bpd = _bars_per_day(timeframe)
        btc = asset_frames.get(self.cfg.btc_asset)
        if btc is None or btc.empty:
            return {}
        sub = _slice(btc, asof_ts_ms)
        if sub.empty:
            return {}
        close = sub["close"].astype(float).reset_index(drop=True)
        dd = _btc_drawdown(close)
        above_ma = _btc_above_ma(close, self.cfg.ma_window_days, bpd)
        rv30 = _realised_vol(close, self.cfg.realised_vol_short_days, bpd)
        rv90 = _realised_vol(close, self.cfg.realised_vol_long_days, bpd)
        return {
            "btc_drawdown": dd if not np.isnan(dd) else float("nan"),
            "btc_above_200dma": float(above_ma) if above_ma is not None
                                else float("nan"),
            "realised_vol_30d": rv30 if rv30 is not None else float("nan"),
            "realised_vol_90d": rv90 if rv90 is not None else float("nan"),
        }

    def target_weights(
        self,
        asof_ts_ms: int,
        asset_frames: Dict[str, pd.DataFrame],
        timeframe: str = "1d",
    ) -> Dict[str, float]:
        bpd = _bars_per_day(timeframe)
        btc = asset_frames.get(self.cfg.btc_asset)
        if btc is None or btc.empty:
            return {}
        sub = _slice(btc, asof_ts_ms)
        if len(sub) < 2:
            return {}
        close = sub["close"].astype(float).reset_index(drop=True)

        dd = _btc_drawdown(close)
        if np.isnan(dd):
            return {}
        above_ma = _btc_above_ma(close, self.cfg.ma_window_days, bpd)
        # Insufficient history for the 200d MA: fall through to defensive
        # cap (40 % BTC) — the conservative choice.
        if above_ma is None:
            above_ma = False

        dd_abs = abs(dd)
        btc_w = _btc_weight_for_drawdown(dd_abs, self.cfg)

        if not above_ma:
            btc_w = min(btc_w, self.cfg.below_ma_cap)
            return {self.cfg.btc_asset: btc_w} if btc_w > 0 else {}

        # BTC above 200d MA + top bucket: optionally fire the alt overlay.
        if (self.cfg.enable_alt_overlay
                and dd_abs < self.cfg.dd_bucket_1
                and self._alt_breadth_strong(asset_frames, asof_ts_ms, bpd)):
            alt_part = self._alt_weights(
                asset_frames, asof_ts_ms, bpd,
                total_weight=self.cfg.alt_overlay_alt_weight,
            )
            if alt_part:
                out = {self.cfg.btc_asset: self.cfg.alt_overlay_btc_weight}
                out.update(alt_part)
                return out

        return {self.cfg.btc_asset: btc_w} if btc_w > 0 else {}
