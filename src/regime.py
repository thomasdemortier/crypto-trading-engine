"""
Market regime detection.

Classifies every candle into a (trend, volatility) regime using ONLY past
and current candle information. There is no lookahead and no smoothing
that peeks ahead — the same partial-vs-full equality property the
indicator tests assert is preserved here.

Regime labels:
  trend:       bull_trend / bear_trend / sideways
  volatility:  high_volatility / low_volatility
  combined:    {trend}|{volatility}     (e.g. "bull_trend|high_volatility")

The detector is purely a *labeller*; it does not place trades or change
risk parameters. Strategies and the regime-filter wrapper consume these
columns to decide whether to act.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import indicators


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegimeConfig:
    ma_short: int = 50
    ma_long: int = 200
    slope_window: int = 20         # bars used to estimate MA200 slope
    flat_ma_spread_pct: float = 1.0  # |MA50-MA200|/MA200*100 below this = sideways
    flat_slope_threshold: float = 0.05  # |slope|/MA200*100 below this = flat trend
    high_vol_atr_pct: float = 5.0    # ATR% above this = high volatility
    atr_period: int = 14


DEFAULT = RegimeConfig()

BULL = "bull_trend"
BEAR = "bear_trend"
SIDEWAYS = "sideways"
HIGH_VOL = "high_volatility"
LOW_VOL = "low_volatility"


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def _ma_slope(series: pd.Series, window: int) -> pd.Series:
    """Discrete slope of `series` over `window` bars: (now - past) / window.
    Uses a backward shift so the value at row i is computed from rows
    [i-window, ..., i] only (no future data)."""
    return (series - series.shift(window)) / float(window)


def add_regime_columns(
    df: pd.DataFrame,
    cfg: Optional[RegimeConfig] = None,
) -> pd.DataFrame:
    """Return a copy of `df` with regime columns appended.

    Required input columns: open, high, low, close, volume.
    Indicator columns (ma50, ma200, atr, atr_pct) are computed if missing.

    Output columns added:
        ma200_slope        — discrete slope of MA200 over `slope_window`
        ma_spread_pct      — |MA50 - MA200| / MA200 * 100
        atr_pct            — ATR / close * 100  (already added if indicators ran)
        trend_regime       — bull_trend / bear_trend / sideways
        volatility_regime  — high_volatility / low_volatility
        regime_label       — "{trend}|{volatility}"
    """
    cfg = cfg or DEFAULT
    out = df.copy()

    if "ma50" not in out.columns:
        out["ma50"] = indicators.sma(out["close"], cfg.ma_short)
    if "ma200" not in out.columns:
        out["ma200"] = indicators.sma(out["close"], cfg.ma_long)
    if "atr" not in out.columns:
        out["atr"] = indicators.atr(out["high"], out["low"], out["close"], cfg.atr_period)
    if "atr_pct" not in out.columns:
        out["atr_pct"] = (out["atr"] / out["close"]) * 100.0

    out["ma200_slope"] = _ma_slope(out["ma200"], cfg.slope_window)
    # Spread normalised to MA200 so it's comparable across assets.
    out["ma_spread_pct"] = (
        (out["ma50"] - out["ma200"]).abs() / out["ma200"].replace(0, np.nan) * 100.0
    )
    # Slope normalised so the threshold is in % per bar.
    slope_pct = (out["ma200_slope"] / out["ma200"].replace(0, np.nan)) * 100.0

    above = out["close"] > out["ma200"]
    below = out["close"] < out["ma200"]
    fast_above_slow = out["ma50"] > out["ma200"]
    fast_below_slow = out["ma50"] < out["ma200"]
    pos_slope = slope_pct > cfg.flat_slope_threshold
    neg_slope = slope_pct < -cfg.flat_slope_threshold
    flat_slope = slope_pct.abs() <= cfg.flat_slope_threshold
    flat_spread = out["ma_spread_pct"] <= cfg.flat_ma_spread_pct

    bull_mask = above & fast_above_slow & pos_slope
    bear_mask = below & fast_below_slow & neg_slope
    sideways_mask = flat_slope & flat_spread

    trend = pd.Series(SIDEWAYS, index=out.index, dtype="object")
    trend = trend.mask(bull_mask, BULL)
    trend = trend.mask(bear_mask, BEAR)
    # Anything that doesn't match a clean bull/bear and isn't sideways
    # falls back to sideways (the most conservative label).
    trend = trend.mask(
        ~(bull_mask | bear_mask | sideways_mask), SIDEWAYS,
    )
    # During warmup (any of the inputs NaN) leave the label as 'unknown'
    # so consumers can detect insufficient history.
    warmup = (
        out["ma50"].isna() | out["ma200"].isna()
        | out["ma200_slope"].isna() | out["atr_pct"].isna()
    )
    trend = trend.mask(warmup, "unknown")

    high_vol_mask = out["atr_pct"] > cfg.high_vol_atr_pct
    vol = pd.Series(LOW_VOL, index=out.index, dtype="object")
    vol = vol.mask(high_vol_mask, HIGH_VOL)
    vol = vol.mask(out["atr_pct"].isna(), "unknown")

    out["trend_regime"] = trend
    out["volatility_regime"] = vol
    out["regime_label"] = trend.astype(str) + "|" + vol.astype(str)
    return out


# ---------------------------------------------------------------------------
# Aggregation helpers (used by the dashboard + scorecard)
# ---------------------------------------------------------------------------
def regime_distribution(df: pd.DataFrame) -> pd.Series:
    """Return percent-of-bars in each `regime_label`. Excludes warmup rows
    (any label containing 'unknown')."""
    if "regime_label" not in df.columns:
        return pd.Series(dtype=float)
    valid = df[~df["regime_label"].str.contains("unknown", na=False)]
    if valid.empty:
        return pd.Series(dtype=float)
    return (
        valid["regime_label"].value_counts(normalize=True) * 100.0
    ).rename("pct_of_bars")


def regime_summary_row(asset: str, timeframe: str, df: pd.DataFrame) -> dict:
    """One-row summary suitable for a per-asset/timeframe results CSV."""
    valid = df[~df["regime_label"].str.contains("unknown", na=False)] \
        if "regime_label" in df.columns else df.iloc[0:0]
    n = len(valid)
    if n == 0:
        return {
            "asset": asset, "timeframe": timeframe, "n_bars": 0,
            "pct_bull": 0.0, "pct_bear": 0.0, "pct_sideways": 0.0,
            "pct_high_vol": 0.0, "pct_low_vol": 0.0,
        }
    trend_counts = valid["trend_regime"].value_counts(normalize=True) * 100.0
    vol_counts = valid["volatility_regime"].value_counts(normalize=True) * 100.0
    return {
        "asset": asset, "timeframe": timeframe, "n_bars": int(n),
        "pct_bull": float(trend_counts.get(BULL, 0.0)),
        "pct_bear": float(trend_counts.get(BEAR, 0.0)),
        "pct_sideways": float(trend_counts.get(SIDEWAYS, 0.0)),
        "pct_high_vol": float(vol_counts.get(HIGH_VOL, 0.0)),
        "pct_low_vol": float(vol_counts.get(LOW_VOL, 0.0)),
    }
