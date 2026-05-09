"""
Technical indicators.

Every indicator is computed from past data only. We use shift-aware operations
and the standard Wilder smoothing (`ewm(alpha=1/period, adjust=False)`) for
RSI/ATR. No `center=True`, no `forward fill`, no future leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. Returns NaN for the first `period` rows."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss is 0 and avg_gain > 0, rsi is 100 by convention.
    out = out.where(avg_loss != 0, 100.0)
    # Before we have `period` of data, return NaN.
    out.iloc[:period] = np.nan
    return out


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def bollinger_bands(close: pd.Series, period: int = 20,
                    num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Standard Bollinger Bands. All values use only past data (rolling
    window includes the current bar but never future bars)."""
    mid = sma(close, period)
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's ATR."""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_indicators(df: pd.DataFrame, cfg: config.StrategyConfig | None = None) -> pd.DataFrame:
    """Return a copy of `df` with indicator columns added.

    Required input columns: open, high, low, close, volume.
    Output columns added: rsi, ma50, ma200, atr, atr_pct, vol_ma.
    """
    cfg = cfg or config.STRATEGY
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing OHLCV columns: {sorted(missing)}")

    out = df.copy()
    out["rsi"] = rsi(out["close"], cfg.rsi_period)
    out["ma50"] = sma(out["close"], cfg.ma_short)
    out["ma200"] = sma(out["close"], cfg.ma_long)
    out["atr"] = atr(out["high"], out["low"], out["close"], cfg.atr_period)
    # ATR as a percentage of price — comparable across assets.
    out["atr_pct"] = (out["atr"] / out["close"]) * 100.0
    out["vol_ma"] = sma(out["volume"], cfg.volume_ma_period)
    return out
