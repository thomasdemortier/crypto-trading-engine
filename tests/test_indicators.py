"""Indicator sanity tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src import indicators


def _synthetic_ohlcv(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, size=n)
    close = 100.0 * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, 0.003, size=n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, size=n)))
    open_ = np.r_[close[0], close[:-1]]
    vol = rng.uniform(100, 500, size=n)
    ts = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({
        "timestamp": (ts.astype("int64") // 10**6).astype("int64"),
        "datetime": ts,
        "open": open_, "high": high, "low": low,
        "close": close, "volume": vol,
    })


def test_rsi_in_valid_range():
    df = _synthetic_ohlcv()
    rsi = indicators.rsi(df["close"], period=14).dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()
    assert len(rsi) > 300


def test_rsi_warmup_is_nan():
    df = _synthetic_ohlcv()
    rsi = indicators.rsi(df["close"], period=14)
    # First 14 values should be NaN
    assert rsi.iloc[:14].isna().all()


def test_sma_correctness():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    sma3 = indicators.sma(s, period=3)
    # First two are NaN, then rolling mean.
    assert sma3.iloc[:2].isna().all()
    assert math.isclose(sma3.iloc[2], 2.0)
    assert math.isclose(sma3.iloc[3], 3.0)
    assert math.isclose(sma3.iloc[5], 5.0)


def test_add_indicators_columns_and_no_lookahead():
    df = _synthetic_ohlcv()
    out = indicators.add_indicators(df)
    for col in ("rsi", "ma50", "ma200", "atr", "atr_pct", "vol_ma"):
        assert col in out.columns

    # Lookahead check: indicator at row i must equal indicator computed on
    # df.iloc[:i+1] only. Sample a few rows.
    for i in (50, 150, 250, 350):
        partial = indicators.add_indicators(df.iloc[: i + 1].copy())
        for col in ("rsi", "ma50", "ma200", "atr"):
            a = out[col].iloc[i]
            b = partial[col].iloc[i]
            if pd.isna(a) and pd.isna(b):
                continue
            assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)


def test_add_indicators_rejects_missing_columns():
    df = pd.DataFrame({"close": [1.0, 2.0]})
    with pytest.raises(ValueError):
        indicators.add_indicators(df)
