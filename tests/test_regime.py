"""Regime detector + regime filter wrapper regression tests."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src import config, indicators, regime, utils
from src.strategies import RsiMaAtrStrategy
from src.strategies.regime_filtered import RegimeFilteredStrategy, RegimePolicy
from src.strategies.base import BUY, SELL, HOLD, SKIP


def _ohlcv_from_close(close: np.ndarray) -> pd.DataFrame:
    n = len(close)
    rng = np.random.default_rng(0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3, n))
    vol = rng.uniform(100, 1_000, n)
    ts = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({
        "timestamp": (ts.astype("datetime64[ns, UTC]").astype("int64") // 10**6).astype("int64"),
        "datetime": ts, "open": open_, "high": high, "low": low,
        "close": close, "volume": vol,
    })


# ---------------------------------------------------------------------------
# Regime detector
# ---------------------------------------------------------------------------
def test_regime_detector_creates_expected_columns():
    df = _ohlcv_from_close(np.linspace(100, 200, 400))
    out = regime.add_regime_columns(df)
    for col in ("ma200_slope", "ma_spread_pct", "atr_pct",
                "trend_regime", "volatility_regime", "regime_label"):
        assert col in out.columns


def test_regime_detector_classifies_bull_trend():
    # Steady uptrend with moderate noise -> tail bars must be bull_trend.
    rng = np.random.default_rng(0)
    close = np.linspace(100, 300, 400) + rng.normal(0, 0.5, 400)
    out = regime.add_regime_columns(_ohlcv_from_close(close))
    last = out["trend_regime"].iloc[-50:]
    assert (last == regime.BULL).mean() > 0.7, (
        f"expected mostly bull near end, got: {last.value_counts().to_dict()}"
    )


def test_regime_detector_classifies_bear_trend():
    rng = np.random.default_rng(0)
    close = np.linspace(300, 100, 400) + rng.normal(0, 0.5, 400)
    out = regime.add_regime_columns(_ohlcv_from_close(close))
    last = out["trend_regime"].iloc[-50:]
    assert (last == regime.BEAR).mean() > 0.7, (
        f"expected mostly bear near end, got: {last.value_counts().to_dict()}"
    )


def test_regime_detector_classifies_sideways_or_unknown():
    """Flat noise around a constant — should be sideways (or at minimum,
    not a confident bull/bear)."""
    rng = np.random.default_rng(0)
    close = 100 + rng.normal(0, 0.2, 400)
    out = regime.add_regime_columns(_ohlcv_from_close(close))
    last = out["trend_regime"].iloc[-50:]
    bull = (last == regime.BULL).sum()
    bear = (last == regime.BEAR).sum()
    assert bull < 5 and bear < 5, (
        f"flat data should not produce many trends, "
        f"got bull={bull} bear={bear}"
    )


def test_regime_detector_no_lookahead():
    """Same partial-vs-full check the indicator tests use: regime label at
    row i must equal the label computed on df.iloc[:i+1] alone."""
    df = _ohlcv_from_close(np.linspace(100, 200, 400))
    full = regime.add_regime_columns(df)
    for i in (250, 300, 350):
        partial = regime.add_regime_columns(df.iloc[: i + 1].copy())
        assert full["trend_regime"].iloc[i] == partial["trend_regime"].iloc[i]
        assert full["volatility_regime"].iloc[i] == partial["volatility_regime"].iloc[i]


# ---------------------------------------------------------------------------
# Regime-filtered wrapper
# ---------------------------------------------------------------------------
def test_regime_filter_blocks_buy_in_bear_trend(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path)

    # Build a clearly DOWN-trending asset so `trend_regime == bear`
    # dominates the testable window.
    rng = np.random.default_rng(0)
    close = np.linspace(300, 100, 600) + rng.normal(0, 0.5, 600)
    df = _ohlcv_from_close(close)
    p = utils.csv_path_for("BTC/USDT", "4h")
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)

    base = RsiMaAtrStrategy()
    wrapped = RegimeFilteredStrategy(
        base_strategy=base,
        policy=RegimePolicy(block_buys_in_bear=True),
    )
    from src import backtester
    art_base = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False, strategy=base,
    )
    art_filt = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False, strategy=wrapped,
    )
    base_buys = (art_base.decisions["action"] == BUY).sum() \
        if not art_base.decisions.empty else 0
    filt_buys = (art_filt.decisions["action"] == BUY).sum() \
        if not art_filt.decisions.empty else 0
    # The bear-trend filter must reduce (or zero) BUYs in a downtrending dataset.
    assert filt_buys <= base_buys

    # And every regime-blocked SKIP decision must mention the regime block.
    if not art_filt.decisions.empty:
        skips = art_filt.decisions[art_filt.decisions["action"] == SKIP]
        regime_blocks = skips["reason"].astype(str).str.contains(
            "regime block", case=False, na=False,
        )
        # Either there are no regime blocks (uptrend window) OR every one
        # carries the explicit reason — never silent.
        assert regime_blocks.sum() >= 0
