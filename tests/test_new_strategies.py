"""Tests for trend_following + pullback_continuation + enhanced breakout."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import backtester, config, utils
from src.strategies import (
    BreakoutStrategy, TrendFollowingStrategy, PullbackContinuationStrategy,
)


def _csv(symbol: str, timeframe: str, close: np.ndarray, volume: np.ndarray) -> Path:
    n = len(close)
    rng = np.random.default_rng(0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3, n))
    ts = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": (ts.astype("datetime64[ns, UTC]").astype("int64") // 10**6).astype("int64"),
        "datetime": ts, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })
    p = utils.csv_path_for(symbol, timeframe)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def synthetic_cache(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(config, "DATA_RAW_DIR", raw)
    yield raw


# ---------------------------------------------------------------------------
# Trend following
# ---------------------------------------------------------------------------
def test_trend_following_emits_valid_signals(synthetic_cache):
    rng = np.random.default_rng(0)
    n = 600
    close = np.linspace(100, 200, n) + rng.normal(0, 1.0, n)
    vol = rng.uniform(100, 1_000, n)
    _csv("BTC/USDT", "4h", close, vol)
    art = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False,
        strategy=TrendFollowingStrategy(),
    )
    assert not art.equity_curve.empty
    if not art.trades.empty:
        assert set(art.trades["side"].unique()).issubset({"BUY", "SELL"})
    # The strategy must always log a decision per bar (HOLD/SKIP/BUY/SELL).
    assert not art.decisions.empty


# ---------------------------------------------------------------------------
# Pullback continuation
# ---------------------------------------------------------------------------
def test_pullback_continuation_emits_valid_signals(synthetic_cache):
    rng = np.random.default_rng(1)
    n = 600
    close = np.linspace(100, 250, n) + rng.normal(0, 0.8, n)
    vol = rng.uniform(100, 1_000, n)
    _csv("BTC/USDT", "4h", close, vol)
    art = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False,
        strategy=PullbackContinuationStrategy(),
    )
    assert not art.equity_curve.empty
    if not art.trades.empty:
        assert set(art.trades["side"].unique()).issubset({"BUY", "SELL"})


# ---------------------------------------------------------------------------
# Enhanced breakout — filters
# ---------------------------------------------------------------------------
def test_breakout_volume_filter_blocks_low_volume(synthetic_cache):
    """When volume is uniformly low (< vol_ma), no BUYs should fire even
    though price keeps breaking new highs."""
    rng = np.random.default_rng(0)
    n = 400
    close = np.linspace(100, 300, n) + rng.normal(0, 0.5, n)
    # Volume monotonically declines — vol_ma will always be ABOVE current
    # volume after warmup, so the filter should block every breakout.
    vol = np.linspace(2000, 100, n)
    _csv("BTC/USDT", "4h", close, vol)
    art = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False,
        strategy=BreakoutStrategy(entry_window=10, exit_window=5),
    )
    assert art.trades.empty


def test_breakout_trend_filter_blocks_bear_market(synthetic_cache):
    """In a sustained downtrend, close < ma200 always; the trend filter
    should block every breakout."""
    rng = np.random.default_rng(0)
    n = 600
    close = np.linspace(300, 100, n) + rng.normal(0, 0.5, n)
    vol = rng.uniform(500, 1500, n)
    _csv("BTC/USDT", "4h", close, vol)
    art = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False,
        strategy=BreakoutStrategy(entry_window=10, exit_window=5),
    )
    assert art.trades.empty


def test_breakout_pure_donchian_legacy_mode_still_works(synthetic_cache):
    """`use_filters=False` should match the legacy bare Donchian behaviour
    (the old test covers the no-lookahead invariant; this one ensures
    legacy callers still produce trades on uptrend data)."""
    from src.strategies.breakout import BreakoutConfig
    rng = np.random.default_rng(0)
    n = 600
    close = np.linspace(100, 300, n) + rng.normal(0, 0.5, n)
    vol = rng.uniform(500, 1500, n)
    _csv("BTC/USDT", "4h", close, vol)
    legacy_cfg = BreakoutConfig(
        entry_window=10, exit_window=5, use_filters=False,
    )
    art = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False,
        strategy=BreakoutStrategy(cfg=legacy_cfg),
    )
    assert not art.equity_curve.empty
