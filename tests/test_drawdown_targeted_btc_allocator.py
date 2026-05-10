"""Tests for the drawdown-targeted BTC allocator strategy."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config, utils
from src.strategies.drawdown_targeted_btc_allocator import (
    DrawdownTargetedBTCAllocatorStrategy, DrawdownTargetedBTCConfig,
    _btc_drawdown, _btc_above_ma, _btc_weight_for_drawdown, _realised_vol,
)


def _frame_from_close(close: np.ndarray, start: str = "2022-01-01"
                       ) -> pd.DataFrame:
    n = len(close)
    ts = pd.date_range(start, periods=n, freq="1D", tz="UTC")
    return pd.DataFrame({
        "timestamp": (ts.astype("datetime64[ns, UTC]").astype("int64")
                       // 10**6).astype("int64"),
        "datetime": ts,
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.ones(n) * 100.0,
    })


def _write(symbol: str, df: pd.DataFrame, timeframe: str = "1d") -> Path:
    p = utils.csv_path_for(symbol, timeframe)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def tmp_dirs(tmp_path, monkeypatch):
    raw = tmp_path / "raw"; raw.mkdir()
    res = tmp_path / "results"; res.mkdir()
    monkeypatch.setattr(config, "DATA_RAW_DIR", raw)
    monkeypatch.setattr(config, "RESULTS_DIR", res)
    yield {"raw": raw, "results": res}


# ---------------------------------------------------------------------------
# Bucket lookups + helpers
# ---------------------------------------------------------------------------
def test_btc_weight_for_drawdown_buckets():
    cfg = DrawdownTargetedBTCConfig()
    assert _btc_weight_for_drawdown(0.0, cfg) == 1.00
    assert _btc_weight_for_drawdown(0.05, cfg) == 1.00
    assert _btc_weight_for_drawdown(0.15, cfg) == 0.70
    assert _btc_weight_for_drawdown(0.25, cfg) == 0.40
    assert _btc_weight_for_drawdown(0.50, cfg) == 0.20
    # Boundaries: 10 % goes to bucket 2 (>= 10 %), 20 % to bucket 3, 35 % to 4.
    assert _btc_weight_for_drawdown(0.10, cfg) == 0.70
    assert _btc_weight_for_drawdown(0.20, cfg) == 0.40
    assert _btc_weight_for_drawdown(0.35, cfg) == 0.20


def test_btc_drawdown_handles_simple_curve():
    close = pd.Series([100.0, 110.0, 120.0, 90.0])  # peak 120, last 90
    dd = _btc_drawdown(close)
    assert pytest.approx(dd, abs=1e-6) == 90.0 / 120.0 - 1.0


def test_btc_above_ma_short_history_returns_none():
    close = pd.Series([100.0] * 10)
    assert _btc_above_ma(close, ma_days=200, bars_per_day=1) is None


def test_btc_above_ma_long_history_works():
    n = 250
    close = pd.Series(np.linspace(100, 200, n))
    out = _btc_above_ma(close, ma_days=200, bars_per_day=1)
    # Last value (200) is well above MA of last 200 entries.
    assert out is True


def test_realised_vol_short_history_returns_none():
    close = pd.Series(np.linspace(100, 110, 10))
    assert _realised_vol(close, window_days=30, bars_per_day=1) is None


def test_realised_vol_positive_for_noisy_series():
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.02, 200))
    out = _realised_vol(pd.Series(close), window_days=30, bars_per_day=1)
    assert out is not None and out > 0


# ---------------------------------------------------------------------------
# Strategy: target_weights behaviour
# ---------------------------------------------------------------------------
def test_no_btc_data_returns_empty():
    s = DrawdownTargetedBTCAllocatorStrategy()
    out = s.target_weights(0, {}, "1d")
    assert out == {}


def test_long_only_no_leverage_no_shorts():
    """The allocator must never emit a weight > 1 in total or < 0 anywhere."""
    rng = np.random.default_rng(7)
    n = 600
    rets = rng.normal(0.0005, 0.03, n)
    btc = 100 * np.cumprod(1 + rets)
    eth = 100 * np.cumprod(1 + rng.normal(0.0004, 0.03, n))
    frames = {
        "BTC/USDT": _frame_from_close(btc),
        "ETH/USDT": _frame_from_close(eth),
    }
    s = DrawdownTargetedBTCAllocatorStrategy()
    # Sample at the last bar (full history available).
    asof = int(frames["BTC/USDT"]["timestamp"].iloc[-1])
    w = s.target_weights(asof, frames, "1d")
    assert all(v >= 0 for v in w.values())
    assert sum(w.values()) <= 1.0 + 1e-9


def test_top_bucket_above_ma_yields_100pct_btc_when_breadth_weak():
    """Strong trend BTC, alts dead -> 100 % BTC."""
    n = 400
    btc = np.linspace(100, 300, n)  # straight uptrend, no drawdown
    flat = np.ones(n) * 100.0
    frames = {"BTC/USDT": _frame_from_close(btc)}
    for alt in ("ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT", "XRP/USDT"):
        frames[alt] = _frame_from_close(flat)
    s = DrawdownTargetedBTCAllocatorStrategy()
    asof = int(frames["BTC/USDT"]["timestamp"].iloc[-1])
    w = s.target_weights(asof, frames, "1d")
    assert w == {"BTC/USDT": 1.0}


def test_top_bucket_above_ma_with_strong_breadth_fires_alt_overlay():
    """BTC trending up + alts trending up -> 70 % BTC + 30 % alts."""
    n = 400
    btc = np.linspace(100, 300, n)
    frames = {"BTC/USDT": _frame_from_close(btc)}
    # 5 alts also trending up (>= min_strong_alts=5).
    for i, alt in enumerate(["ETH/USDT", "SOL/USDT", "AVAX/USDT",
                              "LINK/USDT", "XRP/USDT"]):
        frames[alt] = _frame_from_close(np.linspace(100, 200 + 10 * i, n))
    s = DrawdownTargetedBTCAllocatorStrategy()
    asof = int(frames["BTC/USDT"]["timestamp"].iloc[-1])
    w = s.target_weights(asof, frames, "1d")
    assert w["BTC/USDT"] == pytest.approx(0.70)
    alt_total = sum(v for k, v in w.items() if k != "BTC/USDT")
    assert alt_total == pytest.approx(0.30)


def test_below_ma_caps_btc_at_40pct():
    """Long downtrend in BTC -> below 200d MA -> cap at 40 %."""
    n = 400
    btc = np.linspace(300, 100, n)  # straight downtrend, BTC well below MA
    flat = np.ones(n) * 100.0
    frames = {"BTC/USDT": _frame_from_close(btc)}
    for alt in ("ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT", "XRP/USDT"):
        frames[alt] = _frame_from_close(flat)
    s = DrawdownTargetedBTCAllocatorStrategy()
    asof = int(frames["BTC/USDT"]["timestamp"].iloc[-1])
    w = s.target_weights(asof, frames, "1d")
    assert "BTC/USDT" in w
    assert w["BTC/USDT"] <= 0.40 + 1e-9
    # No alt overlay below the MA.
    assert all(k == "BTC/USDT" for k in w)


def test_drawdown_bucket_2_yields_70pct_above_ma():
    """Slow base, then late rally, then -15 % pullback. The 200d MA is
    dragged down by the long low base, so the post-pullback price is
    still above the MA -> bucket 2 (70 % BTC)."""
    flat = np.ones(220) * 100.0                      # long low base
    rally = np.linspace(100, 200, 180)               # late rally
    drop = np.linspace(200, 200 * 0.85, 20)          # -15 % pullback
    btc = np.concatenate([flat, rally, drop])
    frames = {"BTC/USDT": _frame_from_close(btc)}
    s = DrawdownTargetedBTCAllocatorStrategy()
    asof = int(frames["BTC/USDT"]["timestamp"].iloc[-1])
    w = s.target_weights(asof, frames, "1d")
    assert w == {"BTC/USDT": pytest.approx(0.70)}


def test_diagnostics_returns_dd_regime_and_vol():
    n = 400
    btc = np.linspace(100, 200, n)
    frames = {"BTC/USDT": _frame_from_close(btc)}
    s = DrawdownTargetedBTCAllocatorStrategy()
    asof = int(frames["BTC/USDT"]["timestamp"].iloc[-1])
    diag = s.diagnostics(asof, frames, "1d")
    assert "btc_drawdown" in diag
    assert "btc_above_200dma" in diag
    assert "realised_vol_30d" in diag
    assert "realised_vol_90d" in diag
    assert diag["btc_above_200dma"] == 1.0  # uptrend


def test_no_lookahead_using_only_past_data():
    """target_weights at bar t must be invariant to data after t."""
    n = 400
    rng = np.random.default_rng(11)
    btc_full = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    frames_full = {"BTC/USDT": _frame_from_close(btc_full)}
    # Truncate to first 300 bars and ensure the same target_weights at t=200.
    truncated = frames_full["BTC/USDT"].iloc[:300].copy()
    frames_short = {"BTC/USDT": truncated}
    s = DrawdownTargetedBTCAllocatorStrategy()
    asof = int(frames_full["BTC/USDT"]["timestamp"].iloc[200])
    w_full = s.target_weights(asof, frames_full, "1d")
    w_short = s.target_weights(asof, frames_short, "1d")
    assert w_full == w_short
