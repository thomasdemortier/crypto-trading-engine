"""Tests for `src/relative_value_signals.py`."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from src import relative_value_signals as rvs


def _spot_frame(close_series: np.ndarray, start: str = "2022-01-01"
                  ) -> pd.DataFrame:
    n = len(close_series)
    ts = pd.date_range(start, periods=n, freq="1D", tz="UTC")
    return pd.DataFrame({
        "timestamp": (ts.astype("datetime64[ns, UTC]")
                        .astype("int64") // 10**6).astype("int64"),
        "datetime": ts,
        "open": close_series, "high": close_series * 1.01,
        "low": close_series * 0.99, "close": close_series,
        "volume": np.ones(n) * 100.0,
    })


# ---------------------------------------------------------------------------
# Schema + ratio
# ---------------------------------------------------------------------------
def test_signals_have_locked_schema():
    n = 400
    btc = _spot_frame(np.linspace(100, 200, n))
    eth = _spot_frame(np.linspace(50, 80, n))
    out = rvs.build_signals(btc, eth)
    expected = [
        "timestamp", "datetime", "btc_close", "eth_close",
        "eth_btc_ratio", "ratio_30d_return", "ratio_90d_return",
        "ratio_ma_200", "ratio_above_ma_200", "ratio_z90",
        "btc_30d_return", "btc_90d_return", "eth_30d_return",
        "eth_90d_return", "btc_above_200dma", "eth_above_200dma",
        "btc_realised_vol_30d", "eth_realised_vol_30d",
        "relative_momentum_score", "relative_trend_score", "regime_state",
    ]
    assert list(out.columns) == expected
    # ETH/BTC ratio matches the inputs.
    assert out["eth_btc_ratio"].iloc[0] == pytest.approx(50.0 / 100.0)


def test_empty_inputs_yield_empty_frame():
    out = rvs.build_signals(pd.DataFrame(), pd.DataFrame())
    assert out.empty


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------
def test_eth_leadership_when_ratio_rising():
    """BTC flat, ETH uptrending so ratio is uptrending and above its MA."""
    n = 400
    rng = np.random.default_rng(11)
    btc = _spot_frame(np.full(n, 100.0) + rng.normal(0, 0.5, n))
    eth = _spot_frame(np.linspace(50, 200, n))
    out = rvs.build_signals(btc, eth)
    last = out.iloc[-1]
    assert last["ratio_above_ma_200"] == 1.0
    assert last["ratio_30d_return"] > 0
    assert last["ratio_90d_return"] > 0
    assert last["regime_state"] == rvs.STATE_ETH_LEADERSHIP


def test_btc_leadership_when_ratio_falling_btc_strong():
    """BTC uptrend, ETH flat → ratio falling, BTC > 200d MA."""
    n = 400
    rng = np.random.default_rng(12)
    btc = _spot_frame(np.linspace(100, 300, n))
    eth = _spot_frame(np.full(n, 50.0) + rng.normal(0, 0.5, n))
    out = rvs.build_signals(btc, eth)
    last = out.iloc[-1]
    assert last["btc_above_200dma"] == 1.0
    assert last["regime_state"] == rvs.STATE_BTC_LEADERSHIP


def test_defensive_when_both_below_200dma():
    n = 400
    rng = np.random.default_rng(13)
    btc = _spot_frame(np.linspace(300, 100, n) + rng.normal(0, 0.3, n))
    eth = _spot_frame(np.linspace(150, 50, n) + rng.normal(0, 0.3, n))
    out = rvs.build_signals(btc, eth)
    last = out.iloc[-1]
    assert last["btc_above_200dma"] == 0.0
    assert last["eth_above_200dma"] == 0.0
    assert last["regime_state"] == rvs.STATE_DEFENSIVE


def test_unstable_rotation_when_z90_extreme():
    """Hold BTC and ETH stable, then a sharp ratio spike at the end —
    z90 should be >= 2.0 → unstable_rotation."""
    n = 400
    rng = np.random.default_rng(14)
    btc = _spot_frame(np.full(n, 100.0) + rng.normal(0, 0.1, n))
    eth_close = np.full(n, 50.0) + rng.normal(0, 0.1, n)
    eth_close[-30:] = np.linspace(50.0, 90.0, 30)   # sharp ratio rise
    eth = _spot_frame(eth_close)
    out = rvs.build_signals(btc, eth)
    last = out.iloc[-1]
    assert last["ratio_z90"] > 2.0
    assert last["regime_state"] == rvs.STATE_UNSTABLE_ROTATION


def test_unknown_warmup_rows():
    n = 50
    btc = _spot_frame(np.linspace(100, 110, n))
    eth = _spot_frame(np.linspace(50, 55, n))
    out = rvs.build_signals(btc, eth)
    assert (out["regime_state"].iloc[:50] == rvs.STATE_UNKNOWN).all()


# ---------------------------------------------------------------------------
# Lookahead invariance — partial-vs-full regression
# ---------------------------------------------------------------------------
def test_no_lookahead_truncating_future_doesnt_change_past():
    rng = np.random.default_rng(2026)
    n = 400
    btc_close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    eth_close = 50 * np.cumprod(1 + rng.normal(0.0003, 0.025, n))
    btc_full = _spot_frame(btc_close)
    eth_full = _spot_frame(eth_close)
    full = rvs.build_signals(btc_full, eth_full)
    btc_short = btc_full.iloc[:300].copy()
    eth_short = eth_full.iloc[:300].copy()
    short = rvs.build_signals(btc_short, eth_short)
    cols = ["ratio_above_ma_200", "ratio_z90", "btc_above_200dma",
             "eth_above_200dma", "regime_state"]
    a = full.iloc[:300].reset_index(drop=True)[cols]
    b = short.iloc[:300].reset_index(drop=True)[cols]
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# Source-level safety — no center=True / shift(-N) / forward-direction merge
# ---------------------------------------------------------------------------
_SIGNAL_SOURCE = (Path(__file__).resolve().parents[1]
                    / "src" / "relative_value_signals.py").read_text()


def test_no_center_true_rolling():
    bad = re.compile(r"\.rolling\([^)]*center\s*=\s*True", re.DOTALL)
    assert bad.search(_SIGNAL_SOURCE) is None


def test_no_negative_shift():
    assert re.search(r"\.shift\(\s*-\s*\d", _SIGNAL_SOURCE) is None


def test_no_forward_direction_merge():
    assert 'direction="forward"' not in _SIGNAL_SOURCE
    assert "direction='forward'" not in _SIGNAL_SOURCE


def test_regime_distribution_sums_to_one():
    df = pd.DataFrame({
        "regime_state": [rvs.STATE_ETH_LEADERSHIP, rvs.STATE_BTC_LEADERSHIP,
                          rvs.STATE_DEFENSIVE, rvs.STATE_NEUTRAL],
    })
    dist = rvs.regime_distribution(df)
    assert pytest.approx(sum(dist.values()), rel=1e-6) == 1.0
