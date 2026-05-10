"""Tests for `src/funding_basis_signals.py`.

Verifies:
    * Schema is locked.
    * Regime classifications are correct on synthetic inputs:
        - high funding + high basis + extended price -> crowded_long
        - negative funding + healthy trend          -> cheap_bullish
        - negative funding + weak trend             -> stress_negative_funding
        - below 200d MA                             -> defensive
    * Warm-up rows yield `unknown` (insufficient history for rollers).
    * Rollers are backward-only (truncating future history doesn't
      change past rows).
    * No center=True / shift(-N) / future-data joins in the source.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from src import funding_basis_signals as fbs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def _funding_frame(rate_per_8h_series: np.ndarray,
                    start: str = "2022-01-01") -> pd.DataFrame:
    """Make a synthetic 8h funding frame matching the collector schema."""
    n = len(rate_per_8h_series)
    ts = pd.date_range(start, periods=n, freq="8h", tz="UTC")
    return pd.DataFrame({
        "timestamp": (ts.astype("datetime64[ns, UTC]")
                        .astype("int64") // 10**6).astype("int64"),
        "datetime": ts,
        "funding_rate": rate_per_8h_series.astype("float64"),
        "source": "synthetic", "symbol_or_instrument": "BTCUSDT",
    })


def _klines_frame(close_series: np.ndarray, kind: str,
                    start: str = "2022-01-01") -> pd.DataFrame:
    n = len(close_series)
    ts = pd.date_range(start, periods=n, freq="1D", tz="UTC")
    return pd.DataFrame({
        "timestamp": (ts.astype("datetime64[ns, UTC]")
                        .astype("int64") // 10**6).astype("int64"),
        "datetime": ts,
        "open": close_series, "high": close_series, "low": close_series,
        "close": close_series, "volume": np.ones(n) * 100.0,
        "source": "synthetic", "symbol": "BTCUSDT", "kind": kind,
    })


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_signals_have_locked_schema():
    n = 400
    spot = _spot_frame(np.linspace(100, 200, n))
    funding = _funding_frame(np.full(n * 3, 0.0001))
    mark = _klines_frame(np.linspace(100, 200, n), kind="mark")
    index = _klines_frame(np.linspace(100, 200, n), kind="index")
    out = fbs.build_signals_for_asset("BTC/USDT", spot,
                                          [funding], mark, index)
    expected = [
        "timestamp", "datetime", "asset", "close", "above_200dma",
        "funding_1d_avg", "funding_7d_avg", "funding_30d_avg",
        "funding_z90", "funding_pct_rank_365", "funding_trend_pos",
        "basis", "basis_7d_avg", "basis_30d_avg", "basis_z90",
        "price_return_30d", "price_return_90d", "realised_vol_30d",
        "crowding_score", "carry_attractiveness", "regime_state",
    ]
    assert list(out.columns) == expected


def test_warmup_rows_classified_unknown():
    n = 50
    spot = _spot_frame(np.linspace(100, 110, n))
    funding = _funding_frame(np.full(n * 3, 0.0001))
    mark = _klines_frame(np.linspace(100, 110, n), kind="mark")
    index = _klines_frame(np.linspace(100, 110, n), kind="index")
    out = fbs.build_signals_for_asset("BTC/USDT", spot,
                                          [funding], mark, index)
    # < 200 days history → above_200dma is NaN → state is unknown.
    assert (out["regime_state"].iloc[:50] == fbs.STATE_UNKNOWN).all()


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------
def test_high_funding_high_basis_extended_price_yields_crowded_long():
    n = 400
    # First 380 days: slow ramp 100->150 (uptrend). Last 20 days:
    # fast 150->200 (extended price >+10% over 30d).
    base = np.linspace(100, 150, 380)
    spike = np.linspace(150, 200, 20)
    close = np.concatenate([base, spike])
    spot = _spot_frame(close)

    # Funding flat 0 for 380 days, then very high for last 20 days.
    f = np.zeros(n * 3)
    f[-60:] = 0.0006   # 8h ≈ 65% annualised
    funding = _funding_frame(f)

    # Mark > index by 1.5% for last 20 days (basis stretched).
    mark = _klines_frame(close * np.where(np.arange(n) >= 380, 1.015, 1.0),
                            kind="mark")
    index = _klines_frame(close, kind="index")

    out = fbs.build_signals_for_asset("BTC/USDT", spot,
                                          [funding], mark, index)
    last = out.iloc[-1]
    assert last["regime_state"] == fbs.STATE_CROWDED_LONG


def test_negative_funding_with_healthy_trend_is_cheap_bullish():
    n = 400
    rng = np.random.default_rng(11)
    close = np.linspace(100, 200, n)  # straight uptrend (above 200d MA)
    spot = _spot_frame(close)
    # Funding mildly positive for most of the window then sharply
    # negative for the last ~90 days — clearly negative funding_z90
    # at the final bar.
    f = np.full(n * 3, 0.0001)
    f[-90:] = -0.0003
    funding = _funding_frame(f)
    # Mark has small noise around index → non-zero basis variance, so
    # basis_z90 is finite. Average basis stays near 0 (not stretched).
    noise = rng.normal(0.0, 0.001, n)
    mark = _klines_frame(close * (1.0 + noise), kind="mark")
    index = _klines_frame(close, kind="index")
    out = fbs.build_signals_for_asset("BTC/USDT", spot,
                                          [funding], mark, index)
    last = out.iloc[-1]
    assert last["above_200dma"] == 1.0
    assert last["funding_z90"] < -0.5
    assert last["regime_state"] == fbs.STATE_CHEAP_BULLISH


def test_negative_funding_with_weak_trend_yields_stress_or_defensive():
    n = 400
    rng = np.random.default_rng(12)
    close = np.linspace(200, 100, n)   # straight downtrend
    spot = _spot_frame(close)
    # Mildly noisy negative funding so funding_z90 is finite.
    f = -0.0002 + rng.normal(0.0, 0.00005, n * 3)
    funding = _funding_frame(f)
    noise = rng.normal(0.0, 0.001, n)
    mark = _klines_frame(close * (1.0 + noise), kind="mark")
    index = _klines_frame(close, kind="index")
    out = fbs.build_signals_for_asset("BTC/USDT", spot,
                                          [funding], mark, index)
    last = out.iloc[-1]
    assert last["above_200dma"] == 0.0
    assert last["funding_1d_avg"] < 0
    # Either stress_negative_funding or defensive is allowed by spec.
    assert last["regime_state"] in (fbs.STATE_STRESS_NEGATIVE_FUNDING,
                                         fbs.STATE_DEFENSIVE)


def test_below_200dma_yields_defensive_when_funding_neutral():
    n = 400
    rng = np.random.default_rng(13)
    # 200d uptrend then steep crash below MA.
    base = np.linspace(50, 250, 300)
    crash = np.linspace(250, 90, 100)
    close = np.concatenate([base, crash])
    spot = _spot_frame(close)
    f = 0.0001 + rng.normal(0.0, 0.00005, n * 3)
    funding = _funding_frame(f)
    noise = rng.normal(0.0, 0.001, n)
    mark = _klines_frame(close * (1.0 + noise), kind="mark")
    index = _klines_frame(close, kind="index")
    out = fbs.build_signals_for_asset("BTC/USDT", spot,
                                          [funding], mark, index)
    last = out.iloc[-1]
    assert last["above_200dma"] == 0.0
    assert last["regime_state"] in (fbs.STATE_DEFENSIVE,
                                         fbs.STATE_STRESS_NEGATIVE_FUNDING)


# ---------------------------------------------------------------------------
# Lookahead invariance
# ---------------------------------------------------------------------------
def test_no_lookahead_truncating_future_doesnt_change_past():
    rng = np.random.default_rng(2026)
    n = 400
    rets = rng.normal(0.0005, 0.02, n)
    close = 100 * np.cumprod(1 + rets)
    spot_full = _spot_frame(close)
    funding_full = _funding_frame(rng.normal(0.0001, 0.0001, n * 3))
    mark_full = _klines_frame(close * 1.001, kind="mark")
    index_full = _klines_frame(close, kind="index")
    full = fbs.build_signals_for_asset("BTC/USDT", spot_full,
                                          [funding_full], mark_full,
                                          index_full)
    # Truncate the inputs to first 300 bars and rebuild.
    spot_trunc = spot_full.iloc[:300].copy()
    fund_cut = funding_full[funding_full["timestamp"]
                              <= int(spot_trunc["timestamp"].iloc[-1])]
    mark_trunc = mark_full.iloc[:300]
    index_trunc = index_full.iloc[:300]
    short = fbs.build_signals_for_asset("BTC/USDT", spot_trunc,
                                            [fund_cut], mark_trunc,
                                            index_trunc)
    # Compare the OVERLAP: rows 0..299 should be identical between
    # the two runs because every roller is backward-only.
    cols = ["funding_z90", "basis_z90", "above_200dma",
             "price_return_30d", "regime_state"]
    a = full.iloc[:300].reset_index(drop=True)[cols]
    b = short.iloc[:300].reset_index(drop=True)[cols]
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# Source-level safety: no future-data joins or center=True rollers
# ---------------------------------------------------------------------------
_SIGNAL_SOURCE = (Path(__file__).resolve().parents[1]
                    / "src" / "funding_basis_signals.py").read_text()


def test_no_center_true_rolling():
    """No `.rolling(..., center=True)` API call (docstring mentions
    are fine — we only flag actual calls)."""
    bad = re.compile(r"\.rolling\([^)]*center\s*=\s*True", re.DOTALL)
    assert bad.search(_SIGNAL_SOURCE) is None


def test_no_negative_shift():
    # `shift(-N)` would peek at the future.
    assert re.search(r"\.shift\(\s*-\s*\d", _SIGNAL_SOURCE) is None


def test_merge_uses_backward_direction():
    # We do not use forward-looking merge_asof anywhere.
    assert 'direction="forward"' not in _SIGNAL_SOURCE
    assert "direction='forward'" not in _SIGNAL_SOURCE


def test_regime_distribution_sums_to_one():
    out = pd.DataFrame({
        "asset": ["BTC/USDT"] * 4,
        "regime_state": [fbs.STATE_CROWDED_LONG, fbs.STATE_NEUTRAL_RISK_ON,
                          fbs.STATE_CHEAP_BULLISH, fbs.STATE_DEFENSIVE],
    })
    dist = fbs.regime_distribution(out)
    assert pytest.approx(sum(dist.values()), rel=1e-6) == 1.0
