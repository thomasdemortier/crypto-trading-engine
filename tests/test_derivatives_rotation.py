"""Tests for `src.strategies.derivatives_rotation`.

Construct a tiny universe with hand-crafted signal frames and verify:
  * cash filter behaves like the momentum rotation cash filter
  * symbols with `unknown` signal state are skipped
  * `crowded_long` symbols are excluded when `exclude_crowded_long=True`
  * `healthy_trend` symbols are favoured by the composite score
  * `target_weights` never queries data with timestamp > asof
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from src.strategies.derivatives_rotation import (
    DerivativesRotationConfig,
    DerivativesRotationStrategy,
)


def _make_close_series(start_ms: int, days: int, start_price: float,
                        drift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=drift, scale=0.005, size=days)
    closes = start_price * np.cumprod(1.0 + rets)
    ts = np.arange(days, dtype="int64") * 86_400_000 + start_ms
    return pd.DataFrame({"timestamp": ts, "close": closes})


def _make_signal_row(ts_ms: int, sym: str, return_30d: float,
                     crowding_score: float = 0.0,
                     state: str = "neutral") -> Dict:
    return {
        "timestamp": ts_ms, "symbol": sym, "close": 100.0,
        "return_1d": 0.0, "return_7d": 0.0, "return_30d": return_30d,
        "funding_rate": 0.0, "funding_rate_7d_mean": 0.0,
        "funding_rate_30d_mean": 0.0, "funding_rate_zscore_90d": 0.0,
        "funding_extreme_positive": False, "funding_extreme_negative": False,
        "open_interest": 0.0,
        "open_interest_7d_change_pct": 0.0,
        "open_interest_30d_change_pct": 0.0,
        "open_interest_zscore_90d": 0.0,
        "price_up_oi_up": False, "price_up_oi_down": False,
        "price_down_oi_up": False, "price_down_oi_down": False,
        "crowding_score": crowding_score, "capitulation_score": 0.0,
        "squeeze_risk_score": 0.0, "signal_state": state,
    }


def _build_universe(asof_ts: int) -> Dict[str, pd.DataFrame]:
    """3 symbols with daily bars; BTC chosen so its 200d MA is below
    the latest close (cash filter NOT bearish)."""
    start = asof_ts - 250 * 86_400_000
    btc = _make_close_series(start, 251, 100.0, drift=0.001, seed=1)
    eth = _make_close_series(start, 251, 100.0, drift=0.001, seed=2)
    sol = _make_close_series(start, 251, 100.0, drift=0.001, seed=3)
    return {"BTC/USDT": btc, "ETH/USDT": eth, "SOL/USDT": sol}


# ---------------------------------------------------------------------------
def test_unknown_state_is_filtered_out():
    asof = 1_700_000_000_000
    sigs = pd.DataFrame([
        _make_signal_row(asof - 86_400_000, "BTCUSDT", 0.10, state="neutral"),
        _make_signal_row(asof - 86_400_000, "ETHUSDT", 0.20, state="unknown"),
        _make_signal_row(asof - 86_400_000, "SOLUSDT", 0.30, state="neutral"),
    ])
    strat = DerivativesRotationStrategy(
        sigs, DerivativesRotationConfig(top_n=3, min_assets_required=2),
    )
    weights = strat.target_weights(asof, _build_universe(asof), timeframe="1d")
    # ETH should be excluded for being 'unknown'.
    assert "ETH/USDT" not in weights
    # The other two should be selected.
    assert "BTC/USDT" in weights and "SOL/USDT" in weights


def test_crowded_long_is_excluded():
    asof = 1_700_000_000_000
    sigs = pd.DataFrame([
        _make_signal_row(asof - 86_400_000, "BTCUSDT", 0.10, state="neutral"),
        _make_signal_row(asof - 86_400_000, "ETHUSDT", 0.50,
                         crowding_score=0.9, state="crowded_long"),
        _make_signal_row(asof - 86_400_000, "SOLUSDT", 0.20, state="neutral"),
    ])
    strat = DerivativesRotationStrategy(
        sigs, DerivativesRotationConfig(top_n=3, min_assets_required=2),
    )
    weights = strat.target_weights(asof, _build_universe(asof), timeframe="1d")
    # ETH has the best raw return but is crowded → excluded.
    assert "ETH/USDT" not in weights
    # BTC + SOL share the basket equally.
    assert weights == {"BTC/USDT": 0.5, "SOL/USDT": 0.5}


def test_healthy_trend_bonus_breaks_ties():
    asof = 1_700_000_000_000
    # Same return_30d on both; only ETH has the 'healthy_trend' bonus,
    # so it should outscore BTC.
    sigs = pd.DataFrame([
        _make_signal_row(asof - 86_400_000, "BTCUSDT", 0.10, state="neutral"),
        _make_signal_row(asof - 86_400_000, "ETHUSDT", 0.10, state="healthy_trend"),
        _make_signal_row(asof - 86_400_000, "SOLUSDT", 0.05, state="neutral"),
    ])
    strat = DerivativesRotationStrategy(
        sigs, DerivativesRotationConfig(top_n=1, min_assets_required=2),
    )
    weights = strat.target_weights(asof, _build_universe(asof), timeframe="1d")
    assert weights == {"ETH/USDT": 1.0}


def test_min_assets_required_returns_cash_when_unmet():
    asof = 1_700_000_000_000
    # Only BTC has a usable signal; ETH/SOL are unknown.
    sigs = pd.DataFrame([
        _make_signal_row(asof - 86_400_000, "BTCUSDT", 0.10, state="neutral"),
        _make_signal_row(asof - 86_400_000, "ETHUSDT", 0.10, state="unknown"),
        _make_signal_row(asof - 86_400_000, "SOLUSDT", 0.10, state="unknown"),
    ])
    strat = DerivativesRotationStrategy(
        sigs, DerivativesRotationConfig(top_n=3, min_assets_required=2),
    )
    weights = strat.target_weights(asof, _build_universe(asof), timeframe="1d")
    assert weights == {}


def test_cash_filter_bearish_blocks_entry():
    """Force BTC below its 200d MA on the asof bar — strategy must return {}."""
    asof = 1_700_000_000_000
    # Build a BTC series that falls hard at the end so close < 200d MA.
    days = 251
    start = asof - (days - 1) * 86_400_000
    closes = np.concatenate([
        np.linspace(100.0, 200.0, num=days - 30),  # 200d MA gets pulled up
        np.linspace(200.0, 50.0, num=30),          # then drops below MA
    ])
    btc = pd.DataFrame({
        "timestamp": np.arange(days, dtype="int64") * 86_400_000 + start,
        "close": closes,
    })
    universe = {"BTC/USDT": btc, "ETH/USDT": btc.copy(), "SOL/USDT": btc.copy()}

    sigs = pd.DataFrame([
        _make_signal_row(asof - 86_400_000, "BTCUSDT", 0.10, state="healthy_trend"),
        _make_signal_row(asof - 86_400_000, "ETHUSDT", 0.10, state="healthy_trend"),
        _make_signal_row(asof - 86_400_000, "SOLUSDT", 0.10, state="healthy_trend"),
    ])
    strat = DerivativesRotationStrategy(sigs)
    weights = strat.target_weights(asof, universe, timeframe="1d")
    assert weights == {}


def test_lookahead_signal_with_future_timestamp_is_ignored():
    """A signal row at asof+1d must NEVER influence the asof decision."""
    asof = 1_700_000_000_000
    sigs = pd.DataFrame([
        _make_signal_row(asof + 86_400_000, "ETHUSDT", 1.00,
                         crowding_score=0.0, state="healthy_trend"),
        _make_signal_row(asof - 86_400_000, "BTCUSDT", 0.10, state="neutral"),
        _make_signal_row(asof - 86_400_000, "SOLUSDT", 0.05, state="neutral"),
    ])
    strat = DerivativesRotationStrategy(
        sigs, DerivativesRotationConfig(top_n=1, min_assets_required=2),
    )
    weights = strat.target_weights(asof, _build_universe(asof), timeframe="1d")
    # ETH would dominate IF the future row leaked. It must NOT.
    assert "ETH/USDT" not in weights


def test_empty_signals_returns_cash():
    asof = 1_700_000_000_000
    strat = DerivativesRotationStrategy(pd.DataFrame())
    weights = strat.target_weights(asof, _build_universe(asof), timeframe="1d")
    assert weights == {}
