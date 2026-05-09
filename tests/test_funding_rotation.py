"""Tests for `src.strategies.funding_rotation`.

Hand-built signal frames and a small synthetic universe verify:
  * `unknown` rows are skipped.
  * `crowded_long` rows are excluded (long-only invariant).
  * `capitulation` allowed only when 7d return is not in free-fall.
  * Cash filter behaves like the momentum rotation cash filter.
  * `target_weights` never queries data with timestamp > asof.
  * `min_assets_required` returns cash when not enough qualify.
  * Empty signal frame returns cash.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.strategies.funding_rotation import (
    FundingRotationConfig, FundingRotationStrategy,
)


def _close_series(start_ms: int, days: int, drift: float = 0.0,
                   start_price: float = 100.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=drift, scale=0.005, size=days)
    closes = start_price * np.cumprod(1.0 + rets)
    ts = np.arange(days, dtype="int64") * 86_400_000 + start_ms
    return pd.DataFrame({"timestamp": ts, "close": closes})


def _sig(ts_ms: int, sym: str, return_30d: float, return_7d: float = 0.0,
          state: str = "neutral", attr: float = 0.5, improve: float = 0.5,
          stab: float = 0.5, crowd: float = 0.0) -> dict:
    return {
        "timestamp": ts_ms, "symbol": sym, "close": 100.0,
        "return_1d": 0.0, "return_7d": return_7d, "return_30d": return_30d,
        "funding_rate": 0.0001, "funding_7d_mean": 0.0001,
        "funding_30d_mean": 0.0001, "funding_90d_zscore": 0.0,
        "extreme_positive_funding": False, "extreme_negative_funding": False,
        "funding_trend": 0.0, "funding_normalization": 1.0,
        "price_return_7d": return_7d, "price_return_30d": return_30d,
        "funding_attractiveness": attr,
        "funding_improvement": improve,
        "stabilization_score": stab,
        "crowding_penalty": crowd,
        "funding_state": state,
    }


def _universe(asof_ts: int, bullish: bool = True) -> Dict[str, pd.DataFrame]:
    days = 251
    start = asof_ts - (days - 1) * 86_400_000
    drift = 0.001 if bullish else -0.001
    btc = _close_series(start, days, drift=drift, seed=1)
    eth = _close_series(start, days, drift=drift, seed=2)
    sol = _close_series(start, days, drift=drift, seed=3)
    return {"BTC/USDT": btc, "ETH/USDT": eth, "SOL/USDT": sol}


# ---------------------------------------------------------------------------
def test_unknown_state_skipped():
    asof = 1_700_000_000_000
    sigs = pd.DataFrame([
        _sig(asof - 86_400_000, "BTCUSDT", 0.10, state="neutral"),
        _sig(asof - 86_400_000, "ETHUSDT", 0.20, state="unknown"),
        _sig(asof - 86_400_000, "SOLUSDT", 0.05, state="neutral"),
    ])
    strat = FundingRotationStrategy(
        sigs, FundingRotationConfig(top_n=3, min_assets_required=2),
    )
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    assert "ETH/USDT" not in weights
    assert "BTC/USDT" in weights and "SOL/USDT" in weights


def test_crowded_long_excluded_when_long_only():
    asof = 1_700_000_000_000
    sigs = pd.DataFrame([
        _sig(asof - 86_400_000, "BTCUSDT", 0.10, state="neutral"),
        _sig(asof - 86_400_000, "ETHUSDT", 0.50,
              state="crowded_long", crowd=1.0),
        _sig(asof - 86_400_000, "SOLUSDT", 0.20, state="neutral"),
    ])
    strat = FundingRotationStrategy(
        sigs, FundingRotationConfig(top_n=3, min_assets_required=2),
    )
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    assert "ETH/USDT" not in weights
    assert weights == {"BTC/USDT": 0.5, "SOL/USDT": 0.5}


def test_capitulation_allowed_only_when_price_stabilised():
    asof = 1_700_000_000_000
    sigs = pd.DataFrame([
        # Free-falling capitulation — 7d return < -5 % → must be excluded.
        _sig(asof - 86_400_000, "BTCUSDT", -0.30, return_7d=-0.10,
              state="capitulation"),
        # Stabilised capitulation — 7d return ~0 → admissible.
        _sig(asof - 86_400_000, "ETHUSDT", -0.20, return_7d=0.01,
              state="capitulation"),
        _sig(asof - 86_400_000, "SOLUSDT", 0.10, state="neutral"),
    ])
    strat = FundingRotationStrategy(
        sigs, FundingRotationConfig(
            top_n=2, min_assets_required=2,
            allow_capitulation=True,
            capitulation_max_drawdown_7d=0.05,
        ),
    )
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    assert "BTC/USDT" not in weights
    # ETH (stabilised cap) and SOL (neutral) should be the two picks.
    assert set(weights.keys()) == {"ETH/USDT", "SOL/USDT"}


def test_min_assets_required_returns_cash():
    asof = 1_700_000_000_000
    sigs = pd.DataFrame([
        _sig(asof - 86_400_000, "BTCUSDT", 0.10, state="neutral"),
        _sig(asof - 86_400_000, "ETHUSDT", 0.10, state="unknown"),
        _sig(asof - 86_400_000, "SOLUSDT", 0.10, state="unknown"),
    ])
    strat = FundingRotationStrategy(
        sigs, FundingRotationConfig(top_n=3, min_assets_required=3),
    )
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    assert weights == {}


def test_cash_filter_blocks_when_btc_below_200d_ma():
    asof = 1_700_000_000_000
    days = 251
    start = asof - (days - 1) * 86_400_000
    closes = np.concatenate([
        np.linspace(100.0, 200.0, num=days - 30),
        np.linspace(200.0, 50.0, num=30),
    ])
    btc = pd.DataFrame({
        "timestamp": np.arange(days, dtype="int64") * 86_400_000 + start,
        "close": closes,
    })
    universe = {"BTC/USDT": btc, "ETH/USDT": btc.copy(), "SOL/USDT": btc.copy()}
    sigs = pd.DataFrame([
        _sig(asof - 86_400_000, sym, 0.10, state="neutral")
        for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    ])
    strat = FundingRotationStrategy(sigs)
    assert strat.target_weights(asof, universe, timeframe="1d") == {}


def test_lookahead_future_signal_is_ignored():
    """A signal row at asof+1d must NEVER influence the asof decision."""
    asof = 1_700_000_000_000
    sigs = pd.DataFrame([
        _sig(asof + 86_400_000, "ETHUSDT", 1.0,
              attr=1.0, improve=1.0, state="neutral"),
        _sig(asof - 86_400_000, "BTCUSDT", 0.05, state="neutral"),
        _sig(asof - 86_400_000, "SOLUSDT", 0.05, state="neutral"),
    ])
    strat = FundingRotationStrategy(
        sigs, FundingRotationConfig(top_n=1, min_assets_required=2),
    )
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    assert "ETH/USDT" not in weights


def test_empty_signals_returns_cash():
    asof = 1_700_000_000_000
    strat = FundingRotationStrategy(pd.DataFrame())
    assert strat.target_weights(asof, _universe(asof), timeframe="1d") == {}
