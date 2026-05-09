"""Tests for `src.strategies.market_structure_allocator`."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.strategies.market_structure_allocator import (
    MarketStructureAllocatorConfig, MarketStructureAllocatorStrategy,
)


# ---------------------------------------------------------------------------
# Universe + signal helpers
# ---------------------------------------------------------------------------
def _close_series(start_ms: int, days: int, drift: float = 0.0,
                   start_price: float = 100.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=drift, scale=0.005, size=days)
    closes = start_price * np.cumprod(1.0 + rets)
    ts = np.arange(days, dtype="int64") * 86_400_000 + start_ms
    return pd.DataFrame({"timestamp": ts, "close": closes})


def _universe(asof_ts: int, days: int = 250) -> Dict[str, pd.DataFrame]:
    start = asof_ts - (days - 1) * 86_400_000
    return {
        "BTC/USDT": _close_series(start, days, drift=0.001, seed=1),
        "ETH/USDT": _close_series(start, days, drift=0.002, seed=2),
        "SOL/USDT": _close_series(start, days, drift=0.003, seed=3),
        "AVAX/USDT": _close_series(start, days, drift=0.0, seed=4),
        "LINK/USDT": _close_series(start, days, drift=-0.001, seed=5),
        "XRP/USDT": _close_series(start, days, drift=0.001, seed=6),
        "DOGE/USDT": _close_series(start, days, drift=-0.002, seed=7),
        "ADA/USDT": _close_series(start, days, drift=0.001, seed=8),
        "LTC/USDT": _close_series(start, days, drift=0.0, seed=9),
        "BNB/USDT": _close_series(start, days, drift=0.0015, seed=10),
    }


def _sig_row(ts_ms: int, state: str) -> dict:
    return {"timestamp": ts_ms, "date": "2024-01-01",
             "btc_close": 100.0, "btc_return_30d": 0.0,
             "btc_return_90d": 0.0, "btc_above_200d_ma": True,
             "total_tvl": 0.0, "total_tvl_return_30d": 0.0,
             "total_tvl_return_90d": 0.0,
             "stablecoin_supply": 0.0,
             "stablecoin_supply_return_30d": 0.0,
             "stablecoin_supply_return_90d": 0.0,
             "btc_market_cap": 0.0, "btc_market_cap_return_30d": 0.0,
             "btc_hash_rate": 0.0, "btc_hash_rate_return_30d": 0.0,
             "btc_transactions": 0.0, "btc_transactions_return_30d": 0.0,
             "alt_basket_return_30d": 0.0, "alt_basket_return_90d": 0.0,
             "alt_basket_above_200d_ma_pct": 0.5,
             "alt_basket_vs_btc_30d": 0.0, "alt_basket_vs_btc_90d": 0.0,
             "liquidity_score": 0.0, "onchain_health_score": 0.0,
             "alt_risk_score": 0.0, "defensive_score": 0.0,
             "market_structure_state": state}


# ---------------------------------------------------------------------------
# State -> allocation
# ---------------------------------------------------------------------------
def test_defensive_state_returns_cash():
    asof = 1_700_000_000_000
    sigs = pd.DataFrame([_sig_row(asof - 86_400_000, "defensive")])
    strat = MarketStructureAllocatorStrategy(sigs)
    assert strat.target_weights(asof, _universe(asof), timeframe="1d") == {}


def test_unknown_state_returns_cash():
    asof = 1_700_000_000_000
    sigs = pd.DataFrame([_sig_row(asof - 86_400_000, "unknown")])
    strat = MarketStructureAllocatorStrategy(sigs)
    assert strat.target_weights(asof, _universe(asof), timeframe="1d") == {}


def test_btc_leadership_holds_100pct_btc():
    asof = 1_700_000_000_000
    sigs = pd.DataFrame([_sig_row(asof - 86_400_000, "btc_leadership")])
    strat = MarketStructureAllocatorStrategy(sigs)
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    assert weights == {"BTC/USDT": 1.0}


def test_neutral_holds_100pct_btc():
    asof = 1_700_000_000_000
    sigs = pd.DataFrame([_sig_row(asof - 86_400_000, "neutral")])
    strat = MarketStructureAllocatorStrategy(sigs)
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    assert weights == {"BTC/USDT": 1.0}


def test_alt_risk_on_holds_top_alts():
    asof = 1_700_000_000_000
    sigs = pd.DataFrame([_sig_row(asof - 86_400_000, "alt_risk_on")])
    strat = MarketStructureAllocatorStrategy(
        sigs, MarketStructureAllocatorConfig(alt_top_n=5),
    )
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    # 5 alts, equal-weighted; total = 1.0; BTC NOT in the basket.
    assert "BTC/USDT" not in weights
    assert len(weights) == 5
    assert sum(weights.values()) == pytest.approx(1.0)
    for w in weights.values():
        assert w == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Long-only / leverage invariants
# ---------------------------------------------------------------------------
def test_weights_never_exceed_100pct_in_any_state():
    asof = 1_700_000_000_000
    for state in ("alt_risk_on", "btc_leadership", "neutral",
                    "defensive", "unknown"):
        sigs = pd.DataFrame([_sig_row(asof - 86_400_000, state)])
        strat = MarketStructureAllocatorStrategy(sigs)
        weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
        total = sum(weights.values())
        assert total <= 1.0 + 1e-9, (state, total)
        for w in weights.values():
            assert w >= 0.0


def test_no_leverage_or_short_in_alt_risk_on():
    asof = 1_700_000_000_000
    sigs = pd.DataFrame([_sig_row(asof - 86_400_000, "alt_risk_on")])
    strat = MarketStructureAllocatorStrategy(sigs)
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    for w in weights.values():
        assert 0.0 <= w <= 1.0


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------
def test_assets_without_enough_history_excluded():
    asof = 1_700_000_000_000
    # Build a universe where SOL only has 30 days of history.
    universe = _universe(asof, days=250)
    short_start = asof - 29 * 86_400_000
    universe["SOL/USDT"] = _close_series(short_start, 30, drift=0.05, seed=11)
    sigs = pd.DataFrame([_sig_row(asof - 86_400_000, "alt_risk_on")])
    strat = MarketStructureAllocatorStrategy(
        sigs, MarketStructureAllocatorConfig(alt_top_n=5,
                                                momentum_window_days=90,
                                                min_history_days=90),
    )
    weights = strat.target_weights(asof, universe, timeframe="1d")
    # SOL has the strongest drift but insufficient history → must NOT
    # be in the basket.
    assert "SOL/USDT" not in weights


# ---------------------------------------------------------------------------
# Lookahead invariant
# ---------------------------------------------------------------------------
def test_future_signal_row_is_ignored():
    asof = 1_700_000_000_000
    sigs = pd.DataFrame([
        _sig_row(asof - 86_400_000, "defensive"),
        _sig_row(asof + 86_400_000, "alt_risk_on"),  # future row
    ])
    strat = MarketStructureAllocatorStrategy(sigs)
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    # The defensive row must dominate; the future alt_risk_on must NOT leak.
    assert weights == {}


def test_empty_signal_frame_returns_cash():
    asof = 1_700_000_000_000
    strat = MarketStructureAllocatorStrategy(pd.DataFrame())
    assert strat.target_weights(asof, _universe(asof), timeframe="1d") == {}
