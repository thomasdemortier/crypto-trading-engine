"""Tests for `src.strategies.market_structure_vol_target_allocator`."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.strategies.market_structure_vol_target_allocator import (
    MarketStructureVolTargetConfig, MarketStructureVolTargetAllocatorStrategy,
)


def _close_series(start_ms: int, days: int, drift: float = 0.0,
                   start_price: float = 100.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=drift, scale=0.005, size=days)
    closes = start_price * np.cumprod(1.0 + rets)
    ts = np.arange(days, dtype="int64") * 86_400_000 + start_ms
    return pd.DataFrame({"timestamp": ts, "close": closes})


def _universe(asof_ts: int, days: int = 250) -> Dict[str, pd.DataFrame]:
    start = asof_ts - (days - 1) * 86_400_000
    drifts = {"BTC/USDT": 0.001, "ETH/USDT": 0.002, "SOL/USDT": 0.003,
               "AVAX/USDT": 0.0, "LINK/USDT": -0.001, "XRP/USDT": 0.001,
               "DOGE/USDT": -0.002, "ADA/USDT": 0.001, "LTC/USDT": 0.0,
               "BNB/USDT": 0.0015}
    return {sym: _close_series(start, days, drift=d, seed=i + 1)
             for i, (sym, d) in enumerate(drifts.items())}


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
def test_defensive_holds_30pct_btc_70pct_cash():
    asof = 1_700_000_000_000
    sigs = pd.DataFrame([_sig_row(asof - 86_400_000, "defensive")])
    strat = MarketStructureVolTargetAllocatorStrategy(sigs)
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    assert weights == {"BTC/USDT": pytest.approx(0.30)}
    # Cash share = 1 - sum(weights).
    cash = 1.0 - sum(weights.values())
    assert cash == pytest.approx(0.70)


def test_neutral_holds_70pct_btc_30pct_cash():
    asof = 1_700_000_000_000
    sigs = pd.DataFrame([_sig_row(asof - 86_400_000, "neutral")])
    strat = MarketStructureVolTargetAllocatorStrategy(sigs)
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    assert weights == {"BTC/USDT": pytest.approx(0.70)}
    assert 1.0 - sum(weights.values()) == pytest.approx(0.30)


def test_btc_leadership_holds_100pct_btc():
    asof = 1_700_000_000_000
    sigs = pd.DataFrame([_sig_row(asof - 86_400_000, "btc_leadership")])
    strat = MarketStructureVolTargetAllocatorStrategy(sigs)
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    assert weights == {"BTC/USDT": pytest.approx(1.0)}


def test_alt_risk_on_holds_70pct_alts_30pct_btc():
    asof = 1_700_000_000_000
    sigs = pd.DataFrame([_sig_row(asof - 86_400_000, "alt_risk_on")])
    strat = MarketStructureVolTargetAllocatorStrategy(
        sigs, MarketStructureVolTargetConfig(alt_top_n=5),
    )
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    btc_w = weights.get("BTC/USDT", 0.0)
    alt_total = sum(w for sym, w in weights.items() if sym != "BTC/USDT")
    assert btc_w == pytest.approx(0.30)
    assert alt_total == pytest.approx(0.70)
    # Each of the 5 alts equal-weighted: 0.70 / 5 = 0.14.
    alt_count = len([w for sym, w in weights.items() if sym != "BTC/USDT"])
    assert alt_count == 5
    for sym, w in weights.items():
        if sym == "BTC/USDT":
            continue
        assert w == pytest.approx(0.14)


def test_unknown_state_holds_100pct_cash():
    asof = 1_700_000_000_000
    sigs = pd.DataFrame([_sig_row(asof - 86_400_000, "unknown")])
    strat = MarketStructureVolTargetAllocatorStrategy(sigs)
    assert strat.target_weights(asof, _universe(asof), timeframe="1d") == {}


# ---------------------------------------------------------------------------
# Long-only / leverage invariants
# ---------------------------------------------------------------------------
def test_weights_never_exceed_100pct_in_any_state():
    asof = 1_700_000_000_000
    for state in ("alt_risk_on", "btc_leadership", "neutral",
                    "defensive", "unknown"):
        sigs = pd.DataFrame([_sig_row(asof - 86_400_000, state)])
        strat = MarketStructureVolTargetAllocatorStrategy(sigs)
        weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
        total = sum(weights.values())
        assert total <= 1.0 + 1e-9, (state, total)
        for w in weights.values():
            assert 0.0 <= w <= 1.0


def test_no_leverage_in_alt_risk_on():
    """The 70/30 split must total 100%, never 110% or 130%."""
    asof = 1_700_000_000_000
    sigs = pd.DataFrame([_sig_row(asof - 86_400_000, "alt_risk_on")])
    strat = MarketStructureVolTargetAllocatorStrategy(sigs)
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    assert sum(weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Eligibility / lookahead
# ---------------------------------------------------------------------------
def test_btc_state_returns_cash_when_btc_history_too_short():
    asof = 1_700_000_000_000
    universe = _universe(asof, days=250)
    short_start = asof - 29 * 86_400_000
    universe["BTC/USDT"] = _close_series(short_start, 30, seed=99)
    sigs = pd.DataFrame([_sig_row(asof - 86_400_000, "neutral")])
    strat = MarketStructureVolTargetAllocatorStrategy(
        sigs, MarketStructureVolTargetConfig(min_history_days=90),
    )
    # BTC history < 90 days → can't allocate → cash.
    assert strat.target_weights(asof, universe, timeframe="1d") == {}


def test_future_signal_row_is_ignored():
    asof = 1_700_000_000_000
    sigs = pd.DataFrame([
        _sig_row(asof - 86_400_000, "defensive"),
        _sig_row(asof + 86_400_000, "alt_risk_on"),  # future
    ])
    strat = MarketStructureVolTargetAllocatorStrategy(sigs)
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    # Defensive must dominate; future row must NOT leak.
    assert weights == {"BTC/USDT": pytest.approx(0.30)}


def test_empty_signal_frame_returns_cash():
    asof = 1_700_000_000_000
    strat = MarketStructureVolTargetAllocatorStrategy(pd.DataFrame())
    assert strat.target_weights(asof, _universe(asof), timeframe="1d") == {}
