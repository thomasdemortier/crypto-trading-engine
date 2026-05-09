"""Tests for `src.strategies.sentiment_market_structure_allocator`."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.strategies.sentiment_market_structure_allocator import (
    SentimentMarketStructureAllocatorConfig,
    SentimentMarketStructureAllocatorStrategy,
)


# ---------------------------------------------------------------------------
# Synthetic universe + signal helpers
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
    drifts = {"BTC/USDT": 0.001, "ETH/USDT": 0.002, "SOL/USDT": 0.003,
               "AVAX/USDT": 0.0, "LINK/USDT": -0.001, "XRP/USDT": 0.001,
               "DOGE/USDT": -0.002, "ADA/USDT": 0.001, "LTC/USDT": 0.0,
               "BNB/USDT": 0.0015}
    return {sym: _close_series(start, days, drift=d, seed=i + 1)
             for i, (sym, d) in enumerate(drifts.items())}


def _ms_sig_row(ts_ms: int, state: str,
                 btc_above_200d_ma: bool = True) -> dict:
    return {"timestamp": ts_ms, "date": "2024-01-01",
             "btc_close": 100.0, "btc_return_30d": 0.0,
             "btc_return_90d": 0.0, "btc_above_200d_ma": btc_above_200d_ma,
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


def _sent_row(ts_ms: int, state: str) -> dict:
    return {"timestamp": ts_ms, "date": "2024-01-01",
             "fear_greed_value": 50, "fear_greed_classification": "Neutral",
             "fg_7d_change": 0.0, "fg_30d_change": 0.0,
             "fg_7d_mean": 50.0, "fg_30d_mean": 50.0, "fg_90d_zscore": 0.0,
             "extreme_fear": False, "fear": False, "neutral": True,
             "greed": False, "extreme_greed": False,
             "sentiment_recovering": False, "sentiment_deteriorating": False,
             "sentiment_state": state}


def _build(state_ms: str, state_sent: str, btc_above: bool = True):
    asof = 1_700_000_000_000
    ms = pd.DataFrame([_ms_sig_row(asof - 86_400_000, state_ms,
                                      btc_above_200d_ma=btc_above)])
    sent = pd.DataFrame([_sent_row(asof - 86_400_000, state_sent)])
    strat = SentimentMarketStructureAllocatorStrategy(ms, sent)
    return asof, strat


# ---------------------------------------------------------------------------
# Pass-through behavior
# ---------------------------------------------------------------------------
def test_neutral_sentiment_passes_base_through_unchanged():
    asof, strat = _build("btc_leadership", "neutral")
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    assert weights == {"BTC/USDT": pytest.approx(1.0)}


def test_unknown_sentiment_passes_base_through_unchanged():
    asof, strat = _build("alt_risk_on", "unknown")
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    btc = weights.get("BTC/USDT", 0.0)
    alts = sum(v for k, v in weights.items() if k != "BTC/USDT")
    assert btc == pytest.approx(0.30)
    assert alts == pytest.approx(0.70)


def test_fear_recovery_passes_base_through_unchanged():
    asof, strat = _build("neutral", "fear_recovery")
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    # Base 'neutral' = 70% BTC + 30% cash; overlay leaves it.
    assert weights == {"BTC/USDT": pytest.approx(0.70)}


# ---------------------------------------------------------------------------
# extreme_fear: +20pp BTC, capped at 100%
# ---------------------------------------------------------------------------
def test_extreme_fear_boosts_btc_when_cash_available():
    """Base 'defensive' = 30% BTC + 70% cash. Extreme fear adds 20pp BTC
    from cash → 50% BTC + 50% cash."""
    asof, strat = _build("defensive", "extreme_fear", btc_above=True)
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    assert weights == {"BTC/USDT": pytest.approx(0.50)}


def test_extreme_fear_does_not_boost_when_btc_below_200d_ma():
    """If BTC is below its 200d MA, the +20pp boost is suppressed —
    we don't double-down on bleeding equity."""
    asof, strat = _build("defensive", "extreme_fear", btc_above=False)
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    # Base defensive untouched: 30% BTC.
    assert weights == {"BTC/USDT": pytest.approx(0.30)}


def test_extreme_fear_caps_at_100pct():
    """Base 'btc_leadership' = 100% BTC. No cash to boost from → no change."""
    asof, strat = _build("btc_leadership", "extreme_fear", btc_above=True)
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    assert weights == {"BTC/USDT": pytest.approx(1.0)}


# ---------------------------------------------------------------------------
# extreme_greed: -20pp from alts → BTC if base had BTC, else cash
# ---------------------------------------------------------------------------
def test_extreme_greed_cuts_alts_and_moves_to_btc_when_base_has_btc():
    """Base 'alt_risk_on' = 70% alts + 30% BTC. Extreme greed cuts alts
    by 20pp → 50% alts + 50% BTC."""
    asof, strat = _build("alt_risk_on", "extreme_greed")
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    btc = weights.get("BTC/USDT", 0.0)
    alts = sum(v for k, v in weights.items() if k != "BTC/USDT")
    assert btc == pytest.approx(0.50)
    assert alts == pytest.approx(0.50)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_extreme_greed_with_no_alts_passes_through():
    """Base 'btc_leadership' has no alts to cut."""
    asof, strat = _build("btc_leadership", "extreme_greed")
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    assert weights == {"BTC/USDT": pytest.approx(1.0)}


# ---------------------------------------------------------------------------
# deteriorating: -20pp risky, alts first, then BTC, into cash
# ---------------------------------------------------------------------------
def test_deteriorating_cuts_alts_first_then_btc_into_cash():
    """Base 'alt_risk_on' = 70% alts + 30% BTC. Deteriorating cuts 20pp
    from alts → 50% alts + 30% BTC + 20% cash."""
    asof, strat = _build("alt_risk_on", "deteriorating")
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    btc = weights.get("BTC/USDT", 0.0)
    alts = sum(v for k, v in weights.items() if k != "BTC/USDT")
    assert btc == pytest.approx(0.30)
    assert alts == pytest.approx(0.50)
    cash = 1.0 - sum(weights.values())
    assert cash == pytest.approx(0.20)


def test_deteriorating_falls_through_to_btc_when_no_alts():
    """Base 'btc_leadership' = 100% BTC. Cut 20pp BTC → 80% BTC + 20% cash."""
    asof, strat = _build("btc_leadership", "deteriorating")
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    assert weights == {"BTC/USDT": pytest.approx(0.80)}


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------
def test_total_weights_never_exceed_100pct_in_any_state_combo():
    asof = 1_700_000_000_000
    universe = _universe(asof)
    for ms_state in ("alt_risk_on", "btc_leadership", "neutral",
                       "defensive", "unknown"):
        for sent_state in ("extreme_fear", "fear_recovery", "neutral",
                             "extreme_greed", "deteriorating", "unknown"):
            for btc_above in (True, False):
                ms = pd.DataFrame([_ms_sig_row(asof - 86_400_000, ms_state,
                                                  btc_above_200d_ma=btc_above)])
                sent = pd.DataFrame([_sent_row(asof - 86_400_000, sent_state)])
                strat = SentimentMarketStructureAllocatorStrategy(ms, sent)
                w = strat.target_weights(asof, universe, timeframe="1d")
                total = sum(w.values())
                assert total <= 1.0 + 1e-9, (ms_state, sent_state, total)
                for v in w.values():
                    assert 0.0 <= v <= 1.0


def test_future_sentiment_row_is_ignored():
    asof = 1_700_000_000_000
    ms = pd.DataFrame([_ms_sig_row(asof - 86_400_000, "alt_risk_on")])
    # Future sentiment row would say extreme_greed → cut alts.
    sent = pd.DataFrame([
        _sent_row(asof - 86_400_000, "neutral"),
        _sent_row(asof + 86_400_000, "extreme_greed"),
    ])
    strat = SentimentMarketStructureAllocatorStrategy(ms, sent)
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    btc = weights.get("BTC/USDT", 0.0)
    alts = sum(v for k, v in weights.items() if k != "BTC/USDT")
    # If the future row leaked: alts would be 0.50 not 0.70.
    assert alts == pytest.approx(0.70)
    assert btc == pytest.approx(0.30)


def test_empty_sentiment_passes_base_through():
    asof = 1_700_000_000_000
    ms = pd.DataFrame([_ms_sig_row(asof - 86_400_000, "alt_risk_on")])
    strat = SentimentMarketStructureAllocatorStrategy(ms, pd.DataFrame())
    weights = strat.target_weights(asof, _universe(asof), timeframe="1d")
    btc = weights.get("BTC/USDT", 0.0)
    alts = sum(v for k, v in weights.items() if k != "BTC/USDT")
    assert btc == pytest.approx(0.30)
    assert alts == pytest.approx(0.70)
