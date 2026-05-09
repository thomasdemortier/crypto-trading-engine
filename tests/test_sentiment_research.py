"""Tests for `src.sentiment_research` — placebo + scorecard logic."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src import config, sentiment_research as sr
from src.portfolio_research import (
    PORTFOLIO_FAIL, PORTFOLIO_INCONCLUSIVE,
    PORTFOLIO_PASS, PORTFOLIO_WATCHLIST,
)


# ---------------------------------------------------------------------------
# Random sentiment overlay placebo
# ---------------------------------------------------------------------------
def _make_universe(asof_ts: int, days: int = 250) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(0)
    out = {}
    for i, sym in enumerate(["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT",
                             "LINK/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT",
                             "LTC/USDT", "BNB/USDT"]):
        rets = rng.normal(loc=0.001, scale=0.005, size=days)
        closes = 100.0 * np.cumprod(1.0 + rets)
        ts = (np.arange(days, dtype="int64") * 86_400_000
               + asof_ts - (days - 1) * 86_400_000)
        out[sym] = pd.DataFrame({"timestamp": ts, "close": closes})
    return out


def _ms_row(ts_ms: int, state: str = "neutral") -> dict:
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


def _sent_frame(states):
    rows = []
    for i, s in enumerate(states):
        rows.append({"timestamp": 1_700_000_000_000 + i * 86_400_000,
                      "date": "2024-01-01",
                      "fear_greed_value": 50,
                      "fear_greed_classification": "Neutral",
                      "fg_7d_change": 0.0, "fg_30d_change": 0.0,
                      "fg_7d_mean": 50.0, "fg_30d_mean": 50.0,
                      "fg_90d_zscore": 0.0,
                      "extreme_fear": False, "fear": False,
                      "neutral": True, "greed": False,
                      "extreme_greed": False,
                      "sentiment_recovering": False,
                      "sentiment_deteriorating": False,
                      "sentiment_state": s})
    return pd.DataFrame(rows)


def test_random_overlay_placebo_is_reproducible():
    asof = 1_700_000_000_000
    universe = _make_universe(asof)
    ms = pd.DataFrame([_ms_row(asof - 86_400_000, "alt_risk_on")])
    sent = _sent_frame(["neutral"] * 50 + ["extreme_fear"] * 20
                          + ["extreme_greed"] * 20 + ["deteriorating"] * 10)
    plac1 = sr.SentimentRandomOverlayPlacebo(
        ms, sent, sr.SentimentRandomOverlayPlaceboConfig(seed=11))
    plac2 = sr.SentimentRandomOverlayPlacebo(
        ms, sent, sr.SentimentRandomOverlayPlaceboConfig(seed=11))
    weights1 = [plac1.target_weights(asof + i * 86_400_000, universe)
                 for i in range(20)]
    weights2 = [plac2.target_weights(asof + i * 86_400_000, universe)
                 for i in range(20)]
    assert weights1 == weights2


def test_placebo_uses_empirical_state_distribution():
    """The placebo must read empirical state frequencies from the
    REAL sentiment frame — not consult per-bar real states."""
    asof = 1_700_000_000_000
    universe = _make_universe(asof)
    ms = pd.DataFrame([_ms_row(asof - 86_400_000, "alt_risk_on")])
    # 100% extreme_greed empirical → placebo should always sample greed.
    sent = _sent_frame(["extreme_greed"] * 200)
    plac = sr.SentimentRandomOverlayPlacebo(
        ms, sent, sr.SentimentRandomOverlayPlaceboConfig(seed=42))
    # Pick 30 rebalance bars; verify the alt-cut effect is consistent.
    seen = []
    for i in range(30):
        w = plac.target_weights(asof + i * 86_400_000, universe)
        # extreme_greed cuts alts by 20pp on a 70/30 base → 50% alts.
        alts = sum(v for k, v in w.items() if k != "BTC/USDT")
        seen.append(alts)
    # Most or all should reflect the cut; sampling is deterministic so
    # the empirical-only state forces the same overlay every call.
    assert all(abs(a - 0.50) < 1e-6 for a in seen), seen


# ---------------------------------------------------------------------------
# Scorecard logic (synthetic walk-forward + placebo + comparison)
# ---------------------------------------------------------------------------
def _wf_row(window, oos_return, btc, basket, simple, vt, n_reb=5):
    return {
        "window": window,
        "oos_start_iso": f"2024-01-{window:02d}T00:00:00+00:00",
        "oos_end_iso": f"2024-04-{window:02d}T00:00:00+00:00",
        "oos_return_pct": oos_return,
        "oos_max_drawdown_pct": -10.0, "oos_sharpe": 0.5,
        "btc_oos_return_pct": btc,
        "basket_oos_return_pct": basket,
        "simple_oos_return_pct": simple,
        "vol_target_oos_return_pct": vt,
        "beats_btc": oos_return > btc,
        "beats_basket": oos_return > basket,
        "beats_simple_momentum": oos_return > simple,
        "beats_vol_target": oos_return > vt,
        "profitable": oos_return > 0,
        "n_rebalances": n_reb,
        "n_trades": n_reb * 2,
        "avg_holdings": 3.0,
        "turnover": 0.1,
        "error": None,
    }


def _placebo_summary(strategy_return, median_return, strategy_dd=-25.0):
    return pd.DataFrame([{
        "strategy": "sentiment_market_structure_allocator",
        "strategy_return_pct": strategy_return,
        "strategy_max_drawdown_pct": strategy_dd,
        "strategy_sharpe": 0.5,
        "n_seeds": 20,
        "placebo_median_return_pct": median_return,
        "placebo_median_drawdown_pct": -40.0,
        "placebo_p75_drawdown_pct": -50.0,
        "strategy_beats_median_return": strategy_return > median_return,
        "strategy_beats_median_drawdown": True,
    }])


def _comparison_with_btc_dd(btc_dd=-50.0):
    return pd.DataFrame([
        {"strategy": "sentiment_market_structure_allocator",
         "max_drawdown_pct": -30.0, "total_return_pct": 100.0},
        {"strategy": "BTC_buy_and_hold",
         "max_drawdown_pct": btc_dd, "total_return_pct": 175.0},
        {"strategy": "equal_weight_basket",
         "max_drawdown_pct": -60.0, "total_return_pct": 60.0},
    ])


def _coverage_ok():
    return pd.DataFrame([{
        "source": "alternative.me", "dataset": "fear_greed_index_daily",
        "actual_start": "2018-02-01", "actual_end": "2026-04-01",
        "row_count": 3000, "coverage_days": 3000.0,
        "enough_for_research": True, "largest_gap_days": 1.0, "notes": "ok",
    }])


def _coverage_short():
    return pd.DataFrame([{
        "source": "alternative.me", "dataset": "fear_greed_index_daily",
        "actual_start": "2024-01-01", "actual_end": "2024-04-01",
        "row_count": 90, "coverage_days": 90.0,
        "enough_for_research": False, "largest_gap_days": 1.0,
        "notes": "below_4yr_threshold",
    }])


@pytest.fixture(autouse=True)
def _redirect_results(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    yield


def test_scorecard_inconclusive_when_no_walk_forward(tmp_path):
    out = sr.sentiment_scorecard(walk_forward_df=pd.DataFrame(),
                                    placebo_df=pd.DataFrame(), save=False)
    assert out["verdict"].iloc[0] == PORTFOLIO_INCONCLUSIVE


def test_scorecard_inconclusive_when_sentiment_coverage_short(tmp_path):
    """Even if benchmark + placebo metrics look great, missing sentiment
    coverage drives INCONCLUSIVE — not PASS."""
    _coverage_short().to_csv(
        config.RESULTS_DIR / "sentiment_data_coverage.csv", index=False)
    _comparison_with_btc_dd().to_csv(
        config.RESULTS_DIR / "sentiment_allocator_comparison.csv",
        index=False)
    wf = pd.DataFrame([_wf_row(i, 30.0, 5.0, 5.0, 5.0, 5.0, n_reb=3)
                        for i in range(1, 7)])
    out = sr.sentiment_scorecard(
        walk_forward_df=wf,
        placebo_df=_placebo_summary(strategy_return=200.0,
                                       median_return=10.0,
                                       strategy_dd=-30.0),
        save=False,
    )
    assert out["verdict"].iloc[0] == PORTFOLIO_INCONCLUSIVE


def test_scorecard_weak_strategy_is_fail(tmp_path):
    _coverage_ok().to_csv(
        config.RESULTS_DIR / "sentiment_data_coverage.csv", index=False)
    _comparison_with_btc_dd().to_csv(
        config.RESULTS_DIR / "sentiment_allocator_comparison.csv",
        index=False)
    wf = pd.DataFrame([_wf_row(i, -10.0, 20.0, 15.0, 10.0, 0.0, n_reb=2)
                        for i in range(1, 7)])
    out = sr.sentiment_scorecard(
        walk_forward_df=wf,
        placebo_df=_placebo_summary(strategy_return=-50.0,
                                       median_return=10.0),
        save=False,
    )
    assert out["verdict"].iloc[0] == PORTFOLIO_FAIL


def test_scorecard_beating_placebo_only_is_not_pass(tmp_path):
    _coverage_ok().to_csv(
        config.RESULTS_DIR / "sentiment_data_coverage.csv", index=False)
    _comparison_with_btc_dd().to_csv(
        config.RESULTS_DIR / "sentiment_allocator_comparison.csv",
        index=False)
    # Strategy beats placebo (25 vs 10) but loses to BTC, basket, simple,
    # vol-target in every window.
    wf = pd.DataFrame([_wf_row(i, 5.0, 30.0, 25.0, 20.0, 15.0, n_reb=3)
                        for i in range(1, 7)])
    out = sr.sentiment_scorecard(
        walk_forward_df=wf,
        placebo_df=_placebo_summary(strategy_return=25.0,
                                       median_return=10.0,
                                       strategy_dd=-30.0),
        save=False,
    )
    assert out["verdict"].iloc[0] not in (PORTFOLIO_PASS, PORTFOLIO_WATCHLIST)


def test_scorecard_pass_requires_every_check(tmp_path):
    _coverage_ok().to_csv(
        config.RESULTS_DIR / "sentiment_data_coverage.csv", index=False)
    _comparison_with_btc_dd().to_csv(
        config.RESULTS_DIR / "sentiment_allocator_comparison.csv",
        index=False)
    wf = pd.DataFrame([_wf_row(i, 30.0, 5.0, 5.0, 5.0, 5.0, n_reb=3)
                        for i in range(1, 7)])
    out = sr.sentiment_scorecard(
        walk_forward_df=wf,
        placebo_df=_placebo_summary(strategy_return=200.0,
                                       median_return=10.0,
                                       strategy_dd=-30.0),
        save=False,
    )
    assert out["verdict"].iloc[0] == PORTFOLIO_PASS
    assert int(out["checks_passed"].iloc[0]) == 10
    assert int(out["checks_total"].iloc[0]) == 10
