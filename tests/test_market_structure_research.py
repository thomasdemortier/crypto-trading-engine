"""Tests for `src.market_structure_research` — walk-forward, placebo,
and scorecard logic. Walk-forward window construction and benchmark
alignment use the existing `_build_oos_windows` from `portfolio_research`
(already covered in v1 tests). Here we verify:

  * Walk-forward windows are non-overlapping and chronological.
  * The placebo is deterministic given a fixed seed.
  * The placebo never reads market-structure signals.
  * Beating placebo only is NOT enough for PASS.
  * Weak strategy (< 30% beats benchmarks) is FAIL.
  * Insufficient coverage drives INCONCLUSIVE.
  * Missing data does not crash the scorecard.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from src import config, market_structure_research as msr
from src.portfolio_research import (
    PORTFOLIO_FAIL, PORTFOLIO_INCONCLUSIVE,
    PORTFOLIO_PASS, PORTFOLIO_WATCHLIST,
    _build_oos_windows,
)


# ---------------------------------------------------------------------------
# Walk-forward window mechanics
# ---------------------------------------------------------------------------
def test_oos_windows_are_chronological_and_non_overlapping():
    """Sanity-check that the OOS windows the research module uses are
    chronologically ordered and don't have overlapping OOS periods."""
    day = 86_400_000
    first = pd.Timestamp("2020-01-01", tz="UTC").value // 10**6
    last = first + 1500 * day
    windows = _build_oos_windows(first, last,
                                  in_sample_days=180, oos_days=90, step_days=90)
    assert len(windows) >= 5
    for w in windows:
        assert w["oos_start_ms"] > w["is_end_ms"]
    # Every successive OOS window starts after the previous one starts.
    for prev, cur in zip(windows[:-1], windows[1:]):
        assert cur["oos_start_ms"] > prev["oos_start_ms"]


# ---------------------------------------------------------------------------
# Placebo invariants
# ---------------------------------------------------------------------------
def _make_universe(asof_ts: int, days: int = 200) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(0)
    out = {}
    for i, sym in enumerate(["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT",
                             "LINK/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT",
                             "LTC/USDT", "BNB/USDT"]):
        rets = rng.normal(loc=0.001, scale=0.005, size=days)
        closes = 100.0 * np.cumprod(1.0 + rets)
        ts = np.arange(days, dtype="int64") * 86_400_000 + asof_ts - (days - 1) * 86_400_000
        out[sym] = pd.DataFrame({"timestamp": ts, "close": closes})
    return out


def test_placebo_is_reproducible():
    asof = 1_700_000_000_000
    universe = _make_universe(asof, days=200)
    plac1 = msr.MarketStructureStatePlacebo(
        msr.MarketStructureStatePlaceboConfig(seed=7))
    plac2 = msr.MarketStructureStatePlacebo(
        msr.MarketStructureStatePlaceboConfig(seed=7))
    weights1 = [plac1.target_weights(asof + i * 86_400_000, universe)
                 for i in range(20)]
    weights2 = [plac2.target_weights(asof + i * 86_400_000, universe)
                 for i in range(20)]
    assert weights1 == weights2


def test_placebo_does_not_read_market_structure_signals():
    """The placebo class must not accept or consult a signals frame."""
    plac = msr.MarketStructureStatePlacebo()
    # No `signals_df` ctor argument — pure random allocator.
    assert not hasattr(plac, "_signals")
    # And its output depends only on the seeded RNG + asset_frames keys.


def test_placebo_uses_same_alt_basket_size():
    """When the placebo picks the alt state, it allocates exactly
    `alt_top_n` symbols (matching the strategy's basket size)."""
    asof = 1_700_000_000_000
    universe = _make_universe(asof, days=200)
    # Try many seeds until we observe an "alt" pick.
    for seed in range(50):
        plac = msr.MarketStructureStatePlacebo(
            msr.MarketStructureStatePlaceboConfig(seed=seed, alt_top_n=5))
        weights = plac.target_weights(asof, universe)
        if len(weights) == 5:
            for w in weights.values():
                assert w == pytest.approx(0.2)
            return
    pytest.skip("placebo never picked alt state in 50 seeds (very unlikely)")


# ---------------------------------------------------------------------------
# Scorecard verdicts
# ---------------------------------------------------------------------------
def _wf_row(window, oos_return, btc, basket, simple, n_reb=5):
    return {
        "window": window,
        "oos_start_iso": f"2024-01-{window:02d}T00:00:00+00:00",
        "oos_end_iso": f"2024-04-{window:02d}T00:00:00+00:00",
        "oos_return_pct": oos_return,
        "oos_max_drawdown_pct": -10.0, "oos_sharpe": 0.5,
        "btc_oos_return_pct": btc,
        "basket_oos_return_pct": basket,
        "simple_oos_return_pct": simple,
        "beats_btc": oos_return > btc,
        "beats_basket": oos_return > basket,
        "beats_simple_momentum": oos_return > simple,
        "profitable": oos_return > 0,
        "n_rebalances": n_reb,
        "n_trades": n_reb * 2,
        "avg_holdings": 3.0,
        "turnover": 0.1,
        "error": None,
    }


def _placebo_summary(strategy_return, median_return, strategy_dd=-25.0):
    return pd.DataFrame([{
        "strategy": "market_structure_allocator",
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
        {"strategy": "market_structure_allocator",
         "max_drawdown_pct": -30.0, "total_return_pct": 100.0},
        {"strategy": "BTC_buy_and_hold",
         "max_drawdown_pct": btc_dd, "total_return_pct": 175.0},
        {"strategy": "equal_weight_basket",
         "max_drawdown_pct": -60.0, "total_return_pct": 60.0},
    ])


def _coverage_all_ok():
    return pd.DataFrame([
        {"source": "defillama", "dataset": "total_tvl_all_chains",
         "actual_start": "2017-09-01", "actual_end": "2026-04-01",
         "row_count": 3000, "coverage_days": 3000.0,
         "enough_for_research": True, "largest_gap_days": 1.0,
         "missing_reason": "", "notes": "ok"},
        {"source": "defillama", "dataset": "stablecoin_supply_total",
         "actual_start": "2017-11-01", "actual_end": "2026-04-01",
         "row_count": 3000, "coverage_days": 3000.0,
         "enough_for_research": True, "largest_gap_days": 1.0,
         "missing_reason": "", "notes": "ok"},
    ])


def _coverage_with_gap():
    return pd.DataFrame([
        {"source": "defillama", "dataset": "total_tvl_all_chains",
         "actual_start": "2024-01-01", "actual_end": "2024-04-01",
         "row_count": 90, "coverage_days": 90.0,
         "enough_for_research": False, "largest_gap_days": 1.0,
         "missing_reason": "below_4yr_threshold", "notes": ""},
    ])


@pytest.fixture(autouse=True)
def _redirect_results(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    yield


def test_scorecard_inconclusive_when_no_walk_forward(tmp_path):
    out = msr.market_structure_scorecard(walk_forward_df=pd.DataFrame(),
                                            placebo_df=pd.DataFrame(),
                                            save=False)
    assert out["verdict"].iloc[0] == PORTFOLIO_INCONCLUSIVE


def test_scorecard_weak_strategy_is_fail(tmp_path):
    # All coverage ok, but strategy badly underperforms benchmarks AND placebo.
    _coverage_all_ok().to_csv(
        config.RESULTS_DIR / "market_structure_data_coverage.csv",
        index=False)
    _comparison_with_btc_dd().to_csv(
        config.RESULTS_DIR / "market_structure_allocator_comparison.csv",
        index=False)
    wf = pd.DataFrame([_wf_row(i, -10.0, 20.0, 15.0, 10.0, n_reb=2)
                        for i in range(1, 7)])
    out = msr.market_structure_scorecard(
        walk_forward_df=wf,
        placebo_df=_placebo_summary(strategy_return=-50.0,
                                       median_return=10.0),
        save=False,
    )
    assert out["verdict"].iloc[0] == PORTFOLIO_FAIL


def test_scorecard_inconclusive_when_coverage_missing(tmp_path):
    """Even if benchmark+placebo metrics look great, missing
    market-structure coverage drives INCONCLUSIVE — not PASS."""
    _coverage_with_gap().to_csv(
        config.RESULTS_DIR / "market_structure_data_coverage.csv",
        index=False)
    _comparison_with_btc_dd().to_csv(
        config.RESULTS_DIR / "market_structure_allocator_comparison.csv",
        index=False)
    wf = pd.DataFrame([_wf_row(i, 30.0, 5.0, 5.0, 5.0, n_reb=3)
                        for i in range(1, 7)])
    out = msr.market_structure_scorecard(
        walk_forward_df=wf,
        placebo_df=_placebo_summary(strategy_return=200.0,
                                       median_return=10.0,
                                       strategy_dd=-30.0),
        save=False,
    )
    assert out["verdict"].iloc[0] == PORTFOLIO_INCONCLUSIVE


def test_beating_placebo_only_is_not_pass(tmp_path):
    _coverage_all_ok().to_csv(
        config.RESULTS_DIR / "market_structure_data_coverage.csv",
        index=False)
    _comparison_with_btc_dd().to_csv(
        config.RESULTS_DIR / "market_structure_allocator_comparison.csv",
        index=False)
    # Strategy beats placebo (5% > 0% median) but loses to BTC, basket,
    # and simple momentum in every window.
    wf = pd.DataFrame([_wf_row(i, 5.0, 30.0, 25.0, 20.0, n_reb=3)
                        for i in range(1, 7)])
    out = msr.market_structure_scorecard(
        walk_forward_df=wf,
        placebo_df=_placebo_summary(strategy_return=20.0, median_return=10.0,
                                       strategy_dd=-30.0),
        save=False,
    )
    assert out["verdict"].iloc[0] not in (PORTFOLIO_PASS, PORTFOLIO_WATCHLIST)


def test_pass_requires_every_check(tmp_path):
    _coverage_all_ok().to_csv(
        config.RESULTS_DIR / "market_structure_data_coverage.csv",
        index=False)
    _comparison_with_btc_dd().to_csv(
        config.RESULTS_DIR / "market_structure_allocator_comparison.csv",
        index=False)
    wf = pd.DataFrame([_wf_row(i, 30.0, 5.0, 5.0, 5.0, n_reb=3)
                        for i in range(1, 7)])
    out = msr.market_structure_scorecard(
        walk_forward_df=wf,
        placebo_df=_placebo_summary(strategy_return=200.0,
                                       median_return=10.0,
                                       strategy_dd=-30.0),
        save=False,
    )
    assert out["verdict"].iloc[0] == PORTFOLIO_PASS
    assert int(out["checks_passed"].iloc[0]) == 9
    assert int(out["checks_total"].iloc[0]) == 9


def test_scorecard_handles_missing_csvs_safely(tmp_path):
    """No CSVs anywhere — must produce one well-formed INCONCLUSIVE row,
    not raise."""
    out = msr.market_structure_scorecard(walk_forward_df=None,
                                            placebo_df=None, save=False)
    assert len(out) == 1
    assert out["verdict"].iloc[0] == PORTFOLIO_INCONCLUSIVE
