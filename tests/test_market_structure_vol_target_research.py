"""Tests for the vol-target additions to `src.market_structure_research`."""
from __future__ import annotations

from typing import Dict

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
# Walk-forward window mechanics (sanity)
# ---------------------------------------------------------------------------
def test_vol_target_oos_windows_chronological():
    day = 86_400_000
    first = pd.Timestamp("2020-01-01", tz="UTC").value // 10**6
    last = first + 1500 * day
    windows = _build_oos_windows(first, last,
                                  in_sample_days=180, oos_days=90, step_days=90)
    assert len(windows) >= 5
    for prev, cur in zip(windows[:-1], windows[1:]):
        assert cur["oos_start_ms"] > prev["oos_start_ms"]


# ---------------------------------------------------------------------------
# Vol-state placebo invariants
# ---------------------------------------------------------------------------
def _make_universe(asof_ts: int, days: int = 200) -> Dict[str, pd.DataFrame]:
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


def test_vol_state_placebo_is_reproducible():
    asof = 1_700_000_000_000
    universe = _make_universe(asof, days=200)
    plac1 = msr.MarketStructureVolStatePlacebo(
        msr.MarketStructureVolStatePlaceboConfig(seed=11))
    plac2 = msr.MarketStructureVolStatePlacebo(
        msr.MarketStructureVolStatePlaceboConfig(seed=11))
    weights1 = [plac1.target_weights(asof + i * 86_400_000, universe)
                 for i in range(20)]
    weights2 = [plac2.target_weights(asof + i * 86_400_000, universe)
                 for i in range(20)]
    assert weights1 == weights2


def test_vol_state_placebo_does_not_consult_signals():
    """The class never accepts a signals frame; its output is purely a
    function of seed + asset_frames."""
    plac = msr.MarketStructureVolStatePlacebo()
    assert not hasattr(plac, "_signals")


def test_vol_state_placebo_respects_exposure_bands():
    """Across many rebalance bars the placebo must produce one of:
        empty (cash), {BTC: 1.0}, {BTC: 0.7}, {BTC: 0.3},
        or {alts...: total 0.7, BTC: 0.3}.
    Total weight must always be ≤ 1."""
    asof = 1_700_000_000_000
    universe = _make_universe(asof, days=200)
    seen = set()
    for seed in range(40):
        plac = msr.MarketStructureVolStatePlacebo(
            msr.MarketStructureVolStatePlaceboConfig(seed=seed))
        for _ in range(10):
            w = plac.target_weights(asof, universe)
            total = sum(w.values())
            assert total <= 1.0 + 1e-9, total
            for v in w.values():
                assert 0.0 <= v <= 1.0
            # Bucket the call into a band.
            if not w:
                seen.add("cash")
            elif set(w.keys()) == {"BTC/USDT"}:
                btc = round(w["BTC/USDT"], 2)
                if btc == 1.00:
                    seen.add("btc_only")
                elif btc == 0.70:
                    seen.add("partial_btc")
                elif btc == 0.30:
                    seen.add("defensive_partial")
            else:
                # Alt basket state.
                btc = w.get("BTC/USDT", 0.0)
                alts = sum(v for s, v in w.items() if s != "BTC/USDT")
                if abs(btc - 0.30) < 1e-6 and abs(alts - 0.70) < 1e-6:
                    seen.add("alt_basket")
    # We saw at least three of the five bands in 400 calls.
    assert len(seen) >= 3, seen


# ---------------------------------------------------------------------------
# Scorecard verdicts
# ---------------------------------------------------------------------------
def _wf_row(window, oos_return, btc, basket, simple, original, n_reb=5):
    return {
        "window": window,
        "oos_start_iso": f"2024-01-{window:02d}T00:00:00+00:00",
        "oos_end_iso": f"2024-04-{window:02d}T00:00:00+00:00",
        "oos_return_pct": oos_return,
        "oos_max_drawdown_pct": -10.0, "oos_sharpe": 0.5,
        "btc_oos_return_pct": btc,
        "basket_oos_return_pct": basket,
        "simple_oos_return_pct": simple,
        "original_allocator_oos_return_pct": original,
        "beats_btc": oos_return > btc,
        "beats_basket": oos_return > basket,
        "beats_simple_momentum": oos_return > simple,
        "beats_original_allocator": oos_return > original,
        "profitable": oos_return > 0,
        "n_rebalances": n_reb,
        "n_trades": n_reb * 2,
        "avg_holdings": 3.0,
        "turnover": 0.1,
        "error": None,
    }


def _placebo_summary(strategy_return, median_return, strategy_dd=-25.0):
    return pd.DataFrame([{
        "strategy": "market_structure_vol_target_allocator",
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
        {"strategy": "market_structure_vol_target",
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
    ])


@pytest.fixture(autouse=True)
def _redirect_results(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    yield


def test_vol_scorecard_inconclusive_when_no_walk_forward(tmp_path):
    out = msr.market_structure_vol_target_scorecard(
        walk_forward_df=pd.DataFrame(),
        placebo_df=pd.DataFrame(),
        save=False,
    )
    assert out["verdict"].iloc[0] == PORTFOLIO_INCONCLUSIVE


def test_vol_scorecard_weak_strategy_is_fail(tmp_path):
    _coverage_all_ok().to_csv(
        config.RESULTS_DIR / "market_structure_data_coverage.csv",
        index=False,
    )
    _comparison_with_btc_dd().to_csv(
        config.RESULTS_DIR / "market_structure_vol_target_comparison.csv",
        index=False,
    )
    wf = pd.DataFrame([_wf_row(i, -10.0, 20.0, 15.0, 10.0, 0.0, n_reb=2)
                        for i in range(1, 7)])
    out = msr.market_structure_vol_target_scorecard(
        walk_forward_df=wf,
        placebo_df=_placebo_summary(strategy_return=-50.0,
                                       median_return=10.0),
        save=False,
    )
    assert out["verdict"].iloc[0] == PORTFOLIO_FAIL


def test_vol_scorecard_beating_placebo_only_is_not_pass(tmp_path):
    _coverage_all_ok().to_csv(
        config.RESULTS_DIR / "market_structure_data_coverage.csv",
        index=False,
    )
    _comparison_with_btc_dd().to_csv(
        config.RESULTS_DIR / "market_structure_vol_target_comparison.csv",
        index=False,
    )
    # Strategy beats placebo, beats original (slightly), but loses to
    # BTC, basket, and simple momentum in every window.
    wf = pd.DataFrame([_wf_row(i, 5.0, 30.0, 25.0, 20.0, 1.0, n_reb=3)
                        for i in range(1, 7)])
    out = msr.market_structure_vol_target_scorecard(
        walk_forward_df=wf,
        placebo_df=_placebo_summary(strategy_return=20.0,
                                       median_return=10.0,
                                       strategy_dd=-30.0),
        save=False,
    )
    assert out["verdict"].iloc[0] not in (PORTFOLIO_PASS, PORTFOLIO_WATCHLIST)


def test_vol_scorecard_pass_requires_every_check(tmp_path):
    _coverage_all_ok().to_csv(
        config.RESULTS_DIR / "market_structure_data_coverage.csv",
        index=False,
    )
    _comparison_with_btc_dd().to_csv(
        config.RESULTS_DIR / "market_structure_vol_target_comparison.csv",
        index=False,
    )
    wf = pd.DataFrame([_wf_row(i, 30.0, 5.0, 5.0, 5.0, 5.0, n_reb=3)
                        for i in range(1, 7)])
    out = msr.market_structure_vol_target_scorecard(
        walk_forward_df=wf,
        placebo_df=_placebo_summary(strategy_return=200.0,
                                       median_return=10.0,
                                       strategy_dd=-30.0),
        save=False,
    )
    assert out["verdict"].iloc[0] == PORTFOLIO_PASS
    assert int(out["checks_passed"].iloc[0]) == 10
    assert int(out["checks_total"].iloc[0]) == 10


def test_vol_scorecard_inconclusive_when_coverage_missing(tmp_path):
    """Missing coverage drives INCONCLUSIVE even when benchmarks pass."""
    pd.DataFrame([{
        "source": "defillama", "dataset": "total_tvl_all_chains",
        "actual_start": "2024-01-01", "actual_end": "2024-04-01",
        "row_count": 90, "coverage_days": 90.0,
        "enough_for_research": False, "largest_gap_days": 1.0,
        "missing_reason": "below_4yr_threshold", "notes": "",
    }]).to_csv(
        config.RESULTS_DIR / "market_structure_data_coverage.csv",
        index=False,
    )
    _comparison_with_btc_dd().to_csv(
        config.RESULTS_DIR / "market_structure_vol_target_comparison.csv",
        index=False,
    )
    wf = pd.DataFrame([_wf_row(i, 30.0, 5.0, 5.0, 5.0, 5.0, n_reb=3)
                        for i in range(1, 7)])
    out = msr.market_structure_vol_target_scorecard(
        walk_forward_df=wf,
        placebo_df=_placebo_summary(strategy_return=200.0,
                                       median_return=10.0,
                                       strategy_dd=-30.0),
        save=False,
    )
    assert out["verdict"].iloc[0] == PORTFOLIO_INCONCLUSIVE
