"""Scorecard logic tests for `src.funding_research`.

The scorecard adds two checks the v1 portfolio scorecard didn't have:
  * `beats_simple_momentum_oos` (>50 % of windows)
  * `dd_within_btc_gap_20pp` (strategy DD - BTC DD ≥ -20 pp)

These tests verify each check individually with synthetic walk-forward +
placebo + comparison frames, and verify that beating-placebo-only does
NOT yield a PASS.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest

from src import config, funding_research as fr
from src.portfolio_research import (
    PORTFOLIO_FAIL, PORTFOLIO_INCONCLUSIVE,
    PORTFOLIO_PASS, PORTFOLIO_WATCHLIST,
)


def _wf_row(window: int, oos_return: float, btc: float, basket: float,
             simple: float, n_reb: int = 5) -> dict:
    return {
        "window": window,
        "oos_start_iso": f"2024-01-{window:02d}T00:00:00+00:00",
        "oos_end_iso": f"2024-04-{window:02d}T00:00:00+00:00",
        "oos_return_pct": oos_return,
        "oos_max_drawdown_pct": -10.0,
        "oos_sharpe": 0.5,
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


def _placebo_summary(strategy_return: float, median_return: float,
                      strategy_dd: float = -25.0) -> pd.DataFrame:
    return pd.DataFrame([{
        "strategy": "funding_rotation",
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


def _comparison_with_btc_dd(btc_dd: float = -50.0) -> pd.DataFrame:
    return pd.DataFrame([
        {"strategy": "funding_rotation", "max_drawdown_pct": -30.0,
         "total_return_pct": 100.0},
        {"strategy": "BTC_buy_and_hold", "max_drawdown_pct": btc_dd,
         "total_return_pct": 175.0},
        {"strategy": "equal_weight_basket", "max_drawdown_pct": -60.0,
         "total_return_pct": 60.0},
    ])


@pytest.fixture(autouse=True)
def _redirect_results(tmp_path, monkeypatch):
    """Persist scorecard outputs to a tmp dir so tests don't write to
    the repo's `results/`."""
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    yield


# ---------------------------------------------------------------------------
def test_inconclusive_when_no_walk_forward(tmp_path):
    out = fr.funding_scorecard(walk_forward_df=pd.DataFrame(),
                                  placebo_df=pd.DataFrame(), save=False)
    assert out["verdict"].iloc[0] == PORTFOLIO_INCONCLUSIVE


def test_inconclusive_when_too_few_windows(tmp_path):
    wf = pd.DataFrame([_wf_row(i, 5.0, 4.0, 3.0, 2.0) for i in range(1, 4)])
    out = fr.funding_scorecard(
        walk_forward_df=wf,
        placebo_df=_placebo_summary(20.0, 10.0),
        save=False,
    )
    assert out["verdict"].iloc[0] == PORTFOLIO_INCONCLUSIVE


def test_fail_when_loses_to_btc_basket_and_placebo(tmp_path):
    wf = pd.DataFrame([_wf_row(i, -10.0, 20.0, 15.0, 10.0, n_reb=2)
                        for i in range(1, 7)])
    out = fr.funding_scorecard(
        walk_forward_df=wf,
        placebo_df=_placebo_summary(strategy_return=-50.0, median_return=10.0),
        save=False,
    )
    assert out["verdict"].iloc[0] == PORTFOLIO_FAIL


def test_pass_requires_every_check_including_dd_gap(tmp_path):
    """A run that beats every benchmark + placebo + has tight DD passes."""
    # Persist the comparison frame the scorecard reads from disk.
    cmp_p = config.RESULTS_DIR / "funding_rotation_comparison.csv"
    _comparison_with_btc_dd(btc_dd=-50.0).to_csv(cmp_p, index=False)
    wf = pd.DataFrame([_wf_row(i, 30.0, 5.0, 5.0, 5.0, n_reb=3)
                        for i in range(1, 7)])
    out = fr.funding_scorecard(
        walk_forward_df=wf,
        placebo_df=_placebo_summary(strategy_return=200.0,
                                       median_return=10.0,
                                       strategy_dd=-30.0),
        save=False,
    )
    # strategy_dd (-30) - btc_dd (-50) = 20 >= -20 ✓
    assert out["verdict"].iloc[0] == PORTFOLIO_PASS
    assert int(out["checks_passed"].iloc[0]) == 8


def test_dd_gap_check_blocks_pass_when_dd_too_wide(tmp_path):
    """Strategy beats every benchmark BUT its DD is 50 pp worse than
    BTC's → must NOT be PASS."""
    cmp_p = config.RESULTS_DIR / "funding_rotation_comparison.csv"
    _comparison_with_btc_dd(btc_dd=-50.0).to_csv(cmp_p, index=False)
    wf = pd.DataFrame([_wf_row(i, 30.0, 5.0, 5.0, 5.0, n_reb=3)
                        for i in range(1, 7)])
    out = fr.funding_scorecard(
        walk_forward_df=wf,
        placebo_df=_placebo_summary(strategy_return=200.0,
                                       median_return=10.0,
                                       strategy_dd=-95.0),
        save=False,
    )
    # strategy_dd (-95) - btc_dd (-50) = -45 < -20 → DD gap check fails.
    assert out["verdict"].iloc[0] != PORTFOLIO_PASS


def test_beating_placebo_only_is_not_pass(tmp_path):
    """Strategy beats placebo but loses to BTC, basket, AND simple
    momentum in OOS — must NOT be PASS or WATCHLIST."""
    cmp_p = config.RESULTS_DIR / "funding_rotation_comparison.csv"
    _comparison_with_btc_dd(btc_dd=-50.0).to_csv(cmp_p, index=False)
    wf = pd.DataFrame([_wf_row(i, 5.0, 30.0, 25.0, 20.0, n_reb=3)
                        for i in range(1, 7)])
    out = fr.funding_scorecard(
        walk_forward_df=wf,
        placebo_df=_placebo_summary(strategy_return=20.0,
                                       median_return=10.0,
                                       strategy_dd=-30.0),
        save=False,
    )
    assert out["verdict"].iloc[0] not in (PORTFOLIO_PASS, PORTFOLIO_WATCHLIST)


def test_spot_assets_for_normalises_symbols():
    assert fr._spot_assets_for(["BTCUSDT", "ETH/USDT"]) == [
        "BTC/USDT", "ETH/USDT",
    ]
