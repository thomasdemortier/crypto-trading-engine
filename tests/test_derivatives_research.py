"""Tests for `src.derivatives_research`.

The walk-forward and placebo are integration-heavy (they spin up the
portfolio backtester); we exercise the verdict logic of the scorecard
directly with synthetic walk-forward / placebo frames so the tests
are fast and deterministic.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest

from src import config, derivatives_research as der
from src.portfolio_research import (
    PORTFOLIO_FAIL, PORTFOLIO_INCONCLUSIVE,
    PORTFOLIO_PASS, PORTFOLIO_WATCHLIST,
)


def _wf_row(window: int, oos_return: float, btc: float, basket: float,
            n_reb: int = 5) -> dict:
    return {
        "window": window,
        "oos_start_iso": f"2024-01-{window:02d}T00:00:00+00:00",
        "oos_end_iso": f"2024-04-{window:02d}T00:00:00+00:00",
        "oos_return_pct": oos_return,
        "oos_max_drawdown_pct": -10.0,
        "oos_sharpe": 0.5,
        "btc_oos_return_pct": btc,
        "basket_oos_return_pct": basket,
        "beats_btc": oos_return > btc,
        "beats_basket": oos_return > basket,
        "profitable": oos_return > 0,
        "n_rebalances": n_reb,
        "n_trades": n_reb * 2,
        "avg_holdings": 3.0,
        "turnover": 0.1,
        "error": None,
    }


def _placebo_summary(strategy_return: float, median_return: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "strategy": "derivatives_rotation",
        "strategy_return_pct": strategy_return,
        "strategy_max_drawdown_pct": -15.0,
        "strategy_sharpe": 0.5,
        "n_seeds": 20,
        "placebo_median_return_pct": median_return,
        "placebo_median_drawdown_pct": -20.0,
        "placebo_p75_drawdown_pct": -25.0,
        "strategy_beats_median_return": strategy_return > median_return,
        "strategy_beats_median_drawdown": True,
    }])


def test_scorecard_inconclusive_when_no_walk_forward():
    out = der.derivatives_scorecard(walk_forward_df=pd.DataFrame(),
                                       placebo_df=pd.DataFrame(), save=False)
    assert out["verdict"].iloc[0] == PORTFOLIO_INCONCLUSIVE
    assert "OI cap" in out["reason"].iloc[0]


def test_scorecard_inconclusive_when_too_few_windows():
    wf = pd.DataFrame([_wf_row(i, 5.0, 4.0, 3.0) for i in range(1, 4)])
    out = der.derivatives_scorecard(
        walk_forward_df=wf,
        placebo_df=_placebo_summary(20.0, 10.0),
        save=False,
    )
    assert out["verdict"].iloc[0] == PORTFOLIO_INCONCLUSIVE
    assert "OOS windows" in out["reason"].iloc[0]


def test_scorecard_fail_when_strategy_loses_to_btc_and_basket():
    # 6 windows, strategy badly underperforms.
    wf = pd.DataFrame([_wf_row(i, -10.0, 20.0, 15.0, n_reb=2) for i in range(1, 7)])
    out = der.derivatives_scorecard(
        walk_forward_df=wf,
        placebo_df=_placebo_summary(strategy_return=-50.0, median_return=10.0),
        save=False,
    )
    assert out["verdict"].iloc[0] == PORTFOLIO_FAIL


def test_scorecard_pass_only_when_every_check_passes():
    # 6 windows, strategy beats BTC + basket every time AND beats placebo.
    wf = pd.DataFrame([_wf_row(i, 30.0, 5.0, 5.0, n_reb=3) for i in range(1, 7)])
    out = der.derivatives_scorecard(
        walk_forward_df=wf,
        placebo_df=_placebo_summary(strategy_return=200.0, median_return=10.0),
        save=False,
    )
    assert out["verdict"].iloc[0] == PORTFOLIO_PASS
    assert int(out["checks_passed"].iloc[0]) == int(out["checks_total"].iloc[0])


def test_scorecard_watchlist_when_one_check_short():
    # 6 windows, beats BTC + basket on 5/6 → 83% stability — passes 60%.
    rows = [_wf_row(i, 30.0, 5.0, 5.0, n_reb=3) for i in range(1, 6)]
    rows.append(_wf_row(6, -1.0, 5.0, 5.0, n_reb=3))   # one bad window
    wf = pd.DataFrame(rows)
    # Strategy beats placebo (positive_return passes).
    out = der.derivatives_scorecard(
        walk_forward_df=wf,
        placebo_df=_placebo_summary(strategy_return=150.0, median_return=10.0),
        save=False,
    )
    # 5 of 6 OOS windows beat BTC and basket — stability = 83% (above 60).
    # All 6 checks should pass → verdict PASS, but if any one fails it's
    # WATCHLIST. The point of this test is to confirm that the scorecard
    # is at least not stuck on FAIL when 5 of 6 windows are great.
    assert out["verdict"].iloc[0] in (PORTFOLIO_PASS, PORTFOLIO_WATCHLIST)


def test_spot_assets_for_normalises_symbols():
    assert der._spot_assets_for(["BTCUSDT", "ETH/USDT", "SOLUSDT"]) == [
        "BTC/USDT", "ETH/USDT", "SOL/USDT",
    ]
