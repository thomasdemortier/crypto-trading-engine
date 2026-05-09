"""Strategy scorecard regression tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import scorecard


def _strategy_row(strategy, asset, tf, vs_bh, total_ret, dd, n_trades,
                  exposure=30.0, pf=1.0, sharpe=0.5):
    return {
        "strategy": strategy, "asset": asset, "timeframe": tf,
        "label": strategy,
        "total_return_pct": total_ret,
        "buy_and_hold_return_pct": total_ret - vs_bh,
        "strategy_vs_bh_pct": vs_bh,
        "max_drawdown_pct": dd,
        "win_rate_pct": 50.0,
        "num_trades": n_trades,
        "profit_factor": pf,
        "fees_paid": 1.0,
        "slippage_cost": 0.5,
        "exposure_time_pct": exposure,
        "sharpe_ratio": sharpe,
        "sortino_ratio": 0.5,
        "calmar_ratio": 0.1,
        "starting_capital": 10_000.0,
        "final_portfolio_value": 10_000.0 + total_ret * 100.0,
        "error": None,
    }


def test_scorecard_creates_expected_columns():
    sc = pd.DataFrame([
        _strategy_row("trend_following", "BTC/USDT", "1d", vs_bh=5.0,
                      total_ret=15.0, dd=-3.0, n_trades=12),
    ])
    out = scorecard.build_scorecard(sc, walk_forward_df=None,
                                    robustness_df=None, save=False)
    expected = {
        "strategy_name", "asset", "timeframe", "total_score", "verdict",
        "return_score", "benchmark_score", "drawdown_score",
        "robustness_score", "walk_forward_score", "trade_count_score",
        "notes",
    }
    assert expected.issubset(out.columns)


def test_scorecard_marks_weak_strategy_as_fail():
    """A strategy that underperforms B&H, has 4 trades, and 1% exposure
    must score FAIL or INCONCLUSIVE — never PASS."""
    sc = pd.DataFrame([
        _strategy_row("rsi_ma_atr", "BTC/USDT", "4h", vs_bh=-12.0,
                      total_ret=0.01, dd=-0.05, n_trades=4, exposure=1.2),
    ])
    out = scorecard.build_scorecard(sc, save=False)
    assert out.iloc[0]["verdict"] in ("FAIL", "INCONCLUSIVE")
    assert out.iloc[0]["verdict"] != "PASS"


def test_scorecard_pass_requires_walk_forward_evidence():
    """High benchmark + many trades is NOT enough — without OOS evidence,
    the verdict must stay below PASS."""
    sc = pd.DataFrame([
        _strategy_row("trend_following", "BTC/USDT", "1d", vs_bh=8.0,
                      total_ret=20.0, dd=-5.0, n_trades=30, exposure=60.0),
    ])
    out = scorecard.build_scorecard(sc, walk_forward_df=None,
                                    robustness_df=None, save=False)
    # No WF data -> walk_forward_score is 0 -> verdict can be WATCHLIST but
    # not PASS.
    assert out.iloc[0]["verdict"] != "PASS"


def test_scorecard_pass_when_all_components_strong():
    sc = pd.DataFrame([
        _strategy_row("trend_following", "BTC/USDT", "1d", vs_bh=8.0,
                      total_ret=20.0, dd=-5.0, n_trades=30, exposure=60.0),
    ])
    # Walk-forward: per-strategy keyed (strategy_name column present).
    wf = pd.DataFrame([
        {"strategy_name": "trend_following", "asset": "BTC/USDT",
         "timeframe": "1d", "window": i,
         "strategy_return_pct": 3.0, "strategy_vs_bh_pct": 1.5,
         "num_trades": 4, "max_drawdown_pct": -2.0, "win_rate_pct": 60.0,
         "profit_factor": 1.5, "fees_paid": 0.0, "exposure_time_pct": 50.0,
         "error": None}
        for i in range(7)
    ] + [
        {"strategy_name": "trend_following", "asset": "BTC/USDT",
         "timeframe": "1d", "window": 99,
         "strategy_return_pct": -1.0, "strategy_vs_bh_pct": -2.0,
         "num_trades": 1, "max_drawdown_pct": -1.0, "win_rate_pct": 0.0,
         "profit_factor": 0.0, "fees_paid": 0.0, "exposure_time_pct": 50.0,
         "error": None},
    ])
    rb = pd.DataFrame([
        {"family": "trend_following", "asset": "BTC/USDT", "timeframe": "1d",
         "variant": f"v{i}", "strategy_vs_bh_pct": 3.0, "error": None}
        for i in range(7)
    ] + [
        {"family": "trend_following", "asset": "BTC/USDT", "timeframe": "1d",
         "variant": "vX", "strategy_vs_bh_pct": -1.0, "error": None},
    ])
    out = scorecard.build_scorecard(sc, walk_forward_df=wf, robustness_df=rb,
                                    save=False)
    assert out.iloc[0]["verdict"] == "PASS"


def test_scorecard_inconclusive_for_few_trades():
    sc = pd.DataFrame([
        _strategy_row("trend_following", "BTC/USDT", "1d", vs_bh=5.0,
                      total_ret=10.0, dd=-2.0, n_trades=2),
    ])
    out = scorecard.build_scorecard(sc, save=False)
    assert out.iloc[0]["verdict"] == "INCONCLUSIVE"
