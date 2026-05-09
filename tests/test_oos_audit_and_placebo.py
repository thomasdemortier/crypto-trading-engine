"""Tests for the OOS audit, placebo strategy, placebo comparison, and
data coverage audit."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import (
    backtester, config, oos_audit, performance, research, scorecard, utils,
)
from src.strategies import (
    BENCHMARKS, PLACEBOS, PlaceboRandomStrategy, REGISTRY,
    RsiMaAtrStrategy,
)
from src.strategies.base import BUY, SELL, HOLD, SKIP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _csv(symbol: str, timeframe: str, n: int = 600, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    base = np.linspace(100.0, 200.0, n)
    noise = rng.normal(0, 1.5, n)
    close = np.maximum(base + noise, 1.0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5, n))
    vol = rng.uniform(100, 1_000, n)
    ts = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": (ts.astype("int64") // 10**6).astype("int64"),
        "datetime": ts, "open": open_, "high": high, "low": low,
        "close": close, "volume": vol,
    })
    p = utils.csv_path_for(symbol, timeframe)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def synthetic_cache(tmp_path, monkeypatch):
    raw = tmp_path / "raw"; raw.mkdir()
    res = tmp_path / "results"; res.mkdir()
    monkeypatch.setattr(config, "DATA_RAW_DIR", raw)
    monkeypatch.setattr(config, "RESULTS_DIR", res)
    yield {"raw": raw, "results": res}


# ---------------------------------------------------------------------------
# Phase 2 — placebo strategy
# ---------------------------------------------------------------------------
def test_placebo_registered_and_in_placebos_set():
    assert "placebo_random" in REGISTRY
    assert "placebo_random" in PLACEBOS


def test_placebo_emits_only_valid_signals(synthetic_cache):
    _csv("BTC/USDT", "4h", n=600, seed=1)
    art = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False,
        strategy=PlaceboRandomStrategy(),
    )
    if not art.decisions.empty:
        assert set(art.decisions["action"].unique()).issubset(
            {"BUY", "SELL", "HOLD", "SKIP", "REJECT"}
        )
    if not art.trades.empty:
        assert set(art.trades["side"].unique()).issubset({"BUY", "SELL"})


def test_placebo_is_reproducible_with_fixed_seed(synthetic_cache):
    _csv("BTC/USDT", "4h", n=600, seed=1)
    art1 = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False,
        strategy=PlaceboRandomStrategy(),
    )
    art2 = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False,
        strategy=PlaceboRandomStrategy(),
    )
    # Same seed -> same number of trades and same decision actions.
    assert len(art1.trades) == len(art2.trades)
    if not art1.decisions.empty and not art2.decisions.empty:
        a1 = art1.decisions[["timestamp_ms", "action"]].reset_index(drop=True)
        a2 = art2.decisions[["timestamp_ms", "action"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(a1, a2)


def test_placebo_cannot_get_pass_or_watchlist():
    """Even with a stellar score the scorecard must short-circuit to
    PLACEBO."""
    sc_in = pd.DataFrame([{
        "strategy": "placebo_random", "asset": "BTC/USDT", "timeframe": "1h",
        "label": "placebo_random", "total_return_pct": 50.0,
        "buy_and_hold_return_pct": 5.0,
        "buy_and_hold_max_drawdown_pct": -10.0,
        "strategy_vs_bh_pct": 45.0, "drawdown_vs_bh_pct": 5.0,
        "max_drawdown_pct": -2.0, "win_rate_pct": 80.0,
        "num_trades": 40, "profit_factor": 3.0,
        "fees_paid": 1.0, "slippage_cost": 0.5,
        "exposure_time_pct": 60.0,
        "sharpe_ratio": 2.0, "sortino_ratio": 2.0, "calmar_ratio": 0.8,
        "starting_capital": 10_000.0, "final_portfolio_value": 15_000.0,
        "error": None,
    }])
    sc = scorecard.build_scorecard(sc_in, save=False)
    row = sc.iloc[0]
    assert row["verdict"] == scorecard.PLACEBO_VERDICT
    assert row["verdict"] != "PASS"
    assert row["verdict"] != "WATCHLIST"
    assert bool(row["is_placebo"]) is True


def test_best_picks_excludes_placebo_from_pass_and_watchlist():
    sc_in = pd.DataFrame([
        {"strategy": "placebo_random", "asset": "BTC/USDT", "timeframe": "1h",
         "label": "placebo_random", "total_return_pct": 30.0,
         "buy_and_hold_return_pct": 5.0,
         "buy_and_hold_max_drawdown_pct": -10.0,
         "strategy_vs_bh_pct": 25.0, "drawdown_vs_bh_pct": 5.0,
         "max_drawdown_pct": -2.0, "win_rate_pct": 70.0,
         "num_trades": 30, "profit_factor": 2.0, "fees_paid": 1.0,
         "slippage_cost": 0.5, "exposure_time_pct": 60.0,
         "sharpe_ratio": 1.0, "sortino_ratio": 1.0, "calmar_ratio": 0.4,
         "starting_capital": 10_000.0, "final_portfolio_value": 13_000.0,
         "error": None},
        {"strategy": "trend_following", "asset": "BTC/USDT", "timeframe": "1h",
         "label": "trend_following", "total_return_pct": -5.0,
         "buy_and_hold_return_pct": 2.0,
         "buy_and_hold_max_drawdown_pct": -10.0,
         "strategy_vs_bh_pct": -7.0, "drawdown_vs_bh_pct": -2.0,
         "max_drawdown_pct": -8.0, "win_rate_pct": 30.0,
         "num_trades": 20, "profit_factor": 0.8, "fees_paid": 1.0,
         "slippage_cost": 0.5, "exposure_time_pct": 50.0,
         "sharpe_ratio": -0.5, "sortino_ratio": -0.5, "calmar_ratio": -0.1,
         "starting_capital": 10_000.0, "final_portfolio_value": 9_500.0,
         "error": None},
    ])
    sc = scorecard.build_scorecard(sc_in, save=False)
    picks = scorecard.best_picks(sc)
    # No PASS, no WATCHLIST among tradable.
    assert picks["n_pass"] == 0
    assert picks["n_watchlist"] == 0
    # Placebo appears in the placebos list, not the tradable best.
    assert any(p["strategy_name"] == "placebo_random" for p in picks["placebos"])
    assert picks["best_tradable"]["strategy_name"] != "placebo_random"


# ---------------------------------------------------------------------------
# Phase 1 — OOS audit
# ---------------------------------------------------------------------------
def _wf_frame() -> pd.DataFrame:
    rows = []
    for w in range(1, 8):
        rows.append({
            "strategy_name": "trend_following", "asset": "BTC/USDT",
            "timeframe": "4h", "window": w,
            "oos_start_iso": f"2024-01-{w:02d}T00:00:00+00:00",
            "oos_end_iso": f"2024-01-{w + 3:02d}T00:00:00+00:00",
            "strategy_return_pct": 1.0 if w % 2 else -0.5,
            "buy_and_hold_return_pct": 0.5,
            "strategy_vs_bh_pct": 0.5 if w % 2 else -1.0,
            "max_drawdown_pct": -0.3, "win_rate_pct": 50.0,
            "num_trades": 4, "profit_factor": 1.2, "fees_paid": 0.0,
            "exposure_time_pct": 40.0, "error": None,
        })
    # one error row that should be skipped from the valid summary
    rows.append({
        "strategy_name": "trend_following", "asset": "BTC/USDT",
        "timeframe": "4h", "window": 99,
        "oos_start_iso": None, "oos_end_iso": None,
        "strategy_return_pct": None, "buy_and_hold_return_pct": None,
        "strategy_vs_bh_pct": None, "max_drawdown_pct": None,
        "win_rate_pct": None, "num_trades": 0, "profit_factor": None,
        "fees_paid": None, "exposure_time_pct": None,
        "error": "synthetic error",
    })
    return pd.DataFrame(rows)


def test_audit_produces_audit_and_summary_files(synthetic_cache):
    audit_df, summary_df = oos_audit.audit_walk_forward(_wf_frame(), save=True)
    assert (synthetic_cache["results"] / "oos_audit.csv").exists()
    assert (synthetic_cache["results"] / "oos_audit_summary.csv").exists()
    assert not audit_df.empty
    assert not summary_df.empty


def test_audit_window_counts_and_stability_correct():
    _, summary_df = oos_audit.audit_walk_forward(_wf_frame(), save=False)
    row = summary_df.iloc[0]
    # 7 valid windows, 1 errored row excluded
    assert row["n_windows"] == 7
    assert row["n_total_rows"] == 8
    # 4 of 7 windows have positive return and beat B&H
    assert row["n_profitable"] == 4
    assert row["n_beats_bh"] == 4
    assert row["n_profitable_and_beats_bh"] == 4


def test_audit_handles_missing_walk_forward_safely(synthetic_cache):
    """No CSV present -> must return empty frames, never raise."""
    audit_df, summary_df = oos_audit.audit_walk_forward(save=True)
    assert audit_df.empty
    assert summary_df.empty


def test_audit_flags_low_trade_count_windows():
    df = _wf_frame()
    df["num_trades"] = 1  # every window thin
    audit_df, summary_df = oos_audit.audit_walk_forward(df, save=False)
    assert audit_df["low_trade_count"].all()
    note = summary_df.iloc[0]["notes"]
    assert "<3 trades" in note or "thin" in note.lower() or "trades" in note


# ---------------------------------------------------------------------------
# Phase 3 — placebo comparison
# ---------------------------------------------------------------------------
def _wf_with_placebo() -> pd.DataFrame:
    rows = []
    # Strategy beats placebo: higher stability + higher mean return.
    for w in range(1, 9):
        rows.append({
            "strategy_name": "trend_following", "asset": "BTC/USDT",
            "timeframe": "4h", "window": w,
            "oos_start_iso": None, "oos_end_iso": None,
            "strategy_return_pct": 2.0, "buy_and_hold_return_pct": 0.0,
            "strategy_vs_bh_pct": 2.0, "num_trades": 4,
            "max_drawdown_pct": -1.0, "win_rate_pct": 60.0,
            "profit_factor": 1.5, "fees_paid": 0.0,
            "exposure_time_pct": 50.0, "error": None,
        })
    # Placebo: lower stability + much lower mean return.
    for w in range(1, 9):
        rows.append({
            "strategy_name": "placebo_random", "asset": "BTC/USDT",
            "timeframe": "4h", "window": w,
            "oos_start_iso": None, "oos_end_iso": None,
            "strategy_return_pct": -0.5, "buy_and_hold_return_pct": 0.0,
            "strategy_vs_bh_pct": -0.5, "num_trades": 4,
            "max_drawdown_pct": -2.0, "win_rate_pct": 40.0,
            "profit_factor": 0.8, "fees_paid": 0.0,
            "exposure_time_pct": 40.0, "error": None,
        })
    return pd.DataFrame(rows)


def test_placebo_comparison_detects_strategy_beating_placebo():
    df = research.placebo_comparison(_wf_with_placebo(), save=False)
    row = df[df["strategy_name"] == "trend_following"].iloc[0]
    assert bool(row["strategy_beats_placebo"]) is True


def test_placebo_comparison_detects_strategy_failing_placebo():
    wf = _wf_with_placebo()
    # Flip the strategy to be worse than placebo.
    wf.loc[wf["strategy_name"] == "trend_following", "strategy_return_pct"] = -2.0
    wf.loc[wf["strategy_name"] == "trend_following", "strategy_vs_bh_pct"] = -2.0
    df = research.placebo_comparison(wf, save=False)
    row = df[df["strategy_name"] == "trend_following"].iloc[0]
    assert bool(row["strategy_beats_placebo"]) is False


def test_placebo_comparison_inconclusive_for_low_trade_count():
    wf = _wf_with_placebo()
    wf.loc[wf["strategy_name"] == "trend_following", "num_trades"] = 1
    df = research.placebo_comparison(wf, save=False)
    row = df[df["strategy_name"] == "trend_following"].iloc[0]
    assert "INCONCLUSIVE" in row["notes"]
    assert bool(row["strategy_beats_placebo"]) is False


def test_placebo_excluded_from_comparison_table():
    df = research.placebo_comparison(_wf_with_placebo(), save=False)
    # The placebo itself must not appear as a row being compared.
    assert "placebo_random" not in set(df["strategy_name"].unique())
    # Buy and hold likewise — it's a benchmark.
    assert "buy_and_hold" not in set(df["strategy_name"].unique())


def test_placebo_comparison_handles_empty_input(synthetic_cache):
    df = research.placebo_comparison(pd.DataFrame(), save=False)
    assert df.empty


# ---------------------------------------------------------------------------
# Phase 4 — data coverage audit
# ---------------------------------------------------------------------------
def test_data_coverage_creates_csv_and_flags_missing(synthetic_cache):
    # Only seed BTC/USDT 4h — ETH/USDT 4h will be flagged as missing.
    _csv("BTC/USDT", "4h", n=600, seed=1)
    df = research.data_coverage_audit(
        assets=("BTC/USDT", "ETH/USDT"), timeframes=("4h",),
        requested_lookback_days=730, save=True,
    )
    assert (synthetic_cache["results"] / "data_coverage.csv").exists()
    assert len(df) == 2
    btc_row = df[df["asset"] == "BTC/USDT"].iloc[0]
    eth_row = df[df["asset"] == "ETH/USDT"].iloc[0]
    assert btc_row["candle_count"] > 0
    assert eth_row["candle_count"] == 0
    assert "missing data" in str(eth_row["notes"])


def test_data_coverage_flags_short_history(synthetic_cache):
    _csv("BTC/USDT", "4h", n=200, seed=1)  # ~33 days of 4h bars
    df = research.data_coverage_audit(
        assets=("BTC/USDT",), timeframes=("4h",),
        requested_lookback_days=730, save=False,
    )
    row = df.iloc[0]
    assert bool(row["enough_for_walk_forward"]) is False
    assert "insufficient" in str(row["notes"]).lower() or \
           "only" in str(row["notes"]).lower()


def test_data_coverage_works_without_lookback_days(synthetic_cache):
    _csv("BTC/USDT", "4h", n=600, seed=1)
    df = research.data_coverage_audit(
        assets=("BTC/USDT",), timeframes=("4h",),
        requested_lookback_days=None, save=False,
    )
    assert len(df) == 1
    assert df.iloc[0]["requested_lookback_days"] is None
