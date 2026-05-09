"""Research module regression tests.

Confirms:
* Walk-forward window splits never produce in-sample / out-of-sample overlap.
* Strategy comparison produces one row per (strategy, asset, timeframe).
* Robustness grid produces results for each family.
* Monte Carlo handles too-few-trades safely (returns ok=False, never raises).
* Monte Carlo produces sane summary statistics for a sufficient trade set.
* Research summary correctly reports FAIL when the synthetic strategy
  underperforms B&H, and produces a deterministic shape regardless.
* Research module does not introduce any live-trading or API-key surfaces
  (covered by test_strategies.test_no_strategy_imports_live_trading_or_keys
  but re-asserted here for the research import surface).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config, research, utils
from src.strategies import (
    BreakoutStrategy, BuyAndHoldStrategy,
    MovingAverageCrossStrategy, RsiMaAtrStrategy,
)


def _synthetic_csv(symbol: str, timeframe: str, n: int = 800,
                   seed: int = 0) -> Path:
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
        "timestamp": (ts.astype("datetime64[ns, UTC]").astype("int64") // 10**6).astype("int64"),
        "datetime": ts, "open": open_, "high": high, "low": low,
        "close": close, "volume": vol,
    })
    p = utils.csv_path_for(symbol, timeframe)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def synthetic_cache(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    res = tmp_path / "results"
    res.mkdir()
    monkeypatch.setattr(config, "DATA_RAW_DIR", raw)
    monkeypatch.setattr(config, "RESULTS_DIR", res)
    yield {"raw": raw, "results": res}


# ---------------------------------------------------------------------------
# Walk-forward window split correctness
# ---------------------------------------------------------------------------
def test_walk_forward_window_split_is_disjoint():
    day = 24 * 60 * 60 * 1000
    first = 1_700_000_000_000
    last = first + 365 * day
    windows = research._build_windows(first, last,
                                      in_sample_days=90, oos_days=30,
                                      step_days=30)
    assert windows, "expected at least one walk-forward window"
    for w in windows:
        # in-sample strictly precedes out-of-sample
        assert w["is_end_ms"] < w["oos_start_ms"], (
            f"in-sample end {w['is_end_ms']} must be < OOS start "
            f"{w['oos_start_ms']}"
        )
        # in-sample length is exactly the requested span
        assert w["is_end_ms"] - w["is_start_ms"] == 90 * day - 1
        # out-of-sample length is exactly the requested span
        assert w["oos_end_ms"] - w["oos_start_ms"] == 30 * day - 1


def test_walk_forward_no_window_overlap_within_oos_set():
    day = 24 * 60 * 60 * 1000
    first = 1_700_000_000_000
    last = first + 365 * day
    windows = research._build_windows(first, last, 90, 30, 30)
    # OOS regions of consecutive windows can be adjacent but never overlap.
    for a, b in zip(windows, windows[1:]):
        assert a["oos_end_ms"] < b["oos_start_ms"] or \
               b["oos_start_ms"] >= a["oos_start_ms"] + 30 * day


# ---------------------------------------------------------------------------
# Strategy comparison shape
# ---------------------------------------------------------------------------
def test_strategy_comparison_produces_row_per_combination(synthetic_cache):
    _synthetic_csv("BTC/USDT", "4h", n=600, seed=1)
    _synthetic_csv("ETH/USDT", "4h", n=600, seed=2)
    df = research.strategy_comparison(
        strategies=[RsiMaAtrStrategy(), BuyAndHoldStrategy(),
                    MovingAverageCrossStrategy(fast=20, slow=50),
                    BreakoutStrategy(entry_window=10, exit_window=5)],
        assets=("BTC/USDT", "ETH/USDT"), timeframes=("4h",), save=False,
    )
    # 4 strategies × 2 assets × 1 timeframe = 8 rows expected (some may have
    # error filled if min_history is too long for the synthetic dataset).
    assert len(df) == 4 * 2 * 1
    assert {"strategy", "asset", "timeframe"}.issubset(df.columns)


# ---------------------------------------------------------------------------
# Robustness grid shape
# ---------------------------------------------------------------------------
def test_robustness_grid_produces_per_family_rows(synthetic_cache):
    _synthetic_csv("BTC/USDT", "4h", n=600, seed=1)
    df = research.robustness(
        assets=("BTC/USDT",), timeframes=("4h",), save=False,
    )
    families = set(df.get("family", pd.Series()).dropna().unique())
    assert {"rsi_ma_atr", "ma_cross", "breakout"}.issubset(families)


# ---------------------------------------------------------------------------
# Monte Carlo edge cases
# ---------------------------------------------------------------------------
def test_monte_carlo_handles_empty_trades_safely():
    res = research.monte_carlo_from_trades(pd.DataFrame(), 10_000.0,
                                           n_sim=100, save=False)
    assert res["ok"] is False
    assert "no trades" in res["reason"]


def test_monte_carlo_handles_too_few_trades_safely():
    df = pd.DataFrame([
        {"side": "BUY",  "realized_pnl": 0.0, "timestamp_ms": 1},
        {"side": "SELL", "realized_pnl": 5.0, "timestamp_ms": 2},
        {"side": "BUY",  "realized_pnl": 0.0, "timestamp_ms": 3},
        {"side": "SELL", "realized_pnl": -3.0, "timestamp_ms": 4},
    ])
    res = research.monte_carlo_from_trades(df, 10_000.0, n_sim=100, save=False)
    assert res["ok"] is False
    assert "too few trades" in res["reason"].lower()


def test_monte_carlo_runs_with_enough_trades():
    rng = np.random.default_rng(0)
    n = 50
    pnls = rng.normal(0.5, 5.0, n)  # slight positive drift, high variance
    rows = []
    for i, p in enumerate(pnls):
        rows.append({"side": "BUY",  "realized_pnl": 0.0, "timestamp_ms": 2 * i})
        rows.append({"side": "SELL", "realized_pnl": float(p),
                     "timestamp_ms": 2 * i + 1})
    df = pd.DataFrame(rows)
    res = research.monte_carlo_from_trades(df, 10_000.0, n_sim=500,
                                           seed=123, save=False)
    assert res["ok"] is True
    assert res["n_trades"] == n
    assert res["n_simulations"] == 500
    # Sanity: percentiles ordered correctly
    assert res["p05_final_value"] <= res["median_final_value"] <= res["p95_final_value"]
    # Probability of loss is between 0 and 1
    assert 0.0 <= res["prob_loss"] <= 1.0
    # Worst drawdown is non-positive
    assert res["worst_drawdown_pct"] <= 0.0


# ---------------------------------------------------------------------------
# Research summary verdict shape
# ---------------------------------------------------------------------------
def test_research_summary_returns_expected_checks_with_verdicts():
    summary = research.research_summary(
        timeframe_df=None, walk_forward_df=None,
        strategy_df=None, robustness_df=None,
        monte_carlo_summary=None, save=False,
    )
    expected = {
        "beats_buy_and_hold", "works_on_btc_and_eth",
        "works_across_timeframes", "works_out_of_sample",
        "robust_to_parameters", "statistically_meaningful_trade_count",
        "best_tradable_by_scorecard", "regime_selector_outcome",
        "worth_paper_trading_further",
    }
    assert set(summary["checks"].keys()) == expected
    for v in summary["checks"].values():
        assert v["verdict"] in {"PASS", "FAIL", "INCONCLUSIVE"}
        assert isinstance(v["message"], str) and v["message"]


def test_research_summary_marks_underperformance_as_fail():
    """If every timeframe row underperforms B&H, beats_buy_and_hold must FAIL
    — never silently massage the verdict."""
    bad = pd.DataFrame([
        {"asset": "BTC/USDT", "timeframe": "4h",
         "strategy_vs_bh_pct": -10.0, "num_trades": 4, "error": None},
        {"asset": "ETH/USDT", "timeframe": "4h",
         "strategy_vs_bh_pct": -8.0, "num_trades": 5, "error": None},
    ])
    summary = research.research_summary(
        timeframe_df=bad, walk_forward_df=None, strategy_df=None,
        robustness_df=None, monte_carlo_summary=None, save=False,
    )
    assert summary["checks"]["beats_buy_and_hold"]["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# Research module is paper-only and does not invoke private exchange APIs
# ---------------------------------------------------------------------------
def test_research_module_does_not_reference_private_exchange_apis():
    src_root = Path(__file__).resolve().parents[1] / "src" / "research.py"
    text = src_root.read_text()
    forbidden = (
        "create_order", "create_market_buy_order",
        "create_market_sell_order", "withdraw(", ".privateGet",
        ".privatePost", "apiKey=", "secret=",
    )
    for s in forbidden:
        assert s not in text, f"research.py must not contain {s!r}"


def test_research_module_calls_assert_paper_only_via_backtester():
    """Research helpers route through backtester.run_backtest, which calls
    utils.assert_paper_only(). Confirm the path: flipping LIVE_TRADING_ENABLED
    on must cause research helpers to refuse to run."""
    from src import utils as _u
    # Save and restore so we don't pollute the test session.
    original = config.LIVE_TRADING_ENABLED
    try:
        config.LIVE_TRADING_ENABLED = True
        with pytest.raises(_u.LiveTradingForbiddenError):
            research.timeframe_comparison(
                assets=("BTC/USDT",), timeframes=("4h",), save=False,
            )
    finally:
        config.LIVE_TRADING_ENABLED = original
