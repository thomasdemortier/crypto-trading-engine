"""Tests for the scorecard benchmark fix + per-strategy WF/robustness lookup
+ sideways mean-reversion strategy."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import backtester, config, scorecard, utils, regime
from src.strategies import (
    BENCHMARKS, RsiMaAtrStrategy, BuyAndHoldStrategy,
    SidewaysMeanReversionStrategy, TrendFollowingStrategy,
)
from src.strategies.base import BUY, SELL, HOLD, SKIP


def _strategy_row(strategy, asset, tf, vs_bh, total_ret, dd=-1.0,
                  n_trades=20, exposure=40.0):
    return {
        "strategy": strategy, "asset": asset, "timeframe": tf,
        "label": strategy,
        "total_return_pct": total_ret,
        "buy_and_hold_return_pct": total_ret - vs_bh,
        "strategy_vs_bh_pct": vs_bh,
        "max_drawdown_pct": dd,
        "win_rate_pct": 50.0,
        "num_trades": n_trades,
        "profit_factor": 1.1,
        "fees_paid": 1.0,
        "slippage_cost": 0.5,
        "exposure_time_pct": exposure,
        "sharpe_ratio": 0.5,
        "sortino_ratio": 0.5,
        "calmar_ratio": 0.1,
        "starting_capital": 10_000.0,
        "final_portfolio_value": 10_000.0 + total_ret * 100.0,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Phase 1 — scorecard benchmark handling
# ---------------------------------------------------------------------------
def test_buy_and_hold_marked_as_benchmark():
    sc_in = pd.DataFrame([
        _strategy_row("buy_and_hold", "BTC/USDT", "4h", vs_bh=10.0,
                      total_ret=10.0, n_trades=20),
        _strategy_row("trend_following", "BTC/USDT", "4h", vs_bh=2.0,
                      total_ret=8.0, n_trades=20),
    ])
    sc = scorecard.build_scorecard(sc_in, save=False)
    bh = sc[sc["strategy_name"] == "buy_and_hold"].iloc[0]
    tf = sc[sc["strategy_name"] == "trend_following"].iloc[0]
    assert bool(bh["is_benchmark"]) is True
    assert bool(tf["is_benchmark"]) is False
    assert bh["verdict"] == scorecard.BENCHMARK_VERDICT


def test_buy_and_hold_cannot_get_pass_verdict():
    """Even with a stellar score on every component, a benchmark must be
    labelled BENCHMARK, never PASS."""
    sc_in = pd.DataFrame([
        _strategy_row("buy_and_hold", "BTC/USDT", "1d", vs_bh=12.0,
                      total_ret=20.0, n_trades=30, exposure=80.0),
    ])
    wf = pd.DataFrame([
        {"strategy_name": "buy_and_hold", "asset": "BTC/USDT",
         "timeframe": "1d", "window": i,
         "strategy_return_pct": 5.0, "strategy_vs_bh_pct": 2.0,
         "num_trades": 4, "max_drawdown_pct": -1.0, "win_rate_pct": 80.0,
         "profit_factor": 2.0, "fees_paid": 0.0, "exposure_time_pct": 90.0,
         "error": None}
        for i in range(8)
    ])
    rb = pd.DataFrame([
        {"family": "buy_and_hold", "strategy_name": "buy_and_hold",
         "asset": "BTC/USDT", "timeframe": "1d",
         "variant": f"v{i}", "strategy_vs_bh_pct": 5.0, "error": None}
        for i in range(10)
    ])
    out = scorecard.build_scorecard(sc_in, walk_forward_df=wf,
                                    robustness_df=rb, save=False)
    assert out.iloc[0]["verdict"] == scorecard.BENCHMARK_VERDICT


def test_best_picks_excludes_benchmark_from_pass_and_watchlist():
    sc_in = pd.DataFrame([
        _strategy_row("buy_and_hold", "BTC/USDT", "4h", vs_bh=8.0,
                      total_ret=8.0, n_trades=20, exposure=80.0),
        _strategy_row("trend_following", "BTC/USDT", "4h", vs_bh=-5.0,
                      total_ret=-3.0, n_trades=20),
    ])
    sc = scorecard.build_scorecard(sc_in, save=False)
    picks = scorecard.best_picks(sc)
    # buy_and_hold should appear in benchmarks but contribute 0 to PASS/WATCH.
    assert any(b["strategy_name"] == "buy_and_hold" for b in picks["benchmarks"])
    assert picks["n_pass"] == 0
    # trend_following is FAIL, so n_watchlist also 0.
    assert picks["n_watchlist"] == 0
    # The "best tradable" must be a non-benchmark row.
    assert picks["best_tradable"]["strategy_name"] != "buy_and_hold"


def test_any_tradable_beats_benchmark_flag():
    sc_in = pd.DataFrame([
        _strategy_row("buy_and_hold", "BTC/USDT", "4h", vs_bh=10.0,
                      total_ret=10.0, n_trades=20),
        _strategy_row("trend_following", "BTC/USDT", "4h", vs_bh=3.0,
                      total_ret=5.0, n_trades=20),
    ])
    sc = scorecard.build_scorecard(sc_in, save=False)
    picks = scorecard.best_picks(sc)
    assert picks["any_tradable_beats_benchmark"] is True


# ---------------------------------------------------------------------------
# Phase 2 — per-strategy walk-forward
# ---------------------------------------------------------------------------
def test_scorecard_uses_strategy_specific_walk_forward():
    """When wf_df has a strategy_name column, the scorecard must NEVER
    apply one strategy's WF rows to another strategy's score."""
    sc_in = pd.DataFrame([
        _strategy_row("trend_following", "BTC/USDT", "1d", vs_bh=8.0,
                      total_ret=12.0, n_trades=30),
        _strategy_row("breakout", "BTC/USDT", "1d", vs_bh=-2.0,
                      total_ret=-1.0, n_trades=30),
    ])
    # WF only for trend_following — breakout has NO WF rows.
    wf = pd.DataFrame([
        {"strategy_name": "trend_following", "asset": "BTC/USDT",
         "timeframe": "1d", "window": i,
         "strategy_return_pct": 5.0, "strategy_vs_bh_pct": 2.0,
         "num_trades": 4, "max_drawdown_pct": -2.0, "win_rate_pct": 70.0,
         "profit_factor": 2.0, "fees_paid": 0.0, "exposure_time_pct": 60.0,
         "error": None}
        for i in range(8)
    ])
    sc = scorecard.build_scorecard(sc_in, walk_forward_df=wf, save=False)
    tf_row = sc[sc["strategy_name"] == "trend_following"].iloc[0]
    bo_row = sc[sc["strategy_name"] == "breakout"].iloc[0]
    assert tf_row["walk_forward_score"] > 0
    # breakout has no WF rows -> must score 0, not inherit trend_following's.
    assert bo_row["walk_forward_score"] == 0


def test_scorecard_missing_wf_data_gives_neutral_not_positive():
    sc_in = pd.DataFrame([
        _strategy_row("trend_following", "BTC/USDT", "1d", vs_bh=8.0,
                      total_ret=15.0, n_trades=30),
    ])
    out = scorecard.build_scorecard(sc_in, walk_forward_df=None, save=False)
    assert out.iloc[0]["walk_forward_score"] == 0
    # And the verdict can never reach PASS without WF evidence.
    assert out.iloc[0]["verdict"] != "PASS"


# ---------------------------------------------------------------------------
# Phase 3 — per-strategy robustness lookup
# ---------------------------------------------------------------------------
def test_scorecard_uses_strategy_specific_robustness():
    sc_in = pd.DataFrame([
        _strategy_row("trend_following", "BTC/USDT", "1d", vs_bh=8.0,
                      total_ret=12.0, n_trades=30),
        _strategy_row("breakout", "BTC/USDT", "1d", vs_bh=-2.0,
                      total_ret=-1.0, n_trades=30),
    ])
    rb = pd.DataFrame([
        {"family": "trend_following", "strategy_name": "trend_following",
         "asset": "BTC/USDT", "timeframe": "1d",
         "variant": f"v{i}", "strategy_vs_bh_pct": 4.0, "error": None}
        for i in range(8)
    ])
    sc = scorecard.build_scorecard(sc_in, robustness_df=rb, save=False)
    tf_row = sc[sc["strategy_name"] == "trend_following"].iloc[0]
    bo_row = sc[sc["strategy_name"] == "breakout"].iloc[0]
    assert tf_row["robustness_score"] > 0
    assert bo_row["robustness_score"] == 0  # no rows for breakout


# ---------------------------------------------------------------------------
# Phase 4 — sideways mean reversion
# ---------------------------------------------------------------------------
def _csv(symbol: str, timeframe: str, close: np.ndarray) -> Path:
    n = len(close)
    rng = np.random.default_rng(0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.4, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.4, n))
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
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(config, "DATA_RAW_DIR", raw)
    yield raw


def test_sideways_mr_emits_only_valid_signals(synthetic_cache):
    rng = np.random.default_rng(1)
    n = 600
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))  # random walk → mostly sideways
    _csv("BTC/USDT", "4h", close)
    art = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False,
        strategy=SidewaysMeanReversionStrategy(),
    )
    assert not art.equity_curve.empty
    if not art.trades.empty:
        assert set(art.trades["side"].unique()).issubset({"BUY", "SELL"})
    if not art.decisions.empty:
        assert set(art.decisions["action"].unique()).issubset(
            {"BUY", "SELL", "HOLD", "SKIP", "REJECT"}
        )


def test_sideways_mr_does_not_buy_in_strong_bull_trend(synthetic_cache):
    """Aggressive uptrend — regime is bull_trend, so the MR strategy
    must not open new positions."""
    rng = np.random.default_rng(0)
    n = 600
    close = np.linspace(100, 300, n) + rng.normal(0, 0.3, n)
    _csv("BTC/USDT", "4h", close)
    art = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False,
        strategy=SidewaysMeanReversionStrategy(),
    )
    # Zero BUYs in a sustained uptrend.
    assert art.trades.empty or (art.trades["side"] == "BUY").sum() == 0


def test_sideways_mr_does_not_buy_in_bear_trend(synthetic_cache):
    rng = np.random.default_rng(0)
    n = 600
    close = np.linspace(300, 100, n) + rng.normal(0, 0.3, n)
    _csv("BTC/USDT", "4h", close)
    art = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False,
        strategy=SidewaysMeanReversionStrategy(),
    )
    assert art.trades.empty or (art.trades["side"] == "BUY").sum() == 0


def test_sideways_mr_no_lookahead_in_bollinger(synthetic_cache):
    """Spot-check: BB upper at row i must equal BB upper computed on
    df.iloc[:i+1] alone."""
    from src import indicators
    rng = np.random.default_rng(2)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, 200)))
    full_mid, full_up, full_lo = indicators.bollinger_bands(close, 20, 2.0)
    for i in (50, 100, 150):
        sub = close.iloc[: i + 1]
        m, u, lo = indicators.bollinger_bands(sub, 20, 2.0)
        assert pytest.approx(float(full_mid.iloc[i])) == float(m.iloc[i])
        assert pytest.approx(float(full_up.iloc[i])) == float(u.iloc[i])
        assert pytest.approx(float(full_lo.iloc[i])) == float(lo.iloc[i])


def test_sideways_mr_respects_max_holding_bars():
    """If the strategy enters and the exit conditions never fire, the
    `max_holding_bars` clock must force a SELL."""
    s = SidewaysMeanReversionStrategy()
    s._bars_held["X/USDT"] = s.cfg.max_holding_bars  # simulate already held
    # Build a row where exit conditions don't trigger naturally.
    row = pd.Series({
        "timestamp": 1, "datetime": pd.Timestamp.utcnow(),
        "close": 100.0, "open": 99.5, "high": 100.5, "low": 99.0,
        "smr_bb_mid": 200.0, "smr_bb_upper": 250.0, "smr_bb_lower": 50.0,
        "rsi": 50.0, "smr_rsi_prev": 49.0,
        "atr_pct": 1.0, "smr_ma200_dist_pct": 1.0,
        "smr_close_prev": 99.0, "ma200": 100.0,
        "trend_regime": regime.SIDEWAYS,
        "volatility_regime": regime.LOW_VOL,
    })
    sig = s.signal_for_row("X/USDT", row, in_position=True, cfg=None)
    assert sig.action == SELL
    assert "max holding" in sig.reason


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------
def test_no_live_trading_in_new_modules():
    src = Path(__file__).resolve().parents[1] / "src"
    targets = [
        src / "scorecard.py",
        src / "regime.py",
        src / "strategies" / "sideways_mean_reversion.py",
        src / "strategies" / "regime_filtered.py",
        src / "strategies" / "trend_following.py",
        src / "strategies" / "pullback_continuation.py",
    ]
    forbidden = (
        "create_order", "createMarketBuyOrder", "createMarketSellOrder",
        "create_market_buy_order", "create_market_sell_order",
        ".privateGet", ".privatePost", "apiKey=", "secret=",
    )
    offenders = []
    for p in targets:
        if not p.exists():
            continue
        text = p.read_text()
        for s in forbidden:
            if s in text:
                offenders.append((p.name, s))
    assert not offenders, f"forbidden tokens: {offenders}"
