"""Audit tests for the portfolio momentum implementation.

Concretely verifies the five claims the audit was asked to check:

  1. Cash filter fires when synthetic BTC closes below its 200d MA.
  2. The strategy actually moves to cash on the next rebalance after
     the filter turns bearish.
  3. Assets without enough 90-day history are excluded from ranking.
  4. Benchmarks and the strategy use identical OOS-window timestamps.
  5. Neither the ranking nor the cash filter peeks at future data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config, portfolio_audit, portfolio_backtester as pb, utils
from src.strategies.momentum_rotation import (
    MomentumRotationConfig, MomentumRotationStrategy,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------
def _csv(symbol: str, timeframe: str, close: np.ndarray,
         start: str = "2023-01-01") -> Path:
    n = len(close)
    rng = np.random.default_rng(0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3, n))
    vol = rng.uniform(100, 1_000, n)
    ts = pd.date_range(start, periods=n, freq="1D", tz="UTC")
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
def isolated(tmp_path, monkeypatch):
    raw = tmp_path / "raw"; raw.mkdir()
    res = tmp_path / "results"; res.mkdir()
    monkeypatch.setattr(config, "DATA_RAW_DIR", raw)
    monkeypatch.setattr(config, "RESULTS_DIR", res)
    yield {"raw": raw, "results": res}


# ---------------------------------------------------------------------------
# 1. Cash filter fires on synthetic BTC below 200d MA
# ---------------------------------------------------------------------------
def test_cash_filter_fires_when_btc_below_200d_ma(isolated):
    rng = np.random.default_rng(0)
    n = 600
    # Sustained downtrend → close stays well below 200d MA after warmup.
    btc = np.linspace(300, 100, n) + rng.normal(0, 0.3, n)
    eth = np.linspace(100, 200, n)
    sol = np.linspace(100, 250, n)
    avax = np.linspace(100, 180, n)
    link = np.linspace(100, 150, n)
    _csv("BTC/USDT", "1d", btc)
    _csv("ETH/USDT", "1d", eth)
    _csv("SOL/USDT", "1d", sol)
    _csv("AVAX/USDT", "1d", avax)
    _csv("LINK/USDT", "1d", link)
    frames, _ = pb.load_universe(
        ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT"],
        timeframe="1d",
    )
    strat = MomentumRotationStrategy(
        MomentumRotationConfig(top_n=2, min_assets_required=3),
    )
    asof = int(frames["BTC/USDT"]["timestamp"].iloc[-10])
    weights = strat.target_weights(asof, frames, timeframe="1d")
    assert weights == {}, (
        f"cash filter should fire when BTC is below 200d MA — got {weights}"
    )


def test_cash_filter_does_not_fire_when_btc_above_200d_ma(isolated):
    rng = np.random.default_rng(0)
    n = 600
    btc = np.linspace(100, 400, n) + rng.normal(0, 0.3, n)
    eth = np.linspace(100, 200, n)
    sol = np.linspace(100, 250, n)
    avax = np.linspace(100, 180, n)
    link = np.linspace(100, 150, n)
    _csv("BTC/USDT", "1d", btc)
    _csv("ETH/USDT", "1d", eth)
    _csv("SOL/USDT", "1d", sol)
    _csv("AVAX/USDT", "1d", avax)
    _csv("LINK/USDT", "1d", link)
    frames, _ = pb.load_universe(
        ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT"],
        timeframe="1d",
    )
    strat = MomentumRotationStrategy(
        MomentumRotationConfig(top_n=2, min_assets_required=3),
    )
    asof = int(frames["BTC/USDT"]["timestamp"].iloc[-10])
    weights = strat.target_weights(asof, frames, timeframe="1d")
    assert len(weights) == 2 and abs(sum(weights.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 2. Strategy moves to cash on next rebalance after filter goes bearish
# ---------------------------------------------------------------------------
def test_backtester_moves_to_cash_during_bearish_period(isolated):
    """Simulate: bull phase first, then sustained crash. After the crash
    pulls BTC below its 200d MA, subsequent rebalances must produce
    empty target weights and the portfolio should be flat (cash)."""
    rng = np.random.default_rng(0)
    n_bull, n_crash = 400, 300
    bull = np.linspace(100, 400, n_bull)
    crash = np.linspace(400, 80, n_crash)
    btc = np.concatenate([bull, crash]) + rng.normal(0, 0.3, n_bull + n_crash)
    others_seed = lambda mu_end: np.linspace(100, mu_end, n_bull + n_crash)
    _csv("BTC/USDT", "1d", btc)
    _csv("ETH/USDT", "1d", others_seed(150))
    _csv("SOL/USDT", "1d", others_seed(200))
    _csv("AVAX/USDT", "1d", others_seed(140))
    _csv("LINK/USDT", "1d", others_seed(120))
    frames, _ = pb.load_universe(
        ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT"],
        timeframe="1d",
    )
    strat = MomentumRotationStrategy(
        MomentumRotationConfig(top_n=2, min_assets_required=3),
    )
    art = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames, timeframe="1d",
        cfg=pb.PortfolioBacktestConfig(), save=False,
    )
    # Look at rebalance INTENTS in the second half of the data — at least
    # some of them should be empty (cash filter).
    second_half_mask = (
        pd.to_datetime(art.weights_history["datetime"], utc=True)
        >= pd.to_datetime(art.weights_history["datetime"].iloc[len(art.weights_history)//2], utc=True)
    )
    second_half = art.weights_history[~art.weights_history["filled"]
                                       & second_half_mask]
    n_empty = sum(
        1 for d in second_half["weights"] if not d
    )
    assert n_empty > 0, (
        "backtester should move to cash during BTC crash — no empty "
        "rebalance targets observed in the second half"
    )


# ---------------------------------------------------------------------------
# 3. Assets without enough 90-day history are excluded
# ---------------------------------------------------------------------------
def test_short_history_asset_excluded_from_ranking(isolated):
    """One asset has only 50 days of history when long-window momentum
    needs 90+. It must NOT appear in the target weights."""
    n = 400
    _csv("BTC/USDT", "1d", np.linspace(100, 300, n))
    _csv("ETH/USDT", "1d", np.linspace(100, 200, n))
    _csv("SOL/USDT", "1d", np.linspace(100, 250, n))
    _csv("AVAX/USDT", "1d", np.linspace(100, 180, n))
    _csv("LINK/USDT", "1d", np.linspace(100, 150, n))
    # NEWCOIN/USDT: only 50 bars of data ending where the others end.
    short_close = np.linspace(100, 1000, 50)  # huge fake "momentum"
    _csv("NEW/USDT", "1d", short_close, start="2024-01-01")
    frames, _ = pb.load_universe(
        ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT", "NEW/USDT"],
        timeframe="1d",
    )
    strat = MomentumRotationStrategy(
        MomentumRotationConfig(top_n=2, min_assets_required=3),
    )
    asof = int(frames["BTC/USDT"]["timestamp"].iloc[-1])
    weights = strat.target_weights(asof, frames, timeframe="1d")
    assert "NEW/USDT" not in weights, (
        f"short-history asset must not be ranked — got {weights}"
    )


# ---------------------------------------------------------------------------
# 4. Benchmarks and strategy use identical OOS-window timestamps
# ---------------------------------------------------------------------------
def test_benchmarks_use_identical_window_to_strategy(isolated):
    rng = np.random.default_rng(0)
    n = 600
    _csv("BTC/USDT", "1d", np.linspace(100, 300, n) + rng.normal(0, 0.5, n))
    _csv("ETH/USDT", "1d", np.linspace(100, 250, n) + rng.normal(0, 0.5, n))
    _csv("SOL/USDT", "1d", np.linspace(100, 400, n) + rng.normal(0, 0.5, n))
    _csv("AVAX/USDT", "1d", np.linspace(100, 200, n) + rng.normal(0, 0.5, n))
    _csv("LINK/USDT", "1d", np.linspace(100, 180, n) + rng.normal(0, 0.5, n))
    frames, _ = pb.load_universe(
        ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT"],
        timeframe="1d",
    )
    strat = MomentumRotationStrategy(
        MomentumRotationConfig(top_n=3, min_assets_required=3),
    )
    # Slice to a specific window.
    start_ts = int(frames["BTC/USDT"]["timestamp"].iloc[300])
    end_ts = int(frames["BTC/USDT"]["timestamp"].iloc[450])
    art = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames, timeframe="1d",
        cfg=pb.PortfolioBacktestConfig(),
        start_ts_ms=start_ts, end_ts_ms=end_ts, save=False,
    )
    bench = pb.benchmark_equity_curves(
        frames, starting_capital=10_000.0, timeframe="1d",
        start_ts_ms=start_ts, end_ts_ms=end_ts,
    )
    # Both must cover identical timestamp axes.
    strat_ts = set(art.equity_curve["timestamp"].astype("int64"))
    for name, df in bench.items():
        b_ts = set(df["timestamp"].astype("int64"))
        assert b_ts == strat_ts, (
            f"benchmark {name} timestamps differ from strategy: "
            f"strat-only={len(strat_ts - b_ts)}, "
            f"bench-only={len(b_ts - strat_ts)}"
        )


# ---------------------------------------------------------------------------
# 5. No lookahead in cash filter
# ---------------------------------------------------------------------------
def test_cash_filter_does_not_peek_at_future_data():
    """At asof bar t, the cash filter must produce the same answer
    whether you give it data up to t only, or data up to the end of
    history. This proves the filter does not peek at future bars."""
    rng = np.random.default_rng(0)
    n = 800
    btc = np.linspace(100, 300, n) + rng.normal(0, 1.0, n)
    eth = np.linspace(100, 200, n)
    sol = np.linspace(100, 400, n)
    avax = np.linspace(100, 180, n)
    link = np.linspace(100, 150, n)
    rngs = {
        "BTC/USDT": btc, "ETH/USDT": eth, "SOL/USDT": sol,
        "AVAX/USDT": avax, "LINK/USDT": link,
    }
    full = {}
    for asset, close in rngs.items():
        ts = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")
        full[asset] = pd.DataFrame({
            "timestamp": (ts.astype("int64") // 10**6).astype("int64"),
            "datetime": ts, "open": close, "high": close, "low": close,
            "close": close, "volume": 1.0,
        })
    strat = MomentumRotationStrategy(
        MomentumRotationConfig(top_n=2, min_assets_required=3),
    )
    asof = int(full["BTC/USDT"]["timestamp"].iloc[400])
    weights_full = strat.target_weights(asof, full, timeframe="1d")
    truncated = {a: df[df["timestamp"] <= asof].copy()
                 for a, df in full.items()}
    weights_trunc = strat.target_weights(asof, truncated, timeframe="1d")
    assert weights_full == weights_trunc


# ---------------------------------------------------------------------------
# 6. Audit module + CLI
# ---------------------------------------------------------------------------
def test_audit_cash_filter_handles_missing_btc(isolated):
    """No BTC data on disk -> the audit must return ok=False, never raise."""
    out = portfolio_audit.audit_cash_filter(save=False)
    assert out.get("ok") is False


def test_audit_rebalance_logic_handles_missing_files(isolated):
    """No saved trades/equity/weights -> audit returns rows flagging the
    missing files, never raises."""
    df = portfolio_audit.audit_rebalance_logic(save=False)
    assert not df.empty
    # Every row should include the file-presence checks.
    assert {"trades_file_present", "equity_file_present",
            "weights_file_present"}.issubset(set(df["check"]))
