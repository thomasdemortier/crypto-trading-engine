"""Tests for the multi-asset momentum rotation research module."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config, portfolio_backtester as pb, portfolio_research, utils
from src.strategies.momentum_rotation import (
    MomentumRotationConfig, MomentumRotationStrategy,
    RandomRotationConfig, RandomRotationPlacebo,
)


# ---------------------------------------------------------------------------
# Synthetic universe helpers
# ---------------------------------------------------------------------------
def _csv_with_trend(symbol: str, timeframe: str, n: int = 800,
                    start_price: float = 100.0, end_price: float = 200.0,
                    seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    base = np.linspace(start_price, end_price, n)
    noise = rng.normal(0, 1.0, n)
    close = np.maximum(base + noise, 1.0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5, n))
    vol = rng.uniform(100, 1_000, n)
    ts = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")
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
def synthetic_universe(tmp_path, monkeypatch):
    raw = tmp_path / "raw"; raw.mkdir()
    res = tmp_path / "results"; res.mkdir()
    monkeypatch.setattr(config, "DATA_RAW_DIR", raw)
    monkeypatch.setattr(config, "RESULTS_DIR", res)
    # 5 strong uptrends + 2 weak — gives momentum strategy something to rank.
    _csv_with_trend("BTC/USDT", "1d", n=600, start_price=100, end_price=300, seed=1)
    _csv_with_trend("ETH/USDT", "1d", n=600, start_price=100, end_price=180, seed=2)
    _csv_with_trend("SOL/USDT", "1d", n=600, start_price=100, end_price=400, seed=3)
    _csv_with_trend("AVAX/USDT", "1d", n=600, start_price=100, end_price=120, seed=4)
    _csv_with_trend("LINK/USDT", "1d", n=600, start_price=100, end_price=160, seed=5)
    _csv_with_trend("XRP/USDT", "1d", n=600, start_price=100, end_price=110, seed=6)
    _csv_with_trend("DOGE/USDT", "1d", n=600, start_price=100, end_price=80, seed=7)
    yield {"raw": raw, "results": res}


# ---------------------------------------------------------------------------
# Phase 1 — universe loading + missing-asset graceful handling
# ---------------------------------------------------------------------------
def test_load_universe_handles_missing_assets(synthetic_universe):
    frames, missing = pb.load_universe(
        assets=["BTC/USDT", "ETH/USDT", "DOES_NOT_EXIST/USDT"],
        timeframe="1d",
    )
    assert "BTC/USDT" in frames and "ETH/USDT" in frames
    assert "DOES_NOT_EXIST/USDT" in missing


def test_load_universe_with_report_persists_availability(synthetic_universe):
    frames, avail = portfolio_research.load_universe_with_report(
        assets=["BTC/USDT", "MISSING/USDT"], timeframe="1d", save=True,
    )
    p = synthetic_universe["results"] / "portfolio_universe_availability.csv"
    assert p.exists()
    df = pd.read_csv(p)
    btc = df[df["asset"] == "BTC/USDT"].iloc[0]
    miss = df[df["asset"] == "MISSING/USDT"].iloc[0]
    assert bool(btc["available"]) is True
    assert bool(miss["available"]) is False
    assert int(miss["candle_count"]) == 0


# ---------------------------------------------------------------------------
# Phase 2 — momentum ranking (lookahead-free)
# ---------------------------------------------------------------------------
def test_momentum_ranking_uses_only_past_data(synthetic_universe):
    frames, _ = pb.load_universe(
        assets=["BTC/USDT", "SOL/USDT", "DOGE/USDT", "ETH/USDT", "AVAX/USDT"],
        timeframe="1d",
    )
    strat = MomentumRotationStrategy(
        MomentumRotationConfig(top_n=3, momentum_short_window=30,
                               momentum_long_window=90, min_assets_required=3),
    )
    # Pick a mid-history timestamp.
    asof = int(frames["BTC/USDT"]["timestamp"].iloc[400])
    full_weights = strat.target_weights(asof, frames, timeframe="1d")
    # Truncate every frame so future data is unavailable, then re-rank.
    truncated = {a: df[df["timestamp"] <= asof].copy()
                 for a, df in frames.items()}
    truncated_weights = strat.target_weights(asof, truncated, timeframe="1d")
    assert full_weights == truncated_weights


def test_momentum_picks_top_n_strongest(synthetic_universe):
    frames, _ = pb.load_universe(
        assets=["BTC/USDT", "SOL/USDT", "DOGE/USDT", "ETH/USDT", "AVAX/USDT"],
        timeframe="1d",
    )
    strat = MomentumRotationStrategy(
        MomentumRotationConfig(top_n=2, min_assets_required=3),
    )
    asof = int(frames["BTC/USDT"]["timestamp"].iloc[500])
    weights = strat.target_weights(asof, frames, timeframe="1d")
    assert len(weights) == 2
    # SOL had the steepest synthetic uptrend (100 -> 400) — must be selected.
    assert "SOL/USDT" in weights


def test_momentum_cash_filter_keeps_strategy_in_cash_when_btc_below_ma():
    """If BTC is in a strong downtrend so close < 200d MA, the cash filter
    should return an empty target."""
    rng = np.random.default_rng(0)
    n = 600
    # Constant downtrend so close < 200-day MA at the asof bar.
    btc = np.linspace(300, 100, n) + rng.normal(0, 0.3, n)
    eth = np.linspace(100, 200, n)
    ts = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")
    btc_df = pd.DataFrame({
        "timestamp": (ts.astype("datetime64[ns, UTC]").astype("int64") // 10**6).astype("int64"),
        "datetime": ts, "open": btc, "high": btc + 1, "low": btc - 1,
        "close": btc, "volume": 1.0,
    })
    eth_df = btc_df.copy()
    eth_df["close"] = eth
    eth_df["open"] = eth
    eth_df["high"] = eth + 1
    eth_df["low"] = eth - 1
    frames = {"BTC/USDT": btc_df, "ETH/USDT": eth_df}
    strat = MomentumRotationStrategy(
        MomentumRotationConfig(top_n=2, min_assets_required=2),
    )
    asof = int(btc_df["timestamp"].iloc[-10])
    weights = strat.target_weights(asof, frames, timeframe="1d")
    assert weights == {}, f"cash filter should fire, got {weights}"


# ---------------------------------------------------------------------------
# Phase 3 — portfolio backtester mechanics
# ---------------------------------------------------------------------------
def test_portfolio_backtester_runs_and_produces_artifacts(synthetic_universe):
    frames, _ = pb.load_universe(
        assets=["BTC/USDT", "SOL/USDT", "ETH/USDT", "AVAX/USDT", "LINK/USDT"],
        timeframe="1d",
    )
    art = pb.run_portfolio_backtest(
        portfolio_strategy=MomentumRotationStrategy(
            MomentumRotationConfig(top_n=2, min_assets_required=3),
        ),
        asset_frames=frames, timeframe="1d",
        cfg=pb.PortfolioBacktestConfig(starting_capital=10_000.0),
        save=False,
    )
    assert not art.equity_curve.empty
    assert not art.weights_history.empty
    assert "equity" in art.equity_curve.columns
    assert "cash" in art.equity_curve.columns
    # Equity is bounded below by 0 (long-only, no leverage).
    assert (art.equity_curve["equity"] >= 0).all()


def test_portfolio_weights_never_imply_leverage(synthetic_universe):
    """Even if a strategy asks for sum > 1, the backtester normalises down."""
    class _Greedy:
        name = "greedy"
        def target_weights(self, *, asof_ts_ms, asset_frames, timeframe):
            return {"BTC/USDT": 0.6, "ETH/USDT": 0.6, "SOL/USDT": 0.6}

    frames, _ = pb.load_universe(
        assets=["BTC/USDT", "ETH/USDT", "SOL/USDT"], timeframe="1d",
    )
    art = pb.run_portfolio_backtest(
        portfolio_strategy=_Greedy(), asset_frames=frames, timeframe="1d",
        cfg=pb.PortfolioBacktestConfig(starting_capital=10_000.0),
        save=False,
    )
    # Cash should never go negative (we never buy more than we hold).
    assert (art.equity_curve["cash"] >= -1e-6).all()


def test_weekly_rebalance_fires_at_least_once_per_week(synthetic_universe):
    frames, _ = pb.load_universe(
        assets=["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT"],
        timeframe="1d",
    )
    art = pb.run_portfolio_backtest(
        portfolio_strategy=MomentumRotationStrategy(
            MomentumRotationConfig(top_n=2, min_assets_required=3,
                                   rebalance_frequency="weekly"),
        ),
        asset_frames=frames, timeframe="1d",
        cfg=pb.PortfolioBacktestConfig(rebalance_frequency="weekly"),
        save=False,
    )
    # Span is ~600 days = ~85 weeks. We should see ≥ 50 rebalance attempts.
    n_rebalance_intents = int((~art.weights_history["filled"]).sum())
    assert n_rebalance_intents >= 50


def test_monthly_rebalance_fires_per_month(synthetic_universe):
    frames, _ = pb.load_universe(
        assets=["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT"],
        timeframe="1d",
    )
    art = pb.run_portfolio_backtest(
        portfolio_strategy=MomentumRotationStrategy(
            MomentumRotationConfig(top_n=2, min_assets_required=3,
                                   rebalance_frequency="monthly"),
        ),
        asset_frames=frames, timeframe="1d",
        cfg=pb.PortfolioBacktestConfig(rebalance_frequency="monthly"),
        save=False,
    )
    n_rebalance_intents = int((~art.weights_history["filled"]).sum())
    # 600 days ≈ 20 months, give ourselves slack.
    assert 15 <= n_rebalance_intents <= 25


def test_fees_and_slippage_are_charged(synthetic_universe):
    """A pair of identical backtests with different fee_pct must produce
    different final equities."""
    frames, _ = pb.load_universe(
        assets=["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT"],
        timeframe="1d",
    )
    strat = MomentumRotationStrategy(
        MomentumRotationConfig(top_n=2, min_assets_required=3),
    )
    art_low = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames, timeframe="1d",
        cfg=pb.PortfolioBacktestConfig(fee_pct=0.0001, slippage_pct=0.0),
        save=False,
    )
    art_high = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames, timeframe="1d",
        cfg=pb.PortfolioBacktestConfig(fee_pct=0.005, slippage_pct=0.005),
        save=False,
    )
    final_low = float(art_low.equity_curve["equity"].iloc[-1])
    final_high = float(art_high.equity_curve["equity"].iloc[-1])
    assert final_high < final_low, (
        f"high-fee backtest should end below low-fee, got "
        f"high={final_high}, low={final_low}"
    )


# ---------------------------------------------------------------------------
# Phase 5 — placebo
# ---------------------------------------------------------------------------
def test_random_placebo_is_reproducible_with_fixed_seed(synthetic_universe):
    frames, _ = pb.load_universe(
        assets=["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT"],
        timeframe="1d",
    )
    cfg_a = RandomRotationConfig(top_n=2, seed=7)
    cfg_b = RandomRotationConfig(top_n=2, seed=7)
    p_a = RandomRotationPlacebo(cfg_a)
    p_b = RandomRotationPlacebo(cfg_b)
    art_a = pb.run_portfolio_backtest(
        portfolio_strategy=p_a, asset_frames=frames, timeframe="1d",
        cfg=pb.PortfolioBacktestConfig(), save=False,
    )
    art_b = pb.run_portfolio_backtest(
        portfolio_strategy=p_b, asset_frames=frames, timeframe="1d",
        cfg=pb.PortfolioBacktestConfig(), save=False,
    )
    pd.testing.assert_series_equal(
        art_a.equity_curve["equity"].reset_index(drop=True),
        art_b.equity_curve["equity"].reset_index(drop=True),
        check_names=False,
    )


def test_placebo_audit_runs_multiple_seeds(synthetic_universe):
    df = portfolio_research.portfolio_placebo(
        assets=["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT"],
        timeframe="1d", seeds=(0, 1, 2, 3, 4), save=False,
    )
    # First row is summary, rest are per-seed metrics.
    assert "placebo_median_return_pct" in df.columns
    per_seed = df[df["strategy"] == "placebo_seed_runs"]
    assert len(per_seed) == 5


# ---------------------------------------------------------------------------
# Phase 4 — walk-forward
# ---------------------------------------------------------------------------
def test_portfolio_walk_forward_produces_oos_windows(synthetic_universe):
    df = portfolio_research.portfolio_walk_forward(
        assets=["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT"],
        timeframe="1d",
        in_sample_days=180, oos_days=60, step_days=60, save=False,
    )
    # Synthetic span is ~600 days — we should get a few OOS windows.
    assert not df.empty
    assert "oos_return_pct" in df.columns
    assert "beats_btc" in df.columns
    assert "beats_basket" in df.columns


# ---------------------------------------------------------------------------
# Phase 6 — scorecard
# ---------------------------------------------------------------------------
def test_scorecard_marks_weak_strategy_as_fail(synthetic_universe):
    """Weak walk-forward (all losses, no beats) -> verdict FAIL or
    INCONCLUSIVE, NEVER PASS."""
    wf = pd.DataFrame([
        {"window": i, "oos_return_pct": -2.0, "oos_max_drawdown_pct": -5.0,
         "oos_sharpe": -0.5, "btc_oos_return_pct": 1.0,
         "basket_oos_return_pct": 0.5,
         "beats_btc": False, "beats_basket": False, "profitable": False,
         "n_rebalances": 4, "n_trades": 8, "avg_holdings": 2.0,
         "turnover": 0.1, "error": None}
        for i in range(1, 8)
    ])
    plac = pd.DataFrame([{
        "strategy": "momentum_rotation",
        "strategy_return_pct": -10.0,
        "strategy_max_drawdown_pct": -15.0,
        "strategy_sharpe": -0.5, "n_seeds": 5,
        "placebo_median_return_pct": 5.0,
        "placebo_median_drawdown_pct": -8.0,
        "placebo_p75_drawdown_pct": -10.0,
        "strategy_beats_median_return": False,
        "strategy_beats_median_drawdown": False,
    }])
    sc = portfolio_research.portfolio_scorecard(wf, plac, save=False)
    verdict = sc.iloc[0]["verdict"]
    assert verdict in (portfolio_research.PORTFOLIO_FAIL,
                       portfolio_research.PORTFOLIO_INCONCLUSIVE)
    assert verdict != portfolio_research.PORTFOLIO_PASS


def test_scorecard_does_not_let_placebo_win():
    """Even a stellar synthetic walk-forward must not PASS if it doesn't
    beat the placebo median."""
    wf = pd.DataFrame([
        {"window": i, "oos_return_pct": 8.0, "oos_max_drawdown_pct": -3.0,
         "oos_sharpe": 1.5, "btc_oos_return_pct": 1.0,
         "basket_oos_return_pct": 1.5,
         "beats_btc": True, "beats_basket": True, "profitable": True,
         "n_rebalances": 6, "n_trades": 12, "avg_holdings": 3.0,
         "turnover": 0.5, "error": None}
        for i in range(1, 9)
    ])
    plac = pd.DataFrame([{
        "strategy": "momentum_rotation",
        "strategy_return_pct": 50.0,
        "strategy_max_drawdown_pct": -10.0,
        "strategy_sharpe": 1.5, "n_seeds": 20,
        "placebo_median_return_pct": 80.0,  # placebo beat strategy
        "placebo_median_drawdown_pct": -5.0,
        "placebo_p75_drawdown_pct": -8.0,
        "strategy_beats_median_return": False,
        "strategy_beats_median_drawdown": False,
    }])
    sc = portfolio_research.portfolio_scorecard(wf, plac, save=False)
    verdict = sc.iloc[0]["verdict"]
    assert verdict != portfolio_research.PORTFOLIO_PASS


def test_scorecard_inconclusive_for_too_few_windows():
    wf = pd.DataFrame([
        {"window": 1, "oos_return_pct": 10.0, "oos_max_drawdown_pct": -2.0,
         "btc_oos_return_pct": 5.0, "basket_oos_return_pct": 4.0,
         "beats_btc": True, "beats_basket": True, "profitable": True,
         "n_rebalances": 4, "n_trades": 4, "avg_holdings": 2.0,
         "turnover": 0.2, "error": None}
    ])
    plac = pd.DataFrame()
    sc = portfolio_research.portfolio_scorecard(wf, plac, save=False)
    assert sc.iloc[0]["verdict"] == portfolio_research.PORTFOLIO_INCONCLUSIVE


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------
def test_no_live_trading_in_portfolio_modules():
    src = Path(__file__).resolve().parents[1] / "src"
    targets = [
        src / "portfolio_backtester.py",
        src / "portfolio_research.py",
        src / "strategies" / "momentum_rotation.py",
    ]
    forbidden = (
        "create_order", "createMarketBuyOrder", "create_market_buy_order",
        ".privateGet", ".privatePost", "apiKey=", "secret=",
    )
    for p in targets:
        text = p.read_text()
        for s in forbidden:
            assert s not in text, f"forbidden token {s!r} in {p.name}"
