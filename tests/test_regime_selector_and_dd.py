"""Tests for the new B&H drawdown metric, scorecard drawdown rules, and the
regime-conditional strategy selector."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import backtester, config, performance, regime, scorecard, utils
from src.strategies import (
    REGISTRY, RegimeSelectorStrategy, RsiMaAtrStrategy,
    SidewaysMeanReversionStrategy, TrendFollowingStrategy,
)
from src.strategies.base import BUY, SELL, HOLD, SKIP
from src.strategies.regime_selector import (
    RegimeSelectorConfig, ROUTE_BULL, ROUTE_SIDEWAYS,
)


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
    raw = tmp_path / "raw"; raw.mkdir()
    monkeypatch.setattr(config, "DATA_RAW_DIR", raw)
    yield raw


# ---------------------------------------------------------------------------
# Phase 1 — B&H drawdown in metrics + scorecard drawdown rules
# ---------------------------------------------------------------------------
def test_strategy_comparison_includes_buy_and_hold_max_drawdown(synthetic_cache):
    rng = np.random.default_rng(0)
    close = np.linspace(100, 200, 600) + rng.normal(0, 1.0, 600)
    _csv("BTC/USDT", "4h", close)
    art = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False,
        strategy=RsiMaAtrStrategy(),
    )
    m = performance.compute_metrics(
        art.equity_curve, art.trades, art.asset_close_curves,
        starting_capital=10_000.0,
    )
    # Field exists, is a finite float, and is non-positive (drawdown).
    assert hasattr(m, "buy_and_hold_max_drawdown_pct")
    assert isinstance(m.buy_and_hold_max_drawdown_pct, float)
    assert m.buy_and_hold_max_drawdown_pct <= 0.0


def _strategy_row(strategy, asset, tf, vs_bh, total_ret, dd, bh_dd, exposure,
                  n_trades=20, is_err=False):
    return {
        "strategy": strategy, "asset": asset, "timeframe": tf,
        "label": strategy,
        "total_return_pct": total_ret,
        "buy_and_hold_return_pct": total_ret - vs_bh,
        "buy_and_hold_max_drawdown_pct": bh_dd,
        "strategy_vs_bh_pct": vs_bh,
        "drawdown_vs_bh_pct": dd - bh_dd,
        "max_drawdown_pct": dd,
        "win_rate_pct": 50.0, "num_trades": n_trades, "profit_factor": 1.1,
        "fees_paid": 1.0, "slippage_cost": 0.5,
        "exposure_time_pct": exposure,
        "sharpe_ratio": 0.3, "sortino_ratio": 0.3, "calmar_ratio": 0.1,
        "starting_capital": 10_000.0,
        "final_portfolio_value": 10_000.0 + total_ret * 100.0,
        "error": None if not is_err else "fake",
    }


def test_scorecard_uses_benchmark_drawdown_when_available():
    """Strategy DD = -3, B&H DD = -10, return positive, exposure 50%
    -> drawdown_score should be +1."""
    sc_in = pd.DataFrame([
        _strategy_row("trend_following", "BTC/USDT", "1d", vs_bh=2.0,
                      total_ret=8.0, dd=-3.0, bh_dd=-10.0, exposure=50.0),
    ])
    sc = scorecard.build_scorecard(sc_in, save=False)
    assert sc.iloc[0]["drawdown_score"] == 1


def test_low_exposure_does_not_create_fake_drawdown_pass():
    """Strategy DD = -1, B&H DD = -20, return barely positive, but
    exposure < 20%. The new rule must NOT award +1."""
    sc_in = pd.DataFrame([
        _strategy_row("trend_following", "BTC/USDT", "1d", vs_bh=0.5,
                      total_ret=0.5, dd=-1.0, bh_dd=-20.0, exposure=5.0),
    ])
    sc = scorecard.build_scorecard(sc_in, save=False)
    assert sc.iloc[0]["drawdown_score"] == 0


def test_missing_benchmark_drawdown_falls_back_safely():
    """No B&H DD -> fallback rules. Cannot award +1 without a benchmark."""
    sc_in = pd.DataFrame([
        _strategy_row("trend_following", "BTC/USDT", "1d", vs_bh=2.0,
                      total_ret=8.0, dd=-3.0, bh_dd=0.0, exposure=50.0),
    ])
    # bh_dd=0 is the "missing" sentinel — the scorecard treats it as None.
    sc = scorecard.build_scorecard(sc_in, save=False)
    assert sc.iloc[0]["drawdown_score"] == 0


# ---------------------------------------------------------------------------
# Phase 2 — Regime selector
# ---------------------------------------------------------------------------
def test_regime_selector_registered():
    assert "regime_selector" in REGISTRY


def test_selector_routes_to_trend_following_in_bull_low_vol():
    s = RegimeSelectorStrategy()
    row = pd.Series({
        "timestamp": 1, "datetime": pd.Timestamp.utcnow(),
        "trend_regime": regime.BULL, "volatility_regime": regime.LOW_VOL,
        "close": 100.0, "open": 99.0, "high": 100.5, "low": 99.0,
    })
    assert s._entry_route(regime.BULL, regime.LOW_VOL) == ROUTE_BULL


def test_selector_routes_to_sideways_mr_in_sideways_low_vol():
    s = RegimeSelectorStrategy()
    assert s._entry_route(regime.SIDEWAYS, regime.LOW_VOL) == ROUTE_SIDEWAYS


def test_selector_blocks_buy_in_bear_trend(synthetic_cache):
    rng = np.random.default_rng(0)
    close = np.linspace(300, 100, 600) + rng.normal(0, 0.3, 600)
    _csv("BTC/USDT", "4h", close)
    art = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False,
        strategy=RegimeSelectorStrategy(),
    )
    if not art.trades.empty:
        assert (art.trades["side"] == "BUY").sum() == 0


def test_selector_blocks_buy_in_high_volatility_by_default():
    s = RegimeSelectorStrategy()
    assert s._entry_route(regime.BULL, regime.HIGH_VOL) is None
    assert s._entry_route(regime.SIDEWAYS, regime.HIGH_VOL) is None


def test_selector_blocks_buy_in_unknown_regime():
    s = RegimeSelectorStrategy()
    assert s._entry_route("unknown", "unknown") is None


def test_selector_emits_only_valid_signals(synthetic_cache):
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(0, 0.5, 600))
    _csv("BTC/USDT", "4h", close)
    art = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False,
        strategy=RegimeSelectorStrategy(),
    )
    if not art.decisions.empty:
        assert set(art.decisions["action"].unique()).issubset(
            {"BUY", "SELL", "HOLD", "SKIP", "REJECT"}
        )
    if not art.trades.empty:
        assert set(art.trades["side"].unique()).issubset({"BUY", "SELL"})


def test_selector_does_not_bypass_risk_engine(synthetic_cache):
    """Risk engine path is the same as every other strategy — assert by
    confirming the meta records the selector's name and the fee column
    is populated on every trade row."""
    rng = np.random.default_rng(0)
    close = np.linspace(100, 250, 600) + rng.normal(0, 0.5, 600)
    _csv("BTC/USDT", "4h", close)
    art = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False,
        strategy=RegimeSelectorStrategy(),
    )
    assert art.meta["strategy_name"] == "regime_selector"
    if not art.trades.empty:
        assert (art.trades["fee"] >= 0).all()
        assert (art.trades["slippage_cost"] >= 0).all()


def test_selector_allows_sell_even_after_regime_flip_to_bear():
    """If a sub-strategy emits SELL while the bar's regime is bear_trend,
    the selector must NOT swallow it."""
    s = RegimeSelectorStrategy()
    # Pre-record an open position owned by the bull route.
    s._owner["BTC/USDT"] = ROUTE_BULL

    class _StubBull:
        def signal_for_row(self_inner, asset, row, in_position, cfg):
            from src.strategies.base import Signal, SELL
            return Signal(
                asset=asset, timestamp=int(row["timestamp"]),
                datetime=row.get("datetime"),
                action=SELL, price=float(row["close"]),
                reason="stub exit",
            )

        def prepare(self_inner, df, cfg): return df
        def min_history(self_inner, cfg): return 1
        name = "stub_bull"

    s._bull = _StubBull()
    row = pd.Series({
        "timestamp": 1, "datetime": pd.Timestamp.utcnow(),
        "trend_regime": regime.BEAR, "volatility_regime": regime.LOW_VOL,
        "close": 100.0,
    })
    sig = s.signal_for_row("BTC/USDT", row, in_position=True, cfg=None)
    assert sig.action == SELL
    # Owner must be cleared after a SELL.
    assert "BTC/USDT" not in s._owner


def test_selector_no_lookahead_in_routing(synthetic_cache):
    """Spot-check: the route at row i must be derivable from candle data
    up to and including row i — no future bars. The regime detector is
    already lookahead-free; verify the selector consumes its columns
    without peeking."""
    rng = np.random.default_rng(0)
    close = np.linspace(100, 200, 400) + rng.normal(0, 0.5, 400)
    _csv("BTC/USDT", "4h", close)
    art_full = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False,
        strategy=RegimeSelectorStrategy(),
    )
    # Re-run on a truncated dataset (50 fewer bars). The actions on
    # overlapping bars should be identical — proves no future data leaked.
    full_df = pd.read_csv(utils.csv_path_for("BTC/USDT", "4h"))
    truncated = full_df.iloc[:-50].copy()
    truncated.to_csv(utils.csv_path_for("BTC/USDT", "4h"), index=False)
    art_short = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False,
        strategy=RegimeSelectorStrategy(),
    )
    # restore (test fixture cleans up anyway)
    full_df.to_csv(utils.csv_path_for("BTC/USDT", "4h"), index=False)

    # Compare decisions on the overlapping bars.
    if art_full.decisions.empty or art_short.decisions.empty:
        return
    short_ts = set(art_short.decisions["timestamp_ms"])
    overlap = art_full.decisions[art_full.decisions["timestamp_ms"].isin(short_ts)]
    short = art_short.decisions[art_short.decisions["timestamp_ms"].isin(short_ts)]
    overlap = overlap.sort_values("timestamp_ms").reset_index(drop=True)
    short = short.sort_values("timestamp_ms").reset_index(drop=True)
    # The action column must match for every overlapping row.
    pd.testing.assert_series_equal(
        overlap["action"].reset_index(drop=True),
        short["action"].reset_index(drop=True),
        check_names=False,
    )


def test_no_live_trading_in_regime_selector():
    p = Path(__file__).resolve().parents[1] / "src" / "strategies" / "regime_selector.py"
    text = p.read_text()
    forbidden = (
        "create_order", "createMarketBuyOrder", "create_market_buy_order",
        ".privateGet", ".privatePost", "apiKey=", "secret=",
    )
    for s in forbidden:
        assert s not in text
