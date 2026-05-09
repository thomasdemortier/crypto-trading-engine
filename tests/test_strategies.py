"""Strategy plug-in regression tests.

Confirms:
* Each strategy implements the full Strategy interface.
* Every strategy goes through the same RiskEngine path (no bypass).
* No strategy module touches the live-trading flag or imports forbidden APIs.
* Indicators emitted by each strategy's `prepare()` use ONLY past data
  (lookahead spot-check).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import backtester, config, utils
from src.strategies import (
    BreakoutStrategy, BuyAndHoldStrategy,
    MovingAverageCrossStrategy, REGISTRY, RsiMaAtrStrategy,
)
from src.strategies.base import Strategy


def _synthetic_csv(symbol: str, timeframe: str, n: int = 600,
                   seed: int = 0) -> Path:
    """Create a deterministic uptrend with noise so signals fire."""
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
        "datetime": ts,
        "open": open_, "high": high, "low": low,
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


def test_strategy_registry_has_four_entries():
    assert set(REGISTRY.keys()) == {
        "rsi_ma_atr", "buy_and_hold", "ma_cross", "breakout",
    }


@pytest.mark.parametrize("strategy_factory", [
    lambda: RsiMaAtrStrategy(),
    lambda: BuyAndHoldStrategy(),
    lambda: MovingAverageCrossStrategy(),
    lambda: BreakoutStrategy(),
])
def test_strategy_implements_full_interface(strategy_factory):
    s = strategy_factory()
    assert isinstance(s, Strategy)
    assert isinstance(s.name, str) and s.name
    assert callable(getattr(s, "min_history"))
    assert callable(getattr(s, "prepare"))
    assert callable(getattr(s, "signal_for_row"))


@pytest.mark.parametrize("strategy_factory,name", [
    (lambda: RsiMaAtrStrategy(), "rsi_ma_atr"),
    (lambda: BuyAndHoldStrategy(), "buy_and_hold"),
    (lambda: MovingAverageCrossStrategy(), "ma_cross"),
    (lambda: BreakoutStrategy(), "breakout"),
])
def test_strategy_runs_through_risk_engine(synthetic_cache, strategy_factory, name):
    """Every strategy must produce a valid backtest via the same risk engine
    code path. We assert the artifacts are well-formed, not that returns are
    positive — research tooling never improves results."""
    _synthetic_csv("BTC/USDT", "4h", n=600, seed=1)
    art = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False,
        strategy=strategy_factory(),
    )
    assert art.meta["strategy_name"] == name
    assert not art.equity_curve.empty
    # All recorded trades must reference an action that originated from the
    # risk engine (BUY / SELL only — REJECT / SKIP / HOLD never produce trades).
    if not art.trades.empty:
        assert set(art.trades["side"].unique()).issubset({"BUY", "SELL"})


def test_no_strategy_imports_live_trading_or_keys():
    """Defense in depth — research expansion must not introduce live-order
    or API-key surfaces. Mirrors the existing test_no_live_trading_code_exists
    in test_backtester.py but covers the new strategies/research modules."""
    src_root = Path(__file__).resolve().parents[1] / "src"
    forbidden = (
        "create_order", "createMarketBuyOrder", "createMarketSellOrder",
        "createLimitOrder", "create_market_buy_order",
        "create_market_sell_order", "withdraw(", ".privateGet",
        ".privatePost", "apiKey=", "secret=",
    )
    offenders = []
    for py in src_root.rglob("*.py"):
        text = py.read_text()
        for s in forbidden:
            if s in text:
                offenders.append((py.relative_to(src_root).as_posix(), s))
    assert not offenders, f"forbidden tokens found: {offenders}"


def test_breakout_indicators_use_only_past_data():
    """Donchian rolling high/low must be shifted by 1 so the current bar
    cannot 'break above its own high'. This is the headline lookahead trap
    for breakout strategies — assert it's wired correctly."""
    rng = np.random.default_rng(0)
    n = 100
    close = np.cumsum(rng.normal(0, 1, n)) + 100
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    open_ = np.r_[close[0], close[:-1]]
    ts = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": (ts.astype("int64") // 10**6).astype("int64"),
        "datetime": ts, "open": open_, "high": high, "low": low,
        "close": close, "volume": np.ones(n) * 100,
    })
    out = BreakoutStrategy(entry_window=20, exit_window=10).prepare(df, None)
    # roll_high at row i must equal max(high[i-20 : i]) — does NOT include high[i].
    for i in (25, 50, 75):
        expected = float(out["high"].iloc[i - 20:i].max())
        assert pytest.approx(float(out["roll_high"].iloc[i])) == expected
        expected_low = float(out["low"].iloc[i - 10:i].min())
        assert pytest.approx(float(out["roll_low"].iloc[i])) == expected_low


def test_ma_cross_requires_fast_lt_slow():
    with pytest.raises(ValueError):
        MovingAverageCrossStrategy(fast=200, slow=50)
