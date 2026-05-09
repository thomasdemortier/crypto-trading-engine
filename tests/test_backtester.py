"""End-to-end backtester sanity tests using synthetic data only.

These tests do NOT touch the network; they write fake CSVs into the
project's data/raw directory, run the backtester, then clean up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import backtester, config, utils


def _make_synthetic_csv(symbol: str, timeframe: str, n: int = 600,
                        seed: int = 0) -> Path:
    """Create a deterministic synthetic OHLCV CSV in the cache. The price
    has a clear uptrend so the strategy actually triggers some buys."""
    rng = np.random.default_rng(seed)
    base = np.linspace(100.0, 200.0, n)
    noise = rng.normal(0, 1.5, n)
    close = base + noise
    close = np.maximum(close, 1.0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5, n))
    vol = rng.uniform(100, 1_000, n)
    ts = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": (ts.astype("datetime64[ns, UTC]").astype("int64") // 10**6).astype("int64"),
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
    # Redirect data/raw to tmp_path so we don't pollute the user's cache.
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(config, "DATA_RAW_DIR", raw)
    yield raw


def test_backtest_produces_nonempty_equity_curve(synthetic_cache):
    _make_synthetic_csv("BTC/USDT", "4h", n=600, seed=1)
    _make_synthetic_csv("ETH/USDT", "4h", n=600, seed=2)
    art = backtester.run_backtest(
        assets=["BTC/USDT", "ETH/USDT"], timeframe="4h", save=False,
    )
    assert not art.equity_curve.empty
    assert art.equity_curve["equity"].iloc[0] > 0
    assert art.equity_curve["equity"].iloc[-1] > 0
    # Decisions are always logged (at least HOLDs after warmup).
    assert not art.decisions.empty


def test_no_live_trading_code_exists():
    """Defense in depth: the codebase must not import private ccxt order
    methods or define a function named like create_order."""
    src_root = Path(__file__).resolve().parents[1] / "src"
    forbidden_substrings = (
        "create_order", "createMarketBuyOrder", "createMarketSellOrder",
        "createLimitOrder", "create_market_buy_order",
        "create_market_sell_order", "withdraw(", ".privateGet",
        ".privatePost", "apiKey=", "secret=",
    )
    offenders = []
    for py in src_root.rglob("*.py"):
        text = py.read_text()
        for s in forbidden_substrings:
            if s in text:
                offenders.append((py.name, s))
    assert not offenders, f"forbidden order-placement strings found: {offenders}"


def test_live_trading_flag_is_false_by_default():
    assert config.LIVE_TRADING_ENABLED is False
    # And the guard refuses to bypass.
    from src import utils as u
    u.assert_paper_only()  # must not raise when False
