"""Tests for the drawdown-targeted BTC research orchestrator."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config, drawdown_targeted_research, utils
from src.drawdown_targeted_research import (
    DrawdownTargetedPlacebo, DrawdownTargetedPlaceboConfig,
)


def _csv_with_trend(symbol: str, timeframe: str = "1d", n: int = 800,
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
    ts = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC")
    df = pd.DataFrame({
        "timestamp": (ts.astype("datetime64[ns, UTC]").astype("int64")
                       // 10**6).astype("int64"),
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
    _csv_with_trend("BTC/USDT", n=900, start_price=100, end_price=400, seed=1)
    _csv_with_trend("ETH/USDT", n=900, start_price=100, end_price=180, seed=2)
    _csv_with_trend("SOL/USDT", n=900, start_price=100, end_price=300, seed=3)
    _csv_with_trend("AVAX/USDT", n=900, start_price=100, end_price=120, seed=4)
    _csv_with_trend("LINK/USDT", n=900, start_price=100, end_price=160, seed=5)
    _csv_with_trend("XRP/USDT", n=900, start_price=100, end_price=110, seed=6)
    _csv_with_trend("DOGE/USDT", n=900, start_price=100, end_price=80, seed=7)
    _csv_with_trend("ADA/USDT", n=900, start_price=100, end_price=140, seed=8)
    _csv_with_trend("LTC/USDT", n=900, start_price=100, end_price=120, seed=9)
    _csv_with_trend("BNB/USDT", n=900, start_price=100, end_price=200, seed=10)
    yield {"raw": raw, "results": res}


# ---------------------------------------------------------------------------
# Placebo properties
# ---------------------------------------------------------------------------
def test_placebo_only_uses_strategy_buckets():
    btc_close = np.linspace(100, 200, 100)
    ts = pd.date_range("2022-01-01", periods=100, freq="1D", tz="UTC")
    btc = pd.DataFrame({
        "timestamp": (ts.astype("datetime64[ns, UTC]").astype("int64")
                       // 10**6).astype("int64"),
        "datetime": ts, "open": btc_close, "high": btc_close,
        "low": btc_close, "close": btc_close, "volume": np.ones(100),
    })
    plac = DrawdownTargetedPlacebo(DrawdownTargetedPlaceboConfig(seed=0))
    seen = set()
    for i in range(50, 100):
        asof = int(btc["timestamp"].iloc[i])
        w = plac.target_weights(asof, {"BTC/USDT": btc}, "1d")
        # Allowed values: 1.0, 0.7, 0.4, 0.2 (and {} when w==0 — but 0 is
        # not in the bucket list). All emitted weights must be in the set.
        for v in w.values():
            seen.add(round(float(v), 4))
    assert seen.issubset({1.0, 0.7, 0.4, 0.2})


def test_placebo_seeds_are_reproducible():
    btc_close = np.linspace(100, 200, 100)
    ts = pd.date_range("2022-01-01", periods=100, freq="1D", tz="UTC")
    btc = pd.DataFrame({
        "timestamp": (ts.astype("datetime64[ns, UTC]").astype("int64")
                       // 10**6).astype("int64"),
        "datetime": ts, "open": btc_close, "high": btc_close,
        "low": btc_close, "close": btc_close, "volume": np.ones(100),
    })

    def _run(seed):
        p = DrawdownTargetedPlacebo(DrawdownTargetedPlaceboConfig(seed=seed))
        return [p.target_weights(int(btc["timestamp"].iloc[i]),
                                  {"BTC/USDT": btc}, "1d")
                for i in range(50, 100)]

    assert _run(7) == _run(7)
    # Different seeds usually differ — sanity check: at least one bar differs.
    assert _run(7) != _run(8)


# ---------------------------------------------------------------------------
# Single-window backtest
# ---------------------------------------------------------------------------
def test_run_drawdown_targeted_btc_produces_artifacts(synthetic_universe):
    out = drawdown_targeted_research.run_drawdown_targeted_btc(
        universe=("BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT",
                  "LINK/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT",
                  "LTC/USDT", "BNB/USDT"),
        timeframe="1d", save=True,
    )
    assert out["ok"] is True
    art = out["artifacts"]
    assert not art.equity_curve.empty
    cmp_p = synthetic_universe["results"] / "drawdown_targeted_btc_comparison.csv"
    assert cmp_p.exists()
    cmp_df = pd.read_csv(cmp_p)
    assert "drawdown_targeted_btc_allocator" in cmp_df["strategy"].values
    assert "BTC_buy_and_hold" in cmp_df["strategy"].values
    assert "equal_weight_basket" in cmp_df["strategy"].values
    assert "simple_momentum" in cmp_df["strategy"].values


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def test_walk_forward_caps_at_max_windows(synthetic_universe):
    df = drawdown_targeted_research.drawdown_targeted_btc_walk_forward(
        universe=("BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT",
                  "LINK/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT",
                  "LTC/USDT", "BNB/USDT"),
        timeframe="1d", in_sample_days=120, oos_days=60, step_days=60,
        max_windows=3, save=True,
    )
    assert not df.empty
    assert len(df) <= 3
    cols = ("oos_return_pct", "btc_oos_return_pct", "basket_oos_return_pct",
            "simple_oos_return_pct", "beats_btc", "beats_basket",
            "beats_simple_momentum", "n_rebalances")
    for c in cols:
        assert c in df.columns


# ---------------------------------------------------------------------------
# Placebo + scorecard
# ---------------------------------------------------------------------------
def test_placebo_and_scorecard_pipeline(synthetic_universe):
    seeds = tuple(range(5))
    drawdown_targeted_research.run_drawdown_targeted_btc(
        universe=("BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT",
                  "LINK/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT",
                  "LTC/USDT", "BNB/USDT"),
        timeframe="1d", save=True,
    )
    wf = drawdown_targeted_research.drawdown_targeted_btc_walk_forward(
        universe=("BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT",
                  "LINK/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT",
                  "LTC/USDT", "BNB/USDT"),
        timeframe="1d", in_sample_days=120, oos_days=60, step_days=60,
        max_windows=6, save=True,
    )
    plac = drawdown_targeted_research.drawdown_targeted_btc_placebo(
        universe=("BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT",
                  "LINK/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT",
                  "LTC/USDT", "BNB/USDT"),
        timeframe="1d", seeds=seeds, save=True,
    )
    assert not wf.empty
    assert not plac.empty
    summary = plac.iloc[0]
    assert summary["n_seeds"] == len(seeds)

    sc = drawdown_targeted_research.drawdown_targeted_btc_scorecard(
        walk_forward_df=wf, placebo_df=plac, save=True,
    )
    assert not sc.empty
    row = sc.iloc[0]
    # Verdict is one of the four canonical labels.
    assert row["verdict"] in {"PASS", "WATCHLIST", "FAIL", "INCONCLUSIVE"}
    # Scorecard records every required check name.
    reason = str(row["reason"])
    for check in ("beats_btc_oos_majority", "beats_basket_oos_majority",
                  "beats_simple_momentum_oos_majority",
                  "beats_placebo_median",
                  "oos_stability_at_least_60",
                  "dd_within_btc_gap_20pp",
                  "at_least_10_rebalances"):
        assert check in reason


def test_scorecard_inconclusive_on_no_walk_forward(tmp_path, monkeypatch):
    res = tmp_path / "results"; res.mkdir()
    monkeypatch.setattr(config, "RESULTS_DIR", res)
    sc = drawdown_targeted_research.drawdown_targeted_btc_scorecard(
        walk_forward_df=pd.DataFrame(), placebo_df=pd.DataFrame(),
        save=False,
    )
    assert not sc.empty
    assert sc.iloc[0]["verdict"] == "INCONCLUSIVE"
