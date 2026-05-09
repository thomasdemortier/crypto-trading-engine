"""Tests for the cross-asset regime signals + regime-aware momentum
rotation portfolio strategy."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import (
    config, crypto_regime_signals as crs, portfolio_backtester as pb,
    portfolio_research, utils,
)
from src.strategies.regime_aware_momentum_rotation import (
    RegimeAwareMomentumConfig, RegimeAwareMomentumRotationStrategy,
    RegimeAwareRandomConfig, RegimeAwareRandomPlacebo,
)


# ---------------------------------------------------------------------------
# Fixtures
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


def _seed_universe(strong_alts: bool = True) -> None:
    """Seed BTC + ETH + 5 alts. `strong_alts=True` produces a window
    where the alt basket comfortably outperforms BTC. `False` produces
    a BTC-leadership window."""
    n = 600
    if strong_alts:
        _csv("BTC/USDT", "1d", np.linspace(100, 200, n))
        _csv("ETH/USDT", "1d", np.linspace(100, 350, n))
        _csv("SOL/USDT", "1d", np.linspace(100, 500, n))
        _csv("AVAX/USDT", "1d", np.linspace(100, 300, n))
        _csv("LINK/USDT", "1d", np.linspace(100, 280, n))
        _csv("XRP/USDT", "1d", np.linspace(100, 220, n))
        _csv("DOGE/USDT", "1d", np.linspace(100, 240, n))
    else:
        _csv("BTC/USDT", "1d", np.linspace(100, 400, n))
        _csv("ETH/USDT", "1d", np.linspace(100, 130, n))
        _csv("SOL/USDT", "1d", np.linspace(100, 110, n))
        _csv("AVAX/USDT", "1d", np.linspace(100, 105, n))
        _csv("LINK/USDT", "1d", np.linspace(100, 102, n))
        _csv("XRP/USDT", "1d", np.linspace(100, 110, n))
        _csv("DOGE/USDT", "1d", np.linspace(100, 95, n))


# ---------------------------------------------------------------------------
# Phase A1 — crypto_regime_signals
# ---------------------------------------------------------------------------
def test_signals_eth_btc_ratio_is_eth_over_btc(isolated):
    _seed_universe(strong_alts=True)
    df = crs.compute_regime_signals(timeframe="1d", save=False)
    last = df.iloc[-1]
    btc_close = pd.read_csv(utils.csv_path_for("BTC/USDT", "1d"))["close"].iloc[-1]
    eth_close = pd.read_csv(utils.csv_path_for("ETH/USDT", "1d"))["close"].iloc[-1]
    assert last["eth_btc_ratio"] == pytest.approx(eth_close / btc_close,
                                                  rel=1e-9)


def test_signals_alt_basket_excludes_btc(isolated):
    """Alt basket return should NOT include BTC. We can detect this by
    comparing two universes that differ only in BTC's price path."""
    n = 600
    # First, plant a 'control' universe with BTC flat.
    _csv("BTC/USDT", "1d", np.full(n, 100.0))
    _csv("ETH/USDT", "1d", np.linspace(100, 200, n))
    _csv("SOL/USDT", "1d", np.linspace(100, 200, n))
    _csv("AVAX/USDT", "1d", np.linspace(100, 200, n))
    _csv("LINK/USDT", "1d", np.linspace(100, 200, n))
    _csv("XRP/USDT", "1d", np.linspace(100, 200, n))
    df_flat_btc = crs.compute_regime_signals(timeframe="1d", save=False)
    alt_basket_a = float(df_flat_btc["alt_basket_return_90d_pct"].dropna().iloc[-1])
    # Now spike BTC drastically — alt basket value at last bar must be unchanged.
    _csv("BTC/USDT", "1d", np.linspace(100, 1000, n))
    df_huge_btc = crs.compute_regime_signals(timeframe="1d", save=False)
    alt_basket_b = float(df_huge_btc["alt_basket_return_90d_pct"].dropna().iloc[-1])
    assert alt_basket_a == pytest.approx(alt_basket_b, rel=1e-6), (
        f"alt basket should be invariant to BTC: {alt_basket_a} vs {alt_basket_b}"
    )


def test_signals_breadth_calculation_correct(isolated):
    """Plant 5 assets where exactly 3 are above their 100d MA at the
    last bar. `pct_assets_above_100d_ma` must equal 60%."""
    n = 400
    # 3 uptrending: closes well above their rolling mean.
    _csv("BTC/USDT", "1d", np.linspace(100, 300, n))
    _csv("ETH/USDT", "1d", np.linspace(100, 300, n))
    _csv("SOL/USDT", "1d", np.linspace(100, 300, n))
    # 2 downtrending: closes below rolling mean.
    _csv("AVAX/USDT", "1d", np.linspace(300, 100, n))
    _csv("LINK/USDT", "1d", np.linspace(300, 100, n))
    df = crs.compute_regime_signals(timeframe="1d", save=False)
    last = float(df["pct_assets_above_100d_ma"].dropna().iloc[-1])
    assert last == pytest.approx(60.0, abs=0.01)


def test_signals_no_lookahead(isolated):
    """Each row's signal at row i must equal the signal computed on
    df.iloc[:i+1] alone — the canonical no-lookahead spot-check."""
    _seed_universe(strong_alts=True)
    full = crs.compute_regime_signals(timeframe="1d", save=False)
    # Truncate the underlying CSVs to N-50 bars and recompute.
    for asset in ("BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT",
                  "LINK/USDT", "XRP/USDT", "DOGE/USDT"):
        p = utils.csv_path_for(asset, "1d")
        df = pd.read_csv(p)
        df.iloc[:-50].to_csv(p, index=False)
    truncated = crs.compute_regime_signals(timeframe="1d", save=False)
    # Compare overlapping rows by timestamp.
    overlap_ts = sorted(set(full["timestamp"]) & set(truncated["timestamp"]))
    full_idx = full.set_index("timestamp")
    trunc_idx = truncated.set_index("timestamp")
    for ts in overlap_ts[200:][:50]:  # skip warmup, sample 50 rows
        a = full_idx.loc[ts, "risk_state"]
        b = trunc_idx.loc[ts, "risk_state"]
        assert a == b, f"lookahead at ts {ts}: full={a}, truncated={b}"


def test_signals_defensive_when_btc_below_200d(isolated):
    """Synthetic downtrend in BTC — signals should label most evaluable
    rows as DEFENSIVE."""
    n = 600
    _csv("BTC/USDT", "1d", np.linspace(400, 100, n))
    _csv("ETH/USDT", "1d", np.linspace(100, 120, n))
    _csv("SOL/USDT", "1d", np.linspace(100, 110, n))
    _csv("AVAX/USDT", "1d", np.linspace(100, 105, n))
    _csv("LINK/USDT", "1d", np.linspace(100, 102, n))
    _csv("XRP/USDT", "1d", np.linspace(100, 110, n))
    _csv("DOGE/USDT", "1d", np.linspace(100, 95, n))
    df = crs.compute_regime_signals(timeframe="1d", save=False)
    last = df["risk_state"].iloc[-1]
    assert last == crs.DEFENSIVE


def test_signals_regime_labels_are_deterministic(isolated):
    """Same data twice -> identical risk_state column."""
    _seed_universe(strong_alts=True)
    a = crs.compute_regime_signals(timeframe="1d", save=False)
    b = crs.compute_regime_signals(timeframe="1d", save=False)
    pd.testing.assert_series_equal(
        a["risk_state"].reset_index(drop=True),
        b["risk_state"].reset_index(drop=True),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# Phase A2 — regime-aware strategy routing
# ---------------------------------------------------------------------------
def _make_signals_with(risk_state: str, ts_ms: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "timestamp": ts_ms, "datetime": pd.Timestamp(ts_ms, unit="ms", tz="UTC"),
        "risk_state": risk_state,
    }])


def test_strategy_holds_cash_in_defensive_regime(isolated):
    _seed_universe(strong_alts=True)
    frames, _ = pb.load_universe(
        ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT"],
        timeframe="1d",
    )
    asof = int(frames["BTC/USDT"]["timestamp"].iloc[-1])
    sig = _make_signals_with(crs.DEFENSIVE, asof)
    strat = RegimeAwareMomentumRotationStrategy(signals_df=sig)
    weights = strat.target_weights(asof, frames, timeframe="1d")
    assert weights == {}, f"defensive must produce cash, got {weights}"


def test_strategy_holds_btc_only_in_btc_leadership(isolated):
    _seed_universe(strong_alts=True)
    frames, _ = pb.load_universe(
        ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT"],
        timeframe="1d",
    )
    asof = int(frames["BTC/USDT"]["timestamp"].iloc[-1])
    sig = _make_signals_with(crs.BTC_LEADERSHIP, asof)
    strat = RegimeAwareMomentumRotationStrategy(signals_df=sig)
    weights = strat.target_weights(asof, frames, timeframe="1d")
    assert weights == {"BTC/USDT": 1.0}


def test_strategy_holds_btc_only_in_mixed_or_unknown(isolated):
    _seed_universe(strong_alts=True)
    frames, _ = pb.load_universe(
        ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT"],
        timeframe="1d",
    )
    asof = int(frames["BTC/USDT"]["timestamp"].iloc[-1])
    for label in (crs.MIXED, crs.UNKNOWN):
        sig = _make_signals_with(label, asof)
        strat = RegimeAwareMomentumRotationStrategy(signals_df=sig)
        weights = strat.target_weights(asof, frames, timeframe="1d")
        assert weights == {"BTC/USDT": 1.0}, (
            f"{label} should be BTC-only, got {weights}"
        )


def test_strategy_holds_top_n_in_alt_risk_on(isolated):
    _seed_universe(strong_alts=True)
    frames, _ = pb.load_universe(
        ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT",
         "XRP/USDT", "DOGE/USDT"],
        timeframe="1d",
    )
    asof = int(frames["BTC/USDT"]["timestamp"].iloc[-1])
    sig = _make_signals_with(crs.ALT_RISK_ON, asof)
    strat = RegimeAwareMomentumRotationStrategy(
        signals_df=sig,
        cfg=RegimeAwareMomentumConfig(top_n_alt_risk_on=3,
                                      min_assets_required=3),
    )
    weights = strat.target_weights(asof, frames, timeframe="1d")
    assert len(weights) == 3
    assert sum(weights.values()) == pytest.approx(1.0, rel=1e-9)
    # SOL had the steepest synthetic uptrend (100 -> 500) and must be picked.
    assert "SOL/USDT" in weights


def test_strategy_target_weights_never_exceed_100(isolated):
    """Even in the extreme alt_risk_on path with min_assets very low,
    weights must sum to ≤ 1.0."""
    _seed_universe(strong_alts=True)
    frames, _ = pb.load_universe(
        ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT"],
        timeframe="1d",
    )
    asof = int(frames["BTC/USDT"]["timestamp"].iloc[-1])
    sig = _make_signals_with(crs.ALT_RISK_ON, asof)
    strat = RegimeAwareMomentumRotationStrategy(
        signals_df=sig,
        cfg=RegimeAwareMomentumConfig(top_n_alt_risk_on=3,
                                      min_assets_required=3),
    )
    weights = strat.target_weights(asof, frames, timeframe="1d")
    assert sum(weights.values()) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Walk-forward + placebo + scorecard
# ---------------------------------------------------------------------------
def test_walk_forward_creates_oos_windows(isolated):
    _seed_universe(strong_alts=True)
    df = portfolio_research.regime_aware_portfolio_walk_forward(
        assets=["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT"],
        timeframe="1d",
        in_sample_days=120, oos_days=60, step_days=60, save=False,
    )
    assert not df.empty
    assert "oos_return_pct" in df.columns
    assert "beats_simple_momentum" in df.columns


def test_placebo_is_reproducible(isolated):
    _seed_universe(strong_alts=True)
    frames, _ = pb.load_universe(
        ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT"],
        timeframe="1d",
    )
    sig = crs.compute_regime_signals(asset_frames=frames, timeframe="1d",
                                     save=False)
    a = RegimeAwareRandomPlacebo(signals_df=sig,
                                 cfg=RegimeAwareRandomConfig(seed=7))
    b = RegimeAwareRandomPlacebo(signals_df=sig,
                                 cfg=RegimeAwareRandomConfig(seed=7))
    art_a = pb.run_portfolio_backtest(
        portfolio_strategy=a, asset_frames=frames, timeframe="1d", save=False,
    )
    art_b = pb.run_portfolio_backtest(
        portfolio_strategy=b, asset_frames=frames, timeframe="1d", save=False,
    )
    pd.testing.assert_series_equal(
        art_a.equity_curve["equity"].reset_index(drop=True),
        art_b.equity_curve["equity"].reset_index(drop=True),
        check_names=False,
    )


def test_scorecard_marks_weak_result_as_fail(isolated):
    """Synthetic weak walk-forward + losing placebo comparison must NOT
    yield PASS."""
    wf = pd.DataFrame([
        {"window": i, "oos_return_pct": -1.0, "oos_max_drawdown_pct": -5.0,
         "btc_oos_return_pct": 5.0, "basket_oos_return_pct": 4.0,
         "simple_oos_return_pct": 3.0,
         "beats_btc": False, "beats_basket": False,
         "beats_simple_momentum": False, "profitable": False,
         "n_rebalances": 4, "n_trades": 8, "avg_holdings": 2.0,
         "error": None}
        for i in range(1, 8)
    ])
    plac = pd.DataFrame([{
        "strategy": "regime_aware_momentum_rotation",
        "strategy_return_pct": -10.0,
        "strategy_max_drawdown_pct": -15.0,
        "n_seeds": 5,
        "placebo_median_return_pct": 8.0,
        "placebo_median_drawdown_pct": -10.0,
        "placebo_p75_drawdown_pct": -12.0,
        "strategy_beats_median_return": False,
        "strategy_beats_median_drawdown": False,
    }])
    sc = portfolio_research.regime_aware_portfolio_scorecard(wf, plac,
                                                             save=False)
    assert sc.iloc[0]["verdict"] != portfolio_research.PORTFOLIO_PASS
