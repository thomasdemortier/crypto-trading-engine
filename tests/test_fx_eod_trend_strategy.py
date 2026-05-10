"""Tests for `src/strategies/fx_eod_trend.py`. All offline, all
synthetic. Covers: locked config, signal computation, no-lookahead,
long-or-cash invariants, no-leverage, no-shorts, and the safety
invariants the strategy module must not introduce.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config, fx_research_dataset, safety_lock
from src.strategies import fx_eod_trend


REPO_ROOT = config.REPO_ROOT


# ---------------------------------------------------------------------------
# Synthetic data builder
# ---------------------------------------------------------------------------
def _synthetic_eur_usd(n_days: int = 600, start_close: float = 1.10,
                          drift: float = 0.0001, seed: int = 7
                          ) -> pd.DataFrame:
    """Geometric-random-walk synthetic EUR/USD series in the locked
    dataset schema."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.005, size=n_days)
    closes = start_close * np.exp(np.cumsum(rets))
    dates = pd.date_range("2000-01-03", periods=n_days, freq="B")
    rows = []
    for d, c in zip(dates, closes):
        rows.append({
            "date": d, "asset": "EUR/USD", "source": "ecb_sdmx",
            "base": "EUR", "quote": "USD", "close": float(c),
            "return_1d": float("nan"),
            "log_return_1d": float("nan"),
            "is_derived": False, "source_pair": "",
            "data_quality_status": "ok",
            "notes": "",
        })
    return pd.DataFrame(rows, columns=fx_research_dataset.DATASET_COLUMNS)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_default_config_values_locked():
    cfg = fx_eod_trend.FXEODTrendConfig()
    assert cfg.asset == "EUR/USD"
    assert cfg.lookback_days == 200
    assert cfg.timeframe == "1d"
    assert cfg.mode == fx_eod_trend.MODE_LONG_CASH
    assert cfg.initial_cash == 1.0
    assert cfg.source_filter == "ecb_sdmx"
    assert cfg.strategy_name == "fx_eod_trend_v1"


def test_config_is_frozen():
    cfg = fx_eod_trend.FXEODTrendConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.lookback_days = 50  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------
def test_sma_warmup_is_nan_then_finite():
    cfg = fx_eod_trend.FXEODTrendConfig(lookback_days=10)
    close = pd.Series(np.arange(20, dtype=float) + 1.0)
    sigs = fx_eod_trend.FXEODTrendStrategy.generate_signals(close, cfg)
    # First 9 SMA values must be NaN; from index 9 onward they're set.
    assert sigs["sma"].iloc[:cfg.lookback_days - 1].isna().all()
    assert sigs["sma"].iloc[cfg.lookback_days - 1:].notna().all()


def test_signal_is_binary_and_uses_only_past_data():
    cfg = fx_eod_trend.FXEODTrendConfig(lookback_days=5)
    close = pd.Series([1, 1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=float)
    sigs = fx_eod_trend.FXEODTrendStrategy.generate_signals(close, cfg)
    # SMA at index 4 = mean(close[0:5]) = 1.0; close[4]=1 not > 1 → 0.
    assert sigs["signal"].iloc[4] == 0.0
    # SMA at index 5 = mean(close[1:6]) = (1+1+1+1+2)/5 = 1.2; close[5]=2 > 1.2 → 1.
    assert sigs["signal"].iloc[5] == 1.0


def test_assert_no_lookahead_passes_for_correct_sma():
    cfg = fx_eod_trend.FXEODTrendConfig(lookback_days=5)
    close = pd.Series(np.arange(50, dtype=float) + 1.0)
    sigs = fx_eod_trend.FXEODTrendStrategy.generate_signals(close, cfg)
    fx_eod_trend.assert_no_lookahead(close, sigs["sma"], cfg)  # no raise


def test_assert_no_lookahead_raises_when_sma_uses_future_data():
    cfg = fx_eod_trend.FXEODTrendConfig(lookback_days=5)
    close = pd.Series(np.arange(50, dtype=float) + 1.0)
    sigs = fx_eod_trend.FXEODTrendStrategy.generate_signals(close, cfg)
    # Corrupt the FIRST valid SMA value (one of the spot-checked
    # indices) to one that uses future bars.
    bad = sigs["sma"].copy()
    first_valid = bad.dropna().index[0]
    bad.iloc[first_valid] = float(
        close.iloc[first_valid: first_valid + 5].mean()
    )
    with pytest.raises(ValueError, match="lookahead"):
        fx_eod_trend.assert_no_lookahead(close, bad, cfg)


# ---------------------------------------------------------------------------
# Position lag (no lookahead in the backtest)
# ---------------------------------------------------------------------------
def test_position_is_signal_shifted_by_one_bar():
    sig = pd.Series([np.nan, np.nan, 0.0, 1.0, 1.0, 0.0], dtype=float)
    pos = fx_eod_trend.FXEODTrendStrategy._lag_position(sig)
    assert pd.isna(pos.iloc[0])
    assert pd.isna(pos.iloc[1])
    assert pd.isna(pos.iloc[2])
    assert pos.iloc[3] == 0.0
    assert pos.iloc[4] == 1.0
    assert pos.iloc[5] == 1.0


# ---------------------------------------------------------------------------
# Long-or-cash + no-leverage + no-shorts
# ---------------------------------------------------------------------------
def test_long_cash_invariants_hold_on_synthetic_run():
    cfg = fx_eod_trend.FXEODTrendConfig(lookback_days=20)
    df = _synthetic_eur_usd(n_days=300)
    bt = fx_eod_trend.FXEODTrendStrategy.compute_backtest(df, cfg)
    assert not bt.empty
    p = bt["position"]
    fx_eod_trend.assert_long_cash_only(p)
    assert p.dropna().min() >= 0.0
    assert p.dropna().max() <= 1.0


def test_assert_long_cash_only_rejects_short_position():
    pos = pd.Series([0.0, -1.0, 1.0])
    with pytest.raises(ValueError, match="negative"):
        fx_eod_trend.assert_long_cash_only(pos)


def test_assert_long_cash_only_rejects_leverage():
    pos = pd.Series([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="leverage"):
        fx_eod_trend.assert_long_cash_only(pos)


def test_assert_long_cash_only_rejects_fractional_position():
    pos = pd.Series([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="extras"):
        fx_eod_trend.assert_long_cash_only(pos)


# ---------------------------------------------------------------------------
# Backtest output schema
# ---------------------------------------------------------------------------
def test_backtest_schema_locked():
    cfg = fx_eod_trend.FXEODTrendConfig(lookback_days=20)
    df = _synthetic_eur_usd(n_days=300)
    bt = fx_eod_trend.FXEODTrendStrategy.compute_backtest(df, cfg)
    assert list(bt.columns) == fx_eod_trend.BACKTEST_COLUMNS
    assert (bt["asset"] == "EUR/USD").all()
    assert (bt["close"] > 0).all()


def test_backtest_starts_after_warmup():
    cfg = fx_eod_trend.FXEODTrendConfig(lookback_days=50)
    df = _synthetic_eur_usd(n_days=200)
    bt = fx_eod_trend.FXEODTrendStrategy.compute_backtest(df, cfg)
    # Roughly: 200 - 50 (warmup) - 1 (lag) ≈ 149 active rows
    assert 100 < len(bt) <= 200 - cfg.lookback_days
    # First row's SMA must be defined and the position must be in {0, 1}
    assert pd.notna(bt["sma"].iloc[0])
    assert bt["position"].iloc[0] in (0.0, 1.0)


def test_backtest_returns_match_lagged_signal_times_raw_return():
    cfg = fx_eod_trend.FXEODTrendConfig(lookback_days=20)
    df = _synthetic_eur_usd(n_days=300, seed=42)
    bt = fx_eod_trend.FXEODTrendStrategy.compute_backtest(df, cfg)
    expected = bt["position"] * bt["raw_return"]
    expected = expected.fillna(0.0)
    np.testing.assert_allclose(bt["strategy_return"].values,
                                  expected.values, atol=1e-12)


def test_backtest_position_never_uses_today_close_for_today_return():
    """If we replace today's close with a wildly different value, the
    strategy_return for today must NOT change — the position used for
    today's return is the prior signal, not today's signal."""
    cfg = fx_eod_trend.FXEODTrendConfig(lookback_days=20)
    df_a = _synthetic_eur_usd(n_days=300, seed=11)
    df_b = df_a.copy()
    # Bump the LAST close massively; the lagged position can't see it
    # for that day's return.
    last_idx = df_b.index[-1]
    df_b.loc[last_idx, "close"] = float(df_b.loc[last_idx, "close"] * 1.5)
    bt_a = fx_eod_trend.FXEODTrendStrategy.compute_backtest(df_a, cfg)
    bt_b = fx_eod_trend.FXEODTrendStrategy.compute_backtest(df_b, cfg)
    # All rows except possibly the last must match exactly.
    common = min(len(bt_a), len(bt_b)) - 1
    np.testing.assert_allclose(
        bt_a["strategy_return"].iloc[:common].values,
        bt_b["strategy_return"].iloc[:common].values,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Empty / insufficient input
# ---------------------------------------------------------------------------
def test_backtest_empty_when_dataset_missing_asset():
    cfg = fx_eod_trend.FXEODTrendConfig()
    df = _synthetic_eur_usd(n_days=300)
    df["asset"] = "USD/CHF"  # not the configured asset
    bt = fx_eod_trend.FXEODTrendStrategy.compute_backtest(df, cfg)
    assert bt.empty


def test_backtest_empty_when_too_few_rows():
    cfg = fx_eod_trend.FXEODTrendConfig(lookback_days=200)
    df = _synthetic_eur_usd(n_days=50)
    bt = fx_eod_trend.FXEODTrendStrategy.compute_backtest(df, cfg)
    assert bt.empty


# ---------------------------------------------------------------------------
# Pure-function metrics
# ---------------------------------------------------------------------------
def test_total_return_simple_compound():
    rets = pd.Series([0.10, -0.05, 0.02])
    expected = (1.10 * 0.95 * 1.02) - 1.0
    assert fx_eod_trend.total_return_from_returns(rets) == \
        pytest.approx(expected)


def test_sharpe_zero_when_constant_returns():
    rets = pd.Series([0.0, 0.0, 0.0, 0.0])
    assert fx_eod_trend.annualised_sharpe(rets) == 0.0


def test_max_drawdown_from_equity_correct():
    eq = pd.Series([1.0, 1.2, 0.9, 1.1, 0.8])
    # peak so far: [1.0, 1.2, 1.2, 1.2, 1.2]
    # dd: [0, 0, -0.25, -0.0833, -0.3333]
    assert fx_eod_trend.max_drawdown_from_equity(eq) == \
        pytest.approx(0.8 / 1.2 - 1.0)


def test_trade_count_and_exposure_pct():
    pos = pd.Series([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    assert fx_eod_trend.trade_count_from_position(pos) == 3
    assert fx_eod_trend.exposure_pct_from_position(pos) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Safety invariants — strategy module must not introduce broker / API key /
# execution / paper-trading / live-trading code.
# ---------------------------------------------------------------------------
_MODULE_PATH = Path(fx_eod_trend.__file__)


def _read(path: Path) -> str:
    return path.read_text(errors="ignore")


def test_no_broker_imports_in_strategy_module():
    text = _read(_MODULE_PATH)
    assert "ccxt" not in text
    assert "alpaca" not in text.lower()
    assert "kraken" not in text.lower()
    assert "ig_bank" not in text.lower()
    assert "oanda" not in text.lower()


def test_no_api_key_reads_in_strategy_module():
    text = _read(_MODULE_PATH)
    assert "API_KEY" not in text
    assert "API_SECRET" not in text
    assert "os.environ" not in text
    assert "os.getenv" not in text


def test_no_order_placement_in_strategy_module():
    forbidden = (
        "create_order", "place_order", "submit_order",
        "send_order", "cancel_order", "AddOrder", "CancelOrder",
    )
    text = _read(_MODULE_PATH)
    for token in forbidden:
        assert token not in text, f"{token} found in strategy module"


def test_no_paper_or_live_trading_enablement():
    text = _read(_MODULE_PATH)
    assert "LIVE_TRADING_ENABLED = True" not in text
    assert "paper_trading_allowed = True" not in text
    assert "execution_allowed = True" not in text
    assert "ENABLE_LIVE" not in text
    assert "UNLOCK_TRADING" not in text
    assert "FORCE_TRADE" not in text


def test_safety_lock_remains_locked_after_strategy_import():
    s = safety_lock.status()
    assert not s.execution_allowed
    assert not s.paper_trading_allowed
    assert not s.kraken_connection_allowed
    assert s.safety_lock_status == "locked"
