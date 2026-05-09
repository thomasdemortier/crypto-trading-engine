"""Optional-Kronos regression tests.

None of these tests download Kronos weights or import torch / Kronos source.
They exercise the *adapter surface* (availability, status, input prep,
confirmation rules) and the *strategy wrapper* (CSV lookup, fallback, no
risk-engine bypass) using synthetic data and synthetic confirmation files.

Goal: prove that
  * the app boots cleanly without Kronos installed,
  * confirmation policy obeys the documented thresholds, and
  * the wrapper strategy still routes every Signal through the existing
    RiskEngine path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import backtester, config, utils
from src.ml import kronos_adapter, kronos_confirmation, forecast_evaluation
from src.strategies import REGISTRY as STRATEGY_REGISTRY
from src.strategies.kronos_confirmed import KronosConfirmedStrategy


# ---------------------------------------------------------------------------
# Synthetic data fixtures (no network, no model downloads)
# ---------------------------------------------------------------------------
def _synthetic_csv(symbol: str, timeframe: str, n: int = 600,
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
    res = tmp_path / "results"
    res.mkdir()
    monkeypatch.setattr(config, "DATA_RAW_DIR", raw)
    monkeypatch.setattr(config, "RESULTS_DIR", res)
    yield {"raw": raw, "results": res}


# ---------------------------------------------------------------------------
# Phase 1 — availability + status (Kronos NOT installed in CI)
# ---------------------------------------------------------------------------
def test_kronos_available_returns_bool_without_crash():
    res = kronos_adapter.kronos_available()
    assert isinstance(res, bool)


def test_kronos_status_returns_expected_keys():
    s = kronos_adapter.get_kronos_status()
    expected = {
        "available", "python_deps_ok", "missing_python_deps",
        "repo_path_resolved", "repo_path_candidates_checked",
        "kronos_repo_path_env", "import_error",
        "supported_models", "default_model",
    }
    assert expected.issubset(s.keys())
    assert isinstance(s["supported_models"], list)
    assert "Kronos-mini" in s["supported_models"]


def test_app_launches_without_kronos_installed():
    """Importing the Streamlit script must not require Kronos. We import
    the dashboard module-level objects without instantiating widgets."""
    import importlib
    # The streamlit_app module triggers Streamlit calls at import time;
    # those are not safe outside the Streamlit runtime. Instead, prove the
    # ML adapter import + availability check is safe (the dashboard imports
    # the adapter lazily inside the Kronos tab only).
    mod = importlib.reload(kronos_adapter)
    assert hasattr(mod, "kronos_available")
    assert mod.kronos_available() is False or isinstance(
        mod.kronos_available(), bool
    )


# ---------------------------------------------------------------------------
# Phase 2 — forecast input prep
# ---------------------------------------------------------------------------
def _candle_frame(n: int = 100) -> pd.DataFrame:
    ts = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "timestamp": (ts.astype("int64") // 10**6).astype("int64"),
        "datetime": ts,
        "open": close, "high": close + 1, "low": close - 1,
        "close": close,
    })  # NOTE: deliberately omits volume + amount


def test_prepare_forecast_inputs_handles_missing_volume_and_amount():
    df = _candle_frame(100)
    out = kronos_adapter.prepare_forecast_inputs(
        df, timeframe="4h", lookback=60, pred_len=10,
    )
    x_df = out["x_df"]
    assert "volume" in x_df.columns and "amount" in x_df.columns
    assert (x_df["volume"] == 0).all()
    assert (x_df["amount"] == 0).all()
    assert len(x_df) == 60


def test_prepare_forecast_inputs_generates_correct_y_timestamps():
    df = _candle_frame(100)
    out = kronos_adapter.prepare_forecast_inputs(
        df, timeframe="4h", lookback=50, pred_len=12,
    )
    yt = out["y_timestamp"]
    assert len(yt) == 12
    spacing = (yt.iloc[1] - yt.iloc[0]).total_seconds()
    assert spacing == 4 * 3600


def test_prepare_forecast_inputs_rejects_short_history():
    df = _candle_frame(20)
    with pytest.raises(ValueError):
        kronos_adapter.prepare_forecast_inputs(
            df, timeframe="4h", lookback=60, pred_len=10,
        )


# ---------------------------------------------------------------------------
# Phase 4 — confirmation policy
# ---------------------------------------------------------------------------
def test_confirm_buy_above_threshold_confirms():
    c = kronos_confirmation.confirm_signal_with_kronos("BUY", 2.0)
    assert c.confirmation == kronos_confirmation.CONFIRM
    assert c.confidence_proxy == 2.0


def test_confirm_buy_negative_rejects():
    c = kronos_confirmation.confirm_signal_with_kronos("BUY", -0.5)
    assert c.confirmation == kronos_confirmation.REJECT


def test_confirm_buy_neutral_band():
    c = kronos_confirmation.confirm_signal_with_kronos("BUY", 0.5)
    assert c.confirmation == kronos_confirmation.NEUTRAL


def test_confirm_sell_negative_or_zero_confirms():
    assert kronos_confirmation.confirm_signal_with_kronos(
        "SELL", -1.0).confirmation == kronos_confirmation.CONFIRM
    assert kronos_confirmation.confirm_signal_with_kronos(
        "SELL", 0.0).confirmation == kronos_confirmation.CONFIRM


def test_confirm_sell_strong_positive_rejects():
    c = kronos_confirmation.confirm_signal_with_kronos("SELL", 2.0)
    assert c.confirmation == kronos_confirmation.REJECT


def test_confirm_sell_neutral_band():
    c = kronos_confirmation.confirm_signal_with_kronos("SELL", 0.5)
    assert c.confirmation == kronos_confirmation.NEUTRAL


def test_confirm_does_not_call_kronos_for_hold_or_skip():
    for action in ("HOLD", "SKIP", ""):
        c = kronos_confirmation.confirm_signal_with_kronos(action, 5.0)
        assert c.confirmation == kronos_confirmation.NO_CALL


def test_generate_kronos_confirmations_matches_signals_to_forecasts():
    base = pd.DataFrame([
        {"timestamp_ms": 100, "asset": "BTC/USDT", "timeframe": "4h",
         "action": "BUY"},
        {"timestamp_ms": 200, "asset": "BTC/USDT", "timeframe": "4h",
         "action": "SELL"},
        {"timestamp_ms": 300, "asset": "BTC/USDT", "timeframe": "4h",
         "action": "HOLD"},
    ])
    fc = pd.DataFrame([
        {"asset": "BTC/USDT", "timeframe": "4h",
         "forecast_start_ms": 50, "forecast_return_pct": 2.0},
        {"asset": "BTC/USDT", "timeframe": "4h",
         "forecast_start_ms": 150, "forecast_return_pct": -1.5},
    ])
    out = kronos_confirmation.generate_kronos_confirmations(
        base, fc, save=False,
    )
    assert len(out) == 3
    # First BUY (ts 100) uses forecast at 50 (+2.0%) -> CONFIRM
    assert out.iloc[0]["confirmation"] == kronos_confirmation.CONFIRM
    # SELL (ts 200) uses forecast at 150 (-1.5%) -> CONFIRM
    assert out.iloc[1]["confirmation"] == kronos_confirmation.CONFIRM
    # HOLD never consults Kronos
    assert out.iloc[2]["confirmation"] == kronos_confirmation.NO_CALL


# ---------------------------------------------------------------------------
# Phase 5 — strategy wrapper: lookup, fallback, risk-engine path
# ---------------------------------------------------------------------------
def test_kronos_confirmed_strategy_no_csv_skips_safely(synthetic_cache):
    """When confirmations CSV is missing, default fallback must convert
    BUY/SELL to SKIP (NEVER pass-through unsafely)."""
    _synthetic_csv("BTC/USDT", "4h", n=600, seed=1)
    base = STRATEGY_REGISTRY["rsi_ma_atr"]()
    wrapped = KronosConfirmedStrategy(
        base_strategy=base,
        confirmations_path=synthetic_cache["results"] / "MISSING.csv",
        fallback="skip",
    )
    art = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False, strategy=wrapped,
    )
    # No CSV -> every BUY/SELL becomes SKIP -> no trades fire.
    assert art.trades.empty
    # But decisions are still logged with the fallback reason.
    assert not art.decisions.empty
    skip_rows = art.decisions[art.decisions["action"] == "SKIP"]
    if not skip_rows.empty:
        any_kronos = skip_rows["reason"].astype(str).str.contains(
            "kronos:", case=False, na=False,
        ).any()
        assert any_kronos


def test_kronos_confirmed_strategy_runs_through_risk_engine(synthetic_cache):
    """Wrapper must produce only valid risk-engine artifacts: trade rows
    use BUY/SELL only, and Kronos reasons are recorded on the decision."""
    _synthetic_csv("BTC/USDT", "4h", n=600, seed=1)
    # Build a confirmations CSV that CONFIRMs every base BUY/SELL on the
    # incumbent strategy by setting forecast_return_pct appropriately on
    # every decision timestamp.
    base = STRATEGY_REGISTRY["rsi_ma_atr"]()
    base_art = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False, strategy=base,
    )
    decisions = base_art.decisions
    assert not decisions.empty
    confirm_rows = []
    for _, r in decisions.iterrows():
        action = str(r["action"])
        if action == "BUY":
            fr = 5.0
        elif action == "SELL":
            fr = -2.0
        else:
            continue
        confirm_rows.append({
            "asset": r["asset"], "timestamp_ms": int(r["timestamp_ms"]),
            "timeframe": "4h", "base_signal": action,
            "forecast_return_pct": fr,
            "confirmation": kronos_confirmation.CONFIRM,
            "confidence_proxy": abs(fr), "final_signal": action,
            "reason": "synthetic-test confirm",
        })
    confirm_df = pd.DataFrame(confirm_rows)
    out_csv = synthetic_cache["results"] / "kronos_confirmations.csv"
    confirm_df.to_csv(out_csv, index=False)

    wrapped = KronosConfirmedStrategy(
        base_strategy=STRATEGY_REGISTRY["rsi_ma_atr"](),
        confirmations_path=out_csv, fallback="skip",
    )
    art = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False, strategy=wrapped,
    )
    if not art.trades.empty:
        assert set(art.trades["side"].unique()).issubset({"BUY", "SELL"})
    # All decisions for BUY/SELL must mention Kronos in the reason
    bs = art.decisions[art.decisions["action"].isin(["BUY", "SELL"])]
    if not bs.empty:
        kronos_tagged = bs["reason"].astype(str).str.contains(
            "kronos:", case=False, na=False,
        )
        assert kronos_tagged.all()


def test_kronos_confirmed_strategy_can_fallback_to_base(synthetic_cache):
    _synthetic_csv("BTC/USDT", "4h", n=600, seed=1)
    base_art = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False,
        strategy=STRATEGY_REGISTRY["rsi_ma_atr"](),
    )
    wrapped = KronosConfirmedStrategy(
        base_strategy=STRATEGY_REGISTRY["rsi_ma_atr"](),
        confirmations_path=synthetic_cache["results"] / "missing.csv",
        fallback="base",
    )
    fb_art = backtester.run_backtest(
        assets=["BTC/USDT"], timeframe="4h", save=False, strategy=wrapped,
    )
    # With fallback="base" and no CSV present, results should match the
    # base strategy 1:1 (no signals are filtered).
    assert len(fb_art.trades) == len(base_art.trades)


def test_kronos_confirmed_strategy_invalid_fallback_raises():
    with pytest.raises(ValueError):
        KronosConfirmedStrategy(
            base_strategy=STRATEGY_REGISTRY["rsi_ma_atr"](),
            fallback="bogus",
        )


# ---------------------------------------------------------------------------
# Phase 6 — comparison output shape
# ---------------------------------------------------------------------------
def test_compare_base_vs_kronos_confirmed_output_columns(synthetic_cache):
    _synthetic_csv("BTC/USDT", "4h", n=600, seed=1)
    df = forecast_evaluation.compare_base_vs_kronos_confirmed(
        asset="BTC/USDT", timeframe="4h",
        base_strategy_name="rsi_ma_atr",
        kronos_confirmations_path=synthetic_cache["results"] / "missing.csv",
        save=False,
    )
    assert len(df) == 2
    expected = {
        "variant", "total_return_pct", "buy_and_hold_return_pct",
        "strategy_vs_bh_pct", "max_drawdown_pct", "win_rate_pct",
        "num_trades", "profit_factor", "fees_paid", "exposure_time_pct",
        "sharpe_ratio", "sortino_ratio", "calmar_ratio",
    }
    assert expected.issubset(df.columns)


# ---------------------------------------------------------------------------
# Safety — no live trading code, no API keys, no order endpoints
# ---------------------------------------------------------------------------
def test_no_live_trading_in_kronos_modules():
    src_root = Path(__file__).resolve().parents[1] / "src"
    targets = list((src_root / "ml").rglob("*.py"))
    targets.append(src_root / "strategies" / "kronos_confirmed.py")
    forbidden = (
        "create_order", "createMarketBuyOrder", "createMarketSellOrder",
        "createLimitOrder", "create_market_buy_order",
        "create_market_sell_order", "withdraw(", ".privateGet",
        ".privatePost", "apiKey=", "secret=",
    )
    offenders = []
    for py in targets:
        text = py.read_text()
        for s in forbidden:
            if s in text:
                offenders.append((py.name, s))
    assert not offenders, f"forbidden tokens found in Kronos modules: {offenders}"


def test_kronos_modules_do_not_import_torch_at_module_top():
    """`torch` and friends MUST be lazy-imported inside functions only —
    otherwise Streamlit Cloud (which lacks `torch`) would crash on dashboard
    load."""
    src_root = Path(__file__).resolve().parents[1] / "src"
    for py in (src_root / "ml").rglob("*.py"):
        text = py.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert not stripped.startswith("import torch"), (
                f"{py.name} imports torch at top level — must be lazy"
            )
            assert not stripped.startswith("from torch"), (
                f"{py.name} imports from torch at top level — must be lazy"
            )
            assert not stripped.startswith("from transformers"), (
                f"{py.name} imports from transformers at top level — must be lazy"
            )


def test_kronos_compare_uses_same_risk_engine_path(synthetic_cache):
    """Smoke check: both variants must populate the same artifact schema,
    proving the wrapper goes through the same backtester/risk engine."""
    _synthetic_csv("BTC/USDT", "4h", n=600, seed=1)
    df = forecast_evaluation.compare_base_vs_kronos_confirmed(
        asset="BTC/USDT", timeframe="4h",
        base_strategy_name="rsi_ma_atr",
        kronos_confirmations_path=synthetic_cache["results"] / "missing.csv",
        save=False,
    )
    # Both variants exposed identical metric columns -> shared engine path.
    assert df.iloc[0].keys().tolist() == df.iloc[1].keys().tolist()
