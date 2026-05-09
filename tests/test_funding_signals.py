"""Tests for `src.funding_signals`.

Synthetic spot + funding inputs verify:
  * The output schema is the documented column set in the documented order.
  * 90d funding z-score is NaN until 90 valid observations exist
    (no future leakage).
  * Returns are realised (close[t]/close[t-N] - 1).
  * crowded_long fires when z(funding) > 1.5 AND 30d return > 0.
  * capitulation fires when z(funding) < -1.5 AND 30d return < 0.
  * Every row classifies into one of the documented funding_states.
  * Missing funding (empty frame) → all rows `unknown`.
  * Missing spot → empty frame with the documented columns.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src import config, data_collector, funding_signals as fs
from src import futures_data_collector as fdc


def _spot(n_days: int, drift: float = 0.001, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=drift, scale=0.01, size=n_days)
    closes = 100.0 * np.cumprod(1.0 + rets)
    end = pd.Timestamp("2025-12-31", tz="UTC")
    dates = pd.date_range(end=end, periods=n_days, freq="D", tz="UTC")
    return pd.DataFrame({
        "timestamp": (dates.view("int64") // 10**6),
        "datetime": dates,
        "open": closes, "high": closes * 1.01, "low": closes * 0.99,
        "close": closes, "volume": 1000.0,
    })


def _funding(dates: pd.DatetimeIndex, rates) -> pd.DataFrame:
    """Build a Binance-shaped funding frame: 3 events per day at 0/8/16h.
    `rates` is an iterable of length len(dates)."""
    rows = []
    for d, r in zip(dates, rates):
        for h in (0, 8, 16):
            ts_ms = (d + pd.Timedelta(hours=h)).value // 10**6
            rows.append({"timestamp": ts_ms, "symbol": "BTCUSDT",
                          "funding_rate": float(r), "funding_time": ts_ms,
                          "source": "binance_fapi_v1_funding_rate"})
    return pd.DataFrame(rows)


def _wire(monkeypatch, spot=None, funding=None, raise_spot: bool = False):
    if raise_spot:
        monkeypatch.setattr(data_collector, "load_candles",
                             lambda *a, **k: (_ for _ in ()).throw(
                                 FileNotFoundError("missing")))
    else:
        monkeypatch.setattr(data_collector, "load_candles",
                             lambda *a, **k: spot)
    monkeypatch.setattr(fdc, "load_funding_rate",
                         lambda *a, **k: funding if funding is not None
                         else pd.DataFrame())


# ---------------------------------------------------------------------------
# Schema + lookahead
# ---------------------------------------------------------------------------
def test_signal_columns_match_spec():
    expected = [
        "timestamp", "symbol", "close",
        "return_1d", "return_7d", "return_30d",
        "funding_rate", "funding_7d_mean", "funding_30d_mean",
        "funding_90d_zscore",
        "extreme_positive_funding", "extreme_negative_funding",
        "funding_trend", "funding_normalization",
        "price_return_7d", "price_return_30d",
        "funding_attractiveness", "funding_improvement",
        "stabilization_score", "crowding_penalty",
        "funding_state",
    ]
    assert fs.SIGNAL_COLUMNS == expected


def test_compute_returns_documented_schema(monkeypatch):
    sp = _spot(200, seed=1)
    fn = _funding(pd.to_datetime(sp["datetime"]), [0.0001] * 200)
    _wire(monkeypatch, sp, fn)
    out = fs.compute_funding_signals_for_symbol("BTCUSDT")
    assert list(out.columns) == fs.SIGNAL_COLUMNS
    assert len(out) == 200


def test_zscore_is_nan_until_90d_warmup(monkeypatch):
    """No future leakage: z-score must be NaN until 90 valid funding
    observations exist."""
    sp = _spot(200, seed=2)
    rng = np.random.default_rng(2)
    rates = rng.normal(loc=0.0001, scale=0.0001, size=200)
    fn = _funding(pd.to_datetime(sp["datetime"]), rates)
    _wire(monkeypatch, sp, fn)
    out = fs.compute_funding_signals_for_symbol("BTCUSDT")
    z = out["funding_90d_zscore"]
    assert z.iloc[:89].isna().all()


def test_returns_are_backward_only(monkeypatch):
    sp = _spot(60, seed=3)
    fn = _funding(pd.to_datetime(sp["datetime"]), [0.0001] * 60)
    _wire(monkeypatch, sp, fn)
    out = fs.compute_funding_signals_for_symbol("BTCUSDT")
    closes = sp["close"].to_numpy()
    expected = closes[37] / closes[30] - 1
    assert out["return_7d"].iloc[37] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# State classification
# ---------------------------------------------------------------------------
def test_states_only_use_documented_set(monkeypatch):
    sp = _spot(200, drift=0.001, seed=4)
    rng = np.random.default_rng(4)
    rates = rng.normal(loc=0.0001, scale=0.0002, size=200)
    fn = _funding(pd.to_datetime(sp["datetime"]), rates)
    _wire(monkeypatch, sp, fn)
    out = fs.compute_funding_signals_for_symbol("BTCUSDT")
    assert set(out["funding_state"].unique()).issubset(set(fs.VALID_FUNDING_STATES))


def test_crowded_long_fires_when_funding_z_high_and_uptrend(monkeypatch):
    """Build: 200d uptrend + a funding spike at the END."""
    n = 200
    sp = _spot(n, drift=0.005, seed=5)
    rates = [0.00005] * (n - 5) + [0.001, 0.0012, 0.0015, 0.0018, 0.002]
    fn = _funding(pd.to_datetime(sp["datetime"]), rates)
    _wire(monkeypatch, sp, fn)
    out = fs.compute_funding_signals_for_symbol("BTCUSDT")
    last = out.iloc[-1]
    assert last["funding_90d_zscore"] > fs._FUNDING_EXTREME_Z
    assert last["return_30d"] > 0
    assert last["funding_state"] == "crowded_long"
    assert bool(last["extreme_positive_funding"]) is True


def test_capitulation_fires_when_funding_z_negative_and_downtrend(monkeypatch):
    """Funding sharply negative + price falling = leveraged longs flushed."""
    n = 200
    rng = np.random.default_rng(6)
    rets = list(rng.normal(loc=0.0, scale=0.005, size=n - 30))
    rets.extend(list(rng.normal(loc=-0.005, scale=0.005, size=30)))
    closes = 100.0 * np.cumprod(1.0 + np.array(rets))
    end = pd.Timestamp("2025-12-31", tz="UTC")
    dates = pd.date_range(end=end, periods=n, freq="D", tz="UTC")
    sp = pd.DataFrame({
        "timestamp": dates.view("int64") // 10**6, "datetime": dates,
        "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": 1000.0,
    })
    rates = [0.0001] * (n - 5) + [-0.001, -0.0012, -0.0015, -0.0018, -0.002]
    fn = _funding(pd.to_datetime(sp["datetime"]), rates)
    _wire(monkeypatch, sp, fn)
    out = fs.compute_funding_signals_for_symbol("BTCUSDT")
    last = out.iloc[-1]
    assert last["funding_90d_zscore"] < -fs._FUNDING_EXTREME_Z
    assert last["return_30d"] < 0
    assert last["funding_state"] == "capitulation"
    assert bool(last["extreme_negative_funding"]) is True


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
def test_missing_funding_yields_unknown(monkeypatch):
    sp = _spot(120, seed=7)
    _wire(monkeypatch, sp, funding=pd.DataFrame())
    out = fs.compute_funding_signals_for_symbol("BTCUSDT")
    assert len(out) == 120
    assert out["funding_rate"].isna().all()
    assert (out["funding_state"] == "unknown").all()


def test_missing_spot_returns_empty_frame(monkeypatch):
    _wire(monkeypatch, raise_spot=True)
    out = fs.compute_funding_signals_for_symbol("DOGEUSDT")
    assert out.empty
    assert list(out.columns) == fs.SIGNAL_COLUMNS


def test_compute_all_persists_csv(tmp_path, monkeypatch):
    sp = _spot(120, seed=8)
    fn = _funding(pd.to_datetime(sp["datetime"]), [0.0001] * 120)
    _wire(monkeypatch, sp, fn)
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    out = fs.compute_all_funding_signals(symbols=("BTCUSDT", "ETHUSDT"),
                                            save=True)
    assert (tmp_path / "funding_signals.csv").exists()
    assert len(out) == 240
