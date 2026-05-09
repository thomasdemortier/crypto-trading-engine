"""Tests for `src.market_structure_signals`."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src import (config, data_collector,
                  market_structure_data_collector as mdc,
                  market_structure_signals as mss)


# ---------------------------------------------------------------------------
# Helpers to build synthetic, lookahead-clean inputs.
# ---------------------------------------------------------------------------
def _make_close(n_days: int, drift: float = 0.001, seed: int = 0,
                  start: float = 100.0, end: str = "2025-12-31") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=drift, scale=0.01, size=n_days)
    closes = start * np.cumprod(1.0 + rets)
    end_ts = pd.Timestamp(end, tz="UTC")
    dates = pd.date_range(end=end_ts, periods=n_days, freq="D", tz="UTC")
    return pd.DataFrame({
        "timestamp": dates.view("int64") // 10**6,
        "datetime": dates,
        "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": 1000.0,
    })


def _ms_payload(n_days: int, base_value: float = 1.0,
                 slope: float = 0.001) -> pd.DataFrame:
    """Return a normalised market-structure series with `n_days` daily rows."""
    end_ts = pd.Timestamp("2025-12-31", tz="UTC")
    dates = pd.date_range(end=end_ts, periods=n_days, freq="D", tz="UTC")
    ts = (dates.view("int64") // 10**6).astype("int64")
    values = base_value * (1.0 + slope * np.arange(n_days))
    return pd.DataFrame({
        "timestamp": ts,
        "date": dates.strftime("%Y-%m-%d"),
        "source": "test",
        "dataset": "test",
        "value": values,
    })


def _wire(monkeypatch, btc_df, alt_dfs: Dict[str, pd.DataFrame],
           tvl=None, stables=None, btc_cap=None, btc_hash=None, btc_tx=None):
    by_symbol = {"BTC/USDT": btc_df}
    by_symbol.update(alt_dfs)

    def fake_load_candles(symbol, timeframe="1d"):
        if symbol not in by_symbol:
            raise FileNotFoundError(symbol)
        return by_symbol[symbol]

    monkeypatch.setattr(data_collector, "load_candles", fake_load_candles)
    monkeypatch.setattr(mdc, "load_total_tvl",
                         lambda: (tvl if tvl is not None else pd.DataFrame()))
    monkeypatch.setattr(mdc, "load_stablecoin_supply",
                         lambda: (stables if stables is not None else pd.DataFrame()))
    monkeypatch.setattr(mdc, "load_btc_market_cap",
                         lambda: (btc_cap if btc_cap is not None else pd.DataFrame()))
    monkeypatch.setattr(mdc, "load_btc_hash_rate",
                         lambda: (btc_hash if btc_hash is not None else pd.DataFrame()))
    monkeypatch.setattr(mdc, "load_btc_transactions",
                         lambda: (btc_tx if btc_tx is not None else pd.DataFrame()))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_signal_columns_match_spec():
    expected = [
        "timestamp", "date", "btc_close",
        "btc_return_30d", "btc_return_90d", "btc_above_200d_ma",
        "total_tvl", "total_tvl_return_30d", "total_tvl_return_90d",
        "stablecoin_supply",
        "stablecoin_supply_return_30d", "stablecoin_supply_return_90d",
        "btc_market_cap", "btc_market_cap_return_30d",
        "btc_hash_rate", "btc_hash_rate_return_30d",
        "btc_transactions", "btc_transactions_return_30d",
        "alt_basket_return_30d", "alt_basket_return_90d",
        "alt_basket_above_200d_ma_pct",
        "alt_basket_vs_btc_30d", "alt_basket_vs_btc_90d",
        "liquidity_score", "onchain_health_score",
        "alt_risk_score", "defensive_score",
        "market_structure_state",
    ]
    assert mss.SIGNAL_COLUMNS == expected


def test_timestamp_column_is_milliseconds_since_epoch(monkeypatch):
    """Regression: a previous version stored `timestamp` in kiloseconds
    when the merged `date` column carried `datetime64[ms]` precision —
    the allocator's `_signal_asof(asof_ts_ms)` then never matched the
    asset_frames timestamps (in real ms) and the strategy was always
    cash. Lock the unit explicitly."""
    btc = _make_close(40, seed=42)
    _wire(monkeypatch, btc, alt_dfs={
        s: _make_close(40, seed=i + 43) for i, s in enumerate(mss.ALT_UNIVERSE)
    })
    out = mss.compute_market_structure_signals(save=False)
    # 2025 timestamps in ms are ~1.7e12; in kiloseconds they would be ~1.7e6.
    assert out["timestamp"].iloc[-1] > 1e12, (
        f"timestamp likely not in ms: {out['timestamp'].iloc[-1]}"
    )
    # Round-trip through pd.to_datetime to confirm the dates parse back.
    parsed = pd.to_datetime(out["timestamp"].iloc[-1], unit="ms", utc=True)
    assert parsed.year >= 2024


def test_returns_use_only_past_data(monkeypatch):
    """`btc_return_30d` at row t must equal `close[t]/close[t-30] - 1`,
    not anything forward-shifted."""
    btc = _make_close(60, seed=1)
    _wire(monkeypatch, btc, alt_dfs={
        s: _make_close(60, seed=i + 2) for i, s in enumerate(mss.ALT_UNIVERSE)
    })
    out = mss.compute_market_structure_signals(save=False)
    closes = btc["close"].to_numpy()
    expected = closes[40] / closes[10] - 1.0
    assert out["btc_return_30d"].iloc[40] == pytest.approx(expected, rel=1e-9)


def test_btc_above_200d_ma_uses_only_past_data(monkeypatch):
    n = 250
    btc = _make_close(n, drift=0.005, seed=2)
    _wire(monkeypatch, btc, alt_dfs={
        s: _make_close(n, seed=i + 3) for i, s in enumerate(mss.ALT_UNIVERSE)
    })
    out = mss.compute_market_structure_signals(save=False)
    # Rolling mean needs 200 observations. Row 198 has only 199 → NaN
    # → fillna False. Row 199 is the FIRST row with a finite MA.
    assert bool(out["btc_above_200d_ma"].iloc[198]) is False
    closes = btc["close"]
    ma = closes.rolling(200, min_periods=200).mean()
    for row in (199, 220):
        expected = bool(closes.iloc[row] > ma.iloc[row])
        assert bool(out["btc_above_200d_ma"].iloc[row]) is expected, (
            f"row {row}: close={closes.iloc[row]}, ma={ma.iloc[row]}"
        )


def test_alt_universe_excludes_btc():
    assert "BTC/USDT" not in mss.ALT_UNIVERSE
    assert mss.ALT_UNIVERSE == [
        "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT",
        "XRP/USDT", "DOGE/USDT", "ADA/USDT", "LTC/USDT", "BNB/USDT",
    ]


def test_alt_breadth_is_correct_fraction(monkeypatch):
    """Construct: 5 alts trending up (above 200d MA) + 4 trending down.
    Latest breadth should be 5/9 ≈ 0.556."""
    n = 250
    btc = _make_close(n, drift=0.001, seed=4)
    alts: Dict[str, pd.DataFrame] = {}
    for i, s in enumerate(mss.ALT_UNIVERSE):
        drift = 0.005 if i < 5 else -0.005
        alts[s] = _make_close(n, drift=drift, seed=i + 5)
    _wire(monkeypatch, btc, alt_dfs=alts)
    out = mss.compute_market_structure_signals(save=False)
    last = out.iloc[-1]
    assert 0.4 < float(last["alt_basket_above_200d_ma_pct"]) < 0.7


def test_stablecoin_supply_return_correct(monkeypatch):
    """A 30-day +30 % growth in stables must yield return_30d ≈ 0.30."""
    n = 130
    btc = _make_close(n, seed=6)
    alts = {s: _make_close(n, seed=i + 7) for i, s in enumerate(mss.ALT_UNIVERSE)}
    stables = _ms_payload(n, base_value=100.0, slope=0.01)  # +1%/day
    _wire(monkeypatch, btc, alts, stables=stables)
    out = mss.compute_market_structure_signals(save=False)
    last_row = out.iloc[-1]
    expected = (100.0 * (1.0 + 0.01 * (n - 1))) / (
        100.0 * (1.0 + 0.01 * (n - 31))) - 1.0
    assert last_row["stablecoin_supply_return_30d"] == pytest.approx(
        expected, rel=1e-6)


def test_total_tvl_return_correct(monkeypatch):
    n = 130
    btc = _make_close(n, seed=8)
    alts = {s: _make_close(n, seed=i + 9) for i, s in enumerate(mss.ALT_UNIVERSE)}
    tvl = _ms_payload(n, base_value=50.0, slope=0.005)
    _wire(monkeypatch, btc, alts, tvl=tvl)
    out = mss.compute_market_structure_signals(save=False)
    last = out.iloc[-1]
    expected = (50.0 * (1.0 + 0.005 * (n - 1))) / (
        50.0 * (1.0 + 0.005 * (n - 31))) - 1.0
    assert last["total_tvl_return_30d"] == pytest.approx(expected, rel=1e-6)


def test_defensive_state_when_btc_below_200d_ma(monkeypatch):
    """Construct: BTC trends up 220 days then crashes 30 days. The last
    row should classify as `defensive`."""
    n = 250
    rng = np.random.default_rng(10)
    rets = list(rng.normal(loc=0.005, scale=0.005, size=n - 30))
    rets.extend(list(rng.normal(loc=-0.04, scale=0.005, size=30)))
    closes = 100.0 * np.cumprod(1.0 + np.array(rets))
    end_ts = pd.Timestamp("2025-12-31", tz="UTC")
    dates = pd.date_range(end=end_ts, periods=n, freq="D", tz="UTC")
    btc = pd.DataFrame({
        "timestamp": dates.view("int64") // 10**6, "datetime": dates,
        "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": 1000.0,
    })
    alts = {s: _make_close(n, drift=-0.01, seed=i + 11)
             for i, s in enumerate(mss.ALT_UNIVERSE)}
    _wire(monkeypatch, btc, alts,
           tvl=_ms_payload(n, base_value=50.0, slope=-0.001),
           stables=_ms_payload(n, base_value=100.0, slope=-0.001))
    out = mss.compute_market_structure_signals(save=False)
    last = out.iloc[-1]
    assert bool(last["btc_above_200d_ma"]) is False
    assert last["market_structure_state"] == "defensive"


def test_alt_risk_on_state_when_liquidity_and_breadth_positive(monkeypatch):
    """Construct: alts strongly outperform BTC, TVL+stables growing,
    BTC above 200d MA, breadth ≥ 0.5. Last row should be `alt_risk_on`."""
    n = 260
    btc = _make_close(n, drift=0.002, seed=12)
    alts = {s: _make_close(n, drift=0.006, seed=i + 13)
             for i, s in enumerate(mss.ALT_UNIVERSE)}
    _wire(monkeypatch, btc, alts,
           tvl=_ms_payload(n, base_value=50.0, slope=0.002),
           stables=_ms_payload(n, base_value=100.0, slope=0.001))
    out = mss.compute_market_structure_signals(save=False)
    last = out.iloc[-1]
    assert last["market_structure_state"] == "alt_risk_on"


def test_missing_source_data_yields_unknown(monkeypatch):
    """No TVL, no stables → can't compute the 90d returns those rules
    require → state must fall through to 'unknown'."""
    n = 260
    btc = _make_close(n, seed=14)
    alts = {s: _make_close(n, seed=i + 15) for i, s in enumerate(mss.ALT_UNIVERSE)}
    _wire(monkeypatch, btc, alts)  # tvl=None, stables=None
    out = mss.compute_market_structure_signals(save=False)
    # Last 10 rows should all be 'unknown' because TVL/stables are NaN.
    assert (out["market_structure_state"].iloc[-10:] == "unknown").all()


def test_partial_vs_full_no_lookahead(monkeypatch):
    """Compute the full series. Then truncate the inputs to the first
    180 rows and recompute. The first 180 rows of the full series MUST
    equal the truncated series — proves no future row leaks into past
    feature values."""
    n_full = 260
    btc_full = _make_close(n_full, drift=0.001, seed=20)
    alts_full = {s: _make_close(n_full, drift=0.001, seed=i + 21)
                  for i, s in enumerate(mss.ALT_UNIVERSE)}
    tvl_full = _ms_payload(n_full, base_value=50.0, slope=0.001)
    stables_full = _ms_payload(n_full, base_value=100.0, slope=0.0005)

    _wire(monkeypatch, btc_full, alts_full, tvl=tvl_full, stables=stables_full)
    out_full = mss.compute_market_structure_signals(save=False)

    # Now truncate every input to the first 180 days.
    cut = 180
    btc_part = btc_full.iloc[:cut].reset_index(drop=True)
    alts_part = {s: df.iloc[:cut].reset_index(drop=True)
                  for s, df in alts_full.items()}
    tvl_part = tvl_full.iloc[:cut].reset_index(drop=True)
    stables_part = stables_full.iloc[:cut].reset_index(drop=True)
    _wire(monkeypatch, btc_part, alts_part, tvl=tvl_part, stables=stables_part)
    out_part = mss.compute_market_structure_signals(save=False)

    # Compare numeric columns at the last bar of out_part vs the same
    # row of out_full (row 179).
    cmp_cols = ["btc_return_30d", "btc_return_90d",
                 "total_tvl_return_30d", "total_tvl_return_90d",
                 "stablecoin_supply_return_30d", "stablecoin_supply_return_90d",
                 "alt_basket_return_30d", "alt_basket_return_90d",
                 "alt_basket_above_200d_ma_pct"]
    full_row = out_full.iloc[cut - 1]
    part_row = out_part.iloc[cut - 1]
    for col in cmp_cols:
        a, b = full_row[col], part_row[col]
        if pd.isna(a) and pd.isna(b):
            continue
        assert a == pytest.approx(float(b), rel=1e-9, abs=1e-9), (
            f"column {col} differs between full and truncated runs at row "
            f"{cut-1}: full={a} part={b} — implies lookahead"
        )
