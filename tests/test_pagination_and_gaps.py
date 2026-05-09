"""Tests for the data-collector pagination fix + gap validation.

These tests do NOT touch the network. They use a mock exchange object
that emulates two failure modes seen in the wild:
  * "stuck" — every call returns the SAME 720 most-recent candles
    (Kraken behavior).
  * "deep" — proper backwards pagination, advancing every call
    (Binance behavior).
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest

from src import config, data_collector


TF_MS_4H = 4 * 60 * 60 * 1000


# ---------------------------------------------------------------------------
# Mock exchange helpers
# ---------------------------------------------------------------------------
class _StuckExchange:
    """Emulates Kraken: every call returns the most-recent N candles
    regardless of the `since` argument. The pagination loop should detect
    this and exit early (without an infinite loop)."""
    id = "stuck"
    rateLimit = 0  # ms

    def __init__(self, n: int, end_ms: int, tf_ms: int = TF_MS_4H):
        self._n = n
        self._end = int(end_ms)
        self._tf = int(tf_ms)

    def milliseconds(self) -> int:
        return self._end

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        # Always returns the most-recent N bars, ignoring `since`.
        rows: List[list] = []
        for i in range(self._n):
            ts = self._end - (self._n - 1 - i) * self._tf
            rows.append([ts, 100.0, 101.0, 99.0, 100.5, 1000.0])
        return rows


class _DeepExchange:
    """Emulates Binance: respects `since`, returns up to `limit` bars
    starting from the requested timestamp going forward."""
    id = "deep"
    rateLimit = 0

    def __init__(self, start_ms: int, end_ms: int, tf_ms: int = TF_MS_4H,
                 batch: int = 1000):
        self._start = int(start_ms)
        self._end = int(end_ms)
        self._tf = int(tf_ms)
        self._batch = int(batch)

    def milliseconds(self) -> int:
        return self._end

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        cursor = max(int(since or self._start), self._start)
        if cursor >= self._end:
            return []
        bars = min(int(limit or self._batch), self._batch)
        rows: List[list] = []
        ts = cursor
        while len(rows) < bars and ts < self._end:
            rows.append([ts, 100.0, 101.0, 99.0, 100.5, 1000.0])
            ts += self._tf
        return rows


# ---------------------------------------------------------------------------
# Pagination behavior
# ---------------------------------------------------------------------------
def test_stuck_pagination_breaks_out_quickly():
    """If the exchange ignores `since`, the loop must NOT run away —
    after detecting that `first_ts` doesn't advance, it bails out."""
    end_ms = 1_700_000_000_000
    ex = _StuckExchange(n=720, end_ms=end_ms, tf_ms=TF_MS_4H)
    since = end_ms - 5000 * TF_MS_4H  # request 5000 bars of history
    rows = data_collector._fetch_ohlcv_paginated(
        ex, "BTC/USDT", "4h", since_ms=since, limit_per_call=720,
    )
    # Two iterations max: first iter fetches the 720 stuck bars; second
    # iter detects the stuck condition and breaks.
    df = data_collector._rows_to_df(rows)
    assert len(df) == 720, f"expected exactly 720 unique bars, got {len(df)}"


def test_deep_pagination_walks_full_history():
    """Binance-style: each call advances; we should accumulate the full
    requested span."""
    end_ms = 1_700_000_000_000
    days = 200
    start_ms = end_ms - days * 24 * 3600 * 1000
    ex = _DeepExchange(start_ms=start_ms, end_ms=end_ms,
                       tf_ms=TF_MS_4H, batch=1000)
    rows = data_collector._fetch_ohlcv_paginated(
        ex, "BTC/USDT", "4h", since_ms=start_ms, limit_per_call=1000,
    )
    df = data_collector._rows_to_df(rows)
    expected = days * 24 // 4  # 4h bars in `days` days
    # Allow ±1 for boundary inclusivity.
    assert abs(len(df) - expected) <= 2, (
        f"expected ~{expected} 4h bars over {days} days, got {len(df)}"
    )


# ---------------------------------------------------------------------------
# Gap validation
# ---------------------------------------------------------------------------
def test_validate_gaps_clean_continuous_series():
    end_ms = 1_700_000_000_000
    ts = np.arange(end_ms - 100 * TF_MS_4H, end_ms, TF_MS_4H, dtype="int64")
    df = pd.DataFrame({"timestamp": ts})
    info = data_collector.validate_gaps(df, "4h")
    assert info["gap_count"] == 0
    assert info["actual_bars"] == len(ts)
    assert info["expected_bars"] == len(ts)
    assert info["largest_gap_bars"] == 0


def test_validate_gaps_detects_missing_bars():
    end_ms = 1_700_000_000_000
    ts = np.arange(end_ms - 50 * TF_MS_4H, end_ms, TF_MS_4H, dtype="int64").tolist()
    # Drop bars 20 and 21 to create a hole.
    del ts[20:22]
    df = pd.DataFrame({"timestamp": ts})
    info = data_collector.validate_gaps(df, "4h")
    assert info["gap_count"] == 1
    assert info["largest_gap_bars"] >= 3   # the gap spans 3 bar-widths


def test_validate_gaps_handles_empty():
    info = data_collector.validate_gaps(pd.DataFrame(), "4h")
    assert info == {"expected_bars": 0, "actual_bars": 0,
                    "gap_count": 0, "largest_gap_bars": 0}


# ---------------------------------------------------------------------------
# Per-exchange chunk limit
# ---------------------------------------------------------------------------
def test_chunk_limit_per_exchange_uses_config_map():
    assert data_collector._chunk_limit_for("binance") == 1000
    assert data_collector._chunk_limit_for("kraken") == 720
    # Unknown exchanges fall back to the global default.
    assert data_collector._chunk_limit_for("unknown") == config.FETCH_CHUNK_LIMIT


# ---------------------------------------------------------------------------
# download_symbol — fallback merging without touching the network
# ---------------------------------------------------------------------------
def test_download_symbol_merges_when_primary_short(monkeypatch, tmp_path):
    """Primary returns less than 50% of the requested span (Kraken-style
    truncation); fallback covers the full window. `download_symbol` must
    merge both, dedup, and save."""
    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path)
    end_ms = 1_700_000_000_000
    days = 1000  # ≈ 6000 4h bars expected; threshold = 3000.
    start_ms = end_ms - days * 24 * 3600 * 1000

    # Primary: only the most-recent 720 bars (Kraken-like truncation).
    primary_df = pd.DataFrame({
        "timestamp": np.arange(end_ms - 720 * TF_MS_4H, end_ms,
                               TF_MS_4H, dtype="int64"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "volume": 1000.0,
    })
    primary_df["datetime"] = pd.to_datetime(primary_df["timestamp"], unit="ms",
                                            utc=True)

    # Fallback: covers the full requested window (older history too).
    fallback_df = pd.DataFrame({
        "timestamp": np.arange(start_ms, end_ms, TF_MS_4H, dtype="int64"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "volume": 1000.0,
    })
    fallback_df["datetime"] = pd.to_datetime(fallback_df["timestamp"], unit="ms",
                                             utc=True)

    calls = {"primary": 0, "fallback": 0}

    def _fake(ex_name, symbol, timeframe, days_arg):
        if ex_name == config.PRIMARY_EXCHANGE:
            calls["primary"] += 1
            return primary_df
        calls["fallback"] += 1
        return fallback_df

    monkeypatch.setattr(data_collector, "_fetch_from_exchange", _fake)

    p = data_collector.download_symbol("BTC/USDT", "4h", days=days, refresh=True)
    assert p.exists()
    df = pd.read_csv(p)
    # Primary's recent 720 bars are a subset of the fallback's full
    # window, so dedup yields exactly the fallback length.
    assert len(df) == len(fallback_df)
    assert calls["primary"] == 1
    assert calls["fallback"] == 1


def test_download_symbol_skips_fallback_when_primary_sufficient(monkeypatch, tmp_path):
    """Primary returns enough bars to clear the 50% threshold — fallback
    must NOT be invoked (avoids burning the fallback exchange's rate
    limit when not needed)."""
    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path)
    end_ms = 1_700_000_000_000
    days = 100  # expects 600 4h bars; threshold = 300.

    primary_df = pd.DataFrame({
        "timestamp": np.arange(end_ms - 600 * TF_MS_4H, end_ms,
                               TF_MS_4H, dtype="int64"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "volume": 1000.0,
    })
    primary_df["datetime"] = pd.to_datetime(primary_df["timestamp"], unit="ms",
                                            utc=True)
    calls = {"primary": 0, "fallback": 0}

    def _fake(ex_name, *_args, **_kwargs):
        if ex_name == config.PRIMARY_EXCHANGE:
            calls["primary"] += 1
            return primary_df
        calls["fallback"] += 1
        return None

    monkeypatch.setattr(data_collector, "_fetch_from_exchange", _fake)
    data_collector.download_symbol("BTC/USDT", "4h", days=days, refresh=True)
    assert calls["primary"] == 1
    assert calls["fallback"] == 0
