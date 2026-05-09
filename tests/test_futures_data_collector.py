"""Tests for `src.futures_data_collector`.

These tests do NOT touch the network. The HTTP client is monkey-patched
to return canned Binance-shaped responses so we can exercise pagination,
deduplication, sort order, gap validation, and the coverage audit
without any external dependency.

What we are checking, end-to-end:
  * Funding-rate parser handles a sample Binance response and produces
    the documented column set.
  * Funding-rate pagination walks forward via `startTime` and dedupes
    overlap.
  * Funding-rate stuck-pagination is detected (same defensive pattern
    as the v1 Kraken fix).
  * OI parser produces the documented column set; the empirical ~30-day
    cap is surfaced honestly via the coverage audit `notes` column.
  * Symbol normalisation accepts both `BTC/USDT` and `BTCUSDT`.
  * `available_futures_symbols` only returns symbols with both files.
  * `audit_futures_coverage` flags missing CSVs as `no_csv` and tags
    OI rows that hug the public cap.
  * No private API keys, no live-trading code path, no Kraken import.
"""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from src import config, futures_data_collector as fdc


# ---------------------------------------------------------------------------
# Sample API payloads (shapes copied from the live Binance docs)
# ---------------------------------------------------------------------------
def _funding_row(symbol: str, ts_ms: int, rate: float) -> Dict[str, Any]:
    return {"symbol": symbol, "fundingTime": int(ts_ms),
            "fundingRate": str(rate), "markPrice": "30000.0"}


def _oi_row(symbol: str, ts_ms: int, oi: float, oi_value: float) -> Dict[str, Any]:
    return {"symbol": symbol, "timestamp": int(ts_ms),
            "sumOpenInterest": str(oi), "sumOpenInterestValue": str(oi_value)}


# ---------------------------------------------------------------------------
# Symbol handling
# ---------------------------------------------------------------------------
def test_no_slash_handles_both_forms():
    assert fdc._no_slash("BTC/USDT") == "BTCUSDT"
    assert fdc._no_slash("BTCUSDT") == "BTCUSDT"


# ---------------------------------------------------------------------------
# Funding rate parser + pagination
# ---------------------------------------------------------------------------
def _install_funding_responses(monkeypatch, batches: List[List[Dict[str, Any]]]):
    """Make `_http_get` return the next batch from `batches` per call."""
    queue = list(batches)

    def fake_get(url, params, timeout):
        if not queue:
            return []
        return queue.pop(0)

    monkeypatch.setattr(fdc, "_http_get", fake_get)


def test_fetch_funding_parses_sample_payload(monkeypatch):
    end = pd.Timestamp.utcnow().value // 10**6
    batch = [
        _funding_row("BTCUSDT", end - 16 * 3_600_000, 0.0001),
        _funding_row("BTCUSDT", end - 8 * 3_600_000, -0.0002),
    ]
    _install_funding_responses(monkeypatch, [batch])
    df = fdc.fetch_funding_rate_history("BTCUSDT", days=2, sleep_seconds=0)
    assert list(df.columns) == ["timestamp", "symbol", "funding_rate",
                                 "funding_time", "source"]
    assert len(df) == 2
    # Sorted ascending by funding_time.
    assert df["funding_time"].is_monotonic_increasing
    assert df["funding_rate"].iloc[0] == pytest.approx(0.0001)
    assert df["funding_rate"].iloc[1] == pytest.approx(-0.0002)
    assert (df["source"] == "binance_fapi_v1_funding_rate").all()


def test_fetch_funding_dedupes_overlap_between_batches(monkeypatch):
    end = pd.Timestamp.utcnow().value // 10**6
    t0 = end - 24 * 3_600_000
    t1 = end - 16 * 3_600_000
    t2 = end - 8 * 3_600_000
    # Second batch deliberately repeats `t1` and `t2` to simulate a
    # paging response that overlaps the previous one.
    b1 = [_funding_row("BTCUSDT", t0, 0.0001),
          _funding_row("BTCUSDT", t1, 0.0002)]
    b2 = [_funding_row("BTCUSDT", t1, 0.0002),
          _funding_row("BTCUSDT", t2, 0.0003)]
    # Make the first batch be 1000 rows so pagination tries again — we
    # pad with synthetic earlier rows that shouldn't survive dedup.
    pad = [_funding_row("BTCUSDT", t0 - i * 1000, 0.0) for i in range(1, 999)]
    _install_funding_responses(monkeypatch, [b1 + pad, b2])
    df = fdc.fetch_funding_rate_history("BTCUSDT", days=2, sleep_seconds=0)
    # The two real timestamps + the 998 padding rows = 1000 unique;
    # then b2 adds exactly one more (`t2`).
    assert df["funding_time"].is_unique
    assert t2 in set(df["funding_time"].tolist())


def test_fetch_funding_detects_stuck_pagination(monkeypatch):
    """If an upstream API returns the SAME first row on every batch, we
    must bail out — same defensive pattern that fixed Kraken in v1."""
    end = pd.Timestamp.utcnow().value // 10**6
    fixed = [_funding_row("BTCUSDT", end - i * 8 * 3_600_000, 0.0)
             for i in range(1000)][::-1]
    # Always return the same 1000-row batch — pagination would otherwise loop forever.
    calls = {"n": 0}

    def fake_get(url, params, timeout):
        calls["n"] += 1
        return fixed

    monkeypatch.setattr(fdc, "_http_get", fake_get)
    df = fdc.fetch_funding_rate_history("BTCUSDT", days=400, sleep_seconds=0)
    # After detecting the stuck condition we exit; a sane upper bound:
    assert calls["n"] <= 3, f"stuck pagination not detected, called {calls['n']} times"
    assert df["funding_time"].is_unique


def test_fetch_funding_empty_response_is_safe(monkeypatch):
    monkeypatch.setattr(fdc, "_http_get", lambda *a, **k: [])
    df = fdc.fetch_funding_rate_history("BTCUSDT", days=10, sleep_seconds=0)
    assert df.empty
    assert list(df.columns) == ["timestamp", "symbol", "funding_rate",
                                 "funding_time", "source"]


# ---------------------------------------------------------------------------
# Open-interest parser (~30-day public cap is documented, not enforced
# in code — we just check the parser shape)
# ---------------------------------------------------------------------------
def test_fetch_oi_parses_sample_payload(monkeypatch):
    end = pd.Timestamp.utcnow().value // 10**6
    batch = [
        _oi_row("BTCUSDT", end - 2 * 86_400_000, 500_000.0, 1.5e10),
        _oi_row("BTCUSDT", end - 1 * 86_400_000, 510_000.0, 1.55e10),
    ]
    monkeypatch.setattr(fdc, "_http_get", lambda *a, **k: batch
                        if k.get("params", {}).get("startTime", 0) or True else [])
    # Single-batch return:
    monkeypatch.setattr(fdc, "_http_get", lambda *a, **k: batch)
    df = fdc.fetch_open_interest_history("BTCUSDT", days=3,
                                          period="1d", sleep_seconds=0)
    assert list(df.columns) == ["timestamp", "symbol", "open_interest",
                                 "open_interest_value", "source"]
    # Two unique timestamps, sorted.
    assert len(df) == 2
    assert df["timestamp"].is_monotonic_increasing
    assert df["open_interest"].iloc[1] == pytest.approx(510_000.0)
    assert (df["source"] == "binance_futures_data_oi_hist").all()


# ---------------------------------------------------------------------------
# Gap validators
# ---------------------------------------------------------------------------
def test_validate_funding_gaps_clean_8h_series():
    end = 1_700_000_000_000
    ts = [end - i * 8 * 3_600_000 for i in range(10)][::-1]
    df = pd.DataFrame({"funding_time": ts})
    info = fdc.validate_funding_gaps(df)
    assert info["row_count"] == 10
    assert info["gap_count"] == 0


def test_validate_funding_gaps_detects_missing_funding_event():
    end = 1_700_000_000_000
    ts = [end - i * 8 * 3_600_000 for i in range(10)][::-1]
    # Drop one event to create a 16h gap.
    del ts[5]
    df = pd.DataFrame({"funding_time": ts})
    info = fdc.validate_funding_gaps(df)
    assert info["gap_count"] >= 1
    assert info["largest_gap_hours"] >= 16.0


def test_validate_funding_gaps_handles_empty():
    info = fdc.validate_funding_gaps(pd.DataFrame())
    assert info["row_count"] == 0
    assert info["gap_count"] == 0


def test_validate_oi_gaps_clean_daily_series():
    end = 1_700_000_000_000
    ts = [end - i * 86_400_000 for i in range(15)][::-1]
    df = pd.DataFrame({"timestamp": ts})
    info = fdc.validate_oi_gaps(df, period="1d")
    assert info["row_count"] == 15
    assert info["gap_count"] == 0


# ---------------------------------------------------------------------------
# Loaders + symbol availability
# ---------------------------------------------------------------------------
def test_load_returns_empty_frame_when_no_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    df_f = fdc.load_funding_rate("BTCUSDT")
    df_o = fdc.load_open_interest("BTCUSDT")
    assert df_f.empty and df_o.empty
    # Schema preserved even when empty — downstream callers can rely on it.
    assert "funding_rate" in df_f.columns
    assert "open_interest" in df_o.columns


def test_available_futures_symbols_only_returns_symbols_with_both_files(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    # BTCUSDT: both files exist.
    f_btc = fdc._futures_csv_path("funding_rates", "BTCUSDT")
    o_btc = fdc._futures_csv_path("open_interest", "BTCUSDT")
    f_btc.write_text("timestamp,symbol,funding_rate,funding_time,source\n1,BTCUSDT,0,1,x\n")
    o_btc.write_text("timestamp,symbol,open_interest,open_interest_value,source\n1,BTCUSDT,0,0,x\n")
    # ETHUSDT: only funding exists.
    f_eth = fdc._futures_csv_path("funding_rates", "ETHUSDT")
    f_eth.write_text("timestamp,symbol,funding_rate,funding_time,source\n1,ETHUSDT,0,1,x\n")

    avail = fdc.available_futures_symbols(("BTCUSDT", "ETHUSDT", "SOLUSDT"))
    assert avail == ["BTCUSDT"]


# ---------------------------------------------------------------------------
# Coverage audit
# ---------------------------------------------------------------------------
def test_audit_coverage_flags_missing_csvs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    out = fdc.audit_futures_coverage(symbols=("BTCUSDT",), save=True)
    # One row per (symbol, dataset) — so 2 rows total.
    assert len(out) == 2
    assert (out["row_count"] == 0).all()
    assert (out["missing_reason"] == "no_csv").all()
    # Persisted CSV exists.
    assert (tmp_path / "results" / "futures_data_coverage.csv").exists()


def test_audit_coverage_oi_note_mentions_30_day_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    end = pd.Timestamp.utcnow().value // 10**6
    # 25 daily rows — well under MIN_DAYS_FOR_RESEARCH (120) and within
    # the public OI cap (~30 days).
    oi = pd.DataFrame({
        "timestamp": [end - i * 86_400_000 for i in range(25)][::-1],
        "symbol": ["BTCUSDT"] * 25,
        "open_interest": [500_000.0] * 25,
        "open_interest_value": [1.5e10] * 25,
        "source": ["binance_futures_data_oi_hist"] * 25,
    })
    oi.to_csv(fdc._futures_csv_path("open_interest", "BTCUSDT"), index=False)
    # Empty funding file — keeps the audit row but won't drive a verdict.
    out = fdc.audit_futures_coverage(symbols=("BTCUSDT",), save=False)
    oi_row = out[out["dataset"] == "open_interest"].iloc[0]
    assert oi_row["row_count"] == 25
    assert not oi_row["enough_for_research"]
    assert "OI capped" in oi_row["notes"]


# ---------------------------------------------------------------------------
# Safety: no API keys, no live trading, no Kraken
# ---------------------------------------------------------------------------
def test_module_uses_only_public_endpoints():
    src = inspect.getsource(fdc)
    assert "fapi.binance.com" in src
    # Strip docstrings / comments — the safety claim is about code, not text.
    code_lines = []
    in_doc = False
    for raw in src.splitlines():
        stripped = raw.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            in_doc = not in_doc
            continue
        if in_doc or stripped.startswith("#"):
            continue
        code_lines.append(raw)
    code = "\n".join(code_lines).lower()
    for forbidden in ("/sapi/", "/wapi/", "x-mbx-apikey",
                      "apikey", "api_key", "kraken"):
        assert forbidden not in code, (
            f"{forbidden!r} appears in futures_data_collector code — "
            "public-only invariant"
        )


def test_download_calls_paper_only_guard(monkeypatch, tmp_path):
    """`download_futures_data` must call `assert_paper_only()` before any
    network work — same guard the v1 collector uses."""
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    called = {"n": 0}

    from src import utils as _utils

    real = _utils.assert_paper_only

    def spy():
        called["n"] += 1
        return real()

    monkeypatch.setattr(_utils, "assert_paper_only", spy)
    # Also stub the network functions so the test doesn't even try to
    # talk to fapi.binance.com.
    monkeypatch.setattr(fdc, "fetch_funding_rate_history",
                        lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(fdc, "fetch_open_interest_history",
                        lambda *a, **k: pd.DataFrame())
    fdc.download_futures_data(symbols=("BTCUSDT",), days=1, refresh=True)
    assert called["n"] == 1
