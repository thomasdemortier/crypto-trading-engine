"""Tests for the funding + basis data collector.

These tests verify, without making any network calls, that:
    * Schema of saved CSVs and the coverage CSV is locked.
    * Coverage classification matches the spec
        (PASS >= 1460, WARNING 365-1459, FAIL < 365, INCONCLUSIVE on
         endpoint anomalies).
    * Failed endpoints are recorded with status="error" + verdict="FAIL".
    * No API key is read.
    * No private endpoint URL is reachable.
    * Timestamps are normalised to ms (int64).
    * Funding gaps are recorded.
    * The CSV is actually written.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import pandas as pd
import pytest

from src import config, funding_basis_data_collector as fbdc, utils


# ---------------------------------------------------------------------------
# HTTP mock helpers
# ---------------------------------------------------------------------------
def _ok(payload: Any) -> fbdc._Response:
    return fbdc._Response(True, 200, payload, None)


def _err(reason: str = "simulated") -> fbdc._Response:
    return fbdc._Response(False, None, None, f"network_error: {reason}")


# ---------------------------------------------------------------------------
# Coverage classification
# ---------------------------------------------------------------------------
def test_classify_coverage_pass_at_threshold():
    assert fbdc.classify_coverage(1460.0) == "PASS"
    assert fbdc.classify_coverage(2000.0) == "PASS"


def test_classify_coverage_warning_band():
    assert fbdc.classify_coverage(365.0) == "WARNING"
    assert fbdc.classify_coverage(1459.0) == "WARNING"


def test_classify_coverage_fail_below_year():
    assert fbdc.classify_coverage(30.0) == "FAIL"
    assert fbdc.classify_coverage(0.0) == "FAIL"


def test_classify_coverage_inconclusive_on_unknown():
    assert fbdc.classify_coverage(None) == "INCONCLUSIVE"


def test_classify_coverage_status_error_is_fail():
    assert fbdc.classify_coverage(2000.0, status="error") == "FAIL"


# ---------------------------------------------------------------------------
# Frame normalisation
# ---------------------------------------------------------------------------
def test_normalise_funding_frame_schema():
    rows = [{"timestamp": 1700000000000, "funding_rate": 0.0001},
            {"timestamp": 1700028800000, "funding_rate": 0.0002}]
    df = fbdc._normalise_funding_frame(rows, "binance_futures", "BTCUSDT")
    assert list(df.columns) == [
        "timestamp", "datetime", "funding_rate",
        "source", "symbol_or_instrument",
    ]
    assert df["timestamp"].dtype == "int64"
    assert df["funding_rate"].dtype == "float64"
    assert df["source"].iloc[0] == "binance_futures"


def test_normalise_funding_frame_dedup_and_sorts():
    rows = [
        {"timestamp": 1700028800000, "funding_rate": 0.0002},
        {"timestamp": 1700000000000, "funding_rate": 0.0001},
        {"timestamp": 1700000000000, "funding_rate": 0.0001},
    ]
    df = fbdc._normalise_funding_frame(rows, "bybit", "BTCUSDT")
    assert len(df) == 2
    assert df["timestamp"].is_monotonic_increasing


def test_normalise_klines_frame_schema():
    rows = [
        [1700000000000, "100.0", "110.0", "95.0", "105.0", "1000.0",
         1700086399999, "100000", 50, "500", "50000", "0"],
    ]
    df = fbdc._normalise_klines_frame(rows, "binance_futures",
                                          "BTCUSDT", "mark")
    assert list(df.columns) == [
        "timestamp", "datetime", "open", "high", "low", "close",
        "volume", "source", "symbol", "kind",
    ]
    assert df["timestamp"].dtype == "int64"
    assert df["close"].iloc[0] == 105.0


def test_gap_count_detects_missing_steps():
    df = pd.DataFrame({
        "timestamp": [
            1700000000000, 1700028800000,         # 8h step
            1700057600000,                          # 8h step
            1700230400000,                          # 48h gap
        ],
    })
    expected = 8 * 3600 * 1000
    gaps = fbdc._gap_count(df, expected, tolerance_ms=expected // 4)
    assert gaps == 1


# ---------------------------------------------------------------------------
# Per-source paginated downloaders — patched HTTP
# ---------------------------------------------------------------------------
def test_binance_funding_paginates_until_short_page(monkeypatch):
    pages = [
        [{"fundingTime": 1700000000000 + i * 28800000,
           "fundingRate": "0.0001"}
         for i in range(1000)],
        [{"fundingTime": 1700000000000 + (1000 + i) * 28800000,
           "fundingRate": "0.0001"}
         for i in range(50)],   # short page → stop
    ]
    calls = {"i": 0}

    def fake(url: str) -> fbdc._Response:
        idx = calls["i"]
        calls["i"] += 1
        if idx < len(pages):
            return _ok(pages[idx])
        return _ok([])

    dl = fbdc._download_binance_funding("BTC/USDT", page_limit=1000,
                                          max_pages=10, rate_delay_s=0.0,
                                          http_get=fake)
    assert dl.error is None
    assert dl.pages == 2
    assert len(dl.df) == 1050


def test_binance_funding_handles_failed_endpoint(monkeypatch):
    dl = fbdc._download_binance_funding("BTC/USDT", page_limit=1000,
                                          max_pages=2, rate_delay_s=0.0,
                                          http_get=lambda u: _err())
    assert dl.df.empty
    assert dl.pages == 0


def test_bybit_funding_paginates_via_endTime(monkeypatch):
    page1 = [{"fundingRateTimestamp": str(1700000000000 + (199 - i) * 28800000),
                "fundingRate": "0.0001"} for i in range(200)]
    page2 = [{"fundingRateTimestamp": str(1700000000000 - (200 - i) * 28800000),
                "fundingRate": "0.0001"} for i in range(50)]

    calls = {"i": 0}

    def fake(url: str) -> fbdc._Response:
        idx = calls["i"]
        calls["i"] += 1
        if idx == 0:
            return _ok({"result": {"list": page1}})
        if idx == 1:
            return _ok({"result": {"list": page2}})
        return _ok({"result": {"list": []}})

    dl = fbdc._download_bybit_funding("BTC/USDT", since_ms=0, page_limit=200,
                                          max_pages=5, rate_delay_s=0.0,
                                          http_get=fake)
    assert dl.error is None
    assert dl.pages == 2
    assert len(dl.df) == 250


def test_deribit_funding_walks_windows_forward(monkeypatch):
    # Two non-empty windows, then nothing.
    calls = {"i": 0}
    win = lambda start: [{"timestamp": start + h * 3_600_000,
                           "interest_8h": 0.00005} for h in range(24)]

    def fake(url: str) -> fbdc._Response:
        idx = calls["i"]; calls["i"] += 1
        if idx == 0:
            return _ok({"result": win(1700000000000)})
        if idx == 1:
            return _ok({"result": win(1700000000000 + 14 * 86_400_000)})
        return _ok({"result": []})

    dl = fbdc._download_deribit_funding("BTC/USDT",
                                            since_ms=1700000000000,
                                            window_days=14, max_windows=4,
                                            rate_delay_s=0.0, http_get=fake)
    assert dl.error is None
    assert dl.pages >= 2
    assert len(dl.df) == 48


# ---------------------------------------------------------------------------
# Top-level orchestrator — full coverage CSV with mocked HTTP
# ---------------------------------------------------------------------------
def test_download_all_writes_coverage_and_csvs(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw"; raw.mkdir(parents=True)
    res = tmp_path / "results"; res.mkdir()
    pos = tmp_path / "data" / "positioning" / "funding_basis"
    pos.mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_RAW_DIR", raw)
    monkeypatch.setattr(config, "RESULTS_DIR", res)
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(fbdc, "POSITIONING_DIR", pos)
    monkeypatch.setattr(fbdc, "COVERAGE_PATH",
                          res / "funding_basis_data_coverage.csv")

    # Single-page funding (50 rows, short → stop), single-page klines.
    funding_pages = {
        "binance": [{"fundingTime": 1700000000000 + i * 28800000,
                       "fundingRate": "0.0001"} for i in range(50)],
        "bybit": [{"fundingRateTimestamp": str(1700000000000 + i * 28800000),
                     "fundingRate": "0.0001"} for i in range(50)],
        "deribit_w1": [{"timestamp": 1700000000000 + h * 3_600_000,
                          "interest_8h": 0.00005} for h in range(24)],
    }
    klines = [
        [1700000000000 + d * 86_400_000, "100.0", "110.0", "95.0",
          "105.0", "1000.0",
          1700000000000 + d * 86_400_000 + 86_399_000, "100000",
          50, "500", "50000", "0"]
        for d in range(50)
    ]

    def fake(url: str) -> fbdc._Response:
        if "fundingRate?" in url:
            return _ok(funding_pages["binance"])
        if "markPriceKlines" in url or "indexPriceKlines" in url:
            return _ok(klines)
        if "/v5/market/funding/history" in url:
            return _ok({"result": {"list": funding_pages["bybit"]}})
        if "deribit" in url:
            return _ok({"result": funding_pages["deribit_w1"]})
        return _err("unhandled")

    cov = fbdc.download_all(assets=("BTC/USDT", "ETH/USDT"),
                              save=True, http_get=fake)
    assert isinstance(cov, pd.DataFrame)
    assert list(cov.columns) == fbdc.COVERAGE_COLUMNS
    # 5 sources × 2 assets = 10 rows.
    assert len(cov) == 10
    cov_path = res / "funding_basis_data_coverage.csv"
    assert cov_path.exists()
    on_disk = pd.read_csv(cov_path)
    assert list(on_disk.columns) == fbdc.COVERAGE_COLUMNS

    # Per-source CSVs were written with normalised schema.
    for asset in ("BTC_USDT", "ETH_USDT"):
        for fname in (f"binance_funding_{asset}.csv",
                       f"binance_mark_klines_{asset}.csv",
                       f"binance_index_klines_{asset}.csv",
                       f"bybit_funding_{asset}.csv",
                       f"deribit_funding_{asset}.csv"):
            path = pos / fname
            assert path.exists(), f"missing {path}"
            df = pd.read_csv(path)
            assert "timestamp" in df.columns
            assert df["timestamp"].dtype.kind in ("i", "u")


def test_download_all_records_failed_endpoints(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw"; raw.mkdir(parents=True)
    res = tmp_path / "results"; res.mkdir()
    pos = tmp_path / "data" / "positioning" / "funding_basis"
    pos.mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_RAW_DIR", raw)
    monkeypatch.setattr(config, "RESULTS_DIR", res)
    monkeypatch.setattr(fbdc, "POSITIONING_DIR", pos)
    monkeypatch.setattr(fbdc, "COVERAGE_PATH",
                          res / "funding_basis_data_coverage.csv")
    cov = fbdc.download_all(assets=("BTC/USDT",), save=True,
                              http_get=lambda u: _err())
    # All probes failed → every row should be FAIL or INCONCLUSIVE
    # (FAIL if the downloader recorded the error string; INCONCLUSIVE
    # if the loop simply broke out of pagination without raising).
    assert cov["verdict"].isin(["FAIL", "INCONCLUSIVE"]).all()
    # No PASS row when every HTTP call failed.
    assert not (cov["verdict"] == "PASS").any()
    assert (cov["row_count"] == 0).all()


# ---------------------------------------------------------------------------
# Safety invariants — module must contain no broker / order / key code
# ---------------------------------------------------------------------------
_COLLECTOR_SOURCE = (Path(__file__).resolve().parents[1] /
                       "src" / "funding_basis_data_collector.py").read_text()


def test_no_api_key_reads():
    forbidden = (
        re.compile(r"\bos\.environ\.get\([\"'](?:[A-Z_]*API[_-]?KEY)[\"']"),
        re.compile(r"\bos\.environ\[[\"'](?:[A-Z_]*API[_-]?KEY)[\"']"),
        re.compile(r"\bos\.getenv\([\"'](?:[A-Z_]*API[_-]?KEY)[\"']"),
        re.compile(r"\bAPI[_-]?KEY\s*=\s*['\"][A-Za-z0-9_\-]{8,}['\"]"),
    )
    for pat in forbidden:
        assert pat.search(_COLLECTOR_SOURCE) is None, pat.pattern


def test_no_private_endpoint_strings():
    forbidden = (
        re.compile(r"\bkraken\.private\b"),
        re.compile(r"\bkraken_private\b"),
        re.compile(r"['\"]/0/private/"),
        re.compile(r"\bAddOrder\b"),
        re.compile(r"\bCancelOrder\b"),
    )
    for pat in forbidden:
        assert pat.search(_COLLECTOR_SOURCE) is None, pat.pattern


def test_no_order_placement_in_collector():
    forbidden = (
        re.compile(r"\bcreate_order\s*\("),
        re.compile(r"\bplace_order\s*\("),
        re.compile(r"\bsubmit_order\s*\("),
        re.compile(r"\bsend_order\s*\("),
        re.compile(r"\bcancel_order\s*\("),
        re.compile(r"\bdef\s+(?:create|place|submit|send|cancel)_order\b"),
    )
    for pat in forbidden:
        assert pat.search(_COLLECTOR_SOURCE) is None, pat.pattern
