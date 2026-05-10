"""Tests for `src/fx_crypto_source_audit.py`. All offline."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src import config, fx_crypto_source_audit as fx, safety_lock


def _ok(payload: Any = None, text: str = "") -> fx._Response:
    return fx._Response(True, 200, payload, text or "", None)


def _err(error: str = "simulated") -> fx._Response:
    return fx._Response(False, None, None, None,
                          f"network_error: {error}")


# ---------------------------------------------------------------------------
# Schema + classifier
# ---------------------------------------------------------------------------
def test_audit_columns_locked():
    expected = {
        "market", "source", "asset", "field_type",
        "endpoint_or_source", "requires_api_key", "free_access",
        "actual_start", "actual_end", "coverage_days",
        "granularity", "usable_for_research",
        "decision_status", "notes",
    }
    assert set(fx.AUDIT_COLUMNS) == expected


def test_thresholds_locked():
    assert fx.COVERAGE_PASS_DAYS == 1460
    assert fx.COVERAGE_WARNING_DAYS == 365


def test_classify_pass_at_threshold():
    assert fx.classify_coverage(1460.0) == "PASS"


def test_classify_warning_band():
    assert fx.classify_coverage(800.0) == "WARNING"


def test_classify_fail_below_year():
    assert fx.classify_coverage(30.0) == "FAIL"


def test_classify_unknown_coverage_is_inconclusive():
    assert fx.classify_coverage(None) == "INCONCLUSIVE"


def test_classify_status_error_is_fail():
    assert fx.classify_coverage(2000.0, status="error") == "FAIL"


def test_classify_snapshot_only_is_fail():
    assert fx.classify_coverage(0.0, is_snapshot_only=True) == "FAIL"


def test_classify_requires_key_is_inconclusive():
    assert fx.classify_coverage(2000.0,
                                    requires_key=True) == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# ECB probe with synthetic CSV
# ---------------------------------------------------------------------------
def test_ecb_probe_pass_on_long_csv():
    csv_text = (
        "KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,"
        "TIME_PERIOD,OBS_VALUE\n"
        "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,1999-01-04,1.18\n"
        "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2026-05-08,1.10\n"
    )
    row = fx._probe_ecb_eurquoted(
        "EUR/USD", "USD",
        http_get=lambda url: _ok(text=csv_text),
    )
    assert row["decision_status"] == "PASS"
    assert row["coverage_days"] is not None
    assert row["coverage_days"] > 1460


def test_ecb_probe_inconclusive_on_empty_csv():
    csv_text = (
        "KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,"
        "TIME_PERIOD,OBS_VALUE\n"
    )
    row = fx._probe_ecb_eurquoted(
        "EUR/USD", "USD",
        http_get=lambda url: _ok(text=csv_text),
    )
    assert row["decision_status"] == "INCONCLUSIVE"


def test_ecb_probe_fail_on_network_error():
    row = fx._probe_ecb_eurquoted(
        "EUR/USD", "USD", http_get=lambda url: _err(),
    )
    assert row["decision_status"] == "FAIL"


# ---------------------------------------------------------------------------
# Frankfurter probe
# ---------------------------------------------------------------------------
def test_frankfurter_probe_pass_on_long_payload():
    payload = {
        "amount": 1.0, "base": "EUR",
        "start_date": "2010-01-01", "end_date": "2026-05-08",
        "rates": {
            "2010-01-04": {"USD": 1.44},
            "2026-05-08": {"USD": 1.10},
        },
    }
    row = fx._probe_frankfurter(
        "EUR/USD", http_get=lambda url: _ok(payload=payload),
    )
    assert row["decision_status"] == "PASS"


def test_frankfurter_probe_fail_on_network_error():
    row = fx._probe_frankfurter(
        "EUR/USD", http_get=lambda url: _err(),
    )
    assert row["decision_status"] == "FAIL"


# ---------------------------------------------------------------------------
# LBMA gold probe
# ---------------------------------------------------------------------------
def test_lbma_probe_pass_on_long_payload():
    payload = [
        {"d": "1968-04-01", "v": [37.7, 15.68, None]},
        {"d": "2026-05-08", "v": [3300.0, 2700.0, None]},
    ]
    row = fx._probe_lbma_gold_pm(
        http_get=lambda url: _ok(payload=payload),
    )
    assert row["decision_status"] == "PASS"


def test_lbma_probe_fail_on_empty_payload():
    row = fx._probe_lbma_gold_pm(
        http_get=lambda url: _ok(payload=[]),
    )
    assert row["decision_status"] == "FAIL"


# ---------------------------------------------------------------------------
# Yahoo Finance gold futures probe
# ---------------------------------------------------------------------------
def test_yfinance_gold_probe_pass_on_long_history():
    payload = {
        "chart": {"result": [{"timestamp": [
            1262304000,   # 2010-01-01
            1778457600,   # 2026-05-10
        ]}], "error": None},
    }
    row = fx._probe_yfinance_gold_futures(
        http_get=lambda url: _ok(payload=payload),
    )
    assert row["decision_status"] == "PASS"


def test_yfinance_gold_probe_inconclusive_on_block():
    row = fx._probe_yfinance_gold_futures(
        http_get=lambda url: fx._Response(False, 429, None, None,
                                              "http_error: Too Many "
                                              "Requests"),
    )
    assert row["decision_status"] == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Crypto probes — re-validate the strategy-9 known shapes
# ---------------------------------------------------------------------------
def test_binance_klines_probe_pass_on_long_window():
    earliest = 1568102400000   # 2019-09-10
    payload = [
        [earliest, "1.0", "1.0", "1.0", "1.0", "1.0",
          earliest + 86400000, "100", 1, "1", "1", "0"],
        [1778371200000, "1.0", "1.0", "1.0", "1.0", "1.0",
          1778457599999, "100", 1, "1", "1", "0"],
    ]
    row = fx._probe_binance_klines(
        "BTC", "BTCUSDT",
        http_get=lambda url: _ok(payload=payload),
    )
    assert row["decision_status"] == "PASS"


def test_binance_oi_probe_fail_at_30_days():
    payload = [
        {"timestamp": 1775865600000,
          "sumOpenInterest": "1.0",
          "sumOpenInterestValue": "1.0"},
        {"timestamp": 1778371200000,
          "sumOpenInterest": "1.0",
          "sumOpenInterestValue": "1.0"},
    ]
    row = fx._probe_binance_oi_history(
        "BTC", "BTCUSDT",
        http_get=lambda url: _ok(payload=payload),
    )
    assert row["decision_status"] == "FAIL"


def test_deribit_book_summary_is_snapshot_fail():
    row = fx._probe_deribit_book_summary(
        http_get=lambda url: _ok(payload={"result": [{"foo": 1}]}),
    )
    assert row["decision_status"] == "FAIL"


# ---------------------------------------------------------------------------
# Network failure resilience + run_audit
# ---------------------------------------------------------------------------
def test_run_audit_handles_total_network_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    df = fx.run_audit(save=True, rate_delay_s=0.0,
                          http_get=lambda url: _err())
    assert not df.empty
    out = tmp_path / "fx_crypto_source_audit.csv"
    assert out.exists()
    on_disk = pd.read_csv(out)
    assert list(on_disk.columns) == fx.AUDIT_COLUMNS
    # No PASS row when every HTTP call failed.
    assert not (df["decision_status"] == "PASS").any()


def test_run_audit_writes_to_custom_path(tmp_path, monkeypatch):
    out = tmp_path / "custom_audit.csv"
    df = fx.run_audit(save=True, output_path=out, rate_delay_s=0.0,
                          http_get=lambda url: _err())
    assert out.exists()
    on_disk = pd.read_csv(out)
    assert list(on_disk.columns) == fx.AUDIT_COLUMNS
    assert len(on_disk) == len(df)


def test_run_audit_includes_key_required_inconclusive_rows(tmp_path,
                                                                  monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    df = fx.run_audit(save=False, rate_delay_s=0.0,
                          http_get=lambda url: _err())
    needs_key = df[df["requires_api_key"] == True]  # noqa: E712
    assert not needs_key.empty
    sources = set(needs_key["source"].tolist())
    # The INCONCLUSIVE / requires-key ledger should at minimum cover
    # the documented sources from the spec.
    assert {"oanda_v20", "ig_bank_switzerland", "stooq",
             "dukascopy"} <= sources


# ---------------------------------------------------------------------------
# Summariser
# ---------------------------------------------------------------------------
def test_summarise_counts_per_market():
    df = pd.DataFrame([
        {"market": "forex", "decision_status": "PASS"},
        {"market": "forex", "decision_status": "INCONCLUSIVE"},
        {"market": "crypto", "decision_status": "PASS"},
        {"market": "crypto", "decision_status": "FAIL"},
        {"market": "crypto", "decision_status": "WARNING"},
    ])
    out = fx.summarise(df)
    assert out["fx_pass"] == 1
    assert out["fx_inconclusive"] == 1
    assert out["crypto_pass"] == 1
    assert out["crypto_fail"] == 1
    assert out["crypto_warning"] == 1
    assert out["fx_viable"] is True
    assert out["crypto_viable"] is True


def test_summarise_handles_empty_frame():
    out = fx.summarise(pd.DataFrame())
    assert out["n"] == 0
    assert out["fx_viable"] is False
    assert out["crypto_viable"] is False


# ---------------------------------------------------------------------------
# Safety lock continues to be locked
# ---------------------------------------------------------------------------
def test_safety_lock_remains_locked():
    assert safety_lock.is_execution_allowed() is False
    assert safety_lock.is_paper_trading_allowed() is False
    assert safety_lock.is_kraken_connection_allowed() is False
    assert safety_lock.safety_lock_status() == "locked"


# ---------------------------------------------------------------------------
# Source-level safety
# ---------------------------------------------------------------------------
_SOURCE = (Path(__file__).resolve().parents[1]
              / "src" / "fx_crypto_source_audit.py").read_text()


def test_no_broker_imports():
    bad_import_patterns = (
        re.compile(r"^\s*import\s+ccxt\b", re.MULTILINE),
        re.compile(r"^\s*from\s+ccxt", re.MULTILINE),
        re.compile(r"^\s*import\s+kraken\b", re.MULTILINE),
        re.compile(r"^\s*from\s+kraken\b", re.MULTILINE),
        re.compile(r"^\s*import\s+alpaca", re.MULTILINE),
        re.compile(r"^\s*from\s+alpaca", re.MULTILINE),
        re.compile(r"\bbinance\.client\b"),
        re.compile(r"\bbybit\.client\b"),
    )
    for pat in bad_import_patterns:
        assert pat.search(_SOURCE) is None, pat.pattern


def test_no_api_key_reads():
    forbidden = (
        re.compile(r"\bos\.environ\.get\([\"'](?:[A-Z_]*API[_-]?KEY)[\"']"),
        re.compile(r"\bos\.environ\[[\"'](?:[A-Z_]*API[_-]?KEY)[\"']"),
        re.compile(r"\bos\.getenv\([\"'](?:[A-Z_]*API[_-]?KEY)[\"']"),
        re.compile(r"\bAPI[_-]?KEY\s*=\s*['\"][A-Za-z0-9_\-]{8,}['\"]"),
    )
    for pat in forbidden:
        assert pat.search(_SOURCE) is None, pat.pattern


def test_no_kraken_private_endpoints():
    forbidden = (
        re.compile(r"\bkraken\.private\b"),
        re.compile(r"\bkraken_private\b"),
        re.compile(r"['\"]/0/private/"),
        re.compile(r"\bAddOrder\b"),
        re.compile(r"\bCancelOrder\b"),
    )
    for pat in forbidden:
        assert pat.search(_SOURCE) is None, pat.pattern


def test_no_order_placement_strings():
    forbidden = (
        re.compile(r"\bcreate_order\s*\("),
        re.compile(r"\bplace_order\s*\("),
        re.compile(r"\bsubmit_order\s*\("),
        re.compile(r"\bsend_order\s*\("),
        re.compile(r"\bcancel_order\s*\("),
        re.compile(r"\bdef\s+(?:create|place|submit|send|cancel)_order\b"),
    )
    for pat in forbidden:
        assert pat.search(_SOURCE) is None, pat.pattern


def test_no_paper_or_live_trading_enablement():
    forbidden = (
        re.compile(r"paper_trading_allowed\s*=\s*True", re.IGNORECASE),
        re.compile(r"live_trading_allowed\s*=\s*True", re.IGNORECASE),
        re.compile(r"\bgo_live\b", re.IGNORECASE),
    )
    for pat in forbidden:
        assert pat.search(_SOURCE) is None, pat.pattern


def test_no_strategy_imports():
    bad = ("from .strategies", "from src.strategies",
            "import strategies", "Strategy(",
            "from . import backtester",
            "from . import portfolio_backtester")
    for s in bad:
        assert s not in _SOURCE, s


def test_results_glob_is_gitignored():
    gi = (Path(__file__).resolve().parents[1] / ".gitignore").read_text()
    assert "results/*.csv" in gi
