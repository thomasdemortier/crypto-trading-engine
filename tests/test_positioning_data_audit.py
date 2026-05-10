"""Tests for the positioning-data audit module.

These tests verify:
    * Output schema is locked.
    * Empty / failed endpoint handling produces a well-formed FAIL row.
    * API-key + paid classification rules work without a subscription.
    * Coverage-threshold logic matches the spec
        (PASS >= 1460, WARNING 365–1459, FAIL < 365).
    * Current-snapshot-only data is classified FAIL.
    * The module contains NO strategy code, NO broker imports, NO
      Kraken private endpoints, NO API key reads, NO order placement.
    * The CSV is actually written.
    * Network failures do not crash the audit.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pandas as pd
import pytest

from src import config, positioning_data_audit as pda


# ---------------------------------------------------------------------------
# Schema + empty-row helpers
# ---------------------------------------------------------------------------
def test_audit_columns_are_locked():
    """The CSV schema is part of the contract — adding a column is a
    deliberate change. Keep this list in lock-step with `_AUDIT_COLUMNS`."""
    expected = {
        "source", "dataset", "endpoint_or_source", "asset",
        "requires_api_key", "access_type", "free_access", "paid_access",
        "status", "http_status",
        "actual_start", "actual_end", "row_count", "coverage_days",
        "granularity", "fields_available", "pagination_limit",
        "usable_for_research", "usable_reason", "notes",
    }
    assert set(pda._AUDIT_COLUMNS) == expected
    assert len(pda._AUDIT_COLUMNS) == 20


def test_empty_row_has_every_column():
    row = pda._empty_row("foo", "bar", "/baz", "BTC")
    assert set(row.keys()) == set(pda._AUDIT_COLUMNS)
    assert row["status"] == "not_probed"
    assert row["usable_for_research"] == "FAIL"


# ---------------------------------------------------------------------------
# Coverage classification
# ---------------------------------------------------------------------------
def test_classify_usability_pass_at_threshold():
    out = pda.classify_usability(1460.0)
    assert out["usable_for_research"] == "PASS"


def test_classify_usability_warning_band():
    out = pda.classify_usability(500.0)
    assert out["usable_for_research"] == "WARNING"


def test_classify_usability_fail_below_year():
    out = pda.classify_usability(30.0)
    assert out["usable_for_research"] == "FAIL"
    assert "30" in out["usable_reason"]


def test_classify_usability_unknown_coverage_is_fail():
    out = pda.classify_usability(None)
    assert out["usable_for_research"] == "FAIL"


def test_classify_usability_paid_is_inconclusive():
    out = pda.classify_usability(None, paid_only=True)
    assert out["usable_for_research"] == "INCONCLUSIVE"
    assert "paid" in out["usable_reason"].lower()


def test_classify_usability_snapshot_only_is_fail():
    out = pda.classify_usability(0.0, is_current_snapshot_only=True)
    assert out["usable_for_research"] == "FAIL"
    assert "snapshot" in out["usable_reason"].lower()


def test_classify_usability_status_error_is_fail():
    out = pda.classify_usability(2000.0, status="error")
    assert out["usable_for_research"] == "FAIL"


def test_classify_usability_irrelevant_field_is_fail():
    out = pda.classify_usability(2000.0, field_relevant=False)
    assert out["usable_for_research"] == "FAIL"


# ---------------------------------------------------------------------------
# Network-failure resilience — patch the HTTP layer to simulate failure
# ---------------------------------------------------------------------------
def _failing_response() -> pda._Response:
    return pda._Response(
        ok=False, status_code=None, payload=None,
        error="network_error: simulated", elapsed_s=0.0,
    )


def test_network_failure_does_not_crash_run_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(pda, "_http_get_json",
                          lambda *a, **kw: _failing_response())
    df = pda.run_audit(save=True)
    out = tmp_path / "positioning_data_audit.csv"
    assert out.exists()
    # Every probed row must report status=error AND usable=FAIL.
    probed = df[df["access_type"] != "paid_only"]
    assert not probed.empty
    assert (probed["status"] == "error").all()
    assert (probed["usable_for_research"] == "FAIL").all()


def test_run_audit_includes_paid_sources_as_inconclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(pda, "_http_get_json",
                          lambda *a, **kw: _failing_response())
    df = pda.run_audit(save=False)
    paid = df[df["access_type"] == "paid_only"]
    assert not paid.empty
    assert (paid["usable_for_research"] == "INCONCLUSIVE").all()
    assert (paid["requires_api_key"] == True).all()  # noqa: E712
    sources = set(paid["source"].tolist())
    assert {"coinglass", "cryptoquant", "glassnode",
            "kaiko", "velo_data"} <= sources


def test_run_audit_writes_to_custom_output_path(tmp_path, monkeypatch):
    monkeypatch.setattr(pda, "_http_get_json",
                          lambda *a, **kw: _failing_response())
    out = tmp_path / "custom_audit.csv"
    df = pda.run_audit(save=True, output_path=out)
    assert out.exists()
    on_disk = pd.read_csv(out)
    assert list(on_disk.columns) == pda._AUDIT_COLUMNS
    assert len(on_disk) == len(df)


# ---------------------------------------------------------------------------
# Successful-probe path — patch payload to deep-history success
# ---------------------------------------------------------------------------
def test_binance_funding_pass_when_payload_is_deep(monkeypatch):
    # 2019-09-10 → 2026-05-09: ~2433 days of coverage.
    payload = [
        {"fundingTime": 1568102400000, "fundingRate": "0.0001"},
        {"fundingTime": 1778371200000, "fundingRate": "0.0001"},
    ]
    monkeypatch.setattr(
        pda, "_http_get_json",
        lambda *a, **kw: pda._Response(True, 200, payload, None, 0.05),
    )
    row = pda._probe_binance_funding()
    assert row["status"] == "ok"
    assert row["usable_for_research"] == "PASS"
    assert row["coverage_days"] is not None and row["coverage_days"] > 1460


def test_binance_oi_hist_classified_fail_at_30_days(monkeypatch):
    payload = [{"timestamp": t, "sumOpenInterest": "1.0",
                "sumOpenInterestValue": "1.0"}
               for t in (1775865600000, 1778371200000)]
    monkeypatch.setattr(
        pda, "_http_get_json",
        lambda *a, **kw: pda._Response(True, 200, payload, None, 0.05),
    )
    row = pda._probe_binance_oi_hist()
    assert row["status"] == "ok"
    assert row["coverage_days"] is not None and row["coverage_days"] < 365
    assert row["usable_for_research"] == "FAIL"


def test_deribit_book_summary_classified_snapshot_only_fail(monkeypatch):
    payload = {"result": [{"open_interest": 1.0, "mark_price": 1.0}]}
    monkeypatch.setattr(
        pda, "_http_get_json",
        lambda *a, **kw: pda._Response(True, 200, payload, None, 0.05),
    )
    row = pda._probe_deribit_book_summary()
    assert row["status"] == "ok"
    assert row["usable_for_research"] == "FAIL"
    assert "snapshot" in row["usable_reason"].lower()


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------
def test_summarise_counts_categories():
    df = pd.DataFrame([
        {"usable_for_research": "PASS", "source": "binance_futures",
         "dataset": "funding_rate_history", "coverage_days": 2400.0},
        {"usable_for_research": "WARNING", "source": "bybit",
         "dataset": "long_short_account_ratio", "coverage_days": 500.0},
        {"usable_for_research": "FAIL", "source": "okx",
         "dataset": "rubik_oi", "coverage_days": 180.0},
        {"usable_for_research": "INCONCLUSIVE", "source": "coinglass",
         "dataset": "agg_oi", "coverage_days": None},
    ])
    out = pda.summarise(df)
    assert out["pass"] == 1
    assert out["warning"] == 1
    assert out["fail"] == 1
    assert out["inconclusive"] == 1
    assert out["best"] == "binance_futures / funding_rate_history"


def test_summarise_handles_empty_frame():
    out = pda.summarise(pd.DataFrame())
    assert out == {"n": 0, "pass": 0, "warning": 0, "fail": 0,
                   "inconclusive": 0, "best": None}


# ---------------------------------------------------------------------------
# Safety invariants — module must contain no strategy / broker / order code
# ---------------------------------------------------------------------------
_AUDIT_SOURCE = (Path(__file__).resolve().parents[1] /
                  "src" / "positioning_data_audit.py").read_text()


def test_no_strategy_imports_in_audit_module():
    bad_substrings = (
        "from .strategies", "from src.strategies",
        "import strategies", "Strategy(",
    )
    for s in bad_substrings:
        assert s not in _AUDIT_SOURCE, (
            f"audit module must not pull in strategy code: {s!r}"
        )


def test_no_broker_imports_in_audit_module():
    bad_substrings = ("ccxt", "kraken", "binance.client", "bybit.client",
                       "deribit.client", "okx.client")
    for s in bad_substrings:
        assert s.lower() not in _AUDIT_SOURCE.lower(), (
            f"audit module must not import broker SDKs: {s!r}"
        )


def test_no_kraken_private_endpoints():
    """Mirrors the CI safety check pattern."""
    forbidden = (
        re.compile(r"\bkraken\.private\b"),
        re.compile(r"\bkraken_private\b"),
        re.compile(r"['\"]/0/private/"),
        re.compile(r"\bAddOrder\b"),
        re.compile(r"\bCancelOrder\b"),
    )
    for pat in forbidden:
        assert pat.search(_AUDIT_SOURCE) is None, (
            f"forbidden private-endpoint pattern matched: {pat.pattern}"
        )


def test_no_api_key_reads_in_audit_module():
    forbidden = (
        re.compile(r"\bos\.environ\.get\([\"'](?:[A-Z_]*API[_-]?KEY)[\"']"),
        re.compile(r"\bos\.environ\[[\"'](?:[A-Z_]*API[_-]?KEY)[\"']"),
        re.compile(r"\bos\.getenv\([\"'](?:[A-Z_]*API[_-]?KEY)[\"']"),
        re.compile(r"\bAPI[_-]?KEY\s*=\s*['\"][A-Za-z0-9_\-]{8,}['\"]"),
    )
    for pat in forbidden:
        assert pat.search(_AUDIT_SOURCE) is None, (
            f"audit module must not read API keys: {pat.pattern}"
        )


def test_no_order_placement_in_audit_module():
    forbidden = (
        re.compile(r"\bcreate_order\s*\("),
        re.compile(r"\bplace_order\s*\("),
        re.compile(r"\bsubmit_order\s*\("),
        re.compile(r"\bsend_order\s*\("),
        re.compile(r"\bcancel_order\s*\("),
        re.compile(r"\bdef\s+(?:create|place|submit|send|cancel)_order\b"),
    )
    for pat in forbidden:
        assert pat.search(_AUDIT_SOURCE) is None, (
            f"audit module must not place orders: {pat.pattern}"
        )


def test_classify_usability_threshold_constants_match_spec():
    assert pda.COVERAGE_PASS_DAYS == 1460
    assert pda.COVERAGE_WARNING_DAYS == 365
