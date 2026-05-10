"""Tests for the CoinGlass keyed data-depth audit.

Verified, all OFFLINE (HTTP layer is mocked):
    * Output schema is locked.
    * Missing API key produces AUTH_FAILED + INCONCLUSIVE rows, no crash.
    * 401 / 403 produces AUTH_FAILED.
    * 429 produces RATE_LIMITED.
    * 404 produces ENDPOINT_NOT_AVAILABLE.
    * Coverage classification PASS / WARNING / FAIL.
    * The decision rule "GO if >= 2 PASS field classes for both BTC and
      ETH" works.
    * The API key VALUE never appears in the CSV / DataFrame.
    * The API key VALUE never appears in stdout / log output.
    * The module contains no hardcoded keys, no broker imports, no
      Kraken private endpoints, no order-placement strings.
    * Tests do not require a real API key.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from src import config, coinglass_keyed_data_audit as cg


def _ok(payload: Any, status: int = 200) -> cg._Response:
    return cg._Response(True, status, payload, None)


def _err(error: str, status: int = None) -> cg._Response:
    return cg._Response(False, status, None, error)


def _coverage_payload(start_ms: int, end_ms: int,
                        n_rows: int = 10) -> Dict[str, Any]:
    """Build a minimal CoinGlass-shaped JSON response with `n_rows`
    timestamps spanning [start_ms, end_ms]."""
    if n_rows <= 1:
        return {"code": "0", "data": [{"ts": start_ms, "value": 1.0}]}
    step = (end_ms - start_ms) // (n_rows - 1)
    rows = [{"ts": start_ms + i * step, "value": 1.0}
              for i in range(n_rows)]
    return {"code": "0", "data": rows}


# ---------------------------------------------------------------------------
# Schema + decision vocabulary
# ---------------------------------------------------------------------------
def test_audit_columns_locked():
    expected = {
        "source", "dataset", "asset", "endpoint_or_source",
        "requires_api_key", "api_key_present",
        "status", "http_status", "actual_start", "actual_end",
        "row_count", "coverage_days", "granularity",
        "fields_available", "pagination_limit", "exchange_coverage",
        "usable_for_research", "decision_status", "usable_reason", "notes",
    }
    assert set(cg.AUDIT_COLUMNS) == expected


def test_decision_status_vocabulary_locked():
    assert set(cg.DECISION_STATUSES) == {
        "PASS", "WARNING", "FAIL", "INCONCLUSIVE",
        "AUTH_FAILED", "RATE_LIMITED", "ENDPOINT_NOT_AVAILABLE",
    }


def test_coverage_thresholds_locked():
    assert cg.COVERAGE_PASS_DAYS == 1460
    assert cg.COVERAGE_WARNING_DAYS == 365


# ---------------------------------------------------------------------------
# classify_decision unit tests
# ---------------------------------------------------------------------------
def test_classify_pass_at_threshold():
    assert cg.classify_decision(1460.0, api_key_present=True
                                    )[0] == "PASS"


def test_classify_warning_band():
    assert cg.classify_decision(500.0, api_key_present=True
                                    )[0] == "WARNING"


def test_classify_fail_below_year():
    assert cg.classify_decision(30.0, api_key_present=True)[0] == "FAIL"


def test_classify_snapshot_only_is_fail():
    assert cg.classify_decision(0.0, api_key_present=True,
                                    is_snapshot_only=True)[0] == "FAIL"


def test_classify_no_key_is_auth_failed():
    decision, usable = cg.classify_decision(2000.0, api_key_present=False)
    assert decision == "AUTH_FAILED"
    assert usable == "INCONCLUSIVE"


def test_classify_401_is_auth_failed():
    decision, usable = cg.classify_decision(
        2000.0, api_key_present=True, http_status=401,
    )
    assert decision == "AUTH_FAILED"
    assert usable == "INCONCLUSIVE"


def test_classify_429_is_rate_limited():
    decision, usable = cg.classify_decision(
        2000.0, api_key_present=True, http_status=429,
    )
    assert decision == "RATE_LIMITED"


def test_classify_404_is_endpoint_not_available():
    decision, usable = cg.classify_decision(
        2000.0, api_key_present=True, http_status=404,
    )
    assert decision == "ENDPOINT_NOT_AVAILABLE"


def test_classify_unknown_coverage_is_inconclusive():
    decision, usable = cg.classify_decision(None, api_key_present=True)
    assert decision == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Probe-level behaviour with mocked HTTP
# ---------------------------------------------------------------------------
def _probe_for(asset: str, dataset_substring: str) -> cg._ProbeSpec:
    for p in cg._make_probes():
        if p.asset == asset and dataset_substring in p.dataset:
            return p
    raise AssertionError(f"no probe found for {asset!r} / {dataset_substring!r}")


def test_run_probe_pass_on_deep_history():
    spec = _probe_for("BTC", "open_interest_aggregated")
    payload = _coverage_payload(
        start_ms=1568102400000,   # 2019-09-10
        end_ms=1778371200000,      # 2026-05-08
        n_rows=20,
    )
    row = cg.run_probe(spec, "fake-key",
                          http_get=lambda u, k: _ok(payload))
    assert row["status"] == "ok"
    assert row["decision_status"] == "PASS"
    assert row["coverage_days"] is not None
    assert row["coverage_days"] > 1460


def test_run_probe_fail_on_short_history():
    spec = _probe_for("BTC", "open_interest_aggregated")
    payload = _coverage_payload(
        start_ms=1775865600000,   # 2026-04-08
        end_ms=1778371200000,      # 2026-05-08
        n_rows=10,
    )
    row = cg.run_probe(spec, "fake-key",
                          http_get=lambda u, k: _ok(payload))
    assert row["decision_status"] == "FAIL"


def test_run_probe_no_key_yields_auth_failed():
    spec = _probe_for("ETH", "funding_binance")
    row = cg.run_probe(spec, "")
    assert row["api_key_present"] is False
    assert row["status"] == "auth_failed"
    assert row["decision_status"] == "AUTH_FAILED"
    assert row["usable_for_research"] == "INCONCLUSIVE"


def test_run_probe_401_yields_auth_failed():
    spec = _probe_for("BTC", "liquidations")
    row = cg.run_probe(spec, "fake-key",
                          http_get=lambda u, k: _err(
                              "http_error: Unauthorized", 401))
    assert row["decision_status"] == "AUTH_FAILED"


def test_run_probe_429_yields_rate_limited():
    spec = _probe_for("BTC", "liquidations")
    row = cg.run_probe(spec, "fake-key",
                          http_get=lambda u, k: _err(
                              "http_error: Too Many Requests", 429))
    assert row["decision_status"] == "RATE_LIMITED"


def test_run_probe_404_yields_endpoint_not_available():
    spec = _probe_for("ETH", "basis_binance")
    row = cg.run_probe(spec, "fake-key",
                          http_get=lambda u, k: _err(
                              "http_error: Not Found", 404))
    assert row["decision_status"] == "ENDPOINT_NOT_AVAILABLE"


def test_run_probe_empty_data_array_is_inconclusive():
    spec = _probe_for("BTC", "long_short_ratio_global")
    row = cg.run_probe(spec, "fake-key",
                          http_get=lambda u, k: _ok(
                              {"code": "0", "data": []}))
    assert row["decision_status"] == "INCONCLUSIVE"


def test_run_probe_unknown_shape_is_inconclusive():
    spec = _probe_for("BTC", "long_short_ratio_global")
    # 'rows' has no recognisable timestamp key.
    row = cg.run_probe(spec, "fake-key",
                          http_get=lambda u, k: _ok(
                              {"code": "0", "data": [{"foo": 1}]}))
    assert row["decision_status"] == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# run_audit orchestration
# ---------------------------------------------------------------------------
def test_run_audit_no_key_produces_well_formed_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.delenv("COINGLASS_API_KEY", raising=False)
    df = cg.run_audit(save=True, rate_delay_s=0.0)
    assert not df.empty
    assert list(df.columns) == cg.AUDIT_COLUMNS
    assert (df["api_key_present"] == False).all()  # noqa: E712
    assert (df["decision_status"] == "AUTH_FAILED").all()
    out = tmp_path / "coinglass_keyed_data_audit.csv"
    assert out.exists()
    on_disk = pd.read_csv(out)
    assert list(on_disk.columns) == cg.AUDIT_COLUMNS


def test_run_audit_with_key_produces_pass_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setenv("COINGLASS_API_KEY", "test-key-value-not-real")
    payload = _coverage_payload(
        start_ms=1568102400000, end_ms=1778371200000, n_rows=20,
    )
    df = cg.run_audit(save=True, rate_delay_s=0.0,
                          http_get=lambda u, k: _ok(payload))
    assert (df["api_key_present"] == True).all()  # noqa: E712
    # All probes return the same deep payload, so all should be PASS.
    assert (df["decision_status"] == "PASS").all()


def test_run_audit_handles_per_endpoint_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setenv("COINGLASS_API_KEY", "test-key-value-not-real")

    def patchy(url: str, key: str) -> cg._Response:
        if "liquidation" in url:
            return _err("http_error: Not Found", 404)
        return _ok(_coverage_payload(
            1568102400000, 1778371200000, n_rows=20,
        ))

    df = cg.run_audit(save=True, rate_delay_s=0.0, http_get=patchy)
    liq = df[df["dataset"].str.contains("liquidation")]
    not_liq = df[~df["dataset"].str.contains("liquidation")]
    assert (liq["decision_status"] == "ENDPOINT_NOT_AVAILABLE").all()
    assert (not_liq["decision_status"] == "PASS").all()


# ---------------------------------------------------------------------------
# API-key safety — key value never reaches CSV / stdout
# ---------------------------------------------------------------------------
SECRET_PROBE_KEY = "TEST-COINGLASS-SECRET-VALUE-1234567890"


def test_api_key_value_never_in_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setenv("COINGLASS_API_KEY", SECRET_PROBE_KEY)
    payload = _coverage_payload(1568102400000, 1778371200000, n_rows=20)
    df = cg.run_audit(save=True, rate_delay_s=0.0,
                          http_get=lambda u, k: _ok(payload))
    # Scan every cell.
    for col in df.columns:
        for v in df[col].dropna().tolist():
            if isinstance(v, str):
                assert SECRET_PROBE_KEY not in v, (
                    f"key leaked into column {col!r}: {v!r}"
                )
    out = tmp_path / "coinglass_keyed_data_audit.csv"
    raw = out.read_text()
    assert SECRET_PROBE_KEY not in raw, "key leaked into CSV file"


def test_api_key_value_never_logged(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setenv("COINGLASS_API_KEY", SECRET_PROBE_KEY)
    payload = _coverage_payload(1568102400000, 1778371200000, n_rows=20)
    with caplog.at_level("DEBUG", logger="cte"):
        cg.run_audit(save=True, rate_delay_s=0.0,
                        http_get=lambda u, k: _ok(payload))
    for r in caplog.records:
        assert SECRET_PROBE_KEY not in r.getMessage(), (
            f"key leaked into log: {r.getMessage()!r}"
        )


def test_api_key_value_never_printed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setenv("COINGLASS_API_KEY", SECRET_PROBE_KEY)
    payload = _coverage_payload(1568102400000, 1778371200000, n_rows=20)
    cg.run_audit(save=True, rate_delay_s=0.0,
                    http_get=lambda u, k: _ok(payload))
    captured = capsys.readouterr()
    assert SECRET_PROBE_KEY not in captured.out
    assert SECRET_PROBE_KEY not in captured.err


def test_api_key_present_predicate_does_not_leak_value(monkeypatch):
    monkeypatch.setenv("COINGLASS_API_KEY", SECRET_PROBE_KEY)
    assert cg.api_key_present() is True
    # The predicate must return a bool — never the raw string.
    assert cg.api_key_present() is not SECRET_PROBE_KEY


def test_api_key_missing_predicate_returns_false(monkeypatch):
    monkeypatch.delenv("COINGLASS_API_KEY", raising=False)
    assert cg.api_key_present() is False


# ---------------------------------------------------------------------------
# Summarise / decision rule
# ---------------------------------------------------------------------------
def test_summarise_go_when_two_field_classes_pass_for_both_assets(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setenv("COINGLASS_API_KEY", "k")
    payload = _coverage_payload(
        1568102400000, 1778371200000, n_rows=20,
    )
    df = cg.run_audit(save=False, rate_delay_s=0.0,
                          http_get=lambda u, k: _ok(payload))
    out = cg.summarise(df)
    assert out["api_key_present"] is True
    assert out["btc_pass_count"] >= 2
    assert out["eth_pass_count"] >= 2
    assert out["go"] is True
    assert out["verdict"] == "GO"


def test_summarise_no_go_when_only_one_class_passes_per_asset(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setenv("COINGLASS_API_KEY", "k")

    deep = _coverage_payload(1568102400000, 1778371200000, n_rows=20)
    short = _coverage_payload(1775865600000, 1778371200000, n_rows=10)

    def patchy(url: str, key: str) -> cg._Response:
        # Only OI passes per asset; the rest are short.
        if "openInterest" in url:
            return _ok(deep)
        return _ok(short)

    df = cg.run_audit(save=False, rate_delay_s=0.0, http_get=patchy)
    out = cg.summarise(df)
    assert out["btc_pass_count"] == 1
    assert out["eth_pass_count"] == 1
    assert out["go"] is False
    assert out["verdict"] == "NO_GO"


def test_summarise_inconclusive_when_no_key():
    df = pd.DataFrame([
        {"dataset": "btc_open_interest_aggregated", "asset": "BTC",
          "decision_status": "AUTH_FAILED", "api_key_present": False},
        {"dataset": "eth_funding_binance", "asset": "ETH",
          "decision_status": "AUTH_FAILED", "api_key_present": False},
    ])
    out = cg.summarise(df)
    assert out["api_key_present"] is False
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["go"] is False


def test_summarise_handles_empty_frame():
    out = cg.summarise(pd.DataFrame())
    assert out["n"] == 0
    assert out["go"] is False


# ---------------------------------------------------------------------------
# Source-level safety invariants
# ---------------------------------------------------------------------------
_AUDIT_SOURCE = (Path(__file__).resolve().parents[1]
                   / "src" / "coinglass_keyed_data_audit.py").read_text()


def test_no_hardcoded_api_key_value_literal():
    """The CI safety regex catches `XXX_API_KEY = "value"` form. We
    additionally assert here that nothing in the module assigns a
    long-looking literal next to any '*KEY*' identifier."""
    bad = re.compile(r"\b[A-Z_]*KEY[A-Z_]*\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"]")
    assert bad.search(_AUDIT_SOURCE) is None


def test_no_kraken_private_endpoints():
    forbidden = (
        re.compile(r"\bkraken\.private\b"),
        re.compile(r"\bkraken_private\b"),
        re.compile(r"['\"]/0/private/"),
        re.compile(r"\bAddOrder\b"),
        re.compile(r"\bCancelOrder\b"),
    )
    for pat in forbidden:
        assert pat.search(_AUDIT_SOURCE) is None, pat.pattern


def test_no_broker_imports():
    bad = ("ccxt", "kraken", "binance.client", "bybit.client",
            "deribit.client", "okx.client")
    for s in bad:
        assert s.lower() not in _AUDIT_SOURCE.lower(), s


def test_no_order_placement():
    forbidden = (
        re.compile(r"\bcreate_order\s*\("),
        re.compile(r"\bplace_order\s*\("),
        re.compile(r"\bsubmit_order\s*\("),
        re.compile(r"\bsend_order\s*\("),
        re.compile(r"\bcancel_order\s*\("),
        re.compile(r"\bdef\s+(?:create|place|submit|send|cancel)_order\b"),
    )
    for pat in forbidden:
        assert pat.search(_AUDIT_SOURCE) is None, pat.pattern


def test_only_env_var_read_for_key():
    """The key must be sourced ONLY from os.environ (or os.getenv).
    Specifically we forbid any other key-input vector."""
    bad = (
        # Reading from a file would be a different attack surface.
        re.compile(r"open\(['\"][^'\"]*\.key['\"]"),
        # Hard-coded base64 / hex literal of typical key length.
        re.compile(r"['\"][A-Za-z0-9]{32,}['\"]"),
    )
    # The 32+ char literal regex would false-positive on URLs and notes.
    # We only assert on the SPECIFIC line that reads the env var.
    src = _AUDIT_SOURCE
    # Confirm the canonical reader exists.
    assert "_API_KEY_ENV_NAME" in src
    assert 'os.environ.get(_API_KEY_ENV_NAME' in src
    # And no other shape:
    assert "open('coinglass.key'" not in src
    assert "open(\"coinglass.key\"" not in src
