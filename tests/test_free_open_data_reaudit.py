"""Tests for the free / open data re-audit (offline)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src import config, free_open_data_reaudit as fr


def _ok(payload: Any) -> fr._Response:
    return fr._Response(True, 200, payload, None)


def _err(error: str = "simulated") -> fr._Response:
    return fr._Response(False, None, None, f"network_error: {error}")


# ---------------------------------------------------------------------------
# Schema + classifier
# ---------------------------------------------------------------------------
def test_audit_columns_locked():
    expected = {
        "source", "dataset", "asset", "endpoint_or_source",
        "requires_api_key", "free_access",
        "actual_start", "actual_end",
        "row_count", "coverage_days", "granularity",
        "field_type", "usable_for_research",
        "decision_status", "notes",
    }
    assert set(fr.AUDIT_COLUMNS) == expected


def test_thresholds_locked():
    assert fr.COVERAGE_PASS_DAYS == 1460
    assert fr.COVERAGE_WARNING_DAYS == 365


def test_classify_pass_at_threshold():
    assert fr.classify_coverage(1460.0) == "PASS"


def test_classify_warning_band():
    assert fr.classify_coverage(800.0) == "WARNING"


def test_classify_fail_below_year():
    assert fr.classify_coverage(30.0) == "FAIL"


def test_classify_snapshot_only_is_fail():
    assert fr.classify_coverage(0.0, is_snapshot_only=True) == "FAIL"


def test_classify_unknown_coverage_is_inconclusive():
    assert fr.classify_coverage(None) == "INCONCLUSIVE"


def test_classify_status_error_is_fail():
    assert fr.classify_coverage(2000.0, status="error") == "FAIL"


# ---------------------------------------------------------------------------
# Per-source probe behaviour with mocked HTTP
# ---------------------------------------------------------------------------
def test_binance_klines_pass_on_deep_history():
    # Two timestamps spanning 2019-09 → 2026-05 (~2400 days).
    payload = [
        [1568102400000, "100", "100", "100", "100", "1000",
          1568188799999, "100000", 1, "100", "100", "0"],
        [1778371200000, "100", "100", "100", "100", "1000",
          1778457599999, "100000", 1, "100", "100", "0"],
    ]
    row = fr._probe_binance_klines("BTC", "BTCUSDT",
                                       http_get=lambda u: _ok(payload))
    assert row["decision_status"] == "PASS"
    assert row["field_type"] == "ohlcv"
    assert row["coverage_days"] is not None and row["coverage_days"] > 1460


def test_binance_oi_history_classified_fail_at_short_history():
    payload = [{"timestamp": 1775865600000,
                 "sumOpenInterest": "1.0",
                 "sumOpenInterestValue": "1.0"},
                {"timestamp": 1778371200000,
                 "sumOpenInterest": "1.0",
                 "sumOpenInterestValue": "1.0"}]
    row = fr._probe_binance_oi_history("BTC", "BTCUSDT",
                                            http_get=lambda u: _ok(payload))
    assert row["decision_status"] == "FAIL"
    assert row["field_type"] == "open_interest"


def test_alternative_me_pass_on_long_history():
    payload = {"data": [
        {"value": "30", "value_classification": "Fear",
          "timestamp": "1517443200"},
        {"value": "47", "value_classification": "Neutral",
          "timestamp": "1778371200"},
    ]}
    row = fr._probe_alternative_me_fng(http_get=lambda u: _ok(payload))
    assert row["decision_status"] == "PASS"
    assert row["field_type"] == "sentiment"


def test_defillama_chain_tvl_pass():
    payload = [
        {"date": 1506470400, "tvl": 0},
        {"date": 1778371200, "tvl": 100000},
    ]
    row = fr._probe_defillama_chain_tvl("ETH", "Ethereum",
                                             http_get=lambda u: _ok(payload))
    assert row["decision_status"] == "PASS"
    assert row["field_type"] == "tvl"


def test_defillama_stablecoin_pass():
    payload = [
        {"date": "1511913600", "totalCirculatingUSD": {"peggedUSD": 1}},
        {"date": "1778371200", "totalCirculatingUSD": {"peggedUSD": 1}},
    ]
    row = fr._probe_defillama_stablecoin_total(
        http_get=lambda u: _ok(payload),
    )
    assert row["decision_status"] == "PASS"
    assert row["field_type"] == "stablecoin_or_liquidity"


def test_blockchain_com_chart_pass():
    payload = {
        "values": [
            {"x": 1230940800, "y": 0.0001},
            {"x": 1778284800, "y": 1.0e9},
        ],
    }
    row = fr._probe_blockchain_com_chart(
        "hash-rate", "hash_rate",
        http_get=lambda u: _ok(payload),
    )
    assert row["decision_status"] == "PASS"
    assert row["field_type"] == "onchain_or_market_structure"


def test_coingecko_paid_gate_is_inconclusive():
    """CoinGecko free tier returns 401 for /market_chart?days=max in 2024+
    — the audit should record INCONCLUSIVE, not crash."""
    row = fr._probe_coingecko_market_chart(
        "BTC", "bitcoin",
        http_get=lambda u: fr._Response(False, 401, None,
                                            "http_error: Unauthorized"),
    )
    assert row["decision_status"] == "INCONCLUSIVE"
    assert "API key" in row["notes"]


def test_coinpaprika_paid_gate_is_inconclusive():
    row = fr._probe_coinpaprika_historical(
        "BTC", "btc-bitcoin",
        http_get=lambda u: fr._Response(False, 402, None,
                                            "http_error: Payment Required"),
    )
    assert row["decision_status"] == "INCONCLUSIVE"


def test_kraken_ohlc_capped_history_warning_or_fail():
    rows = [[ts, "100", "100", "100", "100", "100"]
              for ts in (1716163200, 1778371200)]
    payload = {"result": {"XXBTZUSD": rows, "last": 0}, "error": []}
    row = fr._probe_kraken_ohlc("BTC", "XBTUSD",
                                     http_get=lambda u: _ok(payload))
    # ~720 days → WARNING (between 365 and 1460).
    assert row["decision_status"] == "WARNING"


# ---------------------------------------------------------------------------
# Network-failure resilience
# ---------------------------------------------------------------------------
def test_network_failure_does_not_crash_run_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    df = fr.run_audit(save=True, rate_delay_s=0.0,
                          http_get=lambda u: _err())
    assert not df.empty
    out = tmp_path / "free_open_data_reaudit.csv"
    assert out.exists()
    # No PASS rows when every HTTP call failed.
    assert not (df["decision_status"] == "PASS").any()


def test_run_audit_writes_to_custom_path(tmp_path, monkeypatch):
    out = tmp_path / "custom_reaudit.csv"
    df = fr.run_audit(save=True, output_path=out, rate_delay_s=0.0,
                          http_get=lambda u: _err())
    assert out.exists()
    on_disk = pd.read_csv(out)
    assert list(on_disk.columns) == fr.AUDIT_COLUMNS
    assert len(on_disk) == len(df)


# ---------------------------------------------------------------------------
# Decision summariser
# ---------------------------------------------------------------------------
def test_summarise_go_when_two_useful_field_types_pass():
    df = pd.DataFrame([
        {"field_type": "sentiment", "decision_status": "PASS"},
        {"field_type": "tvl", "decision_status": "PASS"},
        {"field_type": "ohlcv", "decision_status": "PASS"},  # baseline
        {"field_type": "open_interest", "decision_status": "FAIL"},
    ])
    out = fr.summarise(df)
    assert out["verdict"] == "GO"
    assert "sentiment" in out["useful_pass_field_types"]
    assert "tvl" in out["useful_pass_field_types"]


def test_summarise_no_go_when_only_baseline_passes():
    df = pd.DataFrame([
        {"field_type": "ohlcv", "decision_status": "PASS"},
        {"field_type": "funding", "decision_status": "PASS"},
        {"field_type": "open_interest", "decision_status": "FAIL"},
    ])
    out = fr.summarise(df)
    assert out["verdict"] == "NO_GO"


def test_summarise_warning_when_only_warnings():
    df = pd.DataFrame([
        {"field_type": "long_short_ratio", "decision_status": "WARNING"},
        {"field_type": "vol_index", "decision_status": "WARNING"},
        {"field_type": "ohlcv", "decision_status": "PASS"},
    ])
    out = fr.summarise(df)
    assert out["verdict"] == "WARNING"


def test_summarise_handles_empty_frame():
    out = fr.summarise(pd.DataFrame())
    assert out["n"] == 0
    assert out["verdict"] == "NO_GO"


# ---------------------------------------------------------------------------
# Source-level safety invariants
# ---------------------------------------------------------------------------
_AUDIT_SOURCE = (Path(__file__).resolve().parents[1]
                   / "src" / "free_open_data_reaudit.py").read_text()


def test_no_api_key_reads():
    forbidden = (
        re.compile(r"\bos\.environ\.get\([\"'](?:[A-Z_]*API[_-]?KEY)[\"']"),
        re.compile(r"\bos\.environ\[[\"'](?:[A-Z_]*API[_-]?KEY)[\"']"),
        re.compile(r"\bos\.getenv\([\"'](?:[A-Z_]*API[_-]?KEY)[\"']"),
        re.compile(r"\bAPI[_-]?KEY\s*=\s*['\"][A-Za-z0-9_\-]{8,}['\"]"),
    )
    for pat in forbidden:
        assert pat.search(_AUDIT_SOURCE) is None, pat.pattern


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
    bad = ("ccxt", "binance.client", "bybit.client",
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


def test_no_strategy_imports():
    bad = ("from .strategies", "from src.strategies",
            "import strategies", "Strategy(")
    for s in bad:
        assert s not in _AUDIT_SOURCE, s
