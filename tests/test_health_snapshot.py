"""Tests for `src/health_snapshot.py`. All offline."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from src import config, health_snapshot as hs, safety_lock


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_schema_locked():
    assert hs.SNAPSHOT_COLUMNS == [
        "snapshot_timestamp",
        "safety_lock_status",
        "execution_allowed",
        "paper_trading_allowed",
        "kraken_connection_allowed",
        "blocked_reason_count",
        "system_health_pass_count",
        "system_health_warning_count",
        "system_health_fail_count",
        "strategy_total_count",
        "strategy_pass_count",
        "strategy_fail_count",
        "strategy_inconclusive_count",
        "portfolio_file_present",
        "portfolio_schema_valid",
        "portfolio_total_market_value",
        "portfolio_risk_classification",
        "portfolio_recommendation",
        "notes",
    ]


def test_collect_returns_all_schema_keys():
    snap = hs.collect_health_snapshot()
    assert set(snap.keys()) == set(hs.SNAPSHOT_COLUMNS)
    assert isinstance(snap["snapshot_timestamp"], str)
    # ISO 8601-ish: starts with YYYY-MM-DD.
    assert re.match(r"^\d{4}-\d{2}-\d{2}", snap["snapshot_timestamp"])


def test_collect_reflects_locked_safety():
    snap = hs.collect_health_snapshot()
    assert snap["execution_allowed"] is False
    assert snap["paper_trading_allowed"] is False
    assert snap["kraken_connection_allowed"] is False
    assert snap["safety_lock_status"] == "locked"
    assert snap["blocked_reason_count"] >= 1


# ---------------------------------------------------------------------------
# append_health_snapshot
# ---------------------------------------------------------------------------
def test_append_writes_header_and_one_row(tmp_path, monkeypatch):
    res = tmp_path / "results"; res.mkdir()
    monkeypatch.setattr(config, "RESULTS_DIR", res)
    monkeypatch.setattr(hs, "DEFAULT_OUTPUT_PATH",
                          res / "health_snapshots.csv")

    out = hs.append_health_snapshot()
    assert out.exists()
    df = pd.read_csv(out)
    assert list(df.columns) == hs.SNAPSHOT_COLUMNS
    assert len(df) == 1


def test_append_appends_without_overwriting(tmp_path, monkeypatch):
    res = tmp_path / "results"; res.mkdir()
    monkeypatch.setattr(config, "RESULTS_DIR", res)
    monkeypatch.setattr(hs, "DEFAULT_OUTPUT_PATH",
                          res / "health_snapshots.csv")
    out = hs.append_health_snapshot()
    hs.append_health_snapshot()
    hs.append_health_snapshot()
    df = pd.read_csv(out)
    assert len(df) == 3


def test_append_refuses_to_write_outside_results(tmp_path):
    bad_path = tmp_path / "rogue" / "snapshot.csv"
    bad_path.parent.mkdir(parents=True)
    with pytest.raises(ValueError, match="results"):
        hs.append_health_snapshot(path=bad_path)


def test_append_with_supplied_snapshot_uses_it(tmp_path, monkeypatch):
    res = tmp_path / "results"; res.mkdir()
    out_path = res / "health_snapshots.csv"
    snap = hs._safe_defaults()
    snap["snapshot_timestamp"] = "2024-01-01T00:00:00+00:00"
    snap["safety_lock_status"] = "locked"
    snap["notes"] = "explicit-snapshot"
    out = hs.append_health_snapshot(path=out_path, snapshot=snap)
    df = pd.read_csv(out)
    assert df.iloc[0]["notes"] == "explicit-snapshot"
    assert df.iloc[0]["snapshot_timestamp"] == "2024-01-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# load_health_snapshots
# ---------------------------------------------------------------------------
def test_load_missing_returns_empty_with_warning(tmp_path):
    df, warn = hs.load_health_snapshots(tmp_path / "absent.csv")
    assert df.empty
    assert list(df.columns) == hs.SNAPSHOT_COLUMNS
    assert warn is not None
    assert "write_health_snapshot" in warn


def test_load_malformed_returns_warning(tmp_path):
    bad = tmp_path / "snapshots.csv"
    bad.write_text("not really a csv,foo,bar\n,,\nincomplete\n")
    df, warn = hs.load_health_snapshots(bad)
    # Either empty + warning, or non-empty but missing required cols → warn.
    assert warn is not None or df.empty


def test_load_round_trip(tmp_path, monkeypatch):
    res = tmp_path / "results"; res.mkdir()
    out_path = res / "health_snapshots.csv"
    hs.append_health_snapshot(path=out_path)
    hs.append_health_snapshot(path=out_path)
    df, warn = hs.load_health_snapshots(out_path)
    assert warn is None
    assert len(df) == 2
    assert list(df.columns) == hs.SNAPSHOT_COLUMNS


# ---------------------------------------------------------------------------
# summarize_health_timeline
# ---------------------------------------------------------------------------
def test_summary_empty_frame_safe():
    out = hs.summarize_health_timeline(pd.DataFrame())
    assert out["row_count"] == 0
    assert out["latest_snapshot_timestamp"] is None
    assert "no snapshots yet" in (out["warnings"][0]).lower()


def test_summary_one_row():
    snap = hs._safe_defaults()
    snap["snapshot_timestamp"] = "2024-01-01T00:00:00+00:00"
    snap["safety_lock_status"] = "locked"
    df = pd.DataFrame([snap], columns=hs.SNAPSHOT_COLUMNS)
    out = hs.summarize_health_timeline(df)
    assert out["row_count"] == 1
    assert out["latest_safety_lock_status"] == "locked"
    assert out["warnings"] == []


def test_summary_emits_warning_when_safety_unlocked():
    snap = hs._safe_defaults()
    snap["snapshot_timestamp"] = "2024-01-01T00:00:00+00:00"
    snap["safety_lock_status"] = "unlocked"
    df = pd.DataFrame([snap], columns=hs.SNAPSHOT_COLUMNS)
    out = hs.summarize_health_timeline(df)
    assert any("safety lock" in w for w in out["warnings"])


def test_summary_emits_warning_when_system_health_fail():
    snap = hs._safe_defaults()
    snap["snapshot_timestamp"] = "2024-01-01T00:00:00+00:00"
    snap["system_health_fail_count"] = 2
    df = pd.DataFrame([snap], columns=hs.SNAPSHOT_COLUMNS)
    out = hs.summarize_health_timeline(df)
    assert any("FAIL" in w for w in out["warnings"])


# ---------------------------------------------------------------------------
# Portfolio missing / present paths
# ---------------------------------------------------------------------------
def test_portfolio_missing_does_not_crash(tmp_path):
    """Force the portfolio file path to a definitely-absent location;
    the snapshot must still produce a row with `portfolio_file_present
    = False`."""
    snap = hs.collect_health_snapshot(
        portfolio_path=tmp_path / "definitely_absent.csv",
    )
    assert snap["portfolio_file_present"] is False
    assert snap["portfolio_schema_valid"] is False
    assert snap["portfolio_risk_classification"] == "UNKNOWN"
    assert snap["portfolio_recommendation"] in (
        "data missing", "do nothing until data is complete",
    )


def test_portfolio_present_path_works(tmp_path):
    p = tmp_path / "portfolio_holdings.csv"
    pd.DataFrame([{
        "asset": "BTC", "quantity": 0.5, "average_cost": 30000.0,
        "currency": "USD", "current_price": 60000.0,
        "price_source": "manual", "notes": "",
    }]).to_csv(p, index=False)
    snap = hs.collect_health_snapshot(portfolio_path=p)
    assert snap["portfolio_file_present"] is True
    assert snap["portfolio_schema_valid"] is True
    assert snap["portfolio_total_market_value"] == pytest.approx(30000.0)
    assert snap["portfolio_risk_classification"] in (
        "LOW", "MODERATE", "HIGH", "EXTREME", "UNKNOWN",
    )


# ---------------------------------------------------------------------------
# Safety-lock invariant
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
              / "src" / "health_snapshot.py").read_text()


def test_no_broker_imports():
    bad_import_patterns = (
        re.compile(r"^\s*import\s+ccxt\b", re.MULTILINE),
        re.compile(r"^\s*from\s+ccxt", re.MULTILINE),
        re.compile(r"^\s*import\s+kraken\b", re.MULTILINE),
        re.compile(r"^\s*from\s+kraken\b", re.MULTILINE),
        re.compile(r"\bbinance\.client\b"),
        re.compile(r"\bbybit\.client\b"),
        re.compile(r"\balpaca\b"),
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
    """The module must not enable trading anywhere — only READ the
    safety_lock state."""
    forbidden = (
        re.compile(r"paper_trading_allowed\s*=\s*True", re.IGNORECASE),
        re.compile(r"live_trading_allowed\s*=\s*True", re.IGNORECASE),
        re.compile(r"is_paper_trading_allowed\s*=\s*lambda", re.IGNORECASE),
        re.compile(r"is_execution_allowed\s*=\s*lambda", re.IGNORECASE),
        re.compile(r"\bgo_live\b", re.IGNORECASE),
    )
    for pat in forbidden:
        assert pat.search(_SOURCE) is None, pat.pattern


def test_no_network_calls():
    bad = ("urllib.request", "urllib.urlopen", "requests.get(",
            "httpx.get(", "aiohttp")
    for s in bad:
        assert s not in _SOURCE, s


# ---------------------------------------------------------------------------
# Generated artefact must not be tracked
# ---------------------------------------------------------------------------
def test_default_output_path_is_under_results():
    assert "results" in str(hs.DEFAULT_OUTPUT_PATH).lower()
    assert hs.DEFAULT_OUTPUT_PATH.name == "health_snapshots.csv"


def test_results_glob_is_gitignored():
    """Sanity check: `.gitignore` covers `results/*.csv`, so the
    snapshot file pattern is automatically ignored."""
    gi = (Path(__file__).resolve().parents[1] / ".gitignore").read_text()
    assert "results/*.csv" in gi
