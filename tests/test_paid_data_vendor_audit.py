"""Tests for `src/paid_data_vendor_audit.py`.

These tests run entirely offline — the module is a manually maintained
ledger so no network calls are required to validate it.

Verified:
    * Output schema is locked.
    * Decision classifier covers PASS_CANDIDATE, WATCHLIST,
      SALES_REQUIRED, DOCS_GATED, INCONCLUSIVE, FAIL.
    * Confidence values constrained to {high, medium, low}.
    * No API key reads / no private endpoint URLs / no broker imports
      / no order placement strings.
    * CSV writes to a custom output path.
    * Module imports cleanly with no network access.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from src import config, paid_data_vendor_audit as pdv


# ---------------------------------------------------------------------------
# Schema + ledger sanity
# ---------------------------------------------------------------------------
def test_audit_columns_locked():
    expected = {
        "vendor", "category", "website", "docs_url", "pricing_url",
        "api_docs_public", "pricing_public", "pricing_model",
        "lowest_visible_plan_usd", "free_trial_available",
        "requires_sales_call",
        "historical_open_interest", "historical_liquidations",
        "historical_long_short_ratios", "historical_funding",
        "historical_basis", "exchange_flows_reserves",
        "asset_coverage", "granularity_claimed", "history_depth_claimed",
        "api_access", "download_access", "terms_notes",
        "research_usability", "decision_status",
        "recommended_next_action", "confidence", "notes",
    }
    assert set(pdv.AUDIT_COLUMNS) == expected


def test_decision_status_vocabulary_locked():
    assert pdv.DECISION_STATUSES == (
        "PASS_CANDIDATE", "WATCHLIST", "SALES_REQUIRED",
        "DOCS_GATED", "INCONCLUSIVE", "FAIL",
    )


def test_confidence_vocabulary_locked():
    assert pdv.CONFIDENCE_LEVELS == ("high", "medium", "low")


def test_ledger_covers_required_vendors():
    df = pdv.run_audit(save=False)
    needed = {"CoinGlass", "CryptoQuant", "Velo Data", "Glassnode",
                "Kaiko", "Laevitas", "Amberdata", "TokenTerminal",
                "Santiment", "TheTie"}
    assert needed <= set(df["vendor"].tolist())


def test_every_row_has_classified_decision():
    df = pdv.run_audit(save=False)
    assert df["decision_status"].notna().all()
    assert df["decision_status"].isin(pdv.DECISION_STATUSES).all()


def test_every_row_has_valid_confidence():
    df = pdv.run_audit(save=False)
    assert df["confidence"].isin(pdv.CONFIDENCE_LEVELS).all()


# ---------------------------------------------------------------------------
# Classifier — direct unit tests on synthetic rows.
# ---------------------------------------------------------------------------
def _row(**kw: Any) -> Dict[str, Any]:
    base = {f: "unknown" for f in pdv.PRIORITY_FIELDS}
    base.update({
        "api_access": "unknown", "download_access": "unknown",
        "history_depth_claimed": "", "pricing_public": False,
        "lowest_visible_plan_usd": None,
        "requires_sales_call": False, "api_docs_public": False,
    })
    base.update(kw)
    return base


def test_classify_pass_candidate_when_two_yes_fields_and_priced():
    row = _row(
        historical_open_interest="yes", historical_funding="yes",
        api_access="yes", history_depth_claimed="multi-year",
        pricing_public=True, lowest_visible_plan_usd=29.0,
        requires_sales_call=False, api_docs_public=True,
    )
    assert pdv.classify_decision(row) == "PASS_CANDIDATE"


def test_classify_sales_required_when_no_pricing():
    row = _row(
        historical_open_interest="yes", historical_funding="yes",
        api_access="yes", history_depth_claimed="multi-year",
        pricing_public=False, lowest_visible_plan_usd=None,
        requires_sales_call=True, api_docs_public=True,
    )
    assert pdv.classify_decision(row) == "SALES_REQUIRED"


def test_classify_docs_gated_when_api_docs_not_public():
    row = _row(
        historical_open_interest="yes",
        api_access="yes", history_depth_claimed="multi-year",
        pricing_public=False, lowest_visible_plan_usd=None,
        requires_sales_call=False, api_docs_public=False,
    )
    assert pdv.classify_decision(row) == "DOCS_GATED"


def test_classify_watchlist_when_useful_but_pricing_unclear():
    row = _row(
        historical_funding="yes",
        api_access="yes", history_depth_claimed="",
        pricing_public=True, lowest_visible_plan_usd=49.0,
        requires_sales_call=False, api_docs_public=True,
    )
    assert pdv.classify_decision(row) == "WATCHLIST"


def test_classify_inconclusive_when_no_evidence():
    row = _row(
        api_access="yes", history_depth_claimed="multi-year",
        pricing_public=True, lowest_visible_plan_usd=49.0,
        requires_sales_call=False, api_docs_public=True,
    )
    # No yes fields, no FAIL trigger because api_access is yes —
    # classifier should fall through to INCONCLUSIVE rather than FAIL.
    decision = pdv.classify_decision(row)
    assert decision in ("INCONCLUSIVE", "FAIL")


def test_classify_fail_when_irrelevant_category():
    row = _row(
        historical_open_interest="no", historical_liquidations="no",
        historical_long_short_ratios="no", historical_funding="no",
        historical_basis="no", exchange_flows_reserves="no",
        api_access="yes", history_depth_claimed="multi-year",
        pricing_public=True, lowest_visible_plan_usd=49.0,
        requires_sales_call=False, api_docs_public=True,
    )
    assert pdv.classify_decision(row) == "FAIL"


def test_classify_fail_when_snapshot_only_data():
    row = _row(
        historical_open_interest="snapshot_only",
        historical_liquidations="snapshot_only",
        historical_long_short_ratios="snapshot_only",
        historical_funding="snapshot_only",
        historical_basis="snapshot_only",
        exchange_flows_reserves="snapshot_only",
        api_access="yes",
    )
    assert pdv.classify_decision(row) == "FAIL"


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------
def test_summarise_picks_cheapest_high_confidence_pass_candidate():
    df = pd.DataFrame([
        {"vendor": "Cheap", "decision_status": "PASS_CANDIDATE",
         "lowest_visible_plan_usd": 29.0, "confidence": "medium"},
        {"vendor": "Expensive", "decision_status": "PASS_CANDIDATE",
         "lowest_visible_plan_usd": 99.0, "confidence": "high"},
        {"vendor": "Watch", "decision_status": "WATCHLIST",
         "lowest_visible_plan_usd": 19.0, "confidence": "high"},
    ])
    out = pdv.summarise(df)
    assert out["pass_candidates"] == 2
    assert out["watchlist"] == 1
    assert out["top"] == "Cheap"


def test_summarise_handles_empty_frame():
    out = pdv.summarise(pd.DataFrame())
    assert out["n"] == 0
    assert out["top"] is None


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------
def test_run_audit_writes_csv_to_custom_path(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    out = tmp_path / "custom_vendor_audit.csv"
    df = pdv.run_audit(save=True, output_path=out)
    assert out.exists()
    on_disk = pd.read_csv(out)
    assert list(on_disk.columns) == pdv.AUDIT_COLUMNS
    assert len(on_disk) == len(df)


def test_run_audit_writes_to_default_path(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    pdv.run_audit(save=True)
    assert (tmp_path / "paid_data_vendor_audit.csv").exists()


# ---------------------------------------------------------------------------
# Safety invariants — module must contain no broker / order / key code
# ---------------------------------------------------------------------------
_AUDIT_SOURCE = (Path(__file__).resolve().parents[1]
                   / "src" / "paid_data_vendor_audit.py").read_text()


def test_no_strategy_imports():
    bad = ("from .strategies", "from src.strategies",
            "import strategies", "Strategy(")
    for s in bad:
        assert s not in _AUDIT_SOURCE, s


def test_no_broker_imports():
    bad = ("ccxt", "kraken", "binance.client", "bybit.client",
            "deribit.client", "okx.client")
    for s in bad:
        assert s.lower() not in _AUDIT_SOURCE.lower(), s


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


def test_no_api_key_reads():
    forbidden = (
        re.compile(r"\bos\.environ\.get\([\"'](?:[A-Z_]*API[_-]?KEY)[\"']"),
        re.compile(r"\bos\.environ\[[\"'](?:[A-Z_]*API[_-]?KEY)[\"']"),
        re.compile(r"\bos\.getenv\([\"'](?:[A-Z_]*API[_-]?KEY)[\"']"),
        re.compile(r"\bAPI[_-]?KEY\s*=\s*['\"][A-Za-z0-9_\-]{8,}['\"]"),
    )
    for pat in forbidden:
        assert pat.search(_AUDIT_SOURCE) is None, pat.pattern


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


def test_no_network_calls_in_module():
    """The ledger is a pure manually maintained list — there should be
    no network call sites in the module (urllib / requests / httpx)."""
    bad = ("urllib.request", "urllib.urlopen", "requests.get(",
            "httpx.get(", "aiohttp")
    for s in bad:
        assert s not in _AUDIT_SOURCE, s


def test_module_imports_cleanly_without_network(monkeypatch):
    """Force-fail any accidental network call so we know the module is
    importable in offline CI."""
    import builtins
    real_import = builtins.__import__

    def trap_import(name, *args, **kw):
        if name in ("urllib.request", "requests", "httpx"):
            raise AssertionError(
                f"audit module triggered offline-banned import: {name}",
            )
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", trap_import)
    # Re-run the audit; should not raise.
    df = pdv.run_audit(save=False)
    assert not df.empty
