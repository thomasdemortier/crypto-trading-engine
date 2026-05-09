"""Tests for `src.sentiment_data_audit`. Network is monkey-patched
so the verdict logic, schema, threshold, and failure handling are
exercised deterministically."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import pytest

from src import config, sentiment_data_audit as sda


class _FakeResp:
    def __init__(self, status_code: int = 200, payload: Any = None):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self._payload


def _altme_payload(n_days: int) -> Dict[str, Any]:
    """alternative.me shape: {'data': [{'timestamp': str, 'value': str,
       'value_classification': str, 'time_until_update': str}, ...]} —
    NEWEST first."""
    base = 1_700_000_000  # epoch seconds
    rows = []
    for i in range(n_days):
        ts = base + i * 86_400
        rows.append({"value": str(50 + (i % 50)),
                      "value_classification": "Neutral",
                      "timestamp": str(ts),
                      "time_until_update": "60000"})
    return {"name": "Fear and Greed Index",
             "data": list(reversed(rows)),  # newest first
             "metadata": {"error": None}}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_audit_columns_match_spec():
    expected = [
        "source", "dataset", "endpoint_or_source",
        "requires_api_key", "free_access",
        "actual_start", "actual_end", "row_count", "coverage_days",
        "usable_for_research", "notes",
    ]
    assert sda.AUDIT_COLUMNS == expected


def test_audit_writes_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path / "raw")

    def fake_get(url, params=None, headers=None, timeout=20):
        if "alternative.me/fng" in url:
            return _FakeResp(200, _altme_payload(5 * 365))
        if "coinmarketcap.com" in url:
            return _FakeResp(404, {"error": "not found"})
        if "coingecko.com" in url:
            return _FakeResp(200, {"community_data": {
                "facebook_likes": None,
                "reddit_average_posts_48h": 1.0,
            }})
        if "reddit.com" in url:
            return _FakeResp(200, {"data": {"subscribers": 8_000_000}})
        return _FakeResp(404, {})

    monkeypatch.setattr(sda.requests, "get", fake_get)
    out = sda.audit_sentiment_data(save=True)
    assert (tmp_path / "sentiment_data_audit.csv").exists()
    assert list(out.columns) == sda.AUDIT_COLUMNS
    # 4 network probes + 1 spot-universe = 5 rows.
    assert len(out) == 5
    # alternative.me row should be the only `usable_for_research=True`
    # in this stub since the other probes all return current-snapshot
    # or a 404, and the spot universe is empty under tmp_path.
    usable = out[out["usable_for_research"]]
    assert len(usable) == 1
    assert usable.iloc[0]["source"] == "alternative.me"


# ---------------------------------------------------------------------------
# Threshold logic
# ---------------------------------------------------------------------------
def test_under_4_years_is_unusable(monkeypatch):
    short = _altme_payload(365)  # 1 year
    monkeypatch.setattr(
        sda.requests, "get",
        lambda *a, **k: _FakeResp(200, short),
    )
    row = sda.probe_alternative_me_fear_greed()
    assert row["coverage_days"] < sda.MIN_DAYS_FOR_RESEARCH
    assert row["usable_for_research"] is False


def test_at_or_over_4_years_is_usable(monkeypatch):
    long = _altme_payload(5 * 365)
    monkeypatch.setattr(
        sda.requests, "get",
        lambda *a, **k: _FakeResp(200, long),
    )
    row = sda.probe_alternative_me_fear_greed()
    assert row["coverage_days"] >= sda.MIN_DAYS_FOR_RESEARCH
    assert row["usable_for_research"] is True


def test_alternative_me_handles_data_in_newest_first_order(monkeypatch):
    """The endpoint returns newest first; our probe must sort to ascending
    so coverage_days is positive, not negative."""
    payload = _altme_payload(120)
    monkeypatch.setattr(
        sda.requests, "get",
        lambda *a, **k: _FakeResp(200, payload),
    )
    row = sda.probe_alternative_me_fear_greed()
    assert row["coverage_days"] > 0


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------
def test_unavailable_source_does_not_raise(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("network unreachable")
    monkeypatch.setattr(sda.requests, "get", boom)
    row = sda.probe_alternative_me_fear_greed()
    assert row["usable_for_research"] is False
    assert "http_error" in row["notes"]


def test_coinmarketcap_marked_paid(monkeypatch):
    monkeypatch.setattr(
        sda.requests, "get",
        lambda *a, **k: _FakeResp(404, {"error": "not found"}),
    )
    row = sda.probe_coinmarketcap_fear_greed_historical()
    assert row["usable_for_research"] is False
    assert row["requires_api_key"] is True
    assert "PRO" in row["notes"]


def test_coingecko_marked_snapshot_only(monkeypatch):
    monkeypatch.setattr(
        sda.requests, "get",
        lambda *a, **k: _FakeResp(200, {"community_data":
                                           {"reddit_average_posts_48h": 1.0}}),
    )
    row = sda.probe_coingecko_bitcoin_community_data()
    assert row["usable_for_research"] is False
    assert "snapshot" in row["notes"].lower() or "pro" in row["notes"].lower()


def test_reddit_subscribers_marked_snapshot_only(monkeypatch):
    monkeypatch.setattr(
        sda.requests, "get",
        lambda *a, **k: _FakeResp(200, {"data": {"subscribers": 8_000_000}}),
    )
    row = sda.probe_reddit_bitcoin_subscribers()
    assert row["usable_for_research"] is False
    assert "snapshot" in row["notes"].lower()


def test_existing_spot_probe_handles_empty_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path)
    row = sda.probe_existing_spot_universe()
    assert row["usable_for_research"] is False
    assert row["row_count"] == 0
    assert "no cached" in row["notes"].lower()


# ---------------------------------------------------------------------------
# Safety invariants — no API keys, no strategy code
# ---------------------------------------------------------------------------
def test_module_uses_no_api_keys_and_no_strategy_code():
    import inspect, re
    src = inspect.getsource(sda)
    code_lines = []
    in_doc = False
    for raw in src.splitlines():
        s = raw.strip()
        if s.startswith('"""') or s.startswith("'''"):
            in_doc = not in_doc
            continue
        if in_doc or s.startswith("#"):
            continue
        code_lines.append(raw)
    code = "\n".join(code_lines)
    for pat in (r"\bapi_key\s*=\s*['\"][A-Za-z0-9_\-]+['\"]",
                r"\bapikey\s*=\s*['\"][A-Za-z0-9_\-]+['\"]",
                r"\bsecret\s*=\s*['\"][A-Za-z0-9_\-]+['\"]",
                r"\bX-MBX-APIKEY\b"):
        assert not re.search(pat, code), f"auth pattern {pat!r} present"
    for pat in (r"\bplace_order\s*\(", r"\bsend_order\s*\(",
                r"\bsubmit_order\s*\(", r"\btarget_weights\s*\(",
                r"\bclass\s+\w*Strategy\b"):
        assert not re.search(pat, code), f"strategy/order pattern {pat!r}"
