"""Tests for `src.sentiment_data_collector`. Network monkey-patched."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import pytest

from src import config, sentiment_data_collector as sdc


class _FakeResp:
    def __init__(self, payload: Any, status: int = 200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self._payload


def _altme_payload(n_days: int) -> Dict[str, Any]:
    """alternative.me shape — newest first."""
    base = 1_700_000_000
    rows = []
    for i in range(n_days):
        ts = base + i * 86_400
        rows.append({"value": str(50 + (i % 50)),
                      "value_classification": "Neutral",
                      "timestamp": str(ts),
                      "time_until_update": "60000"})
    return {"data": list(reversed(rows))}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
def test_normalised_columns_match_spec():
    assert sdc.NORMALISED_COLUMNS == [
        "timestamp", "date", "fear_greed_value",
        "fear_greed_classification", "source",
    ]


def test_coverage_columns_match_spec():
    assert sdc.COVERAGE_COLUMNS == [
        "source", "dataset", "actual_start", "actual_end",
        "row_count", "coverage_days", "enough_for_research",
        "largest_gap_days", "notes",
    ]


# ---------------------------------------------------------------------------
# Parser + normaliser
# ---------------------------------------------------------------------------
def test_parser_extracts_documented_fields():
    payload = _altme_payload(5)
    rows = sdc._parse_payload(payload)
    assert len(rows) == 5
    for r in rows:
        assert "timestamp" in r and "fear_greed_value" in r \
                and "fear_greed_classification" in r
        # Timestamp is in ms.
        assert r["timestamp"] > 1_000_000_000_000


def test_normaliser_dedupes_and_sorts_ascending():
    payload = _altme_payload(20)
    rows = sdc._parse_payload(payload)
    # Add a duplicate.
    rows.append(dict(rows[0]))
    df = sdc._to_normalised_df(rows)
    assert list(df.columns) == sdc.NORMALISED_COLUMNS
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].is_unique
    assert (df["source"] == "alternative.me").all()


def test_parser_skips_malformed_rows():
    rows = sdc._parse_payload({"data": [
        {"timestamp": "1700000000", "value": "50",
         "value_classification": "Neutral"},
        {"timestamp": "BAD", "value": "50",
         "value_classification": "Neutral"},
        {"timestamp": "1700086400"},   # missing value
    ]})
    assert len(rows) == 1


def test_gap_stats_detect_missing_day():
    ts = pd.Series([0, 86_400_000, 86_400_000 * 4])
    gaps, largest = sdc._gap_stats_days(ts)
    assert gaps == 1
    assert largest >= 3.0


# ---------------------------------------------------------------------------
# Download path (monkey-patched HTTP)
# ---------------------------------------------------------------------------
@pytest.fixture
def _redirect_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    yield


def test_download_writes_csv_and_coverage(_redirect_repo, monkeypatch):
    monkeypatch.setattr(
        sdc.requests, "get",
        lambda *a, **k: _FakeResp(_altme_payload(5 * 365)),
    )
    res = sdc.download_sentiment_data(refresh=True)
    p = res["path"]
    assert p.exists()
    df = pd.read_csv(p)
    assert list(df.columns) == sdc.NORMALISED_COLUMNS
    assert len(df) == 5 * 365
    cov = res["coverage_df"]
    assert list(cov.columns) == sdc.COVERAGE_COLUMNS
    assert bool(cov["enough_for_research"].iloc[0]) is True


def test_download_short_history_marked_unusable(_redirect_repo, monkeypatch):
    monkeypatch.setattr(
        sdc.requests, "get",
        lambda *a, **k: _FakeResp(_altme_payload(365)),
    )
    res = sdc.download_sentiment_data(refresh=True)
    cov = res["coverage_df"]
    assert bool(cov["enough_for_research"].iloc[0]) is False
    assert cov["row_count"].iloc[0] == 365


def test_download_handles_network_error(_redirect_repo, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("network down")
    monkeypatch.setattr(sdc.requests, "get", boom)
    res = sdc.download_sentiment_data(refresh=True)
    p = res["path"]
    assert p.exists()
    df = pd.read_csv(p) if p.stat().st_size > 0 else pd.DataFrame(
        columns=sdc.NORMALISED_COLUMNS)
    assert df.empty


def test_download_uses_cache_when_not_refresh(_redirect_repo, monkeypatch):
    """Second call with refresh=False must NOT hit the network."""
    monkeypatch.setattr(
        sdc.requests, "get",
        lambda *a, **k: _FakeResp(_altme_payload(5 * 365)),
    )
    sdc.download_sentiment_data(refresh=True)
    calls = {"n": 0}

    def fake_get(*a, **k):
        calls["n"] += 1
        return _FakeResp(_altme_payload(5 * 365))

    monkeypatch.setattr(sdc.requests, "get", fake_get)
    sdc.download_sentiment_data(refresh=False)
    assert calls["n"] == 0


# ---------------------------------------------------------------------------
# Safety invariants
# ---------------------------------------------------------------------------
def test_module_uses_no_api_keys_and_no_strategy_code():
    import inspect, re
    src = inspect.getsource(sdc)
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
                r"\bsubmit_order\s*\("):
        assert not re.search(pat, code), f"order pattern {pat!r} present"
