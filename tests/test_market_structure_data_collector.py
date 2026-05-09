"""Tests for `src.market_structure_data_collector`.

Exercise parsers, normaliser, gap stats, coverage row builder, and the
end-to-end download path with monkey-patched HTTP. No live network.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import pytest

from src import config, market_structure_data_collector as mdc


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, payload: Any, status: int = 200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self._payload


def _bc_payload(n_days: int) -> Dict[str, Any]:
    base = 1_700_000_000  # epoch seconds
    return {"values": [{"x": base + i * 86_400, "y": float(i + 1)}
                        for i in range(n_days)]}


def _llama_tvl_payload(n_days: int) -> List[Dict[str, Any]]:
    base = 1_500_000_000
    return [{"date": str(base + i * 86_400),
              "totalLiquidityUSD": float(i + 1)}
             for i in range(n_days)]


def _llama_stable_payload(n_days: int) -> List[Dict[str, Any]]:
    base = 1_500_000_000
    return [{"date": str(base + i * 86_400),
              "totalCirculatingUSD": {"peggedUSD": float(i + 1)}}
             for i in range(n_days)]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_normalised_columns_match_spec():
    assert mdc.NORMALISED_COLUMNS == [
        "timestamp", "date", "source", "dataset", "value",
    ]


def test_coverage_columns_match_spec():
    assert mdc.COVERAGE_COLUMNS == [
        "source", "dataset", "actual_start", "actual_end",
        "row_count", "coverage_days", "enough_for_research",
        "largest_gap_days", "missing_reason", "notes",
    ]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def test_bc_values_parser_extracts_xy_pairs():
    payload = _bc_payload(5)
    out = mdc._parse_bc_values(payload)
    assert len(out) == 5
    # ts in ms, monotone increasing.
    assert all(b > a for (a, _), (b, _) in zip(out[:-1], out[1:]))
    # values increase 1..5.
    assert [v for _, v in out] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_llama_parser_picks_first_numeric_value():
    payload = _llama_tvl_payload(3)
    out = mdc._parse_list_top_level(payload)
    assert len(out) == 3
    assert [v for _, v in out] == [1.0, 2.0, 3.0]


def test_llama_parser_flattens_nested_pegged_dict():
    payload = _llama_stable_payload(3)
    out = mdc._parse_list_top_level(payload)
    assert len(out) == 3
    assert [v for _, v in out] == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Normaliser + gap stats
# ---------------------------------------------------------------------------
def test_normaliser_produces_documented_schema_and_dedupes():
    rows = [(1_000, 5.0), (2_000, 6.0), (1_000, 5.0)]
    df = mdc._to_normalised_df(rows, "test_source", "test_dataset")
    assert list(df.columns) == mdc.NORMALISED_COLUMNS
    assert len(df) == 2
    assert df["source"].iloc[0] == "test_source"
    assert df["dataset"].iloc[0] == "test_dataset"
    # Sorted ascending.
    assert df["timestamp"].is_monotonic_increasing


def test_gap_stats_detect_missing_day():
    ts = pd.Series([0, 86_400_000, 86_400_000 * 4])  # 3-day jump
    gaps, largest = mdc._gap_stats_days(ts)
    assert gaps == 1
    assert largest >= 3.0


# ---------------------------------------------------------------------------
# Download path (monkey-patched HTTP)
# ---------------------------------------------------------------------------
@pytest.fixture
def _fake_network(monkeypatch):
    def fake_get(url, params=None, timeout=30):
        if "blockchain.info" in url:
            return _FakeResp(_bc_payload(5 * 365))
        if "stablecoins.llama.fi" in url:
            return _FakeResp(_llama_stable_payload(2000))
        if "api.llama.fi/charts" in url:
            return _FakeResp(_llama_tvl_payload(2000))
        return _FakeResp({}, 404)
    monkeypatch.setattr(mdc.requests, "get", fake_get)


def test_download_all_writes_csvs(tmp_path, monkeypatch, _fake_network):
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path / "data" / "raw")
    res = mdc.download_all_market_structure(refresh=True, sleep_seconds=0.0)
    paths = res["paths"]
    assert len(paths) == 5
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
        df = pd.read_csv(p)
        assert list(df.columns) == mdc.NORMALISED_COLUMNS
    cov = res["coverage_df"]
    assert list(cov.columns) == mdc.COVERAGE_COLUMNS
    # 5 dataset rows + 1 spot-universe row = 6.
    assert len(cov) == 6


def test_download_dataset_logs_and_continues_on_failure(tmp_path, monkeypatch):
    """When the endpoint raises, we still write an empty CSV and the
    coverage row marks the dataset unusable."""
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)

    def boom(*a, **k):
        raise ConnectionError("network down")

    monkeypatch.setattr(mdc.requests, "get", boom)
    spec = mdc._DATASET_SPECS[0]
    p = mdc.download_dataset(spec, refresh=True, sleep_seconds=0.0)
    assert p.exists()
    df = pd.read_csv(p) if p.stat().st_size > 0 else pd.DataFrame(
        columns=mdc.NORMALISED_COLUMNS)
    assert df.empty


# ---------------------------------------------------------------------------
# Coverage logic
# ---------------------------------------------------------------------------
def test_coverage_marks_short_history_unusable(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    spec = mdc._DATASET_SPECS[0]
    # Write a short series (1 year) directly.
    short = mdc._to_normalised_df(
        [(int((pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)).value
               // 10**6), float(i + 1)) for i in range(365)],
        spec.source, spec.dataset,
    )
    short.to_csv(mdc.output_path_for(spec.output_csv), index=False)
    row = mdc.coverage_for_dataset(spec)
    assert row["enough_for_research"] is False
    assert row["row_count"] == 365


def test_coverage_marks_4yr_history_usable(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    spec = mdc._DATASET_SPECS[0]
    long_df = mdc._to_normalised_df(
        [(int((pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)).value
               // 10**6), float(i + 1)) for i in range(5 * 365)],
        spec.source, spec.dataset,
    )
    long_df.to_csv(mdc.output_path_for(spec.output_csv), index=False)
    row = mdc.coverage_for_dataset(spec)
    assert row["enough_for_research"] is True
    assert row["coverage_days"] >= mdc.MIN_DAYS_FOR_RESEARCH


def test_spot_universe_coverage_handles_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path)
    row = mdc._spot_universe_coverage_row()
    assert row["enough_for_research"] is False
    assert row["row_count"] == 0
    assert row["missing_reason"] == "no_csv"


# ---------------------------------------------------------------------------
# Safety invariants
# ---------------------------------------------------------------------------
def test_module_uses_only_public_endpoints():
    import inspect, re
    src = inspect.getsource(mdc)
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
    # Word-boundary auth-key checks (column NAMES like
    # `requires_api_key` are not used in this module, so these are
    # strict no-go patterns).
    for pat in (r"\bapi_key\s*=\s*['\"][A-Za-z0-9_\-]+['\"]",
                r"\bapikey\s*=\s*['\"][A-Za-z0-9_\-]+['\"]",
                r"\bsecret\s*=\s*['\"][A-Za-z0-9_\-]+['\"]",
                r"\bX-MBX-APIKEY\b"):
        assert not re.search(pat, code), f"auth pattern {pat!r} present"
    # No order plumbing.
    for pat in (r"\bplace_order\s*\(", r"\bsend_order\s*\(",
                r"\bsubmit_order\s*\("):
        assert not re.search(pat, code), f"order pattern {pat!r} present"
