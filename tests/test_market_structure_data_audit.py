"""Tests for `src.market_structure_data_audit`.

Tests run offline. Network calls in the probes are stubbed with
monkeypatched `requests.get` so the verdict logic, schema, and CSV
write behaviour can be exercised deterministically.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import pytest

from src import config, market_structure_data_audit as mda


# ---------------------------------------------------------------------------
# Fake HTTP plumbing
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, status_code: int = 200, payload: Any = None):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self._payload


def _bc_payload(n_days: int, *, daily: bool = True) -> Dict[str, Any]:
    """Build a Blockchain.com-shaped {'values': [{'x': ts, 'y': v}, ...]}."""
    base = 1_700_000_000  # arbitrary epoch second
    step = 86_400 if daily else 7 * 86_400
    return {"values": [{"x": base + i * step, "y": float(i)} for i in range(n_days)]}


def _llama_payload(n_rows: int) -> List[Dict[str, Any]]:
    base = 1_500_000_000
    return [{"date": str(base + i * 86_400), "totalLiquidityUSD": float(i)}
             for i in range(n_rows)]


# ---------------------------------------------------------------------------
# Schema + write
# ---------------------------------------------------------------------------
def test_audit_columns_match_spec():
    expected = [
        "source", "dataset", "endpoint_or_source",
        "requires_api_key", "free_access",
        "actual_start", "actual_end", "row_count", "coverage_days",
        "usable_for_research", "notes",
    ]
    assert mda.AUDIT_COLUMNS == expected


def test_audit_writes_csv(tmp_path, monkeypatch):
    """The full audit must produce one row per probe and persist a CSV."""
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    # Make every network probe return a "5-year, daily" payload so the
    # verdict logic returns "usable_for_research=True" everywhere it can.
    def fake_get(url, params=None, timeout=20):
        if "blockchain.info" in url:
            return _FakeResp(200, _bc_payload(5 * 365))
        if "coingecko.com" in url and "market_chart" in url:
            return _FakeResp(200, {
                "prices": [[(1_700_000_000 + i * 86_400) * 1000, 0.0]
                            for i in range(365)],
            })
        if "coingecko.com/api/v3/global" in url:
            return _FakeResp(200, {"data": {"market_cap_percentage":
                                                 {"btc": 50.0, "eth": 15.0}}})
        if "coinpaprika.com" in url:
            return _FakeResp(200, {"bitcoin_dominance_percentage": 50.0})
        if "stablecoins.llama.fi" in url:
            return _FakeResp(200, _llama_payload(2000))
        if "api.llama.fi/charts" in url:
            return _FakeResp(200, _llama_payload(2000))
        return _FakeResp(404, {"error": "unknown"})
    monkeypatch.setattr(mda.requests, "get", fake_get)
    # Stub the spot-universe probe so it doesn't hit the local cache.
    monkeypatch.setattr(mda, "probe_existing_spot_universe",
                         lambda: mda._empty_row(
                             "binance_spot",
                             "existing_universe_daily_ohlcv",
                             "data/raw/", False, "test stub"))
    out = mda.audit_market_structure_data(save=True)
    assert (tmp_path / "market_structure_data_audit.csv").exists()
    assert list(out.columns) == mda.AUDIT_COLUMNS
    # 4 BC probes + 1 CG market_chart + 1 CG global + 1 CoinPaprika +
    # 2 DefiLlama + 1 spot-universe = 10 rows.
    assert len(out) == 10


# ---------------------------------------------------------------------------
# Coverage threshold
# ---------------------------------------------------------------------------
def test_under_4_years_marked_unusable(monkeypatch):
    """A series shorter than the 4-year (1460-day) bar is unusable."""
    short = _bc_payload(365)  # 1 year
    monkeypatch.setattr(
        mda.requests, "get",
        lambda *a, **k: _FakeResp(200, short),
    )
    row = mda.probe_blockchain_com_chart("market-price")
    assert row["coverage_days"] < mda.MIN_DAYS_FOR_RESEARCH
    assert row["usable_for_research"] is False
    assert "under" in row["notes"].lower()


def test_at_or_over_4_years_marked_usable(monkeypatch):
    long = _bc_payload(5 * 365)  # 5 years daily
    monkeypatch.setattr(
        mda.requests, "get",
        lambda *a, **k: _FakeResp(200, long),
    )
    row = mda.probe_blockchain_com_chart("market-price")
    assert row["coverage_days"] >= mda.MIN_DAYS_FOR_RESEARCH
    assert row["usable_for_research"] is True


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------
def test_unavailable_source_does_not_raise(monkeypatch):
    """A probe MUST NOT raise — network failures become a clearly-marked
    unusable row."""
    def boom(*a, **k):
        raise ConnectionError("network unreachable")
    monkeypatch.setattr(mda.requests, "get", boom)
    row = mda.probe_blockchain_com_chart("market-price")
    assert row["usable_for_research"] is False
    assert "http_error" in row["notes"]


def test_coingecko_max_returns_unusable_with_explicit_note(monkeypatch):
    """CoinGecko free returns 401 on `days=max` since 2024 — surface it
    in `notes` so we don't quietly try to use it."""
    monkeypatch.setattr(
        mda.requests, "get",
        lambda *a, **k: _FakeResp(401, {"error": "PRO only"}),
    )
    row = mda.probe_coingecko_btc_market_chart()
    assert row["usable_for_research"] is False
    assert "free tier capped" in row["notes"].lower()
    assert row["requires_api_key"] is False
    assert row["free_access"] is True


def test_global_dominance_endpoints_marked_snapshot_only():
    """Both CoinGecko `/global` and CoinPaprika `/global` are surfaced as
    current-snapshot, NOT historical — even when they succeed."""
    # Stub via direct requests.get monkeypatching at the module level.
    from src import market_structure_data_audit as mod

    def fake_get(url, params=None, timeout=20):
        if "coingecko.com" in url:
            return _FakeResp(200, {"data": {"market_cap_percentage":
                                                 {"btc": 60.0}}})
        if "coinpaprika.com" in url:
            return _FakeResp(200, {"bitcoin_dominance_percentage": 60.0})
        return _FakeResp(404, {})
    import pytest as _pytest
    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(mod.requests, "get", fake_get)
        cg = mod.probe_coingecko_global_dominance()
        cp = mod.probe_coinpaprika_global_dominance()
    for r in (cg, cp):
        assert r["usable_for_research"] is False
        assert ("snapshot" in r["notes"].lower()
                 or "pro" in r["notes"].lower())


# ---------------------------------------------------------------------------
# Safety: no API key, no strategy code in this module
# ---------------------------------------------------------------------------
def test_module_uses_no_api_keys_and_no_strategy_code():
    """Word-boundary check on the audit module's source. The column
    name `requires_api_key` is *describing* whether a source needs a
    key — that is the audit's job — so we look for tokens like
    `apikey=...`, `os.environ['SECRET_..']`, `place_order(...)`, not
    bare substrings."""
    import inspect, re
    src = inspect.getsource(mda)
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

    # Auth-key patterns we never want to see (assignment / header /
    # env-var read). The column NAME `requires_api_key` is fine because
    # it is followed by `:` (type hint) not `=` to an actual key value.
    forbidden_auth = [
        r"\bapi_key\s*=\s*['\"][A-Za-z0-9_\-]+['\"]",
        r"\bapikey\s*=\s*['\"][A-Za-z0-9_\-]+['\"]",
        r"\bsecret\s*=\s*['\"][A-Za-z0-9_\-]+['\"]",
        r"\bX-MBX-APIKEY\b",
        r"\bos\.environ\[['\"][A-Z_]*KEY['\"]\]",
        r"\bos\.getenv\(['\"][A-Z_]*KEY['\"]",
    ]
    for pat in forbidden_auth:
        assert not re.search(pat, code), (
            f"auth-key pattern {pat!r} appears in audit module"
        )

    # Strategy / order plumbing must NOT be in this module — match
    # function calls or definitions, not arbitrary substrings.
    forbidden_strategy = [
        r"\bplace_order\s*\(", r"\bsend_order\s*\(", r"\bsubmit_order\s*\(",
        r"\btarget_weights\s*\(", r"\brebalance\s*\(",
        r"\bclass\s+\w*Strategy\b",
    ]
    for pat in forbidden_strategy:
        assert not re.search(pat, code), (
            f"strategy/order pattern {pat!r} appears in audit module"
        )


def test_existing_spot_probe_handles_empty_cache(tmp_path, monkeypatch):
    """When no `*_1d.csv` files exist, the probe returns a clean
    unusable row (no exception)."""
    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path)
    row = mda.probe_existing_spot_universe()
    assert row["usable_for_research"] is False
    assert row["row_count"] == 0
    assert "no cached" in row["notes"].lower()
