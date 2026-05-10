"""
Free / open data re-audit.

Probes every public-only crypto data endpoint we know about and asks
one question: does the FREE public stack — no API keys, no paid
plans — still offer enough multi-year depth to justify another
strategy branch?

Hard rules:
    * Public endpoints only. No API keys ever read or required.
    * No private endpoints (no order routing, balance, margin).
    * No paid plans, no scraping bypasses, no ToS violations.
    * Network failures NEVER crash the audit — every probe is wrapped
      in a guarded try/except and the row is recorded with
      `status="error"` so the CSV is always produced.

Output: `results/free_open_data_reaudit.csv`. Schema in
`AUDIT_COLUMNS` is locked.

Coverage classifier (locked, NEVER tuned):
    PASS         coverage_days >= 1460
    WARNING      365 <= coverage_days < 1460
    FAIL         < 365 days OR current snapshot only OR endpoint failed
    INCONCLUSIVE source likely works but cannot be verified (paid /
                  endpoint changed)

Strategy decision rule (per the user's spec):
    GO     if the free public stack provides at least 2 useful
            multi-year fields BEYOND vanilla OHLCV + basic funding.
    WARNING (close to the bar) if useful fields exist but capped <1460d.
    NO_GO  if free public stack adds no meaningful new multi-year data
            beyond what was already tested in the failed branches.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from . import config, utils

logger = utils.get_logger("cte.free_open_data_reaudit")


# ---------------------------------------------------------------------------
# Locked constants.
# ---------------------------------------------------------------------------
COVERAGE_PASS_DAYS = 1460
COVERAGE_WARNING_DAYS = 365

DEFAULT_TIMEOUT = 12.0
DEFAULT_USER_AGENT = "cte-free-data-reaudit/0.1"

# Field-type vocabulary — drives the strategy GO gate.
FIELD_OHLCV = "ohlcv"
FIELD_FUNDING = "funding"
FIELD_BASIS = "basis_or_premium"
FIELD_OPEN_INTEREST = "open_interest"
FIELD_LIQUIDATIONS = "liquidations"
FIELD_LONG_SHORT_RATIO = "long_short_ratio"
FIELD_ONCHAIN = "onchain_or_market_structure"
FIELD_SENTIMENT = "sentiment"
FIELD_STABLECOIN = "stablecoin_or_liquidity"
FIELD_TVL = "tvl"
FIELD_VOL_INDEX = "vol_index"

FIELD_TYPES: Tuple[str, ...] = (
    FIELD_OHLCV, FIELD_FUNDING, FIELD_BASIS, FIELD_OPEN_INTEREST,
    FIELD_LIQUIDATIONS, FIELD_LONG_SHORT_RATIO, FIELD_ONCHAIN,
    FIELD_SENTIMENT, FIELD_STABLECOIN, FIELD_TVL, FIELD_VOL_INDEX,
)

# Field types that count as "beyond vanilla OHLCV + basic funding"
# for the GO gate (per spec).
USEFUL_BEYOND_BASELINE: Tuple[str, ...] = (
    FIELD_BASIS,
    FIELD_OPEN_INTEREST,
    FIELD_LIQUIDATIONS,
    FIELD_LONG_SHORT_RATIO,
    FIELD_ONCHAIN,
    FIELD_SENTIMENT,
    FIELD_STABLECOIN,
    FIELD_TVL,
    FIELD_VOL_INDEX,
)


# ---------------------------------------------------------------------------
# CSV schema (locked).
# ---------------------------------------------------------------------------
AUDIT_COLUMNS: List[str] = [
    "source",
    "dataset",
    "asset",
    "endpoint_or_source",
    "requires_api_key",
    "free_access",
    "actual_start",
    "actual_end",
    "row_count",
    "coverage_days",
    "granularity",
    "field_type",
    "usable_for_research",
    "decision_status",
    "notes",
]


# ---------------------------------------------------------------------------
# Coverage classifier.
# ---------------------------------------------------------------------------
def classify_coverage(coverage_days: Optional[float],
                       *, status: str = "ok",
                       is_snapshot_only: bool = False) -> str:
    if status != "ok":
        return "FAIL"
    if is_snapshot_only:
        return "FAIL"
    if coverage_days is None:
        return "INCONCLUSIVE"
    cd = float(coverage_days)
    if cd < COVERAGE_WARNING_DAYS:
        return "FAIL"
    if cd < COVERAGE_PASS_DAYS:
        return "WARNING"
    return "PASS"


# ---------------------------------------------------------------------------
# HTTP helper — fail-soft, never raises.
# ---------------------------------------------------------------------------
@dataclass
class _Response:
    ok: bool
    status_code: Optional[int]
    payload: Optional[Any]
    error: Optional[str]


def _http_get_json(url: str, timeout: float = DEFAULT_TIMEOUT,
                    user_agent: str = DEFAULT_USER_AGENT) -> _Response:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            try:
                return _Response(True, r.status, json.loads(raw), None)
            except ValueError as exc:
                return _Response(False, r.status, None,
                                  f"json_decode: {exc}")
    except urllib.error.HTTPError as exc:
        return _Response(False, exc.code, None,
                          f"http_error: {exc.reason}")
    except urllib.error.URLError as exc:
        return _Response(False, None, None, f"url_error: {exc.reason}")
    except (TimeoutError, OSError) as exc:
        return _Response(False, None, None, f"network_error: {exc}")
    except Exception as exc:  # noqa: BLE001
        return _Response(False, None, None,
                          f"unexpected: {type(exc).__name__}")


# ---------------------------------------------------------------------------
# Date helpers.
# ---------------------------------------------------------------------------
def _ts_to_iso(ts_seconds_or_ms: Optional[Any]) -> Optional[str]:
    if ts_seconds_or_ms is None:
        return None
    try:
        ts = int(ts_seconds_or_ms)
    except (ValueError, TypeError):
        return None
    # Auto-detect seconds vs ms — seconds are typically 10 digits at this
    # epoch; ms 13 digits.
    if ts > 1_000_000_000_000:
        ts_seconds = ts / 1000.0
    else:
        ts_seconds = float(ts)
    try:
        return datetime.fromtimestamp(ts_seconds,
                                        tz=timezone.utc).isoformat()
    except (ValueError, OSError):
        return None


def _coverage_days(start: Optional[Any], end: Optional[Any]) -> Optional[float]:
    if start is None or end is None:
        return None
    try:
        s = int(start)
        e = int(end)
    except (ValueError, TypeError):
        return None
    # Auto-detect units.
    s_secs = s / 1000.0 if s > 1_000_000_000_000 else float(s)
    e_secs = e / 1000.0 if e > 1_000_000_000_000 else float(e)
    return max(0.0, (e_secs - s_secs) / 86_400.0)


def _empty_row(source: str, dataset: str, endpoint: str, asset: str,
                field_type: str, **overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {c: None for c in AUDIT_COLUMNS}
    base.update({
        "source": source, "dataset": dataset,
        "asset": asset, "endpoint_or_source": endpoint,
        "requires_api_key": False, "free_access": True,
        "row_count": 0, "coverage_days": None,
        "field_type": field_type,
        "usable_for_research": "FAIL", "decision_status": "FAIL",
        "notes": None,
    })
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Per-source probes. Each returns one audit row.
# ---------------------------------------------------------------------------
def _probe_binance_klines(asset: str, symbol: str,
                            http_get: Callable[[str], _Response] = _http_get_json
                            ) -> Dict[str, Any]:
    url = (f"https://api.binance.com/api/v3/klines"
            f"?symbol={symbol}&interval=1d&startTime=1500000000000"
            f"&limit=1000")
    r = http_get(url)
    base = _empty_row(
        "binance", f"{asset.lower()}_spot_ohlcv_1d",
        "/api/v3/klines (spot)", asset, FIELD_OHLCV,
        granularity="1d", notes="paginates via startTime; multi-year deep",
    )
    if not r.ok or not isinstance(r.payload, list) or not r.payload:
        base.update({"decision_status": classify_coverage(None,
                                                              status="error"),
                      "usable_for_research": "FAIL",
                      "notes": (r.error or "empty")})
        return base
    earliest = int(r.payload[0][0])
    now_ms = int(time.time() * 1000)
    cov = _coverage_days(earliest, now_ms)
    base.update({
        "actual_start": _ts_to_iso(earliest),
        "actual_end": _ts_to_iso(now_ms),
        "row_count": len(r.payload), "coverage_days": cov,
        "decision_status": classify_coverage(cov),
        "usable_for_research": classify_coverage(cov),
    })
    return base


def _probe_binance_funding(asset: str, symbol: str,
                             http_get: Callable[[str], _Response] = _http_get_json
                             ) -> Dict[str, Any]:
    url = ("https://fapi.binance.com/fapi/v1/fundingRate"
            f"?symbol={symbol}&startTime=1500000000000&limit=1000")
    r = http_get(url)
    base = _empty_row(
        "binance_futures", f"{asset.lower()}_funding_rate_history",
        "/fapi/v1/fundingRate", asset, FIELD_FUNDING,
        granularity="8h",
        notes="paginates via startTime; multi-year deep",
    )
    if not r.ok or not isinstance(r.payload, list) or not r.payload:
        base.update({"decision_status": classify_coverage(None,
                                                              status="error"),
                      "notes": (r.error or "empty")})
        return base
    earliest = int(r.payload[0]["fundingTime"])
    now_ms = int(time.time() * 1000)
    cov = _coverage_days(earliest, now_ms)
    base.update({
        "actual_start": _ts_to_iso(earliest),
        "actual_end": _ts_to_iso(now_ms),
        "row_count": len(r.payload), "coverage_days": cov,
        "decision_status": classify_coverage(cov),
        "usable_for_research": classify_coverage(cov),
    })
    return base


def _probe_binance_oi_history(asset: str, symbol: str,
                                http_get: Callable[[str], _Response] = _http_get_json
                                ) -> Dict[str, Any]:
    url = ("https://fapi.binance.com/futures/data/openInterestHist"
            f"?symbol={symbol}&period=1d&limit=500")
    r = http_get(url)
    base = _empty_row(
        "binance_futures", f"{asset.lower()}_open_interest_history",
        "/futures/data/openInterestHist", asset, FIELD_OPEN_INTEREST,
        granularity="1d", notes="hard-capped ~30 days regardless of limit",
    )
    if not r.ok or not isinstance(r.payload, list) or not r.payload:
        base.update({"decision_status": classify_coverage(None,
                                                              status="error"),
                      "notes": (r.error or "empty")})
        return base
    start = int(r.payload[0]["timestamp"])
    end = int(r.payload[-1]["timestamp"])
    cov = _coverage_days(start, end)
    base.update({
        "actual_start": _ts_to_iso(start), "actual_end": _ts_to_iso(end),
        "row_count": len(r.payload), "coverage_days": cov,
        "decision_status": classify_coverage(cov),
        "usable_for_research": classify_coverage(cov),
    })
    return base


def _probe_binance_basis(asset: str, symbol: str,
                           http_get: Callable[[str], _Response] = _http_get_json
                           ) -> Dict[str, Any]:
    url = ("https://fapi.binance.com/fapi/v1/markPriceKlines"
            f"?symbol={symbol}&interval=1d&limit=1500")
    r = http_get(url)
    base = _empty_row(
        "binance_futures", f"{asset.lower()}_mark_price_klines",
        "/fapi/v1/markPriceKlines", asset, FIELD_BASIS,
        granularity="1d",
        notes="mark + index together = futures-spot basis proxy",
    )
    if not r.ok or not isinstance(r.payload, list) or not r.payload:
        base.update({"decision_status": classify_coverage(None,
                                                              status="error"),
                      "notes": (r.error or "empty")})
        return base
    earliest = int(r.payload[0][0])
    latest = int(r.payload[-1][0])
    cov = _coverage_days(earliest, latest)
    base.update({
        "actual_start": _ts_to_iso(earliest),
        "actual_end": _ts_to_iso(latest),
        "row_count": len(r.payload), "coverage_days": cov,
        "decision_status": classify_coverage(cov),
        "usable_for_research": classify_coverage(cov),
    })
    return base


def _probe_bybit_funding(asset: str, symbol: str,
                           http_get: Callable[[str], _Response] = _http_get_json
                           ) -> Dict[str, Any]:
    # Probe with very-old startTime to find the earliest available row.
    url = ("https://api.bybit.com/v5/market/funding/history"
            f"?category=linear&symbol={symbol}"
            f"&startTime=1500000000000&endTime=1600000000000&limit=200")
    r = http_get(url)
    base = _empty_row(
        "bybit", f"{asset.lower()}_funding_rate_history",
        "/v5/market/funding/history", asset, FIELD_FUNDING,
        granularity="8h",
        notes="paginates via endTime; depth = (now - earliest)",
    )
    if not r.ok:
        base.update({"decision_status": classify_coverage(None,
                                                              status="error"),
                      "notes": (r.error or "no payload")})
        return base
    rows = (r.payload or {}).get("result", {}).get("list", []) or []
    if not rows:
        base.update({"decision_status": "INCONCLUSIVE",
                      "usable_for_research": "INCONCLUSIVE",
                      "notes": "empty list"})
        return base
    earliest = min(int(x["fundingRateTimestamp"]) for x in rows)
    now_ms = int(time.time() * 1000)
    cov = _coverage_days(earliest, now_ms)
    base.update({
        "actual_start": _ts_to_iso(earliest),
        "actual_end": _ts_to_iso(now_ms),
        "row_count": len(rows), "coverage_days": cov,
        "decision_status": classify_coverage(cov),
        "usable_for_research": classify_coverage(cov),
    })
    return base


def _probe_bybit_oi(asset: str, symbol: str,
                      http_get: Callable[[str], _Response] = _http_get_json
                      ) -> Dict[str, Any]:
    url = ("https://api.bybit.com/v5/market/open-interest"
            f"?category=linear&symbol={symbol}&intervalTime=1d&limit=200")
    r = http_get(url)
    base = _empty_row(
        "bybit", f"{asset.lower()}_open_interest_1d",
        "/v5/market/open-interest", asset, FIELD_OPEN_INTEREST,
        granularity="1d",
        notes="capped at recent ~200 days",
    )
    if not r.ok:
        base.update({"decision_status": classify_coverage(None,
                                                              status="error"),
                      "notes": (r.error or "no payload")})
        return base
    rows = (r.payload or {}).get("result", {}).get("list", []) or []
    if not rows:
        base.update({"decision_status": "INCONCLUSIVE",
                      "usable_for_research": "INCONCLUSIVE",
                      "notes": "empty list"})
        return base
    ts = sorted(int(x["timestamp"]) for x in rows)
    cov = _coverage_days(ts[0], ts[-1])
    base.update({
        "actual_start": _ts_to_iso(ts[0]), "actual_end": _ts_to_iso(ts[-1]),
        "row_count": len(rows), "coverage_days": cov,
        "decision_status": classify_coverage(cov),
        "usable_for_research": classify_coverage(cov),
    })
    return base


def _probe_bybit_account_ratio(asset: str, symbol: str,
                                  http_get: Callable[[str], _Response] = _http_get_json
                                  ) -> Dict[str, Any]:
    url = ("https://api.bybit.com/v5/market/account-ratio"
            f"?category=linear&symbol={symbol}&period=1d&limit=500")
    r = http_get(url)
    base = _empty_row(
        "bybit", f"{asset.lower()}_long_short_account_ratio",
        "/v5/market/account-ratio", asset, FIELD_LONG_SHORT_RATIO,
        granularity="1d", notes="capped at recent ~500 days",
    )
    if not r.ok:
        base.update({"decision_status": classify_coverage(None,
                                                              status="error"),
                      "notes": (r.error or "no payload")})
        return base
    rows = (r.payload or {}).get("result", {}).get("list", []) or []
    if not rows:
        base.update({"decision_status": "INCONCLUSIVE",
                      "usable_for_research": "INCONCLUSIVE",
                      "notes": "empty list"})
        return base
    ts = sorted(int(x["timestamp"]) for x in rows)
    cov = _coverage_days(ts[0], ts[-1])
    base.update({
        "actual_start": _ts_to_iso(ts[0]), "actual_end": _ts_to_iso(ts[-1]),
        "row_count": len(rows), "coverage_days": cov,
        "decision_status": classify_coverage(cov),
        "usable_for_research": classify_coverage(cov),
    })
    return base


def _probe_okx_funding(asset: str, instrument: str,
                         http_get: Callable[[str], _Response] = _http_get_json
                         ) -> Dict[str, Any]:
    """OKX paginates back via `after` cursor. Walk a few pages to test depth."""
    cursor: Optional[str] = None
    seen: List[int] = []
    pages = 0
    base_url = ("https://www.okx.com/api/v5/public/funding-rate-history"
                  f"?instId={instrument}&limit=100")
    for _ in range(5):
        url = base_url + (f"&after={cursor}" if cursor else "")
        r = http_get(url)
        if not r.ok:
            break
        rows = (r.payload or {}).get("data", []) or []
        if not rows:
            break
        ts = sorted(int(x["fundingTime"]) for x in rows)
        new_oldest = ts[0]
        if seen and new_oldest >= min(seen):
            break
        seen.extend(ts)
        cursor = str(new_oldest)
        pages += 1
    base = _empty_row(
        "okx", f"{asset.lower()}_funding_rate_history",
        "/api/v5/public/funding-rate-history", asset, FIELD_FUNDING,
        granularity="8h",
        notes=f"walked {pages} cursor pages; public endpoint capped",
    )
    if not seen:
        base.update({"decision_status": "FAIL",
                      "notes": "no rows reachable"})
        return base
    earliest, latest = min(seen), max(seen)
    cov = _coverage_days(earliest, latest)
    base.update({
        "actual_start": _ts_to_iso(earliest),
        "actual_end": _ts_to_iso(latest),
        "row_count": len(seen), "coverage_days": cov,
        "decision_status": classify_coverage(cov),
        "usable_for_research": classify_coverage(cov),
    })
    return base


def _probe_deribit_funding(asset: str, instrument: str,
                              http_get: Callable[[str], _Response] = _http_get_json
                              ) -> Dict[str, Any]:
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - 23 * 86_400_000
    url = ("https://www.deribit.com/api/v2/public/get_funding_rate_history"
            f"?instrument_name={instrument}"
            f"&start_timestamp={start_ts}&end_timestamp={end_ts}")
    r = http_get(url)
    base = _empty_row(
        "deribit", f"{asset.lower()}_perpetual_funding_history",
        "/api/v2/public/get_funding_rate_history", asset, FIELD_FUNDING,
        granularity="~1h",
        notes="range-window endpoint; perp launched 2019-08",
    )
    if not r.ok:
        base.update({"decision_status": classify_coverage(None,
                                                              status="error"),
                      "notes": (r.error or "no payload")})
        return base
    rows = (r.payload or {}).get("result", []) or []
    if not rows:
        base.update({"decision_status": "INCONCLUSIVE",
                      "usable_for_research": "INCONCLUSIVE",
                      "notes": "empty result"})
        return base
    perp_launch_ms = int(datetime(2019, 8, 13, tzinfo=timezone.utc
                                     ).timestamp() * 1000)
    latest = max(int(x["timestamp"]) for x in rows)
    cov = _coverage_days(perp_launch_ms, latest)
    base.update({
        "actual_start": _ts_to_iso(perp_launch_ms),
        "actual_end": _ts_to_iso(latest),
        "row_count": len(rows), "coverage_days": cov,
        "decision_status": classify_coverage(cov),
        "usable_for_research": classify_coverage(cov),
    })
    return base


def _probe_deribit_dvol(asset: str,
                          http_get: Callable[[str], _Response] = _http_get_json
                          ) -> Dict[str, Any]:
    end_ts = int(time.time() * 1000)
    start_ts = int(datetime(2022, 5, 27, tzinfo=timezone.utc
                              ).timestamp() * 1000)
    url = ("https://www.deribit.com/api/v2/public/get_volatility_index_data"
            f"?currency={asset}&start_timestamp={start_ts}"
            f"&end_timestamp={end_ts}&resolution=86400")
    r = http_get(url)
    base = _empty_row(
        "deribit", f"{asset.lower()}_dvol_index",
        "/api/v2/public/get_volatility_index_data", asset, FIELD_VOL_INDEX,
        granularity="1d", notes="DVOL implied-vol index",
    )
    if not r.ok:
        base.update({"decision_status": classify_coverage(None,
                                                              status="error"),
                      "notes": (r.error or "no payload")})
        return base
    rows = (r.payload or {}).get("result", {}).get("data", []) or []
    if not rows:
        base.update({"decision_status": "INCONCLUSIVE",
                      "usable_for_research": "INCONCLUSIVE",
                      "notes": "empty"})
        return base
    ts = sorted(int(x[0]) for x in rows)
    cov = _coverage_days(ts[0], ts[-1])
    base.update({
        "actual_start": _ts_to_iso(ts[0]), "actual_end": _ts_to_iso(ts[-1]),
        "row_count": len(rows), "coverage_days": cov,
        "decision_status": classify_coverage(cov),
        "usable_for_research": classify_coverage(cov),
    })
    return base


def _probe_kraken_ohlc(asset: str, kraken_pair: str,
                          http_get: Callable[[str], _Response] = _http_get_json
                          ) -> Dict[str, Any]:
    url = (f"https://api.kraken.com/0/public/OHLC"
            f"?pair={kraken_pair}&interval=1440")
    r = http_get(url)
    base = _empty_row(
        "kraken", f"{asset.lower()}_spot_ohlc_1d",
        "/0/public/OHLC", asset, FIELD_OHLCV,
        granularity="1d",
        notes="hard-capped at 720 most-recent candles regardless of since",
    )
    if not r.ok:
        base.update({"decision_status": classify_coverage(None,
                                                              status="error"),
                      "notes": (r.error or "no payload")})
        return base
    res = (r.payload or {}).get("result", {}) or {}
    pair_keys = [k for k in res.keys() if k != "last"]
    if not pair_keys:
        base.update({"decision_status": "INCONCLUSIVE",
                      "usable_for_research": "INCONCLUSIVE",
                      "notes": "no pair key in result"})
        return base
    rows = res[pair_keys[0]]
    if not rows:
        base.update({"decision_status": "INCONCLUSIVE",
                      "usable_for_research": "INCONCLUSIVE",
                      "notes": "empty rows"})
        return base
    start = int(rows[0][0])
    end = int(rows[-1][0])
    cov = _coverage_days(start, end)
    base.update({
        "actual_start": _ts_to_iso(start),
        "actual_end": _ts_to_iso(end),
        "row_count": len(rows), "coverage_days": cov,
        "decision_status": classify_coverage(cov),
        "usable_for_research": classify_coverage(cov),
    })
    return base


def _probe_alternative_me_fng(http_get: Callable[[str], _Response] = _http_get_json
                                 ) -> Dict[str, Any]:
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    r = http_get(url)
    base = _empty_row(
        "alternative_me", "fear_and_greed_index",
        "/fng/?limit=0", "ALL", FIELD_SENTIMENT,
        granularity="1d", notes="all-history daily F&G since 2018-02",
    )
    if not r.ok:
        base.update({"decision_status": "FAIL",
                      "notes": (r.error or "no payload")})
        return base
    rows = (r.payload or {}).get("data", []) or []
    if not rows:
        base.update({"decision_status": "INCONCLUSIVE",
                      "notes": "empty data"})
        return base
    ts = sorted(int(x["timestamp"]) for x in rows)
    cov = _coverage_days(ts[0], ts[-1])
    base.update({
        "actual_start": _ts_to_iso(ts[0]), "actual_end": _ts_to_iso(ts[-1]),
        "row_count": len(rows), "coverage_days": cov,
        "decision_status": classify_coverage(cov),
        "usable_for_research": classify_coverage(cov),
    })
    return base


def _probe_defillama_chain_tvl(asset: str, chain: str,
                                  http_get: Callable[[str], _Response] = _http_get_json
                                  ) -> Dict[str, Any]:
    url = f"https://api.llama.fi/v2/historicalChainTvl/{chain}"
    r = http_get(url)
    base = _empty_row(
        "defillama", f"{asset.lower()}_chain_tvl_history",
        f"/v2/historicalChainTvl/{chain}", asset, FIELD_TVL,
        granularity="1d", notes="per-chain daily TVL history",
    )
    if not r.ok or not isinstance(r.payload, list) or not r.payload:
        base.update({"decision_status": classify_coverage(None,
                                                              status="error"),
                      "notes": (r.error or "empty")})
        return base
    ts = sorted(int(x["date"]) for x in r.payload)
    cov = _coverage_days(ts[0], ts[-1])
    base.update({
        "actual_start": _ts_to_iso(ts[0]), "actual_end": _ts_to_iso(ts[-1]),
        "row_count": len(r.payload), "coverage_days": cov,
        "decision_status": classify_coverage(cov),
        "usable_for_research": classify_coverage(cov),
    })
    return base


def _probe_defillama_stablecoin_total(http_get: Callable[[str], _Response] = _http_get_json
                                          ) -> Dict[str, Any]:
    url = "https://stablecoins.llama.fi/stablecoincharts/all"
    r = http_get(url)
    base = _empty_row(
        "defillama", "total_stablecoin_supply",
        "/stablecoincharts/all", "ALL", FIELD_STABLECOIN,
        granularity="1d", notes="total daily stablecoin supply",
    )
    if not r.ok or not isinstance(r.payload, list) or not r.payload:
        base.update({"decision_status": classify_coverage(None,
                                                              status="error"),
                      "notes": (r.error or "empty")})
        return base
    ts = sorted(int(x["date"]) for x in r.payload)
    cov = _coverage_days(ts[0], ts[-1])
    base.update({
        "actual_start": _ts_to_iso(ts[0]), "actual_end": _ts_to_iso(ts[-1]),
        "row_count": len(r.payload), "coverage_days": cov,
        "decision_status": classify_coverage(cov),
        "usable_for_research": classify_coverage(cov),
    })
    return base


def _probe_blockchain_com_chart(metric: str, friendly: str,
                                    field_type: str = FIELD_ONCHAIN,
                                    http_get: Callable[[str], _Response] = _http_get_json
                                    ) -> Dict[str, Any]:
    url = (f"https://api.blockchain.info/charts/{metric}"
            f"?timespan=all&format=json")
    r = http_get(url)
    base = _empty_row(
        "blockchain_com", f"btc_{friendly}",
        f"/charts/{metric}", "BTC", field_type,
        granularity="1d", notes=f"all-history daily BTC {friendly}",
    )
    if not r.ok:
        base.update({"decision_status": classify_coverage(None,
                                                              status="error"),
                      "notes": (r.error or "no payload")})
        return base
    values = (r.payload or {}).get("values", []) or []
    if not values:
        base.update({"decision_status": "INCONCLUSIVE",
                      "notes": "empty values"})
        return base
    ts = sorted(int(v["x"]) for v in values)
    cov = _coverage_days(ts[0], ts[-1])
    base.update({
        "actual_start": _ts_to_iso(ts[0]), "actual_end": _ts_to_iso(ts[-1]),
        "row_count": len(values), "coverage_days": cov,
        "decision_status": classify_coverage(cov),
        "usable_for_research": classify_coverage(cov),
    })
    return base


def _probe_coingecko_market_chart(asset: str, coin_id: str,
                                       http_get: Callable[[str], _Response] = _http_get_json
                                       ) -> Dict[str, Any]:
    url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            f"/market_chart?vs_currency=usd&days=max&interval=daily")
    r = http_get(url)
    base = _empty_row(
        "coingecko_free", f"{asset.lower()}_market_chart_max",
        f"/coins/{coin_id}/market_chart", asset, FIELD_OHLCV,
        granularity="1d", notes="free public tier",
    )
    if not r.ok:
        # 401/403 means free public tier no longer serves this depth.
        base.update({"decision_status": "INCONCLUSIVE",
                      "usable_for_research": "INCONCLUSIVE",
                      "notes": (r.error or "")
                                + " — CoinGecko free tier requires API "
                                  "key as of 2024"})
        return base
    prices = (r.payload or {}).get("prices", []) or []
    if not prices:
        base.update({"decision_status": "INCONCLUSIVE",
                      "notes": "empty prices"})
        return base
    ts = [int(x[0]) for x in prices]
    cov = _coverage_days(min(ts), max(ts))
    base.update({
        "actual_start": _ts_to_iso(min(ts)),
        "actual_end": _ts_to_iso(max(ts)),
        "row_count": len(prices), "coverage_days": cov,
        "decision_status": classify_coverage(cov),
        "usable_for_research": classify_coverage(cov),
    })
    return base


def _probe_coinpaprika_historical(asset: str, ticker: str,
                                       http_get: Callable[[str], _Response] = _http_get_json
                                       ) -> Dict[str, Any]:
    url = (f"https://api.coinpaprika.com/v1/tickers/{ticker}/historical"
            f"?start=2020-01-01&interval=1d")
    r = http_get(url)
    base = _empty_row(
        "coinpaprika_free", f"{asset.lower()}_historical_max",
        f"/tickers/{ticker}/historical", asset, FIELD_OHLCV,
        granularity="1d", notes="free public tier",
    )
    if not r.ok:
        base.update({"decision_status": "INCONCLUSIVE",
                      "usable_for_research": "INCONCLUSIVE",
                      "notes": (r.error or "")
                                + " — CoinPaprika historical requires "
                                  "paid plan as of 2024"})
        return base
    rows = r.payload if isinstance(r.payload, list) else []
    if not rows:
        base.update({"decision_status": "INCONCLUSIVE",
                      "notes": "empty rows"})
        return base
    base.update({
        "row_count": len(rows),
        "coverage_days": _coverage_days(0, 0),
        "decision_status": "WARNING",
        "usable_for_research": "WARNING",
    })
    return base


# ---------------------------------------------------------------------------
# Probe table — every entry returns a single audit row.
# ---------------------------------------------------------------------------
def _make_probes(http_get: Callable[[str], _Response] = _http_get_json
                  ) -> List[Callable[[], Dict[str, Any]]]:
    """Return a flat list of zero-arg callables, one per probe."""
    probes: List[Callable[[], Dict[str, Any]]] = []
    # Spot OHLCV
    for asset, sym in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT"),
                          ("SOL", "SOLUSDT")):
        probes.append(lambda a=asset, s=sym: _probe_binance_klines(
            a, s, http_get=http_get))
    # Funding
    for asset, sym in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")):
        probes.append(lambda a=asset, s=sym: _probe_binance_funding(
            a, s, http_get=http_get))
    # Basis
    for asset, sym in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")):
        probes.append(lambda a=asset, s=sym: _probe_binance_basis(
            a, s, http_get=http_get))
    # OI history (FAIL on public endpoints)
    for asset, sym in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")):
        probes.append(lambda a=asset, s=sym: _probe_binance_oi_history(
            a, s, http_get=http_get))
    # Bybit
    for asset, sym in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")):
        probes.append(lambda a=asset, s=sym: _probe_bybit_funding(
            a, s, http_get=http_get))
        probes.append(lambda a=asset, s=sym: _probe_bybit_oi(
            a, s, http_get=http_get))
        probes.append(lambda a=asset, s=sym: _probe_bybit_account_ratio(
            a, s, http_get=http_get))
    # OKX (BTC + ETH)
    for asset, instr in (("BTC", "BTC-USDT-SWAP"),
                            ("ETH", "ETH-USDT-SWAP")):
        probes.append(lambda a=asset, i=instr: _probe_okx_funding(
            a, i, http_get=http_get))
    # Deribit (BTC + ETH)
    for asset, instr in (("BTC", "BTC-PERPETUAL"),
                            ("ETH", "ETH-PERPETUAL")):
        probes.append(lambda a=asset, i=instr: _probe_deribit_funding(
            a, i, http_get=http_get))
    for asset in ("BTC", "ETH"):
        probes.append(lambda a=asset: _probe_deribit_dvol(
            a, http_get=http_get))
    # Kraken (depth probe — known to be capped)
    probes.append(lambda: _probe_kraken_ohlc(
        "BTC", "XBTUSD", http_get=http_get))
    probes.append(lambda: _probe_kraken_ohlc(
        "ETH", "ETHUSD", http_get=http_get))
    # Alternative.me F&G
    probes.append(lambda: _probe_alternative_me_fng(http_get=http_get))
    # DefiLlama TVL per chain
    for asset, chain in (("BTC", "Bitcoin"), ("ETH", "Ethereum"),
                            ("SOL", "Solana")):
        probes.append(lambda a=asset, c=chain: _probe_defillama_chain_tvl(
            a, c, http_get=http_get))
    # DefiLlama stablecoins
    probes.append(lambda: _probe_defillama_stablecoin_total(
        http_get=http_get))
    # Blockchain.com on-chain
    for metric, friendly in (
        ("hash-rate", "hash_rate"),
        ("n-transactions", "n_transactions_per_day"),
        ("market-cap", "market_cap_usd"),
        ("estimated-transaction-volume-usd", "estimated_tx_volume_usd"),
    ):
        probes.append(lambda m=metric, f=friendly: _probe_blockchain_com_chart(
            m, f, http_get=http_get))
    # CoinGecko free
    for asset, coin in (("BTC", "bitcoin"), ("ETH", "ethereum")):
        probes.append(lambda a=asset, c=coin: _probe_coingecko_market_chart(
            a, c, http_get=http_get))
    # CoinPaprika free
    for asset, ticker in (("BTC", "btc-bitcoin"),
                              ("ETH", "eth-ethereum")):
        probes.append(lambda a=asset, t=ticker: _probe_coinpaprika_historical(
            a, t, http_get=http_get))
    return probes


# ---------------------------------------------------------------------------
# Top-level orchestrator + decision summariser.
# ---------------------------------------------------------------------------
def run_audit(
    save: bool = True,
    output_path: Optional[Path] = None,
    rate_delay_s: float = 0.20,
    http_get: Callable[[str], _Response] = _http_get_json,
) -> pd.DataFrame:
    """Run every probe; write the CSV. Always returns a DataFrame.

    Network failures NEVER crash the audit — the row records the error
    and the audit moves on."""
    utils.assert_paper_only()
    probes = _make_probes(http_get=http_get)
    rows: List[Dict[str, Any]] = []
    for i, probe_fn in enumerate(probes):
        try:
            row = probe_fn()
        except Exception as exc:  # noqa: BLE001 — fail-soft
            logger.warning("probe %d raised %s", i,
                              type(exc).__name__)
            row = _empty_row("unknown", f"probe_{i}", "n/a", "?",
                                FIELD_OHLCV,
                                decision_status="INCONCLUSIVE",
                                usable_for_research="INCONCLUSIVE",
                                notes=f"raised {type(exc).__name__}")
        rows.append(row)
        if i + 1 < len(probes) and rate_delay_s > 0:
            time.sleep(rate_delay_s)

    df = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    if save:
        out = output_path or (config.RESULTS_DIR
                                / "free_open_data_reaudit.csv")
        utils.write_df(df, out)
    return df


def summarise(df: pd.DataFrame) -> Dict[str, Any]:
    """Return per-decision counts plus a 'GO/WARNING/NO_GO' verdict.

    Decision rule (locked, per spec):
        GO       at least 2 useful field types (beyond OHLCV + basic
                   funding) clear PASS.
        WARNING  >= 2 useful field types in WARNING band but none in PASS.
        NO_GO    fewer than 2 useful field types pass even WARNING.
    """
    if df.empty:
        return {"n": 0, "pass": 0, "warning": 0, "fail": 0,
                "inconclusive": 0,
                "useful_pass_field_types": [],
                "useful_warning_field_types": [],
                "verdict": "NO_GO"}

    counts = df["decision_status"].value_counts().to_dict()

    pass_useful = set()
    warning_useful = set()
    for _, row in df.iterrows():
        ft = str(row.get("field_type", ""))
        ds = str(row.get("decision_status", ""))
        if ft not in USEFUL_BEYOND_BASELINE:
            continue
        if ds == "PASS":
            pass_useful.add(ft)
        elif ds == "WARNING":
            warning_useful.add(ft)

    if len(pass_useful) >= 2:
        verdict = "GO"
    elif len(warning_useful) >= 2 or (len(pass_useful)
                                          + len(warning_useful)) >= 2:
        verdict = "WARNING"
    else:
        verdict = "NO_GO"

    return {
        "n": int(len(df)),
        "pass": int(counts.get("PASS", 0)),
        "warning": int(counts.get("WARNING", 0)),
        "fail": int(counts.get("FAIL", 0)),
        "inconclusive": int(counts.get("INCONCLUSIVE", 0)),
        "useful_pass_field_types": sorted(pass_useful),
        "useful_warning_field_types": sorted(warning_useful),
        "verdict": verdict,
    }
