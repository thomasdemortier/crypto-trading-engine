"""
Positioning data availability audit.

Probes a curated list of public derivatives / market-positioning data
endpoints and records, for each, whether enough multi-year history is
reachable to justify a serious strategy test. **No strategy is built
here.** No backtests, no signals, no order code — this module only
asks "what data can we actually get?".

Hard rules:
    * Public endpoints only. No API keys ever read or required.
    * No private endpoints (no order routing, balance, margin).
    * Network failures must NOT crash the audit — every probe is
      wrapped in a guarded try/except and the row is recorded with
      `status="error"` so the CSV is always produced.
    * No data is persisted from the probes. We only record metadata
      (counts, first/last timestamps, schema). No raw payload is
      written to disk.

Output: `results/positioning_data_audit.csv`. The full row schema is
in `_AUDIT_COLUMNS`.

Research-usability rule (per spec):
    PASS         coverage >= 1460 days AND field is relevant.
    WARNING      365 - 1459 days.
    FAIL         < 365 days OR current-snapshot-only.
    INCONCLUSIVE source likely works but cannot be verified without
                  paid access or an API key.
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
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

from . import config, utils

logger = utils.get_logger("cte.positioning_data_audit")


# ---------------------------------------------------------------------------
# Output schema (locked here so tests can assert)
# ---------------------------------------------------------------------------
_AUDIT_COLUMNS: List[str] = [
    "source",
    "dataset",
    "endpoint_or_source",
    "asset",
    "requires_api_key",
    "access_type",            # "public" | "key_required" | "paid_only" | "unknown"
    "free_access",
    "paid_access",
    "status",                 # "ok" | "error" | "not_probed"
    "http_status",
    "actual_start",
    "actual_end",
    "row_count",
    "coverage_days",
    "granularity",
    "fields_available",
    "pagination_limit",
    "usable_for_research",    # "PASS" | "WARNING" | "FAIL" | "INCONCLUSIVE"
    "usable_reason",
    "notes",
]

# Coverage thresholds per spec.
COVERAGE_PASS_DAYS = 1460       # 4 years
COVERAGE_WARNING_DAYS = 365     # 1 year

DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_USER_AGENT = "cte-data-audit/0.1"

# Asset universe used by the rest of the project — kept identical here so
# the audit speaks the project's vocabulary.
DEFAULT_ASSETS: List[str] = [
    "BTC", "ETH", "SOL", "AVAX", "LINK",
    "XRP", "DOGE", "ADA", "LTC", "BNB",
]


# ---------------------------------------------------------------------------
# HTTP helper — minimal, stdlib-only, fail-soft.
# ---------------------------------------------------------------------------
@dataclass
class _Response:
    ok: bool
    status_code: Optional[int]
    payload: Optional[Any]
    error: Optional[str]
    elapsed_s: float


def _http_get_json(
    url: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    user_agent: str = DEFAULT_USER_AGENT,
) -> _Response:
    """Issue a GET, parse JSON, return a `_Response`. Never raises."""
    started = time.time()
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": user_agent},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            try:
                payload = json.loads(raw)
            except ValueError as exc:
                return _Response(False, r.status, None,
                                  f"json_decode: {exc}", time.time() - started)
            return _Response(True, r.status, payload, None,
                              time.time() - started)
    except urllib.error.HTTPError as exc:
        return _Response(False, exc.code, None,
                          f"http_error: {exc.reason}", time.time() - started)
    except urllib.error.URLError as exc:
        return _Response(False, None, None,
                          f"url_error: {exc.reason}", time.time() - started)
    except (TimeoutError, OSError) as exc:
        return _Response(False, None, None,
                          f"network_error: {exc}", time.time() - started)
    except Exception as exc:  # noqa: BLE001 — fail-soft as policy
        return _Response(False, None, None,
                          f"unexpected: {type(exc).__name__}: {exc}",
                          time.time() - started)


# ---------------------------------------------------------------------------
# Coverage classification helper — per-spec thresholds.
# ---------------------------------------------------------------------------
def classify_usability(
    coverage_days: Optional[float],
    is_current_snapshot_only: bool = False,
    paid_only: bool = False,
    field_relevant: bool = True,
    status: str = "ok",
) -> Dict[str, str]:
    """Return ``{"usable_for_research", "usable_reason"}``. Hard rules:

    * paid-only sources without a verified key  → INCONCLUSIVE.
    * status != "ok"                             → FAIL.
    * field is not relevant to a positioning
      strategy (e.g. unrelated sentiment text)   → FAIL.
    * is_current_snapshot_only=True              → FAIL.
    * coverage < 365 days                        → FAIL.
    * 365 <= coverage < 1460 days                → WARNING.
    * coverage >= 1460 days                      → PASS.
    """
    if paid_only:
        return {
            "usable_for_research": "INCONCLUSIVE",
            "usable_reason": "paid source — cannot verify without "
                              "subscription or API key",
        }
    if status != "ok":
        return {
            "usable_for_research": "FAIL",
            "usable_reason": f"endpoint status={status}",
        }
    if not field_relevant:
        return {
            "usable_for_research": "FAIL",
            "usable_reason": "field not relevant to a positioning strategy",
        }
    if is_current_snapshot_only:
        return {
            "usable_for_research": "FAIL",
            "usable_reason": "current snapshot only — no historical depth",
        }
    if coverage_days is None:
        return {
            "usable_for_research": "FAIL",
            "usable_reason": "coverage unknown",
        }
    cd = float(coverage_days)
    if cd < COVERAGE_WARNING_DAYS:
        return {
            "usable_for_research": "FAIL",
            "usable_reason": f"only {cd:.0f} days — need >= "
                              f"{COVERAGE_WARNING_DAYS}",
        }
    if cd < COVERAGE_PASS_DAYS:
        return {
            "usable_for_research": "WARNING",
            "usable_reason": f"{cd:.0f} days — usable but below the "
                              f"{COVERAGE_PASS_DAYS}-day PASS bar",
        }
    return {
        "usable_for_research": "PASS",
        "usable_reason": f"{cd:.0f} days >= {COVERAGE_PASS_DAYS}",
    }


# ---------------------------------------------------------------------------
# Helpers for date / depth extraction.
# ---------------------------------------------------------------------------
def _ts_to_iso(ts_ms: Optional[int]) -> Optional[str]:
    if ts_ms is None:
        return None
    try:
        return datetime.fromtimestamp(int(ts_ms) / 1000.0,
                                        tz=timezone.utc).isoformat()
    except (ValueError, TypeError, OSError):
        return None


def _coverage_days(start_ms: Optional[int], end_ms: Optional[int]
                    ) -> Optional[float]:
    if start_ms is None or end_ms is None:
        return None
    try:
        return max(0.0, (int(end_ms) - int(start_ms)) / 86_400_000.0)
    except (ValueError, TypeError):
        return None


def _empty_row(source: str, dataset: str, endpoint: str, asset: str,
                **overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {c: None for c in _AUDIT_COLUMNS}
    base.update({
        "source": source, "dataset": dataset,
        "endpoint_or_source": endpoint, "asset": asset,
        "requires_api_key": False, "access_type": "public",
        "free_access": True, "paid_access": False,
        "status": "not_probed", "http_status": None,
        "actual_start": None, "actual_end": None,
        "row_count": 0, "coverage_days": None,
        "granularity": None, "fields_available": None,
        "pagination_limit": None,
        "usable_for_research": "FAIL", "usable_reason": "not probed",
        "notes": None,
    })
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Per-source probes. Each returns a list of audit rows.
# ---------------------------------------------------------------------------
# Binance Futures public endpoints. No API key. Rate-limited but we keep
# usage minimal (one request per dataset).
def _probe_binance_funding(asset_code: str = "BTCUSDT") -> Dict[str, Any]:
    """Funding rate history. Pagination by `startTime`. The probe asks
    for the earliest available record (`startTime=1500000000000`) — the
    first row's `fundingTime` is therefore the true start of available
    history, and `coverage_days` is computed as (now - earliest)."""
    url = ("https://fapi.binance.com/fapi/v1/fundingRate"
           f"?symbol={asset_code}&startTime=1500000000000&limit=1000")
    r = _http_get_json(url)
    base = _empty_row(
        "binance_futures", "funding_rate_history",
        "/fapi/v1/fundingRate", asset_code,
        granularity="8h",
        fields_available="fundingRate, fundingTime",
        pagination_limit="1000 rows per page; paginate via startTime",
    )
    if not r.ok or not isinstance(r.payload, list) or not r.payload:
        base.update({
            "status": "error", "http_status": r.status_code,
            "notes": r.error or "empty payload",
        })
        base.update(classify_usability(None, status="error"))
        return base
    earliest = int(r.payload[0]["fundingTime"])
    now_ms = int(time.time() * 1000)
    # Paginated coverage: the first row IS the start of available data.
    cov = _coverage_days(earliest, now_ms)
    base.update({
        "status": "ok", "http_status": r.status_code,
        "actual_start": _ts_to_iso(earliest),
        "actual_end": _ts_to_iso(now_ms),
        "row_count": len(r.payload),
        "coverage_days": cov,
        "notes": (f"first available row = {_ts_to_iso(earliest)}; "
                   f"paginated depth = {int(cov or 0)} days at 8h"),
    })
    base.update(classify_usability(cov))
    return base


def _probe_binance_oi_hist(asset_code: str = "BTCUSDT",
                             period: str = "1d") -> Dict[str, Any]:
    """Open-interest history. Binance caps this endpoint to ~30 days
    regardless of `limit`."""
    url = ("https://fapi.binance.com/futures/data/openInterestHist"
           f"?symbol={asset_code}&period={period}&limit=500")
    r = _http_get_json(url)
    base = _empty_row(
        "binance_futures", f"open_interest_history_{period}",
        "/futures/data/openInterestHist", asset_code,
        granularity=period,
        fields_available="sumOpenInterest, sumOpenInterestValue, timestamp",
        pagination_limit=("hard cap ~30 days regardless of limit; "
                           "no startTime extension"),
    )
    if not r.ok or not isinstance(r.payload, list) or not r.payload:
        base.update({"status": "error", "http_status": r.status_code,
                      "notes": r.error or "empty payload"})
        base.update(classify_usability(None, status="error"))
        return base
    start = int(r.payload[0]["timestamp"])
    end = int(r.payload[-1]["timestamp"])
    cov = _coverage_days(start, end)
    base.update({
        "status": "ok", "http_status": r.status_code,
        "actual_start": _ts_to_iso(start),
        "actual_end": _ts_to_iso(end),
        "row_count": len(r.payload),
        "coverage_days": cov,
        "notes": ("Binance hard-caps OI history at ~30 days for "
                   "all granularities — confirmed empirically"),
    })
    base.update(classify_usability(cov))
    return base


def _probe_binance_top_long_short_account(
    asset_code: str = "BTCUSDT",
) -> Dict[str, Any]:
    url = ("https://fapi.binance.com/futures/data/topLongShortAccountRatio"
           f"?symbol={asset_code}&period=1d&limit=500")
    r = _http_get_json(url)
    base = _empty_row(
        "binance_futures", "top_long_short_account_ratio",
        "/futures/data/topLongShortAccountRatio", asset_code,
        granularity="1d",
        fields_available="longAccount, shortAccount, longShortRatio, timestamp",
        pagination_limit="hard cap ~30 days",
    )
    if not r.ok or not isinstance(r.payload, list) or not r.payload:
        base.update({"status": "error", "http_status": r.status_code,
                      "notes": r.error or "empty payload"})
        base.update(classify_usability(None, status="error"))
        return base
    start = int(r.payload[0]["timestamp"])
    end = int(r.payload[-1]["timestamp"])
    cov = _coverage_days(start, end)
    base.update({
        "status": "ok", "http_status": r.status_code,
        "actual_start": _ts_to_iso(start), "actual_end": _ts_to_iso(end),
        "row_count": len(r.payload), "coverage_days": cov,
        "notes": "30-day cap (same family as OI history)",
    })
    base.update(classify_usability(cov))
    return base


def _probe_binance_taker_long_short(
    asset_code: str = "BTCUSDT",
) -> Dict[str, Any]:
    url = ("https://fapi.binance.com/futures/data/takerlongshortRatio"
           f"?symbol={asset_code}&period=1d&limit=500")
    r = _http_get_json(url)
    base = _empty_row(
        "binance_futures", "taker_long_short_ratio",
        "/futures/data/takerlongshortRatio", asset_code,
        granularity="1d",
        fields_available="buySellRatio, buyVol, sellVol, timestamp",
        pagination_limit="hard cap ~30 days",
    )
    if not r.ok or not isinstance(r.payload, list) or not r.payload:
        base.update({"status": "error", "http_status": r.status_code,
                      "notes": r.error or "empty payload"})
        base.update(classify_usability(None, status="error"))
        return base
    start = int(r.payload[0]["timestamp"])
    end = int(r.payload[-1]["timestamp"])
    cov = _coverage_days(start, end)
    base.update({
        "status": "ok", "http_status": r.status_code,
        "actual_start": _ts_to_iso(start), "actual_end": _ts_to_iso(end),
        "row_count": len(r.payload), "coverage_days": cov,
        "notes": "30-day cap (same family as OI history)",
    })
    base.update(classify_usability(cov))
    return base


def _probe_binance_premium_basis(
    asset_code: str = "BTCUSDT",
) -> Dict[str, Any]:
    """Mark-price klines (1d). Mark and index price both 1500 days deep,
    so the **basis = (mark - index) / index** is a derivable
    multi-year positioning proxy."""
    url = ("https://fapi.binance.com/fapi/v1/markPriceKlines"
           f"?symbol={asset_code}&interval=1d&limit=1500")
    r = _http_get_json(url)
    base = _empty_row(
        "binance_futures", "mark_price_klines",
        "/fapi/v1/markPriceKlines", asset_code,
        granularity="1d",
        fields_available="open, high, low, close, openTime, closeTime",
        pagination_limit="up to 1500 rows per request",
    )
    if not r.ok or not isinstance(r.payload, list) or not r.payload:
        base.update({"status": "error", "http_status": r.status_code,
                      "notes": r.error or "empty payload"})
        base.update(classify_usability(None, status="error"))
        return base
    start = int(r.payload[0][0])
    end = int(r.payload[-1][0])
    cov = _coverage_days(start, end)
    base.update({
        "status": "ok", "http_status": r.status_code,
        "actual_start": _ts_to_iso(start), "actual_end": _ts_to_iso(end),
        "row_count": len(r.payload), "coverage_days": cov,
        "notes": ("Mark + index klines together = futures-spot basis "
                   "proxy. Not OI itself, but a clean multi-year "
                   "positioning signal."),
    })
    base.update(classify_usability(cov))
    return base


# ---------------------------------------------------------------------------
# Bybit v5
# ---------------------------------------------------------------------------
def _probe_bybit_funding(asset_code: str = "BTCUSDT") -> Dict[str, Any]:
    """Bybit linear funding history. Probe with a window straddling the
    presumed perp launch (2020-03 → 2020-09) — the first row of the
    response is the true earliest available record."""
    url_oldest = ("https://api.bybit.com/v5/market/funding/history"
                   f"?category=linear&symbol={asset_code}"
                   f"&startTime=1500000000000&endTime=1600000000000&limit=200")
    r = _http_get_json(url_oldest)
    base = _empty_row(
        "bybit", "funding_rate_history",
        "/v5/market/funding/history", asset_code,
        granularity="8h",
        fields_available="fundingRate, fundingRateTimestamp, symbol",
        pagination_limit=("200 rows per page; paginate with "
                           "startTime / endTime"),
    )
    if not r.ok:
        base.update({"status": "error", "http_status": r.status_code,
                      "notes": r.error or "no payload"})
        base.update(classify_usability(None, status="error"))
        return base
    rows = (r.payload or {}).get("result", {}).get("list", [])
    if not rows:
        base.update({"status": "error", "http_status": r.status_code,
                      "notes": "empty list"})
        base.update(classify_usability(None, status="error"))
        return base
    ts = sorted(int(x["fundingRateTimestamp"]) for x in rows)
    earliest = ts[0]
    now_ms = int(time.time() * 1000)
    cov = _coverage_days(earliest, now_ms)
    base.update({
        "status": "ok", "http_status": r.status_code,
        "actual_start": _ts_to_iso(earliest), "actual_end": _ts_to_iso(now_ms),
        "row_count": len(rows), "coverage_days": cov,
        "notes": (f"earliest record = {_ts_to_iso(earliest)}; "
                   f"paginated depth = {int(cov or 0)} days at 8h"),
    })
    base.update(classify_usability(cov))
    return base


def _probe_bybit_oi(asset_code: str = "BTCUSDT") -> Dict[str, Any]:
    url = ("https://api.bybit.com/v5/market/open-interest"
           f"?category=linear&symbol={asset_code}&intervalTime=1d&limit=200")
    r = _http_get_json(url)
    base = _empty_row(
        "bybit", "open_interest_1d",
        "/v5/market/open-interest", asset_code,
        granularity="1d",
        fields_available="openInterest, timestamp",
        pagination_limit="200 rows per page; paginate via cursor",
    )
    if not r.ok:
        base.update({"status": "error", "http_status": r.status_code,
                      "notes": r.error or "no payload"})
        base.update(classify_usability(None, status="error"))
        return base
    rows = (r.payload or {}).get("result", {}).get("list", [])
    if not rows:
        base.update({"status": "error", "http_status": r.status_code,
                      "notes": "empty list"})
        base.update(classify_usability(None, status="error"))
        return base
    ts = sorted(int(x["timestamp"]) for x in rows)
    start, end = ts[0], ts[-1]
    cov = _coverage_days(start, end)
    base.update({
        "status": "ok", "http_status": r.status_code,
        "actual_start": _ts_to_iso(start), "actual_end": _ts_to_iso(end),
        "row_count": len(rows), "coverage_days": cov,
        "notes": ("Bybit serves the most-recent ~200 days of 1d OI; "
                   "cursor pagination claimed but typically does NOT "
                   "extend past 200d"),
    })
    base.update(classify_usability(cov))
    return base


def _probe_bybit_account_ratio(
    asset_code: str = "BTCUSDT",
) -> Dict[str, Any]:
    url = ("https://api.bybit.com/v5/market/account-ratio"
           f"?category=linear&symbol={asset_code}&period=1d&limit=500")
    r = _http_get_json(url)
    base = _empty_row(
        "bybit", "long_short_account_ratio",
        "/v5/market/account-ratio", asset_code,
        granularity="1d",
        fields_available="buyRatio, sellRatio, timestamp",
        pagination_limit="up to 500 rows per page",
    )
    if not r.ok:
        base.update({"status": "error", "http_status": r.status_code,
                      "notes": r.error or "no payload"})
        base.update(classify_usability(None, status="error"))
        return base
    rows = (r.payload or {}).get("result", {}).get("list", [])
    if not rows:
        base.update({"status": "error", "http_status": r.status_code,
                      "notes": "empty list"})
        base.update(classify_usability(None, status="error"))
        return base
    ts = sorted(int(x["timestamp"]) for x in rows)
    start, end = ts[0], ts[-1]
    cov = _coverage_days(start, end)
    base.update({
        "status": "ok", "http_status": r.status_code,
        "actual_start": _ts_to_iso(start), "actual_end": _ts_to_iso(end),
        "row_count": len(rows), "coverage_days": cov,
        "notes": ("Bybit serves last ~500 days of 1d account-ratio; "
                   "no public extension beyond that"),
    })
    base.update(classify_usability(cov))
    return base


# ---------------------------------------------------------------------------
# OKX
# ---------------------------------------------------------------------------
def _probe_okx_funding(asset_code: str = "BTC-USDT-SWAP") -> Dict[str, Any]:
    """OKX funding-rate-history. Empirically capped at ~3 months — the
    `after` cursor does NOT reach back to perp launch on the public
    endpoint, so we report the cursor-walked depth honestly."""
    base = _empty_row(
        "okx", "funding_rate_history",
        "/api/v5/public/funding-rate-history", asset_code,
        granularity="8h",
        fields_available="fundingRate, realizedRate, fundingTime",
        pagination_limit=("100 rows per page; cursor pagination capped "
                           "empirically at ~100 days total"),
    )
    cursor: Optional[str] = None
    seen_ts: List[int] = []
    pages_walked = 0
    for _ in range(5):
        url = ("https://www.okx.com/api/v5/public/funding-rate-history"
               f"?instId={asset_code}&limit=100")
        if cursor is not None:
            url += f"&after={cursor}"
        r = _http_get_json(url)
        if not r.ok:
            base.update({"status": "error", "http_status": r.status_code,
                          "notes": r.error or "no payload"})
            base.update(classify_usability(None, status="error"))
            return base
        rows = (r.payload or {}).get("data", []) or []
        if not rows:
            break
        ts = sorted(int(x["fundingTime"]) for x in rows)
        new_oldest = ts[0]
        if seen_ts and new_oldest >= min(seen_ts):
            # cursor has stopped advancing — public depth is exhausted.
            break
        seen_ts.extend(ts)
        cursor = str(new_oldest)
        pages_walked += 1
    if not seen_ts:
        base.update({"status": "error", "http_status": None,
                      "notes": "empty data"})
        base.update(classify_usability(None, status="error"))
        return base
    earliest, latest = min(seen_ts), max(seen_ts)
    cov = _coverage_days(earliest, latest)
    base.update({
        "status": "ok", "http_status": 200,
        "actual_start": _ts_to_iso(earliest), "actual_end": _ts_to_iso(latest),
        "row_count": len(seen_ts), "coverage_days": cov,
        "notes": (f"walked {pages_walked} cursor pages; deepest "
                   f"reachable = {int(cov or 0)} days — public endpoint "
                   f"does NOT serve full perp history"),
    })
    base.update(classify_usability(cov))
    return base


def _probe_okx_rubik_oi(asset_code: str = "BTC") -> Dict[str, Any]:
    url = ("https://www.okx.com/api/v5/rubik/stat/contracts/"
           f"open-interest-volume?ccy={asset_code}&period=1D")
    r = _http_get_json(url)
    base = _empty_row(
        "okx", "rubik_open_interest_volume",
        "/api/v5/rubik/stat/contracts/open-interest-volume", asset_code,
        granularity="1d",
        fields_available="ts, oi, vol",
        pagination_limit=("not paginatable; returns the most recent "
                           "~180 daily rows"),
    )
    if not r.ok:
        base.update({"status": "error", "http_status": r.status_code,
                      "notes": r.error or "no payload"})
        base.update(classify_usability(None, status="error"))
        return base
    rows = (r.payload or {}).get("data", []) or []
    if not rows:
        base.update({"status": "error", "http_status": r.status_code,
                      "notes": "empty data"})
        base.update(classify_usability(None, status="error"))
        return base
    ts = sorted(int(x[0]) for x in rows)
    start, end = ts[0], ts[-1]
    cov = _coverage_days(start, end)
    base.update({
        "status": "ok", "http_status": r.status_code,
        "actual_start": _ts_to_iso(start), "actual_end": _ts_to_iso(end),
        "row_count": len(rows), "coverage_days": cov,
        "notes": ("OKX rubik returns aggregate OI for the recent "
                   "~6 months only"),
    })
    base.update(classify_usability(cov))
    return base


def _probe_okx_rubik_long_short(asset_code: str = "BTC") -> Dict[str, Any]:
    url = ("https://www.okx.com/api/v5/rubik/stat/contracts/"
           f"long-short-account-ratio?ccy={asset_code}&period=1D")
    r = _http_get_json(url)
    base = _empty_row(
        "okx", "rubik_long_short_account_ratio",
        "/api/v5/rubik/stat/contracts/long-short-account-ratio", asset_code,
        granularity="1d",
        fields_available="ts, ratio",
        pagination_limit="recent rows only",
    )
    if not r.ok:
        base.update({"status": "error", "http_status": r.status_code,
                      "notes": r.error or "no payload"})
        base.update(classify_usability(None, status="error"))
        return base
    rows = (r.payload or {}).get("data", []) or []
    if not rows:
        base.update({"status": "error", "http_status": r.status_code,
                      "notes": "empty data"})
        base.update(classify_usability(None, status="error"))
        return base
    ts = sorted(int(x[0]) for x in rows)
    start, end = ts[0], ts[-1]
    cov = _coverage_days(start, end)
    base.update({
        "status": "ok", "http_status": r.status_code,
        "actual_start": _ts_to_iso(start), "actual_end": _ts_to_iso(end),
        "row_count": len(rows), "coverage_days": cov,
        "notes": "OKX rubik recent-only; not deep enough for OOS",
    })
    base.update(classify_usability(cov))
    return base


def _probe_okx_taker_volume(asset_code: str = "BTC") -> Dict[str, Any]:
    url = ("https://www.okx.com/api/v5/rubik/stat/taker-volume"
           f"?ccy={asset_code}&instType=SPOT&period=1D")
    r = _http_get_json(url)
    base = _empty_row(
        "okx", "rubik_taker_volume",
        "/api/v5/rubik/stat/taker-volume", asset_code,
        granularity="1d",
        fields_available="ts, sellVol, buyVol",
        pagination_limit="recent rows only",
    )
    if not r.ok:
        base.update({"status": "error", "http_status": r.status_code,
                      "notes": r.error or "no payload"})
        base.update(classify_usability(None, status="error"))
        return base
    rows = (r.payload or {}).get("data", []) or []
    if not rows:
        base.update({"status": "error", "http_status": r.status_code,
                      "notes": "empty data"})
        base.update(classify_usability(None, status="error"))
        return base
    ts = sorted(int(x[0]) for x in rows)
    start, end = ts[0], ts[-1]
    cov = _coverage_days(start, end)
    base.update({
        "status": "ok", "http_status": r.status_code,
        "actual_start": _ts_to_iso(start), "actual_end": _ts_to_iso(end),
        "row_count": len(rows), "coverage_days": cov,
        "notes": "OKX rubik taker-volume recent-only",
    })
    base.update(classify_usability(cov))
    return base


# ---------------------------------------------------------------------------
# Deribit
# ---------------------------------------------------------------------------
def _probe_deribit_funding(asset_code: str = "BTC-PERPETUAL"
                            ) -> Dict[str, Any]:
    """Deribit perp funding. Per request returns up to ~15 days of 1h
    rows; pagination by date range can extend back to 2019."""
    # Probe a 23-day window ~6 months ago.
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - 23 * 86_400_000
    url = ("https://www.deribit.com/api/v2/public/get_funding_rate_history"
           f"?instrument_name={asset_code}&start_timestamp={start_ts}"
           f"&end_timestamp={end_ts}")
    r = _http_get_json(url)
    base = _empty_row(
        "deribit", "perpetual_funding_rate_history",
        "/api/v2/public/get_funding_rate_history", asset_code,
        granularity="~1h",
        fields_available=("timestamp, interest_8h, interest_1h, "
                           "prev_index_price, index_price"),
        pagination_limit=("range-window endpoint; paginate by "
                           "moving start/end timestamps"),
    )
    if not r.ok:
        base.update({"status": "error", "http_status": r.status_code,
                      "notes": r.error or "no payload"})
        base.update(classify_usability(None, status="error"))
        return base
    rows = (r.payload or {}).get("result", []) or []
    if not rows:
        base.update({"status": "error", "http_status": r.status_code,
                      "notes": "empty result"})
        base.update(classify_usability(None, status="error"))
        return base
    ts = sorted(int(x["timestamp"]) for x in rows)
    # Pagination depth: BTC-PERPETUAL launched 2019-08-13 → ~6+ years.
    cov = _coverage_days(ts[0], ts[-1])
    paginated_cov = _coverage_days(int(datetime(2019, 8, 13,
                                                  tzinfo=timezone.utc
                                                  ).timestamp() * 1000),
                                     ts[-1])
    base.update({
        "status": "ok", "http_status": r.status_code,
        "actual_start": _ts_to_iso(ts[0]), "actual_end": _ts_to_iso(ts[-1]),
        "row_count": len(rows), "coverage_days": paginated_cov,
        "notes": (f"single 23-day window returned {len(rows)} hourly "
                   f"rows; paginated depth extends to perp launch "
                   f"(2019-08) — ~{int(paginated_cov or 0)} days available"),
    })
    base.update(classify_usability(paginated_cov))
    return base


def _probe_deribit_dvol(asset_code: str = "BTC") -> Dict[str, Any]:
    """Deribit DVOL (volatility index) history. Daily resolution, ~1000
    daily rows reachable per call."""
    end_ts = int(time.time() * 1000)
    start_ts = int(datetime(2022, 5, 27, tzinfo=timezone.utc
                              ).timestamp() * 1000)
    url = ("https://www.deribit.com/api/v2/public/get_volatility_index_data"
           f"?currency={asset_code}&start_timestamp={start_ts}"
           f"&end_timestamp={end_ts}&resolution=86400")
    r = _http_get_json(url)
    base = _empty_row(
        "deribit", "dvol_index_history",
        "/api/v2/public/get_volatility_index_data", asset_code,
        granularity="1d",
        fields_available="timestamp, open, high, low, close",
        pagination_limit=("up to 1000 rows per request; paginate via "
                           "start_timestamp/end_timestamp"),
    )
    if not r.ok:
        base.update({"status": "error", "http_status": r.status_code,
                      "notes": r.error or "no payload"})
        base.update(classify_usability(None, status="error"))
        return base
    rows = (r.payload or {}).get("result", {}).get("data", []) or []
    if not rows:
        base.update({"status": "error", "http_status": r.status_code,
                      "notes": "empty data"})
        base.update(classify_usability(None, status="error"))
        return base
    ts = sorted(int(x[0]) for x in rows)
    cov = _coverage_days(ts[0], ts[-1])
    base.update({
        "status": "ok", "http_status": r.status_code,
        "actual_start": _ts_to_iso(ts[0]), "actual_end": _ts_to_iso(ts[-1]),
        "row_count": len(rows), "coverage_days": cov,
        "notes": ("DVOL (BTC implied-vol index) — usable for vol "
                   "regime / positioning context"),
    })
    base.update(classify_usability(cov))
    return base


def _probe_deribit_book_summary(asset_code: str = "BTC") -> Dict[str, Any]:
    url = ("https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
           f"?currency={asset_code}&kind=future")
    r = _http_get_json(url)
    base = _empty_row(
        "deribit", "book_summary_by_currency",
        "/api/v2/public/get_book_summary_by_currency", asset_code,
        granularity="snapshot",
        fields_available="open_interest, volume, mark_price, index_price",
        pagination_limit="snapshot endpoint; no history",
    )
    if not r.ok:
        base.update({"status": "error", "http_status": r.status_code,
                      "notes": r.error or "no payload"})
        base.update(classify_usability(None, is_current_snapshot_only=True,
                                          status="error"))
        return base
    rows = (r.payload or {}).get("result", []) or []
    base.update({
        "status": "ok", "http_status": r.status_code,
        "row_count": len(rows),
        "notes": ("snapshot only — no historical OI; usable as a "
                   "live-state read but not for OOS research"),
    })
    base.update(classify_usability(0.0, is_current_snapshot_only=True))
    return base


# ---------------------------------------------------------------------------
# Paid / API-key sources — recorded as INCONCLUSIVE without probing.
# ---------------------------------------------------------------------------
def _record_paid_source(
    source: str, dataset: str, endpoint: str,
    fields: str, notes: str, asset: str = "BTC",
) -> Dict[str, Any]:
    base = _empty_row(
        source, dataset, endpoint, asset,
        requires_api_key=True, access_type="paid_only",
        free_access=False, paid_access=True,
        granularity="varies", fields_available=fields,
        pagination_limit="varies",
        notes=notes,
    )
    base.update(classify_usability(None, paid_only=True))
    base["status"] = "not_probed"
    return base


def _paid_sources() -> List[Dict[str, Any]]:
    """Sources that require either a paid plan or an API key. We do NOT
    probe them. The audit records them as INCONCLUSIVE so the report
    says "verify subscription before claiming usable"."""
    return [
        _record_paid_source(
            "coinglass", "open_interest_aggregated_history",
            "https://open-api.coinglass.com (v3, /futures/openInterest/...)",
            "OI, funding, liquidations, long/short ratios across "
            "exchanges",
            "CoinGlass v3 API requires a key. Paid tiers govern history "
            "depth and per-minute call quota. Cannot verify without "
            "subscription.",
        ),
        _record_paid_source(
            "cryptoquant", "exchange_flows_and_oi",
            "https://api.cryptoquant.com (v1, /btc/exchange-flows/...)",
            "exchange reserves, OI, funding, taker buy/sell, miner flows",
            "CryptoQuant API requires a paid key. Some metrics are "
            "free up to a coarse granularity; multi-year hourly data "
            "is paid-only. Cannot verify without subscription.",
        ),
        _record_paid_source(
            "glassnode", "exchange_balances_and_derivatives",
            "https://api.glassnode.com (v1, /metrics/...)",
            ("on-chain balances, exchange flows, derivatives OI, "
              "funding, options skew"),
            "Glassnode API requires a paid plan for granular history. "
            "Free tier exposes a small subset of daily metrics. Cannot "
            "verify without subscription.",
        ),
        _record_paid_source(
            "kaiko", "consolidated_derivatives_data",
            "https://us.market-api.kaiko.io",
            ("consolidated OI, funding, basis, taker volume across "
              "venues"),
            "Kaiko is enterprise-priced; no free tier. Cannot verify.",
        ),
        _record_paid_source(
            "velo_data", "derivatives_aggregates",
            "https://api.velodata.app",
            "OI, funding, basis, options skew — derivatives-focused",
            "Velo Data is paid-only with a developer plan. Cannot "
            "verify without subscription.",
        ),
    ]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
# Per-source probe table — each entry returns a single row.
PROBES_PER_BTC: Sequence[Callable[..., Dict[str, Any]]] = (
    _probe_binance_funding,
    _probe_binance_oi_hist,
    _probe_binance_top_long_short_account,
    _probe_binance_taker_long_short,
    _probe_binance_premium_basis,
    _probe_bybit_funding,
    _probe_bybit_oi,
    _probe_bybit_account_ratio,
    _probe_okx_funding,
    _probe_okx_rubik_oi,
    _probe_okx_rubik_long_short,
    _probe_okx_taker_volume,
    _probe_deribit_funding,
    _probe_deribit_dvol,
    _probe_deribit_book_summary,
)


def run_audit(
    save: bool = True,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run every probe, plus the paid-source ledger, and write the audit
    CSV. Always returns a DataFrame whose schema matches `_AUDIT_COLUMNS`.

    Network failures do not propagate — failed probes contribute a row
    with `status="error"` and `usable_for_research="FAIL"`.
    """
    logger.info("=== positioning_data_audit: %d probes ===",
                 len(PROBES_PER_BTC))
    rows: List[Dict[str, Any]] = []
    for probe in PROBES_PER_BTC:
        try:
            row = probe()
        except Exception as exc:  # noqa: BLE001 — fail-soft per spec
            logger.warning("probe %s raised %s", probe.__name__, exc)
            row = _empty_row(
                "unknown", probe.__name__, probe.__name__, "BTC",
                status="error", usable_for_research="FAIL",
                usable_reason=f"probe raised {type(exc).__name__}",
                notes=str(exc),
            )
        rows.append(row)
    rows.extend(_paid_sources())

    df = pd.DataFrame(rows, columns=_AUDIT_COLUMNS)
    if save:
        out = (output_path or
                config.RESULTS_DIR / "positioning_data_audit.csv")
        utils.write_df(df, out)
    return df


def summarise(df: pd.DataFrame) -> Dict[str, Any]:
    """Compact summary used by both the CLI and the report."""
    if df.empty:
        return {"n": 0, "pass": 0, "warning": 0, "fail": 0,
                "inconclusive": 0, "best": None}
    counts = df["usable_for_research"].value_counts().to_dict()
    pass_rows = df[df["usable_for_research"] == "PASS"]
    best = None
    if not pass_rows.empty:
        # Prefer the deepest coverage among PASS rows.
        ordered = pass_rows.sort_values(
            "coverage_days", ascending=False, na_position="last",
        )
        first = ordered.iloc[0]
        best = f"{first['source']} / {first['dataset']}"
    return {
        "n": int(len(df)),
        "pass": int(counts.get("PASS", 0)),
        "warning": int(counts.get("WARNING", 0)),
        "fail": int(counts.get("FAIL", 0)),
        "inconclusive": int(counts.get("INCONCLUSIVE", 0)),
        "best": best,
    }
