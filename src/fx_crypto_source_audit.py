"""
FX + Crypto data source audit.

Probes a curated list of free / public Forex and crypto data
endpoints and decides, per source × asset × field, whether enough
multi-year history is reachable to justify a future strategy
research branch.

Hard rules (locked):
    * Public endpoints only. NO API keys ever read or required.
    * No private endpoints (no order routing, balance, margin).
    * No paid plans, no scraping bypasses, no ToS violations.
    * Network failures NEVER crash the audit — every probe is
      wrapped in a guarded try/except and the row is recorded with
      `decision_status="FAIL"` (status="error") so the CSV is
      always produced.
    * Generated CSV at `results/fx_crypto_source_audit.csv` is
      gitignored under the existing `results/*.csv` rule.

Coverage classifier (locked, NEVER tuned):
    PASS         coverage_days >= 1460
    WARNING      365 <= coverage_days < 1460
    FAIL         coverage_days < 365 OR snapshot only OR endpoint
                  failed
    INCONCLUSIVE source likely works but cannot be verified — paid
                  plan, key required, docs gated, or rate-limited
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

logger = utils.get_logger("cte.fx_crypto_source_audit")


# ---------------------------------------------------------------------------
# Locked constants.
# ---------------------------------------------------------------------------
COVERAGE_PASS_DAYS = 1460
COVERAGE_WARNING_DAYS = 365

DEFAULT_TIMEOUT = 12.0
DEFAULT_USER_AGENT = "cte-fx-crypto-audit/0.1"

MARKET_FOREX = "forex"
MARKET_CRYPTO = "crypto"
MARKETS: Tuple[str, ...] = (MARKET_FOREX, MARKET_CRYPTO)

# Field-type vocabulary.
FIELD_OHLCV = "ohlcv"
FIELD_FUNDING = "funding"
FIELD_OPEN_INTEREST = "open_interest"
FIELD_BASIS = "basis_or_premium"
FIELD_LIQUIDATIONS = "liquidations"
FIELD_LONG_SHORT_RATIO = "long_short_ratio"
FIELD_ORDER_BOOK = "order_book_snapshot"
FIELD_VOL_INDEX = "vol_index"
FIELD_REFERENCE_RATE = "reference_rate"

FIELD_TYPES: Tuple[str, ...] = (
    FIELD_OHLCV, FIELD_FUNDING, FIELD_OPEN_INTEREST,
    FIELD_BASIS, FIELD_LIQUIDATIONS, FIELD_LONG_SHORT_RATIO,
    FIELD_ORDER_BOOK, FIELD_VOL_INDEX, FIELD_REFERENCE_RATE,
)

DECISION_PASS = "PASS"
DECISION_WARNING = "WARNING"
DECISION_FAIL = "FAIL"
DECISION_INCONCLUSIVE = "INCONCLUSIVE"
DECISION_STATUSES: Tuple[str, ...] = (
    DECISION_PASS, DECISION_WARNING, DECISION_FAIL, DECISION_INCONCLUSIVE,
)


# ---------------------------------------------------------------------------
# Output schema (locked per spec).
# ---------------------------------------------------------------------------
AUDIT_COLUMNS: List[str] = [
    "market",
    "source",
    "asset",
    "field_type",
    "endpoint_or_source",
    "requires_api_key",
    "free_access",
    "actual_start",
    "actual_end",
    "coverage_days",
    "granularity",
    "usable_for_research",
    "decision_status",
    "notes",
]


# ---------------------------------------------------------------------------
# Coverage classifier.
# ---------------------------------------------------------------------------
def classify_coverage(coverage_days: Optional[float], *,
                       status: str = "ok",
                       is_snapshot_only: bool = False,
                       requires_key: bool = False) -> str:
    if requires_key:
        return DECISION_INCONCLUSIVE
    if status != "ok":
        return DECISION_FAIL
    if is_snapshot_only:
        return DECISION_FAIL
    if coverage_days is None:
        return DECISION_INCONCLUSIVE
    cd = float(coverage_days)
    if cd < COVERAGE_WARNING_DAYS:
        return DECISION_FAIL
    if cd < COVERAGE_PASS_DAYS:
        return DECISION_WARNING
    return DECISION_PASS


# ---------------------------------------------------------------------------
# HTTP helper — fail-soft, never raises.
# ---------------------------------------------------------------------------
@dataclass
class _Response:
    ok: bool
    status_code: Optional[int]
    payload: Optional[Any]
    text: Optional[str]
    error: Optional[str]


def _http_get(url: str, timeout: float = DEFAULT_TIMEOUT,
                user_agent: str = DEFAULT_USER_AGENT,
                parse_json: bool = True) -> _Response:
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": user_agent},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            text = raw.decode("utf-8", errors="replace")
            payload: Any = None
            if parse_json:
                try:
                    payload = json.loads(text)
                except ValueError:
                    payload = None
            return _Response(True, r.status, payload, text, None)
    except urllib.error.HTTPError as exc:
        return _Response(False, exc.code, None, None,
                          f"http_error: {exc.reason}")
    except urllib.error.URLError as exc:
        return _Response(False, None, None, None,
                          f"url_error: {exc.reason}")
    except (TimeoutError, OSError) as exc:
        return _Response(False, None, None, None, f"network_error: {exc}")
    except Exception as exc:  # noqa: BLE001 — fail-soft policy
        return _Response(False, None, None, None,
                          f"unexpected: {type(exc).__name__}")


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
def _ts_to_iso(ts_seconds_or_ms: Optional[Any]) -> Optional[str]:
    if ts_seconds_or_ms is None:
        return None
    try:
        ts = int(ts_seconds_or_ms)
    except (ValueError, TypeError):
        return None
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
    s_secs = s / 1000.0 if s > 1_000_000_000_000 else float(s)
    e_secs = e / 1000.0 if e > 1_000_000_000_000 else float(e)
    return max(0.0, (e_secs - s_secs) / 86_400.0)


def _empty_row(market: str, source: str, asset: str, field_type: str,
                endpoint: str, **overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {c: None for c in AUDIT_COLUMNS}
    base.update({
        "market": market, "source": source, "asset": asset,
        "field_type": field_type, "endpoint_or_source": endpoint,
        "requires_api_key": False, "free_access": True,
        "coverage_days": None,
        "usable_for_research": DECISION_FAIL,
        "decision_status": DECISION_FAIL,
        "notes": None,
    })
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# FOREX probes
# ---------------------------------------------------------------------------
# ECB SDMX — daily reference rates EUR-quoted. Returns 27-year history
# for major fiat pairs; no key required.
def _probe_ecb_eurquoted(asset_pair: str, ccy: str,
                            http_get: Callable[[str], _Response] = _http_get,
                            ) -> Dict[str, Any]:
    url = (f"https://data-api.ecb.europa.eu/service/data/EXR/"
           f"D.{ccy}.EUR.SP00.A?format=csvdata")
    r = http_get(url)
    base = _empty_row(
        MARKET_FOREX, "ecb_sdmx", asset_pair,
        FIELD_REFERENCE_RATE, url,
        granularity="1d", notes=("ECB official EUR-quoted reference "
                                    "rate; daily fix only, no intraday, "
                                    "no spreads, no volume"),
    )
    if not r.ok or not r.text:
        base.update({"decision_status": DECISION_FAIL,
                      "usable_for_research": DECISION_FAIL,
                      "notes": (r.error or "no payload")})
        return base
    rows = [l for l in r.text.split("\n") if l and not l.startswith("KEY,")]
    if not rows:
        base.update({"decision_status": DECISION_INCONCLUSIVE,
                      "usable_for_research": DECISION_INCONCLUSIVE,
                      "notes": "empty CSV"})
        return base
    try:
        first_date = rows[0].split(",")[6]
        last_date = rows[-1].split(",")[6]
        s = datetime.fromisoformat(first_date).replace(
            tzinfo=timezone.utc)
        e = datetime.fromisoformat(last_date).replace(
            tzinfo=timezone.utc)
        cov = (e - s).days
        base.update({
            "actual_start": s.isoformat(),
            "actual_end": e.isoformat(),
            "coverage_days": float(cov),
        })
        base["decision_status"] = classify_coverage(cov)
        base["usable_for_research"] = base["decision_status"]
    except Exception as exc:  # noqa: BLE001
        base.update({"decision_status": DECISION_INCONCLUSIVE,
                      "usable_for_research": DECISION_INCONCLUSIVE,
                      "notes": f"parse error: {type(exc).__name__}"})
    return base


def _probe_frankfurter(asset_pair: str,
                          http_get: Callable[[str], _Response] = _http_get,
                          ) -> Dict[str, Any]:
    """Frankfurter wraps ECB; same data via friendly JSON. No key."""
    url = "https://api.frankfurter.app/2010-01-01..?from=EUR&to=USD,GBP,JPY,CHF"
    r = http_get(url)
    base = _empty_row(
        MARKET_FOREX, "frankfurter_app", asset_pair,
        FIELD_REFERENCE_RATE, url,
        granularity="1d",
        notes=("frankfurter.app wraps the ECB feed; same data, "
                "JSON-friendly; no key"),
    )
    if not r.ok or not isinstance(r.payload, dict):
        base.update({"decision_status": DECISION_FAIL,
                      "usable_for_research": DECISION_FAIL,
                      "notes": (r.error or "no payload")})
        return base
    rates = r.payload.get("rates", {})
    if not rates:
        base.update({"decision_status": DECISION_INCONCLUSIVE,
                      "usable_for_research": DECISION_INCONCLUSIVE,
                      "notes": "empty rates dict"})
        return base
    try:
        keys = sorted(rates.keys())
        first_date = keys[0]
        last_date = keys[-1]
        s = datetime.fromisoformat(first_date).replace(
            tzinfo=timezone.utc)
        e = datetime.fromisoformat(last_date).replace(
            tzinfo=timezone.utc)
        cov = (e - s).days
        base.update({
            "actual_start": s.isoformat(),
            "actual_end": e.isoformat(),
            "coverage_days": float(cov),
        })
        base["decision_status"] = classify_coverage(cov)
        base["usable_for_research"] = base["decision_status"]
    except Exception as exc:  # noqa: BLE001
        base.update({"decision_status": DECISION_INCONCLUSIVE,
                      "usable_for_research": DECISION_INCONCLUSIVE,
                      "notes": f"parse error: {type(exc).__name__}"})
    return base


def _probe_lbma_gold_pm(http_get: Callable[[str], _Response] = _http_get,
                          ) -> Dict[str, Any]:
    """LBMA London PM gold fix — daily, free JSON, since 1968."""
    url = "https://prices.lbma.org.uk/json/gold_pm.json"
    r = http_get(url)
    base = _empty_row(
        MARKET_FOREX, "lbma", "XAU/USD",
        FIELD_REFERENCE_RATE, url,
        granularity="1d",
        notes=("LBMA London PM gold fix — daily, free, since 1968"),
    )
    if not r.ok or not isinstance(r.payload, list) or not r.payload:
        base.update({"decision_status": DECISION_FAIL,
                      "usable_for_research": DECISION_FAIL,
                      "notes": (r.error or "no payload")})
        return base
    try:
        first = r.payload[0].get("d")
        last = r.payload[-1].get("d")
        s = datetime.fromisoformat(first).replace(tzinfo=timezone.utc)
        e = datetime.fromisoformat(last).replace(tzinfo=timezone.utc)
        cov = (e - s).days
        base.update({
            "actual_start": s.isoformat(),
            "actual_end": e.isoformat(),
            "coverage_days": float(cov),
        })
        base["decision_status"] = classify_coverage(cov)
        base["usable_for_research"] = base["decision_status"]
    except Exception as exc:  # noqa: BLE001
        base.update({"decision_status": DECISION_INCONCLUSIVE,
                      "usable_for_research": DECISION_INCONCLUSIVE,
                      "notes": f"parse error: {type(exc).__name__}"})
    return base


def _probe_yfinance_gold_futures(
    http_get: Callable[[str], _Response] = _http_get,
) -> Dict[str, Any]:
    """Yahoo Finance gold futures (GC=F) chart endpoint. Public,
    no key, but Yahoo is known to throttle / block automated access
    intermittently — recorded as a complementary source."""
    url = ("https://query2.finance.yahoo.com/v8/finance/chart/GC=F"
           "?range=10y&interval=1d")
    r = http_get(url)
    base = _empty_row(
        MARKET_FOREX, "yahoo_finance_unofficial", "XAU/USD",
        FIELD_OHLCV, url,
        granularity="1d",
        notes=("Yahoo Finance unofficial chart endpoint — works "
                "today, but Yahoo throttles automated access; "
                "treat as a complement to LBMA, not a primary source"),
    )
    if not r.ok or not isinstance(r.payload, dict):
        base.update({"decision_status": DECISION_INCONCLUSIVE,
                      "usable_for_research": DECISION_INCONCLUSIVE,
                      "notes": (r.error or "no payload")
                                + " — Yahoo's chart endpoint is not "
                                  "officially documented"})
        return base
    try:
        result = r.payload["chart"]["result"][0]
        timestamps = result.get("timestamp", []) or []
        if len(timestamps) < 2:
            base.update({"decision_status": DECISION_INCONCLUSIVE,
                          "usable_for_research": DECISION_INCONCLUSIVE,
                          "notes": "fewer than 2 timestamps returned"})
            return base
        cov = _coverage_days(timestamps[0], timestamps[-1])
        base.update({
            "actual_start": _ts_to_iso(timestamps[0]),
            "actual_end": _ts_to_iso(timestamps[-1]),
            "coverage_days": cov,
        })
        base["decision_status"] = classify_coverage(cov)
        base["usable_for_research"] = base["decision_status"]
    except Exception as exc:  # noqa: BLE001
        base.update({"decision_status": DECISION_INCONCLUSIVE,
                      "usable_for_research": DECISION_INCONCLUSIVE,
                      "notes": f"parse error: {type(exc).__name__}"})
    return base


# Sources that REQUIRE a key — recorded INCONCLUSIVE without probing.
def _record_key_required(market: str, source: str, asset: str,
                            field_type: str, endpoint: str,
                            notes: str) -> Dict[str, Any]:
    row = _empty_row(market, source, asset, field_type, endpoint,
                       requires_api_key=True, free_access=False,
                       notes=notes)
    row["decision_status"] = DECISION_INCONCLUSIVE
    row["usable_for_research"] = DECISION_INCONCLUSIVE
    return row


def _key_required_fx() -> List[Dict[str, Any]]:
    return [
        _record_key_required(
            MARKET_FOREX, "oanda_v20", "EUR/USD", FIELD_OHLCV,
            "https://api-fxpractice.oanda.com/v3/instruments/.../candles",
            "OANDA practice + live REST require an account-bound API "
            "key; no public unauthenticated path",
        ),
        _record_key_required(
            MARKET_FOREX, "ig_bank_switzerland", "EUR/USD", FIELD_OHLCV,
            "https://api.ig.com/gateway/deal/...",
            "IG Bank Switzerland REST requires authenticated session; "
            "no public unauthenticated path",
        ),
        _record_key_required(
            MARKET_FOREX, "stooq", "EUR/USD", FIELD_OHLCV,
            "https://stooq.com/q/d/l/?s=eurusd&i=d",
            "Stooq CSV download now requires apikey (captcha-gated) "
            "as of recent policy change — was free historically",
        ),
        _record_key_required(
            MARKET_FOREX, "dukascopy", "EUR/USD", FIELD_OHLCV,
            "https://datafeed.dukascopy.com/datafeed/.../*.bi5",
            "Dukascopy bi5 binary endpoint returns 503 to public "
            "User-Agents; data accessible only via JForex client / "
            "community libraries that scrape with browser-like UA",
        ),
        _record_key_required(
            MARKET_FOREX, "fred_st_louis", "EUR/USD", FIELD_REFERENCE_RATE,
            "https://api.stlouisfed.org/fred/series/observations",
            "FRED API requires a free key for series observations; "
            "the public CSV download endpoint times out from this "
            "environment",
        ),
    ]


# ---------------------------------------------------------------------------
# CRYPTO probes — re-validate the strategy-9 free-data audit findings
# ---------------------------------------------------------------------------
def _probe_binance_klines(asset: str, symbol: str,
                            http_get: Callable[[str], _Response] = _http_get,
                            ) -> Dict[str, Any]:
    url = ("https://api.binance.com/api/v3/klines"
           f"?symbol={symbol}&interval=1d&startTime=1500000000000"
           "&limit=1000")
    r = http_get(url)
    base = _empty_row(
        MARKET_CRYPTO, "binance", asset, FIELD_OHLCV, url,
        granularity="1d",
        notes="paginates via startTime; multi-year",
    )
    if not r.ok or not isinstance(r.payload, list) or not r.payload:
        base.update({"decision_status": DECISION_FAIL,
                      "usable_for_research": DECISION_FAIL,
                      "notes": (r.error or "empty")})
        return base
    earliest = int(r.payload[0][0])
    now_ms = int(time.time() * 1000)
    cov = _coverage_days(earliest, now_ms)
    base.update({
        "actual_start": _ts_to_iso(earliest),
        "actual_end": _ts_to_iso(now_ms),
        "coverage_days": cov,
    })
    base["decision_status"] = classify_coverage(cov)
    base["usable_for_research"] = base["decision_status"]
    return base


def _probe_binance_funding(asset: str, symbol: str,
                             http_get: Callable[[str], _Response] = _http_get,
                             ) -> Dict[str, Any]:
    url = ("https://fapi.binance.com/fapi/v1/fundingRate"
           f"?symbol={symbol}&startTime=1500000000000&limit=1000")
    r = http_get(url)
    base = _empty_row(
        MARKET_CRYPTO, "binance_futures", asset, FIELD_FUNDING, url,
        granularity="8h",
        notes="paginates via startTime; multi-year",
    )
    if not r.ok or not isinstance(r.payload, list) or not r.payload:
        base.update({"decision_status": DECISION_FAIL,
                      "usable_for_research": DECISION_FAIL,
                      "notes": (r.error or "empty")})
        return base
    earliest = int(r.payload[0]["fundingTime"])
    now_ms = int(time.time() * 1000)
    cov = _coverage_days(earliest, now_ms)
    base.update({
        "actual_start": _ts_to_iso(earliest),
        "actual_end": _ts_to_iso(now_ms),
        "coverage_days": cov,
    })
    base["decision_status"] = classify_coverage(cov)
    base["usable_for_research"] = base["decision_status"]
    return base


def _probe_binance_oi_history(asset: str, symbol: str,
                                http_get: Callable[[str], _Response] = _http_get,
                                ) -> Dict[str, Any]:
    url = ("https://fapi.binance.com/futures/data/openInterestHist"
           f"?symbol={symbol}&period=1d&limit=500")
    r = http_get(url)
    base = _empty_row(
        MARKET_CRYPTO, "binance_futures", asset,
        FIELD_OPEN_INTEREST, url,
        granularity="1d",
        notes="hard-capped ~30 days regardless of limit",
    )
    if not r.ok or not isinstance(r.payload, list) or not r.payload:
        base.update({"decision_status": DECISION_FAIL,
                      "usable_for_research": DECISION_FAIL,
                      "notes": (r.error or "empty")})
        return base
    start = int(r.payload[0]["timestamp"])
    end = int(r.payload[-1]["timestamp"])
    cov = _coverage_days(start, end)
    base.update({
        "actual_start": _ts_to_iso(start), "actual_end": _ts_to_iso(end),
        "coverage_days": cov,
    })
    base["decision_status"] = classify_coverage(cov)
    base["usable_for_research"] = base["decision_status"]
    return base


def _probe_binance_basis(asset: str, symbol: str,
                           http_get: Callable[[str], _Response] = _http_get,
                           ) -> Dict[str, Any]:
    url = ("https://fapi.binance.com/fapi/v1/markPriceKlines"
           f"?symbol={symbol}&interval=1d&limit=1500")
    r = http_get(url)
    base = _empty_row(
        MARKET_CRYPTO, "binance_futures", asset, FIELD_BASIS, url,
        granularity="1d",
        notes=("mark + index klines together = futures-spot "
                "basis proxy"),
    )
    if not r.ok or not isinstance(r.payload, list) or not r.payload:
        base.update({"decision_status": DECISION_FAIL,
                      "usable_for_research": DECISION_FAIL,
                      "notes": (r.error or "empty")})
        return base
    earliest = int(r.payload[0][0])
    latest = int(r.payload[-1][0])
    cov = _coverage_days(earliest, latest)
    base.update({
        "actual_start": _ts_to_iso(earliest),
        "actual_end": _ts_to_iso(latest),
        "coverage_days": cov,
    })
    base["decision_status"] = classify_coverage(cov)
    base["usable_for_research"] = base["decision_status"]
    return base


def _probe_bybit_funding(asset: str, symbol: str,
                           http_get: Callable[[str], _Response] = _http_get,
                           ) -> Dict[str, Any]:
    url = ("https://api.bybit.com/v5/market/funding/history"
           f"?category=linear&symbol={symbol}"
           "&startTime=1500000000000&endTime=1600000000000&limit=200")
    r = http_get(url)
    base = _empty_row(
        MARKET_CRYPTO, "bybit", asset, FIELD_FUNDING, url,
        granularity="8h",
        notes="paginates via endTime; depth = (now - earliest)",
    )
    if not r.ok:
        base.update({"decision_status": DECISION_FAIL,
                      "usable_for_research": DECISION_FAIL,
                      "notes": (r.error or "no payload")})
        return base
    rows = (r.payload or {}).get("result", {}).get("list", []) or []
    if not rows:
        base.update({"decision_status": DECISION_INCONCLUSIVE,
                      "usable_for_research": DECISION_INCONCLUSIVE,
                      "notes": "empty list"})
        return base
    earliest = min(int(x["fundingRateTimestamp"]) for x in rows)
    now_ms = int(time.time() * 1000)
    cov = _coverage_days(earliest, now_ms)
    base.update({
        "actual_start": _ts_to_iso(earliest),
        "actual_end": _ts_to_iso(now_ms),
        "coverage_days": cov,
    })
    base["decision_status"] = classify_coverage(cov)
    base["usable_for_research"] = base["decision_status"]
    return base


def _probe_bybit_oi(asset: str, symbol: str,
                      http_get: Callable[[str], _Response] = _http_get,
                      ) -> Dict[str, Any]:
    url = ("https://api.bybit.com/v5/market/open-interest"
           f"?category=linear&symbol={symbol}&intervalTime=1d&limit=200")
    r = http_get(url)
    base = _empty_row(
        MARKET_CRYPTO, "bybit", asset, FIELD_OPEN_INTEREST, url,
        granularity="1d",
        notes="capped at recent ~200 days",
    )
    if not r.ok:
        base.update({"decision_status": DECISION_FAIL,
                      "usable_for_research": DECISION_FAIL,
                      "notes": (r.error or "no payload")})
        return base
    rows = (r.payload or {}).get("result", {}).get("list", []) or []
    if not rows:
        base.update({"decision_status": DECISION_INCONCLUSIVE,
                      "usable_for_research": DECISION_INCONCLUSIVE,
                      "notes": "empty list"})
        return base
    ts = sorted(int(x["timestamp"]) for x in rows)
    cov = _coverage_days(ts[0], ts[-1])
    base.update({
        "actual_start": _ts_to_iso(ts[0]),
        "actual_end": _ts_to_iso(ts[-1]),
        "coverage_days": cov,
    })
    base["decision_status"] = classify_coverage(cov)
    base["usable_for_research"] = base["decision_status"]
    return base


def _probe_okx_funding(asset: str, instrument: str,
                         http_get: Callable[[str], _Response] = _http_get,
                         ) -> Dict[str, Any]:
    url = ("https://www.okx.com/api/v5/public/funding-rate-history"
           f"?instId={instrument}&limit=100")
    r = http_get(url)
    base = _empty_row(
        MARKET_CRYPTO, "okx", asset, FIELD_FUNDING, url,
        granularity="8h",
        notes="public endpoint cursor-walks back ~100 days only",
    )
    if not r.ok:
        base.update({"decision_status": DECISION_FAIL,
                      "usable_for_research": DECISION_FAIL,
                      "notes": (r.error or "no payload")})
        return base
    rows = (r.payload or {}).get("data", []) or []
    if not rows:
        base.update({"decision_status": DECISION_INCONCLUSIVE,
                      "usable_for_research": DECISION_INCONCLUSIVE,
                      "notes": "empty data"})
        return base
    ts = sorted(int(x["fundingTime"]) for x in rows)
    cov = _coverage_days(ts[0], ts[-1])
    base.update({
        "actual_start": _ts_to_iso(ts[0]), "actual_end": _ts_to_iso(ts[-1]),
        "coverage_days": cov,
    })
    base["decision_status"] = classify_coverage(cov)
    base["usable_for_research"] = base["decision_status"]
    return base


def _probe_deribit_funding(asset: str, instrument: str,
                              http_get: Callable[[str], _Response] = _http_get,
                              ) -> Dict[str, Any]:
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - 23 * 86_400_000
    url = ("https://www.deribit.com/api/v2/public/get_funding_rate_history"
           f"?instrument_name={instrument}"
           f"&start_timestamp={start_ts}&end_timestamp={end_ts}")
    r = http_get(url)
    base = _empty_row(
        MARKET_CRYPTO, "deribit", asset, FIELD_FUNDING, url,
        granularity="~1h",
        notes="range-window endpoint; perp launched 2019-08",
    )
    if not r.ok:
        base.update({"decision_status": DECISION_FAIL,
                      "usable_for_research": DECISION_FAIL,
                      "notes": (r.error or "no payload")})
        return base
    rows = (r.payload or {}).get("result", []) or []
    if not rows:
        base.update({"decision_status": DECISION_INCONCLUSIVE,
                      "usable_for_research": DECISION_INCONCLUSIVE,
                      "notes": "empty result"})
        return base
    perp_launch_ms = int(datetime(2019, 8, 13,
                                     tzinfo=timezone.utc
                                     ).timestamp() * 1000)
    latest = max(int(x["timestamp"]) for x in rows)
    cov = _coverage_days(perp_launch_ms, latest)
    base.update({
        "actual_start": _ts_to_iso(perp_launch_ms),
        "actual_end": _ts_to_iso(latest),
        "coverage_days": cov,
    })
    base["decision_status"] = classify_coverage(cov)
    base["usable_for_research"] = base["decision_status"]
    return base


def _probe_deribit_book_summary(http_get: Callable[[str], _Response] = _http_get,
                                   ) -> Dict[str, Any]:
    url = ("https://www.deribit.com/api/v2/public/"
           "get_book_summary_by_currency?currency=BTC&kind=future")
    r = http_get(url)
    base = _empty_row(
        MARKET_CRYPTO, "deribit", "BTC", FIELD_ORDER_BOOK, url,
        granularity="snapshot",
        notes=("snapshot only — current OI per future contract; "
                "no historical depth on the public endpoint"),
    )
    if not r.ok:
        base.update({"decision_status": DECISION_FAIL,
                      "usable_for_research": DECISION_FAIL,
                      "notes": (r.error or "no payload")})
        return base
    rows = (r.payload or {}).get("result", []) or []
    base["decision_status"] = classify_coverage(
        0.0, is_snapshot_only=True,
    )
    base["usable_for_research"] = base["decision_status"]
    return base


def _probe_kraken_ohlc(asset: str, kraken_pair: str,
                          http_get: Callable[[str], _Response] = _http_get,
                          ) -> Dict[str, Any]:
    url = (f"https://api.kraken.com/0/public/OHLC"
           f"?pair={kraken_pair}&interval=1440")
    r = http_get(url)
    base = _empty_row(
        MARKET_CRYPTO, "kraken", asset, FIELD_OHLCV, url,
        granularity="1d",
        notes="hard-capped at 720 most-recent candles regardless of since",
    )
    if not r.ok:
        base.update({"decision_status": DECISION_FAIL,
                      "usable_for_research": DECISION_FAIL,
                      "notes": (r.error or "no payload")})
        return base
    res = (r.payload or {}).get("result", {}) or {}
    pair_keys = [k for k in res.keys() if k != "last"]
    if not pair_keys:
        base.update({"decision_status": DECISION_INCONCLUSIVE,
                      "usable_for_research": DECISION_INCONCLUSIVE,
                      "notes": "no pair key in result"})
        return base
    rows = res[pair_keys[0]]
    if not rows:
        base.update({"decision_status": DECISION_INCONCLUSIVE,
                      "usable_for_research": DECISION_INCONCLUSIVE,
                      "notes": "empty rows"})
        return base
    start = int(rows[0][0])
    end = int(rows[-1][0])
    cov = _coverage_days(start, end)
    base.update({
        "actual_start": _ts_to_iso(start), "actual_end": _ts_to_iso(end),
        "coverage_days": cov,
    })
    base["decision_status"] = classify_coverage(cov)
    base["usable_for_research"] = base["decision_status"]
    return base


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def _make_probes(http_get: Callable[[str], _Response] = _http_get,
                  ) -> List[Callable[[], Dict[str, Any]]]:
    """Return zero-arg callables, one per probe."""
    probes: List[Callable[[], Dict[str, Any]]] = []
    # Forex — ECB EUR-quoted majors.
    for pair, ccy in (("EUR/USD", "USD"), ("EUR/GBP", "GBP"),
                          ("EUR/JPY", "JPY"), ("EUR/CHF", "CHF")):
        probes.append(lambda p=pair, c=ccy: _probe_ecb_eurquoted(
            p, c, http_get=http_get))
    # Frankfurter (cross-check on EUR/USD).
    probes.append(lambda: _probe_frankfurter("EUR/USD",
                                                  http_get=http_get))
    # Cross-rates derivable from ECB → record explicitly so the
    # report shows them.
    probes.append(lambda: _probe_ecb_eurquoted(
        "USD/JPY (derived)", "JPY", http_get=http_get))
    probes.append(lambda: _probe_ecb_eurquoted(
        "USD/CHF (derived)", "CHF", http_get=http_get))
    probes.append(lambda: _probe_ecb_eurquoted(
        "GBP/USD (derived)", "GBP", http_get=http_get))
    # Gold.
    probes.append(lambda: _probe_lbma_gold_pm(http_get=http_get))
    probes.append(lambda: _probe_yfinance_gold_futures(http_get=http_get))

    # Crypto.
    for asset, sym in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT"),
                          ("SOL", "SOLUSDT")):
        probes.append(lambda a=asset, s=sym: _probe_binance_klines(
            a, s, http_get=http_get))
    for asset, sym in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")):
        probes.append(lambda a=asset, s=sym: _probe_binance_funding(
            a, s, http_get=http_get))
    probes.append(lambda: _probe_binance_oi_history(
        "BTC", "BTCUSDT", http_get=http_get))
    for asset, sym in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")):
        probes.append(lambda a=asset, s=sym: _probe_binance_basis(
            a, s, http_get=http_get))
    probes.append(lambda: _probe_bybit_funding(
        "BTC", "BTCUSDT", http_get=http_get))
    probes.append(lambda: _probe_bybit_oi(
        "BTC", "BTCUSDT", http_get=http_get))
    probes.append(lambda: _probe_okx_funding(
        "BTC", "BTC-USDT-SWAP", http_get=http_get))
    for asset, instr in (("BTC", "BTC-PERPETUAL"),
                              ("ETH", "ETH-PERPETUAL")):
        probes.append(lambda a=asset, i=instr: _probe_deribit_funding(
            a, i, http_get=http_get))
    probes.append(lambda: _probe_deribit_book_summary(http_get=http_get))
    probes.append(lambda: _probe_kraken_ohlc(
        "BTC", "XBTUSD", http_get=http_get))
    return probes


def run_audit(
    save: bool = True,
    output_path: Optional[Path] = None,
    rate_delay_s: float = 0.20,
    http_get: Callable[[str], _Response] = _http_get,
) -> pd.DataFrame:
    """Run every probe + the key-required ledger; write the CSV;
    return the DataFrame."""
    utils.assert_paper_only()
    rows: List[Dict[str, Any]] = []
    probes = _make_probes(http_get=http_get)
    for i, probe_fn in enumerate(probes):
        try:
            row = probe_fn()
        except Exception as exc:  # noqa: BLE001 — fail-soft per spec
            logger.warning("probe %d raised %s",
                              i, type(exc).__name__)
            row = _empty_row("unknown", "unknown", "?", FIELD_OHLCV, "n/a",
                                decision_status=DECISION_INCONCLUSIVE,
                                usable_for_research=DECISION_INCONCLUSIVE,
                                notes=f"probe raised "
                                        f"{type(exc).__name__}")
        rows.append(row)
        if i + 1 < len(probes) and rate_delay_s > 0:
            time.sleep(rate_delay_s)
    rows.extend(_key_required_fx())

    df = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    if save:
        out = output_path or (config.RESULTS_DIR
                                / "fx_crypto_source_audit.csv")
        utils.write_df(df, out)
    return df


def summarise(df: pd.DataFrame) -> Dict[str, Any]:
    """Compact summary for the CLI + dashboard."""
    if df.empty:
        return {"n": 0, "fx_pass": 0, "fx_warning": 0, "fx_fail": 0,
                "fx_inconclusive": 0,
                "crypto_pass": 0, "crypto_warning": 0,
                "crypto_fail": 0, "crypto_inconclusive": 0,
                "fx_viable": False, "crypto_viable": False}

    fx = df[df["market"] == MARKET_FOREX]
    cr = df[df["market"] == MARKET_CRYPTO]

    def _counts(sub: pd.DataFrame) -> Dict[str, int]:
        c = sub["decision_status"].value_counts().to_dict()
        return {
            "pass": int(c.get(DECISION_PASS, 0)),
            "warning": int(c.get(DECISION_WARNING, 0)),
            "fail": int(c.get(DECISION_FAIL, 0)),
            "inconclusive": int(c.get(DECISION_INCONCLUSIVE, 0)),
        }
    fx_c = _counts(fx)
    cr_c = _counts(cr)

    return {
        "n": int(len(df)),
        "fx_pass": fx_c["pass"], "fx_warning": fx_c["warning"],
        "fx_fail": fx_c["fail"], "fx_inconclusive": fx_c["inconclusive"],
        "crypto_pass": cr_c["pass"], "crypto_warning": cr_c["warning"],
        "crypto_fail": cr_c["fail"],
        "crypto_inconclusive": cr_c["inconclusive"],
        # Viability: at least one PASS row in that market means
        # multi-year free data exists for at least one (asset, field).
        "fx_viable": bool(fx_c["pass"] >= 1),
        "crypto_viable": bool(cr_c["pass"] >= 1),
    }
