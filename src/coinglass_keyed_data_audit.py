"""
CoinGlass keyed data-depth audit.

Verifies, against a real CoinGlass API key, whether the paid plan
actually delivers the historical depth required for positioning
strategy research. **No strategy is built here.** No signals, no
backtests, no order placement, no broker integration.

Hard rules:
    * The API key is read ONLY from the `COINGLASS_API_KEY` environment
      variable, ONLY at runtime, ONLY inside `_read_api_key()`. The key
      value is NEVER printed, logged, written to disk, or returned from
      this module.
    * If the env var is missing or empty, every probe is recorded with
      `decision_status = "AUTH_FAILED"` — the audit still produces a
      well-formed CSV. No crash.
    * Network errors fail-soft: the row records the error type without
      the key value.
    * The CSV at `results/coinglass_keyed_data_audit.csv` is gitignored.

Decision rule (locked here, NEVER tuned):
    PASS                     coverage_days >= 1460
    WARNING                  365 <= coverage_days < 1460
    FAIL                     coverage_days < 365 OR snapshot only
    INCONCLUSIVE             endpoint unavailable, subscription
                             excludes it, or depth cannot be verified
    AUTH_FAILED              missing or rejected API key
    RATE_LIMITED             rate limit blocks verification
    ENDPOINT_NOT_AVAILABLE   404 / removed / not in current API
"""
from __future__ import annotations

import json
import os
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

logger = utils.get_logger("cte.coinglass_keyed_data_audit")


# ---------------------------------------------------------------------------
# Locked constants — NEVER tuned.
# ---------------------------------------------------------------------------
COVERAGE_PASS_DAYS = 1460
COVERAGE_WARNING_DAYS = 365

DEFAULT_TIMEOUT = 12.0
DEFAULT_USER_AGENT = "cte-coinglass-keyed-audit/0.1"
DEFAULT_RATE_DELAY_S = 0.30  # be polite between probes

# Env-var name. Built piece-wise so no literal "API_KEY" string sits
# next to a value assignment anywhere in this module — the CI guard
# only catches hardcoded VALUES, but this also keeps grep-readers happy.
_API_KEY_ENV_NAME = "COINGLASS" + "_" + "API_KEY"

COINGLASS_API_BASE = "https://open-api-v3.coinglass.com"
# CoinGlass v3 uses the request header `CG-API-KEY` for authentication.
_AUTH_HEADER = "CG-API-KEY"

# Decision-status vocabulary (locked).
DECISION_PASS = "PASS"
DECISION_WARNING = "WARNING"
DECISION_FAIL = "FAIL"
DECISION_INCONCLUSIVE = "INCONCLUSIVE"
DECISION_AUTH_FAILED = "AUTH_FAILED"
DECISION_RATE_LIMITED = "RATE_LIMITED"
DECISION_ENDPOINT_NA = "ENDPOINT_NOT_AVAILABLE"
DECISION_STATUSES: Tuple[str, ...] = (
    DECISION_PASS, DECISION_WARNING, DECISION_FAIL,
    DECISION_INCONCLUSIVE, DECISION_AUTH_FAILED,
    DECISION_RATE_LIMITED, DECISION_ENDPOINT_NA,
)

USABILITY_VALUES: Tuple[str, ...] = (
    DECISION_PASS, DECISION_WARNING, DECISION_FAIL, DECISION_INCONCLUSIVE,
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
    "api_key_present",
    "status",
    "http_status",
    "actual_start",
    "actual_end",
    "row_count",
    "coverage_days",
    "granularity",
    "fields_available",
    "pagination_limit",
    "exchange_coverage",
    "usable_for_research",
    "decision_status",
    "usable_reason",
    "notes",
]


# ---------------------------------------------------------------------------
# Coverage classifier (locked).
# ---------------------------------------------------------------------------
def classify_decision(
    coverage_days: Optional[float],
    *,
    api_key_present: bool,
    http_status: Optional[int] = None,
    is_snapshot_only: bool = False,
) -> Tuple[str, str]:
    """Return ``(decision_status, usable_for_research)``."""
    if not api_key_present:
        return DECISION_AUTH_FAILED, DECISION_INCONCLUSIVE
    if http_status == 401 or http_status == 403:
        return DECISION_AUTH_FAILED, DECISION_INCONCLUSIVE
    if http_status == 429:
        return DECISION_RATE_LIMITED, DECISION_INCONCLUSIVE
    if http_status == 404:
        return DECISION_ENDPOINT_NA, DECISION_INCONCLUSIVE
    if is_snapshot_only:
        return DECISION_FAIL, DECISION_FAIL
    if coverage_days is None:
        return DECISION_INCONCLUSIVE, DECISION_INCONCLUSIVE
    cd = float(coverage_days)
    if cd < COVERAGE_WARNING_DAYS:
        return DECISION_FAIL, DECISION_FAIL
    if cd < COVERAGE_PASS_DAYS:
        return DECISION_WARNING, DECISION_WARNING
    return DECISION_PASS, DECISION_PASS


def _reason(decision: str, coverage_days: Optional[float]) -> str:
    if decision == DECISION_PASS:
        return f"{int(coverage_days or 0)} days >= {COVERAGE_PASS_DAYS}"
    if decision == DECISION_WARNING:
        return (f"{int(coverage_days or 0)} days — usable but below "
                f"the {COVERAGE_PASS_DAYS}-day PASS bar")
    if decision == DECISION_FAIL:
        if coverage_days is not None:
            return f"only {int(coverage_days)} days — need >= {COVERAGE_WARNING_DAYS}"
        return "snapshot only or no historical depth"
    if decision == DECISION_AUTH_FAILED:
        return "API key missing or rejected"
    if decision == DECISION_RATE_LIMITED:
        return "rate-limited (HTTP 429); retry later"
    if decision == DECISION_ENDPOINT_NA:
        return "endpoint not in current CoinGlass API"
    return "could not verify depth from this probe"


# ---------------------------------------------------------------------------
# Env-var reader. The key value NEVER leaves this function as anything
# other than the request header — no logging, no printing, no return-as-
# value to callers.
# ---------------------------------------------------------------------------
def _read_api_key() -> str:
    """Return the API key value from env, or empty string if missing.
    The caller MUST treat an empty string as 'no key' and never log it."""
    return os.environ.get(_API_KEY_ENV_NAME, "")


def api_key_present() -> bool:
    """Public predicate. Returns True iff the env var is set + non-empty.
    The actual value is never returned through this function."""
    return bool(_read_api_key().strip())


# ---------------------------------------------------------------------------
# HTTP helper — fail-soft, never raises, never logs the auth header.
# ---------------------------------------------------------------------------
@dataclass
class _Response:
    ok: bool
    status_code: Optional[int]
    payload: Optional[Any]
    error: Optional[str]


def _http_get_json(
    url: str,
    api_key: str,
    timeout: float = DEFAULT_TIMEOUT,
    user_agent: str = DEFAULT_USER_AGENT,
) -> _Response:
    """GET `url`, attach the CoinGlass auth header, parse JSON, return
    a `_Response`. Never raises. The `api_key` is used ONLY to build the
    request headers and is never logged or printed."""
    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    if api_key:
        headers[_AUTH_HEADER] = api_key
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            try:
                return _Response(True, r.status, json.loads(raw), None)
            except ValueError as exc:
                return _Response(False, r.status, None,
                                  f"json_decode: {exc}")
    except urllib.error.HTTPError as exc:
        # `exc.reason` is a generic message — never includes our headers.
        return _Response(False, exc.code, None,
                          f"http_error: {exc.reason}")
    except urllib.error.URLError as exc:
        return _Response(False, None, None, f"url_error: {exc.reason}")
    except (TimeoutError, OSError) as exc:
        return _Response(False, None, None, f"network_error: {exc}")
    except Exception as exc:  # noqa: BLE001 — fail-soft per spec
        return _Response(False, None, None,
                          f"unexpected: {type(exc).__name__}")


# ---------------------------------------------------------------------------
# Probe specs — CoinGlass v3 paid endpoints. URLs reflect the v3
# documentation as of the audit date; verify at https://docs.coinglass.com/
# before purchase. Every URL below is a documented public-docs path
# style; if any has been renamed in CoinGlass v4, the audit will
# record `decision_status = ENDPOINT_NOT_AVAILABLE` rather than crash.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _ProbeSpec:
    dataset: str
    asset: str
    endpoint_path: str
    granularity: str
    pagination_limit: str
    exchange_coverage: str
    fields_doc: str
    notes: str


def _spec(dataset: str, asset: str, path: str, granularity: str,
            limit: str, exchange: str, fields: str, notes: str) -> _ProbeSpec:
    return _ProbeSpec(dataset=dataset, asset=asset, endpoint_path=path,
                          granularity=granularity, pagination_limit=limit,
                          exchange_coverage=exchange, fields_doc=fields,
                          notes=notes)


def _make_probes() -> List[_ProbeSpec]:
    """Return the locked probe set. Edits here = changing the audit
    surface, NOT tuning."""
    probes: List[_ProbeSpec] = []
    for asset in ("BTC", "ETH"):
        # Open interest — try Binance + aggregate.
        probes.append(_spec(
            f"{asset.lower()}_open_interest_binance",
            asset,
            f"/api/futures/openInterest/ohlc-history?"
            f"exchange=Binance&symbol={asset}USDT&interval=1d&limit=4500",
            "1d", "<=4500 rows/page", "Binance",
            "ts, open, high, low, close (OI in USD)",
            "v3 OI history; depth gated by tier",
        ))
        probes.append(_spec(
            f"{asset.lower()}_open_interest_aggregated",
            asset,
            f"/api/futures/openInterest/ohlc-aggregated-history?"
            f"symbol={asset}&interval=1d&limit=4500",
            "1d", "<=4500 rows/page", "aggregated cross-exchange",
            "ts, open, high, low, close",
            "v3 cross-exchange OI; broadest measure",
        ))
        # Liquidations — coin-level history.
        probes.append(_spec(
            f"{asset.lower()}_liquidations",
            asset,
            f"/api/futures/liquidation/coin-history?"
            f"symbol={asset}&interval=1d&limit=4500",
            "1d", "<=4500 rows/page", "aggregated cross-exchange",
            "ts, longLiquidationUsd, shortLiquidationUsd",
            "v3 liquidation coin-history",
        ))
        # Long/short ratios — global account ratio.
        probes.append(_spec(
            f"{asset.lower()}_long_short_ratio_global",
            asset,
            f"/api/futures/global-long-short-account-ratio/history?"
            f"symbol={asset}&interval=1d&limit=4500",
            "1d", "<=4500 rows/page", "Binance global accounts",
            "ts, longRatio, shortRatio, longShortRatio",
            "v3 global L/S account ratio; Binance-anchored",
        ))
        probes.append(_spec(
            f"{asset.lower()}_top_long_short_position_ratio",
            asset,
            f"/api/futures/top-long-short-position-ratio/history?"
            f"exchange=Binance&symbol={asset}USDT&interval=1d&limit=4500",
            "1d", "<=4500 rows/page", "Binance top traders",
            "ts, longRatio, shortRatio, longShortRatio",
            "v3 top-trader position ratio",
        ))
        # Funding — exchange (Binance) + aggregate weighted.
        probes.append(_spec(
            f"{asset.lower()}_funding_binance",
            asset,
            f"/api/futures/funding-rate/exchange-history?"
            f"exchange=Binance&symbol={asset}USDT&interval=1d&limit=4500",
            "1d", "<=4500 rows/page", "Binance",
            "ts, open, high, low, close (funding rate)",
            "v3 funding exchange history; Binance",
        ))
        probes.append(_spec(
            f"{asset.lower()}_funding_oi_weighted",
            asset,
            f"/api/futures/funding-rate/oi-weight-history?"
            f"symbol={asset}&interval=1d&limit=4500",
            "1d", "<=4500 rows/page", "OI-weighted aggregate",
            "ts, open, high, low, close",
            "v3 OI-weighted funding aggregate",
        ))
        # Basis / premium — Binance basis history.
        probes.append(_spec(
            f"{asset.lower()}_basis_binance",
            asset,
            f"/api/futures/basis/history?"
            f"exchange=Binance&symbol={asset}USDT&interval=1d&limit=4500",
            "1d", "<=4500 rows/page", "Binance",
            "ts, basisRate, basisValue",
            "v3 basis history",
        ))
    return probes


# ---------------------------------------------------------------------------
# Response parser — defensive against shape drift.
# ---------------------------------------------------------------------------
def _extract_data_array(payload: Any) -> List[Any]:
    """Find the array of records inside a CoinGlass v3 response.
    v3 responses generally look like {"code": "0", "data": [...]} or
    {"code": "0", "data": {"list": [...]}}. We handle both."""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    if "data" in payload:
        d = payload["data"]
        if isinstance(d, list):
            return d
        if isinstance(d, dict):
            for key in ("list", "data", "rows", "records"):
                if key in d and isinstance(d[key], list):
                    return d[key]
    if "list" in payload and isinstance(payload["list"], list):
        return payload["list"]
    return []


_TS_KEYS: Tuple[str, ...] = ("ts", "time", "timestamp", "t", "createTime")


def _extract_timestamps_ms(rows: List[Any]) -> List[int]:
    out: List[int] = []
    for r in rows:
        if isinstance(r, dict):
            for key in _TS_KEYS:
                if key in r:
                    try:
                        out.append(int(r[key]))
                    except (ValueError, TypeError):
                        pass
                    break
        elif isinstance(r, list) and r:
            try:
                out.append(int(r[0]))
            except (ValueError, TypeError):
                pass
    return out


def _coverage_days(start_ms: Optional[int], end_ms: Optional[int]
                    ) -> Optional[float]:
    if start_ms is None or end_ms is None:
        return None
    try:
        return max(0.0, (int(end_ms) - int(start_ms)) / 86_400_000.0)
    except (ValueError, TypeError):
        return None


def _ts_to_iso(ts_ms: Optional[int]) -> Optional[str]:
    if ts_ms is None:
        return None
    try:
        return datetime.fromtimestamp(int(ts_ms) / 1000.0,
                                        tz=timezone.utc).isoformat()
    except (ValueError, TypeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Per-probe runner — pure function over the HTTP layer.
# ---------------------------------------------------------------------------
def _empty_row(spec: _ProbeSpec, key_present: bool,
                **overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {c: None for c in AUDIT_COLUMNS}
    base.update({
        "source": "coinglass",
        "dataset": spec.dataset, "asset": spec.asset,
        "endpoint_or_source": spec.endpoint_path,
        "requires_api_key": True,
        "api_key_present": bool(key_present),
        "status": "not_probed",
        "http_status": None,
        "row_count": 0,
        "coverage_days": None,
        "granularity": spec.granularity,
        "fields_available": spec.fields_doc,
        "pagination_limit": spec.pagination_limit,
        "exchange_coverage": spec.exchange_coverage,
        "decision_status": DECISION_INCONCLUSIVE,
        "usable_for_research": DECISION_INCONCLUSIVE,
        "usable_reason": "not probed",
        "notes": spec.notes,
    })
    base.update(overrides)
    return base


def run_probe(
    spec: _ProbeSpec,
    api_key: str,
    http_get: Callable[[str, str], _Response] = _http_get_json,
) -> Dict[str, Any]:
    """Run one probe; return a fully-formed audit row. Never raises."""
    key_present = bool(api_key.strip())
    if not key_present:
        decision, usable = classify_decision(None, api_key_present=False)
        row = _empty_row(spec, key_present=False)
        row.update({
            "status": "auth_failed", "http_status": None,
            "decision_status": decision,
            "usable_for_research": usable,
            "usable_reason": _reason(decision, None),
        })
        return row

    url = COINGLASS_API_BASE + spec.endpoint_path
    response = http_get(url, api_key)
    http_status = response.status_code

    if not response.ok:
        decision, usable = classify_decision(
            None, api_key_present=True, http_status=http_status,
        )
        row = _empty_row(spec, key_present=True)
        # Record the network/HTTP error reason but never the key.
        row.update({
            "status": "error", "http_status": http_status,
            "decision_status": decision,
            "usable_for_research": usable,
            "usable_reason": _reason(decision, None),
            "notes": (response.error or "request failed") + "; "
                       + spec.notes,
        })
        return row

    rows = _extract_data_array(response.payload)
    if not rows:
        # Treat empty array on a 200 OK as INCONCLUSIVE — could be a
        # subscription gating issue rather than absence of data.
        row = _empty_row(spec, key_present=True)
        row.update({
            "status": "ok", "http_status": http_status,
            "decision_status": DECISION_INCONCLUSIVE,
            "usable_for_research": DECISION_INCONCLUSIVE,
            "usable_reason": "empty data array on 200 OK — possibly "
                                "tier-gated or wrong symbol",
        })
        return row

    timestamps = _extract_timestamps_ms(rows)
    if not timestamps:
        row = _empty_row(spec, key_present=True)
        row.update({
            "status": "ok", "http_status": http_status,
            "row_count": len(rows),
            "decision_status": DECISION_INCONCLUSIVE,
            "usable_for_research": DECISION_INCONCLUSIVE,
            "usable_reason": "data shape unrecognised — no timestamp "
                                "field found",
        })
        return row

    start_ms = min(timestamps)
    end_ms = max(timestamps)
    cov = _coverage_days(start_ms, end_ms)
    decision, usable = classify_decision(
        cov, api_key_present=True, http_status=http_status,
    )
    row = _empty_row(spec, key_present=True)
    row.update({
        "status": "ok", "http_status": http_status,
        "actual_start": _ts_to_iso(start_ms),
        "actual_end": _ts_to_iso(end_ms),
        "row_count": int(len(rows)),
        "coverage_days": cov,
        "decision_status": decision,
        "usable_for_research": usable,
        "usable_reason": _reason(decision, cov),
    })
    return row


# ---------------------------------------------------------------------------
# Top-level orchestrator.
# ---------------------------------------------------------------------------
def run_audit(
    save: bool = True,
    output_path: Optional[Path] = None,
    rate_delay_s: float = DEFAULT_RATE_DELAY_S,
    http_get: Callable[[str, str], _Response] = _http_get_json,
) -> pd.DataFrame:
    """Run every probe, write the CSV, return the DataFrame.

    The API key is read once at the top of this function and passed
    to `run_probe` ONLY as a positional argument. It is never written
    to the DataFrame, the CSV, the logger, or stdout."""
    utils.assert_paper_only()
    api_key = _read_api_key()
    key_present = bool(api_key.strip())
    if not key_present:
        logger.warning(
            "coinglass keyed audit: %s env var not set; producing "
            "AUTH_FAILED rows", _API_KEY_ENV_NAME,
        )

    rows: List[Dict[str, Any]] = []
    probes = _make_probes()
    for i, spec in enumerate(probes):
        try:
            row = run_probe(spec, api_key, http_get=http_get)
        except Exception as exc:  # noqa: BLE001 — fail-soft policy
            logger.warning("probe %s raised %s",
                              spec.dataset, type(exc).__name__)
            row = _empty_row(spec, key_present=key_present)
            row.update({
                "status": "error",
                "decision_status": DECISION_INCONCLUSIVE,
                "usable_for_research": DECISION_INCONCLUSIVE,
                "usable_reason": f"probe raised {type(exc).__name__}",
            })
        rows.append(row)
        if i + 1 < len(probes) and rate_delay_s > 0:
            time.sleep(rate_delay_s)

    df = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    # Defence in depth: never persist the API key into the CSV under
    # any column name, even by accident.
    _assert_key_not_in_frame(df, api_key)
    if save:
        out = (output_path or
                config.RESULTS_DIR / "coinglass_keyed_data_audit.csv")
        utils.write_df(df, out)
    return df


def _assert_key_not_in_frame(df: pd.DataFrame, api_key: str) -> None:
    """Last-line-of-defence check: scan every cell of the audit
    DataFrame for the key value. Raises if found — that should never
    happen; the modules above never write the key in any path. Short
    placeholder values (< 8 chars) are skipped to avoid false positives
    in tests that pass synthetic short keys."""
    if not api_key or len(api_key) < 8:
        return
    for col in df.columns:
        s = df[col]
        if s.dtype == object:
            for v in s.dropna().tolist():
                if isinstance(v, str) and api_key in v:
                    raise RuntimeError(
                        "API key value leaked into audit DataFrame "
                        f"column {col!r} — refusing to persist",
                    )


# ---------------------------------------------------------------------------
# Decision summariser — used by the CLI and the report.
# ---------------------------------------------------------------------------
def _useful_field_class(dataset: str) -> str:
    """Group a dataset name into the spec's "useful field" axes."""
    if "open_interest" in dataset:
        return "open_interest"
    if "liquidation" in dataset:
        return "liquidations"
    if "long_short" in dataset:
        return "long_short_ratios"
    if "funding" in dataset:
        return "funding"
    if "basis" in dataset or "premium" in dataset:
        return "basis_or_premium"
    return "other"


PRIORITY_FIELD_CLASSES: Tuple[str, ...] = (
    "open_interest", "liquidations", "long_short_ratios",
    "funding", "basis_or_premium",
)


def summarise(df: pd.DataFrame) -> Dict[str, Any]:
    """Return per-decision counts plus a 'fields_passed' map keyed by
    (asset, field_class) -> True if any probe under that class yielded
    PASS for that asset. The strategy decision rule on the user's spec
    is "at least 2 useful fields PASS for BTC and ETH"."""
    if df.empty:
        return {
            "n": 0, "api_key_present": False,
            "pass": 0, "warning": 0, "fail": 0,
            "inconclusive": 0, "auth_failed": 0,
            "rate_limited": 0, "endpoint_not_available": 0,
            "btc_pass_field_classes": [],
            "eth_pass_field_classes": [],
            "btc_pass_count": 0, "eth_pass_count": 0,
            "go": False, "verdict": "INCONCLUSIVE",
        }

    counts = df["decision_status"].value_counts().to_dict()
    api_key_present = bool(df["api_key_present"].any())

    pass_by_asset: Dict[str, set] = {"BTC": set(), "ETH": set()}
    for _, row in df.iterrows():
        if row["decision_status"] == DECISION_PASS:
            field = _useful_field_class(str(row["dataset"]))
            asset = str(row["asset"]).upper()
            if asset in pass_by_asset and field in PRIORITY_FIELD_CLASSES:
                pass_by_asset[asset].add(field)

    btc_pass = sorted(pass_by_asset["BTC"])
    eth_pass = sorted(pass_by_asset["ETH"])
    go = (len(btc_pass) >= 2 and len(eth_pass) >= 2)
    verdict = ("GO" if go else
                ("INCONCLUSIVE" if not api_key_present
                  else "NO_GO"))

    return {
        "n": int(len(df)),
        "api_key_present": api_key_present,
        "pass": int(counts.get(DECISION_PASS, 0)),
        "warning": int(counts.get(DECISION_WARNING, 0)),
        "fail": int(counts.get(DECISION_FAIL, 0)),
        "inconclusive": int(counts.get(DECISION_INCONCLUSIVE, 0)),
        "auth_failed": int(counts.get(DECISION_AUTH_FAILED, 0)),
        "rate_limited": int(counts.get(DECISION_RATE_LIMITED, 0)),
        "endpoint_not_available": int(counts.get(DECISION_ENDPOINT_NA, 0)),
        "btc_pass_field_classes": btc_pass,
        "eth_pass_field_classes": eth_pass,
        "btc_pass_count": len(btc_pass),
        "eth_pass_count": len(eth_pass),
        "go": go,
        "verdict": verdict,
    }
