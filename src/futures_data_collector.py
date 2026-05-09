"""
Binance USDT-margined futures public data collector.

Public endpoints only. NO API keys. NO private endpoints. NO order
placement. NO Kraken integration.

Endpoints used:
  * `https://fapi.binance.com/fapi/v1/fundingRate`
        — funding rate history. Paginates ~4 years of data via
        `startTime` + `limit=1000`.
  * `https://fapi.binance.com/futures/data/openInterestHist`
        — open interest history. **Capped at ~30 days** by Binance,
        regardless of `startTime`. Same family of limitation as
        Kraken's spot OHLC cap (which v1 routed around by switching to
        Binance). For OI there is no obvious public alternative — the
        coverage audit surfaces this honestly so any OI-derived
        research can be flagged INCONCLUSIVE on data.

Symbol handling:
  Binance Futures uses concatenated symbols (`BTCUSDT`, not `BTC/USDT`).
  This module accepts either form externally and normalises internally.

Output:
  * `data/futures/funding_rates/{SYMBOL}.csv`   columns: timestamp, symbol,
        funding_rate, funding_time, source
  * `data/futures/open_interest/{SYMBOL}.csv`   columns: timestamp, symbol,
        open_interest, open_interest_value, source
  * `results/futures_data_coverage.csv`         columns: symbol, dataset,
        actual_start, actual_end, row_count, coverage_days,
        enough_for_research, missing_reason, notes

This module never imports nor calls anything that requires authenticated
exchange access. The HTTP client is `requests` (already a transitive
dep of `ccxt`); no new top-level dependency added.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests

from . import config, utils

logger = utils.get_logger("cte.futures_data")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FUNDING_RATE_ENDPOINT = "https://fapi.binance.com/fapi/v1/fundingRate"
OPEN_INTEREST_HIST_ENDPOINT = "https://fapi.binance.com/futures/data/openInterestHist"
DEFAULT_HTTP_TIMEOUT = 20  # seconds
DEFAULT_SLEEP_SECONDS = 0.25  # respect the 2400/min weight budget by spacing calls

# Binance cap on OI history endpoint (empirically + documented).
OI_HISTORY_PUBLIC_DAYS_CAP = 30
# Minimum coverage for a (funding, OI) feature set to be useful in
# walk-forward at 90/30 windows.
MIN_DAYS_FOR_RESEARCH = 120

DEFAULT_FUTURES_SYMBOLS: Tuple[str, ...] = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT",
)


@dataclass(frozen=True)
class FuturesDownloadConfig:
    days: int = 1460
    refresh: bool = False
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS
    http_timeout: int = DEFAULT_HTTP_TIMEOUT


# ---------------------------------------------------------------------------
# Symbol normalisation
# ---------------------------------------------------------------------------
def _no_slash(symbol: str) -> str:
    """`BTC/USDT` -> `BTCUSDT`."""
    return symbol.replace("/", "")


def _futures_csv_path(kind: str, symbol: str) -> Path:
    if kind not in ("funding_rates", "open_interest"):
        raise ValueError(f"unknown kind {kind!r}")
    base = config.REPO_ROOT / "data" / "futures" / kind
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{_no_slash(symbol)}.csv"


# ---------------------------------------------------------------------------
# Funding rate
# ---------------------------------------------------------------------------
def _http_get(url: str, params: Dict[str, Any], timeout: int) -> List[Dict[str, Any]]:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError(f"unexpected response shape from {url}: {type(data)}")
    return data


def fetch_funding_rate_history(
    symbol: str, days: int = 1460,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    http_timeout: int = DEFAULT_HTTP_TIMEOUT,
) -> pd.DataFrame:
    """Walk forward through `fundingRate` in 1000-row batches starting from
    `now - days`. Sorts ascending, dedupes on `funding_time`."""
    sym = _no_slash(symbol)
    end_ms = int(pd.Timestamp.utcnow().value // 10**6)
    start_ms = int((pd.Timestamp.utcnow() - pd.Timedelta(days=days)).value // 10**6)
    cursor = start_ms
    rows: List[Dict[str, Any]] = []
    iter_count = 0
    prev_first = None
    while cursor < end_ms:
        iter_count += 1
        try:
            batch = _http_get(
                FUNDING_RATE_ENDPOINT,
                {"symbol": sym, "startTime": cursor, "limit": 1000},
                timeout=http_timeout,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("funding fetch error (%s, iter=%d): %s",
                           sym, iter_count, e)
            break
        if not batch:
            break
        first_ts = int(batch[0]["fundingTime"])
        if prev_first is not None and first_ts <= prev_first:
            # Stuck pagination — abort defensively.
            logger.warning("funding stuck pagination on %s after %d iters",
                           sym, iter_count)
            break
        prev_first = first_ts
        rows.extend(batch)
        last_ts = int(batch[-1]["fundingTime"])
        cursor = last_ts + 1
        if len(batch) < 1000:
            break  # no more data
        time.sleep(max(0.0, sleep_seconds))
    if not rows:
        return pd.DataFrame(columns=["timestamp", "symbol", "funding_rate",
                                     "funding_time", "source"])
    df = pd.DataFrame(rows)
    df["funding_time"] = df["fundingTime"].astype("int64")
    df["timestamp"] = df["funding_time"]
    df["symbol"] = df["symbol"].astype(str)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df["source"] = "binance_fapi_v1_funding_rate"
    df = df[["timestamp", "symbol", "funding_rate", "funding_time", "source"]]
    df = (df.drop_duplicates(subset=["funding_time"])
            .sort_values("funding_time")
            .reset_index(drop=True))
    return df


# ---------------------------------------------------------------------------
# Open interest history (CAPPED ~30 days by Binance public API)
# ---------------------------------------------------------------------------
def fetch_open_interest_history(
    symbol: str, days: int = 30, period: str = "1d",
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    http_timeout: int = DEFAULT_HTTP_TIMEOUT,
) -> pd.DataFrame:
    """Walk forward through `openInterestHist`. The public endpoint
    rejects `startTime` older than ~30 days with HTTP 400, so the first
    call goes WITHOUT a `startTime` (Binance returns the most recent
    `limit` rows). Subsequent calls advance from the last returned
    timestamp, which lets the collector adapt automatically if Binance
    ever extends the cap."""
    sym = _no_slash(symbol)
    end_ms = int(pd.Timestamp.utcnow().value // 10**6)
    cursor: Optional[int] = None  # first call: no startTime
    rows: List[Dict[str, Any]] = []
    iter_count = 0
    prev_first = None
    max_iters = max(2, int(days // OI_HISTORY_PUBLIC_DAYS_CAP) + 4)
    while iter_count < max_iters:
        iter_count += 1
        params: Dict[str, Any] = {"symbol": sym, "period": period, "limit": 500}
        if cursor is not None:
            params["startTime"] = cursor
        try:
            batch = _http_get(
                OPEN_INTEREST_HIST_ENDPOINT, params, timeout=http_timeout,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("OI fetch error (%s, iter=%d): %s",
                           sym, iter_count, e)
            break
        if not batch:
            break
        first_ts = int(batch[0]["timestamp"])
        if prev_first is not None and first_ts <= prev_first:
            logger.warning("OI stuck pagination on %s after %d iters",
                           sym, iter_count)
            break
        prev_first = first_ts
        rows.extend(batch)
        last_ts = int(batch[-1]["timestamp"])
        cursor = last_ts + 1
        if len(batch) < 500:
            break
        time.sleep(max(0.0, sleep_seconds))
    if not rows:
        return pd.DataFrame(columns=["timestamp", "symbol", "open_interest",
                                     "open_interest_value", "source"])
    df = pd.DataFrame(rows)
    df["timestamp"] = df["timestamp"].astype("int64")
    df["symbol"] = df["symbol"].astype(str)
    df["open_interest"] = df["sumOpenInterest"].astype(float)
    df["open_interest_value"] = df["sumOpenInterestValue"].astype(float)
    df["source"] = "binance_futures_data_oi_hist"
    df = df[["timestamp", "symbol", "open_interest",
             "open_interest_value", "source"]]
    df = (df.drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True))
    return df


# ---------------------------------------------------------------------------
# Gap validation
# ---------------------------------------------------------------------------
def validate_funding_gaps(df: pd.DataFrame,
                          expected_interval_hours: float = 8.0) -> Dict[str, Any]:
    """Funding rate fires every 8h on Binance USDT-perp. Detect gaps."""
    if df is None or df.empty or "funding_time" not in df.columns:
        return {"row_count": 0, "gap_count": 0, "largest_gap_hours": 0.0}
    ts = df["funding_time"].astype("int64").to_numpy()
    if len(ts) < 2:
        return {"row_count": int(len(ts)), "gap_count": 0,
                "largest_gap_hours": 0.0}
    diffs = ts[1:] - ts[:-1]
    expected_ms = int(expected_interval_hours * 3600 * 1000)
    gap_mask = diffs > int(expected_ms * 1.5)
    largest_gap_ms = int(diffs.max())
    return {
        "row_count": int(len(ts)),
        "gap_count": int(gap_mask.sum()),
        "largest_gap_hours": round(largest_gap_ms / 3600 / 1000, 2),
    }


def validate_oi_gaps(df: pd.DataFrame,
                     period: str = "1d") -> Dict[str, Any]:
    if df is None or df.empty or "timestamp" not in df.columns:
        return {"row_count": 0, "gap_count": 0, "largest_gap_days": 0.0}
    ts = df["timestamp"].astype("int64").to_numpy()
    if len(ts) < 2:
        return {"row_count": int(len(ts)), "gap_count": 0,
                "largest_gap_days": 0.0}
    diffs = ts[1:] - ts[:-1]
    period_ms = {"5m": 300_000, "15m": 900_000, "30m": 1_800_000,
                 "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
                 "6h": 21_600_000, "12h": 43_200_000, "1d": 86_400_000}.get(period, 86_400_000)
    gap_mask = diffs > int(period_ms * 1.5)
    largest_gap_ms = int(diffs.max())
    return {
        "row_count": int(len(ts)),
        "gap_count": int(gap_mask.sum()),
        "largest_gap_days": round(largest_gap_ms / 86_400 / 1000, 2),
    }


# ---------------------------------------------------------------------------
# Loaders for downstream code
# ---------------------------------------------------------------------------
def load_funding_rate(symbol: str) -> pd.DataFrame:
    p = _futures_csv_path("funding_rates", symbol)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=["timestamp", "symbol", "funding_rate",
                                     "funding_time", "source"])
    return pd.read_csv(p)


def load_open_interest(symbol: str) -> pd.DataFrame:
    p = _futures_csv_path("open_interest", symbol)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=["timestamp", "symbol", "open_interest",
                                     "open_interest_value", "source"])
    return pd.read_csv(p)


def available_futures_symbols(
    symbols: Sequence[str] = DEFAULT_FUTURES_SYMBOLS,
) -> List[str]:
    """Return symbols where BOTH funding and OI CSVs exist and are non-empty."""
    out: List[str] = []
    for s in symbols:
        f = _futures_csv_path("funding_rates", s)
        o = _futures_csv_path("open_interest", s)
        if f.exists() and f.stat().st_size > 0 \
                and o.exists() and o.stat().st_size > 0:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Public API: download + audit
# ---------------------------------------------------------------------------
def download_futures_data(
    symbols: Sequence[str] = DEFAULT_FUTURES_SYMBOLS,
    days: int = 1460,
    refresh: bool = False,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    http_timeout: int = DEFAULT_HTTP_TIMEOUT,
) -> Dict[str, Any]:
    """Download funding + OI for every symbol. Failures are logged and
    skipped; the run never aborts. Returns a dict of paths + a list of
    symbols where either dataset failed. Coverage is also persisted to
    `results/futures_data_coverage.csv`."""
    utils.assert_paper_only()
    funding_paths: Dict[str, Path] = {}
    oi_paths: Dict[str, Path] = {}
    missing: List[str] = []
    for sym in symbols:
        sym_clean = _no_slash(sym)
        # Funding
        f_path = _futures_csv_path("funding_rates", sym_clean)
        if f_path.exists() and not refresh:
            funding_paths[sym_clean] = f_path
            logger.info("funding cache hit: %s", sym_clean)
        else:
            try:
                df_f = fetch_funding_rate_history(
                    sym_clean, days=days,
                    sleep_seconds=sleep_seconds, http_timeout=http_timeout,
                )
                if df_f.empty:
                    logger.warning("funding empty for %s", sym_clean)
                    missing.append(sym_clean)
                else:
                    utils.write_df(df_f, f_path)
                    funding_paths[sym_clean] = f_path
            except Exception as e:  # noqa: BLE001
                logger.warning("funding download failed (%s): %s", sym_clean, e)
                missing.append(sym_clean)
        # OI
        o_path = _futures_csv_path("open_interest", sym_clean)
        if o_path.exists() and not refresh:
            oi_paths[sym_clean] = o_path
            logger.info("OI cache hit: %s", sym_clean)
        else:
            try:
                df_o = fetch_open_interest_history(
                    sym_clean, days=days, period="1d",
                    sleep_seconds=sleep_seconds, http_timeout=http_timeout,
                )
                if df_o.empty:
                    logger.warning("OI empty for %s", sym_clean)
                else:
                    utils.write_df(df_o, o_path)
                    oi_paths[sym_clean] = o_path
            except Exception as e:  # noqa: BLE001
                logger.warning("OI download failed (%s): %s", sym_clean, e)

    coverage = audit_futures_coverage(symbols=symbols, save=True)
    return {
        "funding_paths": funding_paths,
        "oi_paths": oi_paths,
        "missing": missing,
        "coverage_df": coverage,
    }


def audit_futures_coverage(
    symbols: Sequence[str] = DEFAULT_FUTURES_SYMBOLS,
    save: bool = True,
) -> pd.DataFrame:
    """Build `results/futures_data_coverage.csv` with the same columns the
    spec requires. Each (symbol, dataset) pair is one row."""
    rows: List[Dict[str, Any]] = []
    for sym in symbols:
        sym_clean = _no_slash(sym)
        for kind, validate, key in (
            ("funding_rates", validate_funding_gaps, "funding_time"),
            ("open_interest", validate_oi_gaps, "timestamp"),
        ):
            p = _futures_csv_path(kind, sym_clean)
            if not p.exists() or p.stat().st_size == 0:
                rows.append({
                    "symbol": sym_clean, "dataset": kind,
                    "actual_start": None, "actual_end": None,
                    "row_count": 0, "coverage_days": 0.0,
                    "enough_for_research": False,
                    "missing_reason": "no_csv",
                    "notes": "",
                })
                continue
            df = pd.read_csv(p)
            if df.empty or key not in df.columns:
                rows.append({
                    "symbol": sym_clean, "dataset": kind,
                    "actual_start": None, "actual_end": None,
                    "row_count": int(len(df)), "coverage_days": 0.0,
                    "enough_for_research": False,
                    "missing_reason": "empty_or_malformed",
                    "notes": "",
                })
                continue
            start_ms = int(df[key].iloc[0])
            end_ms = int(df[key].iloc[-1])
            cov_days = (end_ms - start_ms) / 86_400_000
            info = validate(df)
            enough = cov_days >= MIN_DAYS_FOR_RESEARCH
            note_parts = []
            if kind == "open_interest" and cov_days < OI_HISTORY_PUBLIC_DAYS_CAP + 5:
                note_parts.append(
                    f"OI capped near Binance public limit ~"
                    f"{OI_HISTORY_PUBLIC_DAYS_CAP}d"
                )
            if kind == "funding_rates" and info.get("gap_count", 0) > 0:
                note_parts.append(
                    f"{info['gap_count']} funding gap(s); largest "
                    f"{info.get('largest_gap_hours', 0):.1f}h"
                )
            if kind == "open_interest" and info.get("gap_count", 0) > 0:
                note_parts.append(
                    f"{info['gap_count']} OI gap(s); largest "
                    f"{info.get('largest_gap_days', 0):.1f}d"
                )
            rows.append({
                "symbol": sym_clean, "dataset": kind,
                "actual_start": str(pd.to_datetime(start_ms, unit="ms", utc=True)),
                "actual_end": str(pd.to_datetime(end_ms, unit="ms", utc=True)),
                "row_count": int(len(df)),
                "coverage_days": round(cov_days, 2),
                "enough_for_research": bool(enough),
                "missing_reason": "" if enough else (
                    "insufficient_history_for_walk_forward"
                ),
                "notes": "; ".join(note_parts) if note_parts else "ok",
            })
    out = pd.DataFrame(rows)
    if save:
        utils.write_df(out, config.RESULTS_DIR / "futures_data_coverage.csv")
    return out
