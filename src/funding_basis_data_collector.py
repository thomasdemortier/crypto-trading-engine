"""
Funding-rate + futures-basis data collector.

Public endpoints only. No API keys are read or required. No private
endpoints. No order placement. Read-only HTTP.

Data fetched (per asset):
    * Binance Futures funding rate history (paginated by `startTime`).
    * Binance Futures mark-price daily klines.
    * Binance Futures index-price daily klines.
        (mark - index) / index = futures-spot basis proxy.
    * Bybit linear funding rate history (paginated by `startTime`/`endTime`).
    * Deribit perpetual funding rate history (range-windowed).

Anchor calendar:
    The signals module aligns everything to the local spot OHLCV
    cache (which begins 2022-04-01 for BTC/ETH on this project). The
    collector therefore caps pagination at `since_ms = 2022-04-01` —
    no use in pulling 2019 funding when the strategy can't trade on it.

Outputs (gitignored — see `.gitignore: data/positioning/**`):
    data/positioning/funding_basis/binance_funding_<SYM>.csv
    data/positioning/funding_basis/binance_mark_klines_<SYM>.csv
    data/positioning/funding_basis/binance_index_klines_<SYM>.csv
    data/positioning/funding_basis/bybit_funding_<SYM>.csv
    data/positioning/funding_basis/deribit_funding_<INSTR>.csv
    results/funding_basis_data_coverage.csv

Coverage thresholds (locked here, never tuned):
    PASS         coverage >= 1460 days.
    WARNING      365 - 1459 days.
    FAIL         < 365 days OR endpoint failed.
    INCONCLUSIVE the endpoint changed shape or is unreachable.
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

logger = utils.get_logger("cte.funding_basis_data_collector")


# ---------------------------------------------------------------------------
# Public constants — locked thresholds + paths.
# ---------------------------------------------------------------------------
COVERAGE_PASS_DAYS = 1460
COVERAGE_WARNING_DAYS = 365

# Spot anchor — BTC/ETH 1d cache begins 2022-04-01 on this project.
ANCHOR_START_MS = int(datetime(2022, 4, 1, tzinfo=timezone.utc
                                 ).timestamp() * 1000)

DEFAULT_TIMEOUT = 12.0
DEFAULT_USER_AGENT = "cte-funding-basis-collector/0.1"
DEFAULT_RATE_DELAY_S = 0.10  # gentle pacing between paginated calls

# Directory layout. The directory is created lazily.
POSITIONING_DIR: Path = config.REPO_ROOT / "data" / "positioning" / "funding_basis"
COVERAGE_PATH: Path = config.RESULTS_DIR / "funding_basis_data_coverage.csv"

# Asset universe: BTC + ETH only. SOL omitted because the audit did not
# verify SOL funding/basis depth on this branch — adding it later requires
# a fresh probe + a re-run of `download_funding_basis_data`.
DEFAULT_ASSETS: Tuple[str, ...] = ("BTC/USDT", "ETH/USDT")

# Per-exchange symbol mapping — kept here so the collector module is the
# single source of truth.
BINANCE_SYMBOL: Dict[str, str] = {
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
    "SOL/USDT": "SOLUSDT",
}
BYBIT_SYMBOL: Dict[str, str] = {
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
    "SOL/USDT": "SOLUSDT",
}
DERIBIT_INSTRUMENT: Dict[str, str] = {
    "BTC/USDT": "BTC-PERPETUAL",
    "ETH/USDT": "ETH-PERPETUAL",
}


# ---------------------------------------------------------------------------
# Coverage classification (matches the spec — never tuned).
# ---------------------------------------------------------------------------
def classify_coverage(coverage_days: Optional[float], status: str = "ok",
                       ) -> str:
    if status != "ok":
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
# Stdlib HTTP helper — fail-soft. Never raises.
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
        return _Response(False, None, None,
                          f"url_error: {exc.reason}")
    except (TimeoutError, OSError) as exc:
        return _Response(False, None, None, f"network_error: {exc}")
    except Exception as exc:  # noqa: BLE001 — fail-soft policy
        return _Response(False, None, None,
                          f"unexpected: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Date / dataframe helpers
# ---------------------------------------------------------------------------
def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_dt(ts_ms: Any) -> Optional[pd.Timestamp]:
    try:
        return pd.to_datetime(int(ts_ms), unit="ms", utc=True)
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


def _normalise_funding_frame(rows: List[Dict[str, Any]],
                              source: str, symbol_or_instr: str
                              ) -> pd.DataFrame:
    """Standard schema: timestamp (ms), datetime (UTC), funding_rate (float),
    source, symbol_or_instrument. Sorted, dedup'd by timestamp."""
    if not rows:
        return pd.DataFrame(columns=[
            "timestamp", "datetime", "funding_rate",
            "source", "symbol_or_instrument",
        ])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_numeric(df["timestamp"]).astype("int64")
    df["funding_rate"] = pd.to_numeric(df["funding_rate"]).astype("float64")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["source"] = source
    df["symbol_or_instrument"] = symbol_or_instr
    df = (df.drop_duplicates("timestamp")
              .sort_values("timestamp").reset_index(drop=True))
    return df[["timestamp", "datetime", "funding_rate",
                "source", "symbol_or_instrument"]]


def _normalise_klines_frame(rows: List[List[Any]], source: str,
                              symbol: str, kind: str) -> pd.DataFrame:
    """Standard kline schema: timestamp (ms), datetime, open, high, low,
    close, volume, source, symbol, kind."""
    if not rows:
        return pd.DataFrame(columns=[
            "timestamp", "datetime", "open", "high", "low", "close",
            "volume", "source", "symbol", "kind",
        ])
    df = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "qa_volume", "n_trades", "tb_base", "tb_quote",
        "ignore",
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"]).astype("int64")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c]).astype("float64")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["source"] = source
    df["symbol"] = symbol
    df["kind"] = kind
    return df[["timestamp", "datetime", "open", "high", "low", "close",
                "volume", "source", "symbol", "kind"]
              ].drop_duplicates("timestamp").sort_values(
                  "timestamp").reset_index(drop=True)


def _gap_count(df: pd.DataFrame, expected_step_ms: int,
                tolerance_ms: int = 0) -> int:
    """Count timestamp gaps where step > expected_step_ms + tolerance."""
    if len(df) < 2 or "timestamp" not in df.columns:
        return 0
    diffs = df["timestamp"].astype("int64").diff().dropna()
    threshold = expected_step_ms + tolerance_ms
    return int((diffs > threshold).sum())


# ---------------------------------------------------------------------------
# Per-source paginated downloaders. Each returns a tidy DataFrame and the
# pagination metadata used by the coverage row.
# ---------------------------------------------------------------------------
@dataclass
class _DownloadResult:
    df: pd.DataFrame
    pages: int
    error: Optional[str]
    last_http_status: Optional[int]


def _download_binance_funding(
    asset: str, *, since_ms: int = ANCHOR_START_MS,
    page_limit: int = 1000, max_pages: int = 30,
    rate_delay_s: float = DEFAULT_RATE_DELAY_S,
    http_get: Callable[[str], _Response] = _http_get_json,
) -> _DownloadResult:
    """Binance returns up to 1000 rows per call ascending from `startTime`.
    Walk forward until the page is short or the cursor stops moving."""
    sym = BINANCE_SYMBOL.get(asset)
    if sym is None:
        return _DownloadResult(pd.DataFrame(), 0,
                                f"binance: no mapping for {asset}", None)
    cursor = int(since_ms)
    pages = 0
    last_status: Optional[int] = None
    seen: List[Dict[str, Any]] = []
    last_ts: Optional[int] = None
    for _ in range(max_pages):
        url = ("https://fapi.binance.com/fapi/v1/fundingRate"
               f"?symbol={sym}&startTime={cursor}&limit={page_limit}")
        r = http_get(url)
        last_status = r.status_code
        if not r.ok or not isinstance(r.payload, list) or not r.payload:
            break
        page = r.payload
        pages += 1
        for row in page:
            seen.append({
                "timestamp": int(row["fundingTime"]),
                "funding_rate": float(row["fundingRate"]),
            })
        new_last = int(page[-1]["fundingTime"])
        if last_ts is not None and new_last <= last_ts:
            break  # no progress — stop
        last_ts = new_last
        if len(page) < page_limit:
            break
        cursor = new_last + 1  # advance past the last seen row
        if rate_delay_s > 0:
            time.sleep(rate_delay_s)
    df = _normalise_funding_frame(seen, "binance_futures", sym)
    return _DownloadResult(df, pages, None, last_status)


def _download_binance_klines(
    asset: str, kind: str, *, since_ms: int = ANCHOR_START_MS,
    interval: str = "1d", page_limit: int = 1500,
    rate_delay_s: float = DEFAULT_RATE_DELAY_S,
    http_get: Callable[[str], _Response] = _http_get_json,
) -> _DownloadResult:
    """`kind` is one of {'mark', 'index'}. Binance caps mark/index klines
    at 1500 rows per request — single call covers >= 4 years for 1d."""
    sym = BINANCE_SYMBOL.get(asset)
    if sym is None:
        return _DownloadResult(pd.DataFrame(), 0,
                                f"binance: no mapping for {asset}", None)
    if kind == "mark":
        url = ("https://fapi.binance.com/fapi/v1/markPriceKlines"
               f"?symbol={sym}&interval={interval}"
               f"&startTime={since_ms}&limit={page_limit}")
    elif kind == "index":
        # Binance index-price klines use `pair`, not `symbol`.
        url = ("https://fapi.binance.com/fapi/v1/indexPriceKlines"
               f"?pair={sym}&interval={interval}"
               f"&startTime={since_ms}&limit={page_limit}")
    else:
        return _DownloadResult(pd.DataFrame(), 0,
                                f"unknown kline kind: {kind}", None)
    r = http_get(url)
    if not r.ok or not isinstance(r.payload, list) or not r.payload:
        return _DownloadResult(pd.DataFrame(), 0,
                                r.error or "empty payload", r.status_code)
    df = _normalise_klines_frame(r.payload, "binance_futures", sym, kind)
    return _DownloadResult(df, 1, None, r.status_code)


def _download_bybit_funding(
    asset: str, *, since_ms: int = ANCHOR_START_MS,
    page_limit: int = 200, max_pages: int = 60,
    rate_delay_s: float = DEFAULT_RATE_DELAY_S,
    http_get: Callable[[str], _Response] = _http_get_json,
) -> _DownloadResult:
    """Bybit's `/v5/market/funding/history` returns DESCENDING (most
    recent first) and is paginated by moving `endTime` backwards."""
    sym = BYBIT_SYMBOL.get(asset)
    if sym is None:
        return _DownloadResult(pd.DataFrame(), 0,
                                f"bybit: no mapping for {asset}", None)
    end_ms = _now_ms()
    pages = 0
    last_status: Optional[int] = None
    seen: List[Dict[str, Any]] = []
    earliest_seen: Optional[int] = None
    for _ in range(max_pages):
        url = ("https://api.bybit.com/v5/market/funding/history"
               f"?category=linear&symbol={sym}"
               f"&startTime={since_ms}&endTime={end_ms}"
               f"&limit={page_limit}")
        r = http_get(url)
        last_status = r.status_code
        if not r.ok:
            break
        rows = (r.payload or {}).get("result", {}).get("list", []) or []
        if not rows:
            break
        pages += 1
        for row in rows:
            seen.append({
                "timestamp": int(row["fundingRateTimestamp"]),
                "funding_rate": float(row["fundingRate"]),
            })
        ts_in_page = sorted(int(x["fundingRateTimestamp"]) for x in rows)
        page_oldest = ts_in_page[0]
        if earliest_seen is not None and page_oldest >= earliest_seen:
            break  # cursor stalled
        earliest_seen = page_oldest
        if page_oldest <= since_ms:
            break
        end_ms = page_oldest - 1
        if rate_delay_s > 0:
            time.sleep(rate_delay_s)
    df = _normalise_funding_frame(seen, "bybit", sym)
    return _DownloadResult(df, pages, None, last_status)


def _download_deribit_funding(
    asset: str, *, since_ms: int = ANCHOR_START_MS,
    window_days: int = 14, max_windows: int = 200,
    rate_delay_s: float = DEFAULT_RATE_DELAY_S,
    http_get: Callable[[str], _Response] = _http_get_json,
) -> _DownloadResult:
    """Deribit `get_funding_rate_history` is range-windowed. Walk forward
    in `window_days` chunks. Per-window response is hourly so each
    chunk produces ~24 * window_days rows."""
    instr = DERIBIT_INSTRUMENT.get(asset)
    if instr is None:
        return _DownloadResult(pd.DataFrame(), 0,
                                f"deribit: no mapping for {asset}", None)
    cursor = int(since_ms)
    end_total = _now_ms()
    span_ms = window_days * 86_400_000
    pages = 0
    last_status: Optional[int] = None
    seen: List[Dict[str, Any]] = []
    last_cursor: Optional[int] = None
    for _ in range(max_windows):
        if cursor >= end_total:
            break
        win_end = min(cursor + span_ms, end_total)
        url = ("https://www.deribit.com/api/v2/public/"
               f"get_funding_rate_history"
               f"?instrument_name={instr}"
               f"&start_timestamp={cursor}&end_timestamp={win_end}")
        r = http_get(url)
        last_status = r.status_code
        if not r.ok:
            break
        rows = (r.payload or {}).get("result", []) or []
        pages += 1
        for row in rows:
            # Deribit reports an `interest_8h` field — interpret as the
            # 8-hour realised funding rate (matching the convention used
            # by Binance/Bybit `funding_rate`).
            ts = int(row.get("timestamp") or 0)
            rate = row.get("interest_8h")
            if rate is None:
                rate = row.get("interest_1h")
            if rate is None or ts <= 0:
                continue
            seen.append({
                "timestamp": ts,
                "funding_rate": float(rate),
            })
        if last_cursor is not None and cursor <= last_cursor:
            break
        last_cursor = cursor
        cursor = win_end + 1
        if rate_delay_s > 0:
            time.sleep(rate_delay_s)
    df = _normalise_funding_frame(seen, "deribit", instr)
    return _DownloadResult(df, pages, None, last_status)


# ---------------------------------------------------------------------------
# Coverage row builder
# ---------------------------------------------------------------------------
COVERAGE_COLUMNS: List[str] = [
    "source", "asset", "symbol_or_instrument", "dataset",
    "status", "verdict", "row_count", "pages_walked",
    "actual_start", "actual_end", "coverage_days",
    "expected_step_ms", "gap_count", "granularity",
    "last_http_status", "csv_path", "notes",
]


def _coverage_row(*, source: str, asset: str, symbol_or_instr: str,
                   dataset: str, df: pd.DataFrame, dl: _DownloadResult,
                   expected_step_ms: int, granularity: str,
                   csv_path: Path, notes: str = "") -> Dict[str, Any]:
    if df.empty:
        verdict = classify_coverage(None,
                                       status="error" if dl.error else "ok")
        if verdict != "FAIL":
            verdict = "INCONCLUSIVE"
        return {
            "source": source, "asset": asset,
            "symbol_or_instrument": symbol_or_instr, "dataset": dataset,
            "status": "error" if dl.error else "ok",
            "verdict": verdict,
            "row_count": 0, "pages_walked": dl.pages,
            "actual_start": None, "actual_end": None,
            "coverage_days": None, "expected_step_ms": expected_step_ms,
            "gap_count": None, "granularity": granularity,
            "last_http_status": dl.last_http_status,
            "csv_path": str(csv_path), "notes": dl.error or notes,
        }
    start = int(df["timestamp"].iloc[0])
    end = int(df["timestamp"].iloc[-1])
    cov = _coverage_days(start, end)
    gaps = _gap_count(df, expected_step_ms,
                      tolerance_ms=expected_step_ms // 4)
    return {
        "source": source, "asset": asset,
        "symbol_or_instrument": symbol_or_instr, "dataset": dataset,
        "status": "ok", "verdict": classify_coverage(cov),
        "row_count": int(len(df)), "pages_walked": dl.pages,
        "actual_start": (_safe_dt(start).isoformat()
                          if _safe_dt(start) is not None else None),
        "actual_end": (_safe_dt(end).isoformat()
                        if _safe_dt(end) is not None else None),
        "coverage_days": cov,
        "expected_step_ms": expected_step_ms,
        "gap_count": gaps, "granularity": granularity,
        "last_http_status": dl.last_http_status,
        "csv_path": str(csv_path), "notes": notes,
    }


# ---------------------------------------------------------------------------
# Top-level entry — orchestrate every download + write coverage CSV.
# ---------------------------------------------------------------------------
def download_all(
    assets: Sequence[str] = DEFAULT_ASSETS,
    since_ms: int = ANCHOR_START_MS,
    save: bool = True,
    output_dir: Optional[Path] = None,
    coverage_path: Optional[Path] = None,
    http_get: Callable[[str], _Response] = _http_get_json,
) -> pd.DataFrame:
    """Download every required dataset for every asset, write per-source
    CSVs, and emit `results/funding_basis_data_coverage.csv`. Always
    returns a coverage DataFrame, even on partial failure.

    Returns the coverage DataFrame. The per-asset funding/basis frames
    themselves are persisted to the positioning data directory only —
    they are NOT returned.
    """
    utils.assert_paper_only()
    out_dir = output_dir or POSITIONING_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    coverage_rows: List[Dict[str, Any]] = []
    for asset in assets:
        # Binance funding.
        try:
            dl = _download_binance_funding(asset, since_ms=since_ms,
                                              http_get=http_get)
        except Exception as exc:  # noqa: BLE001 — fail-soft
            dl = _DownloadResult(pd.DataFrame(), 0,
                                  f"raised: {type(exc).__name__}: {exc}",
                                  None)
        sym = BINANCE_SYMBOL.get(asset, asset)
        path = out_dir / f"binance_funding_{utils.safe_symbol(asset)}.csv"
        if save and not dl.df.empty:
            utils.write_df(dl.df, path)
        coverage_rows.append(_coverage_row(
            source="binance_futures", asset=asset, symbol_or_instr=sym,
            dataset="funding_rate_history", df=dl.df, dl=dl,
            expected_step_ms=8 * 60 * 60 * 1000, granularity="8h",
            csv_path=path,
            notes=("multi-page paginated via startTime; ascending"),
        ))

        # Binance mark klines (1d).
        try:
            dl = _download_binance_klines(asset, "mark", since_ms=since_ms,
                                              http_get=http_get)
        except Exception as exc:  # noqa: BLE001
            dl = _DownloadResult(pd.DataFrame(), 0,
                                  f"raised: {type(exc).__name__}: {exc}",
                                  None)
        path = out_dir / f"binance_mark_klines_{utils.safe_symbol(asset)}.csv"
        if save and not dl.df.empty:
            utils.write_df(dl.df, path)
        coverage_rows.append(_coverage_row(
            source="binance_futures", asset=asset, symbol_or_instr=sym,
            dataset="mark_price_klines_1d", df=dl.df, dl=dl,
            expected_step_ms=86_400_000, granularity="1d",
            csv_path=path, notes="single page; up to 1500 rows",
        ))

        # Binance index klines (1d).
        try:
            dl = _download_binance_klines(asset, "index", since_ms=since_ms,
                                              http_get=http_get)
        except Exception as exc:  # noqa: BLE001
            dl = _DownloadResult(pd.DataFrame(), 0,
                                  f"raised: {type(exc).__name__}: {exc}",
                                  None)
        path = out_dir / f"binance_index_klines_{utils.safe_symbol(asset)}.csv"
        if save and not dl.df.empty:
            utils.write_df(dl.df, path)
        coverage_rows.append(_coverage_row(
            source="binance_futures", asset=asset, symbol_or_instr=sym,
            dataset="index_price_klines_1d", df=dl.df, dl=dl,
            expected_step_ms=86_400_000, granularity="1d",
            csv_path=path, notes="single page; up to 1500 rows",
        ))

        # Bybit funding.
        try:
            dl = _download_bybit_funding(asset, since_ms=since_ms,
                                            http_get=http_get)
        except Exception as exc:  # noqa: BLE001
            dl = _DownloadResult(pd.DataFrame(), 0,
                                  f"raised: {type(exc).__name__}: {exc}",
                                  None)
        sym_b = BYBIT_SYMBOL.get(asset, asset)
        path = out_dir / f"bybit_funding_{utils.safe_symbol(asset)}.csv"
        if save and not dl.df.empty:
            utils.write_df(dl.df, path)
        coverage_rows.append(_coverage_row(
            source="bybit", asset=asset, symbol_or_instr=sym_b,
            dataset="funding_rate_history", df=dl.df, dl=dl,
            expected_step_ms=8 * 60 * 60 * 1000, granularity="8h",
            csv_path=path,
            notes="multi-page paginated via endTime; descending",
        ))

        # Deribit funding (BTC + ETH only).
        instr = DERIBIT_INSTRUMENT.get(asset)
        if instr is not None:
            try:
                dl = _download_deribit_funding(asset, since_ms=since_ms,
                                                  http_get=http_get)
            except Exception as exc:  # noqa: BLE001
                dl = _DownloadResult(pd.DataFrame(), 0,
                                      f"raised: {type(exc).__name__}: {exc}",
                                      None)
            path = out_dir / f"deribit_funding_{utils.safe_symbol(asset)}.csv"
            if save and not dl.df.empty:
                utils.write_df(dl.df, path)
            coverage_rows.append(_coverage_row(
                source="deribit", asset=asset, symbol_or_instr=instr,
                dataset="perpetual_funding_rate_history",
                df=dl.df, dl=dl,
                expected_step_ms=60 * 60 * 1000, granularity="~1h",
                csv_path=path,
                notes=("range-windowed; walked forward in 14d chunks"),
            ))

    cov_df = pd.DataFrame(coverage_rows, columns=COVERAGE_COLUMNS)
    if save:
        utils.write_df(cov_df, coverage_path or COVERAGE_PATH)
    return cov_df


# ---------------------------------------------------------------------------
# Convenience reader — used by the signals module to load the persisted CSVs.
# ---------------------------------------------------------------------------
def load_funding_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=[
            "timestamp", "datetime", "funding_rate",
            "source", "symbol_or_instrument",
        ])
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"]).astype("int64")
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True,
                                          errors="coerce")
    return df


def load_klines_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=[
            "timestamp", "datetime", "open", "high", "low", "close",
            "volume", "source", "symbol", "kind",
        ])
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"]).astype("int64")
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True,
                                          errors="coerce")
    return df
