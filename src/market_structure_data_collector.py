"""
Market-structure data collector.

Downloads + normalises the free public datasets that the audit
([`src.market_structure_data_audit`]) confirmed are usable for
≥4 years of daily research:

  * DefiLlama total TVL (all chains)
  * DefiLlama stablecoin supply (all)
  * Blockchain.com BTC market cap
  * Blockchain.com BTC hash rate
  * Blockchain.com BTC transactions / day
  * Existing Binance daily spot OHLCV (counted from disk; not re-downloaded)

Outputs are normalised to a fixed long-format schema:
    timestamp (ms), date (UTC midnight string), source, dataset, value

A single coverage row per dataset is written to
`results/market_structure_data_coverage.csv`.

Hard rules:
  * No API keys, no paid endpoints, no private endpoints.
  * No live trading, no broker code, no order code.
  * Failures are logged and skipped; the run never aborts.
  * Raw data is never silently forward-filled — gaps are recorded.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests

from . import config, utils

logger = utils.get_logger("cte.market_structure_collector")

DEFAULT_HTTP_TIMEOUT = 30
DEFAULT_SLEEP_SECONDS = 0.25
DEFAULT_LOOKBACK_DAYS = 1500
MIN_DAYS_FOR_RESEARCH = 4 * 365


# ---------------------------------------------------------------------------
# Schemas (locked here so tests can assert)
# ---------------------------------------------------------------------------
NORMALISED_COLUMNS: List[str] = [
    "timestamp", "date", "source", "dataset", "value",
]
COVERAGE_COLUMNS: List[str] = [
    "source", "dataset", "actual_start", "actual_end",
    "row_count", "coverage_days", "enough_for_research",
    "largest_gap_days", "missing_reason", "notes",
]


# Endpoint constants — exposed at module level so tests can monkeypatch.
DEFILLAMA_TOTAL_TVL_URL = "https://api.llama.fi/charts"
DEFILLAMA_STABLECOIN_URL = "https://stablecoins.llama.fi/stablecoincharts/all"
BLOCKCHAIN_COM_BASE = "https://api.blockchain.info/charts"

# Mapping: dataset_name -> (source, endpoint, output_filename, payload_path)
@dataclass(frozen=True)
class _DatasetSpec:
    source: str
    dataset: str
    endpoint: str
    output_csv: str
    payload_path: str   # e.g. "values" for BC.com or "" for top-level list


_DATASET_SPECS: Tuple[_DatasetSpec, ...] = (
    _DatasetSpec("defillama", "total_tvl_all_chains",
                 DEFILLAMA_TOTAL_TVL_URL,
                 "defillama_total_tvl.csv", "_list_top_level"),
    _DatasetSpec("defillama", "stablecoin_supply_total",
                 DEFILLAMA_STABLECOIN_URL,
                 "defillama_stablecoin_supply.csv", "_list_top_level"),
    _DatasetSpec("blockchain.com", "btc_market_cap_usd",
                 f"{BLOCKCHAIN_COM_BASE}/market-cap",
                 "blockchain_btc_market_cap.csv", "_bc_values"),
    _DatasetSpec("blockchain.com", "btc_hash_rate",
                 f"{BLOCKCHAIN_COM_BASE}/hash-rate",
                 "blockchain_btc_hash_rate.csv", "_bc_values"),
    _DatasetSpec("blockchain.com", "btc_transactions_per_day",
                 f"{BLOCKCHAIN_COM_BASE}/n-transactions",
                 "blockchain_btc_transactions.csv", "_bc_values"),
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _market_structure_dir() -> Path:
    p = config.REPO_ROOT / "data" / "market_structure"
    p.mkdir(parents=True, exist_ok=True)
    return p


def output_path_for(filename: str) -> Path:
    return _market_structure_dir() / filename


# ---------------------------------------------------------------------------
# HTTP + parsers
# ---------------------------------------------------------------------------
def _http_get_json(url: str, params: Optional[Dict[str, Any]] = None,
                    timeout: int = DEFAULT_HTTP_TIMEOUT) -> Any:
    r = requests.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _parse_list_top_level(payload: Any) -> List[Tuple[int, float]]:
    """DefiLlama responses: list of {'date': '<unix>', value-key, ...}."""
    if not isinstance(payload, list):
        raise ValueError(f"unexpected payload shape: {type(payload).__name__}")
    out: List[Tuple[int, float]] = []
    for row in payload:
        if not isinstance(row, dict) or "date" not in row:
            continue
        try:
            ts_s = int(row["date"])
        except (TypeError, ValueError):
            continue
        # The value column varies per endpoint. Pick the first numeric
        # one that isn't `date`. This makes the parser robust to
        # `totalLiquidityUSD` vs `totalCirculatingUSD` etc.
        value: Optional[float] = None
        for k, v in row.items():
            if k == "date":
                continue
            try:
                value = float(v)
                break
            except (TypeError, ValueError):
                continue
            except Exception:  # noqa: BLE001
                continue
        if value is None:
            # Some stablecoin rows have a nested {peggedUSD: ...}; flatten.
            for k, v in row.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        try:
                            value = float(vv)
                            break
                        except Exception:  # noqa: BLE001
                            continue
                if value is not None:
                    break
        if value is None:
            continue
        out.append((ts_s * 1000, value))
    return out


def _parse_bc_values(payload: Any) -> List[Tuple[int, float]]:
    """Blockchain.com responses: {'values': [{'x': ts_s, 'y': value}, ...]}."""
    if not isinstance(payload, dict) or "values" not in payload:
        raise ValueError("missing 'values' key in payload")
    out: List[Tuple[int, float]] = []
    for row in payload["values"]:
        if not isinstance(row, dict):
            continue
        try:
            ts_s = int(row["x"])
            v = float(row["y"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append((ts_s * 1000, v))
    return out


_PARSERS: Dict[str, Callable[[Any], List[Tuple[int, float]]]] = {
    "_list_top_level": _parse_list_top_level,
    "_bc_values": _parse_bc_values,
}


# ---------------------------------------------------------------------------
# Normalisation + gap validation
# ---------------------------------------------------------------------------
def _to_normalised_df(rows: List[Tuple[int, float]],
                       source: str, dataset: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=NORMALISED_COLUMNS)
    df = pd.DataFrame(rows, columns=["timestamp", "value"])
    df["timestamp"] = df["timestamp"].astype("int64")
    df["value"] = df["value"].astype(float)
    df = (df.drop_duplicates(subset=["timestamp"])
              .sort_values("timestamp").reset_index(drop=True))
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True
                                 ).dt.floor("D").dt.strftime("%Y-%m-%d")
    df["source"] = source
    df["dataset"] = dataset
    return df[NORMALISED_COLUMNS]


def _gap_stats_days(timestamps_ms: pd.Series,
                     expected_period_ms: int = 86_400_000) -> Tuple[int, float]:
    if len(timestamps_ms) < 2:
        return 0, 0.0
    ts = timestamps_ms.astype("int64").to_numpy()
    diffs = ts[1:] - ts[:-1]
    gap_threshold = expected_period_ms * 1.5
    gap_count = int((diffs > gap_threshold).sum())
    largest_gap_days = float(diffs.max()) / 86_400_000.0
    return gap_count, round(largest_gap_days, 2)


# ---------------------------------------------------------------------------
# Per-source download
# ---------------------------------------------------------------------------
def download_dataset(spec: _DatasetSpec, *, refresh: bool = False,
                      sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
                      http_timeout: int = DEFAULT_HTTP_TIMEOUT,
                      lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> Path:
    out_path = output_path_for(spec.output_csv)
    if out_path.exists() and not refresh:
        logger.info("market_structure cache hit: %s", spec.dataset)
        return out_path
    parser = _PARSERS[spec.payload_path]
    rows: List[Tuple[int, float]] = []
    try:
        # Blockchain.com supports `timespan=Nyears`; DefiLlama returns
        # full history with no params. Round UP the years so a 1500-day
        # request maps to `5years` (~1825 daily rows), comfortably above
        # the 1460-day threshold rather than 1 day below it.
        params: Dict[str, Any] = {}
        if "blockchain.info" in spec.endpoint:
            years = max(1, -(-lookback_days // 365))   # ceil division
            params = {"timespan": f"{years}years", "format": "json"}
        payload = _http_get_json(spec.endpoint, params=params,
                                  timeout=http_timeout)
        rows = parser(payload)
    except Exception as e:  # noqa: BLE001
        logger.warning("download failed (%s): %s", spec.dataset, e)
    df = _to_normalised_df(rows, spec.source, spec.dataset)
    if df.empty:
        logger.warning("dataset %s wrote 0 rows — endpoint failed?", spec.dataset)
    utils.write_df(df, out_path)
    time.sleep(max(0.0, sleep_seconds))
    return out_path


# ---------------------------------------------------------------------------
# Loaders for downstream code
# ---------------------------------------------------------------------------
def _load(filename: str) -> pd.DataFrame:
    p = output_path_for(filename)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=NORMALISED_COLUMNS)
    df = pd.read_csv(p)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"],
                                          errors="coerce").astype("Int64")
    return df


def load_total_tvl() -> pd.DataFrame:
    return _load("defillama_total_tvl.csv")


def load_stablecoin_supply() -> pd.DataFrame:
    return _load("defillama_stablecoin_supply.csv")


def load_btc_market_cap() -> pd.DataFrame:
    return _load("blockchain_btc_market_cap.csv")


def load_btc_hash_rate() -> pd.DataFrame:
    return _load("blockchain_btc_hash_rate.csv")


def load_btc_transactions() -> pd.DataFrame:
    return _load("blockchain_btc_transactions.csv")


# ---------------------------------------------------------------------------
# Coverage audit
# ---------------------------------------------------------------------------
def _spot_universe_coverage_row() -> Dict[str, Any]:
    """One coverage row that summarises the cached `*_1d.csv` Binance
    spot universe — read from disk only, no network."""
    raw = config.DATA_RAW_DIR
    files = sorted(raw.glob("*_1d.csv")) if raw.exists() else []
    if not files:
        return {
            "source": "binance_spot",
            "dataset": "existing_universe_daily_ohlcv",
            "actual_start": None, "actual_end": None,
            "row_count": 0, "coverage_days": 0.0,
            "enough_for_research": False,
            "largest_gap_days": 0.0,
            "missing_reason": "no_csv",
            "notes": "",
        }
    total_rows = 0
    deepest_days = 0.0
    earliest_ms: Optional[int] = None
    latest_ms: Optional[int] = None
    syms = 0
    for f in files:
        try:
            df = pd.read_csv(f)
            if df.empty or "timestamp" not in df.columns:
                continue
            ts0, ts1 = int(df["timestamp"].iloc[0]), int(df["timestamp"].iloc[-1])
            cov = (ts1 - ts0) / 86_400_000.0
            deepest_days = max(deepest_days, cov)
            earliest_ms = ts0 if earliest_ms is None else min(earliest_ms, ts0)
            latest_ms = ts1 if latest_ms is None else max(latest_ms, ts1)
            total_rows += int(len(df))
            syms += 1
        except Exception:  # noqa: BLE001
            continue
    enough = deepest_days >= MIN_DAYS_FOR_RESEARCH
    return {
        "source": "binance_spot",
        "dataset": "existing_universe_daily_ohlcv",
        "actual_start": (str(pd.to_datetime(earliest_ms, unit="ms", utc=True))
                          if earliest_ms is not None else None),
        "actual_end": (str(pd.to_datetime(latest_ms, unit="ms", utc=True))
                        if latest_ms is not None else None),
        "row_count": int(total_rows),
        "coverage_days": round(deepest_days, 2),
        "enough_for_research": bool(enough),
        "largest_gap_days": 1.0,   # daily bars; coverage_days drives the verdict
        "missing_reason": "" if enough else "deepest_history_below_threshold",
        "notes": f"{syms} symbols cached daily",
    }


def coverage_for_dataset(spec: _DatasetSpec) -> Dict[str, Any]:
    p = output_path_for(spec.output_csv)
    if not p.exists() or p.stat().st_size == 0:
        return {
            "source": spec.source, "dataset": spec.dataset,
            "actual_start": None, "actual_end": None,
            "row_count": 0, "coverage_days": 0.0,
            "enough_for_research": False, "largest_gap_days": 0.0,
            "missing_reason": "no_csv", "notes": "",
        }
    df = pd.read_csv(p)
    if df.empty or "timestamp" not in df.columns:
        return {
            "source": spec.source, "dataset": spec.dataset,
            "actual_start": None, "actual_end": None,
            "row_count": int(len(df)), "coverage_days": 0.0,
            "enough_for_research": False, "largest_gap_days": 0.0,
            "missing_reason": "empty_or_malformed", "notes": "",
        }
    df["timestamp"] = pd.to_numeric(df["timestamp"]).astype("int64")
    start_ms = int(df["timestamp"].iloc[0])
    end_ms = int(df["timestamp"].iloc[-1])
    cov_days = (end_ms - start_ms) / 86_400_000.0
    enough = cov_days >= MIN_DAYS_FOR_RESEARCH
    gap_count, largest_gap_days = _gap_stats_days(df["timestamp"])
    notes_parts = []
    if gap_count > 0:
        notes_parts.append(f"{gap_count} day-gaps; largest {largest_gap_days:.1f}d")
    return {
        "source": spec.source, "dataset": spec.dataset,
        "actual_start": str(pd.to_datetime(start_ms, unit="ms", utc=True)),
        "actual_end": str(pd.to_datetime(end_ms, unit="ms", utc=True)),
        "row_count": int(len(df)),
        "coverage_days": round(cov_days, 2),
        "enough_for_research": bool(enough),
        "largest_gap_days": largest_gap_days,
        "missing_reason": "" if enough else "below_4yr_threshold",
        "notes": "; ".join(notes_parts) if notes_parts else "ok",
    }


def write_coverage(save: bool = True) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for spec in _DATASET_SPECS:
        rows.append(coverage_for_dataset(spec))
    rows.append(_spot_universe_coverage_row())
    out = pd.DataFrame(rows, columns=COVERAGE_COLUMNS)
    if save:
        utils.write_df(out, config.RESULTS_DIR / "market_structure_data_coverage.csv")
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def download_all_market_structure(
    *, refresh: bool = False,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    http_timeout: int = DEFAULT_HTTP_TIMEOUT,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    save_coverage: bool = True,
) -> Dict[str, Any]:
    """Download every market-structure dataset and write the coverage CSV.
    Never aborts on a single source failure."""
    utils.assert_paper_only()
    paths: Dict[str, Path] = {}
    for spec in _DATASET_SPECS:
        try:
            p = download_dataset(spec, refresh=refresh,
                                  sleep_seconds=sleep_seconds,
                                  http_timeout=http_timeout,
                                  lookback_days=lookback_days)
            paths[spec.dataset] = p
        except Exception as e:  # noqa: BLE001
            logger.warning("download_dataset crashed (%s): %s", spec.dataset, e)
    coverage = write_coverage(save=save_coverage)
    return {"paths": paths, "coverage_df": coverage}
