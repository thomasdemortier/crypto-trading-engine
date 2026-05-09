"""
Free-tier sentiment data collector — alternative.me Fear & Greed Index.

The audit (`src.sentiment_data_audit`) confirmed this endpoint:
  * No API key, no rate-limit visible at this volume.
  * `?limit=0&format=json` returns full daily history (~3 016 rows
    since 2018-02-01 as of 2026-05).

Output (data/sentiment/fear_greed.csv):
    timestamp (ms), date (YYYY-MM-DD UTC), fear_greed_value (int 0..100),
    fear_greed_classification (str), source

Coverage row (results/sentiment_data_coverage.csv):
    source, dataset, actual_start, actual_end, row_count, coverage_days,
    enough_for_research, largest_gap_days, notes

Hard rules:
    * No API keys, no paid endpoints, no private endpoints.
    * Failures are logged and skipped; the run never aborts.
    * Raw data is never silently forward-filled — gaps are recorded.
    * No strategy code, no order plumbing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from . import config, utils

logger = utils.get_logger("cte.sentiment_collector")

DEFAULT_HTTP_TIMEOUT = 30
MIN_DAYS_FOR_RESEARCH = 4 * 365

ALTME_FEAR_GREED_URL = "https://api.alternative.me/fng/"
SOURCE_NAME = "alternative.me"
DATASET_NAME = "fear_greed_index_daily"
OUTPUT_FILENAME = "fear_greed.csv"


# ---------------------------------------------------------------------------
# Schemas (locked here so tests can assert)
# ---------------------------------------------------------------------------
NORMALISED_COLUMNS: List[str] = [
    "timestamp", "date", "fear_greed_value",
    "fear_greed_classification", "source",
]
COVERAGE_COLUMNS: List[str] = [
    "source", "dataset", "actual_start", "actual_end",
    "row_count", "coverage_days", "enough_for_research",
    "largest_gap_days", "notes",
]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _sentiment_dir() -> Path:
    p = config.REPO_ROOT / "data" / "sentiment"
    p.mkdir(parents=True, exist_ok=True)
    return p


def output_path() -> Path:
    return _sentiment_dir() / OUTPUT_FILENAME


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def _parse_payload(payload: Any) -> List[Dict[str, Any]]:
    """alternative.me shape:
        {"data": [{"value": "38", "value_classification": "Fear",
                   "timestamp": "1778284800",
                   "time_until_update": "..."}, ...]}
    The endpoint returns NEWEST first; we sort ascending downstream."""
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError("missing 'data' key in alternative.me payload")
    rows: List[Dict[str, Any]] = []
    for r in payload["data"]:
        if not isinstance(r, dict) or "timestamp" not in r:
            continue
        try:
            ts_s = int(r["timestamp"])
            value = int(float(r["value"]))
        except (KeyError, TypeError, ValueError):
            continue
        cls = str(r.get("value_classification", "")).strip()
        rows.append({
            "timestamp": ts_s * 1000,            # → milliseconds
            "fear_greed_value": value,
            "fear_greed_classification": cls,
        })
    return rows


# ---------------------------------------------------------------------------
# Normalisation + gap stats
# ---------------------------------------------------------------------------
def _to_normalised_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=NORMALISED_COLUMNS)
    df = pd.DataFrame(rows)
    df["timestamp"] = df["timestamp"].astype("int64")
    df = (df.drop_duplicates(subset=["timestamp"])
              .sort_values("timestamp").reset_index(drop=True))
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True
                                 ).dt.floor("D").dt.strftime("%Y-%m-%d")
    df["source"] = SOURCE_NAME
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
# Loader
# ---------------------------------------------------------------------------
def load_fear_greed() -> pd.DataFrame:
    p = output_path()
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=NORMALISED_COLUMNS)
    df = pd.read_csv(p)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"],
                                          errors="coerce").astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
def coverage_for_dataset() -> Dict[str, Any]:
    p = output_path()
    if not p.exists() or p.stat().st_size == 0:
        return {
            "source": SOURCE_NAME, "dataset": DATASET_NAME,
            "actual_start": None, "actual_end": None,
            "row_count": 0, "coverage_days": 0.0,
            "enough_for_research": False, "largest_gap_days": 0.0,
            "notes": "no_csv",
        }
    df = pd.read_csv(p)
    if df.empty or "timestamp" not in df.columns:
        return {
            "source": SOURCE_NAME, "dataset": DATASET_NAME,
            "actual_start": None, "actual_end": None,
            "row_count": int(len(df)), "coverage_days": 0.0,
            "enough_for_research": False, "largest_gap_days": 0.0,
            "notes": "empty_or_malformed",
        }
    df["timestamp"] = pd.to_numeric(df["timestamp"]).astype("int64")
    start_ms = int(df["timestamp"].iloc[0])
    end_ms = int(df["timestamp"].iloc[-1])
    cov_days = (end_ms - start_ms) / 86_400_000.0
    enough = cov_days >= MIN_DAYS_FOR_RESEARCH
    gap_count, largest_gap_days = _gap_stats_days(df["timestamp"])
    notes_parts: List[str] = []
    if gap_count > 0:
        notes_parts.append(f"{gap_count} day-gaps; largest "
                            f"{largest_gap_days:.1f}d")
    return {
        "source": SOURCE_NAME, "dataset": DATASET_NAME,
        "actual_start": str(pd.to_datetime(start_ms, unit="ms", utc=True)),
        "actual_end": str(pd.to_datetime(end_ms, unit="ms", utc=True)),
        "row_count": int(len(df)),
        "coverage_days": round(cov_days, 2),
        "enough_for_research": bool(enough),
        "largest_gap_days": largest_gap_days,
        "notes": "; ".join(notes_parts) if notes_parts else "ok",
    }


def write_coverage(save: bool = True) -> pd.DataFrame:
    out = pd.DataFrame([coverage_for_dataset()], columns=COVERAGE_COLUMNS)
    if save:
        utils.write_df(out, config.RESULTS_DIR / "sentiment_data_coverage.csv")
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def download_fear_greed(*, refresh: bool = False,
                          http_timeout: int = DEFAULT_HTTP_TIMEOUT) -> Path:
    """Download the full alternative.me Fear & Greed history. Caches to
    `data/sentiment/fear_greed.csv`."""
    p = output_path()
    if p.exists() and not refresh:
        logger.info("sentiment cache hit: %s", p.name)
        return p
    rows: List[Dict[str, Any]] = []
    try:
        r = requests.get(ALTME_FEAR_GREED_URL,
                          params={"limit": 0, "format": "json"},
                          timeout=http_timeout)
        r.raise_for_status()
        rows = _parse_payload(r.json())
    except Exception as e:  # noqa: BLE001
        logger.warning("alternative.me fetch failed: %s", e)
    df = _to_normalised_df(rows)
    if df.empty:
        logger.warning("alternative.me returned 0 usable rows")
    utils.write_df(df, p)
    return p


def download_sentiment_data(*, refresh: bool = False,
                              http_timeout: int = DEFAULT_HTTP_TIMEOUT,
                              save_coverage: bool = True) -> Dict[str, Any]:
    """Download every sentiment dataset (currently just F&G), persist the
    coverage CSV. Never aborts on failure."""
    utils.assert_paper_only()
    p = download_fear_greed(refresh=refresh, http_timeout=http_timeout)
    coverage = write_coverage(save=save_coverage)
    return {"path": p, "coverage_df": coverage}
