"""
Free-tier sentiment / Fear & Greed data audit.

This module is **not** a strategy. It probes free public endpoints once,
records the actual historical depth they return without an API key, and
writes `results/sentiment_data_audit.csv`. Strategy work on the
sentiment branch cannot start until this audit confirms ≥ 4 years of
daily coverage from at least one source.

Sources probed:

  * **alternative.me Fear & Greed** (`api.alternative.me/fng/?limit=0`)
        Public, no key. `limit=0` returns the full history (8 + years
        as of 2026, beginning 2018-02-01). Daily resolution.

  * **CoinMarketCap data-api Fear & Greed** (current path)
        Currently returns 404 on the public path; historical access
        requires the paid CMC PRO API. Recorded for clarity.

  * **CoinGecko `/coins/bitcoin` community_data**
        Returns CURRENT-snapshot social metrics only (facebook_likes,
        reddit_avg_posts_48h, etc). No historical depth on free tier.

  * **Reddit `/r/Bitcoin/about.json`**
        Public, no key. Returns a current-snapshot subscriber count
        only — no historical time-series.

  * **Existing Binance daily OHLCV** (local cache)
        Counted from disk. No network call. Used as the alignment
        reference for any future sentiment strategy.

Hard rules (carried from the v1 closure + every prior research branch):
  * No API keys.
  * No paid endpoints.
  * Failures are recorded as unusable rows; the audit never raises.
  * No strategy code in this module.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from . import config, utils

logger = utils.get_logger("cte.sentiment_audit")


HTTP_TIMEOUT = 20  # seconds
# 4 years of daily bars is the threshold for "usable for research" —
# matches the prior branches' walk-forward requirements.
MIN_DAYS_FOR_RESEARCH = 4 * 365


# ---------------------------------------------------------------------------
# Audit row schema
# ---------------------------------------------------------------------------
AUDIT_COLUMNS: List[str] = [
    "source", "dataset", "endpoint_or_source",
    "requires_api_key", "free_access",
    "actual_start", "actual_end", "row_count", "coverage_days",
    "usable_for_research", "notes",
]


def _empty_row(source: str, dataset: str, endpoint: str,
                requires_key: bool, notes: str) -> Dict[str, Any]:
    return {
        "source": source, "dataset": dataset, "endpoint_or_source": endpoint,
        "requires_api_key": bool(requires_key),
        "free_access": not bool(requires_key),
        "actual_start": None, "actual_end": None,
        "row_count": 0, "coverage_days": 0.0,
        "usable_for_research": False, "notes": notes,
    }


def _row_from_timeseries(
    source: str, dataset: str, endpoint: str,
    requires_key: bool, ts_seconds: List[int], notes_extra: str = "",
) -> Dict[str, Any]:
    row = _empty_row(source, dataset, endpoint, requires_key, "")
    if not ts_seconds:
        row["notes"] = ("empty response"
                          + (f"; {notes_extra}" if notes_extra else ""))
        return row
    start_s = int(ts_seconds[0])
    end_s = int(ts_seconds[-1])
    cov_days = (end_s - start_s) / 86400.0
    enough = cov_days >= MIN_DAYS_FOR_RESEARCH
    row.update({
        "actual_start": str(pd.to_datetime(start_s, unit="s", utc=True)),
        "actual_end": str(pd.to_datetime(end_s, unit="s", utc=True)),
        "row_count": int(len(ts_seconds)),
        "coverage_days": round(cov_days, 2),
        "usable_for_research": bool(enough),
        "notes": notes_extra or ("ok" if enough else
                                  f"under {MIN_DAYS_FOR_RESEARCH}d threshold"),
    })
    return row


# ---------------------------------------------------------------------------
# Source probes
# ---------------------------------------------------------------------------
def probe_alternative_me_fear_greed() -> Dict[str, Any]:
    """alternative.me F&G with `limit=0` (full history). Daily resolution."""
    url = "https://api.alternative.me/fng/"
    try:
        r = requests.get(url, params={"limit": 0, "format": "json"},
                          timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        j = r.json()
    except Exception as e:  # noqa: BLE001
        return _empty_row("alternative.me", "fear_greed_index_daily", url,
                            requires_key=False, notes=f"http_error: {e}")
    if not isinstance(j, dict) or "data" not in j:
        return _empty_row("alternative.me", "fear_greed_index_daily", url,
                            requires_key=False, notes="unexpected_response")
    rows = j["data"]
    if not isinstance(rows, list):
        return _empty_row("alternative.me", "fear_greed_index_daily", url,
                            requires_key=False, notes="data not a list")
    # The endpoint returns newest first; sort ascending.
    ts: List[int] = []
    for row in rows:
        if not isinstance(row, dict) or "timestamp" not in row:
            continue
        try:
            ts.append(int(row["timestamp"]))
        except (TypeError, ValueError):
            continue
    ts.sort()
    return _row_from_timeseries(
        "alternative.me", "fear_greed_index_daily", url,
        requires_key=False, ts_seconds=ts,
    )


def probe_coinmarketcap_fear_greed_historical() -> Dict[str, Any]:
    """CMC's data-api currently returns 404 on the public path."""
    url = "https://api.coinmarketcap.com/data-api/v3/fear-greed/historical"
    try:
        r = requests.get(url, params={"start": 1, "limit": 100},
                          timeout=HTTP_TIMEOUT)
    except Exception as e:  # noqa: BLE001
        return _empty_row("coinmarketcap", "fear_greed_historical", url,
                            requires_key=True, notes=f"http_error: {e}")
    note = (f"public data-api returned status {r.status_code}; "
              f"historical Fear & Greed requires CMC PRO API")
    return _empty_row("coinmarketcap", "fear_greed_historical", url,
                        requires_key=True, notes=note)


def probe_coingecko_bitcoin_community_data() -> Dict[str, Any]:
    """CoinGecko `/coins/bitcoin` returns current community metrics only —
    no historical time-series on the free tier."""
    url = "https://api.coingecko.com/api/v3/coins/bitcoin"
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        j = r.json()
    except Exception as e:  # noqa: BLE001
        return _empty_row("coingecko", "bitcoin_community_data_history", url,
                            requires_key=False, notes=f"http_error: {e}")
    cd = j.get("community_data") if isinstance(j, dict) else None
    sample = ""
    if isinstance(cd, dict):
        sample = ", ".join(f"{k}={v}" for k, v in list(cd.items())[:3])
    return _empty_row(
        "coingecko", "bitcoin_community_data_history", url,
        requires_key=False,
        notes=("current snapshot only "
                 + (f"(e.g. {sample})" if sample else "")
                 + "; historical community metrics are PRO-only"),
    )


def probe_reddit_bitcoin_subscribers() -> Dict[str, Any]:
    """Reddit `/r/Bitcoin/about.json` — current subscriber count only."""
    url = "https://www.reddit.com/r/Bitcoin/about.json"
    try:
        r = requests.get(url,
                          headers={"User-Agent": "cte-research/1.0"},
                          timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        j = r.json()
    except Exception as e:  # noqa: BLE001
        return _empty_row("reddit", "rbitcoin_subscriber_history", url,
                            requires_key=False, notes=f"http_error: {e}")
    subs = (j.get("data", {}).get("subscribers")
              if isinstance(j, dict) else None)
    return _empty_row(
        "reddit", "rbitcoin_subscriber_history", url, requires_key=False,
        notes=(f"current snapshot only (~{int(subs):,} subscribers); "
                 f"historical Reddit subscriber counts require a paid feed "
                 f"such as Pushshift archives" if isinstance(subs, int) else
                 "current snapshot only; no historical access on free tier"),
    )


def probe_existing_spot_universe() -> Dict[str, Any]:
    """Count cached daily Binance OHLCV files. Pure local I/O."""
    raw_dir: Path = config.DATA_RAW_DIR
    files = list(raw_dir.glob("*_1d.csv")) if raw_dir.exists() else []
    if not files:
        return _empty_row(
            "binance_spot", "existing_universe_daily_ohlcv", str(raw_dir),
            requires_key=False, notes="no cached _1d.csv files",
        )
    deepest_days = 0.0
    total_rows = 0
    earliest_ms: Optional[int] = None
    latest_ms: Optional[int] = None
    syms_ok = 0
    for f in files:
        try:
            df = pd.read_csv(f)
            if df.empty or "timestamp" not in df.columns:
                continue
            ts0 = int(df["timestamp"].iloc[0])
            ts1 = int(df["timestamp"].iloc[-1])
            earliest_ms = (ts0 if earliest_ms is None
                            else min(earliest_ms, ts0))
            latest_ms = (ts1 if latest_ms is None
                          else max(latest_ms, ts1))
            cov = (ts1 - ts0) / 86_400_000.0
            deepest_days = max(deepest_days, cov)
            total_rows += int(len(df))
            syms_ok += 1
        except Exception:  # noqa: BLE001
            continue
    if earliest_ms is None or latest_ms is None:
        return _empty_row(
            "binance_spot", "existing_universe_daily_ohlcv", str(raw_dir),
            requires_key=False, notes="cached files unreadable",
        )
    enough = deepest_days >= MIN_DAYS_FOR_RESEARCH
    return {
        "source": "binance_spot",
        "dataset": "existing_universe_daily_ohlcv",
        "endpoint_or_source": str(raw_dir),
        "requires_api_key": False, "free_access": True,
        "actual_start": str(pd.to_datetime(earliest_ms, unit="ms", utc=True)),
        "actual_end": str(pd.to_datetime(latest_ms, unit="ms", utc=True)),
        "row_count": int(total_rows),
        "coverage_days": round(deepest_days, 2),
        "usable_for_research": bool(enough),
        "notes": f"{syms_ok} symbols cached daily; deepest history "
                  f"{deepest_days:.0f} days",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def audit_sentiment_data(save: bool = True) -> pd.DataFrame:
    """Run every probe in series, build the audit table, persist to
    `results/sentiment_data_audit.csv`."""
    probes: List[Tuple[str, Any]] = [
        ("alternative.me Fear & Greed limit=0",
         probe_alternative_me_fear_greed),
        ("CoinMarketCap Fear & Greed historical",
         probe_coinmarketcap_fear_greed_historical),
        ("CoinGecko bitcoin community_data",
         probe_coingecko_bitcoin_community_data),
        ("Reddit r/Bitcoin about", probe_reddit_bitcoin_subscribers),
        ("existing binance spot universe", probe_existing_spot_universe),
    ]
    rows: List[Dict[str, Any]] = []
    for label, fn in probes:
        try:
            row = fn()
        except Exception as e:  # noqa: BLE001
            logger.warning("audit probe %s raised: %s", label, e)
            row = _empty_row(label.split()[0], label, "", False,
                              f"probe_raised: {e}")
        rows.append(row)
        time.sleep(0.1)
    out = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    if save:
        utils.write_df(
            out, config.RESULTS_DIR / "sentiment_data_audit.csv",
        )
    return out
