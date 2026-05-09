"""
Free-tier market-structure data audit.

This module is **not** a strategy. It probes free public endpoints once,
records the actual historical depth they return without an API key, and
writes `results/market_structure_data_audit.csv`. The downstream task
("can we build a market-structure strategy on free data alone?") cannot
be answered honestly until this audit confirms ≥ 4 years of daily
coverage from at least one source.

Sources probed:

  * **Blockchain.com Charts** (`api.blockchain.info/charts/{name}`)
        Public, no key. `timespan=5years` returns true daily resolution.
        Probed: market-price, market-cap, hash-rate, n-transactions.

  * **CoinGecko v3 free** (`api.coingecko.com/api/v3/...`)
        Free tier — historical depth is now CAPPED to the most-recent
        365 days for `/coins/{id}/market_chart` and the global market
        cap chart is **PRO-only**. Audited so the limitation is recorded
        explicitly and we don't reach for the wrong endpoint later.

  * **CoinPaprika** (`api.coinpaprika.com/v1/global`)
        Current dominance snapshot, no historical on the free tier.

  * **DefiLlama** (`api.llama.fi`, `stablecoins.llama.fi`)
        Public, no key. Historical TVL (per chain or total) and
        historical stablecoin supply since 2017. Useful as a proxy for
        DeFi-leverage and stablecoin-supply regime.

  * **Existing Binance spot universe** (local CSV cache from v1)
        Daily `BTC/USDT` etc. already paginated multi-year via the
        v1 collector. No network call here — counted from disk.

Hard rules (carried over from v1 + the previous failed branch):
  * No API keys.
  * No paid endpoints.
  * No network calls beyond the one probe per source needed for the audit.
  * No strategy code in this module.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from . import config, utils

logger = utils.get_logger("cte.market_structure_audit")


HTTP_TIMEOUT = 20  # seconds
# 4 years of daily bars is the threshold for "usable for research" —
# matches the v1 walk-forward setup (180-day IS + at least 5 OOS windows).
MIN_DAYS_FOR_RESEARCH = 4 * 365


# ---------------------------------------------------------------------------
# Audit row schema (locked here so tests can assert it precisely).
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
    """Build an audit row from a list of unix timestamps (in seconds)."""
    row = _empty_row(source, dataset, endpoint, requires_key, "")
    if not ts_seconds:
        row["notes"] = ("empty response" + (f"; {notes_extra}" if notes_extra else ""))
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
# Source probes — each one is a small, idempotent function returning a
# single audit row. They never raise: a network or shape error becomes
# a row with `usable_for_research=False` and a `notes` description.
# ---------------------------------------------------------------------------
def probe_blockchain_com_chart(name: str,
                                  display: Optional[str] = None) -> Dict[str, Any]:
    """One BC.com chart. `timespan=5years` returns daily resolution."""
    display = display or name
    url = f"https://api.blockchain.info/charts/{name}"
    try:
        r = requests.get(url, params={"timespan": "5years", "format": "json"},
                          timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        j = r.json()
    except Exception as e:  # noqa: BLE001
        return _empty_row("blockchain.com", display, url,
                            requires_key=False, notes=f"http_error: {e}")
    if not isinstance(j, dict) or "values" not in j:
        return _empty_row("blockchain.com", display, url,
                            requires_key=False, notes="unexpected_response")
    ts = [int(v["x"]) for v in j["values"]]
    return _row_from_timeseries("blockchain.com", display, url,
                                  requires_key=False, ts_seconds=ts)


def probe_coingecko_btc_market_chart() -> Dict[str, Any]:
    """CoinGecko free tier — historical depth on `days=max` is capped."""
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    try:
        r = requests.get(url,
                          params={"vs_currency": "usd", "days": "max",
                                   "interval": "daily"},
                          timeout=HTTP_TIMEOUT)
    except Exception as e:  # noqa: BLE001
        return _empty_row("coingecko", "bitcoin_price_usd_max", url,
                            requires_key=False, notes=f"http_error: {e}")
    if r.status_code == 401:
        # Free tier explicitly rejects historical depth on this endpoint.
        return _empty_row(
            "coingecko", "bitcoin_price_usd_max", url,
            requires_key=False,
            notes=("free tier capped to recent ~365d; days=max returns 401 "
                    "(error_code 10012) — pro-only beyond that"),
        )
    try:
        j = r.json()
    except Exception:  # noqa: BLE001
        return _empty_row("coingecko", "bitcoin_price_usd_max", url,
                            requires_key=False, notes=f"non_json: {r.status_code}")
    if "prices" in j and isinstance(j["prices"], list):
        ts_s = [int(p[0]) // 1000 for p in j["prices"]]
        return _row_from_timeseries("coingecko", "bitcoin_price_usd_max", url,
                                      requires_key=False, ts_seconds=ts_s)
    return _empty_row("coingecko", "bitcoin_price_usd_max", url,
                        requires_key=False,
                        notes=f"unexpected_response: status={r.status_code}")


def probe_coingecko_global_dominance() -> Dict[str, Any]:
    """Current snapshot only — historical version is PRO only."""
    url = "https://api.coingecko.com/api/v3/global"
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        j = r.json()
    except Exception as e:  # noqa: BLE001
        return _empty_row("coingecko", "global_dominance_history", url,
                            requires_key=False, notes=f"http_error: {e}")
    btc_dom = (j.get("data", {}).get("market_cap_percentage", {}).get("btc")
                 if isinstance(j, dict) else None)
    return _empty_row(
        "coingecko", "global_dominance_history", url, requires_key=False,
        notes=(f"current snapshot only (BTC dominance now ≈ "
                 f"{btc_dom:.1f}%); historical /global/market_cap_chart is "
                 f"PRO-only on the free tier"
                 if btc_dom is not None else
                 "current snapshot only; historical PRO-only"),
    )


def probe_coinpaprika_global_dominance() -> Dict[str, Any]:
    url = "https://api.coinpaprika.com/v1/global"
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        j = r.json()
    except Exception as e:  # noqa: BLE001
        return _empty_row("coinpaprika", "global_dominance_history", url,
                            requires_key=False, notes=f"http_error: {e}")
    btc_dom = j.get("bitcoin_dominance_percentage") if isinstance(j, dict) else None
    return _empty_row(
        "coinpaprika", "global_dominance_history", url, requires_key=False,
        notes=(f"current snapshot only (BTC dominance now ≈ {btc_dom:.1f}%); "
                 f"historical access requires CoinPaprika PRO"
                 if btc_dom is not None else
                 "current snapshot only; historical requires PRO"),
    )


def probe_defillama_total_tvl() -> Dict[str, Any]:
    url = "https://api.llama.fi/charts"
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        j = r.json()
    except Exception as e:  # noqa: BLE001
        return _empty_row("defillama", "total_tvl_all_chains", url,
                            requires_key=False, notes=f"http_error: {e}")
    if not isinstance(j, list):
        return _empty_row("defillama", "total_tvl_all_chains", url,
                            requires_key=False, notes="unexpected_response")
    ts = [int(p["date"]) for p in j if "date" in p]
    return _row_from_timeseries("defillama", "total_tvl_all_chains", url,
                                  requires_key=False, ts_seconds=ts)


def probe_defillama_stablecoins() -> Dict[str, Any]:
    url = "https://stablecoins.llama.fi/stablecoincharts/all"
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        j = r.json()
    except Exception as e:  # noqa: BLE001
        return _empty_row("defillama", "stablecoin_supply_total", url,
                            requires_key=False, notes=f"http_error: {e}")
    if not isinstance(j, list):
        return _empty_row("defillama", "stablecoin_supply_total", url,
                            requires_key=False, notes="unexpected_response")
    ts = [int(p["date"]) for p in j if "date" in p]
    return _row_from_timeseries("defillama", "stablecoin_supply_total", url,
                                  requires_key=False, ts_seconds=ts)


def probe_existing_spot_universe() -> Dict[str, Any]:
    """Count cached `*_1d.csv` daily spot files. Pure local I/O."""
    raw_dir: Path = config.DATA_RAW_DIR
    files = list(raw_dir.glob("*_1d.csv")) if raw_dir.exists() else []
    if not files:
        return _empty_row(
            "binance_spot", "existing_universe_daily_ohlcv", str(raw_dir),
            requires_key=False, notes="no cached _1d.csv files",
        )
    coverage_days = 0.0
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
            earliest_ms = ts0 if earliest_ms is None else min(earliest_ms, ts0)
            latest_ms = ts1 if latest_ms is None else max(latest_ms, ts1)
            cov = (ts1 - ts0) / 86_400_000.0
            coverage_days = max(coverage_days, cov)  # report deepest history
            total_rows += int(len(df))
            syms_ok += 1
        except Exception:  # noqa: BLE001
            continue
    if earliest_ms is None or latest_ms is None:
        return _empty_row(
            "binance_spot", "existing_universe_daily_ohlcv", str(raw_dir),
            requires_key=False, notes="cached files unreadable",
        )
    enough = coverage_days >= MIN_DAYS_FOR_RESEARCH
    return {
        "source": "binance_spot",
        "dataset": "existing_universe_daily_ohlcv",
        "endpoint_or_source": str(raw_dir),
        "requires_api_key": False, "free_access": True,
        "actual_start": str(pd.to_datetime(earliest_ms, unit="ms", utc=True)),
        "actual_end": str(pd.to_datetime(latest_ms, unit="ms", utc=True)),
        "row_count": int(total_rows),
        "coverage_days": round(coverage_days, 2),
        "usable_for_research": bool(enough),
        "notes": f"{syms_ok} symbols cached daily; deepest history "
                  f"{coverage_days:.0f} days",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def audit_market_structure_data(save: bool = True) -> pd.DataFrame:
    """Run every probe in series, build the audit table, optionally
    persist it to `results/market_structure_data_audit.csv`."""
    probes: List[Tuple[str, Any]] = [
        ("blockchain.com market-price",
         lambda: probe_blockchain_com_chart("market-price",
                                              "btc_market_price_usd")),
        ("blockchain.com market-cap",
         lambda: probe_blockchain_com_chart("market-cap",
                                              "btc_market_cap_usd")),
        ("blockchain.com hash-rate",
         lambda: probe_blockchain_com_chart("hash-rate", "btc_hash_rate")),
        ("blockchain.com n-transactions",
         lambda: probe_blockchain_com_chart("n-transactions",
                                              "btc_transactions_per_day")),
        ("coingecko bitcoin market_chart days=max",
         probe_coingecko_btc_market_chart),
        ("coingecko global dominance", probe_coingecko_global_dominance),
        ("coinpaprika global dominance", probe_coinpaprika_global_dominance),
        ("defillama total TVL", probe_defillama_total_tvl),
        ("defillama stablecoin supply", probe_defillama_stablecoins),
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
        time.sleep(0.1)  # tiny courtesy delay between probes
    out = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    if save:
        utils.write_df(
            out, config.RESULTS_DIR / "market_structure_data_audit.csv",
        )
    return out
