"""
Public OHLCV data collector.

Uses ccxt with the EXCHANGE'S PUBLIC ENDPOINTS ONLY. We never instantiate the
exchange with API keys. The only ccxt method called is `fetch_ohlcv`.

Caching: data is stored as CSV in `data/raw/<symbol>_<timeframe>.csv` and
reused unless `refresh=True` is passed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

from . import config, utils

logger = utils.get_logger("cte.data")


# ---------------------------------------------------------------------------
# Exchange handling
# ---------------------------------------------------------------------------
def _build_exchange(name: str):
    """Construct a public-only ccxt exchange. No keys are passed."""
    utils.assert_paper_only()
    import ccxt  # imported lazily so tests that don't use ccxt run faster
    if not hasattr(ccxt, name):
        raise ValueError(f"unknown exchange: {name}")
    klass = getattr(ccxt, name)
    ex = klass({"enableRateLimit": True, "timeout": 20_000})
    return ex


def _resolve_symbol(exchange, symbol: str) -> Optional[str]:
    """Return a tradeable symbol for this exchange, or None if not listed.

    Kraken often quotes BTC against USD rather than USDT. We optionally fall
    back to the USD variant to keep BTC/USDT and ETH/USDT requests working.
    """
    try:
        markets = exchange.load_markets()
    except Exception as e:  # noqa: BLE001
        logger.warning("could not load markets for %s: %s", exchange.id, e)
        return None

    if symbol in markets:
        return symbol

    if exchange.id == "kraken" and config.KRAKEN_USDT_TO_USD_FALLBACK:
        if symbol.endswith("/USDT"):
            usd_variant = symbol.replace("/USDT", "/USD")
            if usd_variant in markets:
                logger.info("kraken: %s not listed, using %s instead", symbol, usd_variant)
                return usd_variant
    return None


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def _timeframe_ms(timeframe: str) -> int:
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    n = int(timeframe[:-1])
    u = timeframe[-1]
    if u not in units:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    return n * units[u]


def _fetch_ohlcv_paginated(
    exchange, symbol: str, timeframe: str, since_ms: int, limit_per_call: int
) -> List[list]:
    """Walk the public OHLCV endpoint until we reach 'now'."""
    tf_ms = _timeframe_ms(timeframe)
    all_rows: List[list] = []
    cursor = since_ms
    now_ms = exchange.milliseconds()
    while cursor < now_ms:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe,
                                         since=cursor, limit=limit_per_call)
        except Exception as e:  # noqa: BLE001
            logger.warning("fetch_ohlcv error (%s %s): %s", symbol, timeframe, e)
            break
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        # advance one bar past the last to avoid infinite loops on duplicates
        next_cursor = last_ts + tf_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        # be polite even though enableRateLimit handles most of it
        time.sleep(exchange.rateLimit / 1000.0)
    return all_rows


def _rows_to_df(rows: List[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low",
                                     "close", "volume"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df[["timestamp", "datetime", "open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def download_symbol(
    symbol: str,
    timeframe: str = config.DEFAULT_TIMEFRAME,
    days: int = config.DEFAULT_HISTORY_DAYS,
    refresh: bool = False,
) -> Path:
    """Download OHLCV for one symbol/timeframe and return the CSV path.

    If a CSV already exists and `refresh` is False, returns immediately.
    """
    utils.assert_paper_only()
    utils.ensure_dirs([config.DATA_RAW_DIR])

    csv_path = utils.csv_path_for(symbol, timeframe)
    if csv_path.exists() and not refresh:
        logger.info("cache hit: %s (%s)", csv_path.name,
                    f"{csv_path.stat().st_size // 1024} KB")
        return csv_path

    since_ms = int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).timestamp() * 1000)

    last_error: Optional[Exception] = None
    for ex_name in (config.PRIMARY_EXCHANGE, config.FALLBACK_EXCHANGE):
        try:
            ex = _build_exchange(ex_name)
            resolved = _resolve_symbol(ex, symbol)
            if resolved is None:
                logger.info("%s: %s not listed, trying next exchange", ex_name, symbol)
                continue
            logger.info("downloading %s %s from %s (%d days)",
                        resolved, timeframe, ex_name, days)
            rows = _fetch_ohlcv_paginated(ex, resolved, timeframe, since_ms,
                                          config.FETCH_CHUNK_LIMIT)
            if not rows:
                logger.warning("%s returned no rows for %s %s",
                               ex_name, resolved, timeframe)
                continue
            df = _rows_to_df(rows)
            utils.write_df(df, csv_path)
            logger.info("saved %d candles → %s", len(df), csv_path.name)
            return csv_path
        except Exception as e:  # noqa: BLE001
            last_error = e
            logger.warning("%s failed for %s %s: %s", ex_name, symbol, timeframe, e)

    raise RuntimeError(
        f"failed to download {symbol} {timeframe} from any configured "
        f"exchange. last error: {last_error}"
    )


def download_all(
    assets: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    days: int = config.DEFAULT_HISTORY_DAYS,
    refresh: bool = False,
) -> List[Path]:
    """Download every (asset, timeframe) combination."""
    utils.assert_paper_only()
    assets = assets or config.ASSETS
    timeframes = timeframes or [config.DEFAULT_TIMEFRAME]
    paths: List[Path] = []
    for symbol in assets:
        for tf in timeframes:
            try:
                paths.append(download_symbol(symbol, tf, days=days, refresh=refresh))
            except Exception as e:  # noqa: BLE001
                logger.error("could not download %s %s: %s", symbol, tf, e)
    return paths


def load_candles(symbol: str, timeframe: str = config.DEFAULT_TIMEFRAME) -> pd.DataFrame:
    """Load cached candles, or raise FileNotFoundError if not yet downloaded."""
    csv_path = utils.csv_path_for(symbol, timeframe)
    df = utils.read_csv_if_exists(csv_path)
    if df is None:
        raise FileNotFoundError(
            f"No cached data for {symbol} {timeframe}. Run "
            f"`python main.py download` first."
        )
    return utils.parse_timestamp_column(df)
