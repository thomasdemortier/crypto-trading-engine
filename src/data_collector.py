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
from typing import Dict, List, Optional

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
    exchange, symbol: str, timeframe: str, since_ms: int,
    limit_per_call: int,
) -> List[list]:
    """Walk the public OHLCV endpoint forward in batches until we reach now.

    Stuck-pagination detection:
      Some exchanges (notably Kraken's public OHLC endpoint) return the
      most-recent ~720 candles regardless of `since`. After dedup that
      collapses to a single batch's worth of unique rows. We detect this
      by tracking the FIRST timestamp of each batch — if it's not strictly
      greater than the previous batch's first timestamp, we are stuck and
      bail out so the caller can try the fallback exchange.
    """
    tf_ms = _timeframe_ms(timeframe)
    all_rows: List[list] = []
    cursor = since_ms
    now_ms = exchange.milliseconds()
    prev_first_ts: Optional[int] = None
    iter_count = 0
    while cursor < now_ms:
        iter_count += 1
        try:
            batch = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe,
                since=cursor, limit=limit_per_call,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("fetch_ohlcv error (%s %s): %s", symbol, timeframe, e)
            break
        if not batch:
            break
        first_ts = int(batch[0][0])
        last_ts = int(batch[-1][0])
        # Stuck-detection: a healthy paginated response advances on every
        # iteration. If the first timestamp does not strictly increase,
        # the exchange is ignoring `since` (Kraken behavior).
        if prev_first_ts is not None and first_ts <= prev_first_ts:
            logger.warning(
                "fetch_ohlcv: %s.%s appears to ignore `since` (stuck at "
                "first_ts=%d after %d iters). Aborting paginated walk so "
                "caller can try fallback exchange.",
                exchange.id, symbol, first_ts, iter_count,
            )
            # Still accept the data we already have for this exchange.
            all_rows.extend(batch)
            break
        all_rows.extend(batch)
        prev_first_ts = first_ts
        # advance one bar past the last to avoid infinite loops on duplicates
        next_cursor = last_ts + tf_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        # be polite even though enableRateLimit handles most of it
        time.sleep(exchange.rateLimit / 1000.0)
    logger.info("fetch_ohlcv done: %s.%s fetched %d raw rows in %d iters",
                exchange.id, symbol, len(all_rows), iter_count)
    return all_rows


def _rows_to_df(rows: List[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low",
                                     "close", "volume"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df[["timestamp", "datetime", "open", "high", "low", "close", "volume"]]


def validate_gaps(df: pd.DataFrame, timeframe: str) -> Dict[str, int]:
    """Inspect a sorted candle dataframe for missing bars.

    Returns a dict with `expected_bars`, `actual_bars`, `gap_count`
    (number of consecutive-pair gaps where dt > tf_ms * 1.5),
    `largest_gap_bars` (worst gap, in units of timeframe).
    """
    if df is None or df.empty or "timestamp" not in df.columns:
        return {"expected_bars": 0, "actual_bars": 0,
                "gap_count": 0, "largest_gap_bars": 0}
    tf_ms = _timeframe_ms(timeframe)
    ts = pd.to_numeric(df["timestamp"]).astype("int64").to_numpy()
    if len(ts) < 2:
        return {"expected_bars": int(len(ts)), "actual_bars": int(len(ts)),
                "gap_count": 0, "largest_gap_bars": 0}
    diffs = ts[1:] - ts[:-1]
    expected_total_ms = int(ts[-1] - ts[0])
    expected_bars = int(expected_total_ms // tf_ms) + 1
    actual_bars = int(len(ts))
    # A "gap" is any consecutive pair more than 1.5x apart (allowing some
    # exchange-side latency in candle stamping).
    gap_mask = diffs > int(tf_ms * 1.5)
    gap_count = int(gap_mask.sum())
    largest_gap_bars = int(diffs.max() // tf_ms) if gap_count > 0 else 0
    return {
        "expected_bars": expected_bars,
        "actual_bars": actual_bars,
        "gap_count": gap_count,
        "largest_gap_bars": largest_gap_bars,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _chunk_limit_for(exchange_id: str) -> int:
    """Per-exchange max candles per `fetch_ohlcv` call. Falls back to
    the global `FETCH_CHUNK_LIMIT` when the exchange isn't in the map."""
    return int(config.FETCH_CHUNK_LIMITS.get(exchange_id, config.FETCH_CHUNK_LIMIT))


def _fetch_from_exchange(
    ex_name: str, symbol: str, timeframe: str, days: int,
) -> Optional[pd.DataFrame]:
    """Fetch the full requested history from one exchange. Returns the
    deduped+sorted DataFrame, or None on any error / empty response."""
    try:
        ex = _build_exchange(ex_name)
    except Exception as e:  # noqa: BLE001
        logger.warning("could not build %s: %s", ex_name, e)
        return None
    resolved = _resolve_symbol(ex, symbol)
    if resolved is None:
        logger.info("%s: %s not listed", ex_name, symbol)
        return None
    since_ms = int(
        (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days))
        .timestamp() * 1000
    )
    logger.info("downloading %s %s from %s (%d days, chunk_limit=%d)",
                resolved, timeframe, ex_name, days, _chunk_limit_for(ex_name))
    try:
        rows = _fetch_ohlcv_paginated(
            ex, resolved, timeframe, since_ms, _chunk_limit_for(ex_name),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("paginated fetch failed for %s %s on %s: %s",
                       symbol, timeframe, ex_name, e)
        return None
    if not rows:
        return None
    return _rows_to_df(rows)


def download_symbol(
    symbol: str,
    timeframe: str = config.DEFAULT_TIMEFRAME,
    days: int = config.DEFAULT_HISTORY_DAYS,
    refresh: bool = False,
) -> Path:
    """Download OHLCV for one symbol/timeframe and return the CSV path.

    Strategy:
      1. Try the primary exchange (Binance — supports proper backwards
         pagination at 1000 candles per call).
      2. If the primary returns less history than would cover ~50% of the
         requested `days`, also fetch from the fallback (Kraken — capped
         at ~720 most-recent candles per timeframe) and MERGE the two
         (dedup on timestamp).
      3. Save the merged, sorted, deduped frame.

    Cache hit short-circuits immediately when `refresh=False`.
    """
    utils.assert_paper_only()
    utils.ensure_dirs([config.DATA_RAW_DIR])

    csv_path = utils.csv_path_for(symbol, timeframe)
    if csv_path.exists() and not refresh:
        logger.info("cache hit: %s (%s)", csv_path.name,
                    f"{csv_path.stat().st_size // 1024} KB")
        return csv_path

    tf_ms = _timeframe_ms(timeframe)
    expected_bars = max(int(days * 24 * 3600 * 1000 // tf_ms), 1)
    sufficient_threshold = max(int(expected_bars * 0.5), 1)

    primary_df = _fetch_from_exchange(
        config.PRIMARY_EXCHANGE, symbol, timeframe, days,
    )
    primary_n = 0 if primary_df is None or primary_df.empty else len(primary_df)
    logger.info("primary %s -> %d bars (sufficient threshold = %d)",
                config.PRIMARY_EXCHANGE, primary_n, sufficient_threshold)

    if primary_n >= sufficient_threshold:
        utils.write_df(primary_df, csv_path)
        logger.info("saved %d candles → %s", len(primary_df), csv_path.name)
        return csv_path

    # Primary did not give enough — try fallback.
    fallback_df = _fetch_from_exchange(
        config.FALLBACK_EXCHANGE, symbol, timeframe, days,
    )
    fallback_n = 0 if fallback_df is None or fallback_df.empty else len(fallback_df)
    logger.info("fallback %s -> %d bars", config.FALLBACK_EXCHANGE, fallback_n)

    candidates = [df for df in (primary_df, fallback_df) if df is not None and not df.empty]
    if not candidates:
        raise RuntimeError(
            f"failed to download {symbol} {timeframe} from any exchange "
            f"(primary={config.PRIMARY_EXCHANGE}, "
            f"fallback={config.FALLBACK_EXCHANGE})"
        )
    merged = (pd.concat(candidates, ignore_index=True)
              .drop_duplicates(subset="timestamp")
              .sort_values("timestamp")
              .reset_index(drop=True))
    utils.write_df(merged, csv_path)
    logger.info("saved %d merged candles → %s (primary=%d, fallback=%d)",
                len(merged), csv_path.name, primary_n, fallback_n)
    return csv_path


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
