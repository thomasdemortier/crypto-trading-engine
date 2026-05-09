"""
Market-structure daily signals.

Joins:
  * BTC/USDT daily spot OHLCV (cached by `data_collector.load_candles`).
  * The 9 alt assets in the configured allocator universe (used to compute
    breadth and an equal-weight alt basket).
  * DefiLlama total TVL.
  * DefiLlama stablecoin supply total.
  * Blockchain.com BTC market cap, hash rate, transactions/day.

Output: `results/market_structure_signals.csv` with the 28-column schema
documented in the spec, plus a `market_structure_state` classification.

LOOKAHEAD RULES — non-negotiable. Every rolling window is backward
(`min_periods=window`, no `center=True`). Every return is
`x[t]/x[t-N] - 1`. Every state at row `t` is a pure function of features
at the same row, which themselves only depend on data with timestamp ≤ t.
A `partial_vs_full` test verifies this explicitly.

States:
    alt_risk_on    : stablecoin supply 90d > 0, total TVL 90d > 0,
                      alt basket 90d > BTC 90d, alt breadth ≥ 50 %.
    btc_leadership : BTC 90d > alt basket 90d, BTC > 200d MA,
                      alt breadth < 50 %.
    defensive      : BTC < 200d MA, OR (TVL 90d < 0 AND stables 90d < 0),
                      OR alt breadth < 35 %.
    neutral        : nothing extreme fired.
    unknown        : any required input is NaN at row t.

Hard rules: no live trading, no broker code, no API keys. Pure feature
engineering on cached CSVs.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import (config, data_collector,
                market_structure_data_collector as mdc, utils)

logger = utils.get_logger("cte.market_structure_signals")


# ---------------------------------------------------------------------------
# Schema (locked here so tests can assert it)
# ---------------------------------------------------------------------------
SIGNAL_COLUMNS: List[str] = [
    "timestamp", "date", "btc_close",
    "btc_return_30d", "btc_return_90d", "btc_above_200d_ma",
    "total_tvl", "total_tvl_return_30d", "total_tvl_return_90d",
    "stablecoin_supply",
    "stablecoin_supply_return_30d", "stablecoin_supply_return_90d",
    "btc_market_cap", "btc_market_cap_return_30d",
    "btc_hash_rate", "btc_hash_rate_return_30d",
    "btc_transactions", "btc_transactions_return_30d",
    "alt_basket_return_30d", "alt_basket_return_90d",
    "alt_basket_above_200d_ma_pct",
    "alt_basket_vs_btc_30d", "alt_basket_vs_btc_90d",
    "liquidity_score", "onchain_health_score",
    "alt_risk_score", "defensive_score",
    "market_structure_state",
]

VALID_STATES: List[str] = [
    "alt_risk_on", "btc_leadership", "defensive", "neutral", "unknown",
]

# Universe ordering matches the v1 expanded universe; alts = universe minus BTC.
ALLOCATOR_UNIVERSE: List[str] = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT",
    "XRP/USDT", "DOGE/USDT", "ADA/USDT", "LTC/USDT", "BNB/USDT",
]
ALT_UNIVERSE: List[str] = [a for a in ALLOCATOR_UNIVERSE if a != "BTC/USDT"]

_RET_SHORT_DAYS = 30
_RET_LONG_DAYS = 90
_TREND_DAYS = 200

# Thresholds — fixed by spec, not tuning knobs.
_BREADTH_RISK_ON = 0.50
_BREADTH_DEFENSIVE = 0.35


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_daily_close(symbol: str) -> pd.DataFrame:
    df = data_collector.load_candles(symbol, "1d")
    out = df[["timestamp", "close"]].copy()
    out["date"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True
                                  ).dt.floor("D")
    out = (out.drop_duplicates(subset=["date"], keep="last")
              .sort_values("date").reset_index(drop=True))
    return out[["date", "close"]].rename(columns={"close": symbol})


def _load_market_structure_series(loader, value_alias: str) -> pd.DataFrame:
    df = loader()
    if df.empty:
        return pd.DataFrame(columns=["date", value_alias])
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True
                                 ).dt.floor("D")
    df = (df.groupby("date", as_index=False)["value"].mean()
              .sort_values("date").reset_index(drop=True))
    return df.rename(columns={"value": value_alias})


def _build_alt_basket_close(alt_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight alt basket built as the row-wise mean of normalised
    alt closes (each alt's first available close = 1.0). The basket
    series is a synthetic price-index, NOT a return series."""
    parts: List[pd.Series] = []
    for sym, df in alt_frames.items():
        if df is None or df.empty:
            continue
        s = df.set_index("date")[sym].astype(float)
        first = s.dropna().iloc[0] if not s.dropna().empty else np.nan
        if not np.isfinite(first) or first <= 0:
            continue
        parts.append((s / first).rename(sym))
    if not parts:
        return pd.DataFrame(columns=["date", "alt_basket_close"])
    df = pd.concat(parts, axis=1).sort_index()
    df["alt_basket_close"] = df.mean(axis=1, skipna=True)
    out = df[["alt_basket_close"]].dropna(how="all").reset_index()
    return out


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _backward_pct_change(series: pd.Series, n: int) -> pd.Series:
    """Strict backward percent change with no `fill_method` surprises."""
    if len(series) <= n:
        return pd.Series([np.nan] * len(series), index=series.index)
    shifted = series.shift(n)
    return (series / shifted) - 1.0


def _backward_sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


def _classify_row(row: pd.Series) -> str:
    needed = [
        "btc_close", "btc_return_90d", "btc_above_200d_ma",
        "total_tvl_return_90d", "stablecoin_supply_return_90d",
        "alt_basket_return_90d", "alt_basket_above_200d_ma_pct",
    ]
    for k in needed:
        v = row.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "unknown"

    btc_above = bool(row["btc_above_200d_ma"])
    tvl_90d = float(row["total_tvl_return_90d"])
    stables_90d = float(row["stablecoin_supply_return_90d"])
    alt_90d = float(row["alt_basket_return_90d"])
    btc_90d = float(row["btc_return_90d"])
    breadth = float(row["alt_basket_above_200d_ma_pct"])

    # Defensive overrides everything else if any of its triggers fire.
    if (not btc_above
            or (tvl_90d < 0 and stables_90d < 0)
            or breadth < _BREADTH_DEFENSIVE):
        return "defensive"
    # Risk-on requires positive liquidity AND alt outperformance AND breadth.
    if (stables_90d > 0 and tvl_90d > 0
            and alt_90d > btc_90d and breadth >= _BREADTH_RISK_ON):
        return "alt_risk_on"
    # BTC leadership when BTC clearly leads alts.
    if btc_90d > alt_90d and btc_above and breadth < _BREADTH_RISK_ON:
        return "btc_leadership"
    return "neutral"


def compute_market_structure_signals(
    universe: Sequence[str] = ALLOCATOR_UNIVERSE,
    save: bool = True,
) -> pd.DataFrame:
    """Build the daily market-structure signal table. Returns an empty
    documented frame if BTC OHLCV is missing — never raises."""
    try:
        btc = _load_daily_close("BTC/USDT").rename(columns={"BTC/USDT": "btc_close"})
    except FileNotFoundError:
        logger.warning("BTC/USDT daily missing — cannot compute signals")
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    if btc.empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

    alt_frames: Dict[str, pd.DataFrame] = {}
    alt_above_ma_flags = {}
    for sym in (s for s in universe if s != "BTC/USDT"):
        try:
            alt_df = _load_daily_close(sym)
        except FileNotFoundError:
            logger.info("alt missing: %s", sym)
            continue
        if alt_df.empty:
            continue
        alt_frames[sym] = alt_df
        # Track each alt's "above 200d MA" boolean for the breadth metric.
        c = alt_df.set_index("date")[sym].astype(float)
        ma = _backward_sma(c, _TREND_DAYS)
        alt_above_ma_flags[sym] = (c > ma).rename(sym)

    alt_basket = _build_alt_basket_close(alt_frames)

    # Market-structure series.
    tvl = _load_market_structure_series(mdc.load_total_tvl, "total_tvl")
    stables = _load_market_structure_series(mdc.load_stablecoin_supply,
                                              "stablecoin_supply")
    btc_cap = _load_market_structure_series(mdc.load_btc_market_cap,
                                              "btc_market_cap")
    btc_hash = _load_market_structure_series(mdc.load_btc_hash_rate,
                                               "btc_hash_rate")
    btc_tx = _load_market_structure_series(mdc.load_btc_transactions,
                                              "btc_transactions")

    df = btc.copy()
    for piece in (alt_basket, tvl, stables, btc_cap, btc_hash, btc_tx):
        if not piece.empty:
            df = df.merge(piece, on="date", how="left")
    # Ensure expected columns exist even if a source is missing.
    for col in ("alt_basket_close", "total_tvl", "stablecoin_supply",
                 "btc_market_cap", "btc_hash_rate", "btc_transactions"):
        if col not in df.columns:
            df[col] = np.nan

    # BTC features
    df["btc_return_30d"] = _backward_pct_change(df["btc_close"], _RET_SHORT_DAYS)
    df["btc_return_90d"] = _backward_pct_change(df["btc_close"], _RET_LONG_DAYS)
    df["btc_above_200d_ma"] = (
        df["btc_close"] > _backward_sma(df["btc_close"], _TREND_DAYS)
    ).fillna(False)

    # Market-structure data features
    df["total_tvl_return_30d"] = _backward_pct_change(df["total_tvl"], _RET_SHORT_DAYS)
    df["total_tvl_return_90d"] = _backward_pct_change(df["total_tvl"], _RET_LONG_DAYS)
    df["stablecoin_supply_return_30d"] = _backward_pct_change(
        df["stablecoin_supply"], _RET_SHORT_DAYS)
    df["stablecoin_supply_return_90d"] = _backward_pct_change(
        df["stablecoin_supply"], _RET_LONG_DAYS)
    df["btc_market_cap_return_30d"] = _backward_pct_change(
        df["btc_market_cap"], _RET_SHORT_DAYS)
    df["btc_hash_rate_return_30d"] = _backward_pct_change(
        df["btc_hash_rate"], _RET_SHORT_DAYS)
    df["btc_transactions_return_30d"] = _backward_pct_change(
        df["btc_transactions"], _RET_SHORT_DAYS)

    # Alt basket features
    df["alt_basket_return_30d"] = _backward_pct_change(
        df["alt_basket_close"], _RET_SHORT_DAYS)
    df["alt_basket_return_90d"] = _backward_pct_change(
        df["alt_basket_close"], _RET_LONG_DAYS)
    df["alt_basket_vs_btc_30d"] = (
        df["alt_basket_return_30d"] - df["btc_return_30d"]
    )
    df["alt_basket_vs_btc_90d"] = (
        df["alt_basket_return_90d"] - df["btc_return_90d"]
    )

    # Breadth: % of alts above their own 200d MA at row t.
    if alt_above_ma_flags:
        breadth_df = pd.concat(alt_above_ma_flags.values(), axis=1).sort_index()
        breadth_series = breadth_df.mean(axis=1, skipna=True).rename(
            "alt_basket_above_200d_ma_pct").reset_index()
        df = df.merge(breadth_series, on="date", how="left")
    else:
        df["alt_basket_above_200d_ma_pct"] = np.nan

    # Composite scores in [-1, 1] / [0, 1].
    liq = (df["total_tvl_return_90d"].fillna(0.0).clip(-0.5, 0.5) / 0.5
            + df["stablecoin_supply_return_90d"].fillna(0.0).clip(-0.5, 0.5) / 0.5)
    df["liquidity_score"] = (liq / 2.0).clip(-1.0, 1.0)

    onchain = (df["btc_hash_rate_return_30d"].fillna(0.0).clip(-0.5, 0.5) / 0.5
                 + df["btc_transactions_return_30d"].fillna(0.0).clip(-0.5, 0.5) / 0.5)
    df["onchain_health_score"] = (onchain / 2.0).clip(-1.0, 1.0)

    alt_risk = (df["alt_basket_vs_btc_90d"].fillna(0.0).clip(-0.5, 0.5) / 0.5
                  + df["alt_basket_above_200d_ma_pct"].fillna(0.0)
                    .clip(0.0, 1.0))
    df["alt_risk_score"] = (alt_risk / 2.0).clip(-1.0, 1.0)

    defensive = ((~df["btc_above_200d_ma"].astype(bool)).astype(float)
                   + (df["total_tvl_return_90d"].fillna(0.0) < 0).astype(float)
                   + (df["alt_basket_above_200d_ma_pct"].fillna(1.0)
                      < _BREADTH_DEFENSIVE).astype(float))
    df["defensive_score"] = (defensive / 3.0).clip(0.0, 1.0)

    # Force nanosecond precision before int64 conversion — the merged
    # `date` column may carry datetime64[ms, UTC] precision, in which
    # case `astype("int64")` returns ms-since-epoch and dividing by 10**6
    # again (silently) produces kiloseconds. tz_convert preserves the UTC
    # value while letting us reset the unit to nanoseconds cleanly.
    _dates_utc = pd.to_datetime(df["date"], utc=True)
    _dates_ns = _dates_utc.dt.tz_convert("UTC").dt.tz_localize(None
                                  ).astype("datetime64[ns]")
    df["timestamp"] = (_dates_ns.astype("int64") // 10**6).astype("int64")
    df["date"] = _dates_utc.dt.strftime("%Y-%m-%d")
    df["market_structure_state"] = df.apply(_classify_row, axis=1)

    out = df.reindex(columns=SIGNAL_COLUMNS).reset_index(drop=True)
    if save:
        utils.write_df(out, config.RESULTS_DIR / "market_structure_signals.csv")
    return out
