"""
Funding-only signal engineering.

Inputs (per symbol):
    1. Spot OHLCV from `data_collector.load_candles` (daily resolution).
    2. Funding-rate history from `futures_data_collector.load_funding_rate`
       (8h cadence on Binance USDT-perp; ~4 years of history).

Output: a per-symbol daily signal frame with the documented funding-only
column set, and a stacked cross-asset frame written to
`results/funding_signals.csv`.

LOOKAHEAD RULES — non-negotiable. Every rolling stat is computed with
pandas' default backward window (no `center=True`); every return is
`close[t] / close[t-N] - 1`. The 90d z-score at row `t` only depends on
funding values at `t' <= t`. The classification is a pure function of
the engineered features at the same row.

Funding states (per spec):
    neutral       : nothing extreme is going on
    crowded_long  : z(funding) > +1.5 and 30d return > 0
    capitulation  : z(funding) < -1.5 and 30d return < 0
    recovering    : funding regime was negative but improving and the
                     last week of price action stabilized
    unknown       : not enough rolling history yet (warm-up rows)

This module performs no I/O beyond reading cached CSVs and (optionally)
writing `results/funding_signals.csv`. No network, no API keys, no
private endpoints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import config, data_collector, futures_data_collector as fdc, utils

logger = utils.get_logger("cte.funding_signals")


# ---------------------------------------------------------------------------
# Output schema (locked here so tests can assert it precisely).
# ---------------------------------------------------------------------------
SIGNAL_COLUMNS: List[str] = [
    "timestamp", "symbol", "close",
    "return_1d", "return_7d", "return_30d",
    "funding_rate", "funding_7d_mean", "funding_30d_mean",
    "funding_90d_zscore",
    "extreme_positive_funding", "extreme_negative_funding",
    "funding_trend", "funding_normalization",
    "price_return_7d", "price_return_30d",
    "funding_attractiveness", "funding_improvement",
    "stabilization_score", "crowding_penalty",
    "funding_state",
]

VALID_FUNDING_STATES: List[str] = [
    "neutral", "crowded_long", "capitulation", "recovering", "unknown",
]

# Rolling windows + thresholds — fixed by spec, not tuning knobs.
_ROLL_SHORT = 7
_ROLL_LONG = 30
_ROLL_ZSCORE = 90

_FUNDING_EXTREME_Z = 1.5
_RECOVERING_RETURN_FLAT = 0.05      # |7d return| under this counts as "stable"
_FUNDING_NORMALIZATION_BAND = 1.0   # |z| under 1 = "near-normal"


# ---------------------------------------------------------------------------
# Symbol helpers
# ---------------------------------------------------------------------------
def _spot_symbol(symbol: str) -> str:
    if "/" in symbol:
        return symbol
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}/USDT"
    return symbol


def _futures_symbol(symbol: str) -> str:
    return symbol.replace("/", "")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_spot_daily(symbol: str) -> pd.DataFrame:
    df = data_collector.load_candles(_spot_symbol(symbol), "1d")
    out = df[["timestamp", "close"]].copy()
    out["date"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True).dt.floor("D")
    out = (out.drop_duplicates(subset=["date"], keep="last")
              .sort_values("date").reset_index(drop=True))
    return out


def _load_funding_daily(symbol: str) -> pd.DataFrame:
    f = fdc.load_funding_rate(_futures_symbol(symbol))
    if f.empty:
        return pd.DataFrame(columns=["date", "funding_rate"])
    f = f.copy()
    f["date"] = pd.to_datetime(f["funding_time"], unit="ms", utc=True).dt.floor("D")
    daily = (f.groupby("date", as_index=False)["funding_rate"].mean()
              .sort_values("date").reset_index(drop=True))
    return daily


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _backward_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def _classify_state(row: pd.Series) -> str:
    """Pure function of the row's features. Only inputs that exist at or
    before timestamp `t` are touched (the row itself is built from
    backward-rolling stats), so this is lookahead-free by construction.
    """
    fz = row.get("funding_90d_zscore")
    f30 = row.get("funding_30d_mean")
    f_trend = row.get("funding_trend")
    r30 = row.get("return_30d")
    r7 = row.get("return_7d")

    # Need return_30d, return_7d, AND a funding 30d mean for any rule
    # other than crowded/capitulation (those need fz too). Without these
    # the row is in warm-up territory.
    if pd.isna(r30) or pd.isna(r7) or pd.isna(f30):
        return "unknown"

    # crowded_long and capitulation need fz; if fz is NaN we can still
    # return a non-`unknown` state via the recovering/neutral branches.
    if not pd.isna(fz):
        if float(fz) > _FUNDING_EXTREME_Z and float(r30) > 0:
            return "crowded_long"
        if float(fz) < -_FUNDING_EXTREME_Z and float(r30) < 0:
            return "capitulation"

    # Recovering: prior funding regime was negative on average AND the
    # 7d mean is improving (less negative than 30d mean) AND the last
    # week of price was stable.
    if (float(f30) < 0
            and not pd.isna(f_trend) and float(f_trend) > 0
            and abs(float(r7)) < _RECOVERING_RETURN_FLAT):
        return "recovering"
    return "neutral"


def compute_funding_signals_for_symbol(symbol: str) -> pd.DataFrame:
    """Build the per-symbol daily funding-signal table. The schema is
    fixed; missing inputs surface as NaN columns + `unknown` state."""
    sym = _futures_symbol(symbol)
    try:
        spot = _load_spot_daily(symbol)
    except FileNotFoundError:
        logger.warning("spot OHLCV missing for %s — skipping funding signals", sym)
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    if spot.empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

    funding = _load_funding_daily(symbol)
    df = spot[["date", "timestamp", "close"]].copy()
    df = df.merge(funding, on="date", how="left")

    # Returns
    df["return_1d"] = df["close"].pct_change(1)
    df["return_7d"] = df["close"].pct_change(_ROLL_SHORT)
    df["return_30d"] = df["close"].pct_change(_ROLL_LONG)
    # Aliases — kept distinct for the documented schema even if values
    # are identical to return_*; downstream code can rely on the names.
    df["price_return_7d"] = df["return_7d"]
    df["price_return_30d"] = df["return_30d"]

    # Funding rolling stats
    df["funding_7d_mean"] = df["funding_rate"].rolling(
        _ROLL_SHORT, min_periods=_ROLL_SHORT).mean()
    df["funding_30d_mean"] = df["funding_rate"].rolling(
        _ROLL_LONG, min_periods=_ROLL_LONG).mean()
    df["funding_90d_zscore"] = _backward_zscore(df["funding_rate"], _ROLL_ZSCORE)

    df["extreme_positive_funding"] = (
        df["funding_90d_zscore"] > _FUNDING_EXTREME_Z).fillna(False)
    df["extreme_negative_funding"] = (
        df["funding_90d_zscore"] < -_FUNDING_EXTREME_Z).fillna(False)

    # Trend = 7d mean minus 30d mean. Positive = funding rising (worse
    # for longs); negative = funding falling (better for longs).
    df["funding_trend"] = df["funding_7d_mean"] - df["funding_30d_mean"]

    # Normalization = how close current funding is to the 90d mean,
    # in [0,1] where 1 means within ±1σ ("normal") and 0 means very far.
    fz_abs = df["funding_90d_zscore"].abs()
    df["funding_normalization"] = (1.0 - (fz_abs / 3.0).clip(0.0, 1.0))

    # Score components used downstream by the rotation strategy.
    # funding_attractiveness: longs prefer NEGATIVE funding. Map negative
    # 30d mean to a positive score.
    f30 = df["funding_30d_mean"]
    df["funding_attractiveness"] = ((-f30).clip(lower=0.0) / 0.001).clip(0.0, 1.0)
    # funding_improvement: 7d mean dropping below 30d mean is good for
    # longs. Map a NEGATIVE funding_trend to a positive score.
    f_trend = df["funding_trend"]
    df["funding_improvement"] = ((-f_trend).clip(lower=0.0) / 0.0005).clip(0.0, 1.0)
    # stabilization_score: small |7d return| → high score. 0 when |r7|
    # >= the flat threshold; 1 when r7 == 0.
    r7 = df["return_7d"]
    df["stabilization_score"] = (1.0 - (r7.abs() / _RECOVERING_RETURN_FLAT)
                                  ).clip(0.0, 1.0).fillna(0.0)
    # crowding_penalty: positive funding z scaled to [0,1] — used by the
    # composite score to subtract from crowded longs.
    fz = df["funding_90d_zscore"]
    df["crowding_penalty"] = (fz.clip(lower=0.0) / 3.0).clip(0.0, 1.0).fillna(0.0)

    df["symbol"] = sym
    df["funding_state"] = df.apply(_classify_state, axis=1)

    out = df.reindex(columns=SIGNAL_COLUMNS)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cross-asset
# ---------------------------------------------------------------------------
def compute_all_funding_signals(
    symbols: Sequence[str] = fdc.DEFAULT_FUTURES_SYMBOLS,
    save: bool = True,
) -> pd.DataFrame:
    """Build per-symbol funding-signal tables and stack them. Symbols
    with no spot data yield 0 rows and are logged."""
    frames: List[pd.DataFrame] = []
    for sym in symbols:
        try:
            f = compute_funding_signals_for_symbol(sym)
        except Exception as e:  # noqa: BLE001
            logger.warning("funding signals failed for %s: %s", sym, e)
            continue
        if f.empty:
            logger.info("no funding signal rows for %s (missing inputs)", sym)
            continue
        frames.append(f)
    out = (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame(columns=SIGNAL_COLUMNS))
    if save:
        utils.write_df(out, config.RESULTS_DIR / "funding_signals.csv")
    return out
