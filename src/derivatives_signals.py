"""
Derivatives signal engineering: funding rate + open interest + spot.

Inputs (per symbol):
    1. Spot OHLCV from `data_collector.load_candles` (daily resolution).
    2. Funding-rate history from `futures_data_collector.load_funding_rate`
       (8-hour cadence on Binance USDT-perp).
    3. Open-interest history from `futures_data_collector.load_open_interest`
       (1-day cadence; capped at ~30 days by Binance's public endpoint).

Output: a per-symbol daily signal frame with the column set documented
in the Phase 2 spec, plus a cross-asset stacked frame written to
`results/derivatives_signals.csv`.

LOOKAHEAD RULES — these are non-negotiable. Every rolling calculation
uses pandas' default backward window (no `center=True`). All returns are
`close[t] / close[t-N] - 1`, never forward-shifted. Z-scores are
computed on a 90-day backward window so that the value at row `t`
depends only on data at or before `t`.

Signal states (assigned from a current-snapshot snapshot of the engineered
features — no future information used):
    healthy_trend : trending up, OI growing, funding not extreme
    crowded_long  : high crowding score + price-up & OI-up
    capitulation  : high capitulation score + price-down & OI-down
    deleveraging  : OI dropping, price ~flat (unwind without panic)
    neutral       : nothing fires
    unknown       : not enough rolling history yet (warm-up rows)

This module performs no I/O beyond reading the existing cached CSVs and
writing `results/derivatives_signals.csv` (only when `compute_all`'s
`save=True`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import config, data_collector, futures_data_collector as fdc, utils

logger = utils.get_logger("cte.derivatives_signals")


# ---------------------------------------------------------------------------
# Output schema (kept here so tests can assert it precisely).
# ---------------------------------------------------------------------------
SIGNAL_COLUMNS: List[str] = [
    "timestamp", "symbol", "close",
    "return_1d", "return_7d", "return_30d",
    "funding_rate", "funding_rate_7d_mean", "funding_rate_30d_mean",
    "funding_rate_zscore_90d",
    "funding_extreme_positive", "funding_extreme_negative",
    "open_interest",
    "open_interest_7d_change_pct", "open_interest_30d_change_pct",
    "open_interest_zscore_90d",
    "price_up_oi_up", "price_up_oi_down",
    "price_down_oi_up", "price_down_oi_down",
    "crowding_score", "capitulation_score", "squeeze_risk_score",
    "signal_state",
]

VALID_SIGNAL_STATES: List[str] = [
    "healthy_trend", "crowded_long", "capitulation",
    "deleveraging", "neutral", "unknown",
]

# Rolling window sizes (in trading days). These are deterministic — they
# are NOT tuning knobs. Don't change them to chase a PASS.
_ROLL_SHORT = 7
_ROLL_LONG = 30
_ROLL_ZSCORE = 90

# Thresholds — also fixed by spec, not tunable.
_FUNDING_EXTREME_Z = 1.5
_CROWDED_THRESHOLD = 0.65
_CAPITULATION_THRESHOLD = 0.65
_DELEVERAGE_OI_DROP = -0.10
_DELEVERAGE_RETURN_FLAT = 0.05
_HEALTHY_FUNDING_Z_MAX = 1.0


# ---------------------------------------------------------------------------
# Symbol helpers
# ---------------------------------------------------------------------------
def _spot_symbol(symbol: str) -> str:
    """`BTCUSDT` -> `BTC/USDT`. Idempotent."""
    if "/" in symbol:
        return symbol
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}/USDT"
    return symbol


def _futures_symbol(symbol: str) -> str:
    return symbol.replace("/", "")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_spot_daily(symbol: str) -> pd.DataFrame:
    """Returns a daily-indexed close series with columns timestamp, close."""
    df = data_collector.load_candles(_spot_symbol(symbol), "1d")
    out = df[["timestamp", "close"]].copy()
    out["date"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True).dt.floor("D")
    out = (out.drop_duplicates(subset=["date"], keep="last")
              .sort_values("date").reset_index(drop=True))
    return out


def _load_funding_daily(symbol: str) -> pd.DataFrame:
    """Resample 8h funding to a daily mean indexed by UTC date."""
    f = fdc.load_funding_rate(_futures_symbol(symbol))
    if f.empty:
        return pd.DataFrame(columns=["date", "funding_rate"])
    f = f.copy()
    f["date"] = pd.to_datetime(f["funding_time"], unit="ms", utc=True).dt.floor("D")
    daily = (f.groupby("date", as_index=False)["funding_rate"].mean()
              .sort_values("date").reset_index(drop=True))
    return daily


def _load_oi_daily(symbol: str) -> pd.DataFrame:
    o = fdc.load_open_interest(_futures_symbol(symbol))
    if o.empty:
        return pd.DataFrame(columns=["date", "open_interest"])
    o = o.copy()
    o["date"] = pd.to_datetime(o["timestamp"], unit="ms", utc=True).dt.floor("D")
    daily = (o.groupby("date", as_index=False)["open_interest"].mean()
               .sort_values("date").reset_index(drop=True))
    return daily


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _backward_zscore(series: pd.Series, window: int) -> pd.Series:
    """Backward-rolling z-score. NaN until `window` valid values exist."""
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    z = (series - mean) / std.replace(0.0, np.nan)
    return z


def _classify_state(row: pd.Series) -> str:
    """Pure function of the feature columns — no lookahead by construction.

    Each rule only consults the inputs it actually needs, so a row with a
    finite OI series but a degenerate (constant) funding series can still
    classify as `deleveraging` or `capitulation` if those rules' inputs
    are well-defined.
    """
    oi30 = row.get("open_interest_30d_change_pct")
    r30 = row.get("return_30d")
    r7 = row.get("return_7d")
    fz = row.get("funding_rate_zscore_90d")

    # Without OI and 7d/30d returns we cannot say anything.
    if pd.isna(oi30) or pd.isna(r30) or pd.isna(r7):
        return "unknown"

    crowding = float(row.get("crowding_score", 0.0))
    capit = float(row.get("capitulation_score", 0.0))
    if crowding >= _CROWDED_THRESHOLD and bool(row.get("price_up_oi_up", False)):
        return "crowded_long"
    if capit >= _CAPITULATION_THRESHOLD and bool(row.get("price_down_oi_down", False)):
        return "capitulation"
    # `healthy_trend` references funding extremity, so it requires fz.
    if (float(r30) > 0 and float(oi30) > 0 and not pd.isna(fz)
            and abs(float(fz)) < _HEALTHY_FUNDING_Z_MAX):
        return "healthy_trend"
    # `deleveraging` is a pure OI/price unwind — funding not required.
    if (float(oi30) < _DELEVERAGE_OI_DROP
            and abs(float(r7)) < _DELEVERAGE_RETURN_FLAT):
        return "deleveraging"
    return "neutral"


def compute_signals_for_symbol(symbol: str) -> pd.DataFrame:
    """Build the per-symbol daily signal table. Always returns the documented
    schema, even when funding or OI are missing — missing inputs surface as
    NaN columns and `signal_state == 'unknown'`."""
    sym_disp = _futures_symbol(symbol)
    try:
        spot = _load_spot_daily(symbol)
    except FileNotFoundError:
        logger.warning("spot OHLCV missing for %s — skipping signals", sym_disp)
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    if spot.empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

    funding = _load_funding_daily(symbol)
    oi = _load_oi_daily(symbol)

    df = spot[["date", "timestamp", "close"]].copy()
    df = df.merge(funding, on="date", how="left")
    df = df.merge(oi, on="date", how="left")

    # Returns
    df["return_1d"] = df["close"].pct_change(1)
    df["return_7d"] = df["close"].pct_change(_ROLL_SHORT)
    df["return_30d"] = df["close"].pct_change(_ROLL_LONG)

    # Funding rolling stats
    df["funding_rate_7d_mean"] = df["funding_rate"].rolling(
        _ROLL_SHORT, min_periods=_ROLL_SHORT).mean()
    df["funding_rate_30d_mean"] = df["funding_rate"].rolling(
        _ROLL_LONG, min_periods=_ROLL_LONG).mean()
    df["funding_rate_zscore_90d"] = _backward_zscore(
        df["funding_rate"], _ROLL_ZSCORE)

    df["funding_extreme_positive"] = (
        df["funding_rate_zscore_90d"] > _FUNDING_EXTREME_Z).fillna(False)
    df["funding_extreme_negative"] = (
        df["funding_rate_zscore_90d"] < -_FUNDING_EXTREME_Z).fillna(False)

    # OI changes
    df["open_interest_7d_change_pct"] = df["open_interest"].pct_change(_ROLL_SHORT)
    df["open_interest_30d_change_pct"] = df["open_interest"].pct_change(_ROLL_LONG)
    df["open_interest_zscore_90d"] = _backward_zscore(
        df["open_interest"], _ROLL_ZSCORE)

    # Price/OI joint state — uses 1d return and 1d OI delta.
    oi_1d_change = df["open_interest"].pct_change(1)
    df["price_up_oi_up"] = ((df["return_1d"] > 0) & (oi_1d_change > 0)).fillna(False)
    df["price_up_oi_down"] = ((df["return_1d"] > 0) & (oi_1d_change < 0)).fillna(False)
    df["price_down_oi_up"] = ((df["return_1d"] < 0) & (oi_1d_change > 0)).fillna(False)
    df["price_down_oi_down"] = ((df["return_1d"] < 0) & (oi_1d_change < 0)).fillna(False)

    # Composite scores (each in [0, 1]).
    fz = df["funding_rate_zscore_90d"]
    oi30 = df["open_interest_30d_change_pct"]
    r7 = df["return_7d"]

    crowding_funding = (fz.clip(lower=0) / 3.0).clip(0, 1)
    crowding_oi = (oi30.clip(lower=0) / 0.5).clip(0, 1)
    crowding_ret = (r7.clip(lower=0) / 0.2).clip(0, 1)
    df["crowding_score"] = (0.5 * crowding_funding
                            + 0.3 * crowding_oi
                            + 0.2 * crowding_ret).fillna(0.0)

    capit_funding = ((-fz).clip(lower=0) / 3.0).clip(0, 1)
    capit_oi = ((-oi30).clip(lower=0) / 0.5).clip(0, 1)
    capit_ret = ((-r7).clip(lower=0) / 0.2).clip(0, 1)
    df["capitulation_score"] = (0.5 * capit_funding
                                + 0.3 * capit_oi
                                + 0.2 * capit_ret).fillna(0.0)

    # Squeeze risk: high funding magnitude + price moving against the
    # crowded side. Long-squeeze risk: very positive funding (longs paying)
    # while price falls. Short-squeeze risk: very negative funding while
    # price rallies. Either is a 1.0; take the max.
    long_squeeze = ((fz.clip(lower=0) / 3.0).clip(0, 1)
                    * (-r7).clip(lower=0).div(0.1).clip(0, 1))
    short_squeeze = (((-fz).clip(lower=0) / 3.0).clip(0, 1)
                     * r7.clip(lower=0).div(0.1).clip(0, 1))
    df["squeeze_risk_score"] = pd.concat(
        [long_squeeze, short_squeeze], axis=1).max(axis=1).fillna(0.0)

    df["symbol"] = sym_disp
    df["signal_state"] = df.apply(_classify_state, axis=1)

    out = df.reindex(columns=SIGNAL_COLUMNS)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cross-asset
# ---------------------------------------------------------------------------
def compute_all_derivatives_signals(
    symbols: Sequence[str] = fdc.DEFAULT_FUTURES_SYMBOLS,
    save: bool = True,
) -> pd.DataFrame:
    """Build per-symbol signal tables and stack them. Symbols with no
    spot data (or no funding) still produce 0-row contributions and are
    logged."""
    frames: List[pd.DataFrame] = []
    for sym in symbols:
        try:
            f = compute_signals_for_symbol(sym)
        except Exception as e:  # noqa: BLE001
            logger.warning("derivatives signals failed for %s: %s", sym, e)
            continue
        if f.empty:
            logger.info("no signal rows for %s (missing inputs)", sym)
            continue
        frames.append(f)
    if not frames:
        out = pd.DataFrame(columns=SIGNAL_COLUMNS)
    else:
        out = pd.concat(frames, ignore_index=True)
    if save:
        utils.write_df(out, config.RESULTS_DIR / "derivatives_signals.csv")
    return out
