"""
Funding-rate + futures-basis signal engineering.

Builds a daily, lookahead-free signal DataFrame per asset from:

    * Spot OHLCV (the project's existing 1d cache).
    * Binance funding rate history (paginated 8h rate).
    * Binance mark + index klines (1d) — basis = (mark - index) / index.
    * Bybit funding (cross-checked, blended into a robust mean of free
      sources).

Hard rules:
    * Backward-only rolling windows: every roller is `min_periods=window`
      and runs on the closed-bar series — no forward fill of raw funding
      across days.
    * No `center=True`. No `shift(-N)`. No future-data joins. Cross-day
      joins are merge_asof on the funding/basis "as-of-or-before" the
      spot bar's open.
    * Thresholds are LOCKED in `FundingBasisSignalConfig` defaults and
      MUST NOT be tuned after seeing strategy performance.

Output schema (one row per asset per spot bar):
    timestamp, datetime, asset, close, above_200dma,
    funding_1d_avg, funding_7d_avg, funding_30d_avg,
    funding_z90, funding_pct_rank_365,
    funding_trend_pos, basis,
    basis_7d_avg, basis_30d_avg, basis_z90,
    price_return_30d, price_return_90d,
    realised_vol_30d, crowding_score,
    carry_attractiveness, regime_state
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import (config, funding_basis_data_collector as fbdc, portfolio_backtester as pb,
                utils)

logger = utils.get_logger("cte.funding_basis_signals")


# ---------------------------------------------------------------------------
# Locked thresholds — edits after running the strategy = retuning. Don't.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FundingBasisSignalConfig:
    # Window sizes (days) for rollers. All daily-resampled.
    funding_short_days: int = 7
    funding_mid_days: int = 30
    funding_z_days: int = 90
    funding_pct_rank_days: int = 365
    basis_short_days: int = 7
    basis_mid_days: int = 30
    basis_z_days: int = 90
    price_short_days: int = 30
    price_long_days: int = 90
    ma_days: int = 200
    rv_days: int = 30

    # Regime thresholds — locked.
    funding_high_z90: float = 1.5         # crowded-long contributor
    funding_extreme_z90: float = 2.0      # very crowded
    funding_low_z90: float = -0.5         # cheap-bullish contributor
    basis_high_z90: float = 1.0           # stretched basis
    basis_low_z90: float = -1.0
    price_extended_30d_pct: float = 10.0  # late-rally guard
    realised_vol_high_annualised: float = 1.0  # 100 % vol = stress

    # Combined scores.
    crowding_funding_weight: float = 0.5
    crowding_basis_weight: float = 0.3
    crowding_price_weight: float = 0.2

    # Carry attractiveness: lower funding z90 = better; longer-trend
    # positive return = better.
    carry_funding_weight: float = 0.6
    carry_trend_weight: float = 0.4


# Regime state strings — exported for the strategy.
STATE_CROWDED_LONG = "crowded_long"
STATE_NEUTRAL_RISK_ON = "neutral_risk_on"
STATE_CHEAP_BULLISH = "cheap_bullish"
STATE_STRESS_NEGATIVE_FUNDING = "stress_negative_funding"
STATE_DEFENSIVE = "defensive"
STATE_UNKNOWN = "unknown"

REGIME_STATES: List[str] = [
    STATE_CROWDED_LONG, STATE_NEUTRAL_RISK_ON, STATE_CHEAP_BULLISH,
    STATE_STRESS_NEGATIVE_FUNDING, STATE_DEFENSIVE, STATE_UNKNOWN,
]


# ---------------------------------------------------------------------------
# Asof-or-before merge — never lookahead.
# ---------------------------------------------------------------------------
def _merge_asof_at_open(
    spot: pd.DataFrame, side_df: pd.DataFrame,
    side_col: str, *, suffix: str,
) -> pd.DataFrame:
    """Merge `side_df[side_col]` onto `spot` by `timestamp` using
    backward (as-of-or-before) join, so that the value attached to spot
    bar t is the most recent value with `timestamp <= t.open`."""
    if spot.empty or side_df.empty or side_col not in side_df.columns:
        spot = spot.copy()
        spot[f"{suffix}"] = np.nan
        return spot
    left = spot.sort_values("timestamp").copy()
    right = side_df[["timestamp", side_col]].sort_values("timestamp").copy()
    merged = pd.merge_asof(left, right, on="timestamp", direction="backward")
    merged.rename(columns={side_col: suffix}, inplace=True)
    return merged


def _resample_funding_to_daily(funding_df: pd.DataFrame) -> pd.DataFrame:
    """Daily mean of funding (over all observations whose timestamp falls
    within the calendar day, UTC). The resulting timestamp is shifted to
    the END of the calendar day so a backward merge_asof onto a spot bar
    at 00:00 cannot peek at the same-day 08:00 / 16:00 funding ticks.

    Returns columns timestamp + funding_day."""
    if funding_df.empty:
        return pd.DataFrame(columns=["timestamp", "funding_day"])
    f = funding_df.copy()
    f["datetime"] = pd.to_datetime(f["timestamp"], unit="ms", utc=True)
    daily = (f.set_index("datetime")["funding_rate"]
              .resample("1D").mean()
              .dropna())
    # Label each daily mean at the END of its day (start_of_next_day - 1ms)
    # so the value only becomes visible to a spot bar AFTER the day ends.
    end_index = daily.index + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    out = pd.DataFrame({
        "timestamp": (end_index.astype("int64") // 10**6),
        "funding_day": daily.values,
    })
    return out.reset_index(drop=True)


def _basis_from_klines(mark_df: pd.DataFrame,
                        index_df: pd.DataFrame) -> pd.DataFrame:
    """Daily basis = (mark.close - index.close) / index.close. Joined on
    timestamp; rows missing on either side are dropped."""
    if mark_df.empty or index_df.empty:
        return pd.DataFrame(columns=["timestamp", "basis"])
    a = mark_df[["timestamp", "close"]].rename(columns={"close": "mark_close"})
    b = index_df[["timestamp", "close"]].rename(columns={"close": "index_close"})
    j = a.merge(b, on="timestamp", how="inner").dropna()
    j["basis"] = (j["mark_close"] - j["index_close"]) / j["index_close"]
    return j[["timestamp", "basis"]].sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-asset signal builder
# ---------------------------------------------------------------------------
def build_signals_for_asset(
    asset: str, spot_df: pd.DataFrame,
    funding_frames: Sequence[pd.DataFrame],
    mark_df: pd.DataFrame, index_df: pd.DataFrame,
    cfg: Optional[FundingBasisSignalConfig] = None,
) -> pd.DataFrame:
    """Build the full per-asset daily signal frame.

    Args:
        asset: e.g. "BTC/USDT".
        spot_df: spot OHLCV with `timestamp`, `close` (1d).
        funding_frames: one or more standardised funding frames
            (from `_normalise_funding_frame`). Each is daily-resampled,
            then averaged across sources to form a robust daily series.
        mark_df, index_df: standardised mark + index klines (1d).
    """
    cfg = cfg or FundingBasisSignalConfig()
    if spot_df is None or spot_df.empty:
        return pd.DataFrame()

    spot = spot_df.copy()
    spot["timestamp"] = pd.to_numeric(spot["timestamp"]).astype("int64")
    spot = spot[["timestamp", "close"]].sort_values("timestamp").reset_index(
        drop=True)
    spot["datetime"] = pd.to_datetime(spot["timestamp"], unit="ms", utc=True)
    spot["asset"] = asset

    # -- Funding: build a daily series per source, then average.
    daily_per_source: List[pd.Series] = []
    for fdf in funding_frames:
        if fdf is None or fdf.empty:
            continue
        d = _resample_funding_to_daily(fdf)
        if d.empty:
            continue
        s = d.set_index("timestamp")["funding_day"]
        daily_per_source.append(s)
    if daily_per_source:
        # Outer-join all sources, then row-wise mean across whichever
        # sources reported on each day. No forward fill across days.
        funding_panel = pd.concat(daily_per_source, axis=1).sort_index()
        funding_blended = funding_panel.mean(axis=1, skipna=True)
        funding_daily = pd.DataFrame({
            "timestamp": funding_blended.index.astype("int64"),
            "funding_day": funding_blended.values,
        }).dropna()
    else:
        funding_daily = pd.DataFrame(columns=["timestamp", "funding_day"])

    out = _merge_asof_at_open(spot, funding_daily, "funding_day",
                                suffix="funding_1d_avg")

    # -- Basis: derive from Binance mark + index, merge_asof onto spot.
    basis_daily = _basis_from_klines(mark_df, index_df)
    out = _merge_asof_at_open(out, basis_daily, "basis", suffix="basis")

    # -- Rolling stats — every roller is backward-only (default in pandas).
    f1 = out["funding_1d_avg"]
    out["funding_7d_avg"] = f1.rolling(cfg.funding_short_days,
                                          min_periods=cfg.funding_short_days
                                          ).mean()
    out["funding_30d_avg"] = f1.rolling(cfg.funding_mid_days,
                                          min_periods=cfg.funding_mid_days
                                          ).mean()
    z_mean = f1.rolling(cfg.funding_z_days,
                         min_periods=cfg.funding_z_days).mean()
    z_std = f1.rolling(cfg.funding_z_days,
                        min_periods=cfg.funding_z_days).std(ddof=0)
    out["funding_z90"] = (f1 - z_mean) / z_std.replace(0.0, np.nan)
    out["funding_pct_rank_365"] = f1.rolling(
        cfg.funding_pct_rank_days, min_periods=cfg.funding_pct_rank_days,
    ).apply(lambda w: float((w <= w.iloc[-1]).mean() * 100.0), raw=False)
    out["funding_trend_pos"] = (
        out["funding_7d_avg"] > out["funding_30d_avg"]
    ).astype("float64")

    b = out["basis"]
    out["basis_7d_avg"] = b.rolling(cfg.basis_short_days,
                                       min_periods=cfg.basis_short_days
                                       ).mean()
    out["basis_30d_avg"] = b.rolling(cfg.basis_mid_days,
                                        min_periods=cfg.basis_mid_days
                                        ).mean()
    bz_mean = b.rolling(cfg.basis_z_days,
                         min_periods=cfg.basis_z_days).mean()
    bz_std = b.rolling(cfg.basis_z_days,
                        min_periods=cfg.basis_z_days).std(ddof=0)
    out["basis_z90"] = (b - bz_mean) / bz_std.replace(0.0, np.nan)

    close = out["close"].astype("float64")
    out["price_return_30d"] = (close / close.shift(cfg.price_short_days)
                                - 1.0) * 100.0
    out["price_return_90d"] = (close / close.shift(cfg.price_long_days)
                                - 1.0) * 100.0
    out["above_200dma"] = (
        close > close.rolling(cfg.ma_days,
                                min_periods=cfg.ma_days).mean()
    ).astype("float64")

    log_rets = np.log(close).diff()
    out["realised_vol_30d"] = (
        log_rets.rolling(cfg.rv_days, min_periods=cfg.rv_days)
                .std(ddof=1) * np.sqrt(365.0)
    )

    # -- Combined scores. NaN-safe: if any input is NaN, the score is NaN.
    out["crowding_score"] = (
        cfg.crowding_funding_weight * out["funding_z90"].clip(lower=-3.0,
                                                                upper=3.0)
        + cfg.crowding_basis_weight * out["basis_z90"].clip(lower=-3.0,
                                                                upper=3.0)
        + cfg.crowding_price_weight * (
            out["price_return_30d"].clip(lower=-50.0, upper=50.0) / 25.0
        )
    )
    out["carry_attractiveness"] = (
        cfg.carry_funding_weight * (-out["funding_z90"].clip(lower=-3.0,
                                                                 upper=3.0))
        + cfg.carry_trend_weight * (
            out["price_return_90d"].clip(lower=-50.0, upper=50.0) / 25.0
        )
    )

    # -- Regime classification — applied row-by-row, fully vectorised.
    state = pd.Series(STATE_UNKNOWN, index=out.index, dtype="object")

    # Required-not-NaN gate: if any of these are NaN the row is unknown.
    has_inputs = (
        out["funding_z90"].notna()
        & out["basis_z90"].notna()
        & out["price_return_30d"].notna()
        & out["above_200dma"].notna()
        & out["realised_vol_30d"].notna()
    )

    crowded = (
        has_inputs
        & ((out["funding_z90"] > cfg.funding_high_z90)
            | (out["funding_z90"] > cfg.funding_extreme_z90))
        & (out["basis_z90"] > cfg.basis_high_z90)
        & (out["price_return_30d"] > cfg.price_extended_30d_pct)
    )

    cheap_bullish = (
        has_inputs
        & (out["funding_z90"] < cfg.funding_low_z90)
        & (out["above_200dma"] > 0.0)
        & (out["basis_z90"] < cfg.basis_high_z90)
        & ~crowded
    )

    stress_neg = (
        has_inputs
        & (out["funding_1d_avg"] < 0.0)
        & ((out["above_200dma"] == 0.0)
            | (out["realised_vol_30d"] > cfg.realised_vol_high_annualised))
        & ~crowded
    )

    defensive = (
        has_inputs
        & (out["above_200dma"] == 0.0)
        & ~stress_neg
        & ~crowded
    )

    neutral = (
        has_inputs
        & ~crowded & ~cheap_bullish & ~stress_neg & ~defensive
    )

    # Assign in priority order — most-specific first wins.
    state.loc[crowded] = STATE_CROWDED_LONG
    state.loc[stress_neg & (state == STATE_UNKNOWN)] = STATE_STRESS_NEGATIVE_FUNDING
    state.loc[cheap_bullish & (state == STATE_UNKNOWN)] = STATE_CHEAP_BULLISH
    state.loc[defensive & (state == STATE_UNKNOWN)] = STATE_DEFENSIVE
    state.loc[neutral & (state == STATE_UNKNOWN)] = STATE_NEUTRAL_RISK_ON

    out["regime_state"] = state

    cols = [
        "timestamp", "datetime", "asset", "close", "above_200dma",
        "funding_1d_avg", "funding_7d_avg", "funding_30d_avg",
        "funding_z90", "funding_pct_rank_365", "funding_trend_pos",
        "basis", "basis_7d_avg", "basis_30d_avg", "basis_z90",
        "price_return_30d", "price_return_90d", "realised_vol_30d",
        "crowding_score", "carry_attractiveness", "regime_state",
    ]
    return out[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Top-level: load CSVs from the collector and build per-asset signals
# ---------------------------------------------------------------------------
def compute_signals(
    assets: Sequence[str] = fbdc.DEFAULT_ASSETS,
    timeframe: str = "1d",
    cfg: Optional[FundingBasisSignalConfig] = None,
    save: bool = True,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build a stacked signal DataFrame across the requested assets.

    Reads:
        * spot frames via the existing project cache (`pb.load_universe`).
        * funding + mark + index frames from
          `data/positioning/funding_basis/...` (collector outputs).
    """
    cfg = cfg or FundingBasisSignalConfig()
    spot_frames, missing = pb.load_universe(assets=assets, timeframe=timeframe)
    if missing:
        logger.warning("missing spot frames for %s", missing)

    out_rows: List[pd.DataFrame] = []
    for asset in assets:
        spot = spot_frames.get(asset)
        if spot is None or spot.empty:
            logger.warning("skipping %s — no spot data", asset)
            continue
        sym = utils.safe_symbol(asset)
        binance_funding_path = (fbdc.POSITIONING_DIR
                                  / f"binance_funding_{sym}.csv")
        bybit_funding_path = (fbdc.POSITIONING_DIR
                                / f"bybit_funding_{sym}.csv")
        deribit_funding_path = (fbdc.POSITIONING_DIR
                                  / f"deribit_funding_{sym}.csv")
        mark_path = fbdc.POSITIONING_DIR / f"binance_mark_klines_{sym}.csv"
        index_path = fbdc.POSITIONING_DIR / f"binance_index_klines_{sym}.csv"
        funding_frames = []
        for p in (binance_funding_path, bybit_funding_path,
                   deribit_funding_path):
            df = fbdc.load_funding_csv(p)
            if not df.empty:
                funding_frames.append(df)
        mark_df = fbdc.load_klines_csv(mark_path)
        index_df = fbdc.load_klines_csv(index_path)
        sig = build_signals_for_asset(asset, spot, funding_frames,
                                         mark_df, index_df, cfg)
        if sig.empty:
            continue
        out_rows.append(sig)

    if not out_rows:
        out = pd.DataFrame()
    else:
        out = pd.concat(out_rows, ignore_index=True)
    if save:
        utils.write_df(out, output_path
                          or config.RESULTS_DIR / "funding_basis_signals.csv")
    return out


def regime_distribution(signals_df: pd.DataFrame) -> Dict[str, float]:
    """Pct of rows in each regime (per asset, then averaged across assets).
    Useful for the dashboard + report."""
    if signals_df.empty or "regime_state" not in signals_df.columns:
        return {s: 0.0 for s in REGIME_STATES}
    grouped = (signals_df.groupby("asset")["regime_state"]
                  .value_counts(normalize=True).unstack(fill_value=0.0)
                  .mean(axis=0))
    out = {s: 0.0 for s in REGIME_STATES}
    for s in grouped.index:
        out[str(s)] = float(grouped[s])
    return out
