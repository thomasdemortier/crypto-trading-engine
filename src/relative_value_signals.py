"""
BTC/ETH relative-value signal engineering.

Builds a daily, lookahead-free signal DataFrame keyed on the BTC/ETH
spot OHLCV calendar. The "asset" of the signal is conceptually the
PAIR (BTC vs ETH), so this module emits ONE row per spot bar (not one
per asset).

Hard rules:
    * Backward-only rolling windows (`min_periods = window`); no
      `center=True`, no `shift(-N)`, no future joins.
    * Inputs are inner-joined on the spot timestamp axis — if either
      asset is missing on a bar, the row's regime is `unknown`.
    * Thresholds are LOCKED in `RelativeValueSignalConfig` defaults and
      MUST NOT be tuned after seeing strategy performance.

Output schema (one row per spot bar):
    timestamp, datetime, btc_close, eth_close, eth_btc_ratio,
    ratio_30d_return, ratio_90d_return, ratio_ma_200, ratio_above_ma_200,
    ratio_z90, btc_30d_return, btc_90d_return, eth_30d_return,
    eth_90d_return, btc_above_200dma, eth_above_200dma,
    btc_realised_vol_30d, eth_realised_vol_30d, relative_momentum_score,
    relative_trend_score, regime_state
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import config, portfolio_backtester as pb, utils

logger = utils.get_logger("cte.relative_value_signals")


# ---------------------------------------------------------------------------
# Locked thresholds — edits after running the strategy = retuning.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RelativeValueSignalConfig:
    # Window sizes (days).
    ratio_short_days: int = 30
    ratio_long_days: int = 90
    ratio_ma_days: int = 200
    ratio_z_days: int = 90
    price_short_days: int = 30
    price_long_days: int = 90
    ma_days: int = 200
    rv_days: int = 30

    # Regime thresholds — locked.
    ratio_extreme_z90: float = 2.0       # |z90| > this -> unstable rotation
    rv_high_annualised: float = 1.20     # > 120 % annualised vol -> unstable

    # Score weights (relative momentum vs trend).
    rel_momentum_weight: float = 0.6     # weights ratio momentum
    rel_trend_weight: float = 0.4        # weights ratio above 200d MA


STATE_ETH_LEADERSHIP = "eth_leadership"
STATE_BTC_LEADERSHIP = "btc_leadership"
STATE_DEFENSIVE = "defensive"
STATE_UNSTABLE_ROTATION = "unstable_rotation"
STATE_NEUTRAL = "neutral"
STATE_UNKNOWN = "unknown"

REGIME_STATES: List[str] = [
    STATE_ETH_LEADERSHIP, STATE_BTC_LEADERSHIP, STATE_DEFENSIVE,
    STATE_UNSTABLE_ROTATION, STATE_NEUTRAL, STATE_UNKNOWN,
]


# ---------------------------------------------------------------------------
# Per-asset close series helper
# ---------------------------------------------------------------------------
def _close_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    s = pd.Series(df["close"].astype("float64").values,
                    index=df["timestamp"].astype("int64").values)
    return s.sort_index()


# ---------------------------------------------------------------------------
# Per-pair signal builder
# ---------------------------------------------------------------------------
def build_signals(
    btc_df: pd.DataFrame, eth_df: pd.DataFrame,
    cfg: Optional[RelativeValueSignalConfig] = None,
) -> pd.DataFrame:
    """Build the BTC/ETH relative-value signal frame.

    Args:
        btc_df: BTC OHLCV (timestamp, close, ...) for daily bars.
        eth_df: ETH OHLCV (timestamp, close, ...) for daily bars.
    """
    cfg = cfg or RelativeValueSignalConfig()
    if btc_df is None or btc_df.empty or eth_df is None or eth_df.empty:
        return pd.DataFrame()

    btc = _close_series(btc_df)
    eth = _close_series(eth_df)
    common = btc.index.intersection(eth.index)
    if len(common) < 2:
        return pd.DataFrame()
    btc = btc.loc[common].astype("float64")
    eth = eth.loc[common].astype("float64")

    out = pd.DataFrame({
        "timestamp": common.astype("int64"),
        "btc_close": btc.values,
        "eth_close": eth.values,
    })
    out["datetime"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True)

    # ETH/BTC ratio + rollers (backward-only).
    ratio = eth / btc.replace(0.0, np.nan)
    out["eth_btc_ratio"] = ratio.values
    out["ratio_30d_return"] = (
        (ratio / ratio.shift(cfg.ratio_short_days) - 1.0) * 100.0
    ).values
    out["ratio_90d_return"] = (
        (ratio / ratio.shift(cfg.ratio_long_days) - 1.0) * 100.0
    ).values
    ratio_ma = ratio.rolling(cfg.ratio_ma_days,
                              min_periods=cfg.ratio_ma_days).mean()
    out["ratio_ma_200"] = ratio_ma.values
    out["ratio_above_ma_200"] = (ratio > ratio_ma).astype("float64").values
    z_mean = ratio.rolling(cfg.ratio_z_days,
                            min_periods=cfg.ratio_z_days).mean()
    z_std = ratio.rolling(cfg.ratio_z_days,
                           min_periods=cfg.ratio_z_days).std(ddof=0)
    out["ratio_z90"] = ((ratio - z_mean)
                          / z_std.replace(0.0, np.nan)).values

    # Per-asset price returns and 200d MA.
    out["btc_30d_return"] = (
        (btc / btc.shift(cfg.price_short_days) - 1.0) * 100.0
    ).values
    out["btc_90d_return"] = (
        (btc / btc.shift(cfg.price_long_days) - 1.0) * 100.0
    ).values
    out["eth_30d_return"] = (
        (eth / eth.shift(cfg.price_short_days) - 1.0) * 100.0
    ).values
    out["eth_90d_return"] = (
        (eth / eth.shift(cfg.price_long_days) - 1.0) * 100.0
    ).values
    out["btc_above_200dma"] = (
        btc > btc.rolling(cfg.ma_days, min_periods=cfg.ma_days).mean()
    ).astype("float64").values
    out["eth_above_200dma"] = (
        eth > eth.rolling(cfg.ma_days, min_periods=cfg.ma_days).mean()
    ).astype("float64").values

    # Realised vol (annualised log-return std).
    btc_log = np.log(btc).diff()
    eth_log = np.log(eth).diff()
    out["btc_realised_vol_30d"] = (
        btc_log.rolling(cfg.rv_days, min_periods=cfg.rv_days)
                .std(ddof=1) * np.sqrt(365.0)
    ).values
    out["eth_realised_vol_30d"] = (
        eth_log.rolling(cfg.rv_days, min_periods=cfg.rv_days)
                .std(ddof=1) * np.sqrt(365.0)
    ).values

    # Composite scores. Centred so 0 = no edge.
    rel_mom = (out["ratio_30d_return"].clip(-50.0, 50.0) / 25.0
                + out["ratio_90d_return"].clip(-50.0, 50.0) / 25.0) / 2.0
    rel_trend = out["ratio_above_ma_200"] * 2.0 - 1.0  # +1 above, -1 below
    out["relative_momentum_score"] = rel_mom.values
    out["relative_trend_score"] = rel_trend.values

    # Regime classification — vectorised, priority order matters.
    state = pd.Series(STATE_UNKNOWN, index=out.index, dtype="object")
    has_inputs = (
        out["ratio_above_ma_200"].notna()
        & out["ratio_30d_return"].notna()
        & out["ratio_90d_return"].notna()
        & out["btc_above_200dma"].notna()
        & out["eth_above_200dma"].notna()
        & out["ratio_z90"].notna()
        & out["btc_realised_vol_30d"].notna()
        & out["eth_realised_vol_30d"].notna()
    )

    unstable = (
        has_inputs
        & ((out["ratio_z90"].abs() > cfg.ratio_extreme_z90)
            | (out["btc_realised_vol_30d"] > cfg.rv_high_annualised)
            | (out["eth_realised_vol_30d"] > cfg.rv_high_annualised))
    )

    defensive = (
        has_inputs
        & ~unstable
        & (out["btc_above_200dma"] == 0.0)
        & (out["eth_above_200dma"] == 0.0)
    )

    eth_lead = (
        has_inputs
        & ~unstable & ~defensive
        & (out["ratio_above_ma_200"] == 1.0)
        & (out["ratio_30d_return"] > 0.0)
        & (out["ratio_90d_return"] > 0.0)
    )

    btc_lead = (
        has_inputs
        & ~unstable & ~defensive & ~eth_lead
        & (
            (out["ratio_above_ma_200"] == 0.0)
            | ((out["ratio_30d_return"] <= 0.0)
                & (out["ratio_90d_return"] <= 0.0))
          )
        & (out["btc_above_200dma"] == 1.0)
    )

    neutral = (
        has_inputs
        & ~unstable & ~defensive & ~eth_lead & ~btc_lead
    )

    state.loc[unstable] = STATE_UNSTABLE_ROTATION
    state.loc[defensive & (state == STATE_UNKNOWN)] = STATE_DEFENSIVE
    state.loc[eth_lead & (state == STATE_UNKNOWN)] = STATE_ETH_LEADERSHIP
    state.loc[btc_lead & (state == STATE_UNKNOWN)] = STATE_BTC_LEADERSHIP
    state.loc[neutral & (state == STATE_UNKNOWN)] = STATE_NEUTRAL

    out["regime_state"] = state.values

    cols = [
        "timestamp", "datetime", "btc_close", "eth_close",
        "eth_btc_ratio", "ratio_30d_return", "ratio_90d_return",
        "ratio_ma_200", "ratio_above_ma_200", "ratio_z90",
        "btc_30d_return", "btc_90d_return", "eth_30d_return",
        "eth_90d_return", "btc_above_200dma", "eth_above_200dma",
        "btc_realised_vol_30d", "eth_realised_vol_30d",
        "relative_momentum_score", "relative_trend_score", "regime_state",
    ]
    return out[cols].reset_index(drop=True)


def compute_signals(
    timeframe: str = "1d",
    cfg: Optional[RelativeValueSignalConfig] = None,
    save: bool = True,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load BTC/USDT + ETH/USDT spot frames from the project cache and
    build the signal frame."""
    cfg = cfg or RelativeValueSignalConfig()
    frames, missing = pb.load_universe(assets=["BTC/USDT", "ETH/USDT"],
                                          timeframe=timeframe)
    if missing or "BTC/USDT" not in frames or "ETH/USDT" not in frames:
        logger.warning("missing spot frames: %s", missing)
        return pd.DataFrame()
    out = build_signals(frames["BTC/USDT"], frames["ETH/USDT"], cfg)
    if save:
        utils.write_df(
            out, output_path or
            config.RESULTS_DIR / "relative_value_signals.csv",
        )
    return out


def regime_distribution(signals_df: pd.DataFrame) -> Dict[str, float]:
    if signals_df.empty or "regime_state" not in signals_df.columns:
        return {s: 0.0 for s in REGIME_STATES}
    counts = signals_df["regime_state"].value_counts(normalize=True)
    out = {s: 0.0 for s in REGIME_STATES}
    for s in counts.index:
        out[str(s)] = float(counts[s])
    return out
