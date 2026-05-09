"""
Sentiment signal engineering — Fear & Greed only.

Builds a daily signal table from the cached alternative.me Fear & Greed
history.  The signal generator is **not** a strategy — it produces
features the overlay allocator consumes.

LOOKAHEAD RULES (non-negotiable):
    * Every rolling window is backward (`min_periods=window`, no
      `center=True`).
    * 7d / 30d changes are `value[t] - value[t-N]`.
    * The 90d z-score at row `t` only depends on values at `t' <= t`.
    * Classification at row `t` is a pure function of features at the
      same row, all of which depend only on data with timestamp ≤ t.
    * Missing inputs → `unknown`. Never guess, never forward-fill.

States (per spec):
    extreme_fear     value <= 25
    extreme_greed    value >= 75
    fear_recovery    value was below 30 in the last 14 days AND the 7d
                      change is positive (sentiment is recovering)
    deteriorating    7d change is negative AND the previous 30d mean
                      was in greed/neutral territory (>= 50) — i.e.
                      sentiment is rolling over from a positive regime
    neutral          nothing extreme fired
    unknown          warm-up rows where the rule's inputs are NaN

`extreme_fear` and `extreme_greed` take precedence over the other rules.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from . import config, sentiment_data_collector as sdc, utils

logger = utils.get_logger("cte.sentiment_signals")


# ---------------------------------------------------------------------------
# Schema (locked here so tests can assert)
# ---------------------------------------------------------------------------
SIGNAL_COLUMNS: List[str] = [
    "timestamp", "date",
    "fear_greed_value", "fear_greed_classification",
    "fg_7d_change", "fg_30d_change",
    "fg_7d_mean", "fg_30d_mean", "fg_90d_zscore",
    "extreme_fear", "fear", "neutral", "greed", "extreme_greed",
    "sentiment_recovering", "sentiment_deteriorating",
    "sentiment_state",
]

VALID_STATES: List[str] = [
    "extreme_fear", "fear_recovery", "neutral",
    "greed", "extreme_greed", "deteriorating", "unknown",
]

_ROLL_SHORT = 7
_ROLL_LONG = 30
_ROLL_ZSCORE = 90

# Thresholds — fixed by spec, not tuning knobs.
_EXTREME_FEAR_VALUE = 25
_EXTREME_GREED_VALUE = 75
_FEAR_VALUE = 35           # for the boolean flag column
_GREED_VALUE = 65          # for the boolean flag column
_RECOVERY_LOOKBACK_DAYS = 14
_RECOVERY_FEAR_THRESHOLD = 30
_DETERIORATING_PRIOR_VALUE = 50  # 30d mean above this = greed/neutral regime


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _backward_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def _classify_state(row: pd.Series) -> str:
    v = row.get("fear_greed_value")
    if v is None or pd.isna(v):
        return "unknown"
    fg7c = row.get("fg_7d_change")
    fg30m = row.get("fg_30d_mean")

    # Hard precedence — extremes override everything else.
    if float(v) <= _EXTREME_FEAR_VALUE:
        return "extreme_fear"
    if float(v) >= _EXTREME_GREED_VALUE:
        return "extreme_greed"

    # The other rules require backward rolling values; if missing -> unknown.
    if pd.isna(fg7c) or pd.isna(fg30m):
        return "unknown"

    # `fear_recovery` consumes a window-min flag computed below.
    recovering = bool(row.get("sentiment_recovering", False))
    deteriorating = bool(row.get("sentiment_deteriorating", False))
    # Recovery wins over deteriorating only when its trigger fully fires.
    if recovering:
        return "fear_recovery"
    if deteriorating:
        return "deteriorating"
    return "neutral"


def compute_sentiment_signals(save: bool = True) -> pd.DataFrame:
    """Build the daily sentiment signal table. Returns an empty
    documented frame when the cached F&G CSV is missing — never raises."""
    fg = sdc.load_fear_greed()
    if fg is None or fg.empty:
        logger.warning("sentiment cache missing — run "
                        "`python main.py download_sentiment_data` first")
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

    df = fg.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"]).astype("int64")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True
                                      ).dt.strftime("%Y-%m-%d")
    df["fear_greed_value"] = pd.to_numeric(df["fear_greed_value"],
                                              errors="coerce")

    # Boolean classification flags (independent of state classifier).
    v = df["fear_greed_value"]
    df["extreme_fear"] = (v <= _EXTREME_FEAR_VALUE).fillna(False)
    df["fear"] = ((v <= _FEAR_VALUE) & (v > _EXTREME_FEAR_VALUE)).fillna(False)
    df["neutral"] = ((v > _FEAR_VALUE) & (v < _GREED_VALUE)).fillna(False)
    df["greed"] = ((v >= _GREED_VALUE) & (v < _EXTREME_GREED_VALUE)).fillna(False)
    df["extreme_greed"] = (v >= _EXTREME_GREED_VALUE).fillna(False)

    # Rolling stats — strict backward.
    df["fg_7d_change"] = v - v.shift(_ROLL_SHORT)
    df["fg_30d_change"] = v - v.shift(_ROLL_LONG)
    df["fg_7d_mean"] = v.rolling(_ROLL_SHORT, min_periods=_ROLL_SHORT).mean()
    df["fg_30d_mean"] = v.rolling(_ROLL_LONG, min_periods=_ROLL_LONG).mean()
    df["fg_90d_zscore"] = _backward_zscore(v, _ROLL_ZSCORE)

    # Recovery: value was below 30 in the past 14 days AND 7d change > 0.
    rolling_min_14d = v.rolling(_RECOVERY_LOOKBACK_DAYS,
                                  min_periods=_RECOVERY_LOOKBACK_DAYS).min()
    df["sentiment_recovering"] = (
        (rolling_min_14d <= _RECOVERY_FEAR_THRESHOLD)
        & (df["fg_7d_change"] > 0)
        & ~df["extreme_fear"]
        & ~df["extreme_greed"]
    ).fillna(False)

    # Deteriorating: 7d change < 0 AND 30d mean was in greed/neutral
    # territory (>= 50). Locked out when extremes dominate.
    df["sentiment_deteriorating"] = (
        (df["fg_7d_change"] < 0)
        & (df["fg_30d_mean"] >= _DETERIORATING_PRIOR_VALUE)
        & ~df["extreme_fear"]
        & ~df["extreme_greed"]
    ).fillna(False)

    df["sentiment_state"] = df.apply(_classify_state, axis=1)

    out = df.reindex(columns=SIGNAL_COLUMNS).reset_index(drop=True)
    if save:
        utils.write_df(out, config.RESULTS_DIR / "sentiment_signals.csv")
    return out
