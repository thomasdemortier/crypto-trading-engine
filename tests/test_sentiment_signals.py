"""Tests for `src.sentiment_signals`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import sentiment_data_collector as sdc, sentiment_signals as ss


def _fg_frame(values: list, start_date: str = "2022-01-01") -> pd.DataFrame:
    """Build a sentiment_data_collector-shaped CSV equivalent."""
    n = len(values)
    dates = pd.date_range(start=start_date, periods=n, freq="D", tz="UTC")
    # Force ns precision: pandas ≥ 2.x returns `datetime64[us, UTC]`,
    # so a raw `view('int64')` would yield microseconds, not nanoseconds.
    ts = (dates.astype("datetime64[ns, UTC]").view("int64") // 10**6
           ).astype("int64")
    return pd.DataFrame({
        "timestamp": ts,
        "date": dates.strftime("%Y-%m-%d"),
        "fear_greed_value": values,
        "fear_greed_classification": ["Neutral"] * n,
        "source": ["alternative.me"] * n,
    })


def _wire(monkeypatch, df: pd.DataFrame):
    monkeypatch.setattr(sdc, "load_fear_greed", lambda: df)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_signal_columns_match_spec():
    expected = [
        "timestamp", "date",
        "fear_greed_value", "fear_greed_classification",
        "fg_7d_change", "fg_30d_change",
        "fg_7d_mean", "fg_30d_mean", "fg_90d_zscore",
        "extreme_fear", "fear", "neutral", "greed", "extreme_greed",
        "sentiment_recovering", "sentiment_deteriorating",
        "sentiment_state",
    ]
    assert ss.SIGNAL_COLUMNS == expected


# ---------------------------------------------------------------------------
# Lookahead
# ---------------------------------------------------------------------------
def test_rolling_features_use_only_past_data(monkeypatch):
    rng = np.random.default_rng(0)
    values = list(rng.integers(20, 80, size=200))
    _wire(monkeypatch, _fg_frame(values))
    out = ss.compute_sentiment_signals(save=False)
    # `fg_7d_change` at row 100 must equal v[100] - v[93].
    assert out["fg_7d_change"].iloc[100] == pytest.approx(
        float(values[100] - values[93]),
    )
    # 90d z-score: NaN until 90 valid observations exist (row 89 is the
    # first finite z when min_periods=90).
    z = out["fg_90d_zscore"]
    assert z.iloc[:89].isna().all()


def test_partial_vs_full_no_lookahead(monkeypatch):
    """Truncate inputs to first 120 rows; compare to full-history rows
    [0:120). Every column at every row must match — no future data
    can leak into a past feature."""
    rng = np.random.default_rng(1)
    values_full = list(rng.integers(20, 80, size=200))
    _wire(monkeypatch, _fg_frame(values_full))
    out_full = ss.compute_sentiment_signals(save=False)

    cut = 120
    _wire(monkeypatch, _fg_frame(values_full[:cut]))
    out_part = ss.compute_sentiment_signals(save=False)

    cmp_cols = ["fg_7d_change", "fg_30d_change", "fg_7d_mean",
                 "fg_30d_mean", "fg_90d_zscore",
                 "sentiment_recovering", "sentiment_deteriorating"]
    full_row = out_full.iloc[cut - 1]
    part_row = out_part.iloc[cut - 1]
    for col in cmp_cols:
        a, b = full_row[col], part_row[col]
        if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
            assert bool(a) == bool(b), col
            continue
        if pd.isna(a) and pd.isna(b):
            continue
        assert float(a) == pytest.approx(float(b), rel=1e-9, abs=1e-9), col


# ---------------------------------------------------------------------------
# State triggers
# ---------------------------------------------------------------------------
def test_extreme_fear_triggers_when_value_le_25(monkeypatch):
    values = [50] * 100 + [25, 20, 18, 15]
    _wire(monkeypatch, _fg_frame(values))
    out = ss.compute_sentiment_signals(save=False)
    assert out["sentiment_state"].iloc[-1] == "extreme_fear"
    assert bool(out["extreme_fear"].iloc[-1]) is True


def test_extreme_greed_triggers_when_value_ge_75(monkeypatch):
    values = [50] * 100 + [75, 80, 85, 90]
    _wire(monkeypatch, _fg_frame(values))
    out = ss.compute_sentiment_signals(save=False)
    assert out["sentiment_state"].iloc[-1] == "extreme_greed"
    assert bool(out["extreme_greed"].iloc[-1]) is True


def test_fear_recovery_triggers_after_low_then_rising(monkeypatch):
    """Build: 100 days flat 50, then dip to 22 (extreme fear), recover to
    32 (above the 25 threshold), and keep rising. The first day with
    value > 25 AND past-14d-min ≤ 30 AND 7d change > 0 must be
    `fear_recovery`."""
    values = [50] * 100 + [22] * 5 + [27, 30, 32, 34, 36, 38, 40, 42]
    _wire(monkeypatch, _fg_frame(values))
    out = ss.compute_sentiment_signals(save=False)
    last = out.iloc[-1]
    assert int(last["fear_greed_value"]) == 42
    # 7d change: 42 - values[-8] = 42 - 27 = 15 > 0 ✓
    # rolling_min over last 14 days includes the 22s ✓
    assert bool(last["sentiment_recovering"]) is True
    assert last["sentiment_state"] == "fear_recovery"


def test_deteriorating_triggers_when_falling_from_greed(monkeypatch):
    """Build: 30 days at 60 (greed regime), then a steady fall to 45.
    `fg_30d_mean` is in greed/neutral territory and 7d change is
    negative; state must be `deteriorating`."""
    values = [60] * 100 + [58, 56, 54, 52, 50, 48, 46, 45]
    _wire(monkeypatch, _fg_frame(values))
    out = ss.compute_sentiment_signals(save=False)
    last = out.iloc[-1]
    # 30d mean over past 30 days: dominated by the 60s and the descent
    # (still ≥ 50). 7d change = 45 - 56 = -11 < 0.
    assert float(last["fg_30d_mean"]) >= 50.0
    assert float(last["fg_7d_change"]) < 0
    assert bool(last["sentiment_deteriorating"]) is True
    assert last["sentiment_state"] == "deteriorating"


def test_missing_value_yields_unknown(monkeypatch):
    df = _fg_frame([50] * 5)
    df.loc[df.index[-1], "fear_greed_value"] = pd.NA
    _wire(monkeypatch, df)
    out = ss.compute_sentiment_signals(save=False)
    assert out["sentiment_state"].iloc[-1] == "unknown"


def test_state_set_is_documented(monkeypatch):
    rng = np.random.default_rng(2)
    values = list(rng.integers(10, 90, size=300))
    _wire(monkeypatch, _fg_frame(values))
    out = ss.compute_sentiment_signals(save=False)
    assert set(out["sentiment_state"].unique()).issubset(set(ss.VALID_STATES))


def test_warmup_rows_are_unknown(monkeypatch):
    """Until 30 days of values exist, fg_30d_mean is NaN → state cannot
    decide between fear_recovery / deteriorating / neutral and falls to
    unknown unless an extreme triggers."""
    rng = np.random.default_rng(3)
    values = list(rng.integers(40, 60, size=50))   # avoid extremes
    _wire(monkeypatch, _fg_frame(values))
    out = ss.compute_sentiment_signals(save=False)
    # First 29 rows have fg_30d_mean NaN AND no extreme → unknown.
    assert (out["sentiment_state"].iloc[:29] == "unknown").all()


def test_empty_cache_returns_empty_documented_frame(monkeypatch):
    _wire(monkeypatch, pd.DataFrame())
    out = ss.compute_sentiment_signals(save=False)
    assert out.empty
    assert list(out.columns) == ss.SIGNAL_COLUMNS
