"""
Cross-asset crypto regime signals.

Pure measurement, no trading logic. Computes a per-day signals frame from
the expanded universe (BTC, ETH, and altcoins) and labels each day with
one of:

  * `alt_risk_on`   — altcoins broadly outperforming BTC.
  * `btc_leadership` — BTC outperforming altcoins.
  * `defensive`     — broad-market weakness, cash bias.
  * `mixed`         — none of the above cleanly satisfied.
  * `unknown`       — warmup; insufficient history.

Lookahead-free: every value at row t uses ONLY data with timestamp ≤ t.
The same partial-vs-full equality property the indicator tests assert is
preserved here, and is asserted by `tests/test_crypto_regime_signals.py`.

Output: `results/crypto_regime_signals.csv` keyed on the union of
universe timestamps.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config, portfolio_backtester as pb, utils

logger = utils.get_logger("cte.crypto_regime_signals")


# Regime labels (exposed as constants so consumers compare safely).
ALT_RISK_ON = "alt_risk_on"
BTC_LEADERSHIP = "btc_leadership"
DEFENSIVE = "defensive"
MIXED = "mixed"
UNKNOWN = "unknown"
REGIME_LABELS = (ALT_RISK_ON, BTC_LEADERSHIP, DEFENSIVE, MIXED, UNKNOWN)


# Default thresholds (per spec — kept as constants so a future patch can
# expose them without changing call sites).
THRESH_PCT_ABOVE_100D = 60.0
THRESH_PCT_BEATING_BTC_30D = 50.0
THRESH_PCT_BEATING_BTC_90D_BELOW = 40.0
THRESH_PCT_ABOVE_200D_DEFENSIVE = 40.0
BTC_DEFENSIVE_MA_DAYS = 200


def _bars_per_day(timeframe: str) -> int:
    return {"1h": 24, "4h": 6, "1d": 1}.get(timeframe, 1)


def _aligned_close_frame(asset_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a wide DataFrame indexed by timestamp with one column per
    asset (close prices, forward-filled across the union of timestamps)."""
    if not asset_frames:
        return pd.DataFrame()
    all_ts = sorted({int(t) for df in asset_frames.values()
                     for t in df["timestamp"].astype("int64")})
    pieces: Dict[str, pd.Series] = {}
    for asset, df in asset_frames.items():
        s = pd.Series(
            df["close"].astype(float).values,
            index=df["timestamp"].astype("int64").values,
        ).reindex(all_ts).ffill()
        pieces[asset] = s
    out = pd.DataFrame(pieces, index=all_ts)
    out.index.name = "timestamp"
    return out


def _percent_return_lag(s: pd.Series, lag_bars: int) -> pd.Series:
    return (s / s.shift(lag_bars)) - 1.0


# ---------------------------------------------------------------------------
# Public: compute_regime_signals
# ---------------------------------------------------------------------------
def compute_regime_signals(
    asset_frames: Optional[Dict[str, pd.DataFrame]] = None,
    timeframe: str = "1d",
    momentum_short_window: int = 30,
    momentum_long_window: int = 90,
    breadth_short_ma: int = 100,
    breadth_long_ma: int = 200,
    save: bool = True,
) -> pd.DataFrame:
    """Compute per-day cross-asset regime signals from the expanded universe.

    If `asset_frames` is omitted, the expanded universe is loaded from the
    cached candles. Returns a DataFrame keyed on `timestamp` (ms epoch).
    """
    if asset_frames is None:
        asset_frames, _ = pb.load_universe(
            assets=list(config.EXPANDED_UNIVERSE), timeframe=timeframe,
        )
    if not asset_frames or "BTC/USDT" not in asset_frames:
        logger.warning("regime signals: BTC/USDT missing from universe")
        out = pd.DataFrame()
        if save:
            utils.write_df(out, config.RESULTS_DIR / "crypto_regime_signals.csv")
        return out

    closes = _aligned_close_frame(asset_frames)
    bpd = _bars_per_day(timeframe)
    n_short = momentum_short_window * bpd
    n_long = momentum_long_window * bpd
    n_short_ma = breadth_short_ma * bpd
    n_long_ma = breadth_long_ma * bpd

    # ETH/BTC ratio.
    btc = closes["BTC/USDT"]
    eth = closes.get("ETH/USDT")
    if eth is not None:
        eth_btc_ratio = eth / btc
        eth_btc_30 = _percent_return_lag(eth_btc_ratio, n_short) * 100.0
        eth_btc_90 = _percent_return_lag(eth_btc_ratio, n_long) * 100.0
    else:
        eth_btc_ratio = pd.Series(np.nan, index=closes.index)
        eth_btc_30 = pd.Series(np.nan, index=closes.index)
        eth_btc_90 = pd.Series(np.nan, index=closes.index)

    # Per-asset returns at the two momentum horizons.
    ret_short = closes.apply(lambda s: _percent_return_lag(s, n_short) * 100.0)
    ret_long = closes.apply(lambda s: _percent_return_lag(s, n_long) * 100.0)
    btc_ret_short = ret_short["BTC/USDT"]
    btc_ret_long = ret_long["BTC/USDT"]

    # Alt basket = equal-weight non-BTC assets.
    alt_cols = [a for a in closes.columns if a != "BTC/USDT"]
    if alt_cols:
        alt_close_basket = closes[alt_cols].apply(
            lambda col: col / col.dropna().iloc[0] if col.dropna().size > 0 else col,
            axis=0,
        )
        # Equal-weight basket: at any time t the basket value is the
        # average of normalised closes (each normalised at first valid).
        alt_basket = alt_close_basket.mean(axis=1, skipna=True)
        alt_basket_short = _percent_return_lag(alt_basket, n_short) * 100.0
        alt_basket_long = _percent_return_lag(alt_basket, n_long) * 100.0
    else:
        alt_basket_short = pd.Series(np.nan, index=closes.index)
        alt_basket_long = pd.Series(np.nan, index=closes.index)

    # Breadth: % of assets above their MA at each bar.
    pct_above_100 = (
        closes.gt(closes.rolling(n_short_ma, min_periods=n_short_ma).mean())
              .sum(axis=1)
        / closes.notna().sum(axis=1).clip(lower=1) * 100.0
    )
    pct_above_200 = (
        closes.gt(closes.rolling(n_long_ma, min_periods=n_long_ma).mean())
              .sum(axis=1)
        / closes.notna().sum(axis=1).clip(lower=1) * 100.0
    )

    # % assets beating BTC over 30/90 days (BTC excluded from numerator).
    def _pct_beating_btc(rets: pd.DataFrame, btc_col: str) -> pd.Series:
        non_btc = rets.drop(columns=[btc_col], errors="ignore")
        if non_btc.empty:
            return pd.Series(0.0, index=rets.index)
        beats = non_btc.gt(rets[btc_col], axis=0)
        # Only count assets that have a valid return at this row.
        eligible = non_btc.notna()
        denom = eligible.sum(axis=1).clip(lower=1)
        return (beats & eligible).sum(axis=1) / denom * 100.0

    pct_beat_btc_30 = _pct_beating_btc(ret_short, "BTC/USDT")
    pct_beat_btc_90 = _pct_beating_btc(ret_long, "BTC/USDT")

    # BTC defensive flag: BTC < its long MA.
    btc_long_ma = btc.rolling(BTC_DEFENSIVE_MA_DAYS * bpd,
                              min_periods=BTC_DEFENSIVE_MA_DAYS * bpd).mean()
    btc_below_long_ma = btc.lt(btc_long_ma)

    # Alt regime label (independent — used as a column for diagnostics).
    alt_regime = pd.Series("unknown", index=closes.index, dtype="object")
    if alt_cols:
        alt_strong = (alt_basket_long > btc_ret_long) \
                     & (eth_btc_90 > 0) \
                     & (pct_above_100 > THRESH_PCT_ABOVE_100D) \
                     & (pct_beat_btc_30 > THRESH_PCT_BEATING_BTC_30D)
        alt_weak = (btc_ret_long > alt_basket_long) \
                   & (pct_beat_btc_90 < THRESH_PCT_BEATING_BTC_90D_BELOW)
        alt_regime = alt_regime.mask(alt_strong, ALT_RISK_ON)
        alt_regime = alt_regime.mask(alt_weak, BTC_LEADERSHIP)

    # Market regime (defensive vs everything else).
    market_regime = pd.Series("unknown", index=closes.index, dtype="object")
    is_defensive = (btc_below_long_ma
                    | (pct_above_200 < THRESH_PCT_ABOVE_200D_DEFENSIVE))
    market_regime = market_regime.mask(is_defensive, DEFENSIVE)
    market_regime = market_regime.where(market_regime != "unknown", "non_defensive")

    # Combined risk_state — defensive wins; then alt_risk_on; then
    # btc_leadership; else mixed (after warmup).
    risk_state = pd.Series(UNKNOWN, index=closes.index, dtype="object")
    warmup_done = (
        btc_long_ma.notna() & pct_above_100.notna() & pct_above_200.notna()
        & ret_long["BTC/USDT"].notna()
    )
    risk_state = risk_state.where(~warmup_done, MIXED)
    risk_state = risk_state.mask(warmup_done & is_defensive, DEFENSIVE)
    if alt_cols:
        risk_state = risk_state.mask(
            warmup_done & ~is_defensive & alt_strong, ALT_RISK_ON,
        )
        risk_state = risk_state.mask(
            warmup_done & ~is_defensive & alt_weak & ~alt_strong,
            BTC_LEADERSHIP,
        )

    out = pd.DataFrame({
        "timestamp": closes.index,
        "datetime": pd.to_datetime(closes.index, unit="ms", utc=True),
        "eth_btc_ratio": eth_btc_ratio.values,
        "eth_btc_30d_return_pct": eth_btc_30.values,
        "eth_btc_90d_return_pct": eth_btc_90.values,
        "alt_basket_return_30d_pct": alt_basket_short.values,
        "alt_basket_return_90d_pct": alt_basket_long.values,
        "btc_return_30d_pct": btc_ret_short.values,
        "btc_return_90d_pct": btc_ret_long.values,
        "pct_assets_above_100d_ma": pct_above_100.values,
        "pct_assets_above_200d_ma": pct_above_200.values,
        "pct_assets_beating_btc_30d": pct_beat_btc_30.values,
        "pct_assets_beating_btc_90d": pct_beat_btc_90.values,
        "btc_below_200d_ma": btc_below_long_ma.values,
        "alt_regime": alt_regime.values,
        "market_regime": market_regime.values,
        "risk_state": risk_state.values,
    })
    if save:
        utils.write_df(out, config.RESULTS_DIR / "crypto_regime_signals.csv")
    return out


# ---------------------------------------------------------------------------
# Convenience: regime distribution summary
# ---------------------------------------------------------------------------
def regime_distribution(signals_df: pd.DataFrame) -> Dict[str, float]:
    """Return percent-of-bars in each risk_state. Excludes unknown."""
    if signals_df is None or signals_df.empty or "risk_state" not in signals_df.columns:
        return {}
    valid = signals_df[signals_df["risk_state"] != UNKNOWN]
    if valid.empty:
        return {}
    return (
        (valid["risk_state"].value_counts(normalize=True) * 100.0)
        .round(2).to_dict()
    )
