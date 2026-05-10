"""
FX end-of-day trend strategy v1.

Self-contained signal generator + vectorised backtest for a single
locked rule: long EUR/USD when close > N-day SMA, cash otherwise. The
rule is fixed before any results are seen; there is no parameter
search, no optimiser, no broker integration, no order placement, no
leverage, no shorting.

Hard rules (locked):
    * Reads only the validated dataset on disk
      (`data/fx/fx_daily_v1.parquet` via `fx_research_dataset`); the
      strategy module itself never opens a network socket.
    * No API keys, no broker, no execution, no paper trading, no live
      trading.
    * The signal at date `t` may use ONLY data with date ≤ t. The
      position used to capture the return on day `t` is the signal
      from day `t-1` (one-day lag). This is enforced by
      `_lag_position` and asserted in tests.
    * Position is binary {0, 1}. No leverage, no shorts.
    * No transaction costs / spreads / slippage are assumed. ECB
      EUR/USD is a daily reference fix, not a tradable broker quote;
      this is research-only.

The companion research orchestrator lives in
`src/fx_eod_trend_research.py`. The CLI commands wire that
orchestrator into `main.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Locked config
# ---------------------------------------------------------------------------
MODE_LONG_CASH = "long_cash"
TRADING_DAYS_PER_YEAR = 252  # ECB fixings ~252/yr (TARGET holidays excluded)


@dataclass(frozen=True)
class FXEODTrendConfig:
    """Frozen configuration. Defaults are the locked v1 rule:
    EUR/USD, 200-day SMA, long-or-cash, daily."""
    asset: str = "EUR/USD"
    lookback_days: int = 200
    timeframe: str = "1d"
    mode: str = MODE_LONG_CASH
    initial_cash: float = 1.0
    strategy_name: str = "fx_eod_trend_v1"
    source_filter: str = "ecb_sdmx"  # only the ECB-sourced rows


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
class FXEODTrendStrategy:
    """Vectorised long-or-cash trend strategy.

    Pipeline:

        1. `prepare_close_series(df, cfg)` — filter to the asset +
           source, sort by date, drop missing close.
        2. `generate_signals(close, cfg)` — compute the SMA and a
           binary signal `signal_t = (close_t > sma_t)`. Uses ONLY
           data with index ≤ t.
        3. `_lag_position(signal)` — shift the signal forward by one
           bar so it captures the *next* day's return. The first row
           after warmup has `position = NaN` (no prior signal). Cast
           to {0, 1} after dropna.
        4. `compute_backtest(df, cfg)` — return one row per date with:
           date, asset, close, sma, signal, position, raw_return,
           strategy_return, strategy_equity, benchmark_buyhold_return,
           benchmark_buyhold_equity, cash_equity.
    """
    name = "fx_eod_trend_v1"

    # -- Data prep -----------------------------------------------------
    @staticmethod
    def prepare_close_series(df: pd.DataFrame,
                                cfg: FXEODTrendConfig
                                ) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["date", "close"])
        out = df.copy()
        if "asset" in out.columns:
            out = out[out["asset"] == cfg.asset]
        if "source" in out.columns and cfg.source_filter:
            out = out[out["source"] == cfg.source_filter]
        if "data_quality_status" in out.columns:
            out = out[out["data_quality_status"] == "ok"]
        out = out.dropna(subset=["date", "close"])
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date").drop_duplicates(
            subset=["date"], keep="last"
        ).reset_index(drop=True)
        return out[["date", "close"]]

    # -- Signal --------------------------------------------------------
    @staticmethod
    def generate_signals(close: pd.Series,
                            cfg: FXEODTrendConfig) -> pd.DataFrame:
        """Return a DataFrame indexed 0..N-1 with columns: close, sma,
        signal. `sma` and `signal` are NaN for the warmup window."""
        if not isinstance(close, pd.Series):
            close = pd.Series(close)
        sma = close.rolling(window=cfg.lookback_days,
                              min_periods=cfg.lookback_days).mean()
        signal = pd.Series(np.where(close > sma, 1.0, 0.0),
                              index=close.index)
        # Warmup rows have NaN sma → NaN signal (will be dropped later).
        signal = signal.where(sma.notna(), other=np.nan)
        out = pd.DataFrame({
            "close": close.values,
            "sma": sma.values,
            "signal": signal.values,
        })
        return out

    @staticmethod
    def _lag_position(signal: pd.Series) -> pd.Series:
        """One-day lag: position used to earn return on day t is the
        signal that ended day t-1."""
        return signal.shift(1)

    # -- Backtest ------------------------------------------------------
    @classmethod
    def compute_backtest(cls, df: pd.DataFrame,
                            cfg: FXEODTrendConfig) -> pd.DataFrame:
        prep = cls.prepare_close_series(df, cfg)
        if len(prep) <= cfg.lookback_days + 1:
            return pd.DataFrame(columns=BACKTEST_COLUMNS)
        sig_df = cls.generate_signals(prep["close"], cfg)
        position = cls._lag_position(sig_df["signal"])
        raw_return = prep["close"].pct_change()
        # Drop warmup (where position is NaN): start active period when
        # both raw_return and position are defined.
        active_mask = position.notna() & raw_return.notna()
        strategy_return = (position * raw_return).where(active_mask, 0.0)
        # Equity curves anchored at initial_cash on the first active day.
        equity_factor = (1.0 + strategy_return).cumprod()
        equity_factor = equity_factor.where(active_mask).ffill().fillna(1.0)
        strategy_equity = cfg.initial_cash * equity_factor
        # Benchmark buy-and-hold of the same series, starting on the
        # same first active day, at the same initial_cash.
        first_idx = active_mask.idxmax() if active_mask.any() else None
        if first_idx is None:
            return pd.DataFrame(columns=BACKTEST_COLUMNS)
        benchmark_return = raw_return.where(active_mask, 0.0)
        benchmark_equity = cfg.initial_cash * (
            (1.0 + benchmark_return).cumprod()
            .where(active_mask).ffill().fillna(1.0)
        )
        cash_equity = pd.Series(cfg.initial_cash, index=prep.index)
        out = pd.DataFrame({
            "date": prep["date"].values,
            "asset": cfg.asset,
            "close": prep["close"].values,
            "sma": sig_df["sma"].values,
            "signal": sig_df["signal"].values,
            "position": position.values,
            "raw_return": raw_return.values,
            "strategy_return": strategy_return.values,
            "strategy_equity": strategy_equity.values,
            "benchmark_buyhold_return": benchmark_return.values,
            "benchmark_buyhold_equity": benchmark_equity.values,
            "cash_equity": cash_equity.values,
        })
        # Trim warmup so the file starts on the first active day.
        out = out.loc[active_mask.values].reset_index(drop=True)
        return out


# ---------------------------------------------------------------------------
# Backtest output schema (locked)
# ---------------------------------------------------------------------------
BACKTEST_COLUMNS = [
    "date", "asset", "close", "sma", "signal", "position",
    "raw_return", "strategy_return", "strategy_equity",
    "benchmark_buyhold_return", "benchmark_buyhold_equity",
    "cash_equity",
]


# ---------------------------------------------------------------------------
# Pure-function metrics (used by the research orchestrator)
# ---------------------------------------------------------------------------
def _annualisation() -> int:
    return TRADING_DAYS_PER_YEAR


def total_return_from_returns(returns: pd.Series) -> float:
    if returns is None or len(returns) == 0:
        return 0.0
    return float((1.0 + returns.fillna(0.0)).prod() - 1.0)


def annualised_sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe using daily simple returns; rf assumed 0.
    Returns 0.0 if std is zero or fewer than 2 observations."""
    if returns is None or returns.empty:
        return 0.0
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.sqrt(_annualisation()) * r.mean() / sd)


def max_drawdown_from_equity(equity: pd.Series) -> float:
    """Worst peak-to-trough drawdown of an equity curve, expressed as
    a non-positive fraction (e.g. -0.18 means a 18% drawdown)."""
    if equity is None or equity.empty:
        return 0.0
    e = equity.dropna()
    if e.empty:
        return 0.0
    peak = e.cummax()
    dd = e / peak - 1.0
    return float(dd.min())


def trade_count_from_position(position: pd.Series) -> int:
    """Number of times the position changes (signal flips). The first
    transition from NaN to a defined value is not a trade."""
    if position is None or position.empty:
        return 0
    p = position.dropna().astype(float)
    if len(p) < 2:
        return 0
    return int((p.diff().fillna(0.0) != 0.0).sum())


def exposure_pct_from_position(position: pd.Series) -> float:
    if position is None or position.empty:
        return 0.0
    p = position.dropna().astype(float)
    if p.empty:
        return 0.0
    return float(100.0 * p.mean())


# ---------------------------------------------------------------------------
# Invariants the orchestrator (and tests) re-check
# ---------------------------------------------------------------------------
def assert_long_cash_only(position: pd.Series) -> None:
    """Position must be in {0, 1}. Any other value is a violation of
    the locked invariants (no shorts, no leverage)."""
    p = position.dropna().astype(float)
    if p.empty:
        return
    if (p < 0).any():
        raise ValueError("position contains negative values "
                          "(short exposure forbidden)")
    if (p > 1).any():
        raise ValueError("position contains values > 1 "
                          "(leverage forbidden)")
    extras = sorted(set(p.unique()) - {0.0, 1.0})
    if extras:
        raise ValueError(f"position values must be {{0, 1}} only; got "
                          f"extras: {extras}")


def assert_no_lookahead(close: pd.Series, sma: pd.Series,
                          cfg: FXEODTrendConfig) -> None:
    """Sanity check: SMA at index t must equal mean(close[t-N+1:t+1]),
    not an off-by-one window that includes future data."""
    n = cfg.lookback_days
    valid_idx = sma.dropna().index
    for i in valid_idx[:5].tolist() + valid_idx[-5:].tolist():
        expected = float(close.iloc[max(0, i - n + 1): i + 1].mean())
        got = float(sma.iloc[i])
        if not np.isclose(expected, got, atol=1e-12, rtol=1e-12):
            raise ValueError(
                f"SMA lookahead suspect at index {i}: "
                f"expected={expected}, got={got}"
            )
