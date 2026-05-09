"""
Performance analytics. All metrics are computed from the artifacts produced
by the backtester (equity curve + trade log).

Annualisation: we infer the bar interval from successive equity timestamps
and convert the per-bar return series to an annual ratio using
sqrt(bars_per_year). For a 4h timeframe with continuous bars this is
sqrt(6 * 365) ≈ sqrt(2190).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config, utils

logger = utils.get_logger("cte.perf")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _infer_bars_per_year(ts_series: pd.Series) -> float:
    if len(ts_series) < 2:
        return 365.0
    dts = pd.to_datetime(ts_series, unit="ms", utc=True).sort_values()
    median_secs = dts.diff().dropna().dt.total_seconds().median()
    if median_secs <= 0 or pd.isna(median_secs):
        return 365.0
    return (365.0 * 24.0 * 3600.0) / median_secs


def _max_drawdown(equity: pd.Series) -> tuple[float, float]:
    """Return (max_dd_abs, max_dd_pct) where pct is negative."""
    if equity.empty:
        return 0.0, 0.0
    running_max = equity.cummax()
    drawdown_abs = equity - running_max
    drawdown_pct = (equity / running_max) - 1.0
    return float(drawdown_abs.min()), float(drawdown_pct.min())


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0, 0.0) and not pd.isna(b) else 0.0


# ---------------------------------------------------------------------------
# Trade-level metrics
# ---------------------------------------------------------------------------
def _round_trip_pnls(trades: pd.DataFrame) -> pd.Series:
    """Return realised PnL per round-trip (one BUY + matching SELL)."""
    if trades.empty:
        return pd.Series(dtype=float)
    sells = trades[trades["side"] == "SELL"]
    return sells["realized_pnl"].astype(float).reset_index(drop=True)


def _trade_durations_hours(trades: pd.DataFrame) -> List[float]:
    """Approximate by pairing consecutive BUY/SELL rows per asset."""
    if trades.empty:
        return []
    durations: List[float] = []
    for asset, group in trades.groupby("asset"):
        group = group.sort_values("timestamp_ms").reset_index(drop=True)
        open_ts: Optional[int] = None
        for _, row in group.iterrows():
            if row["side"] == "BUY":
                open_ts = int(row["timestamp_ms"])
            elif row["side"] == "SELL" and open_ts is not None:
                close_ts = int(row["timestamp_ms"])
                durations.append((close_ts - open_ts) / 1000 / 3600)
                open_ts = None
    return durations


# ---------------------------------------------------------------------------
# Buy & hold
# ---------------------------------------------------------------------------
def _buy_and_hold_return(price_curves: Dict[str, pd.DataFrame],
                         starting_capital: float,
                         start_ts_ms: Optional[int] = None,
                         end_ts_ms: Optional[int] = None) -> float:
    """Equal-weight allocation across assets, evaluated over the SAME time
    window the strategy traded over.

    `start_ts_ms` / `end_ts_ms` are millisecond UTC epoch bounds; if provided,
    each asset's price curve is sliced to the bars within [start_ts, end_ts]
    before computing entry/exit prices. This avoids the apples-to-oranges
    comparison of measuring B&H over the full dataset (including warmup)
    while the strategy only operates after warmup."""
    if not price_curves:
        return 0.0
    n = len(price_curves)
    per_asset = starting_capital / n
    total_end = 0.0
    for df in price_curves.values():
        sub = df
        if start_ts_ms is not None:
            sub = sub[sub["timestamp"] >= start_ts_ms]
        if end_ts_ms is not None:
            sub = sub[sub["timestamp"] <= end_ts_ms]
        if len(sub) < 2:
            continue
        first = float(sub.iloc[0]["close"])
        last = float(sub.iloc[-1]["close"])
        total_end += per_asset * (last / first if first > 0 else 1.0)
    return (total_end / starting_capital) - 1.0 if starting_capital > 0 else 0.0


def _buy_and_hold_max_drawdown_pct(price_curves: Dict[str, pd.DataFrame],
                                   starting_capital: float,
                                   start_ts_ms: Optional[int] = None,
                                   end_ts_ms: Optional[int] = None) -> float:
    """Maximum drawdown of an equal-weight buy-and-hold portfolio over the
    same window. Returns a negative percent (or 0.0 when not computable).
    Uses per-bar mark-to-market on the union of asset timestamps within
    the window — no future data, no warmup contamination.
    """
    if not price_curves or starting_capital <= 0:
        return 0.0
    n = len(price_curves)
    per_asset = starting_capital / n
    series_list: List[pd.Series] = []
    for df in price_curves.values():
        sub = df
        if start_ts_ms is not None:
            sub = sub[sub["timestamp"] >= start_ts_ms]
        if end_ts_ms is not None:
            sub = sub[sub["timestamp"] <= end_ts_ms]
        if len(sub) < 2:
            continue
        first = float(sub.iloc[0]["close"])
        if first <= 0:
            continue
        units = per_asset / first
        s = pd.Series(
            sub["close"].astype(float).values * units,
            index=pd.to_numeric(sub["timestamp"]).astype("int64").values,
            dtype=float,
        )
        series_list.append(s)
    if not series_list:
        return 0.0
    portfolio = pd.concat(series_list, axis=1).sort_index()
    # Forward-fill within each asset (gaps when one asset has fewer bars)
    # so we always mark-to-market against the most recent available price.
    portfolio = portfolio.ffill().fillna(0.0).sum(axis=1)
    if portfolio.empty:
        return 0.0
    running_max = portfolio.cummax()
    dd = (portfolio / running_max) - 1.0
    return float(dd.min()) * 100.0


# ---------------------------------------------------------------------------
# Public: compute_metrics
# ---------------------------------------------------------------------------
@dataclass
class Metrics:
    starting_capital: float
    final_portfolio_value: float
    total_return_pct: float
    buy_and_hold_return_pct: float
    buy_and_hold_max_drawdown_pct: float  # negative % over the same window
    strategy_vs_bh_pct: float
    drawdown_vs_bh_pct: float             # strat_dd - bh_dd; positive = better than B&H
    max_drawdown_pct: float
    max_drawdown_abs: float
    win_rate_pct: float
    num_trades: int               # count of round-trips (closed)
    num_buys: int
    num_sells: int
    avg_winning_trade: float
    avg_losing_trade: float
    profit_factor: float
    fees_paid: float
    slippage_cost: float
    largest_win: float
    largest_loss: float
    avg_trade_duration_hours: float
    exposure_time_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    bars_per_year: float


def compute_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    price_curves: Dict[str, pd.DataFrame],
    starting_capital: Optional[float] = None,
) -> Metrics:
    starting_capital = starting_capital if starting_capital is not None else float(config.RISK.starting_capital)

    if equity_curve.empty:
        # Empty backtest — return zeros so the dashboard can still render.
        return Metrics(
            starting_capital=starting_capital, final_portfolio_value=starting_capital,
            total_return_pct=0.0, buy_and_hold_return_pct=0.0,
            buy_and_hold_max_drawdown_pct=0.0,
            strategy_vs_bh_pct=0.0, drawdown_vs_bh_pct=0.0,
            max_drawdown_pct=0.0, max_drawdown_abs=0.0, win_rate_pct=0.0,
            num_trades=0, num_buys=0, num_sells=0,
            avg_winning_trade=0.0, avg_losing_trade=0.0, profit_factor=0.0,
            fees_paid=0.0, slippage_cost=0.0, largest_win=0.0, largest_loss=0.0,
            avg_trade_duration_hours=0.0, exposure_time_pct=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
            bars_per_year=365.0,
        )

    eq = equity_curve["equity"].astype(float).reset_index(drop=True)
    final_value = float(eq.iloc[-1])
    total_return_pct = (final_value / starting_capital - 1.0) * 100.0
    # Bound B&H to the SAME window the strategy traded over. Without this,
    # the benchmark uses the full dataset (including warmup) and the
    # comparison is misleading.
    start_ts = int(equity_curve["timestamp"].iloc[0]) if "timestamp" in equity_curve else None
    end_ts = int(equity_curve["timestamp"].iloc[-1]) if "timestamp" in equity_curve else None
    bh = _buy_and_hold_return(price_curves, starting_capital,
                              start_ts_ms=start_ts, end_ts_ms=end_ts) * 100.0
    bh_dd_pct = _buy_and_hold_max_drawdown_pct(
        price_curves, starting_capital,
        start_ts_ms=start_ts, end_ts_ms=end_ts,
    )

    dd_abs, dd_pct = _max_drawdown(eq)
    bars_per_year = _infer_bars_per_year(equity_curve["timestamp"])

    # Period returns (per-bar simple returns)
    rets = eq.pct_change().dropna()
    sharpe = sortino = calmar = 0.0
    if len(rets) > 1 and rets.std(ddof=1) > 0:
        sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(bars_per_year))
        downside = rets[rets < 0]
        if len(downside) > 1 and downside.std(ddof=1) > 0:
            sortino = float(rets.mean() / downside.std(ddof=1) * np.sqrt(bars_per_year))
    if dd_pct < 0:
        # Annualised return / |max DD|
        years = max(len(eq) / bars_per_year, 1e-9)
        ann_ret = (final_value / starting_capital) ** (1.0 / years) - 1.0
        calmar = float(ann_ret / abs(dd_pct))

    pnls = _round_trip_pnls(trades)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = (len(wins) / len(pnls) * 100.0) if len(pnls) > 0 else 0.0
    profit_factor = _safe_div(wins.sum(), -losses.sum()) if len(pnls) > 0 else 0.0

    durations = _trade_durations_hours(trades)
    avg_dur = float(np.mean(durations)) if durations else 0.0

    # Exposure time: fraction of bars with non-zero exposure.
    if "exposure" in equity_curve:
        exposure_pct = float((equity_curve["exposure"] > 0).mean() * 100.0)
    else:
        exposure_pct = 0.0

    fees_paid = float(trades["fee"].sum()) if not trades.empty else 0.0
    slippage_cost = float(trades["slippage_cost"].sum()) if not trades.empty else 0.0

    strat_dd_pct = dd_pct * 100.0
    return Metrics(
        starting_capital=starting_capital,
        final_portfolio_value=final_value,
        total_return_pct=total_return_pct,
        buy_and_hold_return_pct=bh,
        buy_and_hold_max_drawdown_pct=bh_dd_pct,
        strategy_vs_bh_pct=total_return_pct - bh,
        # drawdowns are negative; "less bad" = closer to 0 = larger numerically
        drawdown_vs_bh_pct=(strat_dd_pct - bh_dd_pct),
        max_drawdown_pct=strat_dd_pct,
        max_drawdown_abs=dd_abs,
        win_rate_pct=win_rate,
        num_trades=len(pnls),
        num_buys=int((trades["side"] == "BUY").sum()) if not trades.empty else 0,
        num_sells=int((trades["side"] == "SELL").sum()) if not trades.empty else 0,
        avg_winning_trade=float(wins.mean()) if len(wins) else 0.0,
        avg_losing_trade=float(losses.mean()) if len(losses) else 0.0,
        profit_factor=profit_factor,
        fees_paid=fees_paid,
        slippage_cost=slippage_cost,
        largest_win=float(pnls.max()) if len(pnls) else 0.0,
        largest_loss=float(pnls.min()) if len(pnls) else 0.0,
        avg_trade_duration_hours=avg_dur,
        exposure_time_pct=exposure_pct,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        bars_per_year=bars_per_year,
    )


def save_metrics(metrics: Metrics, path=None) -> None:
    path = path or (config.RESULTS_DIR / "summary_metrics.csv")
    df = pd.DataFrame([asdict(metrics)])
    utils.write_df(df, path)


def drawdown_curve(equity: pd.Series) -> pd.Series:
    """Percentage drawdown series for the dashboard chart."""
    if equity.empty:
        return equity
    return (equity / equity.cummax()) - 1.0


# ---------------------------------------------------------------------------
# Per-asset breakdown
# ---------------------------------------------------------------------------
def per_asset_metrics(
    trades: pd.DataFrame,
    price_curves: Dict[str, pd.DataFrame],
    starting_capital: float,
    equity_curve: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Per-asset breakdown derived from the closed-trade log.

    Portfolio-level metrics (Sharpe, Sortino, drawdown, exposure-time) are
    NOT split per asset because the backtester runs a single shared cash
    pool — those numbers only make sense at the portfolio level.

    The B&H benchmark for each asset is computed on its allocated share
    (`starting_capital / num_assets`) over the same window the strategy
    actually traded over.
    """
    assets = list(price_curves.keys())
    n = max(len(assets), 1)
    allocated = starting_capital / n

    # Window bounds — same as the combined-portfolio computation.
    start_ts = end_ts = None
    if equity_curve is not None and not equity_curve.empty and "timestamp" in equity_curve:
        start_ts = int(equity_curve["timestamp"].iloc[0])
        end_ts = int(equity_curve["timestamp"].iloc[-1])

    rows: List[dict] = []
    for asset in assets:
        sub = trades[trades["asset"] == asset] if not trades.empty else trades
        sells = sub[sub["side"] == "SELL"] if not sub.empty else sub
        buys = sub[sub["side"] == "BUY"] if not sub.empty else sub

        pnls = sells["realized_pnl"].astype(float) if not sells.empty else pd.Series(dtype=float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        win_rate = (len(wins) / len(pnls) * 100.0) if len(pnls) > 0 else 0.0
        pf = _safe_div(wins.sum(), -losses.sum()) if len(pnls) > 0 else 0.0
        fees = float(sub["fee"].sum()) if not sub.empty else 0.0
        slip = float(sub["slippage_cost"].sum()) if not sub.empty else 0.0

        realized_pnl = float(pnls.sum()) if len(pnls) > 0 else 0.0
        realized_return_on_alloc_pct = (
            (realized_pnl / allocated * 100.0) if allocated > 0 else 0.0
        )

        bh_pct = _buy_and_hold_return(
            {asset: price_curves[asset]}, allocated,
            start_ts_ms=start_ts, end_ts_ms=end_ts,
        ) * 100.0

        durations = _trade_durations_hours(sub) if not sub.empty else []
        avg_dur = float(np.mean(durations)) if durations else 0.0

        rows.append({
            "asset": asset,
            "allocated_capital": allocated,
            "realized_pnl": realized_pnl,
            "realized_return_on_allocation_pct": realized_return_on_alloc_pct,
            "buy_and_hold_return_pct": bh_pct,
            "strategy_vs_bh_pct": realized_return_on_alloc_pct - bh_pct,
            "num_trades": len(pnls),
            "num_buys": len(buys),
            "num_sells": len(sells),
            "win_rate_pct": win_rate,
            "profit_factor": pf,
            "avg_winning_trade": float(wins.mean()) if len(wins) else 0.0,
            "avg_losing_trade": float(losses.mean()) if len(losses) else 0.0,
            "largest_win": float(pnls.max()) if len(pnls) else 0.0,
            "largest_loss": float(pnls.min()) if len(pnls) else 0.0,
            "fees_paid": fees,
            "slippage_cost": slip,
            "avg_trade_duration_hours": avg_dur,
        })

    return pd.DataFrame(rows)


def save_per_asset_metrics(df: pd.DataFrame, path=None) -> None:
    path = path or (config.RESULTS_DIR / "per_asset_metrics.csv")
    utils.write_df(df, path)
