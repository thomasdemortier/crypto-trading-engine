"""
Portfolio backtester.

Long-only multi-asset rebalancing simulator. Treats the portfolio as one
shared cash pool plus one position per asset, marked-to-market every bar.
Rebalances at user-specified frequency (weekly or monthly); on each
rebalance bar the strategy returns target weights and the simulator
issues SELL/BUY orders that fill at the NEXT bar's open (no lookahead),
charging fees + slippage symmetrically.

Hard rules (mirroring the single-asset risk engine — no live trading,
no leverage, no shorts):
  * weights MUST sum to <= 1.0; if a strategy returns weights that sum
    to >1, we normalise them down to 1.0 (no implicit leverage).
  * any unallocated weight stays in cash.
  * fees = `fee_pct` per side; slippage = `slippage_pct` worsens fill.

Outputs: equity curve, trade log, weights history.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config, data_collector, utils

logger = utils.get_logger("cte.portfolio")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PortfolioBacktestConfig:
    starting_capital: float = 10_000.0
    fee_pct: float = 0.0010                # 0.10% per side
    slippage_pct: float = 0.0005           # 0.05% per side
    rebalance_frequency: str = "weekly"    # "weekly" or "monthly"
    rebalance_weekday: int = 0             # Mon=0; weekly anchor
    fill_on_next_open: bool = True         # always True in v1


@dataclass
class PortfolioArtifacts:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    weights_history: pd.DataFrame
    asset_close_curves: Dict[str, pd.DataFrame]
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Universe loading
# ---------------------------------------------------------------------------
def load_universe(
    assets: Sequence[str],
    timeframe: str = "1d",
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Return (loaded_frames, missing_assets). Each frame has columns
    timestamp, datetime, open, high, low, close, volume."""
    loaded: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []
    for asset in assets:
        try:
            df = data_collector.load_candles(asset, timeframe)
            if df.empty:
                missing.append(asset)
                continue
            loaded[asset] = df.reset_index(drop=True)
        except FileNotFoundError:
            missing.append(asset)
    return loaded, missing


def _aligned_timestamps(frames: Dict[str, pd.DataFrame]) -> List[int]:
    """Common-superset of timestamps (sorted ascending). We do NOT
    intersect — each asset is forward-filled when missing on a bar where
    another asset trades. This matches how a real portfolio handles
    asynchronous data without dropping the entire date."""
    all_ts: set = set()
    for df in frames.values():
        all_ts.update(int(t) for t in df["timestamp"].astype("int64"))
    return sorted(all_ts)


def _is_rebalance_bar(prev_dt: Optional[pd.Timestamp],
                      cur_dt: pd.Timestamp,
                      frequency: str,
                      weekday: int) -> bool:
    """Decide whether `cur_dt` is a rebalance bar.

    Weekly: the FIRST bar on or after the configured weekday in each
    calendar week. Monthly: the FIRST bar of each calendar month.
    On the very first bar (`prev_dt is None`) we always rebalance.
    """
    if prev_dt is None:
        return True
    if frequency == "weekly":
        # Same week in ISO-ish terms: compare (year, week).
        prev_key = prev_dt.isocalendar()[:2]
        cur_key = cur_dt.isocalendar()[:2]
        return cur_key != prev_key
    if frequency == "monthly":
        return (prev_dt.year, prev_dt.month) != (cur_dt.year, cur_dt.month)
    raise ValueError(f"unsupported rebalance_frequency: {frequency!r}")


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------
def run_portfolio_backtest(
    portfolio_strategy: Any,
    asset_frames: Dict[str, pd.DataFrame],
    timeframe: str = "1d",
    cfg: Optional[PortfolioBacktestConfig] = None,
    start_ts_ms: Optional[int] = None,
    end_ts_ms: Optional[int] = None,
    save: bool = False,
    save_prefix: str = "portfolio_momentum",
) -> PortfolioArtifacts:
    """Run a multi-asset long-only rebalance backtest.

    `portfolio_strategy` must implement
        target_weights(asof_ts_ms, asset_frames, timeframe) -> dict[asset, weight]

    Window slicing:
      The strategy is given the FULL frames (so it can compute its long
      momentum windows). The reported equity curve and trade log only
      cover bars within [start_ts_ms, end_ts_ms].
    """
    utils.assert_paper_only()
    cfg = cfg or PortfolioBacktestConfig()
    if not asset_frames:
        return PortfolioArtifacts(
            equity_curve=pd.DataFrame(), trades=pd.DataFrame(),
            weights_history=pd.DataFrame(), asset_close_curves={},
            meta={"reason": "no asset frames"},
        )

    # Build the universal timestamp axis and per-asset close lookups.
    ts_axis = _aligned_timestamps(asset_frames)
    if start_ts_ms is not None:
        ts_axis = [t for t in ts_axis if t >= int(start_ts_ms)]
    if end_ts_ms is not None:
        ts_axis = [t for t in ts_axis if t <= int(end_ts_ms)]
    if len(ts_axis) < 2:
        return PortfolioArtifacts(
            equity_curve=pd.DataFrame(), trades=pd.DataFrame(),
            weights_history=pd.DataFrame(), asset_close_curves={},
            meta={"reason": "insufficient timestamps after slicing"},
        )

    # Per-asset (timestamp -> close) and (timestamp -> open) lookups.
    closes: Dict[str, pd.Series] = {}
    opens: Dict[str, pd.Series] = {}
    for asset, df in asset_frames.items():
        closes[asset] = pd.Series(
            df["close"].astype(float).values,
            index=df["timestamp"].astype("int64").values,
        )
        opens[asset] = pd.Series(
            df["open"].astype(float).values,
            index=df["timestamp"].astype("int64").values,
        )

    cash: float = float(cfg.starting_capital)
    units: Dict[str, float] = {a: 0.0 for a in asset_frames}
    equity_rows: List[Dict] = []
    trade_rows: List[Dict] = []
    weight_rows: List[Dict] = []
    pending_target: Optional[Dict[str, float]] = None

    prev_dt: Optional[pd.Timestamp] = None
    for i, ts in enumerate(ts_axis):
        dt = pd.to_datetime(ts, unit="ms", utc=True)

        # 1) If we have a pending rebalance from the previous bar, fill it
        #    on THIS bar's open (next-bar-open execution).
        if pending_target is not None:
            cash, units, fills = _execute_rebalance(
                cash, units, pending_target, asset_frames=asset_frames,
                opens_at=ts, opens_lookup=opens, closes_lookup=closes,
                fee_pct=cfg.fee_pct, slippage_pct=cfg.slippage_pct,
                ts=ts, dt=dt,
            )
            trade_rows.extend(fills)
            weight_rows.append({
                "timestamp": ts, "datetime": dt,
                "weights": dict(pending_target),
                "filled": True,
            })
            pending_target = None

        # 2) Mark equity at this bar's close.
        marks = {a: float(s.get(ts, np.nan)) for a, s in closes.items()}
        eq = cash
        for a, n in units.items():
            mark = marks.get(a)
            if mark is not None and not np.isnan(mark):
                eq += n * mark
        exposure = eq - cash
        equity_rows.append({
            "timestamp": ts, "datetime": dt,
            "equity": eq, "cash": cash, "exposure": exposure,
        })

        # 3) Decide on a new target if this bar starts a new
        #    rebalance period.
        if _is_rebalance_bar(prev_dt, dt, cfg.rebalance_frequency,
                             cfg.rebalance_weekday):
            target = portfolio_strategy.target_weights(
                asof_ts_ms=ts, asset_frames=asset_frames, timeframe=timeframe,
            )
            # Defensive normalisation — never imply leverage.
            total = sum(max(0.0, float(w)) for w in target.values())
            if total > 1.0 + 1e-9:
                target = {a: float(w) / total for a, w in target.items()}
            # Defer execution to next bar's open.
            if i < len(ts_axis) - 1:
                pending_target = target
            weight_rows.append({
                "timestamp": ts, "datetime": dt,
                "weights": dict(target),
                "filled": False,
            })
        prev_dt = dt

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trade_rows)
    weights_df = pd.DataFrame(weight_rows)
    asset_curves = {
        a: df[["timestamp", "datetime", "close"]].copy()
        for a, df in asset_frames.items()
    }
    meta = {
        "starting_capital": cfg.starting_capital,
        "rebalance_frequency": cfg.rebalance_frequency,
        "fee_pct": cfg.fee_pct, "slippage_pct": cfg.slippage_pct,
        "n_bars": len(equity_df),
        "n_rebalances": int(weights_df["filled"].sum()) if not weights_df.empty else 0,
        "n_trades": int(len(trades_df)),
        "asset_universe": list(asset_frames.keys()),
        "timeframe": timeframe,
    }
    art = PortfolioArtifacts(
        equity_curve=equity_df, trades=trades_df,
        weights_history=weights_df, asset_close_curves=asset_curves,
        meta=meta,
    )
    if save:
        _save_portfolio_artifacts(art, prefix=save_prefix)
    return art


def _execute_rebalance(
    cash: float,
    units: Dict[str, float],
    target_weights: Dict[str, float],
    asset_frames: Dict[str, pd.DataFrame],
    opens_at: int,
    opens_lookup: Dict[str, pd.Series],
    closes_lookup: Dict[str, pd.Series],
    fee_pct: float,
    slippage_pct: float,
    ts: int,
    dt: pd.Timestamp,
) -> Tuple[float, Dict[str, float], List[Dict]]:
    """Sell-then-buy execution at the open of bar `opens_at`. Fees and
    slippage applied symmetrically on every trade."""
    fills: List[Dict] = []
    # Gather fill prices (skip assets with no open at this bar).
    fill_prices: Dict[str, float] = {}
    for asset in set(list(units.keys()) + list(target_weights.keys())):
        px = opens_lookup.get(asset, pd.Series(dtype=float)).get(opens_at, np.nan)
        if not pd.isna(px):
            fill_prices[asset] = float(px)

    # Compute current portfolio value at fill prices (so target weights
    # are interpreted as fractions of equity at the moment of execution).
    equity_at_fill = cash
    for a, n in units.items():
        if a in fill_prices:
            equity_at_fill += n * fill_prices[a]
        else:
            # No fill price available — mark at last known close.
            mark = float(closes_lookup.get(a, pd.Series(dtype=float)).get(opens_at, np.nan))
            if not np.isnan(mark):
                equity_at_fill += n * mark

    # SELL phase: any asset whose target is below current value (or 0).
    for asset, n_held in list(units.items()):
        if n_held <= 0:
            continue
        target_w = float(target_weights.get(asset, 0.0))
        target_value = max(0.0, target_w * equity_at_fill)
        fill_px = fill_prices.get(asset)
        if fill_px is None:
            continue  # cannot trade an asset with no open today
        slipped_px = fill_px * (1.0 - slippage_pct)
        cur_value = n_held * fill_px
        if target_value < cur_value - 1e-9:
            # Sell down to target.
            sell_value = cur_value - target_value
            sell_units = sell_value / slipped_px
            # Cap by what we hold.
            sell_units = min(sell_units, n_held)
            gross = sell_units * slipped_px
            fee = gross * fee_pct
            slippage_cost = sell_units * (fill_px - slipped_px)
            cash += gross - fee
            units[asset] = n_held - sell_units
            fills.append({
                "timestamp_ms": ts, "datetime_iso": dt.isoformat(),
                "asset": asset, "side": "SELL",
                "price": slipped_px, "size": sell_units,
                "notional": gross, "fee": fee, "slippage_cost": slippage_cost,
                "reason": "rebalance: trim to target",
            })

    # BUY phase: any asset whose target value exceeds current.
    for asset, target_w in target_weights.items():
        target_value = max(0.0, float(target_w)) * equity_at_fill
        fill_px = fill_prices.get(asset)
        if fill_px is None or target_value <= 0:
            continue
        slipped_px = fill_px * (1.0 + slippage_pct)
        n_held = units.get(asset, 0.0)
        cur_value = n_held * fill_px
        gap = target_value - cur_value
        if gap <= 0:
            continue
        # Cap by available cash (defense in depth — should not bind unless
        # weights sum to > 1).
        max_spend = max(0.0, cash) * (1.0 / (1.0 + fee_pct))
        spend = min(gap, max_spend)
        if spend <= 0:
            continue
        units_bought = spend / slipped_px
        fee = spend * fee_pct
        slippage_cost = units_bought * (slipped_px - fill_px)
        cash -= (spend + fee)
        units[asset] = n_held + units_bought
        fills.append({
            "timestamp_ms": ts, "datetime_iso": dt.isoformat(),
            "asset": asset, "side": "BUY",
            "price": slipped_px, "size": units_bought,
            "notional": spend, "fee": fee, "slippage_cost": slippage_cost,
            "reason": "rebalance: top up to target",
        })

    return cash, units, fills


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _save_portfolio_artifacts(art: PortfolioArtifacts,
                              prefix: str = "portfolio_momentum") -> None:
    utils.ensure_dirs([config.RESULTS_DIR])
    utils.write_df(art.equity_curve,
                   config.RESULTS_DIR / f"{prefix}_equity.csv")
    utils.write_df(art.trades,
                   config.RESULTS_DIR / f"{prefix}_trades.csv")
    # Weights history is nested dict per row -> serialise to JSON-ish.
    if not art.weights_history.empty:
        wh = art.weights_history.copy()
        wh["weights_json"] = wh["weights"].apply(
            lambda d: ",".join(f"{k}={v:.4f}" for k, v in (d or {}).items())
        )
        wh = wh.drop(columns=["weights"])
        utils.write_df(wh, config.RESULTS_DIR / f"{prefix}_weights.csv")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def _series_from_frame(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        df["close"].astype(float).values,
        index=df["timestamp"].astype("int64").values,
    )


def benchmark_equity_curves(
    asset_frames: Dict[str, pd.DataFrame],
    starting_capital: float,
    timeframe: str = "1d",
    start_ts_ms: Optional[int] = None,
    end_ts_ms: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Return per-benchmark equity curves over the same window.

    Benchmarks computed:
      * `BTC_buy_and_hold` (if BTC/USDT in universe)
      * `ETH_buy_and_hold` (if ETH/USDT in universe)
      * `equal_weight_basket` — equal-weight all-asset hold from window
        start, no rebalance.
    """
    out: Dict[str, pd.DataFrame] = {}
    ts_axis = _aligned_timestamps(asset_frames)
    if start_ts_ms is not None:
        ts_axis = [t for t in ts_axis if t >= int(start_ts_ms)]
    if end_ts_ms is not None:
        ts_axis = [t for t in ts_axis if t <= int(end_ts_ms)]
    if len(ts_axis) < 2:
        return out

    # Single-asset buy-and-hold curves.
    for asset in ("BTC/USDT", "ETH/USDT"):
        if asset not in asset_frames:
            continue
        s = _series_from_frame(asset_frames[asset]).reindex(ts_axis).ffill()
        if s.dropna().empty:
            continue
        first = float(s.dropna().iloc[0])
        if first <= 0:
            continue
        units = starting_capital / first
        eq = s.astype(float) * units
        df = pd.DataFrame({
            "timestamp": eq.index, "equity": eq.values,
        })
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        out[f"{asset.split('/')[0]}_buy_and_hold"] = df

    # Equal-weight basket: split capital across all assets at window start.
    n = len(asset_frames)
    if n > 0:
        per_asset = starting_capital / n
        components: List[pd.Series] = []
        for asset, df in asset_frames.items():
            s = _series_from_frame(df).reindex(ts_axis).ffill()
            first_val = s.dropna()
            if first_val.empty:
                continue
            first = float(first_val.iloc[0])
            if first <= 0:
                continue
            components.append(s.astype(float) * (per_asset / first))
        if components:
            total = pd.concat(components, axis=1).ffill().sum(axis=1)
            df = pd.DataFrame({
                "timestamp": total.index, "equity": total.values,
            })
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            out["equal_weight_basket"] = df
    return out


# ---------------------------------------------------------------------------
# Metrics on a portfolio equity curve
# ---------------------------------------------------------------------------
def portfolio_metrics(equity_df: pd.DataFrame,
                      starting_capital: float,
                      bars_per_year: float = 365.0) -> Dict[str, float]:
    if equity_df is None or equity_df.empty:
        return {
            "starting_capital": float(starting_capital),
            "final_value": float(starting_capital),
            "total_return_pct": 0.0, "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0, "exposure_time_pct": 0.0,
        }
    eq = equity_df["equity"].astype(float).reset_index(drop=True)
    final = float(eq.iloc[-1])
    tot_ret = (final / starting_capital - 1.0) * 100.0
    running_max = eq.cummax()
    dd_pct = float(((eq / running_max) - 1.0).min()) * 100.0
    rets = eq.pct_change().dropna()
    sharpe = 0.0
    if len(rets) > 1 and rets.std(ddof=1) > 0:
        sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(bars_per_year))
    exposure = 0.0
    if "exposure" in equity_df.columns:
        exposure = float((equity_df["exposure"] > 0).mean() * 100.0)
    return {
        "starting_capital": float(starting_capital),
        "final_value": final,
        "total_return_pct": tot_ret,
        "max_drawdown_pct": dd_pct,
        "sharpe_ratio": sharpe,
        "exposure_time_pct": exposure,
    }
