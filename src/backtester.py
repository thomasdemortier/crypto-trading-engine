"""
Backtester.

Execution model
---------------
* On bar `t` we evaluate indicators using only data up to and including bar `t`.
* If the strategy issues a BUY/SELL on bar `t`, we fill on the OPEN of bar `t+1`.
  (This is the only honest way to avoid lookahead. You can flip
  `BACKTEST.fill_on_signal_close = True` if you want to study an optimistic
  scenario, but the default is the conservative one.)
* Stop-loss checks happen INTRA-bar against the bar's LOW. If pierced, we
  fill at the stop price — itself a simplification (real gaps can fill worse).

Multi-asset handling
--------------------
* We align the per-asset OHLCV frames on their common timestamp index.
* Each bar, we evaluate every asset, run the risk engine, then mark the
  portfolio to the bar close to record an equity point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import config, data_collector, indicators, utils
from .risk_engine import RiskEngine
from .strategies.base import Strategy as _StrategyBase
from .strategies import default_strategy as _default_strategy

logger = utils.get_logger("cte.backtest")


@dataclass
class BacktestArtifacts:
    equity_curve: pd.DataFrame      # columns: timestamp, datetime, equity, cash, exposure
    trades: pd.DataFrame
    decisions: pd.DataFrame
    final_equity: float
    starting_equity: float
    asset_close_curves: Dict[str, pd.DataFrame]  # asset → df(timestamp, close)
    meta: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading & alignment
# ---------------------------------------------------------------------------
def _load_with_indicators(asset: str, timeframe: str,
                          strat_cfg: config.StrategyConfig,
                          strategy_obj: Optional[_StrategyBase] = None) -> pd.DataFrame:
    """Load candles and apply the strategy's required indicators.

    `strategy_obj` is optional only for back-compat — if omitted, the legacy
    full RSI/MA/ATR indicator set is applied via `indicators.add_indicators`.
    """
    df = data_collector.load_candles(asset, timeframe)
    if strategy_obj is None:
        df = indicators.add_indicators(df, strat_cfg)
    else:
        df = strategy_obj.prepare(df, strat_cfg)
    df = df.reset_index(drop=True)
    return df


def _slice_to_window(df: pd.DataFrame, start_ts_ms: Optional[int],
                     end_ts_ms: Optional[int]) -> pd.DataFrame:
    """Keep rows whose `timestamp` falls in the inclusive [start, end] window.

    Indicators are computed on the FULL series before slicing, so the first
    rows of the slice still have correct (non-NaN-only) values.
    """
    if start_ts_ms is None and end_ts_ms is None:
        return df
    sub = df
    if start_ts_ms is not None:
        sub = sub[sub["timestamp"] >= int(start_ts_ms)]
    if end_ts_ms is not None:
        sub = sub[sub["timestamp"] <= int(end_ts_ms)]
    return sub.reset_index(drop=True)


def _align_frames(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Align all per-asset frames to their common set of timestamps. Returns
    frames with identical row counts and matching timestamp indices."""
    if not frames:
        return frames
    common = None
    for df in frames.values():
        ts = set(df["timestamp"].astype("int64").tolist())
        common = ts if common is None else common & ts
    if not common:
        raise RuntimeError("no overlapping timestamps across selected assets")
    common_sorted = sorted(common)
    out = {}
    for asset, df in frames.items():
        sub = df[df["timestamp"].isin(common_sorted)].copy()
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        out[asset] = sub
    return out


# ---------------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------------
def run_backtest(
    assets: Optional[List[str]] = None,
    timeframe: str = config.DEFAULT_TIMEFRAME,
    risk_cfg: Optional[config.RiskConfig] = None,
    strat_cfg: Optional[config.StrategyConfig] = None,
    backtest_cfg: Optional[config.BacktestConfig] = None,
    save: bool = True,
    strategy: Optional[_StrategyBase] = None,
    start_ts_ms: Optional[int] = None,
    end_ts_ms: Optional[int] = None,
) -> BacktestArtifacts:
    """Run the backtester.

    `strategy` defaults to the incumbent RSI/MA/ATR strategy so all
    existing call sites are unchanged. Pass any subclass of
    `src.strategies.base.Strategy` to swap.

    `start_ts_ms` / `end_ts_ms` (inclusive ms epoch UTC) restrict the
    simulation to a window. Indicators are computed on the FULL loaded
    series and only then sliced — this is the honest way to support
    walk-forward windows without losing warmup.
    """
    utils.assert_paper_only()
    assets = assets or config.ASSETS
    risk_cfg = risk_cfg or config.RISK
    strat_cfg = strat_cfg or config.STRATEGY
    backtest_cfg = backtest_cfg or config.BACKTEST
    strategy_obj: _StrategyBase = strategy or _default_strategy()

    # --- Load data + indicators -------------------------------------------
    frames = {
        a: _load_with_indicators(a, timeframe, strat_cfg, strategy_obj)
        for a in assets
    }
    if start_ts_ms is not None or end_ts_ms is not None:
        frames = {a: _slice_to_window(df, start_ts_ms, end_ts_ms)
                  for a, df in frames.items()}
    frames = _align_frames(frames)
    n_bars = len(next(iter(frames.values())))
    logger.info("backtest aligned: %d bars across %d assets (%s, strategy=%s)",
                n_bars, len(assets), timeframe, strategy_obj.name)

    min_hist = strategy_obj.min_history(strat_cfg)
    if n_bars < min_hist + 2:
        raise RuntimeError(
            f"not enough candles ({n_bars}) for backtest. need at least "
            f"{min_hist + 2}."
        )

    engine = RiskEngine(risk_cfg)
    equity_rows: List[dict] = []

    # Order of operations within iteration `i` (representing bar i closing):
    #   1. Process stop-losses against bar i (using bar i's open and low).
    #      Stop-fills happen at min(stop, bar_open) when there's a gap-down,
    #      else at the stop itself.
    #   2. Mark equity at bar i's CLOSE and record the equity point. This is
    #      the "end of bar i" snapshot, AFTER stop exits but BEFORE any new
    #      orders generated by bar i are filled.
    #   3. Generate strategy signal using bar i's data. Fills happen at bar
    #      i+1's OPEN — they will be reflected in the equity point recorded
    #      at iteration i+1. This avoids the anachronism of marking a
    #      just-filled position (at bar i+1 open) at bar i close.
    #
    # The final iteration (i = n_bars - 1) does steps 1 and 2 but NOT step 3,
    # since there is no bar i+1 to fill against.
    for i in range(min_hist, n_bars):
        marks_close: Dict[str, float] = {
            a: float(frames[a].iloc[i]["close"]) for a in assets
        }
        bar_lows: Dict[str, float] = {
            a: float(frames[a].iloc[i]["low"]) for a in assets
        }
        bar_opens: Dict[str, float] = {
            a: float(frames[a].iloc[i]["open"]) for a in assets
        }
        ts = int(frames[assets[0]].iloc[i]["timestamp"])
        dt_iso = pd.to_datetime(ts, unit="ms", utc=True).isoformat()

        # 1) Stop-losses against bar i (gap-aware fill).
        engine.check_stop_losses(ts, dt_iso, bar_lows, bar_opens, marks_close)

        # 2) Mark equity at bar i close (post-stops, pre-new-orders).
        eq = engine.equity(marks_close)
        equity_rows.append({
            "timestamp": ts,
            "datetime": pd.to_datetime(ts, unit="ms", utc=True),
            "equity": eq,
            "cash": engine.cash,
            "exposure": engine.exposure(marks_close),
        })

        # 3) Strategy signal at bar i; fill at bar i+1 open (if it exists).
        if i >= n_bars - 1:
            continue
        bar_opens_next: Dict[str, float] = {
            a: float(frames[a].iloc[i + 1]["open"]) for a in assets
        }
        for asset in assets:
            row = frames[asset].iloc[i]
            in_position = asset in engine.positions
            sig = strategy_obj.signal_for_row(asset, row, in_position, strat_cfg)
            fill_price = (
                float(row["close"]) if backtest_cfg.fill_on_signal_close
                else bar_opens_next[asset]
            )
            engine.evaluate(sig, fill_price, marks_close)

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(engine.trades_as_dicts())
    decisions_df = pd.DataFrame(engine.decisions_as_dicts())
    final_eq = float(equity_df["equity"].iloc[-1])

    asset_curves = {
        a: frames[a][["timestamp", "datetime", "close"]].copy() for a in assets
    }

    first_ts = int(equity_df["timestamp"].iloc[0]) if not equity_df.empty else None
    last_ts = int(equity_df["timestamp"].iloc[-1]) if not equity_df.empty else None
    meta = {
        "assets": list(assets),
        "timeframe": timeframe,
        "starting_capital": float(risk_cfg.starting_capital),
        "num_candles_used": int(len(equity_df)),
        "first_candle_iso": (
            pd.to_datetime(first_ts, unit="ms", utc=True).isoformat()
            if first_ts is not None else None
        ),
        "last_candle_iso": (
            pd.to_datetime(last_ts, unit="ms", utc=True).isoformat()
            if last_ts is not None else None
        ),
        "first_candle_ms": first_ts,
        "last_candle_ms": last_ts,
        "run_timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "fill_on_signal_close": bool(backtest_cfg.fill_on_signal_close),
        "strategy_name": strategy_obj.name,
    }

    artifacts = BacktestArtifacts(
        equity_curve=equity_df,
        trades=trades_df,
        decisions=decisions_df,
        final_equity=final_eq,
        starting_equity=float(risk_cfg.starting_capital),
        asset_close_curves=asset_curves,
        meta=meta,
    )

    if save:
        _save_artifacts(artifacts)

    logger.info("backtest finished: start=%.2f, end=%.2f, trades=%d, decisions=%d",
                artifacts.starting_equity, artifacts.final_equity,
                len(trades_df), len(decisions_df))
    return artifacts


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _save_artifacts(art: BacktestArtifacts) -> None:
    utils.ensure_dirs([config.RESULTS_DIR, config.LOGS_DIR])
    utils.write_df(art.equity_curve, config.RESULTS_DIR / "equity_curve.csv")
    utils.write_df(art.trades, config.LOGS_DIR / "trades.csv")
    utils.write_df(art.decisions, config.LOGS_DIR / "decisions.csv")
    # asset price curves are useful for the dashboard chart
    for asset, df in art.asset_close_curves.items():
        utils.write_df(df, config.RESULTS_DIR / f"price_{utils.safe_symbol(asset)}.csv")
    if art.meta:
        meta_path = config.RESULTS_DIR / "backtest_meta.json"
        meta_path.write_text(json.dumps(art.meta, indent=2, default=str))
