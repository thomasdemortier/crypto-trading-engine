"""
Research toolkit.

Five honest, conservative research lenses on the engine. None of these
modules optimise parameters or pick a winning configuration — they only
measure how a *fixed* strategy behaves across slices of the data we already
have, and they report the results as-is.

All modules:
  * go through the same `RiskEngine` (fees, slippage, position cap, daily
    loss cap, stop-loss) — strategies cannot bypass risk controls
  * use next-bar-open fills (`config.BACKTEST.fill_on_signal_close = False`)
  * compute metrics with `performance.compute_metrics`
  * can be invoked from the CLI (`python main.py research_*`) and from the
    Streamlit Research Lab

The contract for the saved CSVs is documented at the top of each helper.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import backtester, config, data_collector, performance, utils
from . import regime as _regime
from . import scorecard as _scorecard
from .strategies import (
    BreakoutStrategy, BuyAndHoldStrategy,
    MovingAverageCrossStrategy, RsiMaAtrStrategy,
    TrendFollowingStrategy, PullbackContinuationStrategy,
    SidewaysMeanReversionStrategy, RegimeSelectorStrategy,
    PlaceboRandomStrategy, PLACEBOS, BENCHMARKS,
)
from .strategies.base import Strategy
from .strategies.regime_filtered import RegimeFilteredStrategy
from .strategies.regime_selector import RegimeSelectorConfig
from .strategies.breakout import BreakoutConfig
from .strategies.trend_following import TrendFollowingConfig
from .strategies.pullback_continuation import PullbackContinuationConfig
from .strategies.sideways_mean_reversion import SidewaysMeanReversionConfig

logger = utils.get_logger("cte.research")


# Minimum closed-trade count before we treat a single result as statistically
# meaningful. Anything below this triggers an "insufficient" / "INCONCLUSIVE"
# warning in the dashboard. Conservative on purpose.
MIN_TRADES_FOR_CONFIDENCE: int = 10
# A separate, lower bar for Monte Carlo: shuffle-based analysis collapses
# below ~30 trades and is uninformative.
MIN_TRADES_FOR_MONTE_CARLO: int = 30


def _ts_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Helpers — single window backtest with metrics
# ---------------------------------------------------------------------------
def _safe_run(
    *,
    strategy: Strategy,
    asset: str,
    timeframe: str,
    risk_cfg: Optional[config.RiskConfig] = None,
    strat_cfg: Optional[config.StrategyConfig] = None,
    start_ts_ms: Optional[int] = None,
    end_ts_ms: Optional[int] = None,
) -> Tuple[Optional[performance.Metrics], Optional[str], Optional[backtester.BacktestArtifacts]]:
    """Run one window backtest. Return (metrics, error, artifacts).

    Returns (None, "<reason>", None) when the run is impossible (no candle
    file, not enough warmup, alignment failure, etc.) — we never raise to the
    caller because research grids will skip-and-report rather than abort.
    """
    risk_cfg = risk_cfg or config.RISK
    strat_cfg = strat_cfg or config.STRATEGY
    try:
        art = backtester.run_backtest(
            assets=[asset], timeframe=timeframe,
            risk_cfg=risk_cfg, strat_cfg=strat_cfg, save=False,
            strategy=strategy,
            start_ts_ms=start_ts_ms, end_ts_ms=end_ts_ms,
        )
    except utils.LiveTradingForbiddenError:
        # NEVER swallow the safety lock. If someone has flipped
        # LIVE_TRADING_ENABLED, refuse to do research too.
        raise
    except FileNotFoundError as e:
        return None, f"missing data: {e}", None
    except RuntimeError as e:
        return None, str(e), None
    except Exception as e:  # noqa: BLE001
        return None, f"backtest error: {type(e).__name__}: {e}", None

    metrics = performance.compute_metrics(
        art.equity_curve, art.trades, art.asset_close_curves,
        starting_capital=risk_cfg.starting_capital,
    )
    return metrics, None, art


def _metrics_row(asset: str, timeframe: str, label: str,
                 m: performance.Metrics) -> Dict:
    return {
        "asset": asset, "timeframe": timeframe, "label": label,
        "total_return_pct": m.total_return_pct,
        "buy_and_hold_return_pct": m.buy_and_hold_return_pct,
        "buy_and_hold_max_drawdown_pct": m.buy_and_hold_max_drawdown_pct,
        "strategy_vs_bh_pct": m.strategy_vs_bh_pct,
        "drawdown_vs_bh_pct": m.drawdown_vs_bh_pct,
        "max_drawdown_pct": m.max_drawdown_pct,
        "win_rate_pct": m.win_rate_pct,
        "num_trades": int(m.num_trades),
        "profit_factor": m.profit_factor,
        "fees_paid": m.fees_paid,
        "slippage_cost": m.slippage_cost,
        "exposure_time_pct": m.exposure_time_pct,
        "sharpe_ratio": m.sharpe_ratio,
        "sortino_ratio": m.sortino_ratio,
        "calmar_ratio": m.calmar_ratio,
        "starting_capital": m.starting_capital,
        "final_portfolio_value": m.final_portfolio_value,
    }


# ---------------------------------------------------------------------------
# Phase 1 — Multi-timeframe comparison
# ---------------------------------------------------------------------------
def timeframe_comparison(
    assets: Sequence[str] = ("BTC/USDT", "ETH/USDT"),
    timeframes: Sequence[str] = ("1h", "4h", "1d"),
    strategy: Optional[Strategy] = None,
    risk_cfg: Optional[config.RiskConfig] = None,
    strat_cfg: Optional[config.StrategyConfig] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Run the SAME fixed strategy across every (asset, timeframe) pair and
    report metrics.

    CSV columns: see `_metrics_row` plus an `error` column populated when a
    run could not complete (e.g. no candle file)."""
    utils.assert_paper_only()
    strategy = strategy or RsiMaAtrStrategy()
    rows: List[Dict] = []
    for asset in assets:
        for tf in timeframes:
            metrics, err, _ = _safe_run(
                strategy=strategy, asset=asset, timeframe=tf,
                risk_cfg=risk_cfg, strat_cfg=strat_cfg,
            )
            if err is not None:
                rows.append({
                    "asset": asset, "timeframe": tf, "label": strategy.name,
                    "error": err,
                })
                continue
            row = _metrics_row(asset, tf, strategy.name, metrics)
            row["error"] = None
            rows.append(row)
    df = pd.DataFrame(rows)
    if save:
        utils.write_df(df, config.RESULTS_DIR / "research_timeframe_comparison.csv")
    return df


# ---------------------------------------------------------------------------
# Phase 2 — Walk-forward
# ---------------------------------------------------------------------------
def _ms_per_day() -> int:
    return 24 * 60 * 60 * 1000


def _build_windows(first_ts: int, last_ts: int,
                   in_sample_days: int, oos_days: int, step_days: int
                   ) -> List[Dict[str, int]]:
    """Generate a list of {is_start, is_end, oos_start, oos_end} windows.

    `is_end` is exactly one millisecond before `oos_start` so the in-sample
    and out-of-sample regions are strictly disjoint — the test
    `test_walk_forward_no_window_overlap` asserts this invariant.
    """
    day = _ms_per_day()
    is_ms = in_sample_days * day
    oos_ms = oos_days * day
    step_ms = step_days * day

    windows: List[Dict[str, int]] = []
    cursor = first_ts + is_ms
    while cursor + oos_ms <= last_ts:
        is_start = cursor - is_ms
        is_end = cursor - 1
        oos_start = cursor
        oos_end = cursor + oos_ms - 1
        windows.append({
            "is_start_ms": is_start, "is_end_ms": is_end,
            "oos_start_ms": oos_start, "oos_end_ms": oos_end,
        })
        cursor += step_ms
    return windows


def walk_forward(
    assets: Sequence[str] = ("BTC/USDT", "ETH/USDT"),
    timeframes: Sequence[str] = ("1h", "4h", "1d"),
    in_sample_days: int = 90,
    oos_days: int = 30,
    step_days: int = 30,
    strategy: Optional[Strategy] = None,
    risk_cfg: Optional[config.RiskConfig] = None,
    strat_cfg: Optional[config.StrategyConfig] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Walk-forward analysis with disjoint in-sample / out-of-sample windows.

    v1 does NOT optimise parameters — the in-sample portion exists only as
    a warmup buffer for indicators (so MA200, RSI etc. are populated by the
    time we enter the OOS region). The strategy runs unchanged on each OOS
    window and is compared against buy-and-hold over the SAME OOS dates.
    """
    utils.assert_paper_only()
    strategy = strategy or RsiMaAtrStrategy()
    risk_cfg = risk_cfg or config.RISK
    strat_cfg = strat_cfg or config.STRATEGY
    rows: List[Dict] = []
    for asset in assets:
        for tf in timeframes:
            try:
                df = data_collector.load_candles(asset, tf)
            except FileNotFoundError as e:
                rows.append({
                    "asset": asset, "timeframe": tf, "window": 0,
                    "error": f"missing data: {e}",
                })
                continue
            if df.empty:
                rows.append({"asset": asset, "timeframe": tf, "window": 0,
                             "error": "empty candle file"})
                continue
            first_ts = int(df["timestamp"].iloc[0])
            last_ts = int(df["timestamp"].iloc[-1])
            windows = _build_windows(first_ts, last_ts,
                                     in_sample_days, oos_days, step_days)
            if not windows:
                rows.append({
                    "asset": asset, "timeframe": tf, "window": 0,
                    "error": (
                        f"insufficient history for walk-forward "
                        f"({in_sample_days + oos_days} days needed, "
                        f"{(last_ts - first_ts) / _ms_per_day():.0f} available)"
                    ),
                })
                continue

            # Run the full backtest once per (asset, tf) on all data, then
            # slice the equity curve / trades into OOS windows. This is much
            # faster than re-running the backtester per window and gives
            # identical numbers because the simulation is path-deterministic.
            metrics_full, err, art = _safe_run(
                strategy=strategy, asset=asset, timeframe=tf,
                risk_cfg=risk_cfg, strat_cfg=strat_cfg,
            )
            if err is not None or art is None:
                rows.append({
                    "asset": asset, "timeframe": tf, "window": 0,
                    "error": err or "no artifacts",
                })
                continue

            for w_idx, w in enumerate(windows, start=1):
                eq_window = art.equity_curve[
                    (art.equity_curve["timestamp"] >= w["oos_start_ms"]) &
                    (art.equity_curve["timestamp"] <= w["oos_end_ms"])
                ].reset_index(drop=True)
                if len(eq_window) < 2:
                    rows.append({
                        "asset": asset, "timeframe": tf, "window": w_idx,
                        "is_start_iso": _iso(w["is_start_ms"]),
                        "is_end_iso": _iso(w["is_end_ms"]),
                        "oos_start_iso": _iso(w["oos_start_ms"]),
                        "oos_end_iso": _iso(w["oos_end_ms"]),
                        "error": "no equity points in OOS window",
                    })
                    continue
                trades_window = (
                    art.trades if art.trades.empty else
                    art.trades[
                        (art.trades["timestamp_ms"] >= w["oos_start_ms"]) &
                        (art.trades["timestamp_ms"] <= w["oos_end_ms"])
                    ]
                )
                # Per-window starting capital is the equity at OOS start.
                start_cap = float(eq_window["equity"].iloc[0])
                m = performance.compute_metrics(
                    eq_window, trades_window, art.asset_close_curves,
                    starting_capital=start_cap,
                )
                rows.append({
                    "asset": asset, "timeframe": tf, "window": w_idx,
                    "is_start_iso": _iso(w["is_start_ms"]),
                    "is_end_iso": _iso(w["is_end_ms"]),
                    "oos_start_iso": _iso(w["oos_start_ms"]),
                    "oos_end_iso": _iso(w["oos_end_ms"]),
                    "strategy_return_pct": m.total_return_pct,
                    "buy_and_hold_return_pct": m.buy_and_hold_return_pct,
                    "strategy_vs_bh_pct": m.strategy_vs_bh_pct,
                    "max_drawdown_pct": m.max_drawdown_pct,
                    "win_rate_pct": m.win_rate_pct,
                    "num_trades": int(m.num_trades),
                    "profit_factor": m.profit_factor,
                    "fees_paid": m.fees_paid,
                    "exposure_time_pct": m.exposure_time_pct,
                    "error": None,
                })
    df = pd.DataFrame(rows)
    if save:
        utils.write_df(df, config.RESULTS_DIR / "walk_forward_results.csv")
    return df


def _iso(ts_ms: int) -> str:
    return pd.to_datetime(int(ts_ms), unit="ms", utc=True).isoformat()


# ---------------------------------------------------------------------------
# Phase 2b — Per-strategy walk-forward
# ---------------------------------------------------------------------------
def placebo_comparison(
    wf_by_strategy_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Per (strategy, asset, timeframe) compare OOS stability + mean OOS
    return against the placebo. A strategy "beats placebo" only if it has
    BOTH a higher stability score AND a higher mean OOS return.

    If trade count is too low we mark the comparison INCONCLUSIVE rather
    than letting an under-traded strategy spuriously win.
    """
    if wf_by_strategy_df is None:
        p = config.RESULTS_DIR / "walk_forward_by_strategy.csv"
        if not p.exists() or p.stat().st_size == 0:
            empty = pd.DataFrame()
            if save:
                utils.write_df(empty, config.RESULTS_DIR / "placebo_comparison.csv")
            return empty
        wf_by_strategy_df = pd.read_csv(p)

    df = wf_by_strategy_df.copy()
    if "strategy_name" not in df.columns:
        empty = pd.DataFrame()
        if save:
            utils.write_df(empty, config.RESULTS_DIR / "placebo_comparison.csv")
        return empty
    if "error" in df.columns:
        df = df[df["error"].isna()]
    if df.empty:
        empty = pd.DataFrame()
        if save:
            utils.write_df(empty, config.RESULTS_DIR / "placebo_comparison.csv")
        return empty

    # Compute per-(strategy, asset, tf) aggregates.
    def _agg(group: pd.DataFrame) -> Dict[str, float]:
        n = len(group)
        if n == 0:
            return {"stability_pct": 0.0, "mean_return_pct": 0.0,
                    "mean_trades": 0.0, "n_windows": 0}
        n_both = int(((group["strategy_return_pct"] > 0)
                      & (group["strategy_vs_bh_pct"] > 0)).sum())
        return {
            "stability_pct": (n_both / n * 100.0),
            "mean_return_pct": float(group["strategy_return_pct"].mean()),
            "mean_trades": float(group["num_trades"].mean()),
            "n_windows": n,
        }

    aggs = {
        keys: _agg(grp) for keys, grp in
        df.groupby(["strategy_name", "asset", "timeframe"], dropna=False)
    }
    placebo_aggs = {
        (a, t): v for (s, a, t), v in aggs.items() if s in PLACEBOS
    }

    rows = []
    for (strat, asset, tf), agg in aggs.items():
        if strat in PLACEBOS or strat in BENCHMARKS:
            continue
        plac = placebo_aggs.get((asset, tf))
        if plac is None:
            rows.append({
                "strategy_name": strat, "asset": asset, "timeframe": tf,
                "strategy_oos_stability": agg["stability_pct"],
                "placebo_oos_stability": None,
                "strategy_mean_oos_return": agg["mean_return_pct"],
                "placebo_mean_oos_return": None,
                "strategy_mean_trades": agg["mean_trades"],
                "placebo_mean_trades": None,
                "strategy_beats_placebo": False,
                "notes": "no placebo OOS rows for this (asset, timeframe)",
            })
            continue
        # Conservative threshold — the strategy must beat placebo on BOTH
        # stability AND mean OOS return to count.
        if agg["mean_trades"] < 3.0:
            verdict = False
            note = (f"INCONCLUSIVE — only {agg['mean_trades']:.1f} mean "
                    f"trades per window")
        else:
            beats_stab = agg["stability_pct"] > plac["stability_pct"]
            beats_ret = agg["mean_return_pct"] > plac["mean_return_pct"]
            verdict = bool(beats_stab and beats_ret)
            if verdict:
                note = "beats placebo on both stability and mean return"
            elif not beats_stab and not beats_ret:
                note = "does NOT beat placebo on either metric — likely no signal"
            elif beats_stab:
                note = "beats placebo stability but not mean return"
            else:
                note = "beats placebo mean return but not stability"
        rows.append({
            "strategy_name": strat, "asset": asset, "timeframe": tf,
            "strategy_oos_stability": agg["stability_pct"],
            "placebo_oos_stability": plac["stability_pct"],
            "strategy_mean_oos_return": agg["mean_return_pct"],
            "placebo_mean_oos_return": plac["mean_return_pct"],
            "strategy_mean_trades": agg["mean_trades"],
            "placebo_mean_trades": plac["mean_trades"],
            "strategy_beats_placebo": verdict,
            "notes": note,
        })
    out = (pd.DataFrame(rows)
           .sort_values(["strategy_name", "asset", "timeframe"])
           if rows else pd.DataFrame())
    if save:
        utils.write_df(out, config.RESULTS_DIR / "placebo_comparison.csv")
    return out


def data_coverage_audit(
    assets: Sequence[str] = ("BTC/USDT", "ETH/USDT"),
    timeframes: Sequence[str] = ("1h", "4h", "1d"),
    requested_lookback_days: Optional[int] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Audit cached candle coverage. Reports per (asset, timeframe):
    actual_start, actual_end, candle_count, coverage_days,
    enough_for_walk_forward (≥ 120 days = IS + OOS)."""
    rows: List[Dict] = []
    for asset in assets:
        for tf in timeframes:
            try:
                candles = data_collector.load_candles(asset, tf)
            except FileNotFoundError as e:
                rows.append({
                    "asset": asset, "timeframe": tf,
                    "requested_start": None, "actual_start": None,
                    "actual_end": None, "candle_count": 0,
                    "coverage_days": 0.0,
                    "enough_for_walk_forward": False,
                    "requested_lookback_days": requested_lookback_days,
                    "notes": f"missing data: {e}",
                })
                continue
            if candles.empty:
                rows.append({
                    "asset": asset, "timeframe": tf,
                    "requested_start": None, "actual_start": None,
                    "actual_end": None, "candle_count": 0,
                    "coverage_days": 0.0,
                    "enough_for_walk_forward": False,
                    "requested_lookback_days": requested_lookback_days,
                    "notes": "empty candle file",
                })
                continue
            start = pd.to_datetime(candles["datetime"].iloc[0], utc=True,
                                   errors="coerce")
            end = pd.to_datetime(candles["datetime"].iloc[-1], utc=True,
                                 errors="coerce")
            coverage_days = (
                float((end - start).total_seconds() / 86400.0)
                if pd.notna(start) and pd.notna(end) else 0.0
            )
            enough = coverage_days >= 120.0
            gap_info = data_collector.validate_gaps(candles, tf)
            note_parts = []
            if requested_lookback_days is not None:
                requested_start = (pd.Timestamp.now(tz="UTC")
                                   - pd.Timedelta(days=requested_lookback_days))
                if pd.notna(start) and start > requested_start + pd.Timedelta(days=2):
                    note_parts.append(
                        f"only {coverage_days:.0f} days available — "
                        f"requested {requested_lookback_days}"
                    )
            else:
                requested_start = None
            if not enough:
                note_parts.append("insufficient for 90/30 walk-forward")
            if gap_info["gap_count"] > 0:
                note_parts.append(
                    f"{gap_info['gap_count']} gap(s); largest "
                    f"{gap_info['largest_gap_bars']} bars"
                )
            rows.append({
                "asset": asset, "timeframe": tf,
                "requested_start": (str(requested_start)
                                    if requested_start is not None else None),
                "actual_start": str(start), "actual_end": str(end),
                "candle_count": int(len(candles)),
                "expected_bars": gap_info["expected_bars"],
                "gap_count": gap_info["gap_count"],
                "largest_gap_bars": gap_info["largest_gap_bars"],
                "coverage_days": coverage_days,
                "enough_for_walk_forward": enough,
                "requested_lookback_days": requested_lookback_days,
                "notes": "; ".join(note_parts) if note_parts else "ok",
            })
    out = pd.DataFrame(rows)
    if save:
        utils.write_df(out, config.RESULTS_DIR / "data_coverage.csv")
    return out


def _wf_strategies_default() -> List[Tuple[str, Strategy]]:
    """Strategies the per-strategy walk-forward runs by default. We exclude
    `buy_and_hold` (benchmark — its OOS path is trivially correlated with
    the OOS B&H benchmark) but INCLUDE `placebo_random` so the audit can
    compare every tradable strategy against the random baseline."""
    return [
        ("rsi_ma_atr", RsiMaAtrStrategy()),
        ("ma_cross", MovingAverageCrossStrategy()),
        ("breakout", BreakoutStrategy()),
        ("trend_following", TrendFollowingStrategy()),
        ("pullback_continuation", PullbackContinuationStrategy()),
        ("sideways_mean_reversion", SidewaysMeanReversionStrategy()),
        ("regime_selector", RegimeSelectorStrategy()),
        ("placebo_random", PlaceboRandomStrategy()),
    ]


def walk_forward_by_strategy(
    strategies: Optional[Sequence[Tuple[str, Strategy]]] = None,
    assets: Sequence[str] = ("BTC/USDT", "ETH/USDT"),
    timeframes: Sequence[str] = ("1h", "4h", "1d"),
    in_sample_days: int = 90,
    oos_days: int = 30,
    step_days: int = 30,
    risk_cfg: Optional[config.RiskConfig] = None,
    strat_cfg: Optional[config.StrategyConfig] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Run walk-forward for EACH strategy independently. Saves
    `results/walk_forward_by_strategy.csv` with a `strategy_name` column
    so the scorecard can look up per-strategy OOS stability without
    reusing one strategy's WF as a proxy for another.
    """
    utils.assert_paper_only()
    strategies = list(strategies or _wf_strategies_default())
    total = len(strategies)
    frames: List[pd.DataFrame] = []
    for idx, (name, strat) in enumerate(strategies, start=1):
        logger.info(
            "walk_forward_by_strategy [%d/%d] %s on %s × %s",
            idx, total, name, list(assets), list(timeframes),
        )
        try:
            df = walk_forward(
                assets=assets, timeframes=timeframes,
                in_sample_days=in_sample_days, oos_days=oos_days,
                step_days=step_days,
                strategy=strat, risk_cfg=risk_cfg, strat_cfg=strat_cfg,
                save=False,
            )
        except Exception as e:  # noqa: BLE001
            df = pd.DataFrame([{
                "asset": "—", "timeframe": "—", "window": 0,
                "error": f"walk_forward failed: {type(e).__name__}: {e}",
            }])
        df = df.copy()
        df["strategy_name"] = name
        frames.append(df)
    out = (pd.concat(frames, ignore_index=True)
           if frames else pd.DataFrame())
    if save:
        utils.write_df(out, config.RESULTS_DIR / "walk_forward_by_strategy.csv")
    return out


# ---------------------------------------------------------------------------
# Phase 3 — Strategy comparison
# ---------------------------------------------------------------------------
def strategy_comparison(
    strategies: Optional[Sequence[Strategy]] = None,
    assets: Sequence[str] = ("BTC/USDT", "ETH/USDT"),
    timeframes: Sequence[str] = ("1h", "4h", "1d"),
    risk_cfg: Optional[config.RiskConfig] = None,
    strat_cfg: Optional[config.StrategyConfig] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Run multiple strategies through the SAME risk engine, fees, slippage,
    and execution model. The whole point is to make comparison fair."""
    utils.assert_paper_only()
    if strategies is None:
        strategies = [
            RsiMaAtrStrategy(),
            BuyAndHoldStrategy(),
            MovingAverageCrossStrategy(fast=50, slow=200),
            BreakoutStrategy(entry_window=20, exit_window=10),
            TrendFollowingStrategy(),
            PullbackContinuationStrategy(),
            SidewaysMeanReversionStrategy(),
            # Regime-filtered variants — same base strategies through the
            # bear-trend defensive filter. NOT crowned as winners; just
            # measured.
            RegimeFilteredStrategy(RsiMaAtrStrategy()),
            RegimeFilteredStrategy(TrendFollowingStrategy()),
            # Regime-conditional selector: trend_following in bull/low-vol,
            # sideways_mean_reversion in sideways/low-vol, cash otherwise.
            RegimeSelectorStrategy(),
            # Statistical control — random entries, fixed seed. NEVER tradable.
            PlaceboRandomStrategy(),
        ]
    rows: List[Dict] = []
    total = len(strategies) * len(assets) * len(timeframes)
    counter = 0
    for strat in strategies:
        for asset in assets:
            for tf in timeframes:
                counter += 1
                logger.info(
                    "strategy_comparison [%d/%d] %s on %s %s",
                    counter, total, strat.name, asset, tf,
                )
                metrics, err, _ = _safe_run(
                    strategy=strat, asset=asset, timeframe=tf,
                    risk_cfg=risk_cfg, strat_cfg=strat_cfg,
                )
                if err is not None:
                    rows.append({
                        "strategy": strat.name, "asset": asset, "timeframe": tf,
                        "error": err,
                    })
                    continue
                row = _metrics_row(asset, tf, strat.name, metrics)
                row["strategy"] = strat.name
                row["error"] = None
                rows.append(row)
    df = pd.DataFrame(rows)
    if save:
        utils.write_df(df, config.RESULTS_DIR / "strategy_comparison.csv")
    return df


# ---------------------------------------------------------------------------
# Phase 4 — Robustness checks
# ---------------------------------------------------------------------------
def _rsi_variants(strat_cfg: Optional[config.StrategyConfig]) -> List[Tuple[str, Strategy]]:
    """Param grid for RSI/MA/ATR. Each variant returns a (label, Strategy)."""
    out: List[Tuple[str, Strategy]] = []
    base = strat_cfg or config.STRATEGY
    for buy in (30, 35, 40):
        for sell in (60, 65, 70):
            for ma_long in (100, 200):
                cfg_v = config.StrategyConfig(
                    rsi_period=base.rsi_period,
                    rsi_buy_threshold=float(buy),
                    rsi_sell_threshold=float(sell),
                    ma_short=base.ma_short,
                    ma_long=ma_long,
                    atr_period=base.atr_period,
                    atr_pct_max=base.atr_pct_max,
                    volume_ma_period=base.volume_ma_period,
                    min_history_candles=max(ma_long + 20,
                                            base.min_history_candles),
                )
                label = f"rsi_buy={buy}|sell={sell}|ma_long={ma_long}"
                out.append((label, RsiMaAtrStrategy(cfg_v)))
    return out


def _ma_variants() -> List[Tuple[str, Strategy]]:
    out: List[Tuple[str, Strategy]] = []
    for fast in (20, 50):
        for slow in (100, 200):
            if fast >= slow:
                continue
            label = f"fast={fast}|slow={slow}"
            out.append((label, MovingAverageCrossStrategy(fast=fast, slow=slow)))
    return out


def _breakout_variants() -> List[Tuple[str, Strategy]]:
    """Small grid (also kept for the legacy `robustness_results.csv`).
    Used as a quick fragility check, not as a parameter optimiser."""
    out: List[Tuple[str, Strategy]] = []
    for entry in (20, 50):
        for exit_ in (10, 20):
            label = f"entry={entry}|exit={exit_}"
            out.append((label, BreakoutStrategy(entry_window=entry,
                                                exit_window=exit_)))
    return out


def _breakout_variants_extended() -> List[Tuple[str, Strategy]]:
    """Extended grid for `robustness_by_strategy.csv`. Same fragility-test
    intent — never used to crown a winner."""
    out: List[Tuple[str, Strategy]] = []
    for entry in (20, 30, 50):
        for exit_ in (10, 15, 20):
            for vol_w in (20, 30):
                for max_atr in (4.0, 5.0, 6.0):
                    cfg = BreakoutConfig(
                        entry_window=entry, exit_window=exit_,
                        volume_ma_window=vol_w, max_atr_pct=max_atr,
                    )
                    label = (f"entry={entry}|exit={exit_}|"
                             f"vol_w={vol_w}|max_atr={max_atr}")
                    out.append((label, BreakoutStrategy(cfg=cfg)))
    return out


def _trend_following_variants() -> List[Tuple[str, Strategy]]:
    out: List[Tuple[str, Strategy]] = []
    for fast in (20, 30):
        for mid in (50, 75):
            for slow in (150, 200):
                for rsi_rec in (35, 40, 45):
                    cfg = TrendFollowingConfig(
                        fast_ma=fast, mid_ma=mid, slow_ma=slow,
                        rsi_recovery_level=float(rsi_rec),
                    )
                    label = (f"fast={fast}|mid={mid}|slow={slow}|"
                             f"rsi_rec={rsi_rec}")
                    out.append((label, TrendFollowingStrategy(cfg=cfg)))
    return out


def _pullback_variants() -> List[Tuple[str, Strategy]]:
    out: List[Tuple[str, Strategy]] = []
    for pb_ma in (20, 50):
        for rsi_min in (30, 35, 40):
            for rsi_max in (50, 55, 60):
                if rsi_min >= rsi_max:
                    continue
                for max_atr in (4.0, 5.0, 6.0):
                    cfg = PullbackContinuationConfig(
                        pullback_ma=pb_ma, rsi_min=float(rsi_min),
                        rsi_max=float(rsi_max), max_atr_pct=max_atr,
                    )
                    label = (f"pb_ma={pb_ma}|rsi_min={rsi_min}|"
                             f"rsi_max={rsi_max}|max_atr={max_atr}")
                    out.append((label, PullbackContinuationStrategy(cfg=cfg)))
    return out


def _sideways_mr_variants() -> List[Tuple[str, Strategy]]:
    out: List[Tuple[str, Strategy]] = []
    for bb_w in (20, 30):
        for bb_std in (1.8, 2.0, 2.2):
            for rsi_min in (20, 25, 30):
                for rsi_max in (40, 45, 50):
                    if rsi_min >= rsi_max:
                        continue
                    cfg = SidewaysMeanReversionConfig(
                        bb_window=bb_w, bb_std=bb_std,
                        rsi_min=float(rsi_min), rsi_max=float(rsi_max),
                    )
                    label = (f"bb_w={bb_w}|bb_std={bb_std}|"
                             f"rsi_min={rsi_min}|rsi_max={rsi_max}")
                    out.append((label, SidewaysMeanReversionStrategy(cfg=cfg)))
    return out


def _regime_selector_variants() -> List[Tuple[str, Strategy]]:
    """Small, conservative fragility grid for the regime selector.

    Tests two routing axes: which bull-strategy to delegate to (trend vs
    pullback) and the sideways strategy's RSI-band width / max-ATR
    tolerance. Deliberately tiny to avoid optimisation-as-research."""
    out: List[Tuple[str, Strategy]] = []
    bull_choices = [
        ("trend", TrendFollowingStrategy()),
        ("pullback", PullbackContinuationStrategy()),
    ]
    sideways_rsi_bands = [(25, 45), (30, 50)]
    sideways_atr_caps = (4.0, 5.0, 6.0)
    for bull_name, bull_strat in bull_choices:
        for rsi_min, rsi_max in sideways_rsi_bands:
            for max_atr in sideways_atr_caps:
                sw_cfg = SidewaysMeanReversionConfig(
                    rsi_min=float(rsi_min), rsi_max=float(rsi_max),
                    max_atr_pct=max_atr,
                )
                sw_strat = SidewaysMeanReversionStrategy(cfg=sw_cfg)
                label = (f"bull={bull_name}|sideways_rsi={rsi_min}-{rsi_max}"
                         f"|max_atr={max_atr}")
                out.append((label, RegimeSelectorStrategy(
                    bull_strategy=bull_strat, sideways_strategy=sw_strat,
                    name_suffix=label,
                )))
    return out


def robustness(
    assets: Sequence[str] = ("BTC/USDT", "ETH/USDT"),
    timeframes: Sequence[str] = ("1h", "4h", "1d"),
    risk_cfg: Optional[config.RiskConfig] = None,
    strat_cfg: Optional[config.StrategyConfig] = None,
    save: bool = True,
) -> pd.DataFrame:
    """For each strategy family, sweep small parameter variations and
    record one row per (family, variant, asset, timeframe). The point is
    NOT to pick a winner; it is to see whether tiny parameter changes
    obliterate the result (= overfit)."""
    utils.assert_paper_only()
    families: List[Tuple[str, List[Tuple[str, Strategy]]]] = [
        ("rsi_ma_atr", _rsi_variants(strat_cfg)),
        ("ma_cross", _ma_variants()),
        ("breakout", _breakout_variants()),
    ]
    rows: List[Dict] = []
    for family, variants in families:
        for variant_label, strat in variants:
            for asset in assets:
                for tf in timeframes:
                    metrics, err, _ = _safe_run(
                        strategy=strat, asset=asset, timeframe=tf,
                        risk_cfg=risk_cfg, strat_cfg=strat_cfg,
                    )
                    if err is not None:
                        rows.append({
                            "family": family, "variant": variant_label,
                            "asset": asset, "timeframe": tf, "error": err,
                        })
                        continue
                    row = _metrics_row(asset, tf, variant_label, metrics)
                    row["family"] = family
                    row["variant"] = variant_label
                    row["error"] = None
                    rows.append(row)
    df = pd.DataFrame(rows)
    if save:
        utils.write_df(df, config.RESULTS_DIR / "robustness_results.csv")
    return df


def robustness_by_strategy(
    assets: Sequence[str] = ("BTC/USDT", "ETH/USDT"),
    timeframes: Sequence[str] = ("1h", "4h", "1d"),
    risk_cfg: Optional[config.RiskConfig] = None,
    strat_cfg: Optional[config.StrategyConfig] = None,
    families_filter: Optional[Sequence[str]] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Robustness sweep across all tradable strategy families. Each row
    records (strategy_name, parameter_set, asset, timeframe, metrics).

    `families_filter` (e.g. `["regime_selector"]`) limits the sweep to a
    subset — used by `--strategy` on the CLI to avoid the multi-hour 1h
    sweep when only one family matters.

    Output: results/robustness_by_strategy.csv.
    """
    utils.assert_paper_only()
    all_families: List[Tuple[str, List[Tuple[str, Strategy]]]] = [
        ("rsi_ma_atr", _rsi_variants(strat_cfg)),
        ("ma_cross", _ma_variants()),
        ("breakout", _breakout_variants_extended()),
        ("trend_following", _trend_following_variants()),
        ("pullback_continuation", _pullback_variants()),
        ("sideways_mean_reversion", _sideways_mr_variants()),
        ("regime_selector", _regime_selector_variants()),
    ]
    if families_filter:
        keep = set(families_filter)
        families = [(n, vs) for (n, vs) in all_families if n in keep]
        if not families:
            logger.warning(
                "robustness_by_strategy: no families match filter %s",
                families_filter,
            )
    else:
        families = all_families
    total_runs = sum(
        len(vs) * len(assets) * len(timeframes) for _, vs in families
    )
    logger.info(
        "robustness_by_strategy: %d families × variants × %d assets × "
        "%d timeframes = %d total runs",
        len(families), len(assets), len(timeframes), total_runs,
    )
    rows: List[Dict] = []
    counter = 0
    for family, variants in families:
        for variant_label, strat in variants:
            for asset in assets:
                for tf in timeframes:
                    counter += 1
                    if counter == 1 or counter % 25 == 0 or counter == total_runs:
                        logger.info(
                            "robustness [%d/%d] %s::%s on %s %s",
                            counter, total_runs, family, variant_label,
                            asset, tf,
                        )
                    metrics, err, _ = _safe_run(
                        strategy=strat, asset=asset, timeframe=tf,
                        risk_cfg=risk_cfg, strat_cfg=strat_cfg,
                    )
                    if err is not None:
                        rows.append({
                            "strategy_name": family, "family": family,
                            "parameter_set": variant_label,
                            "variant": variant_label,
                            "asset": asset, "timeframe": tf, "error": err,
                        })
                        continue
                    row = _metrics_row(asset, tf, variant_label, metrics)
                    row["strategy_name"] = family
                    row["family"] = family
                    row["parameter_set"] = variant_label
                    row["variant"] = variant_label
                    row["error"] = None
                    rows.append(row)
    df = pd.DataFrame(rows)
    if save:
        utils.write_df(df, config.RESULTS_DIR / "robustness_by_strategy.csv")
    return df


# ---------------------------------------------------------------------------
# Phase 5 — Monte Carlo trade-order simulation
# ---------------------------------------------------------------------------
def monte_carlo_from_trades(
    trades_df: pd.DataFrame,
    starting_capital: float,
    n_sim: int = 1000,
    seed: Optional[int] = 42,
    save: bool = True,
) -> Dict:
    """Shuffle realised round-trip P&L `n_sim` times and report the
    distribution of final equity, probability of loss, and tail percentiles.

    Trade order matters when:
      * there are sequential drawdowns that compound losses, or
      * position sizing references current equity (we do).
    Shuffling answers: "if luck had reordered these same trades, how
    often would I still have ended green?"

    Returns a dict including a `simulations_df` (1000 rows × {final_value,
    return_pct, max_dd_pct}) suitable for the dashboard.
    """
    if trades_df is None or trades_df.empty:
        return {"ok": False, "reason": "no trades", "n_trades": 0}

    sells = trades_df[trades_df["side"] == "SELL"]
    if sells.empty:
        return {"ok": False, "reason": "no closed round-trips", "n_trades": 0}

    pnls = sells["realized_pnl"].astype(float).to_numpy()
    n_trades = len(pnls)
    if n_trades < MIN_TRADES_FOR_MONTE_CARLO:
        return {
            "ok": False,
            "reason": (f"too few trades ({n_trades}) for meaningful "
                       f"Monte Carlo (need ≥ {MIN_TRADES_FOR_MONTE_CARLO})"),
            "n_trades": n_trades,
        }

    rng = np.random.default_rng(seed)
    final_values = np.empty(n_sim, dtype=float)
    return_pcts = np.empty(n_sim, dtype=float)
    max_dds = np.empty(n_sim, dtype=float)

    for i in range(n_sim):
        order = rng.permutation(n_trades)
        path = starting_capital + np.cumsum(pnls[order])
        path = np.concatenate(([starting_capital], path))
        final_values[i] = path[-1]
        return_pcts[i] = (path[-1] / starting_capital - 1.0) * 100.0
        running_max = np.maximum.accumulate(path)
        dd = (path / running_max) - 1.0
        max_dds[i] = float(dd.min()) * 100.0

    sim_df = pd.DataFrame({
        "sim_id": np.arange(n_sim),
        "final_value": final_values,
        "return_pct": return_pcts,
        "max_drawdown_pct": max_dds,
    })

    summary = {
        "ok": True,
        "n_trades": n_trades,
        "n_simulations": n_sim,
        "starting_capital": float(starting_capital),
        "actual_total_pnl": float(pnls.sum()),
        "actual_final_value": float(starting_capital + pnls.sum()),
        "mean_final_value": float(np.mean(final_values)),
        "median_final_value": float(np.median(final_values)),
        "p05_final_value": float(np.percentile(final_values, 5)),
        "p95_final_value": float(np.percentile(final_values, 95)),
        "prob_loss": float((return_pcts < 0).mean()),
        "worst_drawdown_pct": float(np.min(max_dds)),
        "mean_drawdown_pct": float(np.mean(max_dds)),
    }
    if save:
        utils.ensure_dirs([config.RESULTS_DIR])
        utils.write_df(pd.DataFrame([summary]),
                       config.RESULTS_DIR / "monte_carlo_results.csv")
        utils.write_df(sim_df,
                       config.RESULTS_DIR / "monte_carlo_simulations.csv")
    summary["simulations_df"] = sim_df
    return summary


# ---------------------------------------------------------------------------
# Phase 6 — Research summary (PASS / FAIL / INCONCLUSIVE)
# ---------------------------------------------------------------------------
def _verdict(*, pass_: bool, inconclusive: bool, message: str) -> Dict:
    if inconclusive:
        return {"verdict": "INCONCLUSIVE", "message": message}
    return {"verdict": "PASS" if pass_ else "FAIL", "message": message}


def research_summary(
    timeframe_df: Optional[pd.DataFrame] = None,
    walk_forward_df: Optional[pd.DataFrame] = None,
    strategy_df: Optional[pd.DataFrame] = None,
    robustness_df: Optional[pd.DataFrame] = None,
    monte_carlo_summary: Optional[Dict] = None,
    target_strategy_name: str = "rsi_ma_atr",
    scorecard_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> Dict:
    """Produce a brutally honest verdict for each research lens.

    Conventions:
      * PASS    — measurable, robust evidence in favour
      * FAIL    — measurable evidence against
      * INCONCLUSIVE — not enough data to call (e.g. too few trades)
    """
    out: Dict[str, Dict] = {}

    # 1. Beats buy-and-hold (using timeframe comparison) -------------------
    if timeframe_df is None or timeframe_df.empty:
        out["beats_buy_and_hold"] = _verdict(
            pass_=False, inconclusive=True,
            message="No timeframe-comparison results available.")
    else:
        ok = timeframe_df[timeframe_df.get("error").isna()] if "error" in timeframe_df.columns else timeframe_df
        if ok.empty:
            out["beats_buy_and_hold"] = _verdict(
                pass_=False, inconclusive=True,
                message="No completed runs in timeframe comparison.")
        else:
            wins = (ok["strategy_vs_bh_pct"] > 0).sum()
            n = len(ok)
            wins_pct = wins / n * 100.0 if n else 0.0
            beats = wins_pct >= 60.0
            out["beats_buy_and_hold"] = _verdict(
                pass_=beats, inconclusive=False,
                message=(f"Strategy beat buy-and-hold in {wins}/{n} "
                         f"asset+timeframe combinations ({wins_pct:.0f}%). "
                         f"Threshold for PASS: ≥60%."))

    # 2. Works on BTC AND ETH ----------------------------------------------
    if timeframe_df is None or timeframe_df.empty:
        out["works_on_btc_and_eth"] = _verdict(
            pass_=False, inconclusive=True,
            message="No timeframe-comparison results available.")
    else:
        ok = timeframe_df[timeframe_df.get("error").isna()] if "error" in timeframe_df.columns else timeframe_df
        per_asset: Dict[str, bool] = {}
        for asset in sorted(ok["asset"].unique()):
            sub = ok[ok["asset"] == asset]
            per_asset[asset] = bool(((sub["strategy_vs_bh_pct"] > 0).mean()) > 0.5)
        all_pass = bool(per_asset) and all(per_asset.values())
        out["works_on_btc_and_eth"] = _verdict(
            pass_=all_pass, inconclusive=False,
            message=(f"Per-asset majority-of-timeframes wins vs B&H: "
                     f"{per_asset}. PASS requires every tested asset to "
                     f"beat B&H in a majority of timeframes."))

    # 3. Works across timeframes -------------------------------------------
    if timeframe_df is None or timeframe_df.empty:
        out["works_across_timeframes"] = _verdict(
            pass_=False, inconclusive=True,
            message="No timeframe-comparison results available.")
    else:
        ok = timeframe_df[timeframe_df.get("error").isna()] if "error" in timeframe_df.columns else timeframe_df
        per_tf: Dict[str, bool] = {}
        for tf in sorted(ok["timeframe"].unique()):
            sub = ok[ok["timeframe"] == tf]
            per_tf[tf] = bool(((sub["strategy_vs_bh_pct"] > 0).mean()) > 0.5)
        all_pass = bool(per_tf) and all(per_tf.values())
        out["works_across_timeframes"] = _verdict(
            pass_=all_pass, inconclusive=False,
            message=(f"Per-timeframe majority-of-assets wins vs B&H: "
                     f"{per_tf}. PASS requires every tested timeframe to "
                     f"beat B&H in a majority of assets."))

    # 4. Works out of sample (walk-forward) --------------------------------
    if walk_forward_df is None or walk_forward_df.empty:
        out["works_out_of_sample"] = _verdict(
            pass_=False, inconclusive=True,
            message="No walk-forward results available.")
    else:
        ok = walk_forward_df[walk_forward_df.get("error").isna()] if "error" in walk_forward_df.columns else walk_forward_df
        if ok.empty:
            out["works_out_of_sample"] = _verdict(
                pass_=False, inconclusive=True,
                message="No completed walk-forward windows.")
        else:
            stable_wins = ((ok["strategy_return_pct"] > 0)
                           & (ok["strategy_vs_bh_pct"] > 0)).sum()
            n = len(ok)
            score = stable_wins / n * 100.0 if n else 0.0
            pass_ = score >= 70.0
            out["works_out_of_sample"] = _verdict(
                pass_=pass_, inconclusive=False,
                message=(f"Stability score: {score:.1f}% of OOS windows "
                         f"are profitable AND beat B&H ({stable_wins}/{n}). "
                         f"PASS requires ≥70%."))

    # 5. Robust to small parameter changes ---------------------------------
    if robustness_df is None or robustness_df.empty:
        out["robust_to_parameters"] = _verdict(
            pass_=False, inconclusive=True,
            message="No robustness results available.")
    else:
        ok = robustness_df[robustness_df.get("error").isna()] if "error" in robustness_df.columns else robustness_df
        if ok.empty:
            out["robust_to_parameters"] = _verdict(
                pass_=False, inconclusive=True,
                message="No completed robustness runs.")
        else:
            family = ok[ok["family"] == target_strategy_name]
            if family.empty:
                family = ok
            beat = (family["strategy_vs_bh_pct"] > 0).sum()
            tot = len(family)
            beat_pct = beat / tot * 100.0 if tot else 0.0
            pass_ = beat_pct >= 60.0
            out["robust_to_parameters"] = _verdict(
                pass_=pass_, inconclusive=False,
                message=(f"{beat}/{tot} parameter variants of "
                         f"{target_strategy_name} beat B&H "
                         f"({beat_pct:.0f}%). PASS requires ≥60%."))

    # 6. Trades enough to be meaningful -----------------------------------
    if timeframe_df is None or timeframe_df.empty:
        out["statistically_meaningful_trade_count"] = _verdict(
            pass_=False, inconclusive=True,
            message="No timeframe-comparison results to inspect.")
    else:
        ok = timeframe_df[timeframe_df.get("error").isna()] if "error" in timeframe_df.columns else timeframe_df
        if ok.empty:
            out["statistically_meaningful_trade_count"] = _verdict(
                pass_=False, inconclusive=True,
                message="No completed runs to inspect.")
        else:
            mean_trades = float(ok["num_trades"].mean())
            pass_ = mean_trades >= MIN_TRADES_FOR_CONFIDENCE
            out["statistically_meaningful_trade_count"] = _verdict(
                pass_=pass_, inconclusive=False,
                message=(f"Mean closed round-trips per (asset, timeframe): "
                         f"{mean_trades:.1f}. PASS requires "
                         f"≥{MIN_TRADES_FOR_CONFIDENCE}."))

    # 7. Best strategy by scorecard ---------------------------------------
    if scorecard_df is None or scorecard_df.empty:
        out["best_tradable_by_scorecard"] = _verdict(
            pass_=False, inconclusive=True,
            message="No scorecard available (run research_all).")
    else:
        picks = _scorecard.best_picks(scorecard_df)
        best_tr = picks.get("best_tradable")
        beats = picks.get("any_tradable_beats_benchmark")
        if picks["any_pass"]:
            out["best_tradable_by_scorecard"] = _verdict(
                pass_=True, inconclusive=False,
                message=(
                    f"PASS tradable strategies: {picks['n_pass']}; best — "
                    f"{best_tr['strategy_name']} on {best_tr['asset']} "
                    f"{best_tr['timeframe']} (score {best_tr['total_score']}). "
                    f"Tradable beats benchmark: {beats}."),
            )
        elif picks["n_watchlist"] > 0:
            out["best_tradable_by_scorecard"] = _verdict(
                pass_=False, inconclusive=True,
                message=(
                    f"No PASS tradable strategies. Watchlist: "
                    f"{picks['n_watchlist']}, FAIL: {picks['n_fail']}, "
                    f"INCONCLUSIVE: {picks['n_inconclusive']}. "
                    f"Best tradable so far: "
                    f"{(best_tr or {}).get('strategy_name', 'n/a')} "
                    f"(score {(best_tr or {}).get('total_score', 'n/a')}). "
                    f"Tradable beats benchmark: {beats}."),
            )
        else:
            out["best_tradable_by_scorecard"] = _verdict(
                pass_=False, inconclusive=False,
                message=(
                    f"No PASS or WATCHLIST tradable strategies. "
                    f"FAIL: {picks['n_fail']}, INCONCLUSIVE: "
                    f"{picks['n_inconclusive']}. "
                    f"Tradable beats benchmark: {beats}."),
            )

    # 7b. Regime-selector specific check ----------------------------------
    if scorecard_df is None or scorecard_df.empty:
        out["regime_selector_outcome"] = _verdict(
            pass_=False, inconclusive=True,
            message="No scorecard available — cannot evaluate regime selector.",
        )
    else:
        sel_rows = scorecard_df[
            scorecard_df["strategy_name"].astype(str).str.startswith("regime_selector")
        ]
        if sel_rows.empty:
            out["regime_selector_outcome"] = _verdict(
                pass_=False, inconclusive=True,
                message="regime_selector not found in scorecard rows.",
            )
        else:
            sel_best = sel_rows.sort_values("total_score", ascending=False).iloc[0]
            # Compare against the best non-regime-selector tradable.
            tradable = scorecard_df[
                (~scorecard_df["is_benchmark"].astype(bool))
                & (~scorecard_df["strategy_name"].astype(str)
                    .str.startswith("regime_selector"))
            ]
            best_other = (tradable.sort_values("total_score", ascending=False)
                          .iloc[0].to_dict() if not tradable.empty else None)
            beats_others = (
                best_other is not None
                and float(sel_best["total_score"]) > float(best_other["total_score"])
            )
            improved_oos = float(sel_best.get("walk_forward_score", 0)) > 0
            sel_pass = sel_best["verdict"] == "PASS"
            if sel_pass:
                out["regime_selector_outcome"] = _verdict(
                    pass_=True, inconclusive=False,
                    message=(
                        f"regime_selector PASSed on {sel_best['asset']} "
                        f"{sel_best['timeframe']} (score "
                        f"{sel_best['total_score']}). Beats best "
                        f"always-on tradable: {beats_others}. "
                        f"Improved OOS stability: {improved_oos}."),
                )
            else:
                out["regime_selector_outcome"] = _verdict(
                    pass_=False, inconclusive=False,
                    message=(
                        f"regime_selector did NOT reach PASS. Best row: "
                        f"{sel_best['asset']} {sel_best['timeframe']} "
                        f"(verdict {sel_best['verdict']}, score "
                        f"{sel_best['total_score']}). Beats best always-on "
                        f"tradable: {beats_others}. Improved OOS stability: "
                        f"{improved_oos}."),
                )

    # 8. Should it be paper-traded further? -------------------------------
    pass_count = sum(1 for v in out.values() if v["verdict"] == "PASS")
    fail_count = sum(1 for v in out.values() if v["verdict"] == "FAIL")
    incon_count = sum(1 for v in out.values() if v["verdict"] == "INCONCLUSIVE")
    if pass_count >= 5 and fail_count == 0:
        worth = _verdict(
            pass_=True, inconclusive=False,
            message=(f"Worth more paper-trading research — "
                     f"{pass_count} PASS / {fail_count} FAIL / "
                     f"{incon_count} INCONCLUSIVE."))
    elif fail_count >= 3:
        worth = _verdict(
            pass_=False, inconclusive=False,
            message=(f"Not worth paper-trading further as configured — "
                     f"{pass_count} PASS / {fail_count} FAIL / "
                     f"{incon_count} INCONCLUSIVE."))
    else:
        worth = _verdict(
            pass_=False, inconclusive=True,
            message=(f"Mixed evidence — {pass_count} PASS / "
                     f"{fail_count} FAIL / {incon_count} INCONCLUSIVE. "
                     f"Re-run with more data before concluding."))
    out["worth_paper_trading_further"] = worth

    bundle = {
        "generated_at_utc": _ts_now_iso(),
        "checks": out,
        "monte_carlo": (
            {k: v for k, v in (monte_carlo_summary or {}).items()
             if k != "simulations_df"}
            if monte_carlo_summary else None
        ),
    }
    if save:
        rows = [
            {"check": k, "verdict": v["verdict"], "message": v["message"]}
            for k, v in out.items()
        ]
        utils.write_df(pd.DataFrame(rows),
                       config.RESULTS_DIR / "research_summary.csv")
    return bundle


# ---------------------------------------------------------------------------
# Market regime analysis
# ---------------------------------------------------------------------------
def regime_analysis(
    assets: Sequence[str] = ("BTC/USDT", "ETH/USDT"),
    timeframes: Sequence[str] = ("1h", "4h", "1d"),
    save: bool = True,
) -> pd.DataFrame:
    """Per (asset, timeframe) regime distribution. Saves to
    `results/regime_summary.csv` and per-(asset,tf) detail CSVs."""
    utils.assert_paper_only()
    rows: List[Dict] = []
    for asset in assets:
        for tf in timeframes:
            try:
                df = data_collector.load_candles(asset, tf)
            except FileNotFoundError as e:
                rows.append({"asset": asset, "timeframe": tf,
                             "error": f"missing data: {e}"})
                continue
            with_regimes = _regime.add_regime_columns(df)
            row = _regime.regime_summary_row(asset, tf, with_regimes)
            row["error"] = None
            rows.append(row)
            if save:
                detail_path = (
                    config.RESULTS_DIR
                    / f"regime_{utils.safe_symbol(asset)}_{tf}.csv"
                )
                cols = ["timestamp", "datetime", "close", "ma50", "ma200",
                        "ma200_slope", "ma_spread_pct", "atr_pct",
                        "trend_regime", "volatility_regime", "regime_label"]
                cols = [c for c in cols if c in with_regimes.columns]
                utils.write_df(with_regimes[cols], detail_path)
    out = pd.DataFrame(rows)
    if save:
        utils.write_df(out, config.RESULTS_DIR / "regime_summary.csv")
    return out


# ---------------------------------------------------------------------------
# Scorecard wrapper (saves to results/strategy_scorecard.csv)
# ---------------------------------------------------------------------------
def build_scorecard_from_saved(
    save: bool = True,
) -> pd.DataFrame:
    """Re-read the saved Research Lab CSVs and rebuild the scorecard.

    Prefers `walk_forward_by_strategy.csv` and `robustness_by_strategy.csv`
    (strategy-keyed) over the legacy single-strategy files. If neither set
    exists for a given strategy, that component degrades to 0 with a note
    — never reuses one strategy's data as a proxy for another.
    """
    sc_path = config.RESULTS_DIR / "strategy_comparison.csv"
    wf_by_path = config.RESULTS_DIR / "walk_forward_by_strategy.csv"
    wf_legacy_path = config.RESULTS_DIR / "walk_forward_results.csv"
    rb_by_path = config.RESULTS_DIR / "robustness_by_strategy.csv"
    rb_legacy_path = config.RESULTS_DIR / "robustness_results.csv"
    if not sc_path.exists():
        return pd.DataFrame()
    sc = pd.read_csv(sc_path)
    if wf_by_path.exists():
        wf = pd.read_csv(wf_by_path)
    elif wf_legacy_path.exists():
        wf = pd.read_csv(wf_legacy_path)
    else:
        wf = None
    if rb_by_path.exists():
        rb = pd.read_csv(rb_by_path)
    elif rb_legacy_path.exists():
        rb = pd.read_csv(rb_legacy_path)
    else:
        rb = None
    return _scorecard.build_scorecard(sc, wf, rb, save=save)


# ---------------------------------------------------------------------------
# Resumable stage runner
# ---------------------------------------------------------------------------
import json as _json
from datetime import datetime as _dt, timezone as _tz
from . import oos_audit as _oos_audit  # noqa: E402

# Canonical stage names (executed in this order when "all" is requested).
STAGES_IN_ORDER: List[str] = [
    "data_coverage",
    "regimes",
    "strategy_comparison",
    "walk_forward",
    "robustness",
    "scorecard",
    "monte_carlo",
    "oos_audit",
    "placebo_audit",
    "summary",
]
RUN_STATE_PATH = config.RESULTS_DIR / "research_run_state.json"


def _read_run_state() -> Dict:
    if not RUN_STATE_PATH.exists() or RUN_STATE_PATH.stat().st_size == 0:
        return {"started_at": None, "finished_at": None,
                "stages_completed": [], "interrupted": False, "stages": {}}
    try:
        return _json.loads(RUN_STATE_PATH.read_text())
    except Exception:  # noqa: BLE001
        return {"started_at": None, "finished_at": None,
                "stages_completed": [], "interrupted": False, "stages": {}}


def _write_run_state(state: Dict) -> None:
    utils.ensure_dirs([config.RESULTS_DIR])
    RUN_STATE_PATH.write_text(_json.dumps(state, indent=2, default=str))


def _stale_summary_paths() -> List[str]:
    """Files that represent the FINAL verdict — they must be deleted at
    the start of any partial run so an interrupted pipeline cannot leave
    a misleading 'completed' verdict in place."""
    return [
        str(config.RESULTS_DIR / "research_summary.csv"),
        str(config.RESULTS_DIR / "strategy_scorecard.csv"),
    ]


def _maybe_emit_robustness_warning(
    timeframes: Sequence[str], skip_robustness: bool,
) -> None:
    """Print a clear runtime warning before launching a 1h robustness sweep.
    No-op when robustness is skipped or when 1h is not in the timeframe list."""
    if skip_robustness:
        return
    if "1h" not in [str(t) for t in timeframes]:
        return
    logger.warning(
        "robustness on 1h is SLOW. With ~35,000 1h bars × ~250 variants × "
        "%d assets, this stage can take 30-60 minutes. Use "
        "`--skip-robustness` to skip, or `--strategy NAME` to limit to "
        "one family.",
        len(set(["BTC/USDT", "ETH/USDT"])),
    )


def run_stages(
    *,
    stages: Sequence[str] = ("all",),
    assets: Sequence[str] = ("BTC/USDT", "ETH/USDT"),
    timeframes: Sequence[str] = ("1h", "4h", "1d"),
    n_sim: int = 1000,
    skip_robustness: bool = False,
    strategy_filter: Optional[str] = None,
) -> Dict:
    """Run a subset (or all) of the research stages and persist a
    `research_run_state.json` after each one.

    `stages=("all",)` runs everything in `STAGES_IN_ORDER`. Otherwise pass
    a list of explicit stage names. Inputs that a stage requires from a
    prior stage must already exist on disk (we do NOT silently re-run
    upstream dependencies).

    `skip_robustness` toggles the heaviest stage off. `strategy_filter`
    narrows comparison + walk-forward + robustness to one named family.
    """
    utils.assert_paper_only()

    requested: List[str] = []
    if "all" in stages:
        requested = list(STAGES_IN_ORDER)
    else:
        requested = [s for s in stages if s in STAGES_IN_ORDER]
        unknown = [s for s in stages if s not in STAGES_IN_ORDER and s != "all"]
        if unknown:
            raise ValueError(
                f"unknown stage(s) {unknown}. Known: {STAGES_IN_ORDER}"
            )
    if skip_robustness and "robustness" in requested:
        requested.remove("robustness")
    _maybe_emit_robustness_warning(timeframes, skip_robustness or
                                   "robustness" not in requested)

    # Force-delete the FINAL verdict files at the start of a partial run.
    # If the stage runner aborts halfway, the verdict is recreated only
    # when the `summary` stage actually completes — never silently stale.
    if "summary" in requested or "scorecard" in requested:
        for p in _stale_summary_paths():
            try:
                Path(p).unlink()
            except FileNotFoundError:
                pass
            except Exception as e:  # noqa: BLE001
                logger.warning("could not remove %s: %s", p, e)

    state = _read_run_state()
    state["started_at"] = _ts_now_iso()
    state["finished_at"] = None
    state["interrupted"] = True  # flips to False on clean completion
    state["requested_stages"] = list(requested)
    state["assets"] = list(assets)
    state["timeframes"] = list(timeframes)
    state["skip_robustness"] = bool(skip_robustness)
    state["strategy_filter"] = strategy_filter
    state["stages"] = state.get("stages", {})
    state["stages_completed"] = state.get("stages_completed", [])
    _write_run_state(state)

    out: Dict[str, object] = {}

    def _mark(stage: str) -> None:
        state["stages"][stage] = {"finished_at": _ts_now_iso()}
        if stage not in state["stages_completed"]:
            state["stages_completed"].append(stage)
        _write_run_state(state)

    try:
        # --- helper to filter strategies + WF/comparison lists by name ---
        def _filter_strats(strats):
            if not strategy_filter:
                return strats
            return [s for s in strats
                    if (s.name if hasattr(s, "name") else s[0]) == strategy_filter]

        if "data_coverage" in requested:
            logger.info("=== stage: data_coverage ===")
            out["data_coverage_df"] = data_coverage_audit(
                assets=assets, timeframes=timeframes,
            )
            _mark("data_coverage")

        if "regimes" in requested:
            logger.info("=== stage: regimes ===")
            out["regime_df"] = regime_analysis(assets=assets, timeframes=timeframes)
            _mark("regimes")

        if "strategy_comparison" in requested:
            logger.info("=== stage: strategy_comparison ===")
            tf_df = timeframe_comparison(assets=assets, timeframes=timeframes)
            out["timeframe_df"] = tf_df
            sc_df = strategy_comparison(assets=assets, timeframes=timeframes)
            if strategy_filter:
                sc_df = sc_df[sc_df["strategy"].astype(str) == strategy_filter]
                utils.write_df(sc_df, config.RESULTS_DIR / "strategy_comparison.csv")
            out["strategy_df"] = sc_df
            _mark("strategy_comparison")

        if "walk_forward" in requested:
            logger.info("=== stage: walk_forward ===")
            wf_df = walk_forward(assets=assets, timeframes=timeframes)
            wf_strats = _wf_strategies_default()
            if strategy_filter:
                wf_strats = [(n, s) for n, s in wf_strats if n == strategy_filter]
            wf_by_df = walk_forward_by_strategy(
                strategies=wf_strats, assets=assets, timeframes=timeframes,
            )
            out["walk_forward_df"] = wf_df
            out["walk_forward_by_strategy_df"] = wf_by_df
            _mark("walk_forward")

        if "robustness" in requested:
            logger.info("=== stage: robustness ===")
            out["robustness_df"] = robustness(assets=assets, timeframes=timeframes)
            out["robustness_by_strategy_df"] = robustness_by_strategy(
                assets=assets, timeframes=timeframes,
                families_filter=[strategy_filter] if strategy_filter else None,
            )
            _mark("robustness")

        if "scorecard" in requested:
            logger.info("=== stage: scorecard ===")
            sc_df = out.get("strategy_df")
            if sc_df is None:
                sc_path = config.RESULTS_DIR / "strategy_comparison.csv"
                sc_df = pd.read_csv(sc_path) if sc_path.exists() else pd.DataFrame()
            wf_path = config.RESULTS_DIR / "walk_forward_by_strategy.csv"
            rb_path = config.RESULTS_DIR / "robustness_by_strategy.csv"
            wf = pd.read_csv(wf_path) if wf_path.exists() else None
            rb = pd.read_csv(rb_path) if rb_path.exists() else None
            scard_df = _scorecard.build_scorecard(sc_df, wf, rb, save=True)
            out["scorecard_df"] = scard_df
            _mark("scorecard")

        if "monte_carlo" in requested:
            logger.info("=== stage: monte_carlo ===")
            mc_summary: Dict = {"ok": False, "reason": "no saved trades"}
            trades_path = config.LOGS_DIR / "trades.csv"
            if trades_path.exists() and trades_path.stat().st_size > 0:
                try:
                    tdf = pd.read_csv(trades_path)
                    mc_summary = monte_carlo_from_trades(
                        tdf, starting_capital=config.RISK.starting_capital,
                        n_sim=n_sim,
                    )
                except Exception as e:  # noqa: BLE001
                    mc_summary = {"ok": False,
                                  "reason": f"could not load trades: {e}"}
            out["monte_carlo"] = mc_summary
            _mark("monte_carlo")

        if "oos_audit" in requested:
            logger.info("=== stage: oos_audit ===")
            audit_df, summary_df = _oos_audit.audit_walk_forward(save=True)
            out["oos_audit_df"] = audit_df
            out["oos_audit_summary_df"] = summary_df
            _mark("oos_audit")

        if "placebo_audit" in requested:
            logger.info("=== stage: placebo_audit ===")
            out["placebo_comparison_df"] = placebo_comparison(save=True)
            _mark("placebo_audit")

        if "summary" in requested:
            logger.info("=== stage: summary ===")
            tf_df = out.get("timeframe_df")
            if tf_df is None:
                p = config.RESULTS_DIR / "research_timeframe_comparison.csv"
                tf_df = pd.read_csv(p) if p.exists() else None
            wf_df = out.get("walk_forward_df")
            if wf_df is None:
                p = config.RESULTS_DIR / "walk_forward_results.csv"
                wf_df = pd.read_csv(p) if p.exists() else None
            sc_df = out.get("strategy_df")
            if sc_df is None:
                p = config.RESULTS_DIR / "strategy_comparison.csv"
                sc_df = pd.read_csv(p) if p.exists() else None
            rb_df = out.get("robustness_df")
            if rb_df is None:
                p = config.RESULTS_DIR / "robustness_results.csv"
                rb_df = pd.read_csv(p) if p.exists() else None
            scard_df = out.get("scorecard_df")
            if scard_df is None:
                p = config.RESULTS_DIR / "strategy_scorecard.csv"
                scard_df = pd.read_csv(p) if p.exists() else None
            mc_summary = out.get("monte_carlo")
            summary = research_summary(
                timeframe_df=tf_df, walk_forward_df=wf_df,
                strategy_df=sc_df, robustness_df=rb_df,
                monte_carlo_summary=mc_summary,
                scorecard_df=scard_df,
            )
            out["summary"] = summary
            _mark("summary")

        # Clean completion.
        state["interrupted"] = False
        state["finished_at"] = _ts_now_iso()
        _write_run_state(state)
        return out
    except KeyboardInterrupt:
        state["interrupted"] = True
        state["finished_at"] = _ts_now_iso()
        _write_run_state(state)
        logger.warning(
            "research run interrupted after stages: %s",
            state["stages_completed"],
        )
        raise


def run_all(
    assets: Sequence[str] = ("BTC/USDT", "ETH/USDT"),
    timeframes: Sequence[str] = ("1h", "4h", "1d"),
    n_sim: int = 1000,
    skip_robustness: bool = False,
    strategy_filter: Optional[str] = None,
) -> Dict:
    """Back-compat shim — delegates to `run_stages(stages=('all',), ...)`."""
    return run_stages(
        stages=("all",), assets=assets, timeframes=timeframes,
        n_sim=n_sim, skip_robustness=skip_robustness,
        strategy_filter=strategy_filter,
    )
