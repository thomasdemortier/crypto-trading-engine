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
from .strategies import (
    BreakoutStrategy, BuyAndHoldStrategy,
    MovingAverageCrossStrategy, RsiMaAtrStrategy,
)
from .strategies.base import Strategy

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
        "strategy_vs_bh_pct": m.strategy_vs_bh_pct,
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
        ]
    rows: List[Dict] = []
    for strat in strategies:
        for asset in assets:
            for tf in timeframes:
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
    out: List[Tuple[str, Strategy]] = []
    for entry in (20, 50):
        for exit_ in (10, 20):
            label = f"entry={entry}|exit={exit_}"
            out.append((label, BreakoutStrategy(entry_window=entry,
                                                exit_window=exit_)))
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

    # 7. Should it be paper-traded further? -------------------------------
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
# Convenience: run everything
# ---------------------------------------------------------------------------
def run_all(
    assets: Sequence[str] = ("BTC/USDT", "ETH/USDT"),
    timeframes: Sequence[str] = ("1h", "4h", "1d"),
    n_sim: int = 1000,
) -> Dict:
    """Run every research lens and produce a final summary. Saves all CSVs."""
    tf_df = timeframe_comparison(assets=assets, timeframes=timeframes)
    wf_df = walk_forward(assets=assets, timeframes=timeframes)
    sc_df = strategy_comparison(assets=assets, timeframes=timeframes)
    rb_df = robustness(assets=assets, timeframes=timeframes)
    # Monte Carlo on the incumbent strategy's most recent saved trades.
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
            mc_summary = {"ok": False, "reason": f"could not load trades: {e}"}
    summary = research_summary(
        timeframe_df=tf_df, walk_forward_df=wf_df,
        strategy_df=sc_df, robustness_df=rb_df,
        monte_carlo_summary=mc_summary,
    )
    return {
        "timeframe_df": tf_df, "walk_forward_df": wf_df,
        "strategy_df": sc_df, "robustness_df": rb_df,
        "monte_carlo": mc_summary, "summary": summary,
    }
