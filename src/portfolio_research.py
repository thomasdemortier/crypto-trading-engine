"""
Portfolio research orchestration: walk-forward, multi-seed placebo, and
the portfolio scorecard. Mirrors the single-asset research pipeline but
operates at the portfolio level — same conservative bias, same hard
verdict thresholds, same "don't paper-trade unless it beats benchmark
AND beats placebo AND passes OOS" rule.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config, crypto_regime_signals as crs, portfolio_backtester as pb, utils
from .strategies.momentum_rotation import (
    MomentumRotationStrategy, MomentumRotationConfig,
    RandomRotationPlacebo, RandomRotationConfig,
)
from .strategies.regime_aware_momentum_rotation import (
    RegimeAwareMomentumRotationStrategy, RegimeAwareMomentumConfig,
    RegimeAwareRandomPlacebo, RegimeAwareRandomConfig,
)

logger = utils.get_logger("cte.portfolio_research")


# Conservative thresholds shared by walk-forward + scorecard.
MIN_REBALANCES_FOR_CONFIDENCE = 10
MIN_OOS_WINDOWS_FOR_CONFIDENCE = 5
PLACEBO_SEEDS_DEFAULT: Tuple[int, ...] = tuple(range(20))
DEFAULT_TIMEFRAME = "1d"


# ---------------------------------------------------------------------------
# Universe loading + availability report
# ---------------------------------------------------------------------------
def load_universe_with_report(
    assets: Sequence[str] = tuple(config.EXPANDED_UNIVERSE),
    timeframe: str = DEFAULT_TIMEFRAME,
    save: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Load every requested asset, report which are missing.

    Returns (frames, availability_df). Availability is also persisted to
    `results/portfolio_universe_availability.csv` so the dashboard can
    surface it without re-loading.
    """
    frames, missing = pb.load_universe(assets=assets, timeframe=timeframe)
    rows = []
    for asset in assets:
        df = frames.get(asset)
        if df is None or df.empty:
            rows.append({
                "asset": asset, "timeframe": timeframe,
                "available": False, "candle_count": 0,
                "actual_start": None, "actual_end": None,
                "coverage_days": 0.0,
            })
            continue
        start = pd.to_datetime(df["datetime"].iloc[0], utc=True)
        end = pd.to_datetime(df["datetime"].iloc[-1], utc=True)
        cov_days = float((end - start).total_seconds() / 86400.0)
        rows.append({
            "asset": asset, "timeframe": timeframe,
            "available": True, "candle_count": int(len(df)),
            "actual_start": str(start), "actual_end": str(end),
            "coverage_days": cov_days,
        })
    avail_df = pd.DataFrame(rows)
    if save:
        utils.write_df(
            avail_df,
            config.RESULTS_DIR / "portfolio_universe_availability.csv",
        )
    return frames, avail_df


# ---------------------------------------------------------------------------
# Single-window momentum backtest + benchmark comparison
# ---------------------------------------------------------------------------
def run_portfolio_momentum(
    assets: Sequence[str] = tuple(config.EXPANDED_UNIVERSE),
    timeframe: str = DEFAULT_TIMEFRAME,
    momentum_cfg: Optional[MomentumRotationConfig] = None,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """Run a full-history momentum rotation backtest. Compares against
    BTC, ETH, and equal-weight basket benchmarks."""
    utils.assert_paper_only()
    frames, _ = load_universe_with_report(assets=assets, timeframe=timeframe,
                                          save=save)
    if not frames:
        return {"ok": False, "reason": "no asset data"}
    strat = MomentumRotationStrategy(momentum_cfg)
    cfg = backtest_cfg or pb.PortfolioBacktestConfig()
    art = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames,
        timeframe=timeframe, cfg=cfg, save=save,
        save_prefix="portfolio_momentum",
    )
    bench = pb.benchmark_equity_curves(
        frames, starting_capital=cfg.starting_capital,
        timeframe=timeframe,
    )
    metrics = pb.portfolio_metrics(art.equity_curve, cfg.starting_capital)
    bench_metrics = {
        name: pb.portfolio_metrics(df, cfg.starting_capital)
        for name, df in bench.items()
    }
    rows = [{"strategy": "momentum_rotation", **metrics}]
    for name, m in bench_metrics.items():
        rows.append({"strategy": name, **m})
    cmp_df = pd.DataFrame(rows)
    if save:
        utils.write_df(
            cmp_df,
            config.RESULTS_DIR / "portfolio_momentum_comparison.csv",
        )
    return {
        "ok": True, "artifacts": art, "benchmarks": bench,
        "metrics": metrics, "bench_metrics": bench_metrics,
        "comparison_df": cmp_df,
    }


# ---------------------------------------------------------------------------
# Walk-forward (per OOS window: rebuild benchmarks from window start)
# ---------------------------------------------------------------------------
def _build_oos_windows(first_ts: int, last_ts: int,
                       in_sample_days: int = 180,
                       oos_days: int = 90,
                       step_days: int = 90) -> List[Dict[str, int]]:
    day = 24 * 60 * 60 * 1000
    is_ms = in_sample_days * day
    oos_ms = oos_days * day
    step_ms = step_days * day
    windows: List[Dict[str, int]] = []
    cursor = first_ts + is_ms
    while cursor + oos_ms <= last_ts:
        windows.append({
            "is_start_ms": cursor - is_ms, "is_end_ms": cursor - 1,
            "oos_start_ms": cursor, "oos_end_ms": cursor + oos_ms - 1,
        })
        cursor += step_ms
    return windows


def portfolio_walk_forward(
    assets: Sequence[str] = tuple(config.EXPANDED_UNIVERSE),
    timeframe: str = DEFAULT_TIMEFRAME,
    in_sample_days: int = 180,
    oos_days: int = 90,
    step_days: int = 90,
    momentum_cfg: Optional[MomentumRotationConfig] = None,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Walk-forward the momentum rotation strategy. Each OOS window is
    backtested on its own (the strategy still sees full history for
    momentum lookback because we pass the full frames; the EQUITY CURVE is
    sliced to the OOS dates only).

    Reports per window: OOS return, max DD, Sharpe, beats BTC, beats
    equal-weight basket, profitable, turnover, n_rebalances, avg holdings.
    """
    utils.assert_paper_only()
    cfg = backtest_cfg or pb.PortfolioBacktestConfig()
    frames, _ = load_universe_with_report(assets=assets, timeframe=timeframe,
                                          save=False)
    if not frames:
        out = pd.DataFrame()
        if save:
            utils.write_df(out, config.RESULTS_DIR / "portfolio_walk_forward.csv")
        return out

    # Determine common timestamp axis to choose window bounds.
    ts_axis = pb._aligned_timestamps(frames)
    if len(ts_axis) < 2:
        out = pd.DataFrame()
        if save:
            utils.write_df(out, config.RESULTS_DIR / "portfolio_walk_forward.csv")
        return out
    first_ts, last_ts = ts_axis[0], ts_axis[-1]
    windows = _build_oos_windows(first_ts, last_ts,
                                 in_sample_days, oos_days, step_days)

    strat = MomentumRotationStrategy(momentum_cfg)
    rows: List[Dict] = []
    for w_idx, w in enumerate(windows, start=1):
        logger.info(
            "portfolio_walk_forward [%d/%d] OOS %s → %s",
            w_idx, len(windows),
            pd.to_datetime(w["oos_start_ms"], unit="ms", utc=True).date(),
            pd.to_datetime(w["oos_end_ms"], unit="ms", utc=True).date(),
        )
        # Run backtest restricted to OOS dates (strategy sees full history
        # because we always pass the full frames; only the simulator's
        # trading window is sliced).
        art = pb.run_portfolio_backtest(
            portfolio_strategy=strat, asset_frames=frames,
            timeframe=timeframe, cfg=cfg,
            start_ts_ms=w["oos_start_ms"], end_ts_ms=w["oos_end_ms"],
            save=False,
        )
        if art.equity_curve.empty:
            rows.append({
                "window": w_idx,
                "oos_start_iso": pd.to_datetime(w["oos_start_ms"], unit="ms",
                                                utc=True).isoformat(),
                "oos_end_iso": pd.to_datetime(w["oos_end_ms"], unit="ms",
                                              utc=True).isoformat(),
                "error": "no equity points",
            })
            continue
        m = pb.portfolio_metrics(art.equity_curve, cfg.starting_capital)
        # Benchmarks computed over the same OOS window.
        bench = pb.benchmark_equity_curves(
            frames, starting_capital=cfg.starting_capital,
            timeframe=timeframe,
            start_ts_ms=w["oos_start_ms"], end_ts_ms=w["oos_end_ms"],
        )
        btc_ret = pb.portfolio_metrics(
            bench.get("BTC_buy_and_hold", pd.DataFrame()),
            cfg.starting_capital,
        )["total_return_pct"]
        bk_ret = pb.portfolio_metrics(
            bench.get("equal_weight_basket", pd.DataFrame()),
            cfg.starting_capital,
        )["total_return_pct"]
        n_rebalances = int(art.weights_history["filled"].sum()) if not art.weights_history.empty else 0
        avg_holdings = (
            float(art.weights_history["weights"].apply(
                lambda d: len(d) if isinstance(d, dict) else 0,
            ).mean()) if not art.weights_history.empty else 0.0
        )
        # Turnover proxy: total notional traded / (n_rebalances × equity).
        turnover = (
            float(art.trades["notional"].sum()
                  / max(n_rebalances * cfg.starting_capital, 1.0))
            if not art.trades.empty else 0.0
        )
        rows.append({
            "window": w_idx,
            "oos_start_iso": pd.to_datetime(w["oos_start_ms"], unit="ms",
                                            utc=True).isoformat(),
            "oos_end_iso": pd.to_datetime(w["oos_end_ms"], unit="ms",
                                          utc=True).isoformat(),
            "oos_return_pct": m["total_return_pct"],
            "oos_max_drawdown_pct": m["max_drawdown_pct"],
            "oos_sharpe": m["sharpe_ratio"],
            "btc_oos_return_pct": btc_ret,
            "basket_oos_return_pct": bk_ret,
            "beats_btc": bool(m["total_return_pct"] > btc_ret),
            "beats_basket": bool(m["total_return_pct"] > bk_ret),
            "profitable": bool(m["total_return_pct"] > 0),
            "n_rebalances": n_rebalances,
            "n_trades": int(len(art.trades)),
            "avg_holdings": avg_holdings,
            "turnover": turnover,
            "error": None,
        })
    out = pd.DataFrame(rows)
    if save:
        utils.write_df(out, config.RESULTS_DIR / "portfolio_walk_forward.csv")
    return out


# ---------------------------------------------------------------------------
# Multi-seed placebo
# ---------------------------------------------------------------------------
def portfolio_placebo(
    assets: Sequence[str] = tuple(config.EXPANDED_UNIVERSE),
    timeframe: str = DEFAULT_TIMEFRAME,
    seeds: Sequence[int] = PLACEBO_SEEDS_DEFAULT,
    apply_cash_filter: bool = False,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    momentum_cfg: Optional[MomentumRotationConfig] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Run the random-rotation placebo `len(seeds)` times and report
    per-seed metrics + the strategy's metric on the same window. The
    scorecard then asks whether the strategy beat the *median* placebo."""
    utils.assert_paper_only()
    cfg = backtest_cfg or pb.PortfolioBacktestConfig()
    frames, _ = load_universe_with_report(assets=assets, timeframe=timeframe,
                                          save=False)
    if not frames:
        out = pd.DataFrame()
        if save:
            utils.write_df(out, config.RESULTS_DIR / "portfolio_placebo_comparison.csv")
        return out

    # Strategy metric on full window.
    strat = MomentumRotationStrategy(momentum_cfg)
    strat_art = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames, timeframe=timeframe,
        cfg=cfg, save=False,
    )
    strat_m = pb.portfolio_metrics(strat_art.equity_curve, cfg.starting_capital)

    # Per-seed placebo runs.
    rows: List[Dict] = []
    for seed in seeds:
        plac = RandomRotationPlacebo(
            RandomRotationConfig(
                top_n=(momentum_cfg or MomentumRotationConfig()).top_n,
                seed=int(seed),
                apply_cash_filter=apply_cash_filter,
            ),
        )
        art = pb.run_portfolio_backtest(
            portfolio_strategy=plac, asset_frames=frames, timeframe=timeframe,
            cfg=cfg, save=False,
        )
        m = pb.portfolio_metrics(art.equity_curve, cfg.starting_capital)
        rows.append({
            "seed": int(seed),
            "placebo_return_pct": m["total_return_pct"],
            "placebo_max_drawdown_pct": m["max_drawdown_pct"],
            "placebo_sharpe": m["sharpe_ratio"],
            "placebo_n_trades": int(len(art.trades)),
        })
    placebo_df = pd.DataFrame(rows)
    median_ret = float(placebo_df["placebo_return_pct"].median())
    median_dd = float(placebo_df["placebo_max_drawdown_pct"].median())
    p75_dd = float(placebo_df["placebo_max_drawdown_pct"].quantile(0.75))

    summary = pd.DataFrame([{
        "strategy": "momentum_rotation",
        "strategy_return_pct": strat_m["total_return_pct"],
        "strategy_max_drawdown_pct": strat_m["max_drawdown_pct"],
        "strategy_sharpe": strat_m["sharpe_ratio"],
        "n_seeds": len(seeds),
        "placebo_median_return_pct": median_ret,
        "placebo_median_drawdown_pct": median_dd,
        "placebo_p75_drawdown_pct": p75_dd,
        "strategy_beats_median_return": bool(
            strat_m["total_return_pct"] > median_ret
        ),
        "strategy_beats_median_drawdown": bool(
            strat_m["max_drawdown_pct"] > median_dd  # closer to 0 = better
        ),
    }])
    out = pd.concat([summary, placebo_df.assign(strategy="placebo_seed_runs")],
                    ignore_index=True, sort=False)
    if save:
        utils.write_df(
            out,
            config.RESULTS_DIR / "portfolio_placebo_comparison.csv",
        )
    return out


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
PORTFOLIO_PASS = "PASS"
PORTFOLIO_WATCHLIST = "WATCHLIST"
PORTFOLIO_FAIL = "FAIL"
PORTFOLIO_INCONCLUSIVE = "INCONCLUSIVE"


def portfolio_scorecard(
    walk_forward_df: Optional[pd.DataFrame] = None,
    placebo_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Build the portfolio scorecard from saved walk-forward + placebo
    CSVs. Always reads from disk if a frame is omitted."""
    if walk_forward_df is None:
        p = config.RESULTS_DIR / "portfolio_walk_forward.csv"
        walk_forward_df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    if placebo_df is None:
        p = config.RESULTS_DIR / "portfolio_placebo_comparison.csv"
        placebo_df = pd.read_csv(p) if p.exists() else pd.DataFrame()

    if walk_forward_df.empty:
        out = pd.DataFrame([{
            "strategy_name": "momentum_rotation",
            "verdict": PORTFOLIO_INCONCLUSIVE,
            "reason": "no walk-forward data",
        }])
        if save:
            utils.write_df(out, config.RESULTS_DIR / "portfolio_scorecard.csv")
        return out

    ok = walk_forward_df
    if "error" in ok.columns:
        ok = ok[ok["error"].isna()]
    n_windows = int(len(ok))
    if n_windows < MIN_OOS_WINDOWS_FOR_CONFIDENCE:
        out = pd.DataFrame([{
            "strategy_name": "momentum_rotation",
            "n_windows": n_windows,
            "verdict": PORTFOLIO_INCONCLUSIVE,
            "reason": f"only {n_windows} OOS windows — need ≥"
                       f"{MIN_OOS_WINDOWS_FOR_CONFIDENCE}",
        }])
        if save:
            utils.write_df(out, config.RESULTS_DIR / "portfolio_scorecard.csv")
        return out

    avg_oos_return = float(ok["oos_return_pct"].mean())
    avg_oos_dd = float(ok["oos_max_drawdown_pct"].mean())
    pct_profitable = float((ok["oos_return_pct"] > 0).mean() * 100.0)
    pct_beats_btc = float(ok["beats_btc"].mean() * 100.0)
    pct_beats_basket = float(ok["beats_basket"].mean() * 100.0)
    stability = float(((ok["profitable"] & ok["beats_btc"] & ok["beats_basket"])
                       ).mean() * 100.0)
    avg_rebalances = float(ok["n_rebalances"].mean()) if "n_rebalances" in ok.columns else 0.0
    total_rebalances = int(ok["n_rebalances"].sum()) if "n_rebalances" in ok.columns else 0

    # Placebo signals.
    plac_summary = (placebo_df.iloc[[0]]
                    if not placebo_df.empty else pd.DataFrame())
    placebo_median_return = (
        float(plac_summary["placebo_median_return_pct"].iloc[0])
        if not plac_summary.empty
        and "placebo_median_return_pct" in plac_summary.columns
        else float("nan")
    )
    strategy_full_return = (
        float(plac_summary["strategy_return_pct"].iloc[0])
        if not plac_summary.empty
        and "strategy_return_pct" in plac_summary.columns
        else float("nan")
    )
    beats_placebo_median = (
        bool(strategy_full_return > placebo_median_return)
        if not (np.isnan(placebo_median_return) or np.isnan(strategy_full_return))
        else False
    )

    # Verdict rules (per spec).
    pass_required = [
        ("positive_return",
         not np.isnan(strategy_full_return) and strategy_full_return > 0),
        ("beats_btc_oos", pct_beats_btc > 50.0),
        ("beats_basket_oos", pct_beats_basket > 50.0),
        ("beats_placebo_median", beats_placebo_median),
        ("oos_stability_above_60",  stability > 60.0),
        ("at_least_10_rebalances", total_rebalances >= MIN_REBALANCES_FOR_CONFIDENCE),
    ]
    n_pass_satisfied = sum(1 for _, ok_ in pass_required if ok_)
    if all(ok_ for _, ok_ in pass_required):
        verdict = PORTFOLIO_PASS
    elif n_pass_satisfied >= len(pass_required) - 1:
        verdict = PORTFOLIO_WATCHLIST
    elif (pct_beats_btc <= 30.0 and pct_beats_basket <= 30.0
          and not beats_placebo_median):
        verdict = PORTFOLIO_FAIL
    elif total_rebalances < MIN_REBALANCES_FOR_CONFIDENCE:
        verdict = PORTFOLIO_INCONCLUSIVE
    else:
        verdict = PORTFOLIO_FAIL

    out = pd.DataFrame([{
        "strategy_name": "momentum_rotation",
        "n_windows": n_windows,
        "avg_oos_return_pct": avg_oos_return,
        "avg_oos_drawdown_pct": avg_oos_dd,
        "pct_windows_profitable": pct_profitable,
        "pct_windows_beat_btc": pct_beats_btc,
        "pct_windows_beat_basket": pct_beats_basket,
        "stability_score_pct": stability,
        "total_rebalances": total_rebalances,
        "avg_rebalances_per_window": avg_rebalances,
        "strategy_full_return_pct": strategy_full_return,
        "placebo_median_return_pct": placebo_median_return,
        "beats_placebo_median": beats_placebo_median,
        "verdict": verdict,
        "checks_passed": n_pass_satisfied,
        "checks_total": len(pass_required),
        "reason": "; ".join(
            f"{k}={ok_}" for k, ok_ in pass_required
        ),
    }])
    if save:
        utils.write_df(out, config.RESULTS_DIR / "portfolio_scorecard.csv")
    return out


# ---------------------------------------------------------------------------
# One-shot orchestrator
# ---------------------------------------------------------------------------
def run_all_portfolio(
    assets: Sequence[str] = tuple(config.EXPANDED_UNIVERSE),
    timeframe: str = DEFAULT_TIMEFRAME,
    in_sample_days: int = 180,
    oos_days: int = 90,
    step_days: int = 90,
    seeds: Sequence[int] = PLACEBO_SEEDS_DEFAULT,
) -> Dict[str, Any]:
    logger.info("=== portfolio research: full pipeline ===")
    one = run_portfolio_momentum(assets=assets, timeframe=timeframe)
    wf = portfolio_walk_forward(
        assets=assets, timeframe=timeframe,
        in_sample_days=in_sample_days, oos_days=oos_days, step_days=step_days,
    )
    plac = portfolio_placebo(assets=assets, timeframe=timeframe, seeds=seeds)
    sc = portfolio_scorecard(walk_forward_df=wf, placebo_df=plac)
    return {"single": one, "walk_forward": wf, "placebo": plac,
            "scorecard": sc}


# ===========================================================================
# Phase A: regime-aware portfolio research
# ===========================================================================
def run_regime_aware_portfolio(
    assets: Sequence[str] = tuple(config.EXPANDED_UNIVERSE),
    timeframe: str = DEFAULT_TIMEFRAME,
    momentum_cfg: Optional[RegimeAwareMomentumConfig] = None,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    save: bool = True,
) -> Dict[str, Any]:
    utils.assert_paper_only()
    frames, _ = load_universe_with_report(assets=assets, timeframe=timeframe,
                                          save=save)
    if not frames:
        return {"ok": False, "reason": "no asset data"}
    signals = crs.compute_regime_signals(asset_frames=frames,
                                         timeframe=timeframe, save=save)
    cfg = backtest_cfg or pb.PortfolioBacktestConfig()
    strat = RegimeAwareMomentumRotationStrategy(signals_df=signals,
                                                cfg=momentum_cfg)
    art = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames, timeframe=timeframe,
        cfg=cfg, save=save, save_prefix="regime_aware_portfolio",
    )
    bench = pb.benchmark_equity_curves(
        frames, starting_capital=cfg.starting_capital, timeframe=timeframe,
    )
    metrics = pb.portfolio_metrics(art.equity_curve, cfg.starting_capital)
    bench_metrics = {
        name: pb.portfolio_metrics(df, cfg.starting_capital)
        for name, df in bench.items()
    }
    # Also include the simple momentum strategy as a comparison baseline.
    simple_strat = MomentumRotationStrategy(MomentumRotationConfig())
    simple_art = pb.run_portfolio_backtest(
        portfolio_strategy=simple_strat, asset_frames=frames,
        timeframe=timeframe, cfg=cfg, save=False,
    )
    simple_metrics = pb.portfolio_metrics(simple_art.equity_curve,
                                          cfg.starting_capital)
    rows = [{"strategy": "regime_aware_momentum_rotation", **metrics}]
    rows.append({"strategy": "momentum_rotation_simple", **simple_metrics})
    for name, m in bench_metrics.items():
        rows.append({"strategy": name, **m})
    cmp_df = pd.DataFrame(rows)
    if save:
        utils.write_df(
            cmp_df,
            config.RESULTS_DIR / "regime_aware_portfolio_comparison.csv",
        )
    return {
        "ok": True, "artifacts": art, "signals": signals,
        "benchmarks": bench, "metrics": metrics,
        "bench_metrics": bench_metrics,
        "simple_metrics": simple_metrics, "comparison_df": cmp_df,
    }


def regime_aware_portfolio_walk_forward(
    assets: Sequence[str] = tuple(config.EXPANDED_UNIVERSE),
    timeframe: str = DEFAULT_TIMEFRAME,
    in_sample_days: int = 180,
    oos_days: int = 90,
    step_days: int = 90,
    momentum_cfg: Optional[RegimeAwareMomentumConfig] = None,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    save: bool = True,
) -> pd.DataFrame:
    utils.assert_paper_only()
    cfg = backtest_cfg or pb.PortfolioBacktestConfig()
    frames, _ = load_universe_with_report(assets=assets, timeframe=timeframe,
                                          save=False)
    if not frames:
        out = pd.DataFrame()
        if save:
            utils.write_df(
                out,
                config.RESULTS_DIR / "regime_aware_portfolio_walk_forward.csv",
            )
        return out

    signals = crs.compute_regime_signals(asset_frames=frames,
                                         timeframe=timeframe, save=False)
    ts_axis = pb._aligned_timestamps(frames)
    if len(ts_axis) < 2:
        out = pd.DataFrame()
        if save:
            utils.write_df(
                out,
                config.RESULTS_DIR / "regime_aware_portfolio_walk_forward.csv",
            )
        return out
    first_ts, last_ts = ts_axis[0], ts_axis[-1]
    windows = _build_oos_windows(first_ts, last_ts,
                                 in_sample_days, oos_days, step_days)

    strat = RegimeAwareMomentumRotationStrategy(signals_df=signals,
                                                cfg=momentum_cfg)
    simple = MomentumRotationStrategy(MomentumRotationConfig())
    rows: List[Dict] = []
    for w_idx, w in enumerate(windows, start=1):
        logger.info(
            "regime_aware_portfolio_walk_forward [%d/%d] OOS %s -> %s",
            w_idx, len(windows),
            pd.to_datetime(w["oos_start_ms"], unit="ms", utc=True).date(),
            pd.to_datetime(w["oos_end_ms"], unit="ms", utc=True).date(),
        )
        art = pb.run_portfolio_backtest(
            portfolio_strategy=strat, asset_frames=frames,
            timeframe=timeframe, cfg=cfg,
            start_ts_ms=w["oos_start_ms"], end_ts_ms=w["oos_end_ms"],
            save=False,
        )
        if art.equity_curve.empty:
            rows.append({
                "window": w_idx,
                "oos_start_iso": pd.to_datetime(w["oos_start_ms"], unit="ms",
                                                utc=True).isoformat(),
                "oos_end_iso": pd.to_datetime(w["oos_end_ms"], unit="ms",
                                              utc=True).isoformat(),
                "error": "no equity points",
            })
            continue
        m = pb.portfolio_metrics(art.equity_curve, cfg.starting_capital)
        # Simple-momentum benchmark: same OOS window.
        simple_art = pb.run_portfolio_backtest(
            portfolio_strategy=simple, asset_frames=frames,
            timeframe=timeframe, cfg=cfg,
            start_ts_ms=w["oos_start_ms"], end_ts_ms=w["oos_end_ms"],
            save=False,
        )
        simple_m = pb.portfolio_metrics(simple_art.equity_curve,
                                        cfg.starting_capital)
        bench = pb.benchmark_equity_curves(
            frames, starting_capital=cfg.starting_capital,
            timeframe=timeframe,
            start_ts_ms=w["oos_start_ms"], end_ts_ms=w["oos_end_ms"],
        )
        btc_ret = pb.portfolio_metrics(
            bench.get("BTC_buy_and_hold", pd.DataFrame()),
            cfg.starting_capital,
        )["total_return_pct"]
        bk_ret = pb.portfolio_metrics(
            bench.get("equal_weight_basket", pd.DataFrame()),
            cfg.starting_capital,
        )["total_return_pct"]
        n_rebalances = (int(art.weights_history["filled"].sum())
                        if not art.weights_history.empty else 0)
        avg_holdings = (
            float(art.weights_history["weights"].apply(
                lambda d: len(d) if isinstance(d, dict) else 0,
            ).mean()) if not art.weights_history.empty else 0.0
        )
        rows.append({
            "window": w_idx,
            "oos_start_iso": pd.to_datetime(w["oos_start_ms"], unit="ms",
                                            utc=True).isoformat(),
            "oos_end_iso": pd.to_datetime(w["oos_end_ms"], unit="ms",
                                          utc=True).isoformat(),
            "oos_return_pct": m["total_return_pct"],
            "oos_max_drawdown_pct": m["max_drawdown_pct"],
            "oos_sharpe": m["sharpe_ratio"],
            "btc_oos_return_pct": btc_ret,
            "basket_oos_return_pct": bk_ret,
            "simple_oos_return_pct": simple_m["total_return_pct"],
            "beats_btc": bool(m["total_return_pct"] > btc_ret),
            "beats_basket": bool(m["total_return_pct"] > bk_ret),
            "beats_simple_momentum": bool(
                m["total_return_pct"] > simple_m["total_return_pct"]
            ),
            "profitable": bool(m["total_return_pct"] > 0),
            "n_rebalances": n_rebalances,
            "n_trades": int(len(art.trades)),
            "avg_holdings": avg_holdings,
            "error": None,
        })
    out = pd.DataFrame(rows)
    if save:
        utils.write_df(
            out,
            config.RESULTS_DIR / "regime_aware_portfolio_walk_forward.csv",
        )
    return out


def regime_aware_portfolio_placebo(
    assets: Sequence[str] = tuple(config.EXPANDED_UNIVERSE),
    timeframe: str = DEFAULT_TIMEFRAME,
    seeds: Sequence[int] = PLACEBO_SEEDS_DEFAULT,
    momentum_cfg: Optional[RegimeAwareMomentumConfig] = None,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Compare the regime-aware strategy against a regime-aware random
    placebo (random Top-N within the regime-allowed asset set)."""
    utils.assert_paper_only()
    cfg = backtest_cfg or pb.PortfolioBacktestConfig()
    frames, _ = load_universe_with_report(assets=assets, timeframe=timeframe,
                                          save=False)
    if not frames:
        out = pd.DataFrame()
        if save:
            utils.write_df(
                out,
                config.RESULTS_DIR / "regime_aware_portfolio_placebo.csv",
            )
        return out

    signals = crs.compute_regime_signals(asset_frames=frames,
                                         timeframe=timeframe, save=False)
    strat = RegimeAwareMomentumRotationStrategy(signals_df=signals,
                                                cfg=momentum_cfg)
    strat_art = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames, timeframe=timeframe,
        cfg=cfg, save=False,
    )
    strat_m = pb.portfolio_metrics(strat_art.equity_curve, cfg.starting_capital)

    rows: List[Dict] = []
    for seed in seeds:
        plac = RegimeAwareRandomPlacebo(
            signals_df=signals,
            cfg=RegimeAwareRandomConfig(
                top_n_alt_risk_on=(momentum_cfg or RegimeAwareMomentumConfig()).top_n_alt_risk_on,
                seed=int(seed),
            ),
        )
        art = pb.run_portfolio_backtest(
            portfolio_strategy=plac, asset_frames=frames, timeframe=timeframe,
            cfg=cfg, save=False,
        )
        m = pb.portfolio_metrics(art.equity_curve, cfg.starting_capital)
        rows.append({
            "seed": int(seed),
            "placebo_return_pct": m["total_return_pct"],
            "placebo_max_drawdown_pct": m["max_drawdown_pct"],
            "placebo_sharpe": m["sharpe_ratio"],
            "placebo_n_trades": int(len(art.trades)),
        })
    placebo_df = pd.DataFrame(rows)
    median_ret = float(placebo_df["placebo_return_pct"].median())
    median_dd = float(placebo_df["placebo_max_drawdown_pct"].median())
    p75_dd = float(placebo_df["placebo_max_drawdown_pct"].quantile(0.75))
    summary = pd.DataFrame([{
        "strategy": "regime_aware_momentum_rotation",
        "strategy_return_pct": strat_m["total_return_pct"],
        "strategy_max_drawdown_pct": strat_m["max_drawdown_pct"],
        "strategy_sharpe": strat_m["sharpe_ratio"],
        "n_seeds": len(seeds),
        "placebo_median_return_pct": median_ret,
        "placebo_median_drawdown_pct": median_dd,
        "placebo_p75_drawdown_pct": p75_dd,
        "strategy_beats_median_return": bool(
            strat_m["total_return_pct"] > median_ret
        ),
        "strategy_beats_median_drawdown": bool(
            strat_m["max_drawdown_pct"] > median_dd
        ),
    }])
    out = pd.concat([summary,
                     placebo_df.assign(strategy="placebo_seed_runs")],
                    ignore_index=True, sort=False)
    if save:
        utils.write_df(
            out,
            config.RESULTS_DIR / "regime_aware_portfolio_placebo.csv",
        )
    return out


def regime_aware_portfolio_scorecard(
    walk_forward_df: Optional[pd.DataFrame] = None,
    placebo_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Same conservative gates as the simple-momentum scorecard, plus a
    new gate: must beat the simple momentum strategy in a majority of
    OOS windows."""
    if walk_forward_df is None:
        p = config.RESULTS_DIR / "regime_aware_portfolio_walk_forward.csv"
        walk_forward_df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    if placebo_df is None:
        p = config.RESULTS_DIR / "regime_aware_portfolio_placebo.csv"
        placebo_df = pd.read_csv(p) if p.exists() else pd.DataFrame()

    if walk_forward_df.empty:
        out = pd.DataFrame([{
            "strategy_name": "regime_aware_momentum_rotation",
            "verdict": PORTFOLIO_INCONCLUSIVE,
            "reason": "no walk-forward data",
        }])
        if save:
            utils.write_df(
                out,
                config.RESULTS_DIR / "regime_aware_portfolio_scorecard.csv",
            )
        return out

    ok = walk_forward_df
    if "error" in ok.columns:
        ok = ok[ok["error"].isna()]
    n_windows = int(len(ok))
    if n_windows < MIN_OOS_WINDOWS_FOR_CONFIDENCE:
        out = pd.DataFrame([{
            "strategy_name": "regime_aware_momentum_rotation",
            "n_windows": n_windows,
            "verdict": PORTFOLIO_INCONCLUSIVE,
            "reason": f"only {n_windows} OOS windows — need ≥"
                       f"{MIN_OOS_WINDOWS_FOR_CONFIDENCE}",
        }])
        if save:
            utils.write_df(
                out,
                config.RESULTS_DIR / "regime_aware_portfolio_scorecard.csv",
            )
        return out

    avg_oos_return = float(ok["oos_return_pct"].mean())
    avg_oos_dd = float(ok["oos_max_drawdown_pct"].mean())
    pct_profitable = float((ok["oos_return_pct"] > 0).mean() * 100.0)
    pct_beats_btc = float(ok["beats_btc"].mean() * 100.0)
    pct_beats_basket = float(ok["beats_basket"].mean() * 100.0)
    pct_beats_simple = float(ok["beats_simple_momentum"].mean() * 100.0) \
        if "beats_simple_momentum" in ok.columns else 0.0
    stability = float(((ok["profitable"] & ok["beats_btc"] & ok["beats_basket"])
                       ).mean() * 100.0)
    total_rebalances = int(ok["n_rebalances"].sum()) if "n_rebalances" in ok.columns else 0

    plac_summary = (placebo_df.iloc[[0]] if not placebo_df.empty else pd.DataFrame())
    placebo_median_return = (
        float(plac_summary["placebo_median_return_pct"].iloc[0])
        if not plac_summary.empty
        and "placebo_median_return_pct" in plac_summary.columns
        else float("nan")
    )
    strategy_full_return = (
        float(plac_summary["strategy_return_pct"].iloc[0])
        if not plac_summary.empty
        and "strategy_return_pct" in plac_summary.columns
        else float("nan")
    )
    beats_placebo_median = (
        bool(strategy_full_return > placebo_median_return)
        if not (np.isnan(placebo_median_return) or np.isnan(strategy_full_return))
        else False
    )

    pass_required = [
        ("positive_return",
         not np.isnan(strategy_full_return) and strategy_full_return > 0),
        ("beats_btc_oos", pct_beats_btc > 50.0),
        ("beats_basket_oos", pct_beats_basket > 50.0),
        ("beats_simple_momentum_oos", pct_beats_simple > 50.0),
        ("beats_placebo_median", beats_placebo_median),
        ("oos_stability_above_60", stability > 60.0),
        ("at_least_10_rebalances",
         total_rebalances >= MIN_REBALANCES_FOR_CONFIDENCE),
    ]
    n_pass_satisfied = sum(1 for _, ok_ in pass_required if ok_)
    if all(ok_ for _, ok_ in pass_required):
        verdict = PORTFOLIO_PASS
    elif n_pass_satisfied >= len(pass_required) - 1:
        verdict = PORTFOLIO_WATCHLIST
    elif (pct_beats_btc <= 30.0 and pct_beats_basket <= 30.0
          and not beats_placebo_median):
        verdict = PORTFOLIO_FAIL
    elif total_rebalances < MIN_REBALANCES_FOR_CONFIDENCE:
        verdict = PORTFOLIO_INCONCLUSIVE
    else:
        verdict = PORTFOLIO_FAIL

    out = pd.DataFrame([{
        "strategy_name": "regime_aware_momentum_rotation",
        "n_windows": n_windows,
        "avg_oos_return_pct": avg_oos_return,
        "avg_oos_drawdown_pct": avg_oos_dd,
        "pct_windows_profitable": pct_profitable,
        "pct_windows_beat_btc": pct_beats_btc,
        "pct_windows_beat_basket": pct_beats_basket,
        "pct_windows_beat_simple_momentum": pct_beats_simple,
        "stability_score_pct": stability,
        "total_rebalances": total_rebalances,
        "strategy_full_return_pct": strategy_full_return,
        "placebo_median_return_pct": placebo_median_return,
        "beats_placebo_median": beats_placebo_median,
        "verdict": verdict,
        "checks_passed": n_pass_satisfied,
        "checks_total": len(pass_required),
        "reason": "; ".join(f"{k}={ok_}" for k, ok_ in pass_required),
    }])
    if save:
        utils.write_df(
            out,
            config.RESULTS_DIR / "regime_aware_portfolio_scorecard.csv",
        )
    return out


def run_all_regime_aware_portfolio(
    assets: Sequence[str] = tuple(config.EXPANDED_UNIVERSE),
    timeframe: str = DEFAULT_TIMEFRAME,
    in_sample_days: int = 180,
    oos_days: int = 90,
    step_days: int = 90,
    seeds: Sequence[int] = PLACEBO_SEEDS_DEFAULT,
) -> Dict[str, Any]:
    logger.info("=== regime-aware portfolio research: full pipeline ===")
    one = run_regime_aware_portfolio(assets=assets, timeframe=timeframe)
    wf = regime_aware_portfolio_walk_forward(
        assets=assets, timeframe=timeframe,
        in_sample_days=in_sample_days, oos_days=oos_days, step_days=step_days,
    )
    plac = regime_aware_portfolio_placebo(assets=assets, timeframe=timeframe,
                                          seeds=seeds)
    sc = regime_aware_portfolio_scorecard(walk_forward_df=wf, placebo_df=plac)
    return {"single": one, "walk_forward": wf, "placebo": plac,
            "scorecard": sc}
