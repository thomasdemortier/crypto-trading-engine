"""
Derivatives research orchestrator.

Walk-forward + multi-seed placebo + scorecard for the
`DerivativesRotationStrategy`. Mirrors `portfolio_research.py` so the
verdict thresholds, OOS-window construction, and placebo comparison
are identical — there is no "softer" scorecard for the new signal class.

If derivatives signals fail this exact same conservative bar that every
v1 strategy failed, the verdict is FAIL. If the data is too short for
≥5 disjoint OOS windows (likely the case for OI, capped at 30d), the
verdict is INCONCLUSIVE — not PASS.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config, derivatives_signals as ds, portfolio_backtester as pb, utils
from .portfolio_research import (
    DEFAULT_TIMEFRAME, MIN_OOS_WINDOWS_FOR_CONFIDENCE,
    MIN_REBALANCES_FOR_CONFIDENCE, PLACEBO_SEEDS_DEFAULT,
    PORTFOLIO_FAIL, PORTFOLIO_INCONCLUSIVE, PORTFOLIO_PASS,
    PORTFOLIO_WATCHLIST, _build_oos_windows, load_universe_with_report,
)
from .strategies.derivatives_rotation import (
    DerivativesRotationConfig, DerivativesRotationStrategy,
)
from .strategies.momentum_rotation import (
    RandomRotationConfig, RandomRotationPlacebo,
)

logger = utils.get_logger("cte.derivatives_research")


def _ensure_signals(signals_df: Optional[pd.DataFrame],
                    symbols: Sequence[str]) -> pd.DataFrame:
    if signals_df is not None and not signals_df.empty:
        return signals_df
    p = config.RESULTS_DIR / "derivatives_signals.csv"
    if p.exists() and p.stat().st_size > 0:
        return pd.read_csv(p)
    logger.info("derivatives_signals.csv missing — computing on the fly")
    return ds.compute_all_derivatives_signals(symbols=symbols, save=True)


def _spot_assets_for(symbols: Sequence[str]) -> List[str]:
    """Map futures `BTCUSDT` -> spot `BTC/USDT` consistently with how
    the strategy keys `asset_frames`."""
    out: List[str] = []
    for s in symbols:
        if "/" in s:
            out.append(s)
        elif s.endswith("USDT"):
            out.append(f"{s[:-4]}/USDT")
        else:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Single-window full backtest
# ---------------------------------------------------------------------------
def run_derivatives_rotation(
    symbols: Sequence[str] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
                                "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT",
                                "AVAXUSDT", "LTCUSDT"),
    timeframe: str = DEFAULT_TIMEFRAME,
    rotation_cfg: Optional[DerivativesRotationConfig] = None,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    signals_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> Dict[str, Any]:
    utils.assert_paper_only()
    assets = _spot_assets_for(symbols)
    frames, _ = load_universe_with_report(assets=assets, timeframe=timeframe,
                                            save=save)
    if not frames:
        return {"ok": False, "reason": "no asset data"}
    sigs = _ensure_signals(signals_df, symbols)
    strat = DerivativesRotationStrategy(sigs, rotation_cfg)
    cfg = backtest_cfg or pb.PortfolioBacktestConfig()
    art = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames,
        timeframe=timeframe, cfg=cfg, save=save,
        save_prefix="derivatives_rotation",
    )
    bench = pb.benchmark_equity_curves(
        frames, starting_capital=cfg.starting_capital, timeframe=timeframe,
    )
    metrics = pb.portfolio_metrics(art.equity_curve, cfg.starting_capital)
    bench_metrics = {n: pb.portfolio_metrics(d, cfg.starting_capital)
                      for n, d in bench.items()}
    rows = [{"strategy": "derivatives_rotation", **metrics}]
    for name, m in bench_metrics.items():
        rows.append({"strategy": name, **m})
    cmp_df = pd.DataFrame(rows)
    if save:
        utils.write_df(cmp_df,
                       config.RESULTS_DIR / "derivatives_rotation_comparison.csv")
    return {"ok": True, "artifacts": art, "benchmarks": bench,
            "metrics": metrics, "bench_metrics": bench_metrics,
            "comparison_df": cmp_df}


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def derivatives_walk_forward(
    symbols: Sequence[str] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
                                "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT",
                                "AVAXUSDT", "LTCUSDT"),
    timeframe: str = DEFAULT_TIMEFRAME,
    in_sample_days: int = 180,
    oos_days: int = 90,
    step_days: int = 90,
    rotation_cfg: Optional[DerivativesRotationConfig] = None,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    signals_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> pd.DataFrame:
    utils.assert_paper_only()
    cfg = backtest_cfg or pb.PortfolioBacktestConfig()
    assets = _spot_assets_for(symbols)
    frames, _ = load_universe_with_report(assets=assets, timeframe=timeframe,
                                            save=False)
    if not frames:
        out = pd.DataFrame()
        if save:
            utils.write_df(out, config.RESULTS_DIR / "derivatives_walk_forward.csv")
        return out
    sigs = _ensure_signals(signals_df, symbols)

    ts_axis = pb._aligned_timestamps(frames)
    if len(ts_axis) < 2:
        out = pd.DataFrame()
        if save:
            utils.write_df(out, config.RESULTS_DIR / "derivatives_walk_forward.csv")
        return out
    first_ts, last_ts = ts_axis[0], ts_axis[-1]
    # Restrict windows to the span where signals are meaningful — the
    # earliest signal-bearing timestamp.
    if not sigs.empty:
        first_sig_ts = int(sigs["timestamp"].min())
        first_ts = max(first_ts, first_sig_ts)
    windows = _build_oos_windows(first_ts, last_ts,
                                 in_sample_days, oos_days, step_days)

    strat = DerivativesRotationStrategy(sigs, rotation_cfg)
    rows: List[Dict] = []
    for w_idx, w in enumerate(windows, start=1):
        logger.info("derivatives_walk_forward [%d/%d] OOS %s → %s",
                    w_idx, len(windows),
                    pd.to_datetime(w["oos_start_ms"], unit="ms", utc=True).date(),
                    pd.to_datetime(w["oos_end_ms"], unit="ms", utc=True).date())
        art = pb.run_portfolio_backtest(
            portfolio_strategy=strat, asset_frames=frames,
            timeframe=timeframe, cfg=cfg,
            start_ts_ms=w["oos_start_ms"], end_ts_ms=w["oos_end_ms"],
            save=False,
        )
        if art.equity_curve.empty:
            rows.append({"window": w_idx,
                          "oos_start_iso": pd.to_datetime(
                              w["oos_start_ms"], unit="ms",
                              utc=True).isoformat(),
                          "oos_end_iso": pd.to_datetime(
                              w["oos_end_ms"], unit="ms",
                              utc=True).isoformat(),
                          "error": "no equity points"})
            continue
        m = pb.portfolio_metrics(art.equity_curve, cfg.starting_capital)
        bench = pb.benchmark_equity_curves(
            frames, starting_capital=cfg.starting_capital, timeframe=timeframe,
            start_ts_ms=w["oos_start_ms"], end_ts_ms=w["oos_end_ms"],
        )
        btc_ret = pb.portfolio_metrics(
            bench.get("BTC_buy_and_hold", pd.DataFrame()),
            cfg.starting_capital)["total_return_pct"]
        bk_ret = pb.portfolio_metrics(
            bench.get("equal_weight_basket", pd.DataFrame()),
            cfg.starting_capital)["total_return_pct"]
        n_rebalances = (int(art.weights_history["filled"].sum())
                         if not art.weights_history.empty else 0)
        avg_holdings = (
            float(art.weights_history["weights"].apply(
                lambda d: len(d) if isinstance(d, dict) else 0,
            ).mean()) if not art.weights_history.empty else 0.0
        )
        turnover = (float(art.trades["notional"].sum()
                          / max(n_rebalances * cfg.starting_capital, 1.0))
                     if not art.trades.empty else 0.0)
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
        utils.write_df(out, config.RESULTS_DIR / "derivatives_walk_forward.csv")
    return out


# ---------------------------------------------------------------------------
# Multi-seed placebo
# ---------------------------------------------------------------------------
def derivatives_placebo(
    symbols: Sequence[str] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
                                "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT",
                                "AVAXUSDT", "LTCUSDT"),
    timeframe: str = DEFAULT_TIMEFRAME,
    seeds: Sequence[int] = PLACEBO_SEEDS_DEFAULT,
    rotation_cfg: Optional[DerivativesRotationConfig] = None,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    signals_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> pd.DataFrame:
    utils.assert_paper_only()
    cfg = backtest_cfg or pb.PortfolioBacktestConfig()
    assets = _spot_assets_for(symbols)
    frames, _ = load_universe_with_report(assets=assets, timeframe=timeframe,
                                            save=False)
    if not frames:
        out = pd.DataFrame()
        if save:
            utils.write_df(out,
                            config.RESULTS_DIR / "derivatives_placebo_comparison.csv")
        return out
    sigs = _ensure_signals(signals_df, symbols)

    strat = DerivativesRotationStrategy(sigs, rotation_cfg)
    strat_art = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames, timeframe=timeframe,
        cfg=cfg, save=False,
    )
    strat_m = pb.portfolio_metrics(strat_art.equity_curve, cfg.starting_capital)

    rows: List[Dict] = []
    top_n = (rotation_cfg or DerivativesRotationConfig()).top_n
    for seed in seeds:
        plac = RandomRotationPlacebo(
            RandomRotationConfig(top_n=top_n, seed=int(seed),
                                  apply_cash_filter=False),
        )
        art = pb.run_portfolio_backtest(
            portfolio_strategy=plac, asset_frames=frames, timeframe=timeframe,
            cfg=cfg, save=False,
        )
        m = pb.portfolio_metrics(art.equity_curve, cfg.starting_capital)
        rows.append({"seed": int(seed),
                      "placebo_return_pct": m["total_return_pct"],
                      "placebo_max_drawdown_pct": m["max_drawdown_pct"],
                      "placebo_sharpe": m["sharpe_ratio"],
                      "placebo_n_trades": int(len(art.trades))})
    placebo_df = pd.DataFrame(rows)
    median_ret = float(placebo_df["placebo_return_pct"].median())
    median_dd = float(placebo_df["placebo_max_drawdown_pct"].median())
    p75_dd = float(placebo_df["placebo_max_drawdown_pct"].quantile(0.75))
    summary = pd.DataFrame([{
        "strategy": "derivatives_rotation",
        "strategy_return_pct": strat_m["total_return_pct"],
        "strategy_max_drawdown_pct": strat_m["max_drawdown_pct"],
        "strategy_sharpe": strat_m["sharpe_ratio"],
        "n_seeds": len(seeds),
        "placebo_median_return_pct": median_ret,
        "placebo_median_drawdown_pct": median_dd,
        "placebo_p75_drawdown_pct": p75_dd,
        "strategy_beats_median_return": bool(
            strat_m["total_return_pct"] > median_ret),
        "strategy_beats_median_drawdown": bool(
            strat_m["max_drawdown_pct"] > median_dd),
    }])
    out = pd.concat(
        [summary, placebo_df.assign(strategy="placebo_seed_runs")],
        ignore_index=True, sort=False,
    )
    if save:
        utils.write_df(out,
                        config.RESULTS_DIR / "derivatives_placebo_comparison.csv")
    return out


# ---------------------------------------------------------------------------
# Scorecard — same conservative bar as the portfolio scorecard
# ---------------------------------------------------------------------------
def derivatives_scorecard(
    walk_forward_df: Optional[pd.DataFrame] = None,
    placebo_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> pd.DataFrame:
    if walk_forward_df is None:
        p = config.RESULTS_DIR / "derivatives_walk_forward.csv"
        walk_forward_df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    if placebo_df is None:
        p = config.RESULTS_DIR / "derivatives_placebo_comparison.csv"
        placebo_df = pd.read_csv(p) if p.exists() else pd.DataFrame()

    if walk_forward_df.empty:
        out = pd.DataFrame([{
            "strategy_name": "derivatives_rotation",
            "verdict": PORTFOLIO_INCONCLUSIVE,
            "reason": "no walk-forward data — likely OI cap blocks ≥5 OOS windows",
        }])
        if save:
            utils.write_df(out,
                            config.RESULTS_DIR / "derivatives_scorecard.csv")
        return out

    ok = walk_forward_df
    if "error" in ok.columns:
        ok = ok[ok["error"].isna()]
    n_windows = int(len(ok))
    if n_windows < MIN_OOS_WINDOWS_FOR_CONFIDENCE:
        out = pd.DataFrame([{
            "strategy_name": "derivatives_rotation",
            "n_windows": n_windows,
            "verdict": PORTFOLIO_INCONCLUSIVE,
            "reason": f"only {n_windows} OOS windows — need ≥"
                       f"{MIN_OOS_WINDOWS_FOR_CONFIDENCE}",
        }])
        if save:
            utils.write_df(out,
                            config.RESULTS_DIR / "derivatives_scorecard.csv")
        return out

    avg_oos_return = float(ok["oos_return_pct"].mean())
    avg_oos_dd = float(ok["oos_max_drawdown_pct"].mean())
    pct_profitable = float((ok["oos_return_pct"] > 0).mean() * 100.0)
    pct_beats_btc = float(ok["beats_btc"].mean() * 100.0)
    pct_beats_basket = float(ok["beats_basket"].mean() * 100.0)
    stability = float(((ok["profitable"] & ok["beats_btc"] & ok["beats_basket"])
                        ).mean() * 100.0)
    avg_rebalances = (float(ok["n_rebalances"].mean())
                       if "n_rebalances" in ok.columns else 0.0)
    total_rebalances = (int(ok["n_rebalances"].sum())
                         if "n_rebalances" in ok.columns else 0)

    plac_summary = (placebo_df.iloc[[0]] if not placebo_df.empty
                     else pd.DataFrame())
    placebo_median_return = (
        float(plac_summary["placebo_median_return_pct"].iloc[0])
        if not plac_summary.empty
        and "placebo_median_return_pct" in plac_summary.columns
        else float("nan"))
    strategy_full_return = (
        float(plac_summary["strategy_return_pct"].iloc[0])
        if not plac_summary.empty
        and "strategy_return_pct" in plac_summary.columns
        else float("nan"))
    beats_placebo_median = (
        bool(strategy_full_return > placebo_median_return)
        if not (np.isnan(placebo_median_return) or np.isnan(strategy_full_return))
        else False)

    pass_required = [
        ("positive_return", not np.isnan(strategy_full_return)
         and strategy_full_return > 0),
        ("beats_btc_oos", pct_beats_btc > 50.0),
        ("beats_basket_oos", pct_beats_basket > 50.0),
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
        "strategy_name": "derivatives_rotation",
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
        "reason": "; ".join(f"{k}={ok_}" for k, ok_ in pass_required),
    }])
    if save:
        utils.write_df(out, config.RESULTS_DIR / "derivatives_scorecard.csv")
    return out


# ---------------------------------------------------------------------------
# One-shot orchestrator
# ---------------------------------------------------------------------------
def run_all_derivatives(
    symbols: Sequence[str] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
                                "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT",
                                "AVAXUSDT", "LTCUSDT"),
    timeframe: str = DEFAULT_TIMEFRAME,
    in_sample_days: int = 180,
    oos_days: int = 90,
    step_days: int = 90,
    seeds: Sequence[int] = PLACEBO_SEEDS_DEFAULT,
) -> Dict[str, Any]:
    logger.info("=== derivatives research: full pipeline ===")
    sigs = ds.compute_all_derivatives_signals(symbols=symbols, save=True)
    one = run_derivatives_rotation(symbols=symbols, timeframe=timeframe,
                                     signals_df=sigs)
    wf = derivatives_walk_forward(symbols=symbols, timeframe=timeframe,
                                    in_sample_days=in_sample_days,
                                    oos_days=oos_days, step_days=step_days,
                                    signals_df=sigs)
    plac = derivatives_placebo(symbols=symbols, timeframe=timeframe,
                                seeds=seeds, signals_df=sigs)
    sc = derivatives_scorecard(walk_forward_df=wf, placebo_df=plac)
    return {"signals": sigs, "single": one,
            "walk_forward": wf, "placebo": plac, "scorecard": sc}
