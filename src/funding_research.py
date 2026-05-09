"""
Funding-only research orchestrator.

Walk-forward + multi-seed placebo + scorecard for the
`FundingRotationStrategy`. Scorecard PASS criteria are STRICTER than
the v1 portfolio scorecard (see spec):

  PASS requires ALL of:
    1. Positive total return.
    2. Beats BTC OOS (in > 50 % of OOS windows).
    3. Beats equal-weight basket OOS.
    4. Beats simple momentum OOS.
    5. Beats placebo median return.
    6. OOS stability above 60 %.
    7. At least 10 rebalances total.
    8. Max drawdown not worse than BTC by more than 20 percentage points.

Anything weaker is FAIL or INCONCLUSIVE.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import (config, funding_signals as fs, portfolio_backtester as pb,
                utils)
from .portfolio_research import (
    DEFAULT_TIMEFRAME, MIN_OOS_WINDOWS_FOR_CONFIDENCE,
    MIN_REBALANCES_FOR_CONFIDENCE, PLACEBO_SEEDS_DEFAULT,
    PORTFOLIO_FAIL, PORTFOLIO_INCONCLUSIVE, PORTFOLIO_PASS,
    PORTFOLIO_WATCHLIST, _build_oos_windows, load_universe_with_report,
)
from .strategies.funding_rotation import (
    FundingRotationConfig, FundingRotationStrategy,
)
from .strategies.momentum_rotation import (
    MomentumRotationConfig, MomentumRotationStrategy,
    RandomRotationConfig, RandomRotationPlacebo,
)

logger = utils.get_logger("cte.funding_research")

DEFAULT_FUNDING_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
                             "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT",
                             "AVAXUSDT", "LTCUSDT")
# DD divergence ceiling: strategy DD - BTC DD must be ≥ -20 pp (i.e.
# the strategy's drawdown is no worse than BTC's by more than 20 pp).
MAX_DD_VS_BTC_GAP_PP = 20.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_signals(signals_df: Optional[pd.DataFrame],
                     symbols: Sequence[str]) -> pd.DataFrame:
    if signals_df is not None and not signals_df.empty:
        return signals_df
    p = config.RESULTS_DIR / "funding_signals.csv"
    if p.exists() and p.stat().st_size > 0:
        return pd.read_csv(p)
    logger.info("funding_signals.csv missing — computing on the fly")
    return fs.compute_all_funding_signals(symbols=symbols, save=True)


def _spot_assets_for(symbols: Sequence[str]) -> List[str]:
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
def run_funding_rotation(
    symbols: Sequence[str] = DEFAULT_FUNDING_SYMBOLS,
    timeframe: str = DEFAULT_TIMEFRAME,
    rotation_cfg: Optional[FundingRotationConfig] = None,
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
    strat = FundingRotationStrategy(sigs, rotation_cfg)
    cfg = backtest_cfg or pb.PortfolioBacktestConfig()
    art = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames,
        timeframe=timeframe, cfg=cfg, save=save,
        save_prefix="funding_rotation",
    )
    bench = pb.benchmark_equity_curves(
        frames, starting_capital=cfg.starting_capital, timeframe=timeframe,
    )
    # Simple-momentum benchmark — full window, same starting capital.
    simple = MomentumRotationStrategy(MomentumRotationConfig())
    simple_art = pb.run_portfolio_backtest(
        portfolio_strategy=simple, asset_frames=frames,
        timeframe=timeframe, cfg=cfg, save=False,
    )
    bench["simple_momentum"] = simple_art.equity_curve

    metrics = pb.portfolio_metrics(art.equity_curve, cfg.starting_capital)
    bench_metrics = {n: pb.portfolio_metrics(d, cfg.starting_capital)
                      for n, d in bench.items()}
    rows = [{"strategy": "funding_rotation", **metrics}]
    for name, m in bench_metrics.items():
        rows.append({"strategy": name, **m})
    cmp_df = pd.DataFrame(rows)
    if save:
        utils.write_df(cmp_df,
                       config.RESULTS_DIR / "funding_rotation_comparison.csv")
    return {"ok": True, "artifacts": art, "benchmarks": bench,
            "metrics": metrics, "bench_metrics": bench_metrics,
            "comparison_df": cmp_df}


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def funding_walk_forward(
    symbols: Sequence[str] = DEFAULT_FUNDING_SYMBOLS,
    timeframe: str = DEFAULT_TIMEFRAME,
    in_sample_days: int = 180,
    oos_days: int = 90,
    step_days: int = 90,
    rotation_cfg: Optional[FundingRotationConfig] = None,
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
            utils.write_df(out, config.RESULTS_DIR / "funding_walk_forward.csv")
        return out
    sigs = _ensure_signals(signals_df, symbols)

    ts_axis = pb._aligned_timestamps(frames)
    if len(ts_axis) < 2:
        out = pd.DataFrame()
        if save:
            utils.write_df(out, config.RESULTS_DIR / "funding_walk_forward.csv")
        return out
    first_ts, last_ts = ts_axis[0], ts_axis[-1]
    if not sigs.empty and "timestamp" in sigs.columns:
        first_sig_ts = int(pd.to_numeric(sigs["timestamp"]).min())
        first_ts = max(first_ts, first_sig_ts)
    windows = _build_oos_windows(first_ts, last_ts,
                                 in_sample_days, oos_days, step_days)

    strat = FundingRotationStrategy(sigs, rotation_cfg)
    simple_strat = MomentumRotationStrategy(MomentumRotationConfig())
    rows: List[Dict] = []
    for w_idx, w in enumerate(windows, start=1):
        logger.info(
            "funding_walk_forward [%d/%d] OOS %s → %s",
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
            rows.append({"window": w_idx,
                          "oos_start_iso": pd.to_datetime(
                              w["oos_start_ms"], unit="ms", utc=True).isoformat(),
                          "oos_end_iso": pd.to_datetime(
                              w["oos_end_ms"], unit="ms", utc=True).isoformat(),
                          "error": "no equity points"})
            continue
        m = pb.portfolio_metrics(art.equity_curve, cfg.starting_capital)
        bench = pb.benchmark_equity_curves(
            frames, starting_capital=cfg.starting_capital, timeframe=timeframe,
            start_ts_ms=w["oos_start_ms"], end_ts_ms=w["oos_end_ms"],
        )
        simple_art = pb.run_portfolio_backtest(
            portfolio_strategy=simple_strat, asset_frames=frames,
            timeframe=timeframe, cfg=cfg,
            start_ts_ms=w["oos_start_ms"], end_ts_ms=w["oos_end_ms"],
            save=False,
        )
        btc_ret = pb.portfolio_metrics(
            bench.get("BTC_buy_and_hold", pd.DataFrame()),
            cfg.starting_capital)["total_return_pct"]
        bk_ret = pb.portfolio_metrics(
            bench.get("equal_weight_basket", pd.DataFrame()),
            cfg.starting_capital)["total_return_pct"]
        simple_ret = pb.portfolio_metrics(
            simple_art.equity_curve, cfg.starting_capital,
        )["total_return_pct"]
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
            "simple_oos_return_pct": simple_ret,
            "beats_btc": bool(m["total_return_pct"] > btc_ret),
            "beats_basket": bool(m["total_return_pct"] > bk_ret),
            "beats_simple_momentum": bool(m["total_return_pct"] > simple_ret),
            "profitable": bool(m["total_return_pct"] > 0),
            "n_rebalances": n_rebalances,
            "n_trades": int(len(art.trades)),
            "avg_holdings": avg_holdings,
            "turnover": turnover,
            "error": None,
        })
    out = pd.DataFrame(rows)
    if save:
        utils.write_df(out, config.RESULTS_DIR / "funding_walk_forward.csv")
    return out


# ---------------------------------------------------------------------------
# Multi-seed placebo
# ---------------------------------------------------------------------------
def funding_placebo(
    symbols: Sequence[str] = DEFAULT_FUNDING_SYMBOLS,
    timeframe: str = DEFAULT_TIMEFRAME,
    seeds: Sequence[int] = PLACEBO_SEEDS_DEFAULT,
    rotation_cfg: Optional[FundingRotationConfig] = None,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    signals_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Multi-seed random Top-3 rotation as the placebo. Same universe,
    same rebalance frequency. ≥20 seeds by default (per spec)."""
    utils.assert_paper_only()
    cfg = backtest_cfg or pb.PortfolioBacktestConfig()
    assets = _spot_assets_for(symbols)
    frames, _ = load_universe_with_report(assets=assets, timeframe=timeframe,
                                            save=False)
    if not frames:
        out = pd.DataFrame()
        if save:
            utils.write_df(out,
                            config.RESULTS_DIR / "funding_placebo_comparison.csv")
        return out
    sigs = _ensure_signals(signals_df, symbols)
    strat = FundingRotationStrategy(sigs, rotation_cfg)
    strat_art = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames, timeframe=timeframe,
        cfg=cfg, save=False,
    )
    strat_m = pb.portfolio_metrics(strat_art.equity_curve, cfg.starting_capital)

    rows: List[Dict] = []
    top_n = (rotation_cfg or FundingRotationConfig()).top_n
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
        "strategy": "funding_rotation",
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
                        config.RESULTS_DIR / "funding_placebo_comparison.csv")
    return out


# ---------------------------------------------------------------------------
# Scorecard — STRICTER than the v1 portfolio scorecard
# ---------------------------------------------------------------------------
def funding_scorecard(
    walk_forward_df: Optional[pd.DataFrame] = None,
    placebo_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Builds the funding scorecard with the 8-check PASS rule.

    Mirrors `portfolio_scorecard` but adds `beats_simple_momentum_oos`
    and `dd_within_btc_gap` (DD - BTC_DD ≥ -20 pp). All thresholds are
    fixed by spec — none are tuned. Returns one row.
    """
    if walk_forward_df is None:
        p = config.RESULTS_DIR / "funding_walk_forward.csv"
        walk_forward_df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    if placebo_df is None:
        p = config.RESULTS_DIR / "funding_placebo_comparison.csv"
        placebo_df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    cmp_p = config.RESULTS_DIR / "funding_rotation_comparison.csv"
    cmp_df = pd.read_csv(cmp_p) if cmp_p.exists() else pd.DataFrame()

    if walk_forward_df.empty:
        out = pd.DataFrame([{
            "strategy_name": "funding_rotation",
            "verdict": PORTFOLIO_INCONCLUSIVE,
            "reason": "no walk-forward data",
        }])
        if save:
            utils.write_df(out, config.RESULTS_DIR / "funding_scorecard.csv")
        return out

    ok = walk_forward_df
    if "error" in ok.columns:
        ok = ok[ok["error"].isna()]
    n_windows = int(len(ok))
    if n_windows < MIN_OOS_WINDOWS_FOR_CONFIDENCE:
        out = pd.DataFrame([{
            "strategy_name": "funding_rotation",
            "n_windows": n_windows,
            "verdict": PORTFOLIO_INCONCLUSIVE,
            "reason": f"only {n_windows} OOS windows — need ≥"
                       f"{MIN_OOS_WINDOWS_FOR_CONFIDENCE}",
        }])
        if save:
            utils.write_df(out, config.RESULTS_DIR / "funding_scorecard.csv")
        return out

    avg_oos_return = float(ok["oos_return_pct"].mean())
    avg_oos_dd = float(ok["oos_max_drawdown_pct"].mean())
    pct_profitable = float((ok["oos_return_pct"] > 0).mean() * 100.0)
    pct_beats_btc = float(ok["beats_btc"].mean() * 100.0)
    pct_beats_basket = float(ok["beats_basket"].mean() * 100.0)
    pct_beats_simple = (float(ok["beats_simple_momentum"].mean() * 100.0)
                         if "beats_simple_momentum" in ok.columns else 0.0)
    stability = float(((ok["profitable"] & ok["beats_btc"] & ok["beats_basket"])
                        ).mean() * 100.0)
    avg_rebalances = (float(ok["n_rebalances"].mean())
                       if "n_rebalances" in ok.columns else 0.0)
    total_rebalances = (int(ok["n_rebalances"].sum())
                         if "n_rebalances" in ok.columns else 0)

    plac_summary = (placebo_df.iloc[[0]]
                     if not placebo_df.empty else pd.DataFrame())
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
    strategy_full_dd = (
        float(plac_summary["strategy_max_drawdown_pct"].iloc[0])
        if not plac_summary.empty
        and "strategy_max_drawdown_pct" in plac_summary.columns
        else float("nan"))
    beats_placebo_median = (
        bool(strategy_full_return > placebo_median_return)
        if not (np.isnan(placebo_median_return) or np.isnan(strategy_full_return))
        else False)

    # BTC full-window DD from the comparison frame.
    btc_full_dd = float("nan")
    if not cmp_df.empty and "strategy" in cmp_df.columns \
            and "max_drawdown_pct" in cmp_df.columns:
        btc_row = cmp_df[cmp_df["strategy"] == "BTC_buy_and_hold"]
        if not btc_row.empty:
            btc_full_dd = float(btc_row["max_drawdown_pct"].iloc[0])
    # DD gap: drawdowns are negative numbers. "Strategy DD not worse
    # than BTC's by more than 20 pp" -> strategy_dd - btc_dd >= -20.
    dd_gap_ok = (not np.isnan(strategy_full_dd) and not np.isnan(btc_full_dd)
                  and (strategy_full_dd - btc_full_dd) >= -MAX_DD_VS_BTC_GAP_PP)

    pass_required = [
        ("positive_return", not np.isnan(strategy_full_return)
         and strategy_full_return > 0),
        ("beats_btc_oos", pct_beats_btc > 50.0),
        ("beats_basket_oos", pct_beats_basket > 50.0),
        ("beats_simple_momentum_oos", pct_beats_simple > 50.0),
        ("beats_placebo_median", beats_placebo_median),
        ("oos_stability_above_60", stability > 60.0),
        ("at_least_10_rebalances",
         total_rebalances >= MIN_REBALANCES_FOR_CONFIDENCE),
        ("dd_within_btc_gap_20pp", bool(dd_gap_ok)),
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
        "strategy_name": "funding_rotation",
        "n_windows": n_windows,
        "avg_oos_return_pct": avg_oos_return,
        "avg_oos_drawdown_pct": avg_oos_dd,
        "pct_windows_profitable": pct_profitable,
        "pct_windows_beat_btc": pct_beats_btc,
        "pct_windows_beat_basket": pct_beats_basket,
        "pct_windows_beat_simple_momentum": pct_beats_simple,
        "stability_score_pct": stability,
        "total_rebalances": total_rebalances,
        "avg_rebalances_per_window": avg_rebalances,
        "strategy_full_return_pct": strategy_full_return,
        "strategy_full_drawdown_pct": strategy_full_dd,
        "btc_full_drawdown_pct": btc_full_dd,
        "dd_gap_pp": (strategy_full_dd - btc_full_dd) if (
            not np.isnan(strategy_full_dd) and not np.isnan(btc_full_dd)
        ) else float("nan"),
        "placebo_median_return_pct": placebo_median_return,
        "beats_placebo_median": beats_placebo_median,
        "verdict": verdict,
        "checks_passed": n_pass_satisfied,
        "checks_total": len(pass_required),
        "reason": "; ".join(f"{k}={ok_}" for k, ok_ in pass_required),
    }])
    if save:
        utils.write_df(out, config.RESULTS_DIR / "funding_scorecard.csv")
    return out


# ---------------------------------------------------------------------------
# One-shot orchestrator
# ---------------------------------------------------------------------------
def run_all_funding(
    symbols: Sequence[str] = DEFAULT_FUNDING_SYMBOLS,
    timeframe: str = DEFAULT_TIMEFRAME,
    in_sample_days: int = 180,
    oos_days: int = 90,
    step_days: int = 90,
    seeds: Sequence[int] = PLACEBO_SEEDS_DEFAULT,
) -> Dict[str, Any]:
    logger.info("=== funding research: full pipeline ===")
    sigs = fs.compute_all_funding_signals(symbols=symbols, save=True)
    one = run_funding_rotation(symbols=symbols, timeframe=timeframe,
                                 signals_df=sigs)
    wf = funding_walk_forward(symbols=symbols, timeframe=timeframe,
                                in_sample_days=in_sample_days,
                                oos_days=oos_days, step_days=step_days,
                                signals_df=sigs)
    plac = funding_placebo(symbols=symbols, timeframe=timeframe,
                             seeds=seeds, signals_df=sigs)
    sc = funding_scorecard(walk_forward_df=wf, placebo_df=plac)
    return {"signals": sigs, "single": one, "walk_forward": wf,
            "placebo": plac, "scorecard": sc}
