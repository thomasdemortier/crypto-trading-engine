"""
Drawdown-targeted BTC allocator research orchestrator.

Single-window backtest, 14-window walk-forward, multi-seed placebo,
strict scorecard. PASS criteria (specced — NOT tuned after results):

    1. Beats BTC in > 50 % of OOS windows.
    2. Beats equal-weight basket in > 50 % of OOS windows.
    3. Beats placebo median return.
    4. OOS stability >= 60 %.
    5. Max drawdown not worse than BTC by more than 20 pp.
    6. >= 10 rebalances total.
    7. Beats simple momentum in > 50 % of OOS windows (sanity).

`stability` here = fraction of windows where the strategy is profitable
AND beats every primary benchmark (BTC, basket, simple momentum).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import config, portfolio_backtester as pb, utils
from .portfolio_research import (
    DEFAULT_TIMEFRAME, MIN_OOS_WINDOWS_FOR_CONFIDENCE,
    MIN_REBALANCES_FOR_CONFIDENCE, PLACEBO_SEEDS_DEFAULT,
    PORTFOLIO_FAIL, PORTFOLIO_INCONCLUSIVE, PORTFOLIO_PASS,
    PORTFOLIO_WATCHLIST, _build_oos_windows, load_universe_with_report,
)
from .strategies.drawdown_targeted_btc_allocator import (
    DrawdownTargetedBTCAllocatorStrategy, DrawdownTargetedBTCConfig,
)
from .strategies.momentum_rotation import (
    MomentumRotationConfig, MomentumRotationStrategy,
)

logger = utils.get_logger("cte.drawdown_targeted_research")

# DD ceiling: strategy_dd - btc_dd >= -20 pp (drawdowns are negative).
MAX_DD_VS_BTC_GAP_PP = 20.0

# Default universe for this strategy: BTC + the 9 alts used by the alt
# overlay. BTC alone is enough for the core allocator, but the universe
# must include alts for the breadth gate + overlay to fire.
DEFAULT_UNIVERSE: tuple = (
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT",
    "XRP/USDT", "DOGE/USDT", "ADA/USDT", "LTC/USDT", "BNB/USDT",
)


# ---------------------------------------------------------------------------
# Placebo: random exposure-bucket allocator. Picks a BTC weight uniformly
# at random from the 4 buckets at every rebalance. Never reads drawdown,
# 200d MA, vol, or alt breadth — pure null model.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DrawdownTargetedPlaceboConfig:
    btc_asset: str = "BTC/USDT"
    seed: int = 42
    # Exposure buckets — match the strategy's BTC weights so the placebo
    # explores the SAME exposure space.
    btc_weight_buckets: Sequence[float] = (1.00, 0.70, 0.40, 0.20)


class DrawdownTargetedPlacebo:
    """At every rebalance bar, picks a BTC weight uniformly at random
    from the same 4 buckets the strategy uses. NEVER reads drawdown,
    200d MA, vol, or alt breadth."""

    name = "drawdown_targeted_btc_placebo"

    def __init__(
        self, cfg: Optional[DrawdownTargetedPlaceboConfig] = None,
    ) -> None:
        self.cfg = cfg or DrawdownTargetedPlaceboConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

    def reset(self, seed: Optional[int] = None) -> None:
        s = self.cfg.seed if seed is None else int(seed)
        self._rng = np.random.default_rng(s)

    def target_weights(
        self,
        asof_ts_ms: int,
        asset_frames: Dict[str, pd.DataFrame],
        timeframe: str = "1d",
    ) -> Dict[str, float]:
        df = asset_frames.get(self.cfg.btc_asset)
        if df is None or df.empty:
            return {}
        sub = df[df["timestamp"] <= int(asof_ts_ms)]
        if len(sub) < 2:
            return {}
        w = float(self._rng.choice(self.cfg.btc_weight_buckets))
        return {self.cfg.btc_asset: w} if w > 0 else {}


def _enough_btc_history(frames: Dict[str, pd.DataFrame],
                         min_days: int = 4 * 365) -> Optional[str]:
    btc = frames.get("BTC/USDT")
    if btc is None or btc.empty:
        return "no BTC/USDT data"
    if len(btc) < min_days:
        return (f"only {len(btc)} BTC bars; need >= {min_days} for the "
                f"14-window walk-forward")
    return None


def _diagnostics_curve(
    strat: DrawdownTargetedBTCAllocatorStrategy,
    frames: Dict[str, pd.DataFrame], timeframe: str,
) -> pd.DataFrame:
    """Build a per-rebalance diagnostics DataFrame — drawdown, regime, vol,
    breadth — so the dashboard can show what the allocator was reacting
    to. NOT part of the backtest loop, run after the fact."""
    btc = frames.get(strat.cfg.btc_asset)
    if btc is None or btc.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    # Sample once per week from the BTC frame.
    btc_sorted = btc.sort_values("timestamp").reset_index(drop=True)
    btc_sorted["dt"] = pd.to_datetime(btc_sorted["timestamp"], unit="ms",
                                        utc=True)
    last_iso_week = None
    for _, r in btc_sorted.iterrows():
        iso = r["dt"].isocalendar()[:2]
        if iso == last_iso_week:
            continue
        last_iso_week = iso
        diag = strat.diagnostics(int(r["timestamp"]), frames, timeframe)
        if not diag:
            continue
        weights = strat.target_weights(int(r["timestamp"]), frames, timeframe)
        rows.append({
            "timestamp": int(r["timestamp"]),
            "datetime": r["dt"].isoformat(),
            "btc_close": float(r["close"]),
            "btc_drawdown_pct": diag.get("btc_drawdown", float("nan")) * 100.0,
            "btc_above_200dma": diag.get("btc_above_200dma", float("nan")),
            "realised_vol_30d_pct": (diag.get("realised_vol_30d",
                                                 float("nan")) * 100.0
                                       if not np.isnan(diag.get(
                                           "realised_vol_30d", float("nan")))
                                       else float("nan")),
            "realised_vol_90d_pct": (diag.get("realised_vol_90d",
                                                 float("nan")) * 100.0
                                       if not np.isnan(diag.get(
                                           "realised_vol_90d", float("nan")))
                                       else float("nan")),
            "btc_weight": float(weights.get(strat.cfg.btc_asset, 0.0)),
            "n_alts_held": sum(1 for k in weights
                               if k != strat.cfg.btc_asset),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Single-window full backtest
# ---------------------------------------------------------------------------
def run_drawdown_targeted_btc(
    universe: Sequence[str] = DEFAULT_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    strat_cfg: Optional[DrawdownTargetedBTCConfig] = None,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    save: bool = True,
) -> Dict[str, Any]:
    utils.assert_paper_only()
    frames, _ = load_universe_with_report(assets=universe, timeframe=timeframe,
                                            save=save)
    if not frames:
        return {"ok": False, "reason": "no asset data"}
    cfg = backtest_cfg or pb.PortfolioBacktestConfig()
    strat = DrawdownTargetedBTCAllocatorStrategy(strat_cfg)
    art = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames,
        timeframe=timeframe, cfg=cfg, save=save,
        save_prefix="drawdown_targeted_btc",
    )

    bench = pb.benchmark_equity_curves(
        frames, starting_capital=cfg.starting_capital, timeframe=timeframe,
    )
    simple = MomentumRotationStrategy(MomentumRotationConfig())
    simple_art = pb.run_portfolio_backtest(
        portfolio_strategy=simple, asset_frames=frames,
        timeframe=timeframe, cfg=cfg, save=False,
    )
    bench["simple_momentum"] = simple_art.equity_curve

    metrics = pb.portfolio_metrics(art.equity_curve, cfg.starting_capital)
    bench_metrics = {n: pb.portfolio_metrics(d, cfg.starting_capital)
                      for n, d in bench.items()}
    rows = [{"strategy": strat.name, **metrics}]
    for name, m in bench_metrics.items():
        rows.append({"strategy": name, **m})
    cmp_df = pd.DataFrame(rows)

    diag_df = _diagnostics_curve(strat, frames, timeframe)
    if save:
        utils.write_df(
            cmp_df,
            config.RESULTS_DIR / "drawdown_targeted_btc_comparison.csv",
        )
        utils.write_df(
            diag_df,
            config.RESULTS_DIR / "drawdown_targeted_btc_diagnostics.csv",
        )
    return {"ok": True, "artifacts": art, "benchmarks": bench,
            "metrics": metrics, "bench_metrics": bench_metrics,
            "comparison_df": cmp_df, "diagnostics_df": diag_df}


# ---------------------------------------------------------------------------
# 14-window walk-forward
# ---------------------------------------------------------------------------
def drawdown_targeted_btc_walk_forward(
    universe: Sequence[str] = DEFAULT_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    in_sample_days: int = 180,
    oos_days: int = 90,
    step_days: int = 90,
    strat_cfg: Optional[DrawdownTargetedBTCConfig] = None,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    save: bool = True,
    max_windows: int = 14,
) -> pd.DataFrame:
    utils.assert_paper_only()
    cfg = backtest_cfg or pb.PortfolioBacktestConfig()
    frames, _ = load_universe_with_report(assets=universe, timeframe=timeframe,
                                            save=False)
    if not frames:
        out = pd.DataFrame()
        if save:
            utils.write_df(
                out,
                config.RESULTS_DIR / "drawdown_targeted_btc_walk_forward.csv",
            )
        return out

    ts_axis = pb._aligned_timestamps(frames)
    if len(ts_axis) < 2:
        out = pd.DataFrame()
        if save:
            utils.write_df(
                out,
                config.RESULTS_DIR / "drawdown_targeted_btc_walk_forward.csv",
            )
        return out
    first_ts, last_ts = ts_axis[0], ts_axis[-1]
    windows = _build_oos_windows(first_ts, last_ts,
                                  in_sample_days, oos_days, step_days)
    if max_windows is not None and len(windows) > max_windows:
        windows = windows[-max_windows:]  # take the most recent N windows

    strat = DrawdownTargetedBTCAllocatorStrategy(strat_cfg)
    simple = MomentumRotationStrategy(MomentumRotationConfig())
    rows: List[Dict] = []
    for w_idx, w in enumerate(windows, start=1):
        logger.info(
            "drawdown_targeted_btc_walk_forward [%d/%d] OOS %s -> %s",
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
                "oos_start_iso": pd.to_datetime(
                    w["oos_start_ms"], unit="ms", utc=True).isoformat(),
                "oos_end_iso": pd.to_datetime(
                    w["oos_end_ms"], unit="ms", utc=True).isoformat(),
                "error": "no equity points",
            })
            continue
        m = pb.portfolio_metrics(art.equity_curve, cfg.starting_capital)
        bench = pb.benchmark_equity_curves(
            frames, starting_capital=cfg.starting_capital, timeframe=timeframe,
            start_ts_ms=w["oos_start_ms"], end_ts_ms=w["oos_end_ms"],
        )
        simple_art = pb.run_portfolio_backtest(
            portfolio_strategy=simple, asset_frames=frames,
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
        utils.write_df(
            out,
            config.RESULTS_DIR / "drawdown_targeted_btc_walk_forward.csv",
        )
    return out


# ---------------------------------------------------------------------------
# Placebo (multi-seed)
# ---------------------------------------------------------------------------
def drawdown_targeted_btc_placebo(
    universe: Sequence[str] = DEFAULT_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    seeds: Sequence[int] = PLACEBO_SEEDS_DEFAULT,
    strat_cfg: Optional[DrawdownTargetedBTCConfig] = None,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    save: bool = True,
) -> pd.DataFrame:
    utils.assert_paper_only()
    cfg = backtest_cfg or pb.PortfolioBacktestConfig()
    frames, _ = load_universe_with_report(assets=universe, timeframe=timeframe,
                                            save=False)
    if not frames:
        out = pd.DataFrame()
        if save:
            utils.write_df(
                out,
                config.RESULTS_DIR / "drawdown_targeted_btc_placebo.csv",
            )
        return out

    strat = DrawdownTargetedBTCAllocatorStrategy(strat_cfg)
    strat_art = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames, timeframe=timeframe,
        cfg=cfg, save=False,
    )
    strat_m = pb.portfolio_metrics(strat_art.equity_curve, cfg.starting_capital)

    rows: List[Dict] = []
    for seed in seeds:
        plac = DrawdownTargetedPlacebo(
            DrawdownTargetedPlaceboConfig(seed=int(seed)),
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
    median_ret = (float(placebo_df["placebo_return_pct"].median())
                   if not placebo_df.empty else float("nan"))
    median_dd = (float(placebo_df["placebo_max_drawdown_pct"].median())
                  if not placebo_df.empty else float("nan"))
    p75_dd = (float(placebo_df["placebo_max_drawdown_pct"].quantile(0.75))
               if not placebo_df.empty else float("nan"))
    summary = pd.DataFrame([{
        "strategy": "drawdown_targeted_btc_allocator",
        "strategy_return_pct": strat_m["total_return_pct"],
        "strategy_max_drawdown_pct": strat_m["max_drawdown_pct"],
        "strategy_sharpe": strat_m["sharpe_ratio"],
        "n_seeds": len(seeds),
        "placebo_median_return_pct": median_ret,
        "placebo_median_drawdown_pct": median_dd,
        "placebo_p75_drawdown_pct": p75_dd,
        "strategy_beats_median_return": bool(
            (not np.isnan(median_ret))
            and strat_m["total_return_pct"] > median_ret),
        "strategy_beats_median_drawdown": bool(
            (not np.isnan(median_dd))
            and strat_m["max_drawdown_pct"] > median_dd),
    }])
    out = pd.concat(
        [summary, placebo_df.assign(strategy="placebo_seed_runs")],
        ignore_index=True, sort=False,
    )
    if save:
        utils.write_df(
            out, config.RESULTS_DIR / "drawdown_targeted_btc_placebo.csv",
        )
    return out


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
def drawdown_targeted_btc_scorecard(
    walk_forward_df: Optional[pd.DataFrame] = None,
    placebo_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> pd.DataFrame:
    if walk_forward_df is None:
        p = config.RESULTS_DIR / "drawdown_targeted_btc_walk_forward.csv"
        walk_forward_df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    if placebo_df is None:
        p = config.RESULTS_DIR / "drawdown_targeted_btc_placebo.csv"
        placebo_df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    cmp_p = config.RESULTS_DIR / "drawdown_targeted_btc_comparison.csv"
    cmp_df = pd.read_csv(cmp_p) if cmp_p.exists() else pd.DataFrame()

    if walk_forward_df.empty:
        out = pd.DataFrame([{
            "strategy_name": "drawdown_targeted_btc_allocator",
            "verdict": PORTFOLIO_INCONCLUSIVE,
            "reason": "no walk-forward data",
        }])
        if save:
            utils.write_df(
                out,
                config.RESULTS_DIR / "drawdown_targeted_btc_scorecard.csv",
            )
        return out

    ok = walk_forward_df
    if "error" in ok.columns:
        ok = ok[ok["error"].isna()]
    n_windows = int(len(ok))
    if n_windows < MIN_OOS_WINDOWS_FOR_CONFIDENCE:
        out = pd.DataFrame([{
            "strategy_name": "drawdown_targeted_btc_allocator",
            "n_windows": n_windows,
            "verdict": PORTFOLIO_INCONCLUSIVE,
            "reason": f"only {n_windows} OOS windows; need >= "
                       f"{MIN_OOS_WINDOWS_FOR_CONFIDENCE}",
        }])
        if save:
            utils.write_df(
                out,
                config.RESULTS_DIR / "drawdown_targeted_btc_scorecard.csv",
            )
        return out

    avg_oos_return = float(ok["oos_return_pct"].mean())
    avg_oos_dd = float(ok["oos_max_drawdown_pct"].mean())
    pct_profitable = float((ok["oos_return_pct"] > 0).mean() * 100.0)
    pct_beats_btc = float(ok["beats_btc"].mean() * 100.0)
    pct_beats_basket = float(ok["beats_basket"].mean() * 100.0)
    pct_beats_simple = (float(ok["beats_simple_momentum"].mean() * 100.0)
                         if "beats_simple_momentum" in ok.columns else 0.0)
    stab_mask = (ok["profitable"] & ok["beats_btc"] & ok["beats_basket"])
    if "beats_simple_momentum" in ok.columns:
        stab_mask &= ok["beats_simple_momentum"]
    stability = float(stab_mask.mean() * 100.0)
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
        if not (np.isnan(placebo_median_return)
                or np.isnan(strategy_full_return))
        else False)

    btc_full_dd = float("nan")
    if not cmp_df.empty and "strategy" in cmp_df.columns \
            and "max_drawdown_pct" in cmp_df.columns:
        btc_row = cmp_df[cmp_df["strategy"] == "BTC_buy_and_hold"]
        if not btc_row.empty:
            btc_full_dd = float(btc_row["max_drawdown_pct"].iloc[0])
    dd_gap_ok = (not np.isnan(strategy_full_dd) and not np.isnan(btc_full_dd)
                  and (strategy_full_dd - btc_full_dd) >= -MAX_DD_VS_BTC_GAP_PP)

    pass_required = [
        ("beats_btc_oos_majority", pct_beats_btc > 50.0),
        ("beats_basket_oos_majority", pct_beats_basket > 50.0),
        ("beats_simple_momentum_oos_majority", pct_beats_simple > 50.0),
        ("beats_placebo_median", beats_placebo_median),
        ("oos_stability_at_least_60", stability >= 60.0),
        ("dd_within_btc_gap_20pp", bool(dd_gap_ok)),
        ("at_least_10_rebalances",
         total_rebalances >= MIN_REBALANCES_FOR_CONFIDENCE),
    ]
    n_pass_satisfied = sum(1 for _, ok_ in pass_required if ok_)
    if all(ok_ for _, ok_ in pass_required):
        verdict = PORTFOLIO_PASS
    elif n_pass_satisfied >= len(pass_required) - 1:
        verdict = PORTFOLIO_WATCHLIST
    elif total_rebalances < MIN_REBALANCES_FOR_CONFIDENCE:
        verdict = PORTFOLIO_INCONCLUSIVE
    else:
        verdict = PORTFOLIO_FAIL

    out = pd.DataFrame([{
        "strategy_name": "drawdown_targeted_btc_allocator",
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
        utils.write_df(
            out, config.RESULTS_DIR / "drawdown_targeted_btc_scorecard.csv",
        )
    return out


def run_all_drawdown_targeted_btc(
    universe: Sequence[str] = DEFAULT_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    in_sample_days: int = 180,
    oos_days: int = 90,
    step_days: int = 90,
    seeds: Sequence[int] = PLACEBO_SEEDS_DEFAULT,
    max_windows: int = 14,
) -> Dict[str, Any]:
    logger.info("=== drawdown-targeted BTC research: full pipeline ===")
    one = run_drawdown_targeted_btc(universe=universe, timeframe=timeframe)
    wf = drawdown_targeted_btc_walk_forward(
        universe=universe, timeframe=timeframe,
        in_sample_days=in_sample_days, oos_days=oos_days, step_days=step_days,
        max_windows=max_windows,
    )
    plac = drawdown_targeted_btc_placebo(
        universe=universe, timeframe=timeframe, seeds=seeds,
    )
    sc = drawdown_targeted_btc_scorecard(walk_forward_df=wf, placebo_df=plac)
    return {"single": one, "walk_forward": wf,
            "placebo": plac, "scorecard": sc}
