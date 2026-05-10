"""
Funding + basis carry / crowding research orchestrator.

Owns:
    * Single-window full backtest + benchmark comparison.
    * 14-window walk-forward.
    * 20-seed random-bucket placebo.
    * Strict scorecard.
    * Per-rebalance diagnostics.

Hard rules (locked, NOT tuned):
    PASS criteria (each must clear; all gates binding):
        1. Positive full-window return.
        2. Beats BTC in > 50 % of OOS windows.
        3. Beats equal-weight basket in > 50 % of OOS windows.
        4. Beats simple momentum in > 50 % of OOS windows.
        5. Beats placebo median full-window return.
        6. OOS stability >= 60 %.
        7. Max drawdown not worse than BTC by more than 20 pp.
        8. >= 10 real rebalances total.
        9. Data coverage adequate (every required source must clear FAIL).
        10. No lookahead bias — verified by the strategy unit tests.

    Stability = fraction of OOS windows where the strategy is profitable
    AND beats every primary benchmark (BTC, basket, simple momentum).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import (config, funding_basis_data_collector as fbdc,
                funding_basis_signals as fbs, portfolio_backtester as pb,
                utils)
from .portfolio_research import (
    DEFAULT_TIMEFRAME, MIN_OOS_WINDOWS_FOR_CONFIDENCE,
    MIN_REBALANCES_FOR_CONFIDENCE, PLACEBO_SEEDS_DEFAULT,
    PORTFOLIO_FAIL, PORTFOLIO_INCONCLUSIVE, PORTFOLIO_PASS,
    PORTFOLIO_WATCHLIST, _build_oos_windows, load_universe_with_report,
)
from .strategies.funding_basis_carry_allocator import (
    FundingBasisCarryAllocatorStrategy, FundingBasisCarryConfig,
)
from .strategies.momentum_rotation import (
    MomentumRotationConfig, MomentumRotationStrategy,
)

logger = utils.get_logger("cte.funding_basis_research")

# DD ceiling: strategy_dd - btc_dd >= -20 pp. Drawdowns are negative.
MAX_DD_VS_BTC_GAP_PP = 20.0
DEFAULT_UNIVERSE: tuple = ("BTC/USDT", "ETH/USDT")


# ---------------------------------------------------------------------------
# Placebo: random bucket allocator. At each rebalance picks a per-asset
# weight uniformly from the SAME caps the strategy can produce, then
# normalises to Σ ≤ 1. Never reads regime / signal data.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FundingBasisPlaceboConfig:
    seed: int = 42
    asset_buckets: Sequence[float] = (0.80, 0.70, 0.30, 0.0)
    universe: Sequence[str] = DEFAULT_UNIVERSE


class FundingBasisPlacebo:
    """Random allocator that picks one of {0.80, 0.70, 0.30, 0.0} per
    asset uniformly each rebalance, then normalises to Σ ≤ 1. Never
    reads signals."""

    name = "funding_basis_placebo"

    def __init__(
        self, cfg: Optional[FundingBasisPlaceboConfig] = None,
    ) -> None:
        self.cfg = cfg or FundingBasisPlaceboConfig()
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
        weights: Dict[str, float] = {}
        for asset in self.cfg.universe:
            sub = asset_frames.get(asset)
            if sub is None or sub.empty:
                continue
            avail = sub[sub["timestamp"] <= int(asof_ts_ms)]
            if len(avail) < 2:
                continue
            w = float(self._rng.choice(self.cfg.asset_buckets))
            if w > 0.0:
                weights[asset] = w
        total = sum(weights.values())
        if total > 1.0 and total > 0.0:
            scale = 1.0 / total
            weights = {k: float(v) * scale for k, v in weights.items()}
        return weights


# ---------------------------------------------------------------------------
# Data coverage classification helper — read from the collector CSV.
# ---------------------------------------------------------------------------
def _coverage_status() -> Optional[str]:
    """Return None if every REQUIRED source on Binance (funding + mark +
    index) clears FAIL. Otherwise return a short reason string."""
    p = config.RESULTS_DIR / "funding_basis_data_coverage.csv"
    if not p.exists() or p.stat().st_size == 0:
        return "no funding/basis coverage CSV — run download_funding_basis_data"
    cov = pd.read_csv(p)
    if cov.empty:
        return "coverage CSV empty"
    needed_datasets = (
        "funding_rate_history", "mark_price_klines_1d",
        "index_price_klines_1d",
    )
    bad: List[str] = []
    for asset in DEFAULT_UNIVERSE:
        for ds in needed_datasets:
            mask = ((cov["asset"] == asset) & (cov["dataset"] == ds)
                     & (cov["source"] == "binance_futures"))
            sub = cov[mask]
            if sub.empty:
                bad.append(f"missing:{asset}:{ds}")
                continue
            v = str(sub.iloc[0].get("verdict", "FAIL"))
            if v == "FAIL":
                bad.append(f"FAIL:{asset}:{ds}")
    return "; ".join(bad) if bad else None


# ---------------------------------------------------------------------------
# Single-window full backtest
# ---------------------------------------------------------------------------
def run_funding_basis_backtest(
    universe: Sequence[str] = DEFAULT_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    strat_cfg: Optional[FundingBasisCarryConfig] = None,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    signals_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> Dict[str, Any]:
    utils.assert_paper_only()
    frames, _ = load_universe_with_report(assets=universe, timeframe=timeframe,
                                            save=save)
    if not frames:
        return {"ok": False, "reason": "no asset data"}
    cfg = backtest_cfg or pb.PortfolioBacktestConfig()
    sigs = (signals_df if signals_df is not None
              else fbs.compute_signals(assets=universe, timeframe=timeframe,
                                          save=False))
    strat = FundingBasisCarryAllocatorStrategy(sigs, strat_cfg)
    art = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames,
        timeframe=timeframe, cfg=cfg, save=save,
        save_prefix="funding_basis",
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

    diag_rows: List[Dict[str, object]] = []
    if not art.weights_history.empty:
        for _, r in art.weights_history.iterrows():
            ts = int(r["timestamp"])
            for d in strat.diagnostics(ts):
                d["weight"] = float(
                    (r["weights"] or {}).get(d["asset"], 0.0)
                    if isinstance(r["weights"], dict) else 0.0)
                diag_rows.append(d)
    diag_df = pd.DataFrame(diag_rows)

    if save:
        utils.write_df(cmp_df,
                          config.RESULTS_DIR / "funding_basis_comparison.csv")
        utils.write_df(diag_df,
                          config.RESULTS_DIR / "funding_basis_diagnostics.csv")
    return {"ok": True, "artifacts": art, "benchmarks": bench,
            "metrics": metrics, "bench_metrics": bench_metrics,
            "comparison_df": cmp_df, "diagnostics_df": diag_df}


# ---------------------------------------------------------------------------
# Walk-forward — 14 windows by default
# ---------------------------------------------------------------------------
def funding_basis_walk_forward(
    universe: Sequence[str] = DEFAULT_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    in_sample_days: int = 180,
    oos_days: int = 90,
    step_days: int = 90,
    strat_cfg: Optional[FundingBasisCarryConfig] = None,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    signals_df: Optional[pd.DataFrame] = None,
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
            utils.write_df(out, config.RESULTS_DIR
                                / "funding_basis_walk_forward.csv")
        return out
    sigs = (signals_df if signals_df is not None
              else fbs.compute_signals(assets=universe, timeframe=timeframe,
                                          save=False))

    ts_axis = pb._aligned_timestamps(frames)
    if len(ts_axis) < 2:
        out = pd.DataFrame()
        if save:
            utils.write_df(out, config.RESULTS_DIR
                                / "funding_basis_walk_forward.csv")
        return out
    first_ts, last_ts = ts_axis[0], ts_axis[-1]
    windows = _build_oos_windows(first_ts, last_ts,
                                  in_sample_days, oos_days, step_days)
    if max_windows is not None and len(windows) > max_windows:
        windows = windows[-max_windows:]

    strat = FundingBasisCarryAllocatorStrategy(sigs, strat_cfg)
    simple = MomentumRotationStrategy(MomentumRotationConfig())
    rows: List[Dict] = []
    for w_idx, w in enumerate(windows, start=1):
        logger.info(
            "funding_basis_walk_forward [%d/%d] OOS %s -> %s",
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
        eth_ret = pb.portfolio_metrics(
            bench.get("ETH_buy_and_hold", pd.DataFrame()),
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
            "eth_oos_return_pct": eth_ret,
            "basket_oos_return_pct": bk_ret,
            "simple_oos_return_pct": simple_ret,
            "beats_btc": bool(m["total_return_pct"] > btc_ret),
            "beats_eth": bool(m["total_return_pct"] > eth_ret),
            "beats_basket": bool(m["total_return_pct"] > bk_ret),
            "beats_simple_momentum": bool(m["total_return_pct"] > simple_ret),
            "profitable": bool(m["total_return_pct"] > 0),
            "n_rebalances": n_rebalances,
            "n_trades": int(len(art.trades)),
            "avg_holdings": avg_holdings,
            "error": None,
        })
    out = pd.DataFrame(rows)
    if save:
        utils.write_df(out, config.RESULTS_DIR
                            / "funding_basis_walk_forward.csv")
    return out


# ---------------------------------------------------------------------------
# Placebo
# ---------------------------------------------------------------------------
def funding_basis_placebo(
    universe: Sequence[str] = DEFAULT_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    seeds: Sequence[int] = PLACEBO_SEEDS_DEFAULT,
    strat_cfg: Optional[FundingBasisCarryConfig] = None,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    signals_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> pd.DataFrame:
    utils.assert_paper_only()
    cfg = backtest_cfg or pb.PortfolioBacktestConfig()
    frames, _ = load_universe_with_report(assets=universe, timeframe=timeframe,
                                            save=False)
    if not frames:
        out = pd.DataFrame()
        if save:
            utils.write_df(out,
                              config.RESULTS_DIR / "funding_basis_placebo.csv")
        return out
    sigs = (signals_df if signals_df is not None
              else fbs.compute_signals(assets=universe, timeframe=timeframe,
                                          save=False))
    strat = FundingBasisCarryAllocatorStrategy(sigs, strat_cfg)
    strat_art = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames, timeframe=timeframe,
        cfg=cfg, save=False,
    )
    strat_m = pb.portfolio_metrics(strat_art.equity_curve, cfg.starting_capital)

    rows: List[Dict] = []
    for seed in seeds:
        plac = FundingBasisPlacebo(
            FundingBasisPlaceboConfig(seed=int(seed),
                                          universe=tuple(universe)),
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
    plac_df = pd.DataFrame(rows)
    median_ret = (float(plac_df["placebo_return_pct"].median())
                   if not plac_df.empty else float("nan"))
    median_dd = (float(plac_df["placebo_max_drawdown_pct"].median())
                  if not plac_df.empty else float("nan"))
    p75_dd = (float(plac_df["placebo_max_drawdown_pct"].quantile(0.75))
               if not plac_df.empty else float("nan"))
    summary = pd.DataFrame([{
        "strategy": "funding_basis_carry_allocator",
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
    out = pd.concat([summary,
                       plac_df.assign(strategy="placebo_seed_runs")],
                      ignore_index=True, sort=False)
    if save:
        utils.write_df(out,
                          config.RESULTS_DIR / "funding_basis_placebo.csv")
    return out


# ---------------------------------------------------------------------------
# Scorecard — strict, locked thresholds.
# ---------------------------------------------------------------------------
def funding_basis_scorecard(
    walk_forward_df: Optional[pd.DataFrame] = None,
    placebo_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> pd.DataFrame:
    if walk_forward_df is None:
        p = config.RESULTS_DIR / "funding_basis_walk_forward.csv"
        walk_forward_df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    if placebo_df is None:
        p = config.RESULTS_DIR / "funding_basis_placebo.csv"
        placebo_df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    cmp_p = config.RESULTS_DIR / "funding_basis_comparison.csv"
    cmp_df = pd.read_csv(cmp_p) if cmp_p.exists() else pd.DataFrame()

    coverage_issue = _coverage_status()

    if walk_forward_df.empty:
        out = pd.DataFrame([{
            "strategy_name": "funding_basis_carry_allocator",
            "verdict": PORTFOLIO_INCONCLUSIVE,
            "reason": "no walk-forward data",
            "coverage_note": coverage_issue or "ok",
        }])
        if save:
            utils.write_df(out, config.RESULTS_DIR
                                / "funding_basis_scorecard.csv")
        return out

    ok = walk_forward_df
    if "error" in ok.columns:
        ok = ok[ok["error"].isna()]
    n_windows = int(len(ok))
    if n_windows < MIN_OOS_WINDOWS_FOR_CONFIDENCE:
        out = pd.DataFrame([{
            "strategy_name": "funding_basis_carry_allocator",
            "n_windows": n_windows,
            "verdict": PORTFOLIO_INCONCLUSIVE,
            "reason": f"only {n_windows} OOS windows; need >= "
                       f"{MIN_OOS_WINDOWS_FOR_CONFIDENCE}",
            "coverage_note": coverage_issue or "ok",
        }])
        if save:
            utils.write_df(out, config.RESULTS_DIR
                                / "funding_basis_scorecard.csv")
        return out

    avg_oos_return = float(ok["oos_return_pct"].mean())
    avg_oos_dd = float(ok["oos_max_drawdown_pct"].mean())
    pct_profitable = float((ok["oos_return_pct"] > 0).mean() * 100.0)
    pct_beats_btc = float(ok["beats_btc"].mean() * 100.0)
    pct_beats_eth = (float(ok["beats_eth"].mean() * 100.0)
                      if "beats_eth" in ok.columns else 0.0)
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
        ("positive_return", not np.isnan(strategy_full_return)
         and strategy_full_return > 0),
        ("beats_btc_oos_majority", pct_beats_btc > 50.0),
        ("beats_basket_oos_majority", pct_beats_basket > 50.0),
        ("beats_simple_momentum_oos_majority", pct_beats_simple > 50.0),
        ("beats_placebo_median", beats_placebo_median),
        ("oos_stability_at_least_60", stability >= 60.0),
        ("dd_within_btc_gap_20pp", bool(dd_gap_ok)),
        ("at_least_10_rebalances",
         total_rebalances >= MIN_REBALANCES_FOR_CONFIDENCE),
        ("data_coverage_adequate", coverage_issue is None),
    ]
    n_pass_satisfied = sum(1 for _, ok_ in pass_required if ok_)
    if all(ok_ for _, ok_ in pass_required):
        verdict = PORTFOLIO_PASS
    elif coverage_issue is not None:
        verdict = PORTFOLIO_INCONCLUSIVE
    elif total_rebalances < MIN_REBALANCES_FOR_CONFIDENCE:
        verdict = PORTFOLIO_INCONCLUSIVE
    elif n_pass_satisfied >= len(pass_required) - 1:
        verdict = PORTFOLIO_WATCHLIST
    else:
        verdict = PORTFOLIO_FAIL

    out = pd.DataFrame([{
        "strategy_name": "funding_basis_carry_allocator",
        "n_windows": n_windows,
        "avg_oos_return_pct": avg_oos_return,
        "avg_oos_drawdown_pct": avg_oos_dd,
        "pct_windows_profitable": pct_profitable,
        "pct_windows_beat_btc": pct_beats_btc,
        "pct_windows_beat_eth": pct_beats_eth,
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
        "coverage_note": coverage_issue or "ok",
        "verdict": verdict,
        "checks_passed": n_pass_satisfied,
        "checks_total": len(pass_required),
        "reason": "; ".join(f"{k}={ok_}" for k, ok_ in pass_required),
    }])
    if save:
        utils.write_df(out,
                          config.RESULTS_DIR / "funding_basis_scorecard.csv")
    return out


def run_all_funding_basis(
    universe: Sequence[str] = DEFAULT_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    in_sample_days: int = 180,
    oos_days: int = 90,
    step_days: int = 90,
    seeds: Sequence[int] = PLACEBO_SEEDS_DEFAULT,
    max_windows: int = 14,
) -> Dict[str, Any]:
    logger.info("=== funding_basis research: full pipeline ===")
    sigs = fbs.compute_signals(assets=universe, timeframe=timeframe,
                                  save=True)
    one = run_funding_basis_backtest(universe=universe, timeframe=timeframe,
                                        signals_df=sigs)
    wf = funding_basis_walk_forward(universe=universe, timeframe=timeframe,
                                       in_sample_days=in_sample_days,
                                       oos_days=oos_days, step_days=step_days,
                                       signals_df=sigs,
                                       max_windows=max_windows)
    plac = funding_basis_placebo(universe=universe, timeframe=timeframe,
                                    seeds=seeds, signals_df=sigs)
    sc = funding_basis_scorecard(walk_forward_df=wf, placebo_df=plac)
    return {"signals": sigs, "single": one, "walk_forward": wf,
            "placebo": plac, "scorecard": sc}
