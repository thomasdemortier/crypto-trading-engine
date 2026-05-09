"""
Market-structure research orchestrator.

Single-window backtest, walk-forward, multi-seed placebo, scorecard.

Scorecard PASS criteria (8 checks, **stricter** than the v1 portfolio
scorecard, identical to the funding-rotation scorecard):

    1. Positive total return.
    2. Beats BTC OOS (>50 % of OOS windows).
    3. Beats equal-weight basket OOS.
    4. Beats simple momentum OOS.
    5. Beats placebo median return.
    6. OOS stability above 60 %.
    7. At least 10 rebalances total.
    8. Max drawdown not worse than BTC's by more than 20 pp.
    9. Enough market-structure data coverage.

`stability` here means: profitable AND beats BTC AND beats basket AND
beats simple momentum, per spec.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import (config, market_structure_signals as mss,
                market_structure_data_collector as mdc,
                portfolio_backtester as pb, utils)
from .portfolio_research import (
    DEFAULT_TIMEFRAME, MIN_OOS_WINDOWS_FOR_CONFIDENCE,
    MIN_REBALANCES_FOR_CONFIDENCE, PLACEBO_SEEDS_DEFAULT,
    PORTFOLIO_FAIL, PORTFOLIO_INCONCLUSIVE, PORTFOLIO_PASS,
    PORTFOLIO_WATCHLIST, _build_oos_windows, load_universe_with_report,
)
from .strategies.market_structure_allocator import (
    MarketStructureAllocatorConfig, MarketStructureAllocatorStrategy,
)
from .strategies.market_structure_vol_target_allocator import (
    MarketStructureVolTargetAllocatorStrategy,
    MarketStructureVolTargetConfig,
)
from .strategies.momentum_rotation import (
    MomentumRotationConfig, MomentumRotationStrategy,
)

logger = utils.get_logger("cte.market_structure_research")

# DD ceiling: strategy_dd - btc_dd >= -20 pp (drawdowns are negative).
MAX_DD_VS_BTC_GAP_PP = 20.0


# ---------------------------------------------------------------------------
# Placebo: random state-picker (BTC / alt-basket / cash) at each rebalance.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MarketStructureStatePlaceboConfig:
    btc_asset: str = "BTC/USDT"
    alt_universe: Sequence[str] = (
        "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT",
        "XRP/USDT", "DOGE/USDT", "ADA/USDT", "LTC/USDT", "BNB/USDT",
    )
    alt_top_n: int = 5
    seed: int = 42
    states: Sequence[str] = ("btc", "alt", "cash")  # uniform sampling


class MarketStructureStatePlacebo:
    """At every rebalance bar, picks one of {btc, alt, cash} uniformly
    at random with a fixed seed. When `alt`, picks N alt symbols from
    the alt universe at random (no momentum scoring) — that decouples
    placebo behaviour from any market-structure signal."""

    name = "market_structure_state_placebo"

    def __init__(self, cfg: Optional[MarketStructureStatePlaceboConfig] = None) -> None:
        self.cfg = cfg or MarketStructureStatePlaceboConfig()
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
        state = str(self._rng.choice(self.cfg.states))
        if state == "cash":
            return {}
        if state == "btc":
            df = asset_frames.get(self.cfg.btc_asset)
            if df is None or df.empty:
                return {}
            sub = df[df["timestamp"] <= int(asof_ts_ms)]
            if len(sub) < 2:
                return {}
            return {self.cfg.btc_asset: 1.0}
        # state == "alt"
        eligible = []
        for alt in self.cfg.alt_universe:
            df = asset_frames.get(alt)
            if df is None or df.empty:
                continue
            sub = df[df["timestamp"] <= int(asof_ts_ms)]
            if len(sub) >= 2:
                eligible.append(alt)
        if not eligible:
            return {}
        n = min(self.cfg.alt_top_n, len(eligible))
        chosen = list(self._rng.choice(eligible, size=n, replace=False))
        w = 1.0 / n
        return {asset: w for asset in chosen}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_signals(signals_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if signals_df is not None and not signals_df.empty:
        return signals_df
    p = config.RESULTS_DIR / "market_structure_signals.csv"
    if p.exists() and p.stat().st_size > 0:
        return pd.read_csv(p)
    logger.info("market_structure_signals.csv missing — computing on the fly")
    return mss.compute_market_structure_signals(save=True)


def _enough_coverage(min_required: int = 4 * 365) -> Optional[str]:
    """Return None if every required market-structure dataset clears the
    threshold; otherwise return a short reason string for the scorecard."""
    p = config.RESULTS_DIR / "market_structure_data_coverage.csv"
    if not p.exists() or p.stat().st_size == 0:
        return "no coverage CSV — run download_market_structure_data first"
    cov = pd.read_csv(p)
    if cov.empty:
        return "coverage CSV empty"
    bad = cov[~cov["enough_for_research"].astype(bool)]
    if not bad.empty:
        return ("insufficient coverage: "
                + ", ".join(f"{r['source']}:{r['dataset']}"
                             for _, r in bad.iterrows()))
    return None


# ---------------------------------------------------------------------------
# Single-window full backtest
# ---------------------------------------------------------------------------
def run_market_structure_allocator(
    universe: Sequence[str] = mss.ALLOCATOR_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    allocator_cfg: Optional[MarketStructureAllocatorConfig] = None,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    signals_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> Dict[str, Any]:
    utils.assert_paper_only()
    frames, _ = load_universe_with_report(assets=universe, timeframe=timeframe,
                                            save=save)
    if not frames:
        return {"ok": False, "reason": "no asset data"}
    sigs = _ensure_signals(signals_df)
    strat = MarketStructureAllocatorStrategy(sigs, allocator_cfg)
    cfg = backtest_cfg or pb.PortfolioBacktestConfig()
    art = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames,
        timeframe=timeframe, cfg=cfg, save=save,
        save_prefix="market_structure_allocator",
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
    rows = [{"strategy": "market_structure_allocator", **metrics}]
    for name, m in bench_metrics.items():
        rows.append({"strategy": name, **m})
    cmp_df = pd.DataFrame(rows)
    if save:
        utils.write_df(
            cmp_df,
            config.RESULTS_DIR / "market_structure_allocator_comparison.csv",
        )
    return {"ok": True, "artifacts": art, "benchmarks": bench,
            "metrics": metrics, "bench_metrics": bench_metrics,
            "comparison_df": cmp_df}


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def market_structure_walk_forward(
    universe: Sequence[str] = mss.ALLOCATOR_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    in_sample_days: int = 180,
    oos_days: int = 90,
    step_days: int = 90,
    allocator_cfg: Optional[MarketStructureAllocatorConfig] = None,
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
                            config.RESULTS_DIR / "market_structure_walk_forward.csv")
        return out
    sigs = _ensure_signals(signals_df)

    ts_axis = pb._aligned_timestamps(frames)
    if len(ts_axis) < 2:
        out = pd.DataFrame()
        if save:
            utils.write_df(out,
                            config.RESULTS_DIR / "market_structure_walk_forward.csv")
        return out
    first_ts, last_ts = ts_axis[0], ts_axis[-1]
    if not sigs.empty and "timestamp" in sigs.columns:
        first_sig_ts = int(pd.to_numeric(sigs["timestamp"]).min())
        first_ts = max(first_ts, first_sig_ts)
    windows = _build_oos_windows(first_ts, last_ts,
                                  in_sample_days, oos_days, step_days)

    strat = MarketStructureAllocatorStrategy(sigs, allocator_cfg)
    simple = MomentumRotationStrategy(MomentumRotationConfig())
    rows: List[Dict] = []
    for w_idx, w in enumerate(windows, start=1):
        logger.info("market_structure_walk_forward [%d/%d] OOS %s → %s",
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
        utils.write_df(out, config.RESULTS_DIR / "market_structure_walk_forward.csv")
    return out


# ---------------------------------------------------------------------------
# Placebo
# ---------------------------------------------------------------------------
def market_structure_placebo(
    universe: Sequence[str] = mss.ALLOCATOR_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    seeds: Sequence[int] = PLACEBO_SEEDS_DEFAULT,
    allocator_cfg: Optional[MarketStructureAllocatorConfig] = None,
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
                            config.RESULTS_DIR / "market_structure_placebo.csv")
        return out
    sigs = _ensure_signals(signals_df)

    # Strategy run on the full window.
    strat = MarketStructureAllocatorStrategy(sigs, allocator_cfg)
    strat_art = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames, timeframe=timeframe,
        cfg=cfg, save=False,
    )
    strat_m = pb.portfolio_metrics(strat_art.equity_curve, cfg.starting_capital)

    rows: List[Dict] = []
    for seed in seeds:
        plac = MarketStructureStatePlacebo(
            MarketStructureStatePlaceboConfig(seed=int(seed)),
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
        "strategy": "market_structure_allocator",
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
        utils.write_df(out, config.RESULTS_DIR / "market_structure_placebo.csv")
    return out


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
def market_structure_scorecard(
    walk_forward_df: Optional[pd.DataFrame] = None,
    placebo_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> pd.DataFrame:
    if walk_forward_df is None:
        p = config.RESULTS_DIR / "market_structure_walk_forward.csv"
        walk_forward_df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    if placebo_df is None:
        p = config.RESULTS_DIR / "market_structure_placebo.csv"
        placebo_df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    cmp_p = config.RESULTS_DIR / "market_structure_allocator_comparison.csv"
    cmp_df = pd.read_csv(cmp_p) if cmp_p.exists() else pd.DataFrame()

    coverage_issue = _enough_coverage()

    if walk_forward_df.empty:
        out = pd.DataFrame([{
            "strategy_name": "market_structure_allocator",
            "verdict": PORTFOLIO_INCONCLUSIVE,
            "reason": "no walk-forward data",
            "coverage_note": coverage_issue or "ok",
        }])
        if save:
            utils.write_df(out,
                            config.RESULTS_DIR / "market_structure_scorecard.csv")
        return out

    ok = walk_forward_df
    if "error" in ok.columns:
        ok = ok[ok["error"].isna()]
    n_windows = int(len(ok))
    if n_windows < MIN_OOS_WINDOWS_FOR_CONFIDENCE:
        out = pd.DataFrame([{
            "strategy_name": "market_structure_allocator",
            "n_windows": n_windows,
            "verdict": PORTFOLIO_INCONCLUSIVE,
            "reason": f"only {n_windows} OOS windows — need ≥"
                       f"{MIN_OOS_WINDOWS_FOR_CONFIDENCE}",
            "coverage_note": coverage_issue or "ok",
        }])
        if save:
            utils.write_df(out,
                            config.RESULTS_DIR / "market_structure_scorecard.csv")
        return out

    avg_oos_return = float(ok["oos_return_pct"].mean())
    avg_oos_dd = float(ok["oos_max_drawdown_pct"].mean())
    pct_profitable = float((ok["oos_return_pct"] > 0).mean() * 100.0)
    pct_beats_btc = float(ok["beats_btc"].mean() * 100.0)
    pct_beats_basket = float(ok["beats_basket"].mean() * 100.0)
    pct_beats_simple = (float(ok["beats_simple_momentum"].mean() * 100.0)
                         if "beats_simple_momentum" in ok.columns else 0.0)
    # Stability per spec: profitable AND beats BTC AND beats basket AND
    # beats simple momentum.
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
        if not (np.isnan(placebo_median_return) or np.isnan(strategy_full_return))
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
        ("beats_btc_oos", pct_beats_btc > 50.0),
        ("beats_basket_oos", pct_beats_basket > 50.0),
        ("beats_simple_momentum_oos", pct_beats_simple > 50.0),
        ("beats_placebo_median", beats_placebo_median),
        ("oos_stability_above_60", stability > 60.0),
        ("at_least_10_rebalances",
         total_rebalances >= MIN_REBALANCES_FOR_CONFIDENCE),
        ("dd_within_btc_gap_20pp", bool(dd_gap_ok)),
        ("enough_market_structure_coverage", coverage_issue is None),
    ]
    n_pass_satisfied = sum(1 for _, ok_ in pass_required if ok_)
    if all(ok_ for _, ok_ in pass_required):
        verdict = PORTFOLIO_PASS
    elif coverage_issue is not None:
        verdict = PORTFOLIO_INCONCLUSIVE
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
        "strategy_name": "market_structure_allocator",
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
        "coverage_note": coverage_issue or "ok",
        "verdict": verdict,
        "checks_passed": n_pass_satisfied,
        "checks_total": len(pass_required),
        "reason": "; ".join(f"{k}={ok_}" for k, ok_ in pass_required),
    }])
    if save:
        utils.write_df(out, config.RESULTS_DIR / "market_structure_scorecard.csv")
    return out


# ---------------------------------------------------------------------------
# One-shot orchestrator
# ---------------------------------------------------------------------------
def run_all_market_structure(
    universe: Sequence[str] = mss.ALLOCATOR_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    in_sample_days: int = 180,
    oos_days: int = 90,
    step_days: int = 90,
    seeds: Sequence[int] = PLACEBO_SEEDS_DEFAULT,
) -> Dict[str, Any]:
    logger.info("=== market structure research: full pipeline ===")
    sigs = mss.compute_market_structure_signals(save=True)
    one = run_market_structure_allocator(universe=universe, timeframe=timeframe,
                                            signals_df=sigs)
    wf = market_structure_walk_forward(
        universe=universe, timeframe=timeframe,
        in_sample_days=in_sample_days, oos_days=oos_days, step_days=step_days,
        signals_df=sigs,
    )
    plac = market_structure_placebo(
        universe=universe, timeframe=timeframe,
        seeds=seeds, signals_df=sigs,
    )
    sc = market_structure_scorecard(walk_forward_df=wf, placebo_df=plac)
    return {"signals": sigs, "single": one, "walk_forward": wf,
            "placebo": plac, "scorecard": sc}


# ===========================================================================
# Vol-target variant — same signals, softer exposure bands.
# ===========================================================================

# Placebo for the vol-target variant: 5-state random picker mirroring
# the strategy's allocation bands (alt/btc/partial-btc/defensive/cash).
@dataclass(frozen=True)
class MarketStructureVolStatePlaceboConfig:
    btc_asset: str = "BTC/USDT"
    alt_universe: Sequence[str] = (
        "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT",
        "XRP/USDT", "DOGE/USDT", "ADA/USDT", "LTC/USDT", "BNB/USDT",
    )
    alt_top_n: int = 5
    seed: int = 42
    states: Sequence[str] = (
        "alt_basket", "btc_only", "partial_btc",
        "defensive_partial", "cash",
    )
    # Match the strategy's bands so the placebo can land in the same
    # exposure space, not a stricter or looser one.
    alt_basket_alt_weight: float = 0.70
    alt_basket_btc_weight: float = 0.30
    partial_btc_weight: float = 0.70
    defensive_btc_weight: float = 0.30


class MarketStructureVolStatePlacebo:
    """Random 5-state allocator: at each rebalance bar, picks a state
    uniformly at random with a fixed seed, then applies the matching
    exposure band. The placebo NEVER reads `market_structure_state`."""

    name = "market_structure_vol_state_placebo"

    def __init__(
        self,
        cfg: Optional[MarketStructureVolStatePlaceboConfig] = None,
    ) -> None:
        self.cfg = cfg or MarketStructureVolStatePlaceboConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

    def reset(self, seed: Optional[int] = None) -> None:
        s = self.cfg.seed if seed is None else int(seed)
        self._rng = np.random.default_rng(s)

    def _btc_only(self, asset_frames, asof_ts_ms, weight: float) -> Dict[str, float]:
        df = asset_frames.get(self.cfg.btc_asset)
        if df is None or df.empty:
            return {}
        sub = df[df["timestamp"] <= int(asof_ts_ms)]
        if len(sub) < 2:
            return {}
        return {self.cfg.btc_asset: weight}

    def _random_alts(self, asset_frames, asof_ts_ms,
                      total_weight: float) -> Dict[str, float]:
        eligible = []
        for alt in self.cfg.alt_universe:
            df = asset_frames.get(alt)
            if df is None or df.empty:
                continue
            sub = df[df["timestamp"] <= int(asof_ts_ms)]
            if len(sub) >= 2:
                eligible.append(alt)
        if not eligible:
            return {}
        n = min(self.cfg.alt_top_n, len(eligible))
        chosen = list(self._rng.choice(eligible, size=n, replace=False))
        w = total_weight / n
        return {asset: w for asset in chosen}

    def target_weights(
        self,
        asof_ts_ms: int,
        asset_frames: Dict[str, pd.DataFrame],
        timeframe: str = "1d",
    ) -> Dict[str, float]:
        state = str(self._rng.choice(self.cfg.states))
        if state == "cash":
            return {}
        if state == "btc_only":
            return self._btc_only(asset_frames, asof_ts_ms, weight=1.0)
        if state == "partial_btc":
            return self._btc_only(asset_frames, asof_ts_ms,
                                    weight=self.cfg.partial_btc_weight)
        if state == "defensive_partial":
            return self._btc_only(asset_frames, asof_ts_ms,
                                    weight=self.cfg.defensive_btc_weight)
        if state == "alt_basket":
            alt_part = self._random_alts(
                asset_frames, asof_ts_ms,
                total_weight=self.cfg.alt_basket_alt_weight,
            )
            btc_part = self._btc_only(
                asset_frames, asof_ts_ms,
                weight=self.cfg.alt_basket_btc_weight,
            )
            out = dict(alt_part)
            out.update(btc_part)
            return out
        return {}


# ---------------------------------------------------------------------------
# Single-window full backtest — vol-target variant
# ---------------------------------------------------------------------------
def run_market_structure_vol_target(
    universe: Sequence[str] = mss.ALLOCATOR_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    vol_cfg: Optional[MarketStructureVolTargetConfig] = None,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    signals_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> Dict[str, Any]:
    utils.assert_paper_only()
    frames, _ = load_universe_with_report(assets=universe, timeframe=timeframe,
                                            save=save)
    if not frames:
        return {"ok": False, "reason": "no asset data"}
    sigs = _ensure_signals(signals_df)
    cfg = backtest_cfg or pb.PortfolioBacktestConfig()

    strat = MarketStructureVolTargetAllocatorStrategy(sigs, vol_cfg)
    art = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames,
        timeframe=timeframe, cfg=cfg, save=save,
        save_prefix="market_structure_vol_target",
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
    # Original allocator as an explicit benchmark.
    orig = MarketStructureAllocatorStrategy(sigs)
    orig_art = pb.run_portfolio_backtest(
        portfolio_strategy=orig, asset_frames=frames,
        timeframe=timeframe, cfg=cfg, save=False,
    )
    bench["market_structure_allocator_original"] = orig_art.equity_curve

    metrics = pb.portfolio_metrics(art.equity_curve, cfg.starting_capital)
    bench_metrics = {n: pb.portfolio_metrics(d, cfg.starting_capital)
                      for n, d in bench.items()}
    rows = [{"strategy": "market_structure_vol_target", **metrics}]
    for name, m in bench_metrics.items():
        rows.append({"strategy": name, **m})
    cmp_df = pd.DataFrame(rows)
    if save:
        utils.write_df(
            cmp_df,
            config.RESULTS_DIR / "market_structure_vol_target_comparison.csv",
        )
    return {"ok": True, "artifacts": art, "benchmarks": bench,
            "metrics": metrics, "bench_metrics": bench_metrics,
            "comparison_df": cmp_df}


# ---------------------------------------------------------------------------
# Walk-forward — vol-target variant (per-window comparison vs original)
# ---------------------------------------------------------------------------
def market_structure_vol_target_walk_forward(
    universe: Sequence[str] = mss.ALLOCATOR_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    in_sample_days: int = 180,
    oos_days: int = 90,
    step_days: int = 90,
    vol_cfg: Optional[MarketStructureVolTargetConfig] = None,
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
            utils.write_df(
                out,
                config.RESULTS_DIR / "market_structure_vol_target_walk_forward.csv",
            )
        return out
    sigs = _ensure_signals(signals_df)

    ts_axis = pb._aligned_timestamps(frames)
    if len(ts_axis) < 2:
        out = pd.DataFrame()
        if save:
            utils.write_df(
                out,
                config.RESULTS_DIR / "market_structure_vol_target_walk_forward.csv",
            )
        return out
    first_ts, last_ts = ts_axis[0], ts_axis[-1]
    if not sigs.empty and "timestamp" in sigs.columns:
        first_sig_ts = int(pd.to_numeric(sigs["timestamp"]).min())
        first_ts = max(first_ts, first_sig_ts)
    windows = _build_oos_windows(first_ts, last_ts,
                                  in_sample_days, oos_days, step_days)

    vol_strat = MarketStructureVolTargetAllocatorStrategy(sigs, vol_cfg)
    orig_strat = MarketStructureAllocatorStrategy(sigs)
    simple_strat = MomentumRotationStrategy(MomentumRotationConfig())
    rows: List[Dict] = []
    for w_idx, w in enumerate(windows, start=1):
        logger.info(
            "vol_target_walk_forward [%d/%d] OOS %s → %s",
            w_idx, len(windows),
            pd.to_datetime(w["oos_start_ms"], unit="ms", utc=True).date(),
            pd.to_datetime(w["oos_end_ms"], unit="ms", utc=True).date(),
        )
        art = pb.run_portfolio_backtest(
            portfolio_strategy=vol_strat, asset_frames=frames,
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
            portfolio_strategy=simple_strat, asset_frames=frames,
            timeframe=timeframe, cfg=cfg,
            start_ts_ms=w["oos_start_ms"], end_ts_ms=w["oos_end_ms"],
            save=False,
        )
        orig_art = pb.run_portfolio_backtest(
            portfolio_strategy=orig_strat, asset_frames=frames,
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
        orig_ret = pb.portfolio_metrics(
            orig_art.equity_curve, cfg.starting_capital,
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
            "original_allocator_oos_return_pct": orig_ret,
            "beats_btc": bool(m["total_return_pct"] > btc_ret),
            "beats_basket": bool(m["total_return_pct"] > bk_ret),
            "beats_simple_momentum": bool(m["total_return_pct"] > simple_ret),
            "beats_original_allocator": bool(m["total_return_pct"] > orig_ret),
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
            config.RESULTS_DIR / "market_structure_vol_target_walk_forward.csv",
        )
    return out


# ---------------------------------------------------------------------------
# Placebo — vol-target variant
# ---------------------------------------------------------------------------
def market_structure_vol_target_placebo(
    universe: Sequence[str] = mss.ALLOCATOR_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    seeds: Sequence[int] = PLACEBO_SEEDS_DEFAULT,
    vol_cfg: Optional[MarketStructureVolTargetConfig] = None,
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
            utils.write_df(
                out,
                config.RESULTS_DIR / "market_structure_vol_target_placebo.csv",
            )
        return out
    sigs = _ensure_signals(signals_df)
    strat = MarketStructureVolTargetAllocatorStrategy(sigs, vol_cfg)
    strat_art = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames, timeframe=timeframe,
        cfg=cfg, save=False,
    )
    strat_m = pb.portfolio_metrics(strat_art.equity_curve, cfg.starting_capital)

    rows: List[Dict] = []
    for seed in seeds:
        plac = MarketStructureVolStatePlacebo(
            MarketStructureVolStatePlaceboConfig(seed=int(seed)),
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
        "strategy": "market_structure_vol_target_allocator",
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
        utils.write_df(
            out,
            config.RESULTS_DIR / "market_structure_vol_target_placebo.csv",
        )
    return out


# ---------------------------------------------------------------------------
# Scorecard — vol-target variant (10 checks)
# ---------------------------------------------------------------------------
def market_structure_vol_target_scorecard(
    walk_forward_df: Optional[pd.DataFrame] = None,
    placebo_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> pd.DataFrame:
    if walk_forward_df is None:
        p = config.RESULTS_DIR / "market_structure_vol_target_walk_forward.csv"
        walk_forward_df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    if placebo_df is None:
        p = config.RESULTS_DIR / "market_structure_vol_target_placebo.csv"
        placebo_df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    cmp_p = config.RESULTS_DIR / "market_structure_vol_target_comparison.csv"
    cmp_df = pd.read_csv(cmp_p) if cmp_p.exists() else pd.DataFrame()

    coverage_issue = _enough_coverage()

    if walk_forward_df.empty:
        out = pd.DataFrame([{
            "strategy_name": "market_structure_vol_target_allocator",
            "verdict": PORTFOLIO_INCONCLUSIVE,
            "reason": "no walk-forward data",
            "coverage_note": coverage_issue or "ok",
        }])
        if save:
            utils.write_df(
                out,
                config.RESULTS_DIR / "market_structure_vol_target_scorecard.csv",
            )
        return out

    ok = walk_forward_df
    if "error" in ok.columns:
        ok = ok[ok["error"].isna()]
    n_windows = int(len(ok))
    if n_windows < MIN_OOS_WINDOWS_FOR_CONFIDENCE:
        out = pd.DataFrame([{
            "strategy_name": "market_structure_vol_target_allocator",
            "n_windows": n_windows,
            "verdict": PORTFOLIO_INCONCLUSIVE,
            "reason": f"only {n_windows} OOS windows — need ≥"
                       f"{MIN_OOS_WINDOWS_FOR_CONFIDENCE}",
            "coverage_note": coverage_issue or "ok",
        }])
        if save:
            utils.write_df(
                out,
                config.RESULTS_DIR / "market_structure_vol_target_scorecard.csv",
            )
        return out

    avg_oos_return = float(ok["oos_return_pct"].mean())
    avg_oos_dd = float(ok["oos_max_drawdown_pct"].mean())
    pct_profitable = float((ok["oos_return_pct"] > 0).mean() * 100.0)
    pct_beats_btc = float(ok["beats_btc"].mean() * 100.0)
    pct_beats_basket = float(ok["beats_basket"].mean() * 100.0)
    pct_beats_simple = (float(ok["beats_simple_momentum"].mean() * 100.0)
                         if "beats_simple_momentum" in ok.columns else 0.0)
    pct_beats_original = (
        float(ok["beats_original_allocator"].mean() * 100.0)
        if "beats_original_allocator" in ok.columns else 0.0)
    # Stability per spec: profitable AND beats every primary benchmark
    # (BTC, basket, simple momentum, original allocator).
    stab_mask = (ok["profitable"] & ok["beats_btc"] & ok["beats_basket"])
    if "beats_simple_momentum" in ok.columns:
        stab_mask &= ok["beats_simple_momentum"]
    if "beats_original_allocator" in ok.columns:
        stab_mask &= ok["beats_original_allocator"]
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
        if not (np.isnan(placebo_median_return) or np.isnan(strategy_full_return))
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
        ("beats_btc_oos", pct_beats_btc > 50.0),
        ("beats_basket_oos", pct_beats_basket > 50.0),
        ("beats_simple_momentum_oos", pct_beats_simple > 50.0),
        ("beats_original_allocator_oos", pct_beats_original > 50.0),
        ("beats_placebo_median", beats_placebo_median),
        ("oos_stability_above_60", stability > 60.0),
        ("at_least_10_rebalances",
         total_rebalances >= MIN_REBALANCES_FOR_CONFIDENCE),
        ("dd_within_btc_gap_20pp", bool(dd_gap_ok)),
        ("enough_market_structure_coverage", coverage_issue is None),
    ]
    n_pass_satisfied = sum(1 for _, ok_ in pass_required if ok_)
    if all(ok_ for _, ok_ in pass_required):
        verdict = PORTFOLIO_PASS
    elif coverage_issue is not None:
        verdict = PORTFOLIO_INCONCLUSIVE
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
        "strategy_name": "market_structure_vol_target_allocator",
        "n_windows": n_windows,
        "avg_oos_return_pct": avg_oos_return,
        "avg_oos_drawdown_pct": avg_oos_dd,
        "pct_windows_profitable": pct_profitable,
        "pct_windows_beat_btc": pct_beats_btc,
        "pct_windows_beat_basket": pct_beats_basket,
        "pct_windows_beat_simple_momentum": pct_beats_simple,
        "pct_windows_beat_original_allocator": pct_beats_original,
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
        utils.write_df(
            out,
            config.RESULTS_DIR / "market_structure_vol_target_scorecard.csv",
        )
    return out


def run_all_market_structure_vol_target(
    universe: Sequence[str] = mss.ALLOCATOR_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    in_sample_days: int = 180,
    oos_days: int = 90,
    step_days: int = 90,
    seeds: Sequence[int] = PLACEBO_SEEDS_DEFAULT,
) -> Dict[str, Any]:
    logger.info("=== market structure vol-target research: full pipeline ===")
    sigs = mss.compute_market_structure_signals(save=True)
    one = run_market_structure_vol_target(
        universe=universe, timeframe=timeframe, signals_df=sigs,
    )
    wf = market_structure_vol_target_walk_forward(
        universe=universe, timeframe=timeframe,
        in_sample_days=in_sample_days, oos_days=oos_days, step_days=step_days,
        signals_df=sigs,
    )
    plac = market_structure_vol_target_placebo(
        universe=universe, timeframe=timeframe,
        seeds=seeds, signals_df=sigs,
    )
    sc = market_structure_vol_target_scorecard(
        walk_forward_df=wf, placebo_df=plac,
    )
    return {"signals": sigs, "single": one, "walk_forward": wf,
            "placebo": plac, "scorecard": sc}
