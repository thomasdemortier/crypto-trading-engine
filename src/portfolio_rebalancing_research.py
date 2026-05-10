"""
Portfolio rebalancing research orchestrator.

Runs the locked-weight, monthly-rebalance allocator through:
    * single full-window backtest + benchmark comparison
    * 14-window walk-forward analysis
    * 20-seed placebo (random fixed-weight allocator)
    * scorecard against the LOCKED rebalancing-specific PASS criteria
    * diagnostics

PASS criteria (locked, from spec — NEVER tuned):
    1. Sharpe within 0.10 of BTC b&h.
    2. Max drawdown >= 15 percentage points tighter than BTC b&h.
    3. Beats placebo MEDIAN return.
    4. Beats placebo MEDIAN drawdown.
    5. >= 24 rebalances total across the OOS window(s).

These criteria are deliberately NOT 'beat BTC outright on return' —
that is the bar that exhausted every prior long-only allocator. A
fixed-weight rebalancer cannot beat BTC in a BTC bull regime; the
right question is whether it tightens drawdown and delivers
comparable risk-adjusted return.

Hard rules:
    * No network calls.
    * No broker imports.
    * No API key reads.
    * No order placement strings.
    * No paper-trading or live-trading enablement.
    * Generated CSVs are written under `results/` and gitignored.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config, portfolio_backtester as pb, utils
from .portfolio_research import (
    DEFAULT_TIMEFRAME, MIN_OOS_WINDOWS_FOR_CONFIDENCE,
    PORTFOLIO_FAIL, PORTFOLIO_INCONCLUSIVE, PORTFOLIO_PASS,
    PORTFOLIO_WATCHLIST, _build_oos_windows, load_universe_with_report,
)
from .strategies.portfolio_rebalancing_allocator import (
    PortfolioRebalancingAllocator, PortfolioRebalancingConfig,
)

logger = utils.get_logger("cte.portfolio_rebalancing_research")

# Locked rebalancing-specific gates (NEVER tuned).
SHARPE_GAP_LIMIT = 0.10            # |strategy_sharpe - btc_sharpe| <= 0.10
DRAWDOWN_TIGHTNESS_PP = 15.0       # strategy_dd - btc_dd >= 15 pp tighter
MIN_REBALANCES_TOTAL = 24

DEFAULT_UNIVERSE: Tuple[str, ...] = ("BTC/USDT", "ETH/USDT")
PLACEBO_SEEDS_DEFAULT: Tuple[int, ...] = tuple(range(20))


# ---------------------------------------------------------------------------
# Placebo: random fixed-weight allocator.
#
# At construction time the placebo draws a single (w_btc, w_eth) vector
# from Uniform — w_btc ~ U(0, 1), w_eth ~ U(0, 1 - w_btc). It returns
# THAT FIXED vector on every rebalance for the full window. The only
# difference vs the strategy is the weight choice; cadence and
# constraints (long-only, Σ <= 1, no leverage) are identical.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RebalancingPlaceboConfig:
    seed: int = 42
    btc_asset: str = "BTC/USDT"
    eth_asset: str = "ETH/USDT"


class RebalancingPlacebo:
    """Random fixed-weight allocator. Same constraints as the strategy:
    long-only, Σ <= 1, no leverage. Difference: weights drawn at
    construction time from a uniform prior."""

    name = "portfolio_rebalancing_placebo"

    def __init__(
        self, cfg: Optional[RebalancingPlaceboConfig] = None,
    ) -> None:
        self.cfg = cfg or RebalancingPlaceboConfig()
        rng = np.random.default_rng(int(self.cfg.seed))
        self._w_btc = float(rng.uniform(0.0, 1.0))
        # ETH share drawn from the remaining cash budget.
        self._w_eth = float(rng.uniform(0.0, 1.0 - self._w_btc))

    @property
    def weights(self) -> Tuple[float, float]:
        return self._w_btc, self._w_eth

    def target_weights(
        self,
        asof_ts_ms: int,
        asset_frames: Dict[str, pd.DataFrame],
        timeframe: str = "1d",
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        btc = asset_frames.get(self.cfg.btc_asset)
        eth = asset_frames.get(self.cfg.eth_asset)
        if btc is not None and not btc.empty:
            sub = btc[btc["timestamp"] <= int(asof_ts_ms)]
            if len(sub) >= 2 and self._w_btc > 0:
                out[self.cfg.btc_asset] = float(self._w_btc)
        if eth is not None and not eth.empty:
            sub = eth[eth["timestamp"] <= int(asof_ts_ms)]
            if len(sub) >= 2 and self._w_eth > 0:
                out[self.cfg.eth_asset] = float(self._w_eth)
        return out


# ---------------------------------------------------------------------------
# Single-window full backtest
# ---------------------------------------------------------------------------
def run_portfolio_rebalancing_backtest(
    universe: Sequence[str] = DEFAULT_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    strat_cfg: Optional[PortfolioRebalancingConfig] = None,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    save: bool = True,
) -> Dict[str, Any]:
    utils.assert_paper_only()
    frames, _ = load_universe_with_report(
        assets=universe, timeframe=timeframe, save=save,
    )
    if not frames:
        return {"ok": False, "reason": "no asset data"}
    cfg_local = strat_cfg or PortfolioRebalancingConfig()
    cfg = backtest_cfg or pb.PortfolioBacktestConfig(
        rebalance_frequency=cfg_local.rebalance_frequency,
    )
    strat = PortfolioRebalancingAllocator(cfg_local)
    art = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames,
        timeframe=timeframe, cfg=cfg, save=save,
        save_prefix="portfolio_rebalancing_backtest",
    )
    bench = pb.benchmark_equity_curves(
        frames, starting_capital=cfg.starting_capital, timeframe=timeframe,
    )
    metrics = pb.portfolio_metrics(art.equity_curve, cfg.starting_capital)
    bench_metrics = {n: pb.portfolio_metrics(d, cfg.starting_capital)
                      for n, d in bench.items()}
    rows = [{"strategy": strat.name, **metrics}]
    for name, m in bench_metrics.items():
        rows.append({"strategy": name, **m})
    cmp_df = pd.DataFrame(rows)
    if save:
        utils.write_df(
            cmp_df,
            config.RESULTS_DIR / "portfolio_rebalancing_comparison.csv",
        )
    return {
        "ok": True, "artifacts": art, "benchmarks": bench,
        "metrics": metrics, "bench_metrics": bench_metrics,
        "comparison_df": cmp_df,
    }


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def portfolio_rebalancing_walk_forward(
    universe: Sequence[str] = DEFAULT_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    in_sample_days: int = 180,
    oos_days: int = 90,
    step_days: int = 90,
    strat_cfg: Optional[PortfolioRebalancingConfig] = None,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    save: bool = True,
    max_windows: int = 14,
) -> pd.DataFrame:
    utils.assert_paper_only()
    cfg_local = strat_cfg or PortfolioRebalancingConfig()
    cfg = backtest_cfg or pb.PortfolioBacktestConfig(
        rebalance_frequency=cfg_local.rebalance_frequency,
    )
    frames, _ = load_universe_with_report(
        assets=universe, timeframe=timeframe, save=False,
    )
    if not frames:
        out = pd.DataFrame()
        if save:
            utils.write_df(out, config.RESULTS_DIR
                                / "portfolio_rebalancing_walk_forward.csv")
        return out
    ts_axis = pb._aligned_timestamps(frames)
    if len(ts_axis) < 2:
        out = pd.DataFrame()
        if save:
            utils.write_df(out, config.RESULTS_DIR
                                / "portfolio_rebalancing_walk_forward.csv")
        return out
    first_ts, last_ts = ts_axis[0], ts_axis[-1]
    windows = _build_oos_windows(first_ts, last_ts, in_sample_days,
                                    oos_days, step_days)
    if max_windows is not None and len(windows) > max_windows:
        windows = windows[-max_windows:]
    strat = PortfolioRebalancingAllocator(cfg_local)
    rows: List[Dict] = []
    for w_idx, w in enumerate(windows, start=1):
        logger.info(
            "portfolio_rebalancing_walk_forward [%d/%d] OOS %s -> %s",
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
        btc_m = pb.portfolio_metrics(
            bench.get("BTC_buy_and_hold", pd.DataFrame()),
            cfg.starting_capital)
        eth_m = pb.portfolio_metrics(
            bench.get("ETH_buy_and_hold", pd.DataFrame()),
            cfg.starting_capital)
        bk_m = pb.portfolio_metrics(
            bench.get("equal_weight_basket", pd.DataFrame()),
            cfg.starting_capital)
        n_rebalances = (int(art.weights_history["filled"].sum())
                         if not art.weights_history.empty else 0)
        rows.append({
            "window": w_idx,
            "oos_start_iso": pd.to_datetime(w["oos_start_ms"], unit="ms",
                                              utc=True).isoformat(),
            "oos_end_iso": pd.to_datetime(w["oos_end_ms"], unit="ms",
                                            utc=True).isoformat(),
            "oos_return_pct": m["total_return_pct"],
            "oos_max_drawdown_pct": m["max_drawdown_pct"],
            "oos_sharpe": m["sharpe_ratio"],
            "btc_return_pct": btc_m["total_return_pct"],
            "btc_drawdown_pct": btc_m["max_drawdown_pct"],
            "btc_sharpe": btc_m["sharpe_ratio"],
            "eth_return_pct": eth_m["total_return_pct"],
            "basket_return_pct": bk_m["total_return_pct"],
            "n_rebalances": n_rebalances,
            "sharpe_within_010": bool(
                abs(m["sharpe_ratio"] - btc_m["sharpe_ratio"])
                <= SHARPE_GAP_LIMIT),
            "drawdown_15pp_tighter": bool(
                m["max_drawdown_pct"] - btc_m["max_drawdown_pct"]
                >= DRAWDOWN_TIGHTNESS_PP),
            "error": None,
        })
    out = pd.DataFrame(rows)
    if save:
        utils.write_df(out, config.RESULTS_DIR
                            / "portfolio_rebalancing_walk_forward.csv")
    return out


# ---------------------------------------------------------------------------
# Placebo
# ---------------------------------------------------------------------------
def portfolio_rebalancing_placebo(
    universe: Sequence[str] = DEFAULT_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    seeds: Sequence[int] = PLACEBO_SEEDS_DEFAULT,
    strat_cfg: Optional[PortfolioRebalancingConfig] = None,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    save: bool = True,
) -> pd.DataFrame:
    utils.assert_paper_only()
    cfg_local = strat_cfg or PortfolioRebalancingConfig()
    cfg = backtest_cfg or pb.PortfolioBacktestConfig(
        rebalance_frequency=cfg_local.rebalance_frequency,
    )
    frames, _ = load_universe_with_report(
        assets=universe, timeframe=timeframe, save=False,
    )
    if not frames:
        out = pd.DataFrame()
        if save:
            utils.write_df(out, config.RESULTS_DIR
                                / "portfolio_rebalancing_placebo.csv")
        return out

    # Strategy run on the full window for headline numbers.
    strat = PortfolioRebalancingAllocator(cfg_local)
    strat_art = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames,
        timeframe=timeframe, cfg=cfg, save=False,
    )
    strat_m = pb.portfolio_metrics(strat_art.equity_curve, cfg.starting_capital)

    rows: List[Dict] = []
    for seed in seeds:
        plac = RebalancingPlacebo(RebalancingPlaceboConfig(seed=int(seed)))
        art = pb.run_portfolio_backtest(
            portfolio_strategy=plac, asset_frames=frames,
            timeframe=timeframe, cfg=cfg, save=False,
        )
        m = pb.portfolio_metrics(art.equity_curve, cfg.starting_capital)
        rows.append({
            "seed": int(seed),
            "w_btc": float(plac.weights[0]),
            "w_eth": float(plac.weights[1]),
            "placebo_return_pct": m["total_return_pct"],
            "placebo_max_drawdown_pct": m["max_drawdown_pct"],
            "placebo_sharpe": m["sharpe_ratio"],
        })
    plac_df = pd.DataFrame(rows)
    median_ret = (float(plac_df["placebo_return_pct"].median())
                   if not plac_df.empty else float("nan"))
    median_dd = (float(plac_df["placebo_max_drawdown_pct"].median())
                  if not plac_df.empty else float("nan"))
    summary = pd.DataFrame([{
        "strategy": "portfolio_rebalancing_allocator",
        "strategy_return_pct": strat_m["total_return_pct"],
        "strategy_max_drawdown_pct": strat_m["max_drawdown_pct"],
        "strategy_sharpe": strat_m["sharpe_ratio"],
        "n_seeds": len(seeds),
        "placebo_median_return_pct": median_ret,
        "placebo_median_drawdown_pct": median_dd,
        "strategy_beats_median_return": bool(
            (not np.isnan(median_ret))
            and strat_m["total_return_pct"] > median_ret),
        "strategy_beats_median_drawdown": bool(
            (not np.isnan(median_dd))
            and strat_m["max_drawdown_pct"] > median_dd),
        "placebo_return_percentile": (
            float((plac_df["placebo_return_pct"]
                     <= strat_m["total_return_pct"]).mean() * 100.0)
            if not plac_df.empty else float("nan")),
        "placebo_drawdown_percentile": (
            float((plac_df["placebo_max_drawdown_pct"]
                     <= strat_m["max_drawdown_pct"]).mean() * 100.0)
            if not plac_df.empty else float("nan")),
    }])
    out = pd.concat(
        [summary, plac_df.assign(strategy="placebo_seed_runs")],
        ignore_index=True, sort=False,
    )
    if save:
        utils.write_df(out, config.RESULTS_DIR
                            / "portfolio_rebalancing_placebo.csv")
    return out


# ---------------------------------------------------------------------------
# Scorecard — strict, locked thresholds.
# ---------------------------------------------------------------------------
def portfolio_rebalancing_scorecard(
    walk_forward_df: Optional[pd.DataFrame] = None,
    placebo_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> pd.DataFrame:
    if walk_forward_df is None:
        p = config.RESULTS_DIR / "portfolio_rebalancing_walk_forward.csv"
        walk_forward_df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    if placebo_df is None:
        p = config.RESULTS_DIR / "portfolio_rebalancing_placebo.csv"
        placebo_df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    cmp_p = config.RESULTS_DIR / "portfolio_rebalancing_comparison.csv"
    cmp_df = pd.read_csv(cmp_p) if cmp_p.exists() else pd.DataFrame()

    if (walk_forward_df.empty or placebo_df.empty or cmp_df.empty):
        out = pd.DataFrame([{
            "strategy_name": "portfolio_rebalancing_allocator",
            "verdict": PORTFOLIO_INCONCLUSIVE,
            "notes": "missing walk-forward, placebo, or comparison CSV",
        }])
        if save:
            utils.write_df(out, config.RESULTS_DIR
                                / "portfolio_rebalancing_scorecard.csv")
        return out

    # Full-window numbers from the comparison CSV.
    strat_row = cmp_df[
        cmp_df["strategy"] == "portfolio_rebalancing_allocator"]
    btc_row = cmp_df[cmp_df["strategy"] == "BTC_buy_and_hold"]
    basket_row = cmp_df[cmp_df["strategy"] == "equal_weight_basket"]
    if strat_row.empty or btc_row.empty:
        out = pd.DataFrame([{
            "strategy_name": "portfolio_rebalancing_allocator",
            "verdict": PORTFOLIO_INCONCLUSIVE,
            "notes": "comparison CSV missing strategy or BTC b&h row",
        }])
        if save:
            utils.write_df(out, config.RESULTS_DIR
                                / "portfolio_rebalancing_scorecard.csv")
        return out
    strat_return = float(strat_row["total_return_pct"].iloc[0])
    strat_dd = float(strat_row["max_drawdown_pct"].iloc[0])
    strat_sharpe = float(strat_row["sharpe_ratio"].iloc[0])
    btc_return = float(btc_row["total_return_pct"].iloc[0])
    btc_dd = float(btc_row["max_drawdown_pct"].iloc[0])
    btc_sharpe = float(btc_row["sharpe_ratio"].iloc[0])
    basket_return = (float(basket_row["total_return_pct"].iloc[0])
                       if not basket_row.empty else float("nan"))

    # Placebo summary lives in the first row of placebo_df.
    placebo_summary = placebo_df.iloc[[0]]
    beats_placebo_return = bool(
        placebo_summary["strategy_beats_median_return"].iloc[0])
    beats_placebo_drawdown = bool(
        placebo_summary["strategy_beats_median_drawdown"].iloc[0])
    placebo_return_pct_q = float(
        placebo_summary["placebo_return_percentile"].iloc[0])
    placebo_drawdown_pct_q = float(
        placebo_summary["placebo_drawdown_percentile"].iloc[0])

    # Walk-forward gives us the rebalance count (sum across windows).
    ok = walk_forward_df
    if "error" in ok.columns:
        ok = ok[ok["error"].isna()]
    n_windows = int(len(ok))
    total_rebalances = (int(ok["n_rebalances"].sum())
                         if "n_rebalances" in ok.columns else 0)

    # Per-criterion gates.
    pass_sharpe = (abs(strat_sharpe - btc_sharpe) <= SHARPE_GAP_LIMIT)
    pass_drawdown = ((strat_dd - btc_dd) >= DRAWDOWN_TIGHTNESS_PP)
    pass_return_placebo = beats_placebo_return
    pass_drawdown_placebo = beats_placebo_drawdown
    pass_min_rebalances = (total_rebalances >= MIN_REBALANCES_TOTAL)

    gates = [
        ("pass_sharpe_within_010", pass_sharpe),
        ("pass_drawdown_15pp_tighter", pass_drawdown),
        ("pass_beats_placebo_return", pass_return_placebo),
        ("pass_beats_placebo_drawdown", pass_drawdown_placebo),
        ("pass_min_24_rebalances", pass_min_rebalances),
    ]
    n_pass = sum(1 for _, v in gates if v)

    if n_windows < MIN_OOS_WINDOWS_FOR_CONFIDENCE:
        verdict = PORTFOLIO_INCONCLUSIVE
        notes = (f"only {n_windows} OOS windows; "
                  f"need >= {MIN_OOS_WINDOWS_FOR_CONFIDENCE}")
    elif all(v for _, v in gates):
        verdict = PORTFOLIO_PASS
        notes = "all locked rebalancing gates clear"
    elif n_pass >= len(gates) - 1:
        verdict = PORTFOLIO_WATCHLIST
        notes = "one gate missed; rest clear"
    else:
        verdict = PORTFOLIO_FAIL
        notes = ("locked rebalancing gates failed: "
                  + "; ".join(f"{k}={v}" for k, v in gates))

    out_row: Dict[str, Any] = {
        "strategy_name": "portfolio_rebalancing_allocator",
        "verdict": verdict,
        "total_return": strat_return,
        "btc_total_return": btc_return,
        "equal_weight_total_return": basket_return,
        "sharpe": strat_sharpe,
        "btc_sharpe": btc_sharpe,
        "max_drawdown": strat_dd,
        "btc_max_drawdown": btc_dd,
        "drawdown_improvement_pp": float(strat_dd - btc_dd),
        "placebo_return_percentile": placebo_return_pct_q,
        "placebo_drawdown_percentile": placebo_drawdown_pct_q,
        "rebalance_count": total_rebalances,
        "pass_sharpe_within_010": pass_sharpe,
        "pass_drawdown_15pp_tighter": pass_drawdown,
        "pass_beats_placebo_return": pass_return_placebo,
        "pass_beats_placebo_drawdown": pass_drawdown_placebo,
        "pass_min_24_rebalances": pass_min_rebalances,
        "notes": notes,
    }
    out = pd.DataFrame([out_row])
    if save:
        utils.write_df(out, config.RESULTS_DIR
                            / "portfolio_rebalancing_scorecard.csv")
    return out


# ---------------------------------------------------------------------------
# Run-all
# ---------------------------------------------------------------------------
def run_all_portfolio_rebalancing(
    universe: Sequence[str] = DEFAULT_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    in_sample_days: int = 180,
    oos_days: int = 90,
    step_days: int = 90,
    seeds: Sequence[int] = PLACEBO_SEEDS_DEFAULT,
    max_windows: int = 14,
) -> Dict[str, Any]:
    logger.info("=== portfolio rebalancing research: full pipeline ===")
    one = run_portfolio_rebalancing_backtest(
        universe=universe, timeframe=timeframe,
    )
    wf = portfolio_rebalancing_walk_forward(
        universe=universe, timeframe=timeframe,
        in_sample_days=in_sample_days, oos_days=oos_days,
        step_days=step_days, max_windows=max_windows,
    )
    plac = portfolio_rebalancing_placebo(
        universe=universe, timeframe=timeframe, seeds=seeds,
    )
    sc = portfolio_rebalancing_scorecard(
        walk_forward_df=wf, placebo_df=plac,
    )
    return {"single": one, "walk_forward": wf,
            "placebo": plac, "scorecard": sc}
