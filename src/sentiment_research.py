"""
Sentiment-overlay research orchestrator.

Single-window backtest, walk-forward, multi-seed placebo, scorecard.

Scorecard PASS criteria (10 checks, identical bar to the previous best
vol-target scorecard but adds `enough_sentiment_data_coverage` and
swaps `beats_original_allocator` for `beats_market_structure_vol_target`):

    1. Positive total return.
    2. Beats BTC OOS (>50 % windows).
    3. Beats equal-weight basket OOS.
    4. Beats simple momentum OOS.
    5. Beats market-structure vol-target OOS.
    6. Beats placebo median return.
    7. OOS stability above 60 %.
    8. At least 10 rebalances total.
    9. Max drawdown not worse than BTC's by more than 20 pp.
   10. Enough sentiment data coverage.

`stability` = profitable AND beats BTC AND beats basket AND beats simple
momentum AND beats vol-target — same OOS window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import (config, market_structure_signals as mss,
                portfolio_backtester as pb,
                sentiment_data_collector as sdc,
                sentiment_signals as ss, utils)
from .portfolio_research import (
    DEFAULT_TIMEFRAME, MIN_OOS_WINDOWS_FOR_CONFIDENCE,
    MIN_REBALANCES_FOR_CONFIDENCE, PLACEBO_SEEDS_DEFAULT,
    PORTFOLIO_FAIL, PORTFOLIO_INCONCLUSIVE, PORTFOLIO_PASS,
    PORTFOLIO_WATCHLIST, _build_oos_windows, load_universe_with_report,
)
from .strategies.market_structure_vol_target_allocator import (
    MarketStructureVolTargetAllocatorStrategy,
)
from .strategies.momentum_rotation import (
    MomentumRotationConfig, MomentumRotationStrategy,
)
from .strategies.sentiment_market_structure_allocator import (
    SentimentMarketStructureAllocatorConfig,
    SentimentMarketStructureAllocatorStrategy,
)

logger = utils.get_logger("cte.sentiment_research")

MAX_DD_VS_BTC_GAP_PP = 20.0
MIN_SENTIMENT_DAYS_FOR_RESEARCH = 4 * 365


# ---------------------------------------------------------------------------
# Random sentiment-overlay placebo: same base allocator, RANDOMLY apply
# overlay actions at the empirical frequency observed in the real
# sentiment classification.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SentimentRandomOverlayPlaceboConfig:
    seed: int = 42
    # The overlay action probabilities are derived from the actual
    # state distribution of the loaded sentiment table (so the placebo
    # fires overlays at the same empirical rate as the real strategy).
    overlay_states: Sequence[str] = (
        "extreme_fear", "fear_recovery", "neutral",
        "extreme_greed", "deteriorating", "unknown",
    )


class SentimentRandomOverlayPlacebo:
    """Same base vol-target allocator, but the SENTIMENT state at each
    rebalance is drawn at random from the empirical distribution rather
    than the actual state column. Tests whether the *real* sentiment
    information adds anything beyond a randomised overlay with the same
    activation frequency."""

    name = "sentiment_random_overlay_placebo"

    def __init__(
        self,
        market_structure_signals_df: pd.DataFrame,
        sentiment_signals_df: pd.DataFrame,
        cfg: Optional[SentimentRandomOverlayPlaceboConfig] = None,
    ) -> None:
        self.cfg = cfg or SentimentRandomOverlayPlaceboConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        # Empirical state distribution of the real sentiment series.
        if (sentiment_signals_df is None or sentiment_signals_df.empty
                or "sentiment_state" not in sentiment_signals_df.columns):
            self._states = list(self.cfg.overlay_states)
            self._probs = None
        else:
            counts = sentiment_signals_df["sentiment_state"].value_counts()
            self._states = list(counts.index)
            self._probs = counts.to_numpy(dtype=float)
            self._probs = self._probs / self._probs.sum()
        # Pre-build the wrapper allocator we'll mutate per call.
        self._wrapper = SentimentMarketStructureAllocatorStrategy(
            market_structure_signals_df,
            sentiment_signals_df=pd.DataFrame(),  # we'll inject per-call
        )
        self._market_signals_df = market_structure_signals_df

    def reset(self, seed: Optional[int] = None) -> None:
        s = self.cfg.seed if seed is None else int(seed)
        self._rng = np.random.default_rng(s)

    def target_weights(
        self,
        asof_ts_ms: int,
        asset_frames: Dict[str, pd.DataFrame],
        timeframe: str = "1d",
    ) -> Dict[str, float]:
        # Draw a synthetic sentiment state with the empirical distribution.
        state = (str(self._rng.choice(self._states, p=self._probs))
                  if self._probs is not None
                  else str(self._rng.choice(self._states)))
        synthetic_sent = pd.DataFrame([{
            "timestamp": int(asof_ts_ms),
            "date": pd.to_datetime(asof_ts_ms, unit="ms",
                                     utc=True).strftime("%Y-%m-%d"),
            "fear_greed_value": 50,
            "fear_greed_classification": "Synthetic",
            "fg_7d_change": 0.0, "fg_30d_change": 0.0,
            "fg_7d_mean": 50.0, "fg_30d_mean": 50.0, "fg_90d_zscore": 0.0,
            "extreme_fear": False, "fear": False, "neutral": True,
            "greed": False, "extreme_greed": False,
            "sentiment_recovering": False, "sentiment_deteriorating": False,
            "sentiment_state": state,
        }])
        wrapper = SentimentMarketStructureAllocatorStrategy(
            self._market_signals_df, synthetic_sent,
        )
        return wrapper.target_weights(asof_ts_ms, asset_frames,
                                        timeframe=timeframe)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_market_structure_signals(
    signals_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if signals_df is not None and not signals_df.empty:
        return signals_df
    p = config.RESULTS_DIR / "market_structure_signals.csv"
    if p.exists() and p.stat().st_size > 0:
        return pd.read_csv(p)
    logger.info("market_structure_signals.csv missing — computing on the fly")
    return mss.compute_market_structure_signals(save=True)


def _ensure_sentiment_signals(
    signals_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if signals_df is not None and not signals_df.empty:
        return signals_df
    p = config.RESULTS_DIR / "sentiment_signals.csv"
    if p.exists() and p.stat().st_size > 0:
        return pd.read_csv(p)
    logger.info("sentiment_signals.csv missing — computing on the fly")
    return ss.compute_sentiment_signals(save=True)


def _enough_sentiment_coverage() -> Optional[str]:
    p = config.RESULTS_DIR / "sentiment_data_coverage.csv"
    if not p.exists() or p.stat().st_size == 0:
        return ("no sentiment coverage CSV — run "
                "`python main.py download_sentiment_data` first")
    cov = pd.read_csv(p)
    if cov.empty:
        return "sentiment coverage CSV empty"
    if not bool(cov["enough_for_research"].iloc[0]):
        return (f"insufficient sentiment coverage: "
                  f"{int(cov['row_count'].iloc[0])} rows, "
                  f"{float(cov['coverage_days'].iloc[0])}d "
                  f"(need ≥ {MIN_SENTIMENT_DAYS_FOR_RESEARCH}d)")
    return None


# ---------------------------------------------------------------------------
# Single-window backtest
# ---------------------------------------------------------------------------
def run_sentiment_allocator(
    universe: Sequence[str] = mss.ALLOCATOR_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    overlay_cfg: Optional[SentimentMarketStructureAllocatorConfig] = None,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    market_signals_df: Optional[pd.DataFrame] = None,
    sentiment_signals_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> Dict[str, Any]:
    utils.assert_paper_only()
    frames, _ = load_universe_with_report(assets=universe, timeframe=timeframe,
                                            save=save)
    if not frames:
        return {"ok": False, "reason": "no asset data"}
    ms = _ensure_market_structure_signals(market_signals_df)
    sent = _ensure_sentiment_signals(sentiment_signals_df)
    cfg = backtest_cfg or pb.PortfolioBacktestConfig()

    strat = SentimentMarketStructureAllocatorStrategy(ms, sent, overlay_cfg)
    art = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames,
        timeframe=timeframe, cfg=cfg, save=save,
        save_prefix="sentiment_allocator",
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
    vt = MarketStructureVolTargetAllocatorStrategy(ms)
    vt_art = pb.run_portfolio_backtest(
        portfolio_strategy=vt, asset_frames=frames,
        timeframe=timeframe, cfg=cfg, save=False,
    )
    bench["market_structure_vol_target"] = vt_art.equity_curve

    metrics = pb.portfolio_metrics(art.equity_curve, cfg.starting_capital)
    bench_metrics = {n: pb.portfolio_metrics(d, cfg.starting_capital)
                      for n, d in bench.items()}
    rows = [{"strategy": "sentiment_market_structure_allocator", **metrics}]
    for name, m in bench_metrics.items():
        rows.append({"strategy": name, **m})
    cmp_df = pd.DataFrame(rows)
    if save:
        utils.write_df(cmp_df,
                        config.RESULTS_DIR / "sentiment_allocator_comparison.csv")
    return {"ok": True, "artifacts": art, "benchmarks": bench,
            "metrics": metrics, "bench_metrics": bench_metrics,
            "comparison_df": cmp_df}


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def sentiment_walk_forward(
    universe: Sequence[str] = mss.ALLOCATOR_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    in_sample_days: int = 180,
    oos_days: int = 90,
    step_days: int = 90,
    overlay_cfg: Optional[SentimentMarketStructureAllocatorConfig] = None,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    market_signals_df: Optional[pd.DataFrame] = None,
    sentiment_signals_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> pd.DataFrame:
    utils.assert_paper_only()
    cfg = backtest_cfg or pb.PortfolioBacktestConfig()
    frames, _ = load_universe_with_report(assets=universe, timeframe=timeframe,
                                            save=False)
    if not frames:
        out = pd.DataFrame()
        if save:
            utils.write_df(out, config.RESULTS_DIR / "sentiment_walk_forward.csv")
        return out
    ms = _ensure_market_structure_signals(market_signals_df)
    sent = _ensure_sentiment_signals(sentiment_signals_df)

    ts_axis = pb._aligned_timestamps(frames)
    if len(ts_axis) < 2:
        out = pd.DataFrame()
        if save:
            utils.write_df(out, config.RESULTS_DIR / "sentiment_walk_forward.csv")
        return out
    first_ts, last_ts = ts_axis[0], ts_axis[-1]
    if not ms.empty and "timestamp" in ms.columns:
        first_ts = max(first_ts, int(pd.to_numeric(ms["timestamp"]).min()))
    if not sent.empty and "timestamp" in sent.columns:
        first_ts = max(first_ts, int(pd.to_numeric(sent["timestamp"]).min()))
    windows = _build_oos_windows(first_ts, last_ts,
                                  in_sample_days, oos_days, step_days)

    sent_strat = SentimentMarketStructureAllocatorStrategy(
        ms, sent, overlay_cfg,
    )
    vt_strat = MarketStructureVolTargetAllocatorStrategy(ms)
    simple_strat = MomentumRotationStrategy(MomentumRotationConfig())
    rows: List[Dict] = []
    for w_idx, w in enumerate(windows, start=1):
        logger.info(
            "sentiment_walk_forward [%d/%d] OOS %s → %s",
            w_idx, len(windows),
            pd.to_datetime(w["oos_start_ms"], unit="ms", utc=True).date(),
            pd.to_datetime(w["oos_end_ms"], unit="ms", utc=True).date(),
        )
        art = pb.run_portfolio_backtest(
            portfolio_strategy=sent_strat, asset_frames=frames,
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
        vt_art = pb.run_portfolio_backtest(
            portfolio_strategy=vt_strat, asset_frames=frames,
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
        vt_ret = pb.portfolio_metrics(
            vt_art.equity_curve, cfg.starting_capital,
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
            "vol_target_oos_return_pct": vt_ret,
            "beats_btc": bool(m["total_return_pct"] > btc_ret),
            "beats_basket": bool(m["total_return_pct"] > bk_ret),
            "beats_simple_momentum": bool(m["total_return_pct"] > simple_ret),
            "beats_vol_target": bool(m["total_return_pct"] > vt_ret),
            "profitable": bool(m["total_return_pct"] > 0),
            "n_rebalances": n_rebalances,
            "n_trades": int(len(art.trades)),
            "avg_holdings": avg_holdings,
            "turnover": turnover,
            "error": None,
        })
    out = pd.DataFrame(rows)
    if save:
        utils.write_df(out, config.RESULTS_DIR / "sentiment_walk_forward.csv")
    return out


# ---------------------------------------------------------------------------
# Placebo
# ---------------------------------------------------------------------------
def sentiment_placebo(
    universe: Sequence[str] = mss.ALLOCATOR_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    seeds: Sequence[int] = PLACEBO_SEEDS_DEFAULT,
    overlay_cfg: Optional[SentimentMarketStructureAllocatorConfig] = None,
    backtest_cfg: Optional[pb.PortfolioBacktestConfig] = None,
    market_signals_df: Optional[pd.DataFrame] = None,
    sentiment_signals_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> pd.DataFrame:
    utils.assert_paper_only()
    cfg = backtest_cfg or pb.PortfolioBacktestConfig()
    frames, _ = load_universe_with_report(assets=universe, timeframe=timeframe,
                                            save=False)
    if not frames:
        out = pd.DataFrame()
        if save:
            utils.write_df(out, config.RESULTS_DIR / "sentiment_placebo.csv")
        return out
    ms = _ensure_market_structure_signals(market_signals_df)
    sent = _ensure_sentiment_signals(sentiment_signals_df)

    strat = SentimentMarketStructureAllocatorStrategy(ms, sent, overlay_cfg)
    strat_art = pb.run_portfolio_backtest(
        portfolio_strategy=strat, asset_frames=frames, timeframe=timeframe,
        cfg=cfg, save=False,
    )
    strat_m = pb.portfolio_metrics(strat_art.equity_curve, cfg.starting_capital)

    rows: List[Dict] = []
    for seed in seeds:
        plac = SentimentRandomOverlayPlacebo(
            ms, sent,
            SentimentRandomOverlayPlaceboConfig(seed=int(seed)),
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
        "strategy": "sentiment_market_structure_allocator",
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
        utils.write_df(out, config.RESULTS_DIR / "sentiment_placebo.csv")
    return out


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
def sentiment_scorecard(
    walk_forward_df: Optional[pd.DataFrame] = None,
    placebo_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> pd.DataFrame:
    if walk_forward_df is None:
        p = config.RESULTS_DIR / "sentiment_walk_forward.csv"
        walk_forward_df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    if placebo_df is None:
        p = config.RESULTS_DIR / "sentiment_placebo.csv"
        placebo_df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    cmp_p = config.RESULTS_DIR / "sentiment_allocator_comparison.csv"
    cmp_df = pd.read_csv(cmp_p) if cmp_p.exists() else pd.DataFrame()

    coverage_issue = _enough_sentiment_coverage()

    if walk_forward_df.empty:
        out = pd.DataFrame([{
            "strategy_name": "sentiment_market_structure_allocator",
            "verdict": PORTFOLIO_INCONCLUSIVE,
            "reason": "no walk-forward data",
            "coverage_note": coverage_issue or "ok",
        }])
        if save:
            utils.write_df(out, config.RESULTS_DIR / "sentiment_scorecard.csv")
        return out

    ok = walk_forward_df
    if "error" in ok.columns:
        ok = ok[ok["error"].isna()]
    n_windows = int(len(ok))
    if n_windows < MIN_OOS_WINDOWS_FOR_CONFIDENCE:
        out = pd.DataFrame([{
            "strategy_name": "sentiment_market_structure_allocator",
            "n_windows": n_windows,
            "verdict": PORTFOLIO_INCONCLUSIVE,
            "reason": f"only {n_windows} OOS windows — need ≥"
                       f"{MIN_OOS_WINDOWS_FOR_CONFIDENCE}",
            "coverage_note": coverage_issue or "ok",
        }])
        if save:
            utils.write_df(out, config.RESULTS_DIR / "sentiment_scorecard.csv")
        return out

    avg_oos_return = float(ok["oos_return_pct"].mean())
    avg_oos_dd = float(ok["oos_max_drawdown_pct"].mean())
    pct_profitable = float((ok["oos_return_pct"] > 0).mean() * 100.0)
    pct_beats_btc = float(ok["beats_btc"].mean() * 100.0)
    pct_beats_basket = float(ok["beats_basket"].mean() * 100.0)
    pct_beats_simple = (float(ok["beats_simple_momentum"].mean() * 100.0)
                         if "beats_simple_momentum" in ok.columns else 0.0)
    pct_beats_vt = (float(ok["beats_vol_target"].mean() * 100.0)
                     if "beats_vol_target" in ok.columns else 0.0)
    stab_mask = (ok["profitable"] & ok["beats_btc"] & ok["beats_basket"])
    if "beats_simple_momentum" in ok.columns:
        stab_mask &= ok["beats_simple_momentum"]
    if "beats_vol_target" in ok.columns:
        stab_mask &= ok["beats_vol_target"]
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
        ("beats_btc_oos", pct_beats_btc > 50.0),
        ("beats_basket_oos", pct_beats_basket > 50.0),
        ("beats_simple_momentum_oos", pct_beats_simple > 50.0),
        ("beats_market_structure_vol_target_oos", pct_beats_vt > 50.0),
        ("beats_placebo_median", beats_placebo_median),
        ("oos_stability_above_60", stability > 60.0),
        ("at_least_10_rebalances",
         total_rebalances >= MIN_REBALANCES_FOR_CONFIDENCE),
        ("dd_within_btc_gap_20pp", bool(dd_gap_ok)),
        ("enough_sentiment_data_coverage", coverage_issue is None),
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
        "strategy_name": "sentiment_market_structure_allocator",
        "n_windows": n_windows,
        "avg_oos_return_pct": avg_oos_return,
        "avg_oos_drawdown_pct": avg_oos_dd,
        "pct_windows_profitable": pct_profitable,
        "pct_windows_beat_btc": pct_beats_btc,
        "pct_windows_beat_basket": pct_beats_basket,
        "pct_windows_beat_simple_momentum": pct_beats_simple,
        "pct_windows_beat_vol_target": pct_beats_vt,
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
        utils.write_df(out, config.RESULTS_DIR / "sentiment_scorecard.csv")
    return out


def run_all_sentiment(
    universe: Sequence[str] = mss.ALLOCATOR_UNIVERSE,
    timeframe: str = DEFAULT_TIMEFRAME,
    in_sample_days: int = 180,
    oos_days: int = 90,
    step_days: int = 90,
    seeds: Sequence[int] = PLACEBO_SEEDS_DEFAULT,
) -> Dict[str, Any]:
    logger.info("=== sentiment overlay research: full pipeline ===")
    sdc.download_sentiment_data(refresh=False)
    ms = mss.compute_market_structure_signals(save=True)
    sent = ss.compute_sentiment_signals(save=True)
    one = run_sentiment_allocator(
        universe=universe, timeframe=timeframe,
        market_signals_df=ms, sentiment_signals_df=sent,
    )
    wf = sentiment_walk_forward(
        universe=universe, timeframe=timeframe,
        in_sample_days=in_sample_days, oos_days=oos_days, step_days=step_days,
        market_signals_df=ms, sentiment_signals_df=sent,
    )
    plac = sentiment_placebo(
        universe=universe, timeframe=timeframe, seeds=seeds,
        market_signals_df=ms, sentiment_signals_df=sent,
    )
    sc = sentiment_scorecard(walk_forward_df=wf, placebo_df=plac)
    return {"signals": sent, "single": one, "walk_forward": wf,
            "placebo": plac, "scorecard": sc}
