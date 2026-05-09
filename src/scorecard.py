"""
Strategy scorecard.

Aggregates the per-strategy / per-asset / per-timeframe results from the
Research Lab CSVs (strategy_comparison + walk_forward + robustness) into
a single conservative ranking with a PASS / WATCHLIST / FAIL /
INCONCLUSIVE verdict per row.

Scoring is intentionally simple, transparent, and biased *against* false
positives — one good return cannot earn a PASS. The goal is to identify
strategies worth deeper paper-trading research, not to crown a winner.

Score components (all optional — missing inputs degrade to 0 with a note):

  benchmark_score   from strategy_vs_bh_pct (per row)
  drawdown_score    from max_drawdown_pct + total_return_pct + exposure
  trade_count_score from num_trades
  walk_forward_score    from walk_forward_results.csv aggregated to (asset, tf)
  robustness_score      from robustness_results.csv aggregated to (family)

Total = sum of components.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from . import config, utils
from .strategies import BENCHMARKS, PLACEBOS

logger = utils.get_logger("cte.scorecard")


MIN_TRADES_FOR_CONFIDENCE = 10
MIN_OOS_WINDOWS_FOR_CONFIDENCE = 5
BENCHMARK_VERDICT = "BENCHMARK"
PLACEBO_VERDICT = "PLACEBO"


def _is_benchmark(strategy_name: str) -> bool:
    """A strategy is a benchmark if its base name (before any wrapper
    suffix like '+regime') is in `BENCHMARKS`."""
    if not strategy_name:
        return False
    return strategy_name.split("+")[0] in BENCHMARKS


def _is_placebo(strategy_name: str) -> bool:
    """Statistical control strategies (random, null) — never tradable."""
    if not strategy_name:
        return False
    return strategy_name.split("+")[0] in PLACEBOS


# ---------------------------------------------------------------------------
# Per-component scorers
# ---------------------------------------------------------------------------
def _benchmark_score(strategy_vs_bh_pct: float) -> int:
    if pd.isna(strategy_vs_bh_pct):
        return 0
    if strategy_vs_bh_pct >= 1.0:
        return 2
    if strategy_vs_bh_pct <= -1.0:
        return -2
    return 0


def _drawdown_score(max_dd_pct: float, total_return_pct: float,
                    bh_max_dd_pct: Optional[float],
                    exposure_time_pct: float) -> int:
    """
    Conservative drawdown scoring with explicit B&H comparison when
    available.

    +1 only when ALL of:
       * we have a real B&H drawdown to compare against
       * strategy DD is smaller (closer to 0) than B&H DD
       * strategy return is positive
       * exposure_time_pct >= 20% (cash sitting on the sidelines must
         NOT earn drawdown credit just by being inactive)

     0 (neutral) when low DD coincides with low exposure — "looking safe
       because you didn't trade" is not an edge.

    -1 when DD is large AND return is non-positive AND exposure was real.

    Without a B&H DD comparator we fall back to the same conservative
    rules but never award +1 (we cannot prove the strategy actually
    managed risk better than holding).
    """
    if pd.isna(max_dd_pct) or pd.isna(total_return_pct):
        return 0
    bh_dd = (bh_max_dd_pct
             if bh_max_dd_pct is not None and not pd.isna(bh_max_dd_pct)
             and bh_max_dd_pct != 0.0
             else None)
    if bh_dd is not None:
        smaller_dd = max_dd_pct > bh_dd  # both negative; "smaller" = closer to 0
        if smaller_dd and total_return_pct > 0 and exposure_time_pct >= 20:
            return 1
        if max_dd_pct < bh_dd and total_return_pct <= 0 and exposure_time_pct >= 20:
            return -1
        return 0
    # Fallback (no benchmark DD) — never positive.
    if max_dd_pct < -15 and total_return_pct <= 0 and exposure_time_pct >= 20:
        return -1
    return 0


def _trade_count_score(num_trades: int) -> int:
    if num_trades >= MIN_TRADES_FOR_CONFIDENCE:
        return 1
    if num_trades >= 5:
        return 0
    return -1


def _walk_forward_score(
    wf_df: pd.DataFrame,
    asset: str,
    timeframe: str,
    strategy_name: Optional[str] = None,
) -> tuple:
    """Returns (score, note).

    Prefers strategy-specific WF rows when `strategy_name` is given AND a
    `strategy_name` column is present (i.e. caller passed
    `walk_forward_by_strategy.csv`). When no matching rows exist, returns
    (0, note) — never re-uses another strategy's WF as a proxy.
    """
    if wf_df is None or wf_df.empty:
        return 0, "no walk-forward data"
    df = wf_df
    if "error" in df.columns:
        df = df[df["error"].isna()]
    if df.empty:
        return 0, "all walk-forward rows errored"

    has_strategy_col = "strategy_name" in df.columns
    if has_strategy_col and strategy_name:
        sub = df[(df["asset"] == asset) & (df["timeframe"] == timeframe)
                 & (df["strategy_name"] == strategy_name)]
        if sub.empty:
            return 0, f"no per-strategy WF rows for {strategy_name}"
    elif has_strategy_col:
        return 0, "per-strategy WF available but no strategy_name supplied"
    else:
        # Legacy frame (incumbent only) — only score the incumbent.
        if strategy_name and strategy_name.split("+")[0] != "rsi_ma_atr":
            return 0, "no per-strategy WF data (legacy WF is rsi_ma_atr only)"
        sub = df[(df["asset"] == asset) & (df["timeframe"] == timeframe)]
        if sub.empty:
            return 0, "no walk-forward windows for this (asset, timeframe)"

    n = len(sub)
    wins = ((sub["strategy_return_pct"] > 0)
            & (sub["strategy_vs_bh_pct"] > 0)).sum()
    if n < MIN_OOS_WINDOWS_FOR_CONFIDENCE:
        return 0, f"only {n} OOS windows — too few to score"
    pct = wins / n * 100.0
    if pct > 70:
        return 3, f"OOS stability {pct:.0f}% (>{70}% PASS bar)"
    if pct >= 50:
        return 1, f"OOS stability {pct:.0f}% (50-70%)"
    return -2, f"OOS stability {pct:.0f}% (<50%)"


def _robustness_score(rb_df: pd.DataFrame, strategy_name: str,
                      asset: str, timeframe: str) -> tuple:
    """Per-family robustness score. Family = the first segment of the
    strategy name (e.g. 'rsi_ma_atr'). Strategies with no family in the
    robustness grid get 0 + a note."""
    if rb_df is None or rb_df.empty:
        return 0, "no robustness data"
    if "error" in rb_df.columns:
        rb_df = rb_df[rb_df["error"].isna()]
    if rb_df.empty:
        return 0, "no completed robustness rows"
    family = strategy_name.split("+")[0]  # strip wrapper suffixes
    sub = rb_df[(rb_df["family"] == family)
                & (rb_df["asset"] == asset)
                & (rb_df["timeframe"] == timeframe)]
    if sub.empty:
        return 0, f"no robustness rows for family {family!r}"
    n = len(sub)
    beat = (sub["strategy_vs_bh_pct"] > 0).sum()
    pct = beat / n * 100.0
    if pct > 60:
        return 2, f"{beat}/{n} variants beat B&H ({pct:.0f}%)"
    if pct >= 40:
        return 0, f"{beat}/{n} variants beat B&H ({pct:.0f}%)"
    return -2, f"{beat}/{n} variants beat B&H ({pct:.0f}% — fragile)"


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def _verdict(total: int, num_trades: int, wf_score: int,
             is_benchmark: bool = False, is_placebo: bool = False) -> str:
    # Hard short-circuits — placebos and benchmarks are never tradable.
    if is_placebo:
        return PLACEBO_VERDICT
    if is_benchmark:
        return BENCHMARK_VERDICT
    if num_trades < 5:
        return "INCONCLUSIVE"
    if total >= 5 and wf_score > 0:
        return "PASS"
    if total >= 2:
        return "WATCHLIST"
    return "FAIL"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_scorecard(
    strategy_comparison_df: pd.DataFrame,
    walk_forward_df: Optional[pd.DataFrame] = None,
    robustness_df: Optional[pd.DataFrame] = None,
    save: bool = True,
    save_path=None,
) -> pd.DataFrame:
    """Build the scorecard from the saved Research Lab CSVs.

    `strategy_comparison_df` is the source of truth for which (strategy,
    asset, timeframe) rows to score. Walk-forward and robustness data are
    used as auxiliary signals where available.
    """
    if strategy_comparison_df is None or strategy_comparison_df.empty:
        out = pd.DataFrame()
        if save:
            utils.write_df(out, save_path or (config.RESULTS_DIR / "strategy_scorecard.csv"))
        return out

    rows: List[Dict] = []
    for _, r in strategy_comparison_df.iterrows():
        strat = str(r.get("strategy"))
        is_bench = _is_benchmark(strat)
        is_plac = _is_placebo(strat)
        if "error" in r.index and pd.notna(r.get("error")):
            rows.append({
                "strategy_name": strat, "asset": r.get("asset"),
                "timeframe": r.get("timeframe"),
                "is_benchmark": is_bench, "is_placebo": is_plac,
                "total_score": 0,
                "verdict": (PLACEBO_VERDICT if is_plac else
                            (BENCHMARK_VERDICT if is_bench else "INCONCLUSIVE")),
                "return_score": 0, "benchmark_score": 0,
                "drawdown_score": 0, "robustness_score": 0,
                "walk_forward_score": 0, "trade_count_score": 0,
                "notes": f"backtest failed: {r['error']}",
            })
            continue

        asset = str(r.get("asset"))
        tf = str(r.get("timeframe"))
        vs_bh = float(r.get("strategy_vs_bh_pct", 0.0))
        max_dd = float(r.get("max_drawdown_pct", 0.0))
        bh_dd_raw = r.get("buy_and_hold_max_drawdown_pct")
        bh_dd = (float(bh_dd_raw)
                 if bh_dd_raw is not None and not pd.isna(bh_dd_raw)
                 and float(bh_dd_raw) != 0.0
                 else None)
        exposure = float(r.get("exposure_time_pct", 0.0))
        n_trades = int(r.get("num_trades", 0))
        total_return = float(r.get("total_return_pct", 0.0))

        bench = _benchmark_score(vs_bh)
        dd = _drawdown_score(max_dd, total_return, bh_dd, exposure)
        tcount = _trade_count_score(n_trades)
        wf, wf_note = _walk_forward_score(walk_forward_df, asset, tf,
                                           strategy_name=strat)
        rb, rb_note = _robustness_score(robustness_df, strat, asset, tf)

        # `return_score` is a small extra pull from raw return — capped so
        # a single lucky window cannot dominate.
        if total_return >= 5:
            return_score = 1
        elif total_return <= -5:
            return_score = -1
        else:
            return_score = 0

        total = bench + dd + tcount + wf + rb + return_score
        verdict = _verdict(total, n_trades, wf,
                           is_benchmark=is_bench, is_placebo=is_plac)

        notes_parts: List[str] = []
        if n_trades < MIN_TRADES_FOR_CONFIDENCE:
            notes_parts.append(f"only {n_trades} closed trades")
        if exposure < 10:
            notes_parts.append(f"exposure {exposure:.1f}%")
        notes_parts.append(f"WF: {wf_note}")
        notes_parts.append(f"robustness: {rb_note}")
        notes = "; ".join(notes_parts)

        rows.append({
            "strategy_name": strat, "asset": asset, "timeframe": tf,
            "is_benchmark": is_bench, "is_placebo": is_plac,
            "total_score": int(total), "verdict": verdict,
            "return_score": int(return_score),
            "benchmark_score": int(bench),
            "drawdown_score": int(dd),
            "robustness_score": int(rb),
            "walk_forward_score": int(wf),
            "trade_count_score": int(tcount),
            "notes": notes,
        })

    out = pd.DataFrame(rows).sort_values(
        ["total_score", "strategy_name"], ascending=[False, True],
    )
    if save:
        utils.write_df(
            out, save_path or (config.RESULTS_DIR / "strategy_scorecard.csv"),
        )
    return out


def best_picks(scorecard_df: pd.DataFrame) -> Dict[str, object]:
    """Summary used by the dashboard + research_summary.

    Three disjoint groups are reported:
      * benchmarks (buy_and_hold) — reference only.
      * placebos (placebo_random) — control only; also excluded from
        PASS / WATCHLIST counts.
      * tradable — everything else; the only group eligible for PASS.
    """
    empty = {
        "any_pass": False, "n_pass": 0, "n_watchlist": 0, "n_fail": 0,
        "n_inconclusive": 0,
        "best_tradable": None, "passes": [], "watchlist": [],
        "benchmarks": [], "best_benchmark": None,
        "placebos": [], "best_placebo": None,
        "any_tradable_beats_benchmark": False,
    }
    if scorecard_df is None or scorecard_df.empty:
        return empty

    if "is_benchmark" in scorecard_df.columns:
        bench_mask = scorecard_df["is_benchmark"].astype(bool)
    else:
        bench_mask = pd.Series(False, index=scorecard_df.index)
    if "is_placebo" in scorecard_df.columns:
        plac_mask = scorecard_df["is_placebo"].astype(bool)
    else:
        plac_mask = pd.Series(False, index=scorecard_df.index)
    tradable = scorecard_df[~(bench_mask | plac_mask)]
    benchmarks = scorecard_df[bench_mask]
    placebos = scorecard_df[plac_mask]

    pass_rows = tradable[tradable["verdict"] == "PASS"]
    watch_rows = tradable[tradable["verdict"] == "WATCHLIST"]
    fail_rows = tradable[tradable["verdict"] == "FAIL"]
    inconclusive_rows = tradable[tradable["verdict"] == "INCONCLUSIVE"]

    best_tradable = tradable.iloc[0].to_dict() if not tradable.empty else None
    best_benchmark = benchmarks.iloc[0].to_dict() if not benchmarks.empty else None
    best_placebo = placebos.iloc[0].to_dict() if not placebos.empty else None

    any_beats = bool((tradable.get("benchmark_score", 0) > 0).any()) \
        if not tradable.empty else False

    cols_lite = ["strategy_name", "asset", "timeframe", "total_score"]
    cols_lite = [c for c in cols_lite if c in scorecard_df.columns]
    return {
        "any_pass": not pass_rows.empty,
        "n_pass": int(len(pass_rows)),
        "n_watchlist": int(len(watch_rows)),
        "n_fail": int(len(fail_rows)),
        "n_inconclusive": int(len(inconclusive_rows)),
        "best_tradable": best_tradable,
        "best_benchmark": best_benchmark,
        "best_placebo": best_placebo,
        "any_tradable_beats_benchmark": any_beats,
        "passes": pass_rows[cols_lite].to_dict("records"),
        "watchlist": watch_rows[cols_lite].to_dict("records"),
        "benchmarks": benchmarks[cols_lite].to_dict("records") if not benchmarks.empty else [],
        "placebos": placebos[cols_lite].to_dict("records") if not placebos.empty else [],
    }
