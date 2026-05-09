"""
Out-of-sample methodology audit.

Reads `results/walk_forward_by_strategy.csv` and verifies the basic
mechanics of the walk-forward pipeline:

  * window counts per strategy / asset / timeframe
  * window start + end timestamps
  * pairwise overlap detection (within a single asset+timeframe stream)
  * trade count + exposure per window
  * profitable / beat-B&H counts and stability score per group

Saves two CSVs:

  * `results/oos_audit.csv`         — one row per OOS window with audit flags.
  * `results/oos_audit_summary.csv` — aggregates per (strategy, asset, tf).

The audit never re-runs backtests. It is purely an inspection of saved
research artifacts so we can answer "is the OOS evaluation itself
broken?" before iterating on strategies again.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd

from . import config, utils


# Conservative thresholds the audit uses to flag thin windows.
MIN_TRADES_PER_WINDOW_FOR_CONFIDENCE = 3
MIN_OOS_WINDOWS_FOR_CONFIDENCE = 5


def _load_wf_csv(path: Optional[str] = None) -> pd.DataFrame:
    p = path or (config.RESULTS_DIR / "walk_forward_by_strategy.csv")
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(p)


def _detect_overlap(group: pd.DataFrame) -> bool:
    """Return True if any two OOS windows in this (strategy, asset, tf)
    group have overlapping time spans. Windows ordered by start time;
    a window starting before the previous one ended is an overlap."""
    if "oos_start_iso" not in group.columns or "oos_end_iso" not in group.columns:
        return False
    sub = group.dropna(subset=["oos_start_iso", "oos_end_iso"]).copy()
    if sub.empty:
        return False
    sub["_s"] = pd.to_datetime(sub["oos_start_iso"], utc=True, errors="coerce")
    sub["_e"] = pd.to_datetime(sub["oos_end_iso"], utc=True, errors="coerce")
    sub = sub.dropna(subset=["_s", "_e"]).sort_values("_s")
    prev_end = None
    for _, row in sub.iterrows():
        if prev_end is not None and row["_s"] < prev_end:
            return True
        prev_end = row["_e"]
    return False


def audit_walk_forward(
    wf_df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the audit and summary frames.

    `wf_df` is the contents of `walk_forward_by_strategy.csv`. If
    omitted, we load it from disk. Returns (per_window_df, summary_df).
    """
    if wf_df is None:
        wf_df = _load_wf_csv()
    if wf_df is None or wf_df.empty:
        empty_audit = pd.DataFrame()
        empty_summary = pd.DataFrame()
        if save:
            utils.write_df(empty_audit, config.RESULTS_DIR / "oos_audit.csv")
            utils.write_df(empty_summary, config.RESULTS_DIR / "oos_audit_summary.csv")
        return empty_audit, empty_summary

    # Per-window audit rows (preserve every input row, add flags).
    audit_rows = []
    has_strategy_col = "strategy_name" in wf_df.columns
    if not has_strategy_col:
        # legacy WF CSV — treat the whole thing as one strategy.
        wf_df = wf_df.copy()
        wf_df["strategy_name"] = "rsi_ma_atr"

    for _, r in wf_df.iterrows():
        n_trades = int(r["num_trades"]) if pd.notna(r.get("num_trades")) else 0
        ret = float(r["strategy_return_pct"]) if pd.notna(r.get("strategy_return_pct")) else float("nan")
        vs_bh = float(r["strategy_vs_bh_pct"]) if pd.notna(r.get("strategy_vs_bh_pct")) else float("nan")
        audit_rows.append({
            "strategy_name": r.get("strategy_name"),
            "asset": r.get("asset"),
            "timeframe": r.get("timeframe"),
            "window": r.get("window"),
            "oos_start_iso": r.get("oos_start_iso"),
            "oos_end_iso": r.get("oos_end_iso"),
            "num_trades": n_trades,
            "exposure_time_pct": (
                float(r.get("exposure_time_pct"))
                if pd.notna(r.get("exposure_time_pct")) else 0.0
            ),
            "strategy_return_pct": ret,
            "buy_and_hold_return_pct": (
                float(r.get("buy_and_hold_return_pct"))
                if pd.notna(r.get("buy_and_hold_return_pct")) else float("nan")
            ),
            "strategy_vs_bh_pct": vs_bh,
            "is_profitable": bool(ret > 0) if pd.notna(ret) else False,
            "beats_buy_and_hold": bool(vs_bh > 0) if pd.notna(vs_bh) else False,
            "profitable_and_beats_bh": (
                bool(ret > 0 and vs_bh > 0)
                if pd.notna(ret) and pd.notna(vs_bh) else False
            ),
            "low_trade_count": bool(n_trades < MIN_TRADES_PER_WINDOW_FOR_CONFIDENCE),
            "errored": bool(pd.notna(r.get("error"))) if "error" in r else False,
        })
    audit_df = pd.DataFrame(audit_rows)

    # Summary per (strategy, asset, timeframe).
    summary_rows = []
    group_cols = ["strategy_name", "asset", "timeframe"]
    for keys, group in wf_df.groupby(group_cols, dropna=False):
        strat, asset, tf = keys
        valid = group
        if "error" in group.columns:
            valid = group[group["error"].isna()]
        n_windows = int(len(valid))
        n_total = int(len(group))
        if n_windows == 0:
            summary_rows.append({
                "strategy_name": strat, "asset": asset, "timeframe": tf,
                "n_windows": 0, "n_total_rows": n_total,
                "windows_overlap": False, "earliest_oos_start": None,
                "latest_oos_end": None, "mean_trades_per_window": 0.0,
                "median_trades_per_window": 0.0,
                "mean_exposure_pct": 0.0,
                "n_profitable": 0, "n_beats_bh": 0,
                "n_profitable_and_beats_bh": 0,
                "stability_score_pct": 0.0,
                "enough_windows_for_confidence": False,
                "notes": "no valid OOS windows",
            })
            continue
        starts = pd.to_datetime(valid["oos_start_iso"], utc=True, errors="coerce")
        ends = pd.to_datetime(valid["oos_end_iso"], utc=True, errors="coerce")
        n_prof = int((valid["strategy_return_pct"] > 0).sum())
        n_beat = int((valid["strategy_vs_bh_pct"] > 0).sum())
        n_both = int(((valid["strategy_return_pct"] > 0)
                      & (valid["strategy_vs_bh_pct"] > 0)).sum())
        stability = (n_both / n_windows * 100.0) if n_windows else 0.0
        notes = []
        if n_windows < MIN_OOS_WINDOWS_FOR_CONFIDENCE:
            notes.append(
                f"only {n_windows} valid windows — need ≥"
                f"{MIN_OOS_WINDOWS_FOR_CONFIDENCE} for confidence"
            )
        if _detect_overlap(valid):
            notes.append("WARNING: OOS windows overlap")
        thin_windows = (
            valid["num_trades"] < MIN_TRADES_PER_WINDOW_FOR_CONFIDENCE
        ).sum()
        if thin_windows > 0:
            notes.append(f"{int(thin_windows)} window(s) with <"
                         f"{MIN_TRADES_PER_WINDOW_FOR_CONFIDENCE} trades")
        summary_rows.append({
            "strategy_name": strat, "asset": asset, "timeframe": tf,
            "n_windows": n_windows, "n_total_rows": n_total,
            "windows_overlap": _detect_overlap(valid),
            "earliest_oos_start": (str(starts.min()) if not starts.empty else None),
            "latest_oos_end": (str(ends.max()) if not ends.empty else None),
            "mean_trades_per_window": float(valid["num_trades"].mean()),
            "median_trades_per_window": float(valid["num_trades"].median()),
            "mean_exposure_pct": float(valid["exposure_time_pct"].mean())
                if "exposure_time_pct" in valid.columns else 0.0,
            "n_profitable": n_prof, "n_beats_bh": n_beat,
            "n_profitable_and_beats_bh": n_both,
            "stability_score_pct": stability,
            "enough_windows_for_confidence": n_windows >= MIN_OOS_WINDOWS_FOR_CONFIDENCE,
            "notes": "; ".join(notes) if notes else "ok",
        })
    summary_df = pd.DataFrame(summary_rows)

    if save:
        utils.write_df(audit_df, config.RESULTS_DIR / "oos_audit.csv")
        utils.write_df(summary_df, config.RESULTS_DIR / "oos_audit_summary.csv")
    return audit_df, summary_df
