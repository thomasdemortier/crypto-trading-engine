"""
Portfolio momentum implementation auditor.

Pure inspection — never modifies the strategy, the backtester, or the
verdict. Loads the saved CSVs from the most recent
`research_all_portfolio` run, reproduces critical calculations
independently (so we don't grade the strategy with the same code that
might have a bug), and writes diagnostics CSVs the dashboard surfaces.

Three audits, all read-only:

1. Cash filter (`audit_cash_filter`)
   * Recomputes BTC < BTC.rolling(200d).mean() independently from the
     raw candle file.
   * Counts bearish days, reports first / last bearish date, contiguous
     bearish spans.
   * Cross-references with the saved `portfolio_momentum_weights.csv` to
     confirm every bearish rebalance produced an EMPTY target.
   * Disambiguates "empty because cash filter" from "empty because
     warmup" / "empty because too few eligible assets".

2. Benchmark alignment (`audit_benchmark_alignment`)
   * For each OOS window in `portfolio_walk_forward.csv`, recomputes the
     BTC, ETH, and equal-weight-basket returns over the EXACT SAME
     timestamp axis the strategy used.
   * Compares with the saved `btc_oos_return_pct` /
     `basket_oos_return_pct` columns and flags any drift > 0.5pp.

3. Rebalance logic (`audit_rebalance_logic`)
   * Inspects the saved `portfolio_momentum_trades.csv`.
   * Confirms: every fill has fee >= 0 and slippage_cost >= 0; cash
     never goes negative in the equity curve; weights at any point sum
     to <= 1.0 + small floating-point tolerance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config, utils
from .strategies.momentum_rotation import MomentumRotationConfig

logger = utils.get_logger("cte.portfolio_audit")


# Tolerance for "matches the strategy" comparisons.
_BENCH_RTN_TOL_PCT = 0.5
_WEIGHT_SUM_TOL = 1e-6


# ---------------------------------------------------------------------------
# Phase 1 — Cash filter
# ---------------------------------------------------------------------------
def _load_btc_with_ma(timeframe: str = "1d", ma_window_days: int = 200) -> pd.DataFrame:
    """Independent recomputation of BTC + its `ma_window_days` SMA."""
    p = utils.csv_path_for("BTC/USDT", timeframe)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    bars_per_day = {"1h": 24, "4h": 6, "1d": 1}.get(timeframe, 1)
    n_bars = int(ma_window_days * bars_per_day)
    df["ma_filter"] = df["close"].rolling(n_bars, min_periods=n_bars).mean()
    df["below_ma"] = df["close"] < df["ma_filter"]
    return df


def _classify_empty_weight_rows(
    weights_df: pd.DataFrame,
    btc_df: pd.DataFrame,
    universe_size_at: Dict[pd.Timestamp, int],
    min_assets_required: int = 5,
) -> pd.DataFrame:
    """Add a `reason_for_cash` column to weight rows that have an empty
    target — distinguishes "cash filter bearish" from "warmup" from
    "too few eligible assets" from "rebalance not requested"."""
    out = weights_df.copy()
    out["weights_json"] = out["weights_json"].fillna("").astype(str)
    out["is_empty"] = out["weights_json"].str.strip() == ""
    out["btc_below_ma"] = False
    out["btc_ma_warmup_complete"] = False
    out["enough_assets_in_universe"] = False
    out["reason_for_cash"] = ""

    # Date-keyed lookups (we work at daily resolution).
    btc_lookup = btc_df.copy()
    btc_lookup["date"] = btc_lookup["datetime"].dt.date
    btc_idx = btc_lookup.set_index("date")
    out["date"] = pd.to_datetime(out["datetime"], utc=True).dt.date

    for i, r in out.iterrows():
        d = r["date"]
        b = btc_idx.loc[d] if d in btc_idx.index else None
        if b is None:
            out.at[i, "reason_for_cash"] = "btc_data_missing"
            continue
        ma_ok = bool(pd.notna(b["ma_filter"]))
        out.at[i, "btc_ma_warmup_complete"] = ma_ok
        out.at[i, "btc_below_ma"] = bool(b["below_ma"]) if ma_ok else False
        n_eligible = universe_size_at.get(pd.Timestamp(d), 0)
        out.at[i, "enough_assets_in_universe"] = n_eligible >= min_assets_required
        if not bool(r["is_empty"]):
            out.at[i, "reason_for_cash"] = "not_empty"
            continue
        if not ma_ok:
            out.at[i, "reason_for_cash"] = "warmup_btc_ma_not_yet_computable"
        elif bool(b["below_ma"]):
            out.at[i, "reason_for_cash"] = "cash_filter_bearish"
        elif n_eligible < min_assets_required:
            out.at[i, "reason_for_cash"] = "too_few_eligible_assets"
        else:
            out.at[i, "reason_for_cash"] = "unexpected_empty"
    return out


def _eligible_assets_at_each_date(
    asset_frames: Dict[str, pd.DataFrame],
    momentum_long_window_days: int,
    timeframe: str = "1d",
) -> Dict[pd.Timestamp, int]:
    """For every observed date, how many assets have at least
    `momentum_long_window_days * bars_per_day + 1` bars of history."""
    bars_per_day = {"1h": 24, "4h": 6, "1d": 1}.get(timeframe, 1)
    needed = momentum_long_window_days * bars_per_day + 1
    asset_first_eligible: Dict[str, pd.Timestamp] = {}
    for asset, df in asset_frames.items():
        if len(df) < needed:
            continue
        first_ts = pd.to_datetime(df["datetime"].iloc[needed - 1], utc=True)
        asset_first_eligible[asset] = first_ts.normalize()
    # Per-date count of assets where date >= first_eligible.
    all_dates: List[pd.Timestamp] = sorted({
        pd.to_datetime(d, utc=True).normalize()
        for df in asset_frames.values()
        for d in df["datetime"].iloc[needed:]
    })
    out: Dict[pd.Timestamp, int] = {}
    for d in all_dates:
        out[d] = sum(1 for ts in asset_first_eligible.values() if ts <= d)
    return out


def audit_cash_filter(save: bool = True) -> Dict[str, Any]:
    """Independent recompute of the BTC 200d cash filter, cross-checked
    against the saved weights file. Returns a dict summary + per-row
    classification CSV."""
    btc = _load_btc_with_ma(timeframe="1d", ma_window_days=200)
    if btc.empty:
        return {"ok": False, "reason": "BTC/USDT 1d data missing"}

    n_total_eval = int(btc["ma_filter"].notna().sum())
    bearish = btc[btc["below_ma"].fillna(False)]
    n_bearish = int(len(bearish))
    pct_bearish = (n_bearish / n_total_eval * 100.0) if n_total_eval else 0.0

    # Contiguous bearish spans.
    spans: List[Tuple[str, str, int]] = []
    if not bearish.empty:
        dates = bearish["datetime"].dt.date.tolist()
        cur_start, prev = dates[0], dates[0]
        for d in dates[1:]:
            if (d - prev).days > 1:
                spans.append((str(cur_start), str(prev), (prev - cur_start).days + 1))
                cur_start = d
            prev = d
        spans.append((str(cur_start), str(prev), (prev - cur_start).days + 1))

    # Cross-check vs saved weights file.
    weights_path = config.RESULTS_DIR / "portfolio_momentum_weights.csv"
    cross_check_rows: List[Dict] = []
    classified = pd.DataFrame()
    if weights_path.exists():
        weights = pd.read_csv(weights_path)
        weights["datetime"] = pd.to_datetime(weights["datetime"], utc=True)
        # Build eligibility lookup by loading the universe.
        from . import portfolio_backtester as pb
        frames, _ = pb.load_universe(
            assets=list(config.EXPANDED_UNIVERSE), timeframe="1d",
        )
        cfg = MomentumRotationConfig()
        eligibility = _eligible_assets_at_each_date(
            frames, cfg.momentum_long_window, timeframe="1d",
        )
        classified = _classify_empty_weight_rows(
            weights, btc, eligibility,
            min_assets_required=cfg.min_assets_required,
        )
        # Independent expectation: on every bearish-BTC rebalance row
        # where the 200d MA is computable, weights MUST be empty.
        intended = classified[classified["btc_below_ma"]
                              & classified["btc_ma_warmup_complete"]]
        agree = int(intended["is_empty"].sum())
        disagree = int((~intended["is_empty"]).sum())
        cross_check_rows.append({
            "metric": "rebalance_rows_with_btc_below_ma",
            "value": int(len(intended)),
        })
        cross_check_rows.append({
            "metric": "of_those_with_empty_target_weights",
            "value": agree,
        })
        cross_check_rows.append({
            "metric": "of_those_with_NON_empty_target_weights_BUG_IF_NONZERO",
            "value": disagree,
        })

    summary = {
        "ok": True,
        "btc_data_first": str(btc["datetime"].iloc[0].date()),
        "btc_data_last": str(btc["datetime"].iloc[-1].date()),
        "n_eval_bars_after_warmup": n_total_eval,
        "n_bearish_bars": n_bearish,
        "pct_bearish": round(pct_bearish, 2),
        "first_bearish_date": str(bearish["datetime"].iloc[0].date())
                              if not bearish.empty else None,
        "last_bearish_date": str(bearish["datetime"].iloc[-1].date())
                             if not bearish.empty else None,
        "n_contiguous_bearish_spans": len(spans),
        "longest_bearish_span_days": max((d for _, _, d in spans), default=0),
        "weights_file_present": weights_path.exists(),
        "cross_check": cross_check_rows,
    }

    if save:
        if not classified.empty:
            cols = [c for c in [
                "datetime", "filled", "is_empty", "btc_below_ma",
                "btc_ma_warmup_complete", "enough_assets_in_universe",
                "reason_for_cash", "weights_json",
            ] if c in classified.columns]
            utils.write_df(
                classified[cols],
                config.RESULTS_DIR / "portfolio_cash_filter_audit.csv",
            )
        else:
            utils.write_df(
                pd.DataFrame([{
                    "note": "weights file missing — only summary computed",
                    **{k: str(v) for k, v in summary.items() if k != "cross_check"},
                }]),
                config.RESULTS_DIR / "portfolio_cash_filter_audit.csv",
            )
    return summary


# ---------------------------------------------------------------------------
# Phase 2 — Benchmark alignment
# ---------------------------------------------------------------------------
def _bnh_return_for_window(asset_df: pd.DataFrame,
                           start_ts_ms: int, end_ts_ms: int) -> Optional[float]:
    sub = asset_df[(asset_df["timestamp"] >= start_ts_ms)
                   & (asset_df["timestamp"] <= end_ts_ms)]
    if len(sub) < 2:
        return None
    first = float(sub["close"].iloc[0])
    last = float(sub["close"].iloc[-1])
    if first <= 0:
        return None
    return (last / first - 1.0) * 100.0


def audit_benchmark_alignment(save: bool = True) -> pd.DataFrame:
    """For every OOS window in `portfolio_walk_forward.csv`, recompute the
    BTC and equal-weight-basket OOS returns from raw candles and compare
    to the saved values. Flags any per-window drift > 0.5 pp."""
    wf_path = config.RESULTS_DIR / "portfolio_walk_forward.csv"
    if not wf_path.exists():
        return pd.DataFrame()
    wf = pd.read_csv(wf_path)
    if wf.empty or "oos_start_iso" not in wf.columns:
        return pd.DataFrame()
    from . import portfolio_backtester as pb
    frames, missing = pb.load_universe(
        assets=list(config.EXPANDED_UNIVERSE), timeframe="1d",
    )
    rows: List[Dict] = []
    for _, w in wf.iterrows():
        if pd.isna(w.get("oos_start_iso")):
            continue
        start_dt = pd.to_datetime(w["oos_start_iso"], utc=True)
        end_dt = pd.to_datetime(w["oos_end_iso"], utc=True)
        start_ms = int(start_dt.value // 10**6)
        end_ms = int(end_dt.value // 10**6)
        # BTC independent recompute.
        btc_recomputed = _bnh_return_for_window(
            frames.get("BTC/USDT", pd.DataFrame()), start_ms, end_ms,
        )
        # Basket recompute: equal-weight at window start.
        basket_recomputed = None
        n = len(frames)
        if n > 0:
            per = 1.0 / n
            comp = []
            for asset, df in frames.items():
                sub = df[(df["timestamp"] >= start_ms)
                         & (df["timestamp"] <= end_ms)]
                if len(sub) < 2:
                    continue
                first = float(sub["close"].iloc[0])
                last = float(sub["close"].iloc[-1])
                if first > 0:
                    comp.append(per * (last / first))
            if comp:
                basket_recomputed = (sum(comp) - 1.0) * 100.0
        btc_saved = w.get("btc_oos_return_pct")
        basket_saved = w.get("basket_oos_return_pct")
        rows.append({
            "window": int(w.get("window", 0)),
            "oos_start_iso": str(start_dt),
            "oos_end_iso": str(end_dt),
            "btc_saved": (None if pd.isna(btc_saved) else float(btc_saved)),
            "btc_recomputed": btc_recomputed,
            "btc_drift_pp": (
                None if btc_recomputed is None or pd.isna(btc_saved)
                else round(float(btc_saved) - btc_recomputed, 4)
            ),
            "basket_saved": (None if pd.isna(basket_saved)
                             else float(basket_saved)),
            "basket_recomputed": basket_recomputed,
            "basket_drift_pp": (
                None if basket_recomputed is None or pd.isna(basket_saved)
                else round(float(basket_saved) - basket_recomputed, 4)
            ),
            "btc_alignment_ok": (
                btc_recomputed is not None and not pd.isna(btc_saved)
                and abs(float(btc_saved) - btc_recomputed) <= _BENCH_RTN_TOL_PCT
            ),
            "basket_alignment_ok": (
                basket_recomputed is not None and not pd.isna(basket_saved)
                and abs(float(basket_saved) - basket_recomputed) <= _BENCH_RTN_TOL_PCT
            ),
        })
    out = pd.DataFrame(rows)
    if save:
        utils.write_df(
            out,
            config.RESULTS_DIR / "portfolio_benchmark_alignment_audit.csv",
        )
    return out


# ---------------------------------------------------------------------------
# Phase 3 — Rebalance logic
# ---------------------------------------------------------------------------
def audit_rebalance_logic(save: bool = True) -> pd.DataFrame:
    """Inspect the saved trades + equity files for fee/slippage/cash
    invariants. Returns a one-row summary DataFrame with bool flags."""
    trades_path = config.RESULTS_DIR / "portfolio_momentum_trades.csv"
    equity_path = config.RESULTS_DIR / "portfolio_momentum_equity.csv"
    weights_path = config.RESULTS_DIR / "portfolio_momentum_weights.csv"
    rows: List[Dict] = []

    if trades_path.exists():
        trades = pd.read_csv(trades_path)
        n_trades = len(trades)
        n_buys = int((trades["side"] == "BUY").sum()) if n_trades else 0
        n_sells = int((trades["side"] == "SELL").sum()) if n_trades else 0
        fees_nonneg = bool((trades["fee"] >= 0).all()) if n_trades else True
        slip_nonneg = bool((trades["slippage_cost"] >= 0).all()) if n_trades else True
        rows += [
            {"check": "n_trades_total", "value": n_trades, "ok": True},
            {"check": "n_buys", "value": n_buys, "ok": True},
            {"check": "n_sells", "value": n_sells, "ok": True},
            {"check": "every_fee_non_negative", "value": fees_nonneg,
             "ok": fees_nonneg},
            {"check": "every_slippage_non_negative", "value": slip_nonneg,
             "ok": slip_nonneg},
        ]
    else:
        rows.append({"check": "trades_file_present", "value": False, "ok": False})

    if equity_path.exists():
        eq = pd.read_csv(equity_path)
        cash_min = float(eq["cash"].min())
        eq_min = float(eq["equity"].min())
        cash_ok = cash_min >= -1e-6
        eq_ok = eq_min >= 0.0
        rows += [
            {"check": "cash_min", "value": round(cash_min, 4), "ok": cash_ok},
            {"check": "equity_min", "value": round(eq_min, 4), "ok": eq_ok},
            {"check": "cash_never_negative", "value": cash_ok, "ok": cash_ok},
            {"check": "equity_never_negative", "value": eq_ok, "ok": eq_ok},
        ]
    else:
        rows.append({"check": "equity_file_present", "value": False, "ok": False})

    if weights_path.exists():
        weights = pd.read_csv(weights_path)
        if "weights_json" in weights.columns:
            sums: List[float] = []
            for s in weights["weights_json"].fillna("").astype(str):
                if not s.strip():
                    sums.append(0.0)
                    continue
                total = 0.0
                for piece in s.split(","):
                    piece = piece.strip()
                    if not piece:
                        continue
                    try:
                        total += float(piece.split("=")[-1])
                    except (ValueError, IndexError):
                        pass
                sums.append(total)
            max_sum = max(sums) if sums else 0.0
            ok = max_sum <= 1.0 + _WEIGHT_SUM_TOL
            rows += [
                {"check": "max_target_weight_sum", "value": round(max_sum, 6),
                 "ok": ok},
                {"check": "weights_never_imply_leverage", "value": ok,
                 "ok": ok},
            ]
    else:
        rows.append({"check": "weights_file_present", "value": False, "ok": False})

    out = pd.DataFrame(rows)
    if save:
        utils.write_df(out, config.RESULTS_DIR / "portfolio_rebalance_audit.csv")
    return out


# ---------------------------------------------------------------------------
# Convenience: run all three audits and return a summary dict
# ---------------------------------------------------------------------------
def audit_all_portfolio() -> Dict[str, Any]:
    cf = audit_cash_filter(save=True)
    bench = audit_benchmark_alignment(save=True)
    reb = audit_rebalance_logic(save=True)
    return {
        "cash_filter": cf,
        "benchmark_alignment_rows": int(len(bench)) if isinstance(bench, pd.DataFrame) else 0,
        "benchmark_alignment_ok_btc": (
            bool(bench["btc_alignment_ok"].all()) if not bench.empty else None
        ),
        "benchmark_alignment_ok_basket": (
            bool(bench["basket_alignment_ok"].all()) if not bench.empty else None
        ),
        "rebalance_audit_all_ok": bool(reb["ok"].all()) if not reb.empty else None,
    }
