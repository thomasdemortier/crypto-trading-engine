"""
Kronos forecast evaluation + base-vs-Kronos-confirmed comparison.

Two responsibilities:

1. `evaluate_kronos_forecasts` — rolling-window evaluator that asks
   Kronos to forecast `pred_len` candles ahead at every `step` bars,
   then compares the forecast against the actual close. The output CSV
   feeds both the dashboard and the confirmation generator.

2. `compare_base_vs_kronos_confirmed` — runs the SAME base strategy
   twice through the existing backtester: once standalone, once wrapped
   in `KronosConfirmedStrategy` (which reads the saved confirmations
   CSV). Identical risk engine, fees, slippage, and execution model on
   both sides — the only difference is the Kronos confirmation gate.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import backtester, config, data_collector, performance, utils
from ..strategies import REGISTRY as STRATEGY_REGISTRY
from ..strategies.base import Strategy
from . import kronos_adapter

logger = utils.get_logger("cte.kronos.eval")


# ---------------------------------------------------------------------------
# Phase 3 — rolling forecast evaluation
# ---------------------------------------------------------------------------
def evaluate_kronos_forecasts(
    asset: str,
    timeframe: str,
    candles_df: Optional[pd.DataFrame] = None,
    model_name: str = kronos_adapter.DEFAULT_MODEL,
    lookback: int = kronos_adapter.DEFAULT_LOOKBACK,
    pred_len: int = kronos_adapter.DEFAULT_PRED_LEN,
    step: int = kronos_adapter.DEFAULT_PRED_LEN,
    max_windows: int = 20,
    device: str = "cpu",
    T: float = kronos_adapter.DEFAULT_T,
    top_p: float = kronos_adapter.DEFAULT_TOP_P,
    sample_count: int = kronos_adapter.DEFAULT_SAMPLE_COUNT,
    save: bool = True,
    save_path: Optional[Any] = None,
) -> pd.DataFrame:
    """Run rolling Kronos forecasts and report direction accuracy + error.

    The output DataFrame's columns match the Phase 3 spec — one row per
    window plus a header summary. We never silently mask Kronos failures:
    each window catches its own exception and records the reason.

    Lazy: imports Kronos only inside this function.
    """
    if candles_df is None:
        candles_df = data_collector.load_candles(asset, timeframe)
    candles_df = candles_df.copy().reset_index(drop=True)
    candles_df["datetime"] = pd.to_datetime(
        candles_df["datetime"], utc=True, errors="coerce",
    )

    spec = kronos_adapter.KRONOS_MODELS[model_name]
    lookback = min(int(lookback), int(spec.max_context))

    # Window cursor moves from the first valid endpoint to the last point
    # where we still have `pred_len` future bars to score against.
    first_end = lookback
    last_end = len(candles_df) - pred_len
    if last_end <= first_end:
        raise RuntimeError(
            f"need at least {lookback + pred_len} candles for evaluation, "
            f"have {len(candles_df)}"
        )
    cursors = list(range(first_end, last_end + 1, max(1, int(step))))
    if max_windows is not None and max_windows > 0:
        cursors = cursors[:int(max_windows)]

    predictor = None
    rows = []
    for w_idx, end_idx in enumerate(cursors, start=1):
        try:
            if predictor is None:
                predictor, _ = kronos_adapter.load_kronos_predictor(
                    model_name=model_name, device=device,
                )
            pred_df = kronos_adapter.run_kronos_forecast(
                candles=candles_df, timeframe=timeframe,
                model_name=model_name, lookback=lookback,
                pred_len=pred_len, device=device,
                T=T, top_p=top_p, sample_count=sample_count,
                end_index=end_idx, predictor=predictor,
            )
            current_close = float(candles_df.iloc[end_idx - 1]["close"])
            actual_close = float(candles_df.iloc[end_idx + pred_len - 1]["close"])
            forecast_close = float(pred_df["close"].iloc[-1])
            forecast_ret = (forecast_close / current_close - 1.0) * 100.0
            actual_ret = (actual_close / current_close - 1.0) * 100.0
            rows.append({
                "asset": asset,
                "timeframe": timeframe,
                "window_id": w_idx,
                "forecast_start": pd.to_datetime(
                    candles_df.iloc[end_idx - 1]["datetime"], utc=True,
                ).isoformat(),
                "forecast_end": pd.to_datetime(
                    candles_df.iloc[end_idx + pred_len - 1]["datetime"], utc=True,
                ).isoformat(),
                "forecast_start_ms": int(
                    pd.Timestamp(candles_df.iloc[end_idx - 1]["datetime"]).value // 10**6
                ),
                "forecast_end_ms": int(
                    pd.Timestamp(candles_df.iloc[end_idx + pred_len - 1]["datetime"]).value // 10**6
                ),
                "current_close": current_close,
                "forecast_close": forecast_close,
                "actual_close": actual_close,
                "forecast_return_pct": forecast_ret,
                "actual_return_pct": actual_ret,
                "direction_correct": bool(
                    (forecast_ret > 0 and actual_ret > 0)
                    or (forecast_ret < 0 and actual_ret < 0)
                ),
                "abs_error_pct": abs(forecast_ret - actual_ret),
                "bias_pct": forecast_ret - actual_ret,
                "model_name": model_name,
                "lookback": lookback,
                "pred_len": pred_len,
                "device": device,
                "error": None,
            })
        except Exception as e:  # noqa: BLE001
            rows.append({
                "asset": asset, "timeframe": timeframe, "window_id": w_idx,
                "forecast_start": None, "forecast_end": None,
                "forecast_start_ms": None, "forecast_end_ms": None,
                "current_close": None, "forecast_close": None,
                "actual_close": None, "forecast_return_pct": None,
                "actual_return_pct": None, "direction_correct": None,
                "abs_error_pct": None, "bias_pct": None,
                "model_name": model_name, "lookback": lookback,
                "pred_len": pred_len, "device": device,
                "error": f"{type(e).__name__}: {e}",
            })
    out = pd.DataFrame(rows)
    if save:
        utils.write_df(out, save_path or (config.RESULTS_DIR / "kronos_forecast_evaluation.csv"))
    return out


def summarise_forecast_evaluation(eval_df: pd.DataFrame) -> Dict[str, object]:
    """Aggregate the rolling-evaluation DataFrame into the dashboard's
    headline metrics. Safe with an empty / all-error frame."""
    if eval_df is None or eval_df.empty:
        return {"ok": False, "reason": "no evaluation rows"}
    ok = eval_df[eval_df["error"].isna()] if "error" in eval_df.columns else eval_df
    n = len(ok)
    if n == 0:
        return {"ok": False, "reason": "every window errored", "n_windows": 0}
    return {
        "ok": True,
        "n_windows": int(n),
        "direction_accuracy_pct": float(ok["direction_correct"].mean() * 100.0),
        "mape_pct": float(ok["abs_error_pct"].mean()),
        "avg_forecast_return_pct": float(ok["forecast_return_pct"].mean()),
        "avg_actual_return_pct": float(ok["actual_return_pct"].mean()),
        "avg_bias_pct": float(ok["bias_pct"].mean()),
        "worst_abs_error_pct": float(ok["abs_error_pct"].max()),
    }


# ---------------------------------------------------------------------------
# Phase 6 — base vs Kronos-confirmed comparison
# ---------------------------------------------------------------------------
def _row_for_metrics(
    label: str, asset: str, timeframe: str, m: performance.Metrics,
) -> Dict[str, object]:
    return {
        "variant": label, "asset": asset, "timeframe": timeframe,
        "total_return_pct": m.total_return_pct,
        "buy_and_hold_return_pct": m.buy_and_hold_return_pct,
        "strategy_vs_bh_pct": m.strategy_vs_bh_pct,
        "max_drawdown_pct": m.max_drawdown_pct,
        "win_rate_pct": m.win_rate_pct,
        "num_trades": int(m.num_trades),
        "profit_factor": m.profit_factor,
        "fees_paid": m.fees_paid,
        "slippage_cost": m.slippage_cost,
        "exposure_time_pct": m.exposure_time_pct,
        "sharpe_ratio": m.sharpe_ratio,
        "sortino_ratio": m.sortino_ratio,
        "calmar_ratio": m.calmar_ratio,
        "starting_capital": m.starting_capital,
        "final_portfolio_value": m.final_portfolio_value,
    }


def compare_base_vs_kronos_confirmed(
    asset: str,
    timeframe: str,
    base_strategy_name: str = "rsi_ma_atr",
    kronos_confirmations_path: Optional[Any] = None,
    risk_cfg: Optional[config.RiskConfig] = None,
    strat_cfg: Optional[config.StrategyConfig] = None,
    save: bool = True,
    save_path: Optional[Any] = None,
) -> pd.DataFrame:
    """Run the base strategy and the Kronos-confirmed wrapper through the
    SAME backtester / risk engine and report per-variant metrics side by
    side. Calling this never invokes Kronos — it relies on a precomputed
    confirmations CSV.

    Returns a 2-row DataFrame: one row for the base strategy, one row
    for `base + kronos_confirmed`. The `variant` column distinguishes them.
    """
    risk_cfg = risk_cfg or config.RISK
    strat_cfg = strat_cfg or config.STRATEGY
    if base_strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"unknown base strategy {base_strategy_name!r}. Known: "
            f"{list(STRATEGY_REGISTRY)}"
        )
    base_strategy = STRATEGY_REGISTRY[base_strategy_name]()

    # Lazy import to avoid a circular dep with src.strategies.kronos_confirmed
    # at module load time.
    from ..strategies.kronos_confirmed import KronosConfirmedStrategy

    kronos_strategy = KronosConfirmedStrategy(
        base_strategy=base_strategy,
        confirmations_path=kronos_confirmations_path,
        fallback="skip",
    )

    rows = []
    for label, strat in (
        (base_strategy_name, base_strategy),
        (f"{base_strategy_name}+kronos_confirmed", kronos_strategy),
    ):
        art = backtester.run_backtest(
            assets=[asset], timeframe=timeframe,
            risk_cfg=risk_cfg, strat_cfg=strat_cfg, save=False,
            strategy=strat,
        )
        m = performance.compute_metrics(
            art.equity_curve, art.trades, art.asset_close_curves,
            starting_capital=risk_cfg.starting_capital,
        )
        rows.append(_row_for_metrics(label, asset, timeframe, m))

    out = pd.DataFrame(rows)
    if save:
        utils.write_df(out, save_path or (config.RESULTS_DIR / "kronos_confirmation_comparison.csv"))
    return out
