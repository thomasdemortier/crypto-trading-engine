"""
Kronos confirmation logic.

Pure rule mapping from a (base_signal_action, forecast_return_pct) pair to
a confirmation verdict. NEVER calls Kronos. NEVER places trades. NEVER
touches the risk engine. The output is a plain dict / DataFrame that the
strategy wrapper looks up at backtest time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import pandas as pd

from .. import config, utils

logger = utils.get_logger("cte.kronos.confirm")


# Verdict labels — exposed as constants so strategy wrappers compare safely.
CONFIRM = "CONFIRM"
REJECT = "REJECT"
NEUTRAL = "NEUTRAL"
NO_CALL = "DO_NOT_CALL_KRONOS"

# Default thresholds. These are the Kronos confirmation policy defaults
# from the spec — change with caution; they intentionally err on the side
# of skipping uncertain trades rather than acting on them.
DEFAULT_BUY_CONFIRM_PCT = 1.0
DEFAULT_SELL_CONFIRM_PCT = 0.0
DEFAULT_NEUTRAL_BAND_PCT = 1.0


@dataclass(frozen=True)
class Confirmation:
    base_signal: str
    forecast_return_pct: float
    confirmation: str          # CONFIRM / REJECT / NEUTRAL / DO_NOT_CALL_KRONOS
    confidence_proxy: float    # abs(forecast_return_pct) — NOT a true confidence
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "base_signal": self.base_signal,
            "forecast_return_pct": float(self.forecast_return_pct),
            "confirmation": self.confirmation,
            "confidence_proxy": float(self.confidence_proxy),
            "reason": self.reason,
        }


def confirm_signal_with_kronos(
    base_signal: str,
    forecast_return_pct: float,
    buy_confirm_threshold_pct: float = DEFAULT_BUY_CONFIRM_PCT,
    sell_confirm_threshold_pct: float = DEFAULT_SELL_CONFIRM_PCT,
    neutral_band_pct: float = DEFAULT_NEUTRAL_BAND_PCT,
) -> Confirmation:
    """Map (base_signal, forecast_return_pct) to a verdict.

    Rules (per spec):
        BUY:  CONFIRM if return >= +buy_threshold (default +1.0%)
              REJECT  if return <  0
              NEUTRAL otherwise (0 <= return < +buy_threshold)
        SELL: CONFIRM if return <= 0
              REJECT  if return >  +neutral_band (default +1.0%)
              NEUTRAL otherwise (0 < return <= +neutral_band)
        HOLD/SKIP: DO_NOT_CALL_KRONOS — keep original signal.
    """
    sig = (base_signal or "").upper()
    if sig in ("HOLD", "SKIP", ""):
        return Confirmation(
            base_signal=sig,
            forecast_return_pct=float(forecast_return_pct),
            confirmation=NO_CALL,
            confidence_proxy=abs(float(forecast_return_pct)),
            reason="kronos not consulted (base signal is hold/skip)",
        )

    fr = float(forecast_return_pct)
    proxy = abs(fr)

    if sig == "BUY":
        if fr >= buy_confirm_threshold_pct:
            return Confirmation(sig, fr, CONFIRM, proxy,
                                f"forecast +{fr:.2f}% >= +{buy_confirm_threshold_pct}%")
        if fr < 0.0:
            return Confirmation(sig, fr, REJECT, proxy,
                                f"forecast {fr:+.2f}% < 0%")
        return Confirmation(sig, fr, NEUTRAL, proxy,
                            f"forecast {fr:+.2f}% in [0, +{buy_confirm_threshold_pct}%) neutral band")

    if sig == "SELL":
        if fr <= sell_confirm_threshold_pct:
            return Confirmation(sig, fr, CONFIRM, proxy,
                                f"forecast {fr:+.2f}% <= {sell_confirm_threshold_pct}%")
        if fr > neutral_band_pct:
            return Confirmation(sig, fr, REJECT, proxy,
                                f"forecast +{fr:.2f}% > +{neutral_band_pct}% (price still rising)")
        return Confirmation(sig, fr, NEUTRAL, proxy,
                            f"forecast {fr:+.2f}% in (0, +{neutral_band_pct}%] neutral band")

    # Unknown action — never confirm.
    return Confirmation(sig, fr, NO_CALL, proxy,
                        f"unknown base signal {base_signal!r}")


# ---------------------------------------------------------------------------
# Bulk generator: align rule-based signals to the most-recent applicable
# Kronos forecast, then write confirmations.csv.
# ---------------------------------------------------------------------------
def _final_signal(base_action: str, conf: Confirmation, in_position_hint: Optional[bool]) -> str:
    """Apply the confirmation policy to derive the final action label.

    Mirrors the logic in `KronosConfirmedStrategy.signal_for_row` — kept here
    so the saved CSV is self-explanatory and downstream tooling does not need
    to recompute it. The strategy wrapper still re-derives the final action
    at backtest time using its known `in_position` state.
    """
    if conf.confirmation == CONFIRM:
        return base_action
    if conf.confirmation == REJECT:
        if base_action == "BUY":
            return "SKIP"
        if base_action == "SELL":
            return "HOLD" if in_position_hint else "SKIP"
        return base_action
    if conf.confirmation == NEUTRAL:
        if base_action == "BUY":
            return "SKIP"
        return base_action  # SELL kept; HOLD/SKIP unchanged
    return base_action  # NO_CALL or unknown


def generate_kronos_confirmations(
    base_signals_df: pd.DataFrame,
    forecast_eval_df: pd.DataFrame,
    buy_confirm_threshold_pct: float = DEFAULT_BUY_CONFIRM_PCT,
    sell_confirm_threshold_pct: float = DEFAULT_SELL_CONFIRM_PCT,
    neutral_band_pct: float = DEFAULT_NEUTRAL_BAND_PCT,
    save: bool = True,
    save_path=None,
) -> pd.DataFrame:
    """Match each (asset, timestamp) row in `base_signals_df` to its most
    recent Kronos forecast in `forecast_eval_df`, then apply the
    confirmation policy.

    `base_signals_df` must contain at least: timestamp_ms, asset, timeframe,
        action (BUY / SELL / HOLD / SKIP).
    `forecast_eval_df` must contain at least: asset, timeframe,
        forecast_start_ms, forecast_return_pct.
    """
    required_signal = {"timestamp_ms", "asset", "timeframe", "action"}
    missing_sig = required_signal - set(base_signals_df.columns)
    if missing_sig:
        raise ValueError(f"base_signals_df missing columns: {sorted(missing_sig)}")
    required_fc = {"asset", "timeframe", "forecast_start_ms", "forecast_return_pct"}
    missing_fc = required_fc - set(forecast_eval_df.columns)
    if missing_fc:
        raise ValueError(f"forecast_eval_df missing columns: {sorted(missing_fc)}")

    rows = []
    for _, sig_row in base_signals_df.iterrows():
        asset = sig_row["asset"]
        tf = sig_row["timeframe"]
        ts = int(sig_row["timestamp_ms"])
        action = str(sig_row["action"])
        in_position_hint = bool(sig_row.get("in_position", False))

        sub = forecast_eval_df[
            (forecast_eval_df["asset"] == asset)
            & (forecast_eval_df["timeframe"] == tf)
            & (forecast_eval_df["forecast_start_ms"] <= ts)
        ]
        if sub.empty:
            forecast_return = float("nan")
            kronos_dir = "unknown"
        else:
            latest = sub.sort_values("forecast_start_ms").iloc[-1]
            forecast_return = float(latest["forecast_return_pct"])
            kronos_dir = "up" if forecast_return > 0 else "down" if forecast_return < 0 else "flat"

        if pd.isna(forecast_return):
            conf = Confirmation(
                base_signal=action.upper(),
                forecast_return_pct=0.0,
                confirmation=NO_CALL,
                confidence_proxy=0.0,
                reason="no Kronos forecast covers this timestamp",
            )
            final_action = action  # keep original; wrapper will fall back
        else:
            conf = confirm_signal_with_kronos(
                action, forecast_return,
                buy_confirm_threshold_pct=buy_confirm_threshold_pct,
                sell_confirm_threshold_pct=sell_confirm_threshold_pct,
                neutral_band_pct=neutral_band_pct,
            )
            final_action = _final_signal(action, conf, in_position_hint)

        rows.append({
            "timestamp": pd.to_datetime(ts, unit="ms", utc=True).isoformat(),
            "timestamp_ms": ts,
            "asset": asset,
            "timeframe": tf,
            "base_signal": action,
            "forecast_return_pct": forecast_return,
            "kronos_direction": kronos_dir,
            "confirmation": conf.confirmation,
            "confidence_proxy": conf.confidence_proxy,
            "final_signal": final_action,
            "reason": conf.reason,
        })
    out = pd.DataFrame(rows)
    if save:
        utils.write_df(out, save_path or (config.RESULTS_DIR / "kronos_confirmations.csv"))
    return out
