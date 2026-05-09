"""
Sideways mean-reversion (long-only).

Trades only when the regime detector classifies the bar as `sideways`.
Buys oversold pulls toward the lower Bollinger Band when RSI is recovering
and price is not far from MA200. Exits at the middle/upper band, on RSI
exhaustion, on a regime flip to bear, or after `max_holding_bars`.

Designed because the cached BTC + ETH 4h data is ~55-60% sideways — every
existing rule-based strategy in this repo is built for trends and is
necessarily mismatched to that regime. This strategy is the first attempt
at the regime/strategy match. It is NOT optimised, NOT tuned to backtest
results, and explicitly subject to the same risk engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from .. import indicators, regime as _regime
from .base import Strategy, Signal, BUY, SELL, HOLD, SKIP


@dataclass(frozen=True)
class SidewaysMeanReversionConfig:
    bb_window: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    rsi_min: float = 25.0
    rsi_max: float = 45.0
    rsi_exit: float = 60.0
    max_atr_pct: float = 5.0
    atr_period: int = 14
    max_ma200_distance_pct: float = 5.0
    max_holding_bars: int = 20
    near_lower_band_pct: float = 0.5  # within this % above the lower band counts


class SidewaysMeanReversionStrategy(Strategy):
    name = "sideways_mean_reversion"

    def __init__(self, cfg: Optional[SidewaysMeanReversionConfig] = None) -> None:
        self.cfg = cfg or SidewaysMeanReversionConfig()
        # Per-asset bar-counter so we can enforce max_holding_bars without
        # passing the entry timestamp through the Signal interface.
        self._bars_held: Dict[str, int] = {}

    def min_history(self, cfg: Any) -> int:
        # Need MA200 + slope buffer + Bollinger window.
        return max(200 + 25, self.cfg.bb_window + 5)

    def prepare(self, df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
        c = self.cfg
        out = df.copy()
        # Bollinger Bands.
        mid, upper, lower = indicators.bollinger_bands(
            out["close"], c.bb_window, c.bb_std,
        )
        out["smr_bb_mid"] = mid
        out["smr_bb_upper"] = upper
        out["smr_bb_lower"] = lower
        # RSI + ATR + MA200 distance.
        if "rsi" not in out.columns:
            out["rsi"] = indicators.rsi(out["close"], c.rsi_period)
        out["smr_rsi_prev"] = out["rsi"].shift(1)
        if "atr" not in out.columns:
            out["atr"] = indicators.atr(out["high"], out["low"], out["close"], c.atr_period)
        if "atr_pct" not in out.columns:
            out["atr_pct"] = (out["atr"] / out["close"]) * 100.0
        if "ma200" not in out.columns:
            out["ma200"] = indicators.sma(out["close"], 200)
        out["smr_ma200_dist_pct"] = (
            (out["close"] - out["ma200"]).abs() / out["ma200"] * 100.0
        )
        out["smr_close_prev"] = out["close"].shift(1)
        # Regime columns (idempotent — won't recompute if already there).
        if "trend_regime" not in out.columns:
            out = _regime.add_regime_columns(out)
        return out

    def signal_for_row(
        self, asset: str, row: pd.Series, in_position: bool, cfg: Any,
    ) -> Signal:
        c = self.cfg
        ts = int(row["timestamp"]) if pd.notna(row.get("timestamp")) else 0
        close = float(row["close"])

        # Update the bars-held counter BEFORE we possibly emit a SELL.
        if not in_position:
            self._bars_held[asset] = 0
        else:
            self._bars_held[asset] = self._bars_held.get(asset, 0) + 1

        bb_mid = row.get("smr_bb_mid")
        bb_upper = row.get("smr_bb_upper")
        bb_lower = row.get("smr_bb_lower")
        rsi_now = row.get("rsi")
        rsi_prev = row.get("smr_rsi_prev")
        atr_pct = row.get("atr_pct")
        ma200_dist = row.get("smr_ma200_dist_pct")
        close_prev = row.get("smr_close_prev")
        trend = row.get("trend_regime")

        if any(pd.isna(x) for x in (
            bb_mid, bb_upper, bb_lower, rsi_now, rsi_prev, atr_pct,
            ma200_dist, close_prev,
        )):
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=SKIP, price=close,
                reason="insufficient indicator history",
            )
        if pd.isna(trend) or trend == "unknown":
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=HOLD, price=close, reason="regime warmup",
            )

        # ---- Exit logic ------------------------------------------------
        if in_position:
            if trend == _regime.BEAR:
                self._bars_held[asset] = 0
                return Signal(
                    asset=asset, timestamp=ts, datetime=row.get("datetime"),
                    action=SELL, price=close,
                    reason="regime flip to bear_trend",
                    rsi=float(rsi_now), ma200=float(row.get("ma200", float("nan"))),
                )
            if close >= float(bb_upper):
                self._bars_held[asset] = 0
                return Signal(
                    asset=asset, timestamp=ts, datetime=row.get("datetime"),
                    action=SELL, price=close,
                    reason=f"close {close:.2f} >= bb_upper {float(bb_upper):.2f}",
                    rsi=float(rsi_now), ma200=float(row.get("ma200", float("nan"))),
                )
            if close >= float(bb_mid):
                self._bars_held[asset] = 0
                return Signal(
                    asset=asset, timestamp=ts, datetime=row.get("datetime"),
                    action=SELL, price=close,
                    reason=f"close {close:.2f} >= bb_mid {float(bb_mid):.2f}",
                    rsi=float(rsi_now), ma200=float(row.get("ma200", float("nan"))),
                )
            rsi_exhausted = (
                float(rsi_now) > c.rsi_exit and float(rsi_now) < float(rsi_prev)
            )
            if rsi_exhausted:
                self._bars_held[asset] = 0
                return Signal(
                    asset=asset, timestamp=ts, datetime=row.get("datetime"),
                    action=SELL, price=close,
                    reason=(f"rsi exhaustion {float(rsi_prev):.1f}->"
                            f"{float(rsi_now):.1f} above {c.rsi_exit}"),
                    rsi=float(rsi_now),
                )
            if self._bars_held[asset] >= c.max_holding_bars:
                self._bars_held[asset] = 0
                return Signal(
                    asset=asset, timestamp=ts, datetime=row.get("datetime"),
                    action=SELL, price=close,
                    reason=f"max holding bars ({c.max_holding_bars}) exceeded",
                    rsi=float(rsi_now),
                )
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=HOLD, price=close,
                reason=f"in position, bars held {self._bars_held[asset]}",
                rsi=float(rsi_now),
            )

        # ---- Entry filters ---------------------------------------------
        if trend != _regime.SIDEWAYS:
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=HOLD, price=close,
                reason=f"trend regime {trend} != sideways",
            )
        # Within X% above the lower band counts as "near".
        near_lower = (
            close <= float(bb_lower)
            or (close - float(bb_lower)) / float(bb_lower) * 100.0 <= c.near_lower_band_pct
        )
        if not near_lower:
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=HOLD, price=close,
                reason=(f"close {close:.2f} not near bb_lower "
                        f"{float(bb_lower):.2f}"),
            )
        rsi_in_band = c.rsi_min <= float(rsi_now) <= c.rsi_max
        rsi_recovering = float(rsi_now) >= float(rsi_prev)
        green_candle = close > float(close_prev)
        if not (rsi_in_band and rsi_recovering and green_candle):
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=HOLD, price=close,
                reason=(f"missing entry filter: rsi_band={rsi_in_band} "
                        f"recovering={rsi_recovering} green={green_candle}"),
            )
        if float(atr_pct) > c.max_atr_pct:
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=SKIP, price=close,
                reason=f"atr_pct {float(atr_pct):.2f}% > {c.max_atr_pct}%",
            )
        if float(ma200_dist) > c.max_ma200_distance_pct:
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=SKIP, price=close,
                reason=(f"ma200 distance {float(ma200_dist):.2f}% > "
                        f"{c.max_ma200_distance_pct}%"),
            )
        return Signal(
            asset=asset, timestamp=ts, datetime=row.get("datetime"),
            action=BUY, price=close,
            reason=(f"sideways MR: close near bb_lower, rsi "
                    f"{float(rsi_prev):.1f}->{float(rsi_now):.1f} in band, "
                    f"green candle"),
            rsi=float(rsi_now),
        )
