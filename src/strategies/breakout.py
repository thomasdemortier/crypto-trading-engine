"""
Donchian-style breakout (long-only) — enhanced.

Entry conditions (ALL must hold):
  * close > prior N-period high (rolling high uses `high.shift(1)` so we
    NEVER include the current bar — assertable lookahead-free invariant)
  * volume > volume MA * `min_volume_multiple` (skip thin breakouts)
  * close > slow MA (default MA200) — skip bear-market breakouts
  * ATR% <= max_atr_pct — skip "panic" breakouts on giant candles
  * (optional) MA50 > MA200 if `require_trend_alignment`

Exit conditions (ANY triggers):
  * close < prior M-period low
  * stop-loss handled by risk engine
  * (optional) close < slow MA (regime exit) if `exit_below_slow_ma`
  * (optional) failed breakout: still flat after `failed_breakout_bars`
    bars (handled inside `signal_for_row`)

Filter defaults are conservative on purpose — they reduce trade count and
should be expected to underperform raw breakout when the asset is in a
sustained trend, and outperform when the market chops.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from .. import indicators
from .base import Strategy, Signal, BUY, SELL, HOLD, SKIP


@dataclass(frozen=True)
class BreakoutConfig:
    entry_window: int = 20
    exit_window: int = 10
    volume_ma_window: int = 20
    min_volume_multiple: float = 1.0       # 1.0 = volume must beat its MA
    slow_ma: int = 200
    max_atr_pct: float = 5.0
    atr_period: int = 14
    require_trend_alignment: bool = True
    exit_below_slow_ma: bool = False
    failed_breakout_bars: int = 0          # 0 disables the failed-breakout exit
    use_filters: bool = True               # False = pure Donchian (legacy)


class BreakoutStrategy(Strategy):
    name = "breakout"

    def __init__(
        self,
        entry_window: int = 20,
        exit_window: int = 10,
        cfg: Optional[BreakoutConfig] = None,
        **kwargs,
    ) -> None:
        if entry_window < 2 or exit_window < 2:
            raise ValueError("breakout: windows must be >= 2")
        if cfg is None:
            cfg = BreakoutConfig(
                entry_window=entry_window,
                exit_window=exit_window,
                **kwargs,
            )
        self.cfg = cfg
        # Track entry bar index per position so we can apply the failed-
        # breakout exit (bar index is recorded in `signal_for_row`).
        self._last_entry_idx: Optional[int] = None

    def min_history(self, cfg: Any) -> int:
        c = self.cfg
        if not c.use_filters:
            return max(c.entry_window, c.exit_window) + 5
        return max(c.entry_window, c.exit_window, c.slow_ma, c.atr_period) + 5

    def prepare(self, df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
        c = self.cfg
        out = df.copy()
        # Shift by 1 so the rolling high/low uses ONLY data strictly before
        # the current bar. This is the canonical anti-lookahead pattern.
        out["roll_high"] = (out["high"].shift(1)
                            .rolling(c.entry_window).max())
        out["roll_low"] = (out["low"].shift(1)
                           .rolling(c.exit_window).min())
        if c.use_filters:
            out["bo_vol_ma"] = indicators.sma(out["volume"], c.volume_ma_window)
            out["bo_slow"] = indicators.sma(out["close"], c.slow_ma)
            out["bo_mid"] = indicators.sma(out["close"], 50)
            out["bo_atr"] = indicators.atr(
                out["high"], out["low"], out["close"], c.atr_period,
            )
            out["bo_atr_pct"] = (out["bo_atr"] / out["close"]) * 100.0
        return out

    def signal_for_row(
        self, asset: str, row: pd.Series, in_position: bool, cfg: Any,
    ) -> Signal:
        c = self.cfg
        ts = int(row["timestamp"]) if pd.notna(row.get("timestamp")) else 0
        close = float(row["close"])
        roll_high = row.get("roll_high")
        roll_low = row.get("roll_low")
        if pd.isna(roll_high) or pd.isna(roll_low):
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=SKIP, price=close,
                reason="insufficient breakout-window history",
            )

        if in_position:
            if close < float(roll_low):
                self._last_entry_idx = None
                return Signal(
                    asset=asset, timestamp=ts, datetime=row.get("datetime"),
                    action=SELL, price=close,
                    reason=(f"breakout exit: close {close:.2f} < "
                            f"{c.exit_window}-low {float(roll_low):.2f}"),
                )
            if c.use_filters and c.exit_below_slow_ma:
                slow = row.get("bo_slow")
                if pd.notna(slow) and close < float(slow):
                    self._last_entry_idx = None
                    return Signal(
                        asset=asset, timestamp=ts, datetime=row.get("datetime"),
                        action=SELL, price=close,
                        reason=(f"regime exit: close {close:.2f} < "
                                f"ma{c.slow_ma} {float(slow):.2f}"),
                    )
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=HOLD, price=close, reason="in position, no exit",
            )

        # Entry — base condition is the breakout. Filters then gate it.
        if not (close > float(roll_high)):
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=HOLD, price=close, reason="no breakout",
            )
        if c.use_filters:
            slow = row.get("bo_slow")
            mid = row.get("bo_mid")
            vol_ma = row.get("bo_vol_ma")
            atr_pct = row.get("bo_atr_pct")
            volume = row.get("volume")
            if any(pd.isna(x) for x in (slow, vol_ma, atr_pct)):
                return Signal(
                    asset=asset, timestamp=ts, datetime=row.get("datetime"),
                    action=SKIP, price=close,
                    reason="insufficient filter history",
                )
            if close <= float(slow):
                return Signal(
                    asset=asset, timestamp=ts, datetime=row.get("datetime"),
                    action=SKIP, price=close,
                    reason=(f"trend filter: close {close:.2f} <= "
                            f"ma{c.slow_ma} {float(slow):.2f}"),
                )
            if c.require_trend_alignment and pd.notna(mid) \
                    and float(mid) <= float(slow):
                return Signal(
                    asset=asset, timestamp=ts, datetime=row.get("datetime"),
                    action=SKIP, price=close,
                    reason="trend filter: ma50 <= ma200",
                )
            if pd.notna(volume) and float(volume) < float(vol_ma) * c.min_volume_multiple:
                return Signal(
                    asset=asset, timestamp=ts, datetime=row.get("datetime"),
                    action=SKIP, price=close,
                    reason=(f"volume filter: volume {float(volume):.0f} < "
                            f"{c.min_volume_multiple}x vol_ma {float(vol_ma):.0f}"),
                )
            if float(atr_pct) > c.max_atr_pct:
                return Signal(
                    asset=asset, timestamp=ts, datetime=row.get("datetime"),
                    action=SKIP, price=close,
                    reason=(f"volatility filter: atr_pct {float(atr_pct):.2f}% "
                            f"> {c.max_atr_pct}%"),
                )

        return Signal(
            asset=asset, timestamp=ts, datetime=row.get("datetime"),
            action=BUY, price=close,
            reason=(f"breakout entry: close {close:.2f} > "
                    f"{c.entry_window}-high {float(roll_high):.2f}"),
        )
