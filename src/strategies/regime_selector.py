"""
Regime-conditional strategy selector (long-only).

Routes signals to a sub-strategy based on the current market regime
detected by `src.regime.add_regime_columns`.

Default routing:
  bull_trend  + low_volatility   -> trend_following
  bull_trend  + high_volatility  -> SKIP new entries
  sideways    + low_volatility   -> sideways_mean_reversion
  sideways    + high_volatility  -> SKIP new entries
  bear_trend                     -> SKIP new entries (defensive cash)
  unknown     (warmup)           -> SKIP new entries

Invariants enforced by the implementation and tests:
  * The selector only emits BUY / SELL / HOLD / SKIP — never bypasses
    the risk engine.
  * SELL signals from a delegated strategy ALWAYS pass through. The
    selector never blocks an exit, even in bear_trend.
  * BUY signals are only allowed when the current regime route allows
    entries.
  * If a position is already open, the sub-strategy that *opened* it
    handles the exit (so its bookkeeping — bars-held counters etc. —
    stays consistent). The selector does not arbitrarily switch the
    exit logic mid-trade.
  * No lookahead — regime columns use only past + current bars.
"""
from __future__ import annotations

from dataclasses import dataclass, replace as _dc_replace
from typing import Any, Dict, Optional

import pandas as pd

from .. import regime as _regime
from .base import Strategy, Signal, BUY, SELL, HOLD, SKIP


@dataclass(frozen=True)
class RegimeSelectorConfig:
    allow_bull_high_vol: bool = False
    allow_sideways_high_vol: bool = False
    allow_bear_entries: bool = False
    allow_unknown_entries: bool = False


# Routing key constants used in `_owner` and decision logs.
ROUTE_BULL = "bull"
ROUTE_SIDEWAYS = "sideways"


class RegimeSelectorStrategy(Strategy):
    name = "regime_selector"

    def __init__(
        self,
        bull_strategy: Optional[Strategy] = None,
        sideways_strategy: Optional[Strategy] = None,
        cfg: Optional[RegimeSelectorConfig] = None,
        regime_cfg: Optional[_regime.RegimeConfig] = None,
        name_suffix: Optional[str] = None,
    ) -> None:
        # Late imports to avoid circular dependencies inside the package.
        from .trend_following import TrendFollowingStrategy
        from .sideways_mean_reversion import SidewaysMeanReversionStrategy

        self._bull = bull_strategy or TrendFollowingStrategy()
        self._sideways = sideways_strategy or SidewaysMeanReversionStrategy()
        self.cfg = cfg or RegimeSelectorConfig()
        self._regime_cfg = regime_cfg or _regime.DEFAULT
        if name_suffix:
            self.name = f"regime_selector+{name_suffix}"
        # Per-asset bookkeeping: which sub-strategy currently owns the
        # open position (so the same one handles the exit).
        self._owner: Dict[str, str] = {}

    # ---- Strategy interface ------------------------------------------------
    def min_history(self, cfg: Any) -> int:
        # We need history for both sub-strategies AND for the regime
        # detector's slope window. Whichever is largest wins.
        return max(
            self._bull.min_history(cfg),
            self._sideways.min_history(cfg),
            self._regime_cfg.ma_long + self._regime_cfg.slope_window + 5,
        )

    def prepare(self, df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
        # Apply regime columns first so the sub-strategies' prepare()
        # methods (which may also call add_regime_columns) detect them and
        # skip recompute.
        out = _regime.add_regime_columns(df, self._regime_cfg)
        out = self._bull.prepare(out, cfg)
        out = self._sideways.prepare(out, cfg)
        return out

    def _entry_route(self, trend: str, vol: str) -> Optional[str]:
        """Return ROUTE_BULL / ROUTE_SIDEWAYS / None.
        None means: do NOT open a new position in this regime."""
        if trend == _regime.BEAR:
            return None if not self.cfg.allow_bear_entries else ROUTE_BULL
        if trend == "unknown" or vol == "unknown":
            return None if not self.cfg.allow_unknown_entries else ROUTE_BULL
        if trend == _regime.BULL:
            if vol == _regime.LOW_VOL:
                return ROUTE_BULL
            return ROUTE_BULL if self.cfg.allow_bull_high_vol else None
        if trend == _regime.SIDEWAYS:
            if vol == _regime.LOW_VOL:
                return ROUTE_SIDEWAYS
            return ROUTE_SIDEWAYS if self.cfg.allow_sideways_high_vol else None
        return None

    def signal_for_row(
        self, asset: str, row: pd.Series, in_position: bool, cfg: Any,
    ) -> Signal:
        ts = int(row["timestamp"]) if pd.notna(row.get("timestamp")) else 0
        close = float(row["close"])
        trend = row.get("trend_regime")
        vol = row.get("volatility_regime")

        # ---- Already in a position: delegate exit to the OWNER ----------
        if in_position:
            owner = self._owner.get(asset)
            if owner is None:
                # Defensive — we found ourselves in a position with no
                # recorded owner (e.g. carried over from another strategy
                # in the same backtest instance). Default to bull route.
                owner = ROUTE_BULL
                self._owner[asset] = owner
            sub = self._bull if owner == ROUTE_BULL else self._sideways
            sig = sub.signal_for_row(asset, row, True, cfg)
            if sig.action == SELL:
                self._owner.pop(asset, None)
            return _dc_replace(
                sig, reason=f"{sig.reason} [via {owner}]",
            )

        # ---- Not in a position: pick a route based on the regime --------
        route = self._entry_route(str(trend) if trend is not None else "unknown",
                                  str(vol) if vol is not None else "unknown")
        if route is None:
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=SKIP, price=close,
                reason=f"regime selector: blocked entry "
                       f"(trend={trend}, vol={vol})",
            )

        sub = self._bull if route == ROUTE_BULL else self._sideways
        sig = sub.signal_for_row(asset, row, False, cfg)
        # Only BUY signals route through; if the sub-strategy says
        # HOLD/SKIP we keep that. SELL while not-in-position is meaningless
        # and the sub will not produce one.
        if sig.action == BUY:
            self._owner[asset] = route
            return _dc_replace(
                sig, reason=f"{sig.reason} [via {route}]",
            )
        return _dc_replace(
            sig, reason=f"{sig.reason} [via {route}]",
        )
