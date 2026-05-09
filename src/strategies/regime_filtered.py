"""
Regime-filtered strategy wrapper.

Wraps any base strategy with a market-regime gate. The wrapper NEVER
bypasses the risk engine — it only converts BUY/SELL/HOLD/SKIP labels
before they reach the engine.

Default policy:
  * Bear trend  -> block new BUY signals (defensive cash filter).
  * High vol    -> block new BUY signals if `block_buys_in_high_vol`.
  * Sideways    -> pass through unless `block_buys_in_sideways` set.
  * Bull / low vol -> pass through.

SELL signals are NEVER blocked — exits must always be allowed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace as _dc_replace
from typing import Any, Optional

import pandas as pd

from .. import regime as _regime
from .base import Strategy, Signal, BUY, SELL, HOLD, SKIP


@dataclass(frozen=True)
class RegimePolicy:
    block_buys_in_bear: bool = True
    block_buys_in_high_vol: bool = False
    block_buys_in_sideways: bool = False


class RegimeFilteredStrategy(Strategy):
    """Wrap `base_strategy` with a regime-aware signal gate."""

    def __init__(
        self,
        base_strategy: Strategy,
        policy: Optional[RegimePolicy] = None,
        regime_cfg: Optional[_regime.RegimeConfig] = None,
        name_suffix: str = "regime",
    ) -> None:
        if base_strategy is None:
            raise ValueError("base_strategy is required")
        self._base = base_strategy
        self._policy = policy or RegimePolicy()
        self._regime_cfg = regime_cfg or _regime.DEFAULT
        self.name = f"{base_strategy.name}+{name_suffix}"

    # ---- Strategy interface ---------------------------------------------
    @property
    def base(self) -> Strategy:
        return self._base

    def min_history(self, cfg: Any) -> int:
        # Need at least the regime detector's slope window in addition to
        # whatever the base strategy needs.
        return max(
            self._base.min_history(cfg),
            self._regime_cfg.ma_long + self._regime_cfg.slope_window + 5,
        )

    def prepare(self, df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
        out = self._base.prepare(df, cfg)
        if "regime_label" not in out.columns:
            out = _regime.add_regime_columns(out, self._regime_cfg)
        return out

    def signal_for_row(
        self, asset: str, row: pd.Series, in_position: bool, cfg: Any,
    ) -> Signal:
        base_sig = self._base.signal_for_row(asset, row, in_position, cfg)

        # SELL / HOLD / SKIP always pass through — exits matter and we never
        # want to silently keep a position the base strategy wants to close.
        if base_sig.action != BUY:
            return base_sig

        trend = row.get("trend_regime")
        vol = row.get("volatility_regime")
        if pd.isna(trend) or trend == "unknown":
            return base_sig  # warmup — defer to base

        block_reason: Optional[str] = None
        if self._policy.block_buys_in_bear and trend == _regime.BEAR:
            block_reason = "bear_trend"
        elif self._policy.block_buys_in_high_vol and vol == _regime.HIGH_VOL:
            block_reason = "high_volatility"
        elif self._policy.block_buys_in_sideways and trend == _regime.SIDEWAYS:
            block_reason = "sideways"

        if block_reason is None:
            return base_sig
        return _dc_replace(
            base_sig, action=SKIP,
            reason=f"regime block ({block_reason}): base wanted BUY",
        )
