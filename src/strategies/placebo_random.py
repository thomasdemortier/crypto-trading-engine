"""
Placebo random strategy — STATISTICAL CONTROL ONLY.

Generates random BUY/SELL signals using a fixed-seed PRNG. The point is
to provide a baseline against which real strategies must be compared:
if a "real" strategy's OOS stability is no better than placebo's, the
strategy has no signal.

Hard rules:
  * Long-only, no leverage, no shorts.
  * Same risk engine path, same fees, same slippage, same next-bar-open
    fills as every other strategy.
  * Fixed seed -> reproducible signal stream. Same data + same seed
    always produce the same trades.
  * NEVER tradable. Cannot receive PASS / WATCHLIST verdicts; the
    scorecard short-circuits to a "PLACEBO" verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from .base import Strategy, Signal, BUY, SELL, HOLD, SKIP


@dataclass(frozen=True)
class PlaceboRandomConfig:
    seed: int = 42
    entry_prob_per_bar: float = 0.02   # ~2% chance to BUY when flat
    exit_prob_per_bar: float = 0.10    # ~10% chance to SELL when in position


class PlaceboRandomStrategy(Strategy):
    name = "placebo_random"

    def __init__(self, cfg: Optional[PlaceboRandomConfig] = None) -> None:
        self.cfg = cfg or PlaceboRandomConfig()
        self._rng: Optional[np.random.Generator] = None

    def min_history(self, cfg: Any) -> int:
        return 2

    def prepare(self, df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
        # Reset RNG every time the backtester re-prepares the data so that
        # the same (data, seed) always produces the same actions.
        self._rng = np.random.default_rng(self.cfg.seed)
        return df.copy()

    def signal_for_row(
        self, asset: str, row: pd.Series, in_position: bool, cfg: Any,
    ) -> Signal:
        if self._rng is None:
            self._rng = np.random.default_rng(self.cfg.seed)
        ts = int(row["timestamp"]) if pd.notna(row.get("timestamp")) else 0
        close = float(row["close"])
        u = float(self._rng.random())

        if in_position:
            if u < self.cfg.exit_prob_per_bar:
                return Signal(
                    asset=asset, timestamp=ts, datetime=row.get("datetime"),
                    action=SELL, price=close,
                    reason=f"placebo: random exit (u={u:.4f})",
                )
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=HOLD, price=close,
                reason="placebo: hold (random)",
            )
        if u < self.cfg.entry_prob_per_bar:
            return Signal(
                asset=asset, timestamp=ts, datetime=row.get("datetime"),
                action=BUY, price=close,
                reason=f"placebo: random entry (u={u:.4f})",
            )
        return Signal(
            asset=asset, timestamp=ts, datetime=row.get("datetime"),
            action=HOLD, price=close,
            reason="placebo: skip (random)",
        )
