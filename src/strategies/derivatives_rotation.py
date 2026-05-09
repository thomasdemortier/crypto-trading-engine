"""
Derivatives-aware portfolio rotation.

This is a multi-asset, long-only portfolio strategy in the same family
as `momentum_rotation.MomentumRotationStrategy`, but every selection
decision is filtered through the precomputed derivatives-signal table
from `derivatives_signals.compute_all_derivatives_signals`.

Selection rules at each rebalance bar `t` (only data with timestamp
`<= t` is consulted):

  1. Cash filter — if BTC/USDT spot is below its 200d MA at `t`, return {}.
  2. Look up the most recent signal row for each candidate symbol at or
     before `t`. Symbols with no signal row yet, or whose latest signal
     state is `'unknown'`, are dropped from the candidate pool.
  3. Exclude any symbol whose latest `signal_state == 'crowded_long'`.
  4. Score the remaining candidates with a composite that prefers healthy
     trends and penalises crowding:

        combined = return_30d
                   - crowding_penalty * crowding_score
                   + healthy_bonus    * (1 if signal_state == 'healthy_trend' else 0)
                   - capit_penalty    * (1 if signal_state == 'capitulation' else 0)

     The weights are fixed by spec — they are NOT tuned to chase a PASS
     verdict. If this strategy fails the scorecard, that's the verdict.
  5. Take the Top-N (default 3), equal-weight the survivors.

Lookahead invariant: `target_weights` only ever calls `_signal_asof`,
which performs an `index <= asof` slice and returns the *last* row.
There is no forward fill, no future leak.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


def _no_slash(symbol: str) -> str:
    return symbol.replace("/", "")


def _bars_per_day(timeframe: str) -> int:
    return {"1h": 24, "4h": 6, "1d": 1}.get(timeframe, 1)


@dataclass(frozen=True)
class DerivativesRotationConfig:
    top_n: int = 3
    rebalance_frequency: str = "weekly"   # informational only
    momentum_window_days: int = 30
    cash_filter_asset: str = "BTC/USDT"
    cash_filter_ma: int = 200
    min_assets_required: int = 3
    # Composite-score knobs — fixed by spec, not tuning knobs.
    crowding_penalty: float = 0.5
    healthy_bonus: float = 0.05
    capit_penalty: float = 0.1
    exclude_crowded_long: bool = True


class DerivativesRotationStrategy:
    """Portfolio strategy filtered by precomputed derivatives signals."""

    name = "derivatives_rotation"

    def __init__(self, signals_df: pd.DataFrame,
                 cfg: Optional[DerivativesRotationConfig] = None) -> None:
        self.cfg = cfg or DerivativesRotationConfig()
        self._sig_by_symbol: Dict[str, pd.DataFrame] = {}
        if signals_df is not None and not signals_df.empty:
            df = signals_df.copy()
            df["timestamp"] = df["timestamp"].astype("int64")
            for sym, g in df.groupby("symbol"):
                g = g.sort_values("timestamp").reset_index(drop=True)
                self._sig_by_symbol[sym] = g
                self._sig_by_symbol[_no_slash(sym)] = g

    # ---- helpers ----------------------------------------------------------
    def _signal_asof(self, sym: str, asof_ts_ms: int) -> Optional[pd.Series]:
        sig = (self._sig_by_symbol.get(sym)
               or self._sig_by_symbol.get(_no_slash(sym)))
        if sig is None or sig.empty:
            return None
        sub = sig[sig["timestamp"] <= int(asof_ts_ms)]
        if sub.empty:
            return None
        return sub.iloc[-1]

    def _cash_filter_bearish(self, asset_frames: Dict[str, pd.DataFrame],
                             asof_ts_ms: int, bars_per_day: int) -> bool:
        df = asset_frames.get(self.cfg.cash_filter_asset)
        if df is None or df.empty:
            return False
        sub = df[df["timestamp"] <= int(asof_ts_ms)]
        ma_bars = self.cfg.cash_filter_ma * bars_per_day
        if len(sub) < ma_bars + 1:
            return False
        last = float(sub["close"].iloc[-1])
        ma = float(sub["close"].iloc[-ma_bars:].mean())
        return last < ma

    def _composite_score(self, row: pd.Series) -> Optional[float]:
        r30 = row.get("return_30d")
        crowding = float(row.get("crowding_score", 0.0) or 0.0)
        state = row.get("signal_state", "unknown")
        if r30 is None or pd.isna(r30):
            return None
        score = float(r30) - self.cfg.crowding_penalty * crowding
        if state == "healthy_trend":
            score += self.cfg.healthy_bonus
        elif state == "capitulation":
            score -= self.cfg.capit_penalty
        return score

    # ---- public API -------------------------------------------------------
    def target_weights(
        self,
        asof_ts_ms: int,
        asset_frames: Dict[str, pd.DataFrame],
        timeframe: str = "1d",
    ) -> Dict[str, float]:
        bars_per_day = _bars_per_day(timeframe)
        if self._cash_filter_bearish(asset_frames, asof_ts_ms, bars_per_day):
            return {}
        scores: Dict[str, float] = {}
        for asset in asset_frames.keys():
            row = self._signal_asof(asset, asof_ts_ms)
            if row is None:
                continue
            state = row.get("signal_state", "unknown")
            if state == "unknown":
                continue
            if self.cfg.exclude_crowded_long and state == "crowded_long":
                continue
            s = self._composite_score(row)
            if s is None:
                continue
            scores[asset] = s
        if len(scores) < self.cfg.min_assets_required:
            return {}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[: self.cfg.top_n]
        n = len(top)
        if n == 0:
            return {}
        w = 1.0 / n
        return {asset: w for asset, _ in top}
