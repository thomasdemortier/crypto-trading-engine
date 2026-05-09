"""
Funding-only portfolio rotation.

Same `target_weights(asof_ts_ms, asset_frames, timeframe)` contract as
`MomentumRotationStrategy` and `DerivativesRotationStrategy`. The
selection rules at each rebalance bar `t`:

  1. Cash filter — if BTC/USDT spot is below its 200d MA at `t`, return {}.
  2. Look up the most-recent funding-signal row at or before `t` for each
     candidate. Rows with `funding_state == 'unknown'` are dropped.
  3. Exclude `crowded_long` rows (long-only, no short, no leverage).
  4. Optionally allow `capitulation` only after price stabilises — i.e.
     the row's 7d return is no worse than `-capitulation_max_drawdown_7d`.
  5. Composite score (per spec):

         score = 0.4 * return_30d
                 + 0.3 * funding_attractiveness
                 + 0.2 * funding_improvement
                 + 0.1 * stabilization_score
                 - crowding_penalty * crowding_penalty_weight

     The weights are fixed by spec — they are NOT tuning knobs. If this
     strategy fails the scorecard, it fails.
  6. Take the Top-N (default 3); equal-weight the survivors.

Lookahead invariant: `_signal_asof` slices the per-symbol signal frame
on `timestamp <= asof_ts_ms` and returns only the last surviving row.
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
class FundingRotationConfig:
    top_n: int = 3
    rebalance_frequency: str = "weekly"
    cash_filter_asset: str = "BTC/USDT"
    cash_filter_ma: int = 200
    min_assets_required: int = 3
    # Composite-score weights — fixed per spec, not tuned.
    momentum_weight: float = 0.4
    funding_attr_weight: float = 0.3
    funding_improve_weight: float = 0.2
    stabilization_weight: float = 0.1
    crowding_penalty_weight: float = 0.5
    # Behavioural toggles.
    exclude_crowded_long: bool = True
    allow_capitulation: bool = True
    capitulation_max_drawdown_7d: float = 0.05  # |7d return| ceiling


class FundingRotationStrategy:
    """Funding-only portfolio strategy."""

    name = "funding_rotation"

    def __init__(self, signals_df: pd.DataFrame,
                 cfg: Optional[FundingRotationConfig] = None) -> None:
        self.cfg = cfg or FundingRotationConfig()
        self._sig_by_symbol: Dict[str, pd.DataFrame] = {}
        if signals_df is not None and not signals_df.empty:
            df = signals_df.copy()
            df["timestamp"] = df["timestamp"].astype("int64")
            for sym, g in df.groupby("symbol"):
                g = g.sort_values("timestamp").reset_index(drop=True)
                # Index by both forms so lookups work whether the asset
                # key is `BTC/USDT` (spot) or `BTCUSDT` (futures).
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
        if r30 is None or pd.isna(r30):
            return None
        attr = float(row.get("funding_attractiveness", 0.0) or 0.0)
        improve = float(row.get("funding_improvement", 0.0) or 0.0)
        stab = float(row.get("stabilization_score", 0.0) or 0.0)
        crowd = float(row.get("crowding_penalty", 0.0) or 0.0)
        c = self.cfg
        return (c.momentum_weight * float(r30)
                + c.funding_attr_weight * attr
                + c.funding_improve_weight * improve
                + c.stabilization_weight * stab
                - c.crowding_penalty_weight * crowd)

    def _row_admissible(self, row: pd.Series) -> bool:
        state = row.get("funding_state", "unknown")
        if state == "unknown":
            return False
        if self.cfg.exclude_crowded_long and state == "crowded_long":
            return False
        if state == "capitulation":
            if not self.cfg.allow_capitulation:
                return False
            r7 = row.get("return_7d")
            if r7 is None or pd.isna(r7):
                return False
            # Allow only when last week stabilised — ban free-falling
            # capitulation prints.
            if float(r7) < -self.cfg.capitulation_max_drawdown_7d:
                return False
        return True

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
            if not self._row_admissible(row):
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
