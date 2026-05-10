"""
Portfolio rebalancing allocator (long-only, fixed-weight, monthly).

This is the project's first **strategy research family** after the
strategy-universe-selection branch concluded that portfolio
rebalancing was the only universe that fit the existing engine
without leverage, shorting, paid data, or broker integration.

Locked configuration (NEVER tuned, never optimised after seeing
results):

    target weights      BTC/USDT = 0.60
                         ETH/USDT = 0.30
                         cash      = 0.10   (the 10 % unallocated weight
                                              the portfolio backtester
                                              already treats as cash; we
                                              do NOT need a synthetic
                                              USDC frame)
    rebalance frequency  monthly

Hard rules:
    * Long-only. Σ weights ≤ 1.0. No leverage. No shorts.
    * No dynamic tuning. No optimiser. No fitting weights to
      historical performance.
    * Lookahead-free: every signal is sliced on
      `timestamp <= asof_ts_ms`.
    * Missing assets: the strategy degrades gracefully — it allocates
      only among the assets present in `asset_frames` and renormalises
      the locked weight ratio of those that survive (BTC and ETH
      always hold the configured ratio; cash bucket stays at the
      locked size).
    * No paper trading enablement. No live trading enablement. No
      broker imports. No API key reads. No order placement strings.

Per-bar contract (matches every other portfolio strategy in the
project):

    target_weights(asof_ts_ms, asset_frames, timeframe)
        -> dict[str, float]

The backtester treats unallocated weight (i.e. anything not returned
in the dict) as cash. Returning `{"BTC/USDT": 0.60, "ETH/USDT": 0.30}`
therefore implicitly leaves 0.10 in cash, which IS the locked stable /
cash bucket of this strategy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class PortfolioRebalancingConfig:
    """Locked configuration. Edits = retuning. Don't."""

    btc_asset: str = "BTC/USDT"
    eth_asset: str = "ETH/USDT"

    # Locked weights — sum to 0.90; remaining 0.10 is cash.
    btc_weight: float = 0.60
    eth_weight: float = 0.30
    cash_weight: float = 0.10  # informational only; the backtester
                                # treats unallocated weight as cash.

    # Locked rebalance frequency — passed to PortfolioBacktestConfig.
    rebalance_frequency: str = "monthly"

    name: str = "portfolio_rebalancing_allocator"


def _has_history(df: Optional[pd.DataFrame], asof_ts_ms: int) -> bool:
    """Asset is usable if it has at least 2 historical bars at or
    before `asof_ts_ms`. This is the same lookahead-free convention
    used by the other portfolio strategies in this project."""
    if df is None or df.empty or "timestamp" not in df.columns:
        return False
    sub = df[df["timestamp"] <= int(asof_ts_ms)]
    return len(sub) >= 2


class PortfolioRebalancingAllocator:
    """Fixed-weight long-only allocator. Returns the locked weight
    vector at every rebalance bar — no signal, no optimiser, no
    fitting."""

    name = "portfolio_rebalancing_allocator"

    def __init__(
        self,
        cfg: Optional[PortfolioRebalancingConfig] = None,
    ) -> None:
        self.cfg = cfg or PortfolioRebalancingConfig()
        # Defensive structural assertions — the locked rules.
        if self.cfg.btc_weight < 0 or self.cfg.eth_weight < 0:
            raise ValueError("portfolio_rebalancing: weights must be "
                              "non-negative (no shorts)")
        total = (self.cfg.btc_weight + self.cfg.eth_weight
                  + self.cfg.cash_weight)
        if total > 1.0 + 1e-9:
            raise ValueError("portfolio_rebalancing: weights must sum "
                              "to <= 1 (no leverage)")
        if self.cfg.rebalance_frequency not in ("weekly", "monthly"):
            raise ValueError("portfolio_rebalancing: rebalance "
                              "frequency must be 'weekly' or 'monthly'")

    # ---- public API --------------------------------------------------------
    def diagnostics(self) -> Dict[str, float]:
        """Static read-out of the locked config — useful for the
        dashboard and the report."""
        return {
            "btc_weight": float(self.cfg.btc_weight),
            "eth_weight": float(self.cfg.eth_weight),
            "cash_weight": float(self.cfg.cash_weight),
            "rebalance_frequency": self.cfg.rebalance_frequency,
        }

    def target_weights(
        self,
        asof_ts_ms: int,
        asset_frames: Dict[str, pd.DataFrame],
        timeframe: str = "1d",
    ) -> Dict[str, float]:
        """Return the locked weight vector among the assets that have
        at least 2 historical bars at or before `asof_ts_ms`. Missing
        assets are dropped. The cash bucket is preserved at the locked
        `cash_weight` size; the missing-asset weight redistributes
        proportionally between the remaining risk legs (BTC / ETH).

        Returns an empty dict if BOTH BTC and ETH are missing — the
        strategy refuses to allocate cash-only because the next
        rebalance bar will see the assets again and it never tries to
        guess returns."""
        btc_df = asset_frames.get(self.cfg.btc_asset)
        eth_df = asset_frames.get(self.cfg.eth_asset)
        btc_ok = _has_history(btc_df, asof_ts_ms)
        eth_ok = _has_history(eth_df, asof_ts_ms)

        risk_total = self.cfg.btc_weight + self.cfg.eth_weight
        if risk_total <= 0:
            return {}

        if btc_ok and eth_ok:
            return {
                self.cfg.btc_asset: float(self.cfg.btc_weight),
                self.cfg.eth_asset: float(self.cfg.eth_weight),
            }
        if btc_ok and not eth_ok:
            # Redistribute ETH's share onto BTC, keep cash bucket.
            return {self.cfg.btc_asset: float(risk_total)}
        if eth_ok and not btc_ok:
            return {self.cfg.eth_asset: float(risk_total)}
        return {}
