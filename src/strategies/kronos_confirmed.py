"""
Kronos-confirmed strategy wrapper.

Wraps an existing rule-based strategy with a Kronos confirmation gate.
At every bar, the wrapper:
  1. Asks the BASE strategy what it would do.
  2. For HOLD / SKIP: passes through unchanged (Kronos is never consulted).
  3. For BUY / SELL: looks up a precomputed Kronos confirmation in
     `results/kronos_confirmations.csv` (keyed by (asset, timestamp_ms)).
     - CONFIRM   -> keep base signal, append Kronos reason.
     - REJECT    -> BUY becomes SKIP; SELL becomes HOLD if in position
                    else SKIP.
     - NEUTRAL   -> BUY becomes SKIP; SELL is kept (often risk-driven).
     - missing   -> fallback (default: SKIP for safety; "base" keeps base).

This wrapper NEVER calls Kronos at backtest time — Kronos is offline-only.
It NEVER bypasses the risk engine: every emitted Signal is routed through
`RiskEngine.evaluate()` exactly like every other strategy in the registry.
"""
from __future__ import annotations

from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .. import config, utils
from .base import Strategy, Signal, BUY, SELL, HOLD, SKIP


_DEFAULT_CONFIRMATIONS_PATH = config.RESULTS_DIR / "kronos_confirmations.csv"


class KronosConfirmedStrategy(Strategy):
    """Strategy wrapper. Constructed in code (not via the global registry)
    because it depends on a base strategy instance and an artifact path.
    """

    name = "kronos_confirmed"

    def __init__(
        self,
        base_strategy: Strategy,
        confirmations_path: Optional[Path] = None,
        fallback: str = "skip",  # "skip" or "base"
    ) -> None:
        if base_strategy is None:
            raise ValueError("base_strategy is required")
        if fallback not in ("skip", "base"):
            raise ValueError(
                f"fallback must be 'skip' or 'base', got {fallback!r}"
            )
        self._base = base_strategy
        self._confirmations_path = (
            Path(confirmations_path) if confirmations_path is not None
            else _DEFAULT_CONFIRMATIONS_PATH
        )
        self._fallback = fallback
        self._lookup: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None

    # ---- Strategy interface ------------------------------------------------
    @property
    def base(self) -> Strategy:
        return self._base

    def min_history(self, cfg: Any) -> int:
        return self._base.min_history(cfg)

    def prepare(self, df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
        return self._base.prepare(df, cfg)

    def signal_for_row(
        self, asset: str, row: pd.Series, in_position: bool, cfg: Any,
    ) -> Signal:
        base_sig = self._base.signal_for_row(asset, row, in_position, cfg)

        # Kronos is only consulted for entry/exit — pass HOLD/SKIP through.
        if base_sig.action not in (BUY, SELL):
            return base_sig

        if self._lookup is None:
            self._lookup = self._load_lookup()

        ts = int(row["timestamp"]) if pd.notna(row.get("timestamp")) else 0
        conf = self._lookup.get((asset, ts))

        if conf is None:
            if self._fallback == "base":
                return _dc_replace(
                    base_sig,
                    reason=f"{base_sig.reason} | kronos: no confirmation, fallback=base",
                )
            return _dc_replace(
                base_sig, action=SKIP,
                reason=(f"kronos: no confirmation for "
                        f"{asset}@{ts}; fallback SKIP for safety"),
            )

        verdict = str(conf.get("confirmation", "")).upper()
        fr = float(conf.get("forecast_return_pct", 0.0))
        kronos_tag = (
            f"kronos: {verdict} (forecast_return_pct={fr:+.2f}%)"
        )

        if verdict == "CONFIRM":
            return _dc_replace(
                base_sig, reason=f"{base_sig.reason} | {kronos_tag}",
            )
        if verdict == "REJECT":
            if base_sig.action == BUY:
                return _dc_replace(
                    base_sig, action=SKIP,
                    reason=f"BUY blocked by {kronos_tag}",
                )
            # SELL: keep HOLD if we're in a position (Kronos says price will
            # rise, so don't exit); else SKIP.
            new_action = HOLD if in_position else SKIP
            return _dc_replace(
                base_sig, action=new_action,
                reason=f"SELL blocked by {kronos_tag}",
            )
        if verdict == "NEUTRAL":
            if base_sig.action == BUY:
                return _dc_replace(
                    base_sig, action=SKIP,
                    reason=f"BUY skipped on neutral kronos ({kronos_tag})",
                )
            return _dc_replace(
                base_sig,
                reason=f"{base_sig.reason} | {kronos_tag} (kept SELL)",
            )

        # Unknown verdict (incl. DO_NOT_CALL_KRONOS) — keep original.
        return _dc_replace(
            base_sig, reason=f"{base_sig.reason} | {kronos_tag} (unrecognised)",
        )

    # ---- Internals ---------------------------------------------------------
    def _load_lookup(self) -> Dict[Tuple[str, int], Dict[str, Any]]:
        """Read the saved confirmations CSV into a (asset, timestamp_ms)
        dict. Missing or malformed file => empty dict (the wrapper falls
        back per `self._fallback`)."""
        p = self._confirmations_path
        if not p.exists() or p.stat().st_size == 0:
            return {}
        try:
            df = pd.read_csv(p)
        except Exception:  # noqa: BLE001
            return {}
        required = {"asset", "timestamp_ms", "confirmation"}
        if not required.issubset(df.columns):
            return {}
        out: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for _, r in df.iterrows():
            try:
                key = (str(r["asset"]), int(r["timestamp_ms"]))
            except Exception:  # noqa: BLE001
                continue
            out[key] = {
                "confirmation": r.get("confirmation"),
                "forecast_return_pct": r.get("forecast_return_pct", 0.0),
                "final_signal": r.get("final_signal"),
                "reason": r.get("reason"),
            }
        return out
