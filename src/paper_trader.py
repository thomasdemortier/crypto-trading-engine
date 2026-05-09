"""
Paper trader.

Pulls the most recent candles for each configured asset, runs the same
indicator + strategy + risk pipeline as the backtester, and updates a
JSON-backed paper portfolio. **Never places live orders** — the only ccxt
calls are public OHLCV fetches via `data_collector.download_symbol`.

State file: `results/paper_state.json`. It contains cash, open positions,
the trade log (appended), and the decisions log (appended).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import config, data_collector, indicators, strategy, utils
from .risk_engine import RiskEngine, Position

logger = utils.get_logger("cte.paper")

STATE_PATH = config.RESULTS_DIR / "paper_state.json"


# ---------------------------------------------------------------------------
# State serialisation
# ---------------------------------------------------------------------------
def _engine_to_state(engine: RiskEngine) -> dict:
    return {
        "cash": engine.cash,
        "positions": {a: asdict(p) for a, p in engine.positions.items()},
        "trades": engine.trades_as_dicts(),
        "decisions": engine.decisions_as_dicts(),
        "rejected_count": engine.rejected_count,
    }


def _state_to_engine(state: dict, cfg: config.RiskConfig) -> RiskEngine:
    engine = RiskEngine(cfg)
    engine.cash = float(state.get("cash", cfg.starting_capital))
    engine.positions = {
        a: Position(**p) for a, p in state.get("positions", {}).items()
    }
    # We intentionally do not re-hydrate trades/decisions into the live
    # engine — they're append-only artifacts that we re-read for display.
    return engine


def load_state(cfg: config.RiskConfig) -> tuple[RiskEngine, dict]:
    utils.ensure_dirs([config.RESULTS_DIR])
    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text())
            return _state_to_engine(state, cfg), state
        except Exception as e:  # noqa: BLE001
            logger.warning("could not parse %s, starting fresh: %s", STATE_PATH, e)
    return RiskEngine(cfg), {"trades": [], "decisions": []}


def save_state(engine: RiskEngine, prior: dict) -> None:
    state = _engine_to_state(engine)
    # Preserve append-only history across runs.
    prior_trades = prior.get("trades", [])
    prior_decisions = prior.get("decisions", [])
    # Only newly appended trades/decisions during this tick are in engine.
    state["trades"] = prior_trades + state["trades"]
    state["decisions"] = prior_decisions + state["decisions"]
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


# ---------------------------------------------------------------------------
# Single tick
# ---------------------------------------------------------------------------
def run_tick(
    timeframe: str = config.DEFAULT_TIMEFRAME,
    assets: Optional[List[str]] = None,
    refresh: bool = True,
) -> Dict[str, dict]:
    """Run one paper-trading evaluation pass over the most recent candle of
    each asset. Returns a per-asset dict of decision metadata."""
    utils.assert_paper_only()
    assets = assets or config.ASSETS
    risk_cfg = config.RISK
    strat_cfg = config.STRATEGY

    engine, prior = load_state(risk_cfg)
    out: Dict[str, dict] = {}

    # Step 1: pull data and indicators for every requested asset, so we can
    # build a complete `marks` dict and stop-loss inputs BEFORE we start
    # mutating engine state. This fixes the "marks dict missing other open
    # positions" bug that under-counted equity in multi-position portfolios.
    asset_frames: Dict[str, "pd.DataFrame"] = {}
    for asset in assets:
        try:
            data_collector.download_symbol(asset, timeframe, refresh=refresh)
            df = data_collector.load_candles(asset, timeframe)
            df = indicators.add_indicators(df, strat_cfg)
        except Exception as e:  # noqa: BLE001
            logger.error("paper tick: data error for %s: %s", asset, e)
            out[asset] = {"error": str(e)}
            continue
        if len(df) < strat_cfg.min_history_candles + 1:
            out[asset] = {"error": "not enough history yet"}
            continue
        asset_frames[asset] = df

    if not asset_frames:
        save_state(engine, prior)
        return out

    # Build full mark/low/open dicts from the most recent closed bar of every
    # asset (also catches any open positions in assets the user didn't pass
    # but still has on file).
    marks: Dict[str, float] = {}
    bar_lows: Dict[str, float] = {}
    bar_opens: Dict[str, float] = {}
    for a, df in asset_frames.items():
        last_closed = df.iloc[-2]
        marks[a] = float(last_closed["close"])
        bar_lows[a] = float(last_closed["low"])
        bar_opens[a] = float(last_closed["open"])

    # Step 2: process stop-losses on the most recent CLOSED bar BEFORE
    # evaluating new signals — matches what the backtester does each bar.
    if engine.positions:
        ref_ts = int(asset_frames[next(iter(asset_frames))].iloc[-2]["timestamp"])
        ref_iso = pd.to_datetime(ref_ts, unit="ms", utc=True).isoformat()
        engine.check_stop_losses(ref_ts, ref_iso, bar_lows, bar_opens, marks)

    # Step 3: evaluate signals. The signal uses the last CLOSED bar; the fill
    # price is the OPEN of the current forming bar (best available proxy).
    for asset, df in asset_frames.items():
        signal_row = df.iloc[-2]    # last fully closed bar
        next_row = df.iloc[-1]      # current/forming bar
        in_position = asset in engine.positions
        sig = strategy.signal_for_row(asset, signal_row, in_position, strat_cfg)
        fill_price = (
            float(next_row["open"]) if pd.notna(next_row["open"])
            else float(signal_row["close"])
        )
        decision = engine.evaluate(sig, fill_price, marks)
        out[asset] = {
            "action": decision.action,
            "accepted": decision.accepted,
            "reason": decision.reason,
            "price": decision.price,
            "size": decision.size,
            "portfolio_value": decision.portfolio_value,
        }
        logger.info("paper %s — %s @ %.2f (%s)",
                    asset, decision.action, decision.price, decision.reason)

    save_state(engine, prior)
    return out


def get_state_for_display() -> dict:
    """Read-only loader used by the dashboard's paper-mode panel."""
    if not STATE_PATH.exists():
        return {"cash": config.RISK.starting_capital, "positions": {},
                "trades": [], "decisions": [], "rejected_count": 0}
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {"cash": config.RISK.starting_capital, "positions": {},
                "trades": [], "decisions": [], "rejected_count": 0}
