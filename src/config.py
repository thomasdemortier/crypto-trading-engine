"""
Single source of truth for configuration.

Hard safety lock: LIVE_TRADING_ENABLED is False and there is no execution module
in version 1. Every module imports `LIVE_TRADING_ENABLED` and calls
`assert_paper_only()` (see utils.py) before doing anything that touches state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# HARD SAFETY LOCK — DO NOT TOGGLE WITHOUT WRITING A SEPARATE EXECUTION MODULE
# ---------------------------------------------------------------------------
LIVE_TRADING_ENABLED: bool = False


# ---------------------------------------------------------------------------
# Paths (all relative to the repo root — no absolute Mac paths)
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_RAW_DIR: Path = REPO_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = REPO_ROOT / "data" / "processed"
LOGS_DIR: Path = REPO_ROOT / "logs"
RESULTS_DIR: Path = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# Market / exchange settings (PUBLIC DATA ONLY — no API keys ever)
# ---------------------------------------------------------------------------
PRIMARY_EXCHANGE: str = "binance"
FALLBACK_EXCHANGE: str = "kraken"
ASSETS: List[str] = ["BTC/USDT", "ETH/USDT"]
# Larger universe used by the portfolio momentum rotation research module
# (NOT used by the single-asset strategies). The download command will
# silently skip any symbol the exchange does not list, and the portfolio
# backtester reports availability in `data_coverage.csv`.
EXPANDED_UNIVERSE: List[str] = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT",
    "XRP/USDT", "DOGE/USDT", "ADA/USDT", "LTC/USDT", "BNB/USDT",
]
TIMEFRAMES: List[str] = ["1h", "4h", "1d"]
DEFAULT_TIMEFRAME: str = "4h"

# Kraken uses non-standard names for some pairs; ccxt unifies most of them but
# some pairs are quoted in USD rather than USDT on Kraken. The collector falls
# back to a USD-quoted equivalent when a USDT pair is missing.
KRAKEN_USDT_TO_USD_FALLBACK: bool = True

# How much history to download by default (days back from "now")
DEFAULT_HISTORY_DAYS: int = 1460  # ~4 years — required for the 90/30 walk-forward

# Per-exchange pagination chunk size. Kraken caps near 720 of the most-recent
# bars regardless of `since`, so true backwards history requires Binance
# (which supports `startTime` paging at 1000 candles per call).
FETCH_CHUNK_LIMITS: dict = {
    "binance": 1000,
    "kraken": 720,
}
# Back-compat alias for older call sites.
FETCH_CHUNK_LIMIT: int = 1000


# ---------------------------------------------------------------------------
# Strategy parameters (deterministic — no ML, no parameter optimization)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyConfig:
    rsi_period: int = 14
    rsi_buy_threshold: float = 35.0
    rsi_sell_threshold: float = 65.0
    ma_short: int = 50
    ma_long: int = 200
    atr_period: int = 14
    atr_pct_max: float = 5.0  # skip new entries if ATR% > this
    volume_ma_period: int = 20
    min_history_candles: int = 220  # need MA200 plus a buffer


# ---------------------------------------------------------------------------
# Risk parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskConfig:
    starting_capital: float = 10_000.0
    base_currency: str = "USDT"
    max_position_pct: float = 0.05    # max 5% of portfolio per asset
    risk_per_trade_pct: float = 0.01  # 1% of portfolio at risk per trade
    max_daily_loss_pct: float = 0.02  # 2% — risk-off for the rest of the day
    fee_pct: float = 0.0010           # 0.10% per side
    slippage_pct: float = 0.0005      # 0.05% per side
    leverage_enabled: bool = False
    margin_enabled: bool = False
    shorting_enabled: bool = False
    averaging_down_enabled: bool = False
    # Stop-loss distance is derived: max_position_value * stop_dist_pct = risk_amount
    # We expose the explicit stop distance instead so the risk engine can size
    # the position to satisfy *both* the position cap and the risk cap.
    stop_loss_pct: float = 0.05  # 5% below entry — see risk_engine docstring


# ---------------------------------------------------------------------------
# Backtester settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BacktestConfig:
    # Use NEXT bar open as fill price. This is the only way to be honest about
    # lookahead bias. Setting this to True would let the backtester fill at the
    # signalling bar's close — convenient but optimistic.
    fill_on_signal_close: bool = False


STRATEGY = StrategyConfig()
RISK = RiskConfig()
BACKTEST = BacktestConfig()


def summary() -> dict:
    """Return a flat dict of the active config for the dashboard's debug panel."""
    return {
        "LIVE_TRADING_ENABLED": LIVE_TRADING_ENABLED,
        "primary_exchange": PRIMARY_EXCHANGE,
        "fallback_exchange": FALLBACK_EXCHANGE,
        "assets": ASSETS,
        "timeframes": TIMEFRAMES,
        "default_timeframe": DEFAULT_TIMEFRAME,
        "starting_capital": RISK.starting_capital,
        "max_position_pct": RISK.max_position_pct,
        "risk_per_trade_pct": RISK.risk_per_trade_pct,
        "max_daily_loss_pct": RISK.max_daily_loss_pct,
        "fee_pct": RISK.fee_pct,
        "slippage_pct": RISK.slippage_pct,
        "stop_loss_pct": RISK.stop_loss_pct,
        "rsi_buy_threshold": STRATEGY.rsi_buy_threshold,
        "rsi_sell_threshold": STRATEGY.rsi_sell_threshold,
        "ma_short": STRATEGY.ma_short,
        "ma_long": STRATEGY.ma_long,
        "atr_pct_max": STRATEGY.atr_pct_max,
        "fill_on_signal_close": BACKTEST.fill_on_signal_close,
    }
