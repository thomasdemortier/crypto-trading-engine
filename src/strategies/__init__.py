"""Strategy plug-ins.

Each strategy is a self-contained signal factory. Every strategy goes through
the SAME risk engine (fees, slippage, position sizing, daily-loss circuit
breaker, stop-loss). Strategies cannot bypass risk controls.
"""
from .base import Strategy, Signal, BUY, SELL, HOLD, SKIP
from .rsi_ma_atr import RsiMaAtrStrategy
from .buy_and_hold import BuyAndHoldStrategy
from .moving_average_cross import MovingAverageCrossStrategy
from .breakout import BreakoutStrategy

__all__ = [
    "Strategy", "Signal", "BUY", "SELL", "HOLD", "SKIP",
    "RsiMaAtrStrategy", "BuyAndHoldStrategy",
    "MovingAverageCrossStrategy", "BreakoutStrategy",
    "REGISTRY", "default_strategy",
]


REGISTRY = {
    "rsi_ma_atr": RsiMaAtrStrategy,
    "buy_and_hold": BuyAndHoldStrategy,
    "ma_cross": MovingAverageCrossStrategy,
    "breakout": BreakoutStrategy,
}


def default_strategy() -> Strategy:
    """The repo's incumbent strategy. Kept stable so the existing dashboard
    and CLI behaviour are unchanged."""
    return RsiMaAtrStrategy()
