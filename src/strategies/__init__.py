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
from .trend_following import TrendFollowingStrategy
from .pullback_continuation import PullbackContinuationStrategy
from .sideways_mean_reversion import SidewaysMeanReversionStrategy
from .regime_selector import RegimeSelectorStrategy
from .placebo_random import PlaceboRandomStrategy

__all__ = [
    "Strategy", "Signal", "BUY", "SELL", "HOLD", "SKIP",
    "RsiMaAtrStrategy", "BuyAndHoldStrategy",
    "MovingAverageCrossStrategy", "BreakoutStrategy",
    "TrendFollowingStrategy", "PullbackContinuationStrategy",
    "SidewaysMeanReversionStrategy", "RegimeSelectorStrategy",
    "PlaceboRandomStrategy",
    "REGISTRY", "BENCHMARKS", "PLACEBOS", "default_strategy",
]


REGISTRY = {
    "rsi_ma_atr": RsiMaAtrStrategy,
    "buy_and_hold": BuyAndHoldStrategy,
    "ma_cross": MovingAverageCrossStrategy,
    "breakout": BreakoutStrategy,
    "trend_following": TrendFollowingStrategy,
    "pullback_continuation": PullbackContinuationStrategy,
    "sideways_mean_reversion": SidewaysMeanReversionStrategy,
    "regime_selector": RegimeSelectorStrategy,
    "placebo_random": PlaceboRandomStrategy,
}

# Names that act as BENCHMARKS, not tradable strategies. The scorecard
# reports them in a separate section and excludes them from PASS / WATCHLIST.
BENCHMARKS = {"buy_and_hold"}

# Names that act as STATISTICAL PLACEBOS — random or null strategies used
# only to validate the research methodology. Cannot receive PASS / WATCHLIST.
# A real strategy's WATCHLIST score must be evaluated against the placebo
# baseline.
PLACEBOS = {"placebo_random"}


def default_strategy() -> Strategy:
    """The repo's incumbent strategy. Kept stable so the existing dashboard
    and CLI behaviour are unchanged."""
    return RsiMaAtrStrategy()
