"""
Risk engine: the only thing that can authorize a simulated fill.

Responsibilities
----------------
* Position sizing — uses the *more conservative* of:
    (a) max_position_pct of equity, and
    (b) the size implied by risk_per_trade_pct given the configured stop_loss_pct.
* Hard caps — never approve a trade that exceeds either cap, never open a
  duplicate position for an asset already held, never average down (unless
  RiskConfig.averaging_down_enabled is True, which it is not by default).
* Daily loss circuit breaker — once realised+unrealised drawdown for the
  current UTC date exceeds max_daily_loss_pct, all NEW entries are rejected
  for the remainder of the day. Existing exits are still allowed.
* Cash, fees, slippage — applied symmetrically on entry and exit.
* Audit log — every approve/reject decision is captured with reason.

The engine is fully deterministic and stateful. The backtester drives it bar
by bar; the paper trader drives it on the latest candle.

Hard refusals (raise immediately rather than silently no-op):
* leverage_enabled, margin_enabled, shorting_enabled — these MUST stay False
  in v1. If anything flips them, we refuse to construct the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

from . import config, utils

logger = utils.get_logger("cte.risk")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Position:
    asset: str
    entry_price: float        # the executed entry price (post-slippage)
    size: float               # units held (e.g. BTC count)
    cost_basis: float         # cash actually paid, including fees
    stop_loss_price: float    # absolute price level
    opened_at_ts: int         # ms epoch
    opened_at_iso: str

    @property
    def notional(self) -> float:
        """Last-known notional uses cost_basis as a fallback when no mark is set.
        For mark-to-market we expose `unrealized_pnl(current_price)`."""
        return self.cost_basis

    def unrealized_pnl(self, current_price: float) -> float:
        gross = self.size * current_price - self.size * self.entry_price
        # fees on entry are already deducted from cash; we don't double-count
        return gross


@dataclass
class TradeRecord:
    timestamp_ms: int
    datetime_iso: str
    asset: str
    side: str               # 'BUY' or 'SELL'
    price: float            # executed price (after slippage)
    size: float             # units
    notional: float         # price * size (pre-fee)
    fee: float              # absolute fee in USDT
    slippage_cost: float    # absolute slippage cost in USDT
    realized_pnl: float     # 0 for BUY entries; PnL on SELL exits
    portfolio_value: float  # equity AFTER this trade
    reason: str             # plain-language reason


@dataclass
class Decision:
    timestamp_ms: int
    datetime_iso: str
    asset: str
    action: str             # BUY / SELL / HOLD / SKIP / REJECT
    accepted: bool
    reason: str
    price: float
    size: float
    portfolio_value: float


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class RiskEngine:
    def __init__(self, cfg: config.RiskConfig | None = None) -> None:
        utils.assert_paper_only()
        self.cfg = cfg or config.RISK
        if self.cfg.leverage_enabled or self.cfg.margin_enabled or self.cfg.shorting_enabled:
            raise ValueError(
                "RiskEngine refuses to start: leverage/margin/short flags must "
                "all be False in version 1."
            )
        self.cash: float = float(self.cfg.starting_capital)
        self.positions: Dict[str, Position] = {}
        self.trades: List[TradeRecord] = []
        self.decisions: List[Decision] = []

        # Daily loss tracking — keyed by UTC date
        self._day_start_equity: Dict[date, float] = {}
        self._risk_off_days: set = set()

        # Audit counters
        self.rejected_count: int = 0

        logger.info(
            "RiskEngine ready: cash=%.2f, max_position=%.0f%%, risk/trade=%.0f%%, "
            "max_daily_loss=%.0f%%, fee=%.3f%%, slippage=%.3f%%",
            self.cash,
            self.cfg.max_position_pct * 100,
            self.cfg.risk_per_trade_pct * 100,
            self.cfg.max_daily_loss_pct * 100,
            self.cfg.fee_pct * 100,
            self.cfg.slippage_pct * 100,
        )

    # ---- Equity and exposure ----------------------------------------------
    def equity(self, marks: Dict[str, float]) -> float:
        """Total portfolio value: cash + sum(position * current_mark)."""
        e = self.cash
        for asset, pos in self.positions.items():
            mark = marks.get(asset, pos.entry_price)
            e += pos.size * mark
        return e

    def exposure(self, marks: Dict[str, float]) -> float:
        return sum(pos.size * marks.get(asset, pos.entry_price)
                   for asset, pos in self.positions.items())

    # ---- Daily loss circuit breaker ---------------------------------------
    def _today_utc(self, ts_ms: int) -> date:
        return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date()

    def _update_daily_state(self, ts_ms: int, marks: Dict[str, float]) -> bool:
        """Snapshot day-start equity; flip risk-off if drawdown exceeded.
        Returns True if currently in risk-off mode for this UTC day."""
        today = self._today_utc(ts_ms)
        equity = self.equity(marks)
        if today not in self._day_start_equity:
            self._day_start_equity[today] = equity
        start = self._day_start_equity[today]
        if start > 0:
            day_pl_pct = (equity - start) / start
            if day_pl_pct <= -self.cfg.max_daily_loss_pct:
                self._risk_off_days.add(today)
        return today in self._risk_off_days

    # ---- Position sizing --------------------------------------------------
    def _size_position(self, equity: float, price: float) -> float:
        """Return a units count that respects BOTH the max-position cap and
        the per-trade risk cap. If the result is zero or negative, the caller
        should reject the trade."""
        if price <= 0 or equity <= 0:
            return 0.0
        cap_by_position = (equity * self.cfg.max_position_pct) / price
        # Risk = size * stop_distance ; we want size <= risk_budget / stop_distance
        stop_distance = price * self.cfg.stop_loss_pct
        if stop_distance <= 0:
            return 0.0
        cap_by_risk = (equity * self.cfg.risk_per_trade_pct) / stop_distance
        return max(min(cap_by_position, cap_by_risk), 0.0)

    # ---- Public API: feed a strategy signal -------------------------------
    def evaluate(
        self,
        signal,                 # strategy.Signal — kept untyped to avoid cycles
        fill_price: float,      # the candle open we'll execute at
        marks: Dict[str, float] # asset → mark price for equity calc
    ):
        """Decide what (if anything) to do with this signal.

        Returns the Decision; if it was accepted and produced a fill, also
        appends a TradeRecord. The caller (backtester / paper trader) inspects
        engine.trades[-1] when it needs the trade details.
        """
        utils.assert_paper_only()
        ts = signal.timestamp
        iso = signal.datetime.isoformat() if signal.datetime is not None else ""
        asset = signal.asset

        # Always update daily state first so risk-off blocks new entries.
        risk_off_today = self._update_daily_state(ts, marks)
        eq = self.equity(marks)

        # HOLD / SKIP do not trigger any state change — log and return.
        if signal.action in ("HOLD", "SKIP"):
            d = Decision(ts, iso, asset, signal.action, accepted=True,
                         reason=signal.reason, price=signal.price, size=0.0,
                         portfolio_value=eq)
            self.decisions.append(d)
            return d

        if signal.action == "BUY":
            return self._handle_buy(signal, fill_price, eq, marks, risk_off_today)

        if signal.action == "SELL":
            return self._handle_sell(signal, fill_price, eq, marks)

        # Unknown action — defensive.
        d = Decision(ts, iso, asset, "REJECT", accepted=False,
                     reason=f"unknown action {signal.action!r}",
                     price=signal.price, size=0.0, portfolio_value=eq)
        self.decisions.append(d)
        self.rejected_count += 1
        return d

    # ---- Internals: BUY / SELL --------------------------------------------
    def _handle_buy(self, sig, fill_price, equity_before, marks, risk_off_today):
        ts, iso, asset = sig.timestamp, (sig.datetime.isoformat() if sig.datetime is not None else ""), sig.asset

        if risk_off_today:
            return self._reject(ts, iso, asset, fill_price, 0.0, equity_before,
                                "daily loss limit breached — risk off for today")

        if asset in self.positions:
            if not self.cfg.averaging_down_enabled:
                return self._reject(ts, iso, asset, fill_price, 0.0, equity_before,
                                    f"already long {asset}; averaging-down disabled")
            # (averaging down logic intentionally not implemented in v1)

        # Apply slippage first so sizing is honest about the price we will
        # actually pay. This guarantees `size * slipped_price <= cap`.
        slipped_price = fill_price * (1.0 + self.cfg.slippage_pct)
        size = self._size_position(equity_before, slipped_price)
        if size <= 0:
            return self._reject(ts, iso, asset, slipped_price, 0.0, equity_before,
                                "computed position size is zero")

        notional = size * slipped_price
        fee = notional * self.cfg.fee_pct
        slippage_cost = size * (slipped_price - fill_price)
        cash_needed = notional + fee

        if cash_needed > self.cash:
            return self._reject(ts, iso, asset, slipped_price, size, equity_before,
                                f"insufficient cash: need {cash_needed:.2f}, have {self.cash:.2f}")

        # Final cap check (paranoia) against equity_before, since notional was
        # sized exactly against that. Tolerance is generous to absorb fp noise.
        if (notional / max(equity_before, 1e-9)) > (self.cfg.max_position_pct + 1e-6):
            return self._reject(ts, iso, asset, slipped_price, size, equity_before,
                                f"position cap breach: {notional/equity_before*100:.4f}%")

        self.cash -= cash_needed
        self.positions[asset] = Position(
            asset=asset,
            entry_price=slipped_price,
            size=size,
            cost_basis=cash_needed,
            stop_loss_price=slipped_price * (1.0 - self.cfg.stop_loss_pct),
            opened_at_ts=ts,
            opened_at_iso=iso,
        )

        eq_after = self.equity({**marks, asset: slipped_price})
        self.trades.append(TradeRecord(
            timestamp_ms=ts, datetime_iso=iso, asset=asset, side="BUY",
            price=slipped_price, size=size, notional=notional,
            fee=fee, slippage_cost=slippage_cost, realized_pnl=0.0,
            portfolio_value=eq_after, reason=sig.reason,
        ))
        d = Decision(ts, iso, asset, "BUY", accepted=True, reason=sig.reason,
                     price=slipped_price, size=size, portfolio_value=eq_after)
        self.decisions.append(d)
        logger.info("BUY  %s size=%.6f @ %.2f (eq=%.2f) — %s",
                    asset, size, slipped_price, eq_after, sig.reason)
        return d

    def _handle_sell(self, sig, fill_price, equity_before, marks):
        ts, iso, asset = sig.timestamp, (sig.datetime.isoformat() if sig.datetime is not None else ""), sig.asset
        if asset not in self.positions:
            # Nothing to sell — record as HOLD with explanation.
            d = Decision(ts, iso, asset, "HOLD", accepted=True,
                         reason=f"sell signal but no open {asset} position",
                         price=fill_price, size=0.0, portfolio_value=equity_before)
            self.decisions.append(d)
            return d

        pos = self.positions[asset]
        # Slippage on exit hurts (we sell at a slightly worse price)
        slipped_price = fill_price * (1.0 - self.cfg.slippage_pct)
        gross = pos.size * slipped_price
        fee = gross * self.cfg.fee_pct
        slippage_cost = pos.size * (fill_price - slipped_price)
        proceeds = gross - fee
        realized = proceeds - pos.cost_basis

        self.cash += proceeds
        del self.positions[asset]
        eq_after = self.equity(marks)
        self.trades.append(TradeRecord(
            timestamp_ms=ts, datetime_iso=iso, asset=asset, side="SELL",
            price=slipped_price, size=pos.size, notional=gross,
            fee=fee, slippage_cost=slippage_cost, realized_pnl=realized,
            portfolio_value=eq_after, reason=sig.reason,
        ))
        d = Decision(ts, iso, asset, "SELL", accepted=True, reason=sig.reason,
                     price=slipped_price, size=pos.size, portfolio_value=eq_after)
        self.decisions.append(d)
        logger.info("SELL %s size=%.6f @ %.2f pnl=%.2f (eq=%.2f) — %s",
                    asset, pos.size, slipped_price, realized, eq_after, sig.reason)
        return d

    # ---- Stop-loss check (called by backtester each bar) ------------------
    def check_stop_losses(self, ts_ms: int, dt_iso: str,
                          bar_lows: Dict[str, float],
                          bar_opens: Dict[str, float],
                          marks: Dict[str, float]) -> List[Decision]:
        """Honestly model stop-loss execution.

        - If the bar GAPPED through the stop (open below stop), the realistic
          fill is at the open, NOT at the stop price. Returning to the stop
          price would be optimistic.
        - If the bar opened above the stop and traded through it intra-bar
          (low <= stop), we fill at the stop price. This matches the typical
          assumption for liquid spot markets.

        Returns the list of stop-out decisions issued this bar.
        """
        decisions: List[Decision] = []
        for asset in list(self.positions.keys()):
            pos = self.positions[asset]
            low = bar_lows.get(asset)
            bar_open = bar_opens.get(asset)
            if low is None:
                continue
            stop = pos.stop_loss_price
            triggered = (low <= stop) or (bar_open is not None and bar_open <= stop)
            if not triggered:
                continue
            # Pessimistic fill on gap-down: the worse of (stop, open).
            if bar_open is not None and bar_open < stop:
                fill_price = bar_open
                reason = (f"stop-loss gap-down: open {bar_open:.2f} "
                          f"below stop {stop:.2f}")
            else:
                fill_price = stop
                reason = f"stop-loss hit at {stop:.2f}"

            from .strategy import Signal as Sig  # local import to avoid cycles
            from datetime import datetime as _dt
            dummy = Sig(
                asset=asset, timestamp=ts_ms,
                datetime=_dt.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
                action="SELL", price=fill_price, reason=reason,
                rsi=float("nan"), ma50=float("nan"), ma200=float("nan"),
                atr_pct=float("nan"), trend_status="unknown",
                volatility_status="unknown",
            )
            dec = self._handle_sell(dummy, fill_price,
                                    self.equity(marks), marks)
            decisions.append(dec)
        return decisions

    # ---- Helpers ----------------------------------------------------------
    def _reject(self, ts, iso, asset, price, size, eq, reason: str) -> Decision:
        self.rejected_count += 1
        d = Decision(ts, iso, asset, "REJECT", accepted=False, reason=reason,
                     price=price, size=size, portfolio_value=eq)
        self.decisions.append(d)
        logger.info("REJECT %s @ %.2f — %s", asset, price, reason)
        return d

    # ---- Serialisation ----------------------------------------------------
    def trades_as_dicts(self) -> List[dict]:
        return [asdict(t) for t in self.trades]

    def decisions_as_dicts(self) -> List[dict]:
        return [asdict(d) for d in self.decisions]
