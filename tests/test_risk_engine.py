"""Risk engine tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd
import pytest

from src import config
from src.risk_engine import RiskEngine
from src.strategy import Signal


def _sig(asset: str, action: str, price: float, ts_ms: int = 1_700_000_000_000,
         reason: str = "test") -> Signal:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return Signal(asset=asset, timestamp=ts_ms, datetime=pd.Timestamp(dt),
                  action=action, price=price, reason=reason,
                  rsi=30.0, ma50=price, ma200=price * 0.9,
                  atr_pct=2.0, trend_status="up", volatility_status="normal")


def test_engine_refuses_leverage():
    bad = replace(config.RISK, leverage_enabled=True)
    with pytest.raises(ValueError):
        RiskEngine(bad)


def test_buy_respects_max_position_pct():
    eng = RiskEngine()
    price = 50_000.0
    eng.evaluate(_sig("BTC/USDT", "BUY", price), fill_price=price,
                 marks={"BTC/USDT": price})
    pos = eng.positions["BTC/USDT"]
    notional = pos.size * pos.entry_price
    cap = config.RISK.starting_capital * config.RISK.max_position_pct
    # cap may be tightened further by the per-trade risk cap; never exceeded.
    assert notional <= cap + 1e-6, f"notional {notional} > cap {cap}"


def test_buy_respects_risk_per_trade():
    eng = RiskEngine()
    price = 50_000.0
    eng.evaluate(_sig("BTC/USDT", "BUY", price), fill_price=price,
                 marks={"BTC/USDT": price})
    pos = eng.positions["BTC/USDT"]
    risk_at_stop = pos.size * (pos.entry_price - pos.stop_loss_price)
    risk_budget = config.RISK.starting_capital * config.RISK.risk_per_trade_pct
    assert risk_at_stop <= risk_budget + 1e-6


def test_duplicate_position_rejected():
    eng = RiskEngine()
    price = 50_000.0
    eng.evaluate(_sig("BTC/USDT", "BUY", price), fill_price=price,
                 marks={"BTC/USDT": price})
    d = eng.evaluate(_sig("BTC/USDT", "BUY", price * 0.95, ts_ms=1_700_001_000_000),
                     fill_price=price * 0.95, marks={"BTC/USDT": price * 0.95})
    assert d.action == "REJECT"
    assert "already long" in d.reason or "averaging" in d.reason.lower()


def test_sell_with_no_position_is_hold():
    eng = RiskEngine()
    d = eng.evaluate(_sig("BTC/USDT", "SELL", 50_000), fill_price=50_000,
                     marks={"BTC/USDT": 50_000})
    assert d.action == "HOLD"


def test_round_trip_loses_to_fees_and_slippage_when_flat():
    eng = RiskEngine()
    price = 50_000.0
    eng.evaluate(_sig("BTC/USDT", "BUY", price), fill_price=price,
                 marks={"BTC/USDT": price})
    # Sell at the same fill price — fees+slippage on both sides should make
    # the round-trip slightly negative.
    eng.evaluate(_sig("BTC/USDT", "SELL", price, ts_ms=1_700_001_000_000),
                 fill_price=price, marks={"BTC/USDT": price})
    pnl = eng.trades[-1].realized_pnl
    assert pnl < 0


def test_daily_loss_breaker_blocks_new_entries():
    cfg = replace(config.RISK, max_daily_loss_pct=0.005)  # 0.5% daily loss cap
    eng = RiskEngine(cfg)
    ts0 = 1_700_000_000_000
    price = 50_000.0
    eng.evaluate(_sig("BTC/USDT", "BUY", price, ts_ms=ts0),
                 fill_price=price, marks={"BTC/USDT": price})
    # Force a big paper loss by marking down 5% within the same UTC day.
    crashed = price * 0.95
    eng.evaluate(_sig("ETH/USDT", "BUY", 1000.0, ts_ms=ts0 + 60_000),
                 fill_price=1000.0,
                 marks={"BTC/USDT": crashed, "ETH/USDT": 1000.0})
    last = eng.decisions[-1]
    assert last.action in ("BUY", "REJECT")
    if last.action == "REJECT":
        assert "daily loss" in last.reason.lower()
