"""Tests for the BTC/ETH relative-value allocator strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.relative_value_signals import (
    REGIME_STATES, STATE_BTC_LEADERSHIP, STATE_DEFENSIVE,
    STATE_ETH_LEADERSHIP, STATE_NEUTRAL, STATE_UNKNOWN,
    STATE_UNSTABLE_ROTATION,
)
from src.strategies.relative_value_btc_eth_allocator import (
    RelativeValueAllocatorConfig, RelativeValueBTCETHAllocatorStrategy,
)


def _signal_row(ts_ms: int, state: str, *,
                  btc_above: float = 1.0, eth_above: float = 1.0,
                  ratio_above_ma_200: float = 1.0,
                  ratio_30d_return: float = 0.0,
                  ratio_90d_return: float = 0.0,
                  ratio_z90: float = 0.0) -> dict:
    return {
        "timestamp": int(ts_ms),
        "regime_state": state,
        "btc_above_200dma": float(btc_above),
        "eth_above_200dma": float(eth_above),
        "ratio_above_ma_200": float(ratio_above_ma_200),
        "ratio_30d_return": float(ratio_30d_return),
        "ratio_90d_return": float(ratio_90d_return),
        "ratio_z90": float(ratio_z90),
    }


def _two_bar_frames(assets, ts0=1700000000000, ts1=1700086400000):
    return {
        a: pd.DataFrame({
            "timestamp": [ts0, ts1],
            "datetime": pd.to_datetime([ts0, ts1], unit="ms", utc=True),
            "open": [100.0, 101.0], "high": [102.0, 103.0],
            "low": [99.0, 100.0], "close": [101.0, 102.0],
            "volume": [10.0, 11.0],
        }) for a in assets
    }


# ---------------------------------------------------------------------------
# Empty / unknown
# ---------------------------------------------------------------------------
def test_empty_signals_yields_empty():
    s = RelativeValueBTCETHAllocatorStrategy(pd.DataFrame())
    assert s.target_weights(1700000000000,
                              _two_bar_frames(["BTC/USDT", "ETH/USDT"])) == {}


def test_unknown_state_yields_cash():
    sigs = pd.DataFrame([_signal_row(1700000000000, STATE_UNKNOWN)])
    s = RelativeValueBTCETHAllocatorStrategy(sigs)
    assert s.target_weights(1700086400000,
                              _two_bar_frames(["BTC/USDT", "ETH/USDT"])) == {}


def test_defensive_yields_cash():
    sigs = pd.DataFrame([
        _signal_row(1700000000000, STATE_DEFENSIVE,
                     btc_above=0.0, eth_above=0.0),
    ])
    s = RelativeValueBTCETHAllocatorStrategy(sigs)
    assert s.target_weights(1700086400000,
                              _two_bar_frames(["BTC/USDT", "ETH/USDT"])) == {}


# ---------------------------------------------------------------------------
# Leadership states
# ---------------------------------------------------------------------------
def test_eth_leadership_with_eth_above_ma_yields_full_eth():
    sigs = pd.DataFrame([
        _signal_row(1700000000000, STATE_ETH_LEADERSHIP,
                     btc_above=1.0, eth_above=1.0),
    ])
    s = RelativeValueBTCETHAllocatorStrategy(sigs)
    out = s.target_weights(1700086400000,
                              _two_bar_frames(["BTC/USDT", "ETH/USDT"]))
    assert out == {"ETH/USDT": pytest.approx(1.0)}


def test_eth_leadership_with_eth_below_ma_yields_50_eth_50_cash():
    sigs = pd.DataFrame([
        _signal_row(1700000000000, STATE_ETH_LEADERSHIP,
                     btc_above=0.0, eth_above=0.0),
    ])
    s = RelativeValueBTCETHAllocatorStrategy(sigs)
    out = s.target_weights(1700086400000,
                              _two_bar_frames(["BTC/USDT", "ETH/USDT"]))
    assert out == {"ETH/USDT": pytest.approx(0.5)}


def test_btc_leadership_with_btc_above_ma_yields_full_btc():
    sigs = pd.DataFrame([
        _signal_row(1700000000000, STATE_BTC_LEADERSHIP,
                     btc_above=1.0, eth_above=0.0),
    ])
    s = RelativeValueBTCETHAllocatorStrategy(sigs)
    out = s.target_weights(1700086400000,
                              _two_bar_frames(["BTC/USDT", "ETH/USDT"]))
    assert out == {"BTC/USDT": pytest.approx(1.0)}


def test_btc_leadership_with_btc_below_ma_yields_50_btc_50_cash():
    sigs = pd.DataFrame([
        _signal_row(1700000000000, STATE_BTC_LEADERSHIP,
                     btc_above=0.0, eth_above=0.0),
    ])
    s = RelativeValueBTCETHAllocatorStrategy(sigs)
    out = s.target_weights(1700086400000,
                              _two_bar_frames(["BTC/USDT", "ETH/USDT"]))
    assert out == {"BTC/USDT": pytest.approx(0.5)}


# ---------------------------------------------------------------------------
# Neutral / unstable
# ---------------------------------------------------------------------------
def test_neutral_with_both_above_ma_splits_50_50():
    sigs = pd.DataFrame([
        _signal_row(1700000000000, STATE_NEUTRAL,
                     btc_above=1.0, eth_above=1.0),
    ])
    s = RelativeValueBTCETHAllocatorStrategy(sigs)
    out = s.target_weights(1700086400000,
                              _two_bar_frames(["BTC/USDT", "ETH/USDT"]))
    assert out == {"BTC/USDT": pytest.approx(0.5),
                    "ETH/USDT": pytest.approx(0.5)}


def test_neutral_with_only_btc_above_ma_yields_full_btc():
    sigs = pd.DataFrame([
        _signal_row(1700000000000, STATE_NEUTRAL,
                     btc_above=1.0, eth_above=0.0),
    ])
    s = RelativeValueBTCETHAllocatorStrategy(sigs)
    out = s.target_weights(1700086400000,
                              _two_bar_frames(["BTC/USDT", "ETH/USDT"]))
    assert out == {"BTC/USDT": pytest.approx(1.0)}


def test_neutral_with_neither_above_ma_yields_cash():
    sigs = pd.DataFrame([
        _signal_row(1700000000000, STATE_NEUTRAL,
                     btc_above=0.0, eth_above=0.0),
    ])
    s = RelativeValueBTCETHAllocatorStrategy(sigs)
    out = s.target_weights(1700086400000,
                              _two_bar_frames(["BTC/USDT", "ETH/USDT"]))
    assert out == {}


def test_unstable_rotation_with_btc_above_ma_yields_50_btc_50_cash():
    sigs = pd.DataFrame([
        _signal_row(1700000000000, STATE_UNSTABLE_ROTATION,
                     btc_above=1.0, eth_above=0.0),
    ])
    s = RelativeValueBTCETHAllocatorStrategy(sigs)
    out = s.target_weights(1700086400000,
                              _two_bar_frames(["BTC/USDT", "ETH/USDT"]))
    assert out == {"BTC/USDT": pytest.approx(0.5)}


def test_unstable_rotation_with_btc_below_ma_yields_cash():
    sigs = pd.DataFrame([
        _signal_row(1700000000000, STATE_UNSTABLE_ROTATION,
                     btc_above=0.0, eth_above=0.0),
    ])
    s = RelativeValueBTCETHAllocatorStrategy(sigs)
    out = s.target_weights(1700086400000,
                              _two_bar_frames(["BTC/USDT", "ETH/USDT"]))
    assert out == {}


# ---------------------------------------------------------------------------
# Long-only / safety invariants
# ---------------------------------------------------------------------------
def test_weights_never_exceed_one():
    """Every state must keep Σ weights ≤ 1, ≥ 0."""
    states = (STATE_ETH_LEADERSHIP, STATE_BTC_LEADERSHIP, STATE_DEFENSIVE,
                STATE_UNSTABLE_ROTATION, STATE_NEUTRAL, STATE_UNKNOWN)
    rows = [_signal_row(1700000000000 + 86400000 * i, st, btc_above=1.0,
                          eth_above=1.0) for i, st in enumerate(states)]
    sigs = pd.DataFrame(rows)
    s = RelativeValueBTCETHAllocatorStrategy(sigs)
    frames = _two_bar_frames(["BTC/USDT", "ETH/USDT"])
    for r in rows:
        # asof a tiny amount after signal timestamp — picks that signal row.
        out = s.target_weights(int(r["timestamp"]) + 1, frames)
        assert all(v >= 0.0 for v in out.values())
        assert sum(out.values()) <= 1.0 + 1e-9


def test_missing_eth_frame_does_not_crash():
    sigs = pd.DataFrame([
        _signal_row(1700000000000, STATE_NEUTRAL,
                     btc_above=1.0, eth_above=1.0),
    ])
    s = RelativeValueBTCETHAllocatorStrategy(sigs)
    # Only BTC frame supplied — neutral state should fall back to BTC-only.
    out = s.target_weights(1700086400000, _two_bar_frames(["BTC/USDT"]))
    assert "ETH/USDT" not in out
    # With ETH missing the neutral both-above branch can't fire, so we
    # land in the BTC-only-above branch -> 100 % BTC.
    assert out == {"BTC/USDT": pytest.approx(1.0)}


def test_missing_btc_frame_eth_leadership_yields_eth():
    sigs = pd.DataFrame([
        _signal_row(1700000000000, STATE_ETH_LEADERSHIP,
                     btc_above=1.0, eth_above=1.0),
    ])
    s = RelativeValueBTCETHAllocatorStrategy(sigs)
    out = s.target_weights(1700086400000, _two_bar_frames(["ETH/USDT"]))
    assert out == {"ETH/USDT": pytest.approx(1.0)}


def test_asof_uses_only_past_signal_rows():
    """If the latest row in `_signals` is in the future, allocator must
    use the most recent row with `timestamp <= asof`."""
    sigs = pd.DataFrame([
        _signal_row(1700000000000, STATE_ETH_LEADERSHIP,
                     btc_above=1.0, eth_above=1.0),
        _signal_row(1800000000000, STATE_DEFENSIVE,
                     btc_above=0.0, eth_above=0.0),
    ])
    s = RelativeValueBTCETHAllocatorStrategy(sigs)
    out = s.target_weights(1700086400000,
                              _two_bar_frames(["BTC/USDT", "ETH/USDT"]))
    assert out == {"ETH/USDT": pytest.approx(1.0)}


def test_diagnostics_returns_state_and_ratio_inputs():
    sigs = pd.DataFrame([
        _signal_row(1700000000000, STATE_ETH_LEADERSHIP,
                     btc_above=1.0, eth_above=1.0,
                     ratio_30d_return=5.0, ratio_90d_return=10.0,
                     ratio_z90=0.4),
    ])
    s = RelativeValueBTCETHAllocatorStrategy(sigs)
    d = s.diagnostics(1700086400000)
    assert d["regime_state"] == STATE_ETH_LEADERSHIP
    assert d["ratio_30d_return"] == 5.0
    assert d["ratio_90d_return"] == 10.0
    assert d["ratio_z90"] == 0.4
