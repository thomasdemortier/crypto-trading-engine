"""Tests for the funding+basis carry allocator."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.funding_basis_signals import (
    REGIME_STATES, STATE_CHEAP_BULLISH, STATE_CROWDED_LONG, STATE_DEFENSIVE,
    STATE_NEUTRAL_RISK_ON, STATE_STRESS_NEGATIVE_FUNDING, STATE_UNKNOWN,
)
from src.strategies.funding_basis_carry_allocator import (
    FundingBasisCarryAllocatorStrategy, FundingBasisCarryConfig, _state_cap,
)


def _signal_row(asset: str, ts_ms: int, state: str,
                  carry: float = 0.0,
                  funding_z90: float = 0.0,
                  basis_z90: float = 0.0,
                  above_200dma: float = 1.0,
                  ) -> dict:
    return {
        "timestamp": int(ts_ms), "asset": asset,
        "regime_state": state,
        "carry_attractiveness": float(carry),
        "funding_z90": float(funding_z90),
        "basis_z90": float(basis_z90),
        "above_200dma": float(above_200dma),
        "crowding_score": 0.0,
    }


def _two_bar_frames(assets, ts0=1700000000000, ts1=1700086400000):
    out = {}
    for asset in assets:
        out[asset] = pd.DataFrame({
            "timestamp": [ts0, ts1],
            "datetime": pd.to_datetime([ts0, ts1], unit="ms", utc=True),
            "open": [100.0, 101.0], "high": [102.0, 103.0],
            "low": [99.0, 100.0], "close": [101.0, 102.0],
            "volume": [10.0, 11.0],
        })
    return out


# ---------------------------------------------------------------------------
# State cap lookup
# ---------------------------------------------------------------------------
def test_state_caps_match_locked_defaults():
    cfg = FundingBasisCarryConfig()
    assert _state_cap(STATE_CHEAP_BULLISH, cfg) == 0.80
    assert _state_cap(STATE_NEUTRAL_RISK_ON, cfg) == 0.70
    assert _state_cap(STATE_CROWDED_LONG, cfg) == 0.30
    assert _state_cap(STATE_STRESS_NEGATIVE_FUNDING, cfg) == 0.0
    assert _state_cap(STATE_DEFENSIVE, cfg) == 0.0
    assert _state_cap(STATE_UNKNOWN, cfg) == 0.0


# ---------------------------------------------------------------------------
# Empty / unknown handling
# ---------------------------------------------------------------------------
def test_empty_signals_returns_empty_weights():
    s = FundingBasisCarryAllocatorStrategy(pd.DataFrame())
    out = s.target_weights(1700000000000, _two_bar_frames(["BTC/USDT"]))
    assert out == {}


def test_unknown_state_yields_cash():
    sigs = pd.DataFrame([
        _signal_row("BTC/USDT", 1700000000000, STATE_UNKNOWN),
    ])
    s = FundingBasisCarryAllocatorStrategy(sigs)
    out = s.target_weights(1700086400000, _two_bar_frames(["BTC/USDT"]))
    assert out == {}


# ---------------------------------------------------------------------------
# Long-only / weight-cap invariants
# ---------------------------------------------------------------------------
def test_weights_never_exceed_one():
    sigs = pd.DataFrame([
        _signal_row("BTC/USDT", 1700000000000, STATE_NEUTRAL_RISK_ON,
                     carry=0.5),
        _signal_row("ETH/USDT", 1700000000000, STATE_NEUTRAL_RISK_ON,
                     carry=0.4),
    ])
    s = FundingBasisCarryAllocatorStrategy(sigs)
    out = s.target_weights(
        1700086400000,
        _two_bar_frames(["BTC/USDT", "ETH/USDT"]),
    )
    assert all(v >= 0 for v in out.values())
    assert sum(out.values()) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Allocation rule cases
# ---------------------------------------------------------------------------
def test_two_neutral_assets_split_70_30_by_score():
    """BTC has higher carry, so it gets the 70 % slot."""
    sigs = pd.DataFrame([
        _signal_row("BTC/USDT", 1700000000000, STATE_NEUTRAL_RISK_ON,
                     carry=0.9),
        _signal_row("ETH/USDT", 1700000000000, STATE_NEUTRAL_RISK_ON,
                     carry=0.1),
    ])
    s = FundingBasisCarryAllocatorStrategy(sigs)
    out = s.target_weights(
        1700086400000,
        _two_bar_frames(["BTC/USDT", "ETH/USDT"]),
    )
    assert out["BTC/USDT"] == pytest.approx(0.70)
    assert out["ETH/USDT"] == pytest.approx(0.30)


def test_one_cheap_bullish_asset_gets_80pct():
    sigs = pd.DataFrame([
        _signal_row("BTC/USDT", 1700000000000, STATE_CHEAP_BULLISH,
                     carry=1.0),
        _signal_row("ETH/USDT", 1700000000000, STATE_DEFENSIVE),
    ])
    s = FundingBasisCarryAllocatorStrategy(sigs)
    out = s.target_weights(
        1700086400000,
        _two_bar_frames(["BTC/USDT", "ETH/USDT"]),
    )
    assert out == {"BTC/USDT": pytest.approx(0.80)}


def test_crowded_long_caps_at_30pct_when_no_tradable():
    sigs = pd.DataFrame([
        _signal_row("BTC/USDT", 1700000000000, STATE_CROWDED_LONG,
                     carry=1.0),
        _signal_row("ETH/USDT", 1700000000000, STATE_DEFENSIVE),
    ])
    s = FundingBasisCarryAllocatorStrategy(sigs)
    out = s.target_weights(
        1700086400000,
        _two_bar_frames(["BTC/USDT", "ETH/USDT"]),
    )
    assert out == {"BTC/USDT": pytest.approx(0.30)}


def test_stress_negative_funding_excludes_asset():
    sigs = pd.DataFrame([
        _signal_row("BTC/USDT", 1700000000000,
                     STATE_STRESS_NEGATIVE_FUNDING),
        _signal_row("ETH/USDT", 1700000000000, STATE_NEUTRAL_RISK_ON,
                     carry=0.5),
    ])
    s = FundingBasisCarryAllocatorStrategy(sigs)
    out = s.target_weights(
        1700086400000,
        _two_bar_frames(["BTC/USDT", "ETH/USDT"]),
    )
    # Only ETH is tradable. Single-asset cap = min(0.80, neutral cap=0.70).
    assert out == {"ETH/USDT": pytest.approx(0.70)}


def test_defensive_yields_cash_when_no_other_tradable():
    sigs = pd.DataFrame([
        _signal_row("BTC/USDT", 1700000000000, STATE_DEFENSIVE),
        _signal_row("ETH/USDT", 1700000000000, STATE_DEFENSIVE),
    ])
    s = FundingBasisCarryAllocatorStrategy(sigs)
    out = s.target_weights(
        1700086400000,
        _two_bar_frames(["BTC/USDT", "ETH/USDT"]),
    )
    assert out == {}


def test_missing_asset_frame_does_not_crash():
    sigs = pd.DataFrame([
        _signal_row("BTC/USDT", 1700000000000, STATE_NEUTRAL_RISK_ON,
                     carry=0.5),
        _signal_row("ETH/USDT", 1700000000000, STATE_NEUTRAL_RISK_ON,
                     carry=0.4),
    ])
    s = FundingBasisCarryAllocatorStrategy(sigs)
    # Only BTC frame supplied — ETH should be silently dropped. Single
    # tradable asset under neutral_risk_on caps at 0.70 (its per-state
    # cap, which is below the single-asset top-up of 0.80).
    out = s.target_weights(
        1700086400000,
        _two_bar_frames(["BTC/USDT"]),
    )
    assert "ETH/USDT" not in out
    assert out["BTC/USDT"] == pytest.approx(0.70)


def test_asof_uses_only_past_signal_rows():
    """If the latest row in `_signals` is in the future relative to asof,
    the allocator must use the most recent past row instead."""
    sigs = pd.DataFrame([
        _signal_row("BTC/USDT", 1700000000000, STATE_NEUTRAL_RISK_ON,
                     carry=0.5),
        _signal_row("BTC/USDT", 1800000000000, STATE_DEFENSIVE),
    ])
    s = FundingBasisCarryAllocatorStrategy(sigs)
    out = s.target_weights(
        1700086400000,
        _two_bar_frames(["BTC/USDT"]),
    )
    # Defensive row is in the future; allocator must see neutral_risk_on
    # → single-asset cap of 0.70.
    assert out == {"BTC/USDT": pytest.approx(0.70)}
