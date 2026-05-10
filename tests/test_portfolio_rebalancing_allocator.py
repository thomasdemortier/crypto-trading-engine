"""Tests for `src/strategies/portfolio_rebalancing_allocator.py`."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import safety_lock
from src.strategies.portfolio_rebalancing_allocator import (
    PortfolioRebalancingAllocator, PortfolioRebalancingConfig,
)


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
# Config / defaults
# ---------------------------------------------------------------------------
def test_default_config_locked():
    cfg = PortfolioRebalancingConfig()
    assert cfg.btc_weight == 0.60
    assert cfg.eth_weight == 0.30
    assert cfg.cash_weight == 0.10
    assert cfg.btc_asset == "BTC/USDT"
    assert cfg.eth_asset == "ETH/USDT"
    assert cfg.rebalance_frequency == "monthly"


def test_weights_sum_at_most_one():
    cfg = PortfolioRebalancingConfig()
    total = cfg.btc_weight + cfg.eth_weight + cfg.cash_weight
    assert total <= 1.0 + 1e-9
    assert total == pytest.approx(1.0)


def test_no_negative_weights_in_default():
    cfg = PortfolioRebalancingConfig()
    assert cfg.btc_weight >= 0
    assert cfg.eth_weight >= 0
    assert cfg.cash_weight >= 0


# ---------------------------------------------------------------------------
# Constructor invariants
# ---------------------------------------------------------------------------
def test_negative_weights_raise():
    with pytest.raises(ValueError, match="non-negative"):
        PortfolioRebalancingAllocator(
            PortfolioRebalancingConfig(btc_weight=-0.1),
        )


def test_overweight_raises():
    with pytest.raises(ValueError, match="leverage"):
        PortfolioRebalancingAllocator(
            PortfolioRebalancingConfig(btc_weight=0.9, eth_weight=0.5,
                                            cash_weight=0.0),
        )


def test_unknown_rebalance_frequency_raises():
    with pytest.raises(ValueError, match="rebalance"):
        PortfolioRebalancingAllocator(
            PortfolioRebalancingConfig(rebalance_frequency="quarterly"),
        )


# ---------------------------------------------------------------------------
# target_weights — happy path
# ---------------------------------------------------------------------------
def test_target_weights_returns_locked_vector_when_both_assets_present():
    s = PortfolioRebalancingAllocator()
    out = s.target_weights(
        1700086400000, _two_bar_frames(["BTC/USDT", "ETH/USDT"]),
    )
    assert out["BTC/USDT"] == pytest.approx(0.60)
    assert out["ETH/USDT"] == pytest.approx(0.30)
    # Cash is implicit: returned weights must sum to 0.90.
    assert sum(out.values()) == pytest.approx(0.90)


def test_no_short_exposure():
    s = PortfolioRebalancingAllocator()
    out = s.target_weights(
        1700086400000, _two_bar_frames(["BTC/USDT", "ETH/USDT"]),
    )
    for v in out.values():
        assert v >= 0


def test_no_leverage():
    s = PortfolioRebalancingAllocator()
    out = s.target_weights(
        1700086400000, _two_bar_frames(["BTC/USDT", "ETH/USDT"]),
    )
    assert sum(out.values()) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Lookahead invariance
# ---------------------------------------------------------------------------
def test_no_lookahead_fixed_weights_invariant_of_history():
    """Fixed weights — by definition lookahead-free. Asserting it
    explicitly so any future edit that introduces a signal-driven
    weight calc trips this test."""
    s = PortfolioRebalancingAllocator()
    frames_short = _two_bar_frames(["BTC/USDT", "ETH/USDT"])
    frames_long = {
        a: pd.concat([
            df,
            pd.DataFrame({
                "timestamp": [1800000000000],
                "datetime": pd.to_datetime([1800000000000], unit="ms",
                                              utc=True),
                "open": [200.0], "high": [202.0], "low": [199.0],
                "close": [201.0], "volume": [10.0],
            }),
        ], ignore_index=True)
        for a, df in frames_short.items()
    }
    asof = 1700086400000  # before the future bar
    a = s.target_weights(asof, frames_short)
    b = s.target_weights(asof, frames_long)
    assert a == b


# ---------------------------------------------------------------------------
# Missing-asset handling
# ---------------------------------------------------------------------------
def test_missing_eth_redistributes_to_btc():
    s = PortfolioRebalancingAllocator()
    out = s.target_weights(
        1700086400000, _two_bar_frames(["BTC/USDT"]),
    )
    # Locked risk_total = 0.90; ETH share folds onto BTC.
    assert "ETH/USDT" not in out
    assert out["BTC/USDT"] == pytest.approx(0.90)
    assert sum(out.values()) <= 1.0 + 1e-9


def test_missing_btc_redistributes_to_eth():
    s = PortfolioRebalancingAllocator()
    out = s.target_weights(
        1700086400000, _two_bar_frames(["ETH/USDT"]),
    )
    assert "BTC/USDT" not in out
    assert out["ETH/USDT"] == pytest.approx(0.90)


def test_both_missing_returns_empty():
    s = PortfolioRebalancingAllocator()
    out = s.target_weights(1700086400000, {})
    assert out == {}


def test_short_history_treated_as_missing():
    """Asset frame with only 1 bar at-or-before the asof is treated
    as missing (lookahead-free convention)."""
    s = PortfolioRebalancingAllocator()
    eth = pd.DataFrame({
        "timestamp": [1700086400000],
        "datetime": pd.to_datetime([1700086400000], unit="ms", utc=True),
        "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0],
        "volume": [1.0],
    })
    btc = _two_bar_frames(["BTC/USDT"])["BTC/USDT"]
    out = s.target_weights(1700086400000, {"BTC/USDT": btc, "ETH/USDT": eth})
    # ETH has only 1 bar at or before asof → treated as missing.
    assert "ETH/USDT" not in out
    assert out["BTC/USDT"] == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
def test_diagnostics_returns_locked_config():
    s = PortfolioRebalancingAllocator()
    d = s.diagnostics()
    assert d["btc_weight"] == 0.60
    assert d["eth_weight"] == 0.30
    assert d["cash_weight"] == 0.10
    assert d["rebalance_frequency"] == "monthly"


# ---------------------------------------------------------------------------
# Safety lock continues to be locked
# ---------------------------------------------------------------------------
def test_safety_lock_remains_locked():
    assert safety_lock.is_execution_allowed() is False
    assert safety_lock.is_paper_trading_allowed() is False
    assert safety_lock.is_kraken_connection_allowed() is False
    assert safety_lock.safety_lock_status() == "locked"


# ---------------------------------------------------------------------------
# Source-level safety
# ---------------------------------------------------------------------------
_SOURCE = (Path(__file__).resolve().parents[1]
              / "src" / "strategies"
              / "portfolio_rebalancing_allocator.py").read_text()


def test_no_broker_imports():
    bad_import_patterns = (
        re.compile(r"^\s*import\s+ccxt\b", re.MULTILINE),
        re.compile(r"^\s*from\s+ccxt", re.MULTILINE),
        re.compile(r"^\s*import\s+kraken\b", re.MULTILINE),
        re.compile(r"^\s*from\s+kraken\b", re.MULTILINE),
        re.compile(r"^\s*import\s+alpaca", re.MULTILINE),
        re.compile(r"^\s*from\s+alpaca", re.MULTILINE),
        re.compile(r"\bbinance\.client\b"),
        re.compile(r"\bbybit\.client\b"),
    )
    for pat in bad_import_patterns:
        assert pat.search(_SOURCE) is None, pat.pattern


def test_no_api_key_reads():
    forbidden = (
        re.compile(r"\bos\.environ\.get\([\"'](?:[A-Z_]*API[_-]?KEY)[\"']"),
        re.compile(r"\bos\.environ\[[\"'](?:[A-Z_]*API[_-]?KEY)[\"']"),
        re.compile(r"\bos\.getenv\([\"'](?:[A-Z_]*API[_-]?KEY)[\"']"),
        re.compile(r"\bAPI[_-]?KEY\s*=\s*['\"][A-Za-z0-9_\-]{8,}['\"]"),
    )
    for pat in forbidden:
        assert pat.search(_SOURCE) is None, pat.pattern


def test_no_kraken_private_endpoints():
    forbidden = (
        re.compile(r"\bkraken\.private\b"),
        re.compile(r"\bkraken_private\b"),
        re.compile(r"['\"]/0/private/"),
        re.compile(r"\bAddOrder\b"),
        re.compile(r"\bCancelOrder\b"),
    )
    for pat in forbidden:
        assert pat.search(_SOURCE) is None, pat.pattern


def test_no_order_placement_strings():
    forbidden = (
        re.compile(r"\bcreate_order\s*\("),
        re.compile(r"\bplace_order\s*\("),
        re.compile(r"\bsubmit_order\s*\("),
        re.compile(r"\bsend_order\s*\("),
        re.compile(r"\bcancel_order\s*\("),
        re.compile(r"\bdef\s+(?:create|place|submit|send|cancel)_order\b"),
    )
    for pat in forbidden:
        assert pat.search(_SOURCE) is None, pat.pattern


def test_no_paper_or_live_enablement():
    forbidden = (
        re.compile(r"paper_trading_allowed\s*=\s*True", re.IGNORECASE),
        re.compile(r"live_trading_allowed\s*=\s*True", re.IGNORECASE),
        re.compile(r"\bgo_live\b", re.IGNORECASE),
    )
    for pat in forbidden:
        assert pat.search(_SOURCE) is None, pat.pattern


def test_no_optimizer_or_fit():
    """Strategy is fixed-weight by spec — no optimiser, no fit."""
    bad = ("from scipy.optimize", "scipy.optimize",
            ".fit(", "minimize(", "least_squares(",
            "RandomForest", "XGB", "torch.optim")
    for s in bad:
        assert s not in _SOURCE, (
            f"strategy must be fixed-weight; found {s!r}"
        )
