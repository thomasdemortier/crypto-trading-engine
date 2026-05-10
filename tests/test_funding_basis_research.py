"""Tests for the funding+basis research orchestrator + scorecard.

These tests focus on the verdict logic and on the safety of the module:
no broker imports, no order placement, scorecard FAILs when BTC /
placebo gates are missed, INCONCLUSIVE when data coverage or rebalance
count is insufficient.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config, funding_basis_research as fbr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_wf(rows: int = 14, beats_btc: int = 8, beats_basket: int = 8,
              beats_simple: int = 8, profitable: int = 8,
              n_rebalances: int = 13) -> pd.DataFrame:
    out = []
    for i in range(rows):
        out.append({
            "window": i + 1,
            "oos_start_iso": f"2024-{(i % 12) + 1:02d}-01T00:00:00+00:00",
            "oos_end_iso": f"2024-{(i % 12) + 1:02d}-15T23:59:59+00:00",
            "oos_return_pct": 5.0 if i < profitable else -2.0,
            "oos_max_drawdown_pct": -8.0,
            "oos_sharpe": 0.5,
            "btc_oos_return_pct": 4.0 if i < beats_btc else 8.0,
            "eth_oos_return_pct": 3.0,
            "basket_oos_return_pct": 4.0 if i < beats_basket else 8.0,
            "simple_oos_return_pct": 4.0 if i < beats_simple else 8.0,
            "beats_btc": i < beats_btc,
            "beats_eth": True,
            "beats_basket": i < beats_basket,
            "beats_simple_momentum": i < beats_simple,
            "profitable": i < profitable,
            "n_rebalances": n_rebalances,
            "n_trades": 4 * n_rebalances,
            "avg_holdings": 1.5,
            "error": None,
        })
    return pd.DataFrame(out)


def _placebo_summary(strategy_return: float = 50.0,
                       strategy_dd: float = -25.0,
                       median_return: float = 30.0,
                       median_dd: float = -45.0) -> pd.DataFrame:
    return pd.DataFrame([{
        "strategy": "funding_basis_carry_allocator",
        "strategy_return_pct": strategy_return,
        "strategy_max_drawdown_pct": strategy_dd,
        "strategy_sharpe": 0.4,
        "n_seeds": 20,
        "placebo_median_return_pct": median_return,
        "placebo_median_drawdown_pct": median_dd,
        "placebo_p75_drawdown_pct": median_dd - 5.0,
        "strategy_beats_median_return": strategy_return > median_return,
        "strategy_beats_median_drawdown": strategy_dd > median_dd,
    }])


def _coverage_pass(tmp_path: Path) -> Path:
    """Write a coverage CSV with PASS verdicts so the data-coverage
    gate clears."""
    rows = []
    for asset in ("BTC/USDT", "ETH/USDT"):
        for ds in ("funding_rate_history", "mark_price_klines_1d",
                    "index_price_klines_1d"):
            rows.append({
                "source": "binance_futures", "asset": asset,
                "symbol_or_instrument": "BTCUSDT", "dataset": ds,
                "status": "ok", "verdict": "PASS",
                "row_count": 1000, "pages_walked": 5,
                "actual_start": "2022-04-01", "actual_end": "2026-04-01",
                "coverage_days": 1500, "expected_step_ms": 86400000,
                "gap_count": 0, "granularity": "1d",
                "last_http_status": 200,
                "csv_path": "x.csv", "notes": "",
            })
    df = pd.DataFrame(rows)
    p = tmp_path / "funding_basis_data_coverage.csv"
    df.to_csv(p, index=False)
    return p


def _comparison_with_btc_dd(tmp_path: Path, btc_dd: float = -60.0) -> Path:
    cmp = pd.DataFrame([
        {"strategy": "funding_basis_carry_allocator",
         "starting_capital": 10000.0, "final_value": 15000.0,
         "total_return_pct": 50.0, "max_drawdown_pct": -25.0,
         "sharpe_ratio": 0.4, "exposure_time_pct": 60.0},
        {"strategy": "BTC_buy_and_hold",
         "starting_capital": 10000.0, "final_value": 17500.0,
         "total_return_pct": 75.0, "max_drawdown_pct": btc_dd,
         "sharpe_ratio": 0.5, "exposure_time_pct": 100.0},
    ])
    p = tmp_path / "funding_basis_comparison.csv"
    cmp.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Scorecard verdict logic
# ---------------------------------------------------------------------------
def test_scorecard_pass_when_every_gate_clears(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    _coverage_pass(tmp_path)
    _comparison_with_btc_dd(tmp_path)
    wf = _make_wf(beats_btc=10, beats_basket=10, beats_simple=10,
                   profitable=10, n_rebalances=13)
    plac = _placebo_summary(strategy_return=50.0, median_return=20.0,
                                strategy_dd=-25.0)
    sc = fbr.funding_basis_scorecard(walk_forward_df=wf, placebo_df=plac,
                                         save=False)
    assert sc.iloc[0]["verdict"] == "PASS"
    assert int(sc.iloc[0]["checks_passed"]) == int(sc.iloc[0]["checks_total"])


def test_scorecard_fail_when_btc_gate_misses(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    _coverage_pass(tmp_path)
    _comparison_with_btc_dd(tmp_path)
    wf = _make_wf(beats_btc=2, beats_basket=10, beats_simple=10,
                   profitable=10)
    plac = _placebo_summary(strategy_return=50.0, median_return=20.0)
    sc = fbr.funding_basis_scorecard(walk_forward_df=wf, placebo_df=plac,
                                         save=False)
    assert sc.iloc[0]["verdict"] in ("FAIL", "WATCHLIST")
    assert "beats_btc_oos_majority=False" in str(sc.iloc[0]["reason"])


def test_scorecard_fail_when_placebo_gate_misses(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    _coverage_pass(tmp_path)
    _comparison_with_btc_dd(tmp_path)
    wf = _make_wf(beats_btc=10, beats_basket=10, beats_simple=10,
                   profitable=10)
    plac = _placebo_summary(strategy_return=10.0, median_return=20.0)
    sc = fbr.funding_basis_scorecard(walk_forward_df=wf, placebo_df=plac,
                                         save=False)
    assert "beats_placebo_median=False" in str(sc.iloc[0]["reason"])


def test_scorecard_inconclusive_when_coverage_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    # No coverage CSV at all → coverage_status returns a reason.
    _comparison_with_btc_dd(tmp_path)
    wf = _make_wf(beats_btc=10, beats_basket=10, beats_simple=10,
                   profitable=10)
    plac = _placebo_summary()
    sc = fbr.funding_basis_scorecard(walk_forward_df=wf, placebo_df=plac,
                                         save=False)
    assert sc.iloc[0]["verdict"] == "INCONCLUSIVE"


def test_scorecard_inconclusive_when_too_few_rebalances(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    _coverage_pass(tmp_path)
    _comparison_with_btc_dd(tmp_path)
    wf = _make_wf(beats_btc=10, beats_basket=10, beats_simple=10,
                   profitable=10, n_rebalances=0)
    plac = _placebo_summary()
    sc = fbr.funding_basis_scorecard(walk_forward_df=wf, placebo_df=plac,
                                         save=False)
    assert sc.iloc[0]["verdict"] == "INCONCLUSIVE"


def test_scorecard_inconclusive_when_no_walk_forward(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    sc = fbr.funding_basis_scorecard(walk_forward_df=pd.DataFrame(),
                                         placebo_df=pd.DataFrame(),
                                         save=False)
    assert sc.iloc[0]["verdict"] == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Coverage gate helper
# ---------------------------------------------------------------------------
def test_coverage_status_returns_none_when_all_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    _coverage_pass(tmp_path)
    assert fbr._coverage_status() is None


def test_coverage_status_returns_reason_when_any_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    _coverage_pass(tmp_path)
    cov = pd.read_csv(tmp_path / "funding_basis_data_coverage.csv")
    cov.loc[0, "verdict"] = "FAIL"
    cov.to_csv(tmp_path / "funding_basis_data_coverage.csv", index=False)
    msg = fbr._coverage_status()
    assert msg is not None
    assert "FAIL" in msg


# ---------------------------------------------------------------------------
# Placebo class
# ---------------------------------------------------------------------------
def test_placebo_only_emits_known_buckets_or_normalised_caps():
    """Placebo emits raw weights ∈ {0.80, 0.70, 0.30, 0.0}. When both
    BTC + ETH draw a non-zero bucket and they sum > 1, the result is
    normalised to Σ = 1. This test asserts the invariants:
        * Σ weights ≤ 1.
        * Every emitted weight is either zero, one of the raw caps, or
          arises from normalising two raw caps together (so each
          asset's weight equals its raw cap divided by the pair sum)."""
    cfg = fbr.FundingBasisPlaceboConfig(seed=0)
    plac = fbr.FundingBasisPlacebo(cfg)
    btc = pd.DataFrame({
        "timestamp": [1700000000000, 1700086400000],
        "datetime": pd.to_datetime([1700000000000, 1700086400000],
                                       unit="ms", utc=True),
        "open": [100.0, 101.0], "high": [102.0, 103.0],
        "low": [99.0, 100.0], "close": [101.0, 102.0],
        "volume": [10.0, 11.0],
    })
    eth = btc.copy()
    raw_caps = (0.80, 0.70, 0.30)
    allowed = set()
    for a in raw_caps:
        allowed.add(round(a, 4))
        for b in raw_caps:
            if a + b > 1.0:
                allowed.add(round(a / (a + b), 4))
    for _ in range(120):
        w = plac.target_weights(1700086400000,
                                    {"BTC/USDT": btc, "ETH/USDT": eth})
        # Σ ≤ 1 in every iteration.
        assert sum(w.values()) <= 1.0 + 1e-9
        for v in w.values():
            assert round(float(v), 4) in allowed


def test_placebo_seeds_are_reproducible():
    cfg_a = fbr.FundingBasisPlaceboConfig(seed=7)
    cfg_b = fbr.FundingBasisPlaceboConfig(seed=7)
    btc = pd.DataFrame({
        "timestamp": [1700000000000, 1700086400000],
        "datetime": pd.to_datetime([1700000000000, 1700086400000],
                                       unit="ms", utc=True),
        "open": [100.0, 101.0], "high": [102.0, 103.0],
        "low": [99.0, 100.0], "close": [101.0, 102.0],
        "volume": [10.0, 11.0],
    })
    a = fbr.FundingBasisPlacebo(cfg_a)
    b = fbr.FundingBasisPlacebo(cfg_b)
    seq_a = [a.target_weights(1700086400000, {"BTC/USDT": btc})
              for _ in range(20)]
    seq_b = [b.target_weights(1700086400000, {"BTC/USDT": btc})
              for _ in range(20)]
    assert seq_a == seq_b


# ---------------------------------------------------------------------------
# Source-level safety
# ---------------------------------------------------------------------------
_RESEARCH_SOURCE = (Path(__file__).resolve().parents[1]
                      / "src" / "funding_basis_research.py").read_text()


def test_no_broker_imports():
    bad = ("ccxt", "kraken", "binance.client", "bybit.client",
            "okx.client", "deribit.client")
    for s in bad:
        assert s.lower() not in _RESEARCH_SOURCE.lower(), s


def test_no_order_placement_in_research():
    forbidden = (
        re.compile(r"\bcreate_order\s*\("),
        re.compile(r"\bplace_order\s*\("),
        re.compile(r"\bsubmit_order\s*\("),
        re.compile(r"\bsend_order\s*\("),
        re.compile(r"\bcancel_order\s*\("),
        re.compile(r"\bdef\s+(?:create|place|submit|send|cancel)_order\b"),
    )
    for pat in forbidden:
        assert pat.search(_RESEARCH_SOURCE) is None, pat.pattern


def test_no_api_key_reads_in_research():
    forbidden = (
        re.compile(r"\bos\.environ\.get\([\"'](?:[A-Z_]*API[_-]?KEY)[\"']"),
        re.compile(r"\bAPI[_-]?KEY\s*=\s*['\"][A-Za-z0-9_\-]{8,}['\"]"),
    )
    for pat in forbidden:
        assert pat.search(_RESEARCH_SOURCE) is None, pat.pattern
