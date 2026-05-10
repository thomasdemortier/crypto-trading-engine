"""Tests for the relative-value research orchestrator + scorecard."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config, relative_value_research as rvr


def _make_wf(rows: int = 14, *, beats_btc: int = 8, beats_eth: int = 8,
              beats_basket: int = 8, beats_simple: int = 8,
              profitable: int = 8, n_rebalances: int = 13) -> pd.DataFrame:
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
            "eth_oos_return_pct": 4.0 if i < beats_eth else 8.0,
            "basket_oos_return_pct": 4.0 if i < beats_basket else 8.0,
            "simple_oos_return_pct": 4.0 if i < beats_simple else 8.0,
            "beats_btc": i < beats_btc,
            "beats_eth": i < beats_eth,
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
        "strategy": "relative_value_btc_eth_allocator",
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


def _comparison_with_btc_dd(tmp_path: Path, btc_dd: float = -60.0) -> Path:
    cmp = pd.DataFrame([
        {"strategy": "relative_value_btc_eth_allocator",
         "starting_capital": 10000.0, "final_value": 15000.0,
         "total_return_pct": 50.0, "max_drawdown_pct": -25.0,
         "sharpe_ratio": 0.4, "exposure_time_pct": 60.0},
        {"strategy": "BTC_buy_and_hold",
         "starting_capital": 10000.0, "final_value": 17500.0,
         "total_return_pct": 75.0, "max_drawdown_pct": btc_dd,
         "sharpe_ratio": 0.5, "exposure_time_pct": 100.0},
    ])
    p = tmp_path / "relative_value_comparison.csv"
    cmp.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Scorecard verdict logic
# ---------------------------------------------------------------------------
def test_scorecard_pass_when_every_gate_clears(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    _comparison_with_btc_dd(tmp_path)
    wf = _make_wf(beats_btc=10, beats_eth=10, beats_basket=10,
                   beats_simple=10, profitable=10, n_rebalances=13)
    plac = _placebo_summary(strategy_return=50.0, median_return=20.0,
                                strategy_dd=-25.0)
    sc = rvr.relative_value_scorecard(walk_forward_df=wf, placebo_df=plac,
                                          save=False)
    assert sc.iloc[0]["verdict"] == "PASS"


def test_scorecard_fail_when_btc_gate_misses(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    _comparison_with_btc_dd(tmp_path)
    wf = _make_wf(beats_btc=2, beats_eth=10, beats_basket=10,
                   beats_simple=10, profitable=10)
    plac = _placebo_summary(strategy_return=50.0, median_return=20.0)
    sc = rvr.relative_value_scorecard(walk_forward_df=wf, placebo_df=plac,
                                          save=False)
    assert sc.iloc[0]["verdict"] in ("FAIL", "WATCHLIST")
    assert "beats_btc_oos_majority=False" in str(sc.iloc[0]["reason"])


def test_scorecard_fail_when_placebo_gate_misses(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    _comparison_with_btc_dd(tmp_path)
    wf = _make_wf(beats_btc=10, beats_eth=10, beats_basket=10,
                   beats_simple=10, profitable=10)
    plac = _placebo_summary(strategy_return=10.0, median_return=20.0)
    sc = rvr.relative_value_scorecard(walk_forward_df=wf, placebo_df=plac,
                                          save=False)
    assert "beats_placebo_median=False" in str(sc.iloc[0]["reason"])


def test_scorecard_inconclusive_when_too_few_rebalances(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    _comparison_with_btc_dd(tmp_path)
    wf = _make_wf(beats_btc=10, beats_eth=10, beats_basket=10,
                   beats_simple=10, profitable=10, n_rebalances=0)
    plac = _placebo_summary()
    sc = rvr.relative_value_scorecard(walk_forward_df=wf, placebo_df=plac,
                                          save=False)
    assert sc.iloc[0]["verdict"] == "INCONCLUSIVE"


def test_scorecard_inconclusive_when_no_walk_forward(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    sc = rvr.relative_value_scorecard(walk_forward_df=pd.DataFrame(),
                                          placebo_df=pd.DataFrame(),
                                          save=False)
    assert sc.iloc[0]["verdict"] == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Placebo class
# ---------------------------------------------------------------------------
def test_placebo_emits_only_long_only_bucketed_weights():
    cfg = rvr.RelativeValuePlaceboConfig(seed=0)
    plac = rvr.RelativeValuePlacebo(cfg)
    btc = pd.DataFrame({
        "timestamp": [1700000000000, 1700086400000],
        "datetime": pd.to_datetime([1700000000000, 1700086400000],
                                       unit="ms", utc=True),
        "open": [100.0, 101.0], "high": [102.0, 103.0],
        "low": [99.0, 100.0], "close": [101.0, 102.0],
        "volume": [10.0, 11.0],
    })
    eth = btc.copy()
    raw = {0.0, 0.5, 1.0}
    for _ in range(120):
        w = plac.target_weights(1700086400000,
                                    {"BTC/USDT": btc, "ETH/USDT": eth})
        for v in w.values():
            assert v >= 0.0
            assert round(float(v), 4) in {round(x, 4) for x in raw}
        assert sum(w.values()) <= 1.0 + 1e-9


def test_placebo_seeds_are_reproducible():
    btc = pd.DataFrame({
        "timestamp": [1700000000000, 1700086400000],
        "datetime": pd.to_datetime([1700000000000, 1700086400000],
                                       unit="ms", utc=True),
        "open": [100.0, 101.0], "high": [102.0, 103.0],
        "low": [99.0, 100.0], "close": [101.0, 102.0],
        "volume": [10.0, 11.0],
    })
    a = rvr.RelativeValuePlacebo(rvr.RelativeValuePlaceboConfig(seed=11))
    b = rvr.RelativeValuePlacebo(rvr.RelativeValuePlaceboConfig(seed=11))
    seq_a = [a.target_weights(1700086400000, {"BTC/USDT": btc})
              for _ in range(20)]
    seq_b = [b.target_weights(1700086400000, {"BTC/USDT": btc})
              for _ in range(20)]
    assert seq_a == seq_b


# ---------------------------------------------------------------------------
# Source-level safety
# ---------------------------------------------------------------------------
_RESEARCH_SOURCE = (Path(__file__).resolve().parents[1]
                      / "src" / "relative_value_research.py").read_text()
_SIGNALS_SOURCE = (Path(__file__).resolve().parents[1]
                     / "src" / "relative_value_signals.py").read_text()
_STRATEGY_SOURCE = (Path(__file__).resolve().parents[1]
                      / "src" / "strategies"
                      / "relative_value_btc_eth_allocator.py").read_text()


def test_no_broker_imports_in_any_module():
    bad = ("ccxt", "kraken", "binance.client", "bybit.client",
            "okx.client", "deribit.client")
    for src in (_RESEARCH_SOURCE, _SIGNALS_SOURCE, _STRATEGY_SOURCE):
        for s in bad:
            assert s.lower() not in src.lower(), s


def test_no_order_placement_in_any_module():
    forbidden = (
        re.compile(r"\bcreate_order\s*\("),
        re.compile(r"\bplace_order\s*\("),
        re.compile(r"\bsubmit_order\s*\("),
        re.compile(r"\bsend_order\s*\("),
        re.compile(r"\bcancel_order\s*\("),
        re.compile(r"\bdef\s+(?:create|place|submit|send|cancel)_order\b"),
    )
    for src in (_RESEARCH_SOURCE, _SIGNALS_SOURCE, _STRATEGY_SOURCE):
        for pat in forbidden:
            assert pat.search(src) is None, pat.pattern


def test_no_api_key_reads_in_any_module():
    forbidden = (
        re.compile(r"\bos\.environ\.get\([\"'](?:[A-Z_]*API[_-]?KEY)[\"']"),
        re.compile(r"\bAPI[_-]?KEY\s*=\s*['\"][A-Za-z0-9_\-]{8,}['\"]"),
    )
    for src in (_RESEARCH_SOURCE, _SIGNALS_SOURCE, _STRATEGY_SOURCE):
        for pat in forbidden:
            assert pat.search(src) is None, pat.pattern


def test_no_kraken_private_endpoints_in_any_module():
    forbidden = (
        re.compile(r"\bkraken\.private\b"),
        re.compile(r"\bkraken_private\b"),
        re.compile(r"['\"]/0/private/"),
        re.compile(r"\bAddOrder\b"),
        re.compile(r"\bCancelOrder\b"),
    )
    for src in (_RESEARCH_SOURCE, _SIGNALS_SOURCE, _STRATEGY_SOURCE):
        for pat in forbidden:
            assert pat.search(src) is None, pat.pattern
