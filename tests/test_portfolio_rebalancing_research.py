"""Tests for `src/portfolio_rebalancing_research.py`."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from src import config, portfolio_rebalancing_research as prr, safety_lock


# ---------------------------------------------------------------------------
# Locked thresholds
# ---------------------------------------------------------------------------
def test_locked_thresholds():
    assert prr.SHARPE_GAP_LIMIT == 0.10
    assert prr.DRAWDOWN_TIGHTNESS_PP == 15.0
    assert prr.MIN_REBALANCES_TOTAL == 24


def test_default_seeds_at_least_20():
    assert len(prr.PLACEBO_SEEDS_DEFAULT) >= 20


# ---------------------------------------------------------------------------
# Placebo determinism + boundedness
# ---------------------------------------------------------------------------
def test_placebo_seed_is_deterministic():
    a = prr.RebalancingPlacebo(prr.RebalancingPlaceboConfig(seed=7))
    b = prr.RebalancingPlacebo(prr.RebalancingPlaceboConfig(seed=7))
    assert a.weights == b.weights


def test_placebo_weights_are_long_only_no_leverage():
    for seed in range(20):
        plac = prr.RebalancingPlacebo(
            prr.RebalancingPlaceboConfig(seed=seed),
        )
        w_btc, w_eth = plac.weights
        assert 0.0 <= w_btc <= 1.0
        assert 0.0 <= w_eth <= 1.0
        assert (w_btc + w_eth) <= 1.0 + 1e-9


def test_placebo_target_weights_respect_constraints():
    """Returned dict must be long-only and Σ ≤ 1 for every seed."""
    btc = pd.DataFrame({
        "timestamp": [1700000000000, 1700086400000],
        "datetime": pd.to_datetime([1700000000000, 1700086400000],
                                       unit="ms", utc=True),
        "open": [100.0, 101.0], "high": [102.0, 103.0],
        "low": [99.0, 100.0], "close": [101.0, 102.0],
        "volume": [10.0, 11.0],
    })
    eth = btc.copy()
    for seed in range(20):
        plac = prr.RebalancingPlacebo(
            prr.RebalancingPlaceboConfig(seed=seed),
        )
        out = plac.target_weights(
            1700086400000, {"BTC/USDT": btc, "ETH/USDT": eth},
        )
        for v in out.values():
            assert v >= 0
        assert sum(out.values()) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Scorecard verdict logic — with synthetic walk_forward + placebo frames
# ---------------------------------------------------------------------------
def _placebo_summary(beats_return: bool = True,
                       beats_drawdown: bool = True,
                       median_return: float = 0.0,
                       median_dd: float = -50.0,
                       strat_return: float = 30.0,
                       strat_dd: float = -40.0) -> pd.DataFrame:
    return pd.DataFrame([{
        "strategy": "portfolio_rebalancing_allocator",
        "strategy_return_pct": strat_return,
        "strategy_max_drawdown_pct": strat_dd,
        "strategy_sharpe": 0.45,
        "n_seeds": 20,
        "placebo_median_return_pct": median_return,
        "placebo_median_drawdown_pct": median_dd,
        "strategy_beats_median_return": beats_return,
        "strategy_beats_median_drawdown": beats_drawdown,
        "placebo_return_percentile": 75.0,
        "placebo_drawdown_percentile": 70.0,
    }])


def _comparison(strat_return: float = 30.0, strat_dd: float = -45.0,
                  strat_sharpe: float = 0.45,
                  btc_return: float = 75.0, btc_dd: float = -66.0,
                  btc_sharpe: float = 0.50) -> pd.DataFrame:
    return pd.DataFrame([
        {"strategy": "portfolio_rebalancing_allocator",
          "starting_capital": 10000.0, "final_value": 13000.0,
          "total_return_pct": strat_return,
          "max_drawdown_pct": strat_dd,
          "sharpe_ratio": strat_sharpe,
          "exposure_time_pct": 60.0},
        {"strategy": "BTC_buy_and_hold",
          "starting_capital": 10000.0, "final_value": 17500.0,
          "total_return_pct": btc_return,
          "max_drawdown_pct": btc_dd,
          "sharpe_ratio": btc_sharpe,
          "exposure_time_pct": 100.0},
        {"strategy": "equal_weight_basket",
          "starting_capital": 10000.0, "final_value": 12000.0,
          "total_return_pct": 20.0,
          "max_drawdown_pct": -60.0,
          "sharpe_ratio": 0.30, "exposure_time_pct": 100.0},
    ])


def _walk_forward(rows: int = 10, n_rebalances_per_window: int = 3
                    ) -> pd.DataFrame:
    out = []
    for i in range(rows):
        out.append({
            "window": i + 1,
            "oos_start_iso": "2024-01-01T00:00:00+00:00",
            "oos_end_iso": "2024-04-01T00:00:00+00:00",
            "oos_return_pct": 5.0,
            "oos_max_drawdown_pct": -10.0,
            "oos_sharpe": 0.4,
            "btc_return_pct": 10.0,
            "btc_drawdown_pct": -25.0,
            "btc_sharpe": 0.45,
            "eth_return_pct": 8.0,
            "basket_return_pct": 9.0,
            "n_rebalances": n_rebalances_per_window,
            "sharpe_within_010": True,
            "drawdown_15pp_tighter": True,
            "error": None,
        })
    return pd.DataFrame(out)


def _write_comparison(tmp_path, **kw) -> None:
    _comparison(**kw).to_csv(
        tmp_path / "portfolio_rebalancing_comparison.csv", index=False,
    )


def test_scorecard_pass_when_every_locked_gate_clears(tmp_path,
                                                            monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    _write_comparison(tmp_path,
                       strat_sharpe=0.45, btc_sharpe=0.50,
                       strat_dd=-40.0, btc_dd=-66.0)
    wf = _walk_forward(rows=10, n_rebalances_per_window=3)  # 30 rebal
    plac = _placebo_summary(beats_return=True, beats_drawdown=True)
    sc = prr.portfolio_rebalancing_scorecard(
        walk_forward_df=wf, placebo_df=plac, save=False,
    )
    assert sc.iloc[0]["verdict"] == "PASS"
    assert bool(sc.iloc[0]["pass_sharpe_within_010"]) is True
    assert bool(sc.iloc[0]["pass_drawdown_15pp_tighter"]) is True
    assert bool(sc.iloc[0]["pass_beats_placebo_return"]) is True
    assert bool(sc.iloc[0]["pass_beats_placebo_drawdown"]) is True
    assert bool(sc.iloc[0]["pass_min_24_rebalances"]) is True


def test_scorecard_fail_when_sharpe_too_far(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    _write_comparison(tmp_path, strat_sharpe=0.20, btc_sharpe=0.50,
                       strat_dd=-40.0, btc_dd=-66.0)
    wf = _walk_forward(rows=10, n_rebalances_per_window=3)
    plac = _placebo_summary()
    sc = prr.portfolio_rebalancing_scorecard(
        walk_forward_df=wf, placebo_df=plac, save=False,
    )
    assert bool(sc.iloc[0]["pass_sharpe_within_010"]) is False
    assert sc.iloc[0]["verdict"] in ("FAIL", "WATCHLIST")


def test_scorecard_fail_when_drawdown_not_tight_enough(tmp_path,
                                                              monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    # Strat DD only 10pp tighter than BTC — fails the 15pp gate.
    _write_comparison(tmp_path, strat_dd=-56.0, btc_dd=-66.0,
                       strat_sharpe=0.45, btc_sharpe=0.50)
    wf = _walk_forward(rows=10, n_rebalances_per_window=3)
    plac = _placebo_summary()
    sc = prr.portfolio_rebalancing_scorecard(
        walk_forward_df=wf, placebo_df=plac, save=False,
    )
    assert bool(sc.iloc[0]["pass_drawdown_15pp_tighter"]) is False


def test_scorecard_fail_when_placebo_return_lost(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    _write_comparison(tmp_path, strat_sharpe=0.45, btc_sharpe=0.50,
                       strat_dd=-40.0, btc_dd=-66.0)
    wf = _walk_forward(rows=10, n_rebalances_per_window=3)
    plac = _placebo_summary(beats_return=False, beats_drawdown=True)
    sc = prr.portfolio_rebalancing_scorecard(
        walk_forward_df=wf, placebo_df=plac, save=False,
    )
    assert bool(sc.iloc[0]["pass_beats_placebo_return"]) is False


def test_scorecard_fail_when_placebo_drawdown_lost(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    _write_comparison(tmp_path, strat_sharpe=0.45, btc_sharpe=0.50,
                       strat_dd=-40.0, btc_dd=-66.0)
    wf = _walk_forward(rows=10, n_rebalances_per_window=3)
    plac = _placebo_summary(beats_return=True, beats_drawdown=False)
    sc = prr.portfolio_rebalancing_scorecard(
        walk_forward_df=wf, placebo_df=plac, save=False,
    )
    assert bool(sc.iloc[0]["pass_beats_placebo_drawdown"]) is False


def test_scorecard_fail_when_too_few_rebalances(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    _write_comparison(tmp_path, strat_sharpe=0.45, btc_sharpe=0.50,
                       strat_dd=-40.0, btc_dd=-66.0)
    # 5 windows × 4 rebalances = 20 < 24 requirement.
    wf = _walk_forward(rows=5, n_rebalances_per_window=4)
    plac = _placebo_summary()
    sc = prr.portfolio_rebalancing_scorecard(
        walk_forward_df=wf, placebo_df=plac, save=False,
    )
    # 20 < 24 → fail this gate.
    assert bool(sc.iloc[0]["pass_min_24_rebalances"]) is False


def test_scorecard_inconclusive_when_insufficient_windows(tmp_path,
                                                                  monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    _write_comparison(tmp_path)
    wf = _walk_forward(rows=2, n_rebalances_per_window=4)
    plac = _placebo_summary()
    sc = prr.portfolio_rebalancing_scorecard(
        walk_forward_df=wf, placebo_df=plac, save=False,
    )
    assert sc.iloc[0]["verdict"] == "INCONCLUSIVE"


def test_scorecard_inconclusive_on_missing_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    sc = prr.portfolio_rebalancing_scorecard(
        walk_forward_df=pd.DataFrame(),
        placebo_df=pd.DataFrame(),
        save=False,
    )
    assert sc.iloc[0]["verdict"] == "INCONCLUSIVE"


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
              / "src" / "portfolio_rebalancing_research.py").read_text()


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


def test_no_paper_or_live_trading_enablement():
    forbidden = (
        re.compile(r"paper_trading_allowed\s*=\s*True", re.IGNORECASE),
        re.compile(r"live_trading_allowed\s*=\s*True", re.IGNORECASE),
        re.compile(r"\bgo_live\b", re.IGNORECASE),
    )
    for pat in forbidden:
        assert pat.search(_SOURCE) is None, pat.pattern


def test_results_glob_is_gitignored():
    """Sanity check: `.gitignore` covers `results/*.csv` so every
    generated artefact from this research module is automatically
    excluded."""
    gi = (Path(__file__).resolve().parents[1] / ".gitignore").read_text()
    assert "results/*.csv" in gi
