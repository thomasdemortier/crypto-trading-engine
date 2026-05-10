"""Tests for `src/fx_eod_trend_research.py`.

Covers: quality-guard semantics, deterministic placebo seeds, scorecard
PASS/FAIL/INCONCLUSIVE logic, write-path restriction, and the locked
safety invariants for the orchestrator + strategy registry entry.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src import (
    config,
    fx_data_quality as fxq,
    fx_eod_trend_research as fxr,
    fx_research_dataset as fxd,
    safety_lock,
    strategy_registry,
)
from src.strategies import fx_eod_trend


REPO_ROOT = config.REPO_ROOT


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------
def _synthetic_eur_usd_long(n_days: int = 600, seed: int = 9
                                ) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0001, 0.005, size=n_days)
    closes = 1.10 * np.exp(np.cumsum(rets))
    dates = pd.date_range("2000-01-03", periods=n_days, freq="B")
    rows = []
    for d, c in zip(dates, closes):
        rows.append({
            "date": d, "asset": "EUR/USD", "source": "ecb_sdmx",
            "base": "EUR", "quote": "USD", "close": float(c),
            "return_1d": float("nan"),
            "log_return_1d": float("nan"),
            "is_derived": False, "source_pair": "",
            "data_quality_status": "ok", "notes": "",
        })
    return pd.DataFrame(rows, columns=fxd.DATASET_COLUMNS)


# ---------------------------------------------------------------------------
# Quality guard
# ---------------------------------------------------------------------------
def test_load_and_guard_blocks_when_dataset_missing(tmp_path):
    bogus = tmp_path / "missing_fx.parquet"
    with pytest.raises((fxr.FxEodTrendBlockedError,
                         fxq.FxDatasetMissingError, FileNotFoundError)):
        # The orchestrator's guard goes through fx_data_quality, which
        # returns INCONCLUSIVE → blocked.
        try:
            fxr.load_and_guard(bogus)
        except FileNotFoundError:
            raise


def test_load_and_guard_blocks_on_fail_verdict(monkeypatch):
    """If the quality module returns FAIL, the guard must refuse."""
    fake_report = fxq.QualityReport(
        verdict=fxq.FAIL,
        checks=[fxq.CheckResult("schema", fxq.FAIL, "synthetic FAIL")],
        summary={"rows": 1234, "loaded": True},
    )
    monkeypatch.setattr(
        fxq, "run_fx_data_quality_checks", lambda path=None: fake_report,
    )
    with pytest.raises(fxr.FxEodTrendBlockedError):
        fxr.load_and_guard()


def test_load_and_guard_blocks_on_inconclusive_verdict(monkeypatch):
    fake_report = fxq.QualityReport(
        verdict=fxq.INCONCLUSIVE,
        checks=[fxq.CheckResult("load_dataset", fxq.INCONCLUSIVE,
                                  "synthetic INCONCLUSIVE")],
        summary={"rows": 0, "loaded": False},
    )
    monkeypatch.setattr(
        fxq, "run_fx_data_quality_checks", lambda path=None: fake_report,
    )
    with pytest.raises(fxr.FxEodTrendBlockedError):
        fxr.load_and_guard()


def test_load_and_guard_allows_warning_verdict(monkeypatch):
    """WARNING is acceptable — current FX dataset's verdict."""
    df = _synthetic_eur_usd_long(n_days=300)
    fake_report = fxq.QualityReport(
        verdict=fxq.WARNING,
        checks=[fxq.CheckResult("schema", fxq.PASS, "ok")],
        summary={"rows": 300, "loaded": True},
    )
    monkeypatch.setattr(
        fxq, "run_fx_data_quality_checks", lambda path=None: fake_report,
    )
    monkeypatch.setattr(fxq, "load_fx_dataset", lambda path=None: df)
    out_df, out_report = fxr.load_and_guard()
    assert out_report.verdict == fxq.WARNING
    assert len(out_df) == 300


def test_load_and_guard_allows_pass_verdict(monkeypatch):
    df = _synthetic_eur_usd_long(n_days=300)
    fake_report = fxq.QualityReport(
        verdict=fxq.PASS,
        checks=[fxq.CheckResult("schema", fxq.PASS, "ok")],
        summary={"rows": 300, "loaded": True},
    )
    monkeypatch.setattr(
        fxq, "run_fx_data_quality_checks", lambda path=None: fake_report,
    )
    monkeypatch.setattr(fxq, "load_fx_dataset", lambda path=None: df)
    out_df, out_report = fxr.load_and_guard()
    assert out_report.verdict == fxq.PASS
    assert len(out_df) == 300


# ---------------------------------------------------------------------------
# Walk-forward + placebo
# ---------------------------------------------------------------------------
def test_walk_forward_window_count_and_schema():
    cfg = fx_eod_trend.FXEODTrendConfig(lookback_days=20)
    df = _synthetic_eur_usd_long(n_days=600)
    wf = fxr.run_walk_forward(cfg, df, n_windows=5)
    assert len(wf) == 5
    assert list(wf.columns) == fxr.WALK_FORWARD_COLUMNS


def test_placebo_minimum_seeds_enforced():
    cfg = fx_eod_trend.FXEODTrendConfig(lookback_days=20)
    df = _synthetic_eur_usd_long(n_days=300)
    with pytest.raises(ValueError, match="locked minimum"):
        fxr.run_placebo(cfg, df, n_seeds=5)


def test_placebo_is_deterministic_for_same_seeds():
    cfg = fx_eod_trend.FXEODTrendConfig(lookback_days=20)
    df = _synthetic_eur_usd_long(n_days=300)
    pb_a = fxr.run_placebo(cfg, df, n_seeds=20)
    pb_b = fxr.run_placebo(cfg, df, n_seeds=20)
    pd.testing.assert_frame_equal(pb_a, pb_b)


def test_placebo_schema_and_seed_count():
    cfg = fx_eod_trend.FXEODTrendConfig(lookback_days=20)
    df = _synthetic_eur_usd_long(n_days=300)
    pb = fxr.run_placebo(cfg, df, n_seeds=20)
    assert len(pb) == 20
    assert list(pb.columns) == fxr.PLACEBO_COLUMNS
    # All seeds 0..19, each placebo binary → max 1.0, min 0.0
    assert sorted(pb["seed"].tolist()) == list(range(20))


def test_placebo_matched_exposure_close_to_strategy_exposure():
    cfg = fx_eod_trend.FXEODTrendConfig(lookback_days=20)
    df = _synthetic_eur_usd_long(n_days=600)
    bt = fxr.run_full_window_backtest(cfg, df)
    pb = fxr.run_placebo(cfg, df, n_seeds=20)
    strat_exposure = float(bt["position"].mean())
    # Each placebo's exposure is round(n*p)/n → within 1/n of strategy.
    n = len(bt)
    tol = 100.0 / n + 1e-9  # in percentage points
    for ex in pb["exposure_pct"]:
        assert abs(ex - 100.0 * strat_exposure) <= tol


# ---------------------------------------------------------------------------
# Scorecard verdict logic
# ---------------------------------------------------------------------------
def _backtest_with_known_returns(strategy_returns: List[float],
                                    benchmark_returns: List[float],
                                    positions: List[float]) -> pd.DataFrame:
    n = len(strategy_returns)
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    s_eq = pd.Series(strategy_returns).add(1.0).cumprod()
    b_eq = pd.Series(benchmark_returns).add(1.0).cumprod()
    return pd.DataFrame({
        "date": dates,
        "asset": "EUR/USD",
        "close": 1.0 + np.arange(n) * 0.0,
        "sma": 1.0,
        "signal": positions,
        "position": positions,
        "raw_return": benchmark_returns,
        "strategy_return": strategy_returns,
        "strategy_equity": s_eq.values,
        "benchmark_buyhold_return": benchmark_returns,
        "benchmark_buyhold_equity": b_eq.values,
        "cash_equity": 1.0,
    })


def _placebo_frame(returns: List[float], drawdowns: List[float],
                      ) -> pd.DataFrame:
    rows = []
    for s, (r, dd) in enumerate(zip(returns, drawdowns)):
        rows.append({
            "seed": s, "exposure_pct": 50.0, "trade_count": 30,
            "total_return": r, "sharpe": 0.0, "max_drawdown": dd,
            "design": "matched_exposure_random_binary_seeded",
        })
    return pd.DataFrame(rows, columns=fxr.PLACEBO_COLUMNS)


def test_scorecard_pass_when_every_check_holds():
    cfg = fx_eod_trend.FXEODTrendConfig()
    # Strategy: strong positive return, strong sharpe, tight DD, 25 trades.
    rng = np.random.default_rng(0)
    pos_arr = (rng.random(80) > 0.4).astype(float)
    # Force at least 25 flips by alternating mid-section.
    pos_arr[20:60] = ((np.arange(40) % 2)).astype(float)
    strat_rets = (pos_arr * 0.002).tolist()
    # Benchmark: mildly positive but worse drawdown / sharpe.
    bench_rets = ([0.001] * 40 + [-0.005] * 20 + [0.001] * 20)
    bt = _backtest_with_known_returns(strat_rets, bench_rets,
                                          pos_arr.tolist())
    placebo_returns = [-0.05, -0.02, 0.00, 0.02, 0.04,
                          -0.04, -0.01, 0.01, 0.03, 0.05,
                          -0.03, 0.00, 0.02, -0.02, 0.01,
                          0.00, 0.02, -0.01, 0.03, -0.02]
    placebo_dds = [-0.10, -0.12, -0.11, -0.13, -0.09,
                       -0.10, -0.12, -0.11, -0.13, -0.09,
                       -0.10, -0.12, -0.11, -0.13, -0.09,
                       -0.10, -0.12, -0.11, -0.13, -0.09]
    pb = _placebo_frame(placebo_returns, placebo_dds)
    sc = fxr.compute_scorecard(cfg, bt, pb)
    row = sc.iloc[0]
    assert row["verdict"] == fxr.VERDICT_PASS, row.get("notes")
    assert row["checks_passed"] == row["checks_total"]


def test_scorecard_fail_when_drawdown_not_tight_enough():
    cfg = fx_eod_trend.FXEODTrendConfig()
    # Strategy and benchmark have ~equal max drawdowns → fail
    # the 5-pp tighter requirement.
    pos = ([1.0, 0.0] * 40)
    strat = ([0.001, -0.05, 0.001, 0.001] * 20)
    bench = ([0.001, -0.05, 0.001, 0.001] * 20)
    bt = _backtest_with_known_returns(strat, bench, pos)
    pb = _placebo_frame([-0.10] * 20, [-0.30] * 20)
    sc = fxr.compute_scorecard(cfg, bt, pb)
    assert sc.iloc[0]["verdict"] == fxr.VERDICT_FAIL
    assert "drawdown_improvement_pp" in (sc.iloc[0]["notes"] or "")


def test_scorecard_fail_when_total_return_negative():
    cfg = fx_eod_trend.FXEODTrendConfig()
    pos = [1.0] * 60
    strat = [-0.01] * 60
    bench = [0.005] * 60
    bt = _backtest_with_known_returns(strat, bench, pos)
    pb = _placebo_frame([0.00] * 20, [-0.10] * 20)
    sc = fxr.compute_scorecard(cfg, bt, pb)
    assert sc.iloc[0]["verdict"] == fxr.VERDICT_FAIL
    assert not bool(sc.iloc[0]["pass_positive_return"])


def test_scorecard_fail_when_trade_count_too_low():
    cfg = fx_eod_trend.FXEODTrendConfig()
    # Always-long, never flips → 0 trades → fails min_trade_count.
    pos = [1.0] * 60
    strat = [0.001] * 60
    bench = [0.0] * 60
    bt = _backtest_with_known_returns(strat, bench, pos)
    pb = _placebo_frame([-0.10] * 20, [-0.30] * 20)
    sc = fxr.compute_scorecard(cfg, bt, pb)
    assert sc.iloc[0]["verdict"] == fxr.VERDICT_FAIL
    assert not bool(sc.iloc[0]["pass_min_trade_count"])


def test_scorecard_inconclusive_when_backtest_empty():
    cfg = fx_eod_trend.FXEODTrendConfig()
    sc = fxr.compute_scorecard(
        cfg,
        pd.DataFrame(columns=fx_eod_trend.BACKTEST_COLUMNS),
        pd.DataFrame(columns=fxr.PLACEBO_COLUMNS),
    )
    assert sc.iloc[0]["verdict"] == fxr.VERDICT_INCONCLUSIVE


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def test_writer_rejects_path_outside_results(tmp_path):
    bad = tmp_path / "fx_eod_trend_backtest.csv"
    with pytest.raises(fxr.FxEodTrendWritePathError):
        fxr.write_csv(pd.DataFrame({"a": [1]}), bad)


def test_writer_writes_inside_results():
    target = config.RESULTS_DIR / "fx_eod_trend_test_only.csv"
    try:
        fxr.write_csv(pd.DataFrame({"a": [1, 2, 3]}), target)
        assert target.exists()
        loaded = pd.read_csv(target)
        assert loaded["a"].tolist() == [1, 2, 3]
    finally:
        if target.exists():
            target.unlink()


# ---------------------------------------------------------------------------
# Strategy registry — fx_eod_trend entry must be paper/live False
# ---------------------------------------------------------------------------
def test_strategy_registry_includes_fx_eod_trend():
    rows = strategy_registry.build_registry()
    families = {r["strategy_family"] for r in rows}
    assert "fx_eod_trend" in families


def test_strategy_registry_fx_eod_trend_paper_live_false():
    rows = strategy_registry.build_registry()
    by_family: Dict[str, Dict[str, Any]] = {
        r["strategy_family"]: r for r in rows
    }
    fx = by_family["fx_eod_trend"]
    assert fx["paper_trading_allowed"] is False
    assert fx["live_trading_allowed"] is False


def test_no_strategy_in_registry_is_paper_or_live_allowed():
    rows = strategy_registry.build_registry()
    for r in rows:
        assert r["paper_trading_allowed"] is False, r["strategy_family"]
        assert r["live_trading_allowed"] is False, r["strategy_family"]


# ---------------------------------------------------------------------------
# Safety invariants
# ---------------------------------------------------------------------------
_RESEARCH_MODULE_PATH = Path(fxr.__file__)


def _read(path: Path) -> str:
    return path.read_text(errors="ignore")


def test_no_broker_imports_in_research_module():
    text = _read(_RESEARCH_MODULE_PATH)
    assert "ccxt" not in text
    assert "alpaca" not in text.lower()
    assert "kraken" not in text.lower()
    assert "ig_bank" not in text.lower()
    assert "oanda" not in text.lower()


def test_no_api_key_reads_in_research_module():
    text = _read(_RESEARCH_MODULE_PATH)
    assert "API_KEY" not in text
    assert "API_SECRET" not in text
    assert "os.environ" not in text
    assert "os.getenv" not in text


def test_no_order_placement_in_research_module():
    forbidden = (
        "create_order", "place_order", "submit_order",
        "send_order", "cancel_order", "AddOrder", "CancelOrder",
    )
    text = _read(_RESEARCH_MODULE_PATH)
    for token in forbidden:
        assert token not in text, f"{token} found in research module"


def test_no_paper_or_live_trading_enablement_in_research_module():
    text = _read(_RESEARCH_MODULE_PATH)
    assert "LIVE_TRADING_ENABLED = True" not in text
    assert "paper_trading_allowed = True" not in text
    assert "execution_allowed = True" not in text
    assert "ENABLE_LIVE" not in text
    assert "UNLOCK_TRADING" not in text
    assert "FORCE_TRADE" not in text


def test_safety_lock_remains_locked_after_research_import():
    s = safety_lock.status()
    assert not s.execution_allowed
    assert not s.paper_trading_allowed
    assert not s.kraken_connection_allowed


# ---------------------------------------------------------------------------
# .gitignore must exclude the generated FX EOD trend result CSVs
# ---------------------------------------------------------------------------
def test_gitignore_excludes_results_csv_for_fx_eod_trend():
    text = (REPO_ROOT / ".gitignore").read_text()
    # The existing generic rule covers all four output CSVs.
    assert "results/*.csv" in text
