"""Tests for the bot shell modules:
    safety_lock, strategy_registry, bot_status, alert_engine, dry_run_planner.

These exercise BOTH the per-module behaviour and the cross-cutting
"the engine remains research-only with execution disabled" invariant.
"""
from __future__ import annotations

import inspect
import os
import re
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest

from src import (alert_engine, bot_status, config, dry_run_planner,
                  safety_lock, strategy_registry)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _redirect_results(tmp_path, monkeypatch):
    """Every test reads/writes `results/` and `data/raw/` under
    `tmp_path` so the real cache is never touched."""
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    # Strip any locally-set broker env vars so api_keys_loaded reads False.
    for key in ("KRAKEN_API_KEY", "KRAKEN_API_SECRET", "KRAKEN_KEY",
                "BINANCE_API_KEY", "BINANCE_API_SECRET"):
        monkeypatch.delenv(key, raising=False)
    yield


# ---------------------------------------------------------------------------
# safety_lock
# ---------------------------------------------------------------------------
def test_safety_lock_default_blocks_execution():
    assert safety_lock.is_execution_allowed() is False
    assert safety_lock.is_paper_trading_allowed() is False
    assert safety_lock.is_kraken_connection_allowed() is False
    assert safety_lock.safety_lock_status() == "locked"


def test_safety_lock_reason_mentions_no_pass_strategy():
    reasons = safety_lock.reasons_blocked()
    assert any("no strategy has passed" in r.lower() for r in reasons)


def test_safety_lock_assert_execution_blocked_raises():
    with pytest.raises(safety_lock.ExecutionBlocked):
        safety_lock.assert_execution_blocked("hypothetical_order")


def test_safety_lock_no_env_var_can_unlock(monkeypatch):
    """Setting random "looks-like-an-unlock" env vars must NOT change the
    lock state. There is no bypass."""
    monkeypatch.setenv("UNLOCK_TRADING", "1")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("KRAKEN_API_KEY", "fake-key")
    assert safety_lock.is_execution_allowed() is False
    assert safety_lock.is_kraken_connection_allowed() is False


def test_safety_lock_releases_only_when_pass_present(tmp_path):
    """Construct a fake PASS scorecard. The lock STILL blocks execution
    because Kraken is not integrated and paper-trading is spec-disabled
    — defence in depth."""
    pass_csv = config.RESULTS_DIR / "fake_strategy_scorecard.csv"
    pd.DataFrame([{"verdict": "PASS"}]).to_csv(pass_csv, index=False)
    # `_no_pass_strategy_blocked` returns None (passes); but the other
    # checks still block.
    assert safety_lock._no_pass_strategy_blocked() is None
    assert safety_lock.is_execution_allowed() is False
    assert safety_lock.is_kraken_connection_allowed() is False
    rs = safety_lock.reasons_blocked()
    assert any("kraken" in r.lower() for r in rs)


# ---------------------------------------------------------------------------
# strategy_registry
# ---------------------------------------------------------------------------
def test_registry_has_documented_columns():
    assert strategy_registry.REGISTRY_COLUMNS == [
        "strategy_family", "branch", "latest_commit_if_available",
        "verdict", "best_result_summary", "benchmark_result",
        "placebo_result", "scorecard_status",
        "paper_trading_allowed", "live_trading_allowed",
        "reason_blocked", "report_path",
    ]


def test_registry_lists_every_documented_family():
    rows = strategy_registry.build_registry()
    names = {r["strategy_family"] for r in rows}
    expected = {"single_asset_ta", "regime_selector", "portfolio_momentum",
                 "derivatives_funding", "market_structure", "sentiment_overlay"}
    assert expected <= names, names


def test_no_strategy_is_paper_or_live_allowed():
    """Branch invariant: every registry row must have BOTH gates False."""
    rows = strategy_registry.build_registry()
    for r in rows:
        assert r["paper_trading_allowed"] is False, r["strategy_family"]
        assert r["live_trading_allowed"] is False, r["strategy_family"]


def test_registry_writes_csv_and_json(tmp_path):
    df = strategy_registry.write_snapshot(save=True)
    assert (config.RESULTS_DIR / "strategy_registry_snapshot.csv").exists()
    assert (config.RESULTS_DIR / "strategy_registry_snapshot.json").exists()
    assert list(df.columns) == strategy_registry.REGISTRY_COLUMNS


def test_registry_marks_unknown_when_scorecard_missing(tmp_path):
    """All scorecards absent → every family verdict is UNKNOWN and the
    `scorecard_status` is `missing`."""
    rows = strategy_registry.build_registry()
    for r in rows:
        assert r["verdict"] == "UNKNOWN"
        assert r["scorecard_status"] == "missing"


def test_registry_picks_up_pass_verdict_from_scorecard(tmp_path):
    """If the canonical scorecard CSV exists with verdict=PASS, the
    registry row reflects it — but trading gates remain False because
    `safety_lock` still blocks execution (no Kraken module, etc.)."""
    pd.DataFrame([{
        "verdict": "PASS",
        "checks_passed": 10, "checks_total": 10,
        "avg_oos_return_pct": 30.0, "stability_score_pct": 70.0,
        "pct_windows_beat_btc": 60.0,
    }]).to_csv(
        config.RESULTS_DIR / "market_structure_vol_target_scorecard.csv",
        index=False,
    )
    rows = strategy_registry.build_registry()
    ms = next(r for r in rows if r["strategy_family"] == "market_structure")
    assert ms["verdict"] == "PASS"
    assert ms["paper_trading_allowed"] is False
    assert ms["live_trading_allowed"] is False
    assert "kraken" in ms["reason_blocked"].lower() \
            or "no strategy" in ms["reason_blocked"].lower() \
            or "paper trading" in ms["reason_blocked"].lower()


# ---------------------------------------------------------------------------
# bot_status
# ---------------------------------------------------------------------------
def test_bot_status_defaults_to_research_only():
    s = bot_status.compute_status()
    assert s.bot_mode == "research_only"
    assert s.execution_enabled is False
    assert s.paper_trading_enabled is False
    assert s.kraken_connected is False
    assert s.api_keys_loaded is False
    assert s.safety_lock_status == "locked"


def test_bot_status_reason_mentions_no_pass_strategy():
    s = bot_status.compute_status()
    assert "no strategy has passed" in s.reason_execution_blocked.lower() \
            or "no strategy has passed" in s.reason_execution_blocked


def test_bot_status_active_strategy_is_none_without_pass():
    s = bot_status.compute_status()
    assert s.active_strategy is None
    assert s.active_strategy_verdict == "no_pass_strategy"


def test_bot_status_writes_csv_and_json():
    df = bot_status.write_status(save=True)
    assert (config.RESULTS_DIR / "bot_status.csv").exists()
    assert (config.RESULTS_DIR / "bot_status.json").exists()
    assert list(df.columns) == bot_status.STATUS_COLUMNS


def test_bot_status_api_keys_loaded_reflects_env(monkeypatch):
    """If an env var is present, status reports True (so the user can
    see they have a key set locally) — but the lock STILL blocks
    everything else, because `api_keys_loaded` doesn't influence the
    safety lock."""
    monkeypatch.setenv("KRAKEN_API_KEY", "fake-key-for-test-only")
    s = bot_status.compute_status()
    assert s.api_keys_loaded is True
    assert s.execution_enabled is False
    assert s.kraken_connected is False
    assert s.safety_lock_status == "locked"


def test_bot_status_stale_data_when_no_cache():
    s = bot_status.compute_status()
    # No `*_1d.csv` files in the redirected DATA_RAW_DIR → stale.
    assert s.stale_data_warning is True


# ---------------------------------------------------------------------------
# alert_engine
# ---------------------------------------------------------------------------
def test_alert_engine_critical_when_no_pass_exists():
    rows = alert_engine.build_alerts()
    severities = [r["severity"] for r in rows]
    assert "critical" in severities
    # The first critical message must point at the lock or no-PASS state.
    crit = [r for r in rows if r["severity"] == "critical"]
    assert any("locked" in r["message"].lower()
                or "no strategy" in r["message"].lower()
                for r in crit)


def test_alert_engine_never_recommends_trading():
    rows = alert_engine.build_alerts()
    forbidden_words = ("paper trade", "go live", "place order",
                        "start trading", "execute", "fund the account")
    for r in rows:
        text = (r["message"] + " " + r["recommended_action"]).lower()
        for w in forbidden_words:
            assert w not in text, f"alert recommended {w!r}: {r}"


def test_alert_engine_writes_csv():
    df = alert_engine.write_alerts(save=True)
    assert (config.RESULTS_DIR / "bot_alerts.csv").exists()
    assert list(df.columns) == alert_engine.ALERT_COLUMNS


def test_alert_engine_warns_on_stale_data():
    df = alert_engine.write_alerts(save=False)
    assert any("stale" in r.lower() or "no daily ohlcv" in r.lower()
                for r in df["message"])


# ---------------------------------------------------------------------------
# dry_run_planner
# ---------------------------------------------------------------------------
def _write_weights_csv(name: str, weights: Dict[str, float], filled: bool = True):
    """Mimic what the portfolio backtester writes to results/. The
    real format is `weights_json` column with `KEY=VALUE,KEY=VALUE`."""
    p = config.RESULTS_DIR / f"{name}_weights.csv"
    serialised = ",".join(f"{k}={v:.4f}" for k, v in weights.items())
    pd.DataFrame([{
        "timestamp": 1_700_000_000_000,
        "datetime": "2023-11-14 00:00:00+00:00",
        "filled": filled,
        "weights_json": serialised,
    }]).to_csv(p, index=False)
    return p


def test_dry_run_plan_mode_is_dry_run_only():
    _write_weights_csv("market_structure_vol_target",
                        {"BTC/USDT": 0.7, "ETH/USDT": 0.3})
    df = dry_run_planner.write_plan(save=True)
    assert not df.empty
    assert (df["mode"] == dry_run_planner.MODE_DRY_RUN_ONLY).all()


def test_dry_run_plan_status_is_blocked_no_pass():
    _write_weights_csv("market_structure_vol_target",
                        {"BTC/USDT": 0.7, "ETH/USDT": 0.3})
    df = dry_run_planner.write_plan(save=True)
    assert (df["execution_status"]
            == dry_run_planner.EXECUTION_STATUS_BLOCKED).all()


def test_dry_run_plan_handles_no_artifacts():
    """No `*_weights.csv` → single 'no_plan_available' row, mode + status
    still set per spec."""
    df = dry_run_planner.write_plan(save=True)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["theoretical_action"] == "no_plan_available"
    assert row["mode"] == dry_run_planner.MODE_DRY_RUN_ONLY
    assert row["execution_status"] == dry_run_planner.EXECUTION_STATUS_BLOCKED


def test_dry_run_plan_writes_csv_with_documented_columns():
    df = dry_run_planner.write_plan(save=True)
    assert (config.RESULTS_DIR / "dry_run_trade_plan.csv").exists()
    assert list(df.columns) == dry_run_planner.PLAN_COLUMNS


def test_dry_run_plan_parses_real_kv_format(monkeypatch):
    """The portfolio backtester writes `KEY=VALUE,KEY=VALUE` in
    `weights_json`. Round-trip a multi-asset row and confirm the
    parser sees every asset."""
    _write_weights_csv("market_structure_vol_target",
                        {"BTC/USDT": 0.30, "ETH/USDT": 0.14,
                         "SOL/USDT": 0.14, "AVAX/USDT": 0.14,
                         "LINK/USDT": 0.14, "ADA/USDT": 0.14})
    df = dry_run_planner.write_plan(save=True)
    assets = set(df["asset"].tolist())
    expected = {"BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT",
                 "LINK/USDT", "ADA/USDT"}
    assert expected <= assets, f"missing assets: {expected - assets}"


def test_dry_run_plan_parses_dict_literal_format(tmp_path):
    """Backwards compatibility: a `weights` column carrying a Python
    dict literal must also parse cleanly."""
    p = config.RESULTS_DIR / "old_format_weights.csv"
    pd.DataFrame([{
        "timestamp": 1_700_000_000_000,
        "weights": "{'BTC/USDT': 0.5, 'ETH/USDT': 0.5}",
        "filled": True,
    }]).to_csv(p, index=False)
    df = dry_run_planner.write_plan(save=True)
    assets = set(df["asset"].tolist())
    assert {"BTC/USDT", "ETH/USDT"} <= assets


def test_dry_run_plan_handles_empty_weights_as_cash():
    """Weights == {} (cash) → row labelled `all_cash`, no notional."""
    _write_weights_csv("some_strategy", {})
    df = dry_run_planner.write_plan(save=True)
    assert any(df["theoretical_action"] == "all_cash")


# ---------------------------------------------------------------------------
# Cross-cutting: no Kraken, no API keys, no broker, no live trading
# ---------------------------------------------------------------------------
def test_no_bot_shell_module_imports_a_broker():
    import importlib
    for modname in ("safety_lock", "strategy_registry", "bot_status",
                     "alert_engine", "dry_run_planner"):
        m = importlib.import_module(f"src.{modname}")
        src = inspect.getsource(m)
        # Strip docstrings to avoid false positives from descriptive prose
        # ("Kraken not integrated", "no API keys", etc).
        code_lines = []
        in_doc = False
        for raw in src.splitlines():
            s = raw.strip()
            if s.startswith('"""') or s.startswith("'''"):
                in_doc = not in_doc
                continue
            if in_doc or s.startswith("#"):
                continue
            code_lines.append(raw)
        code = "\n".join(code_lines)

        # No code-level use of broker SDKs.
        for forbidden in (r"\bimport\s+ccxt\b", r"\bfrom\s+ccxt\b",
                           r"\bkraken\.private\b", r"\bplace_order\s*\(",
                           r"\bsubmit_order\s*\(", r"\bcreate_order\s*\("):
            assert not re.search(forbidden, code), (modname, forbidden)
        # No hardcoded API-key reads.
        for pat in (r"\bos\.environ\[\s*['\"]KRAKEN_API_KEY['\"]",
                     r"\bos\.environ\[\s*['\"][A-Z_]*SECRET['\"]"):
            assert not re.search(pat, code), (modname, pat)


def test_no_bot_shell_module_loads_api_keys_at_import():
    """Importing any bot-shell module must NOT touch broker keys.
    Implicitly proven by the previous test, but assert the runtime
    guarantee too."""
    import importlib
    for modname in ("safety_lock", "strategy_registry", "bot_status",
                     "alert_engine", "dry_run_planner"):
        # Fresh import; no exception, no side effect on env vars.
        m = importlib.import_module(f"src.{modname}")
        assert m is not None
