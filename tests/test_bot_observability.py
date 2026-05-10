"""Tests for the bot-observability layer:
    bot_status_history, alert_history, decision_journal, system_health
+ the unlock procedure document.

Cross-cutting: every module remains read-only, never recommends
trading, never imports a broker.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest

from src import (alert_engine, alert_history, bot_status,
                  bot_status_history, config, decision_journal,
                  dry_run_planner, safety_lock, strategy_registry,
                  system_health)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _redirect_paths(tmp_path, monkeypatch):
    """Every test reads/writes under tmp_path; the real cache is never
    touched. We also strip broker env vars."""
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "DATA_PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    for sub in ("results", "raw", "processed", "logs", "src", "tests",
                 "reports", "docs"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    for key in ("KRAKEN_API_KEY", "KRAKEN_API_SECRET", "KRAKEN_KEY",
                "BINANCE_API_KEY", "BINANCE_API_SECRET"):
        monkeypatch.delenv(key, raising=False)
    yield


# ---------------------------------------------------------------------------
# bot_status_history
# ---------------------------------------------------------------------------
def test_status_history_creates_csv():
    df = bot_status_history.record_status(save=True)
    p = config.RESULTS_DIR / bot_status_history.OUTPUT_FILENAME
    assert p.exists()
    assert list(df.columns) == bot_status_history.HISTORY_COLUMNS
    assert len(df) == 1


def test_status_history_appends_safely():
    bot_status_history.record_status(save=True)
    df2 = bot_status_history.record_status(save=True)
    # Two distinct calls — at least 1 row, possibly 2 (depending on
    # whether the timestamps differ down to the nanosecond).
    assert len(df2) >= 1
    # Schema must remain stable across appends.
    assert list(df2.columns) == bot_status_history.HISTORY_COLUMNS


def test_status_history_dedupes_identical_timestamp(monkeypatch):
    """Two `record_status` calls with the same timestamp must collapse
    to one row, not duplicate. We monkey-patch the module's
    `_utcnow_iso` indirection rather than subclassing pandas'
    `pd.Timestamp` (subclassing the Cython class fails on Python 3.12
    + pandas 2.2+ with `TypeError: type ... is not an acceptable base
    type`)."""
    frozen_iso = "2026-05-09T18:54:16+00:00"
    monkeypatch.setattr(bot_status_history, "_utcnow_iso",
                         lambda: frozen_iso)
    bot_status_history.record_status(save=True)
    df = bot_status_history.record_status(save=True)
    # Only one row should carry that exact iso — the second call should
    # have replaced the first, not appended.
    assert (df["timestamp"] == frozen_iso).sum() == 1
    assert len(df) == 1


def test_status_history_never_records_api_keys(monkeypatch):
    """Even if env vars are set, the history columns are booleans /
    strings — no key value can leak."""
    monkeypatch.setenv("KRAKEN_API_KEY", "VERY-SECRET-DO-NOT-LEAK")
    df = bot_status_history.record_status(save=True)
    p = config.RESULTS_DIR / bot_status_history.OUTPUT_FILENAME
    text = p.read_text()
    assert "VERY-SECRET-DO-NOT-LEAK" not in text
    # api_keys_loaded is the boolean truthy reflection only.
    assert bool(df["api_keys_loaded"].iloc[-1]) is True


def test_status_history_handles_missing_file_safely():
    """No file present → a fresh frame is created."""
    p = config.RESULTS_DIR / bot_status_history.OUTPUT_FILENAME
    assert not p.exists()
    df = bot_status_history.record_status(save=True)
    assert p.exists()
    assert len(df) == 1


# ---------------------------------------------------------------------------
# alert_history
# ---------------------------------------------------------------------------
def test_alert_history_creates_csv():
    df = alert_history.record_alerts(save=True)
    p = config.RESULTS_DIR / alert_history.OUTPUT_FILENAME
    assert p.exists()
    assert list(df.columns) == alert_history.HISTORY_COLUMNS


def test_alert_history_dedupes_repeat_alerts():
    """Calling record_alerts twice with the same engine output must
    NOT produce duplicate hashes — `occurrence_count` increases."""
    df1 = alert_history.record_alerts(save=True)
    n1 = len(df1)
    df2 = alert_history.record_alerts(save=True)
    # Same number of unique hashes; at least one occurrence_count
    # should have been bumped (so total count rose).
    assert df1["alert_hash"].nunique() == df2["alert_hash"].nunique()
    assert df2["occurrence_count"].sum() > df1["occurrence_count"].sum()


def test_alert_history_marks_active_on_current_run():
    df = alert_history.record_alerts(save=True)
    # Every row from the alert engine on this run is currently active.
    assert df["active"].astype(bool).all()


def test_alert_history_keeps_resolved_alerts(monkeypatch):
    """Inject a synthetic prior row whose hash is NOT in the current
    alert set; record_alerts must keep that row but set active=False."""
    p = config.RESULTS_DIR / alert_history.OUTPUT_FILENAME
    fake_hash = "deadbeef00000000"
    pd.DataFrame([{
        "timestamp": "2024-01-01T00:00:00+00:00",
        "severity": "warning", "category": "synthetic",
        "message": "an old alert that no longer fires",
        "recommended_action": "continue research; do not trade",
        "alert_hash": fake_hash,
        "first_seen": "2024-01-01T00:00:00+00:00",
        "last_seen": "2024-01-01T00:00:00+00:00",
        "occurrence_count": 5, "active": True,
    }], columns=alert_history.HISTORY_COLUMNS).to_csv(p, index=False)
    df = alert_history.record_alerts(save=True)
    old = df[df["alert_hash"] == fake_hash]
    assert len(old) == 1
    assert bool(old["active"].iloc[0]) is False
    # Count is preserved, not bumped.
    assert int(old["occurrence_count"].iloc[0]) == 5


def test_alert_history_never_recommends_trading():
    df = alert_history.record_alerts(save=True)
    forbidden = ("paper trade", "go live", "place order",
                  "start trading", "execute", "fund the account")
    for _, row in df.iterrows():
        text = (str(row["message"]) + " " +
                  str(row["recommended_action"])).lower()
        for w in forbidden:
            assert w not in text, f"{w!r} in alert: {dict(row)}"


# ---------------------------------------------------------------------------
# decision_journal
# ---------------------------------------------------------------------------
def test_decision_journal_creates_csv():
    df = decision_journal.record_decision(save=True)
    p = config.RESULTS_DIR / decision_journal.OUTPUT_FILENAME
    assert p.exists()
    assert list(df.columns) == decision_journal.JOURNAL_COLUMNS


def test_decision_journal_execution_status_always_blocked():
    df = decision_journal.record_decision(save=True)
    assert (df["execution_status"]
            == decision_journal.EXECUTION_STATUS_BLOCKED).all()


def test_decision_journal_says_no_valid_strategy_when_no_pass():
    """No PASS scorecard exists in the redirected results dir → decision
    must be NO_VALID_STRATEGY."""
    df = decision_journal.record_decision(save=True)
    assert df["decision"].iloc[-1] == decision_journal.DECISION_NO_VALID_STRATEGY


def test_decision_journal_says_blocked_when_pass_exists_but_lock_holds():
    """Inject a synthetic PASS scorecard under one of the registry's
    expected filenames. The lock still blocks (Kraken / paper / live
    flag) → decision = EXECUTION_BLOCKED."""
    pd.DataFrame([{
        "verdict": "PASS",
        "checks_passed": 10, "checks_total": 10,
    }]).to_csv(
        # match the registry's expected scorecard filename for
        # `market_structure` family.
        config.RESULTS_DIR / "market_structure_vol_target_scorecard.csv",
        index=False,
    )
    # Provide a fresh daily cache so stale_data_warning is False.
    fresh_ts_ms = int(pd.Timestamp.utcnow().value // 10**6)
    (config.DATA_RAW_DIR / "BTC_USDT_1d.csv").write_text(
        f"timestamp,close\n{fresh_ts_ms},30000\n"
    )
    df = decision_journal.record_decision(save=True)
    assert df["decision"].iloc[-1] == decision_journal.DECISION_EXECUTION_BLOCKED


def test_decision_journal_decision_is_in_allowed_set():
    df = decision_journal.record_decision(save=True)
    assert (df["decision"]
            .isin(decision_journal.ALLOWED_DECISIONS)).all()


def test_decision_journal_appends_safely():
    decision_journal.record_decision(save=True)
    df = decision_journal.record_decision(save=True)
    assert len(df) == 2


def test_decision_journal_never_calls_a_broker():
    """Source-level: no broker imports / order calls / private endpoints."""
    src = inspect.getsource(decision_journal)
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
    for pat in (r"\bimport\s+ccxt\b", r"\bfrom\s+ccxt\b",
                 r"\bplace_order\s*\(", r"\bsubmit_order\s*\(",
                 r"\bcreate_order\s*\("):
        assert not re.search(pat, code), pat


# ---------------------------------------------------------------------------
# system_health
# ---------------------------------------------------------------------------
def test_system_health_writes_csv_and_json():
    df = system_health.write_health(save=True)
    assert (config.RESULTS_DIR / system_health.OUTPUT_CSV).exists()
    assert (config.RESULTS_DIR / system_health.OUTPUT_JSON).exists()
    assert list(df.columns) == system_health.HEALTH_COLUMNS


def test_system_health_has_safety_lock_check():
    rows = system_health.run_health_checks()
    by_name = {r["check_name"]: r for r in rows}
    assert "safety_lock_locked" in by_name
    # The lock is locked in test environment → status is PASS.
    assert by_name["safety_lock_locked"]["status"] == system_health.STATUS_PASS


def test_system_health_confirms_no_kraken_module():
    rows = system_health.run_health_checks()
    by_name = {r["check_name"]: r for r in rows}
    r = by_name["no_kraken_execution_module"]
    assert r["status"] == system_health.STATUS_PASS


def test_system_health_detects_simulated_execution_module(tmp_path):
    """If a kraken_execution.py-style file appears in src/, the check
    must FAIL with critical severity."""
    src = config.REPO_ROOT / "src"
    (src / "kraken_execution.py").write_text("# simulated rogue module\n")
    rows = system_health.run_health_checks()
    by_name = {r["check_name"]: r for r in rows}
    r = by_name["no_kraken_execution_module"]
    assert r["status"] == system_health.STATUS_FAIL
    assert r["severity"] == "critical"


def test_system_health_no_api_keys_check_passes_on_clean_repo():
    """Real `src/` content carries no broker key literals."""
    # Create a benign src file in the redirected REPO_ROOT.
    (config.REPO_ROOT / "src" / "benign.py").write_text(
        "x = 'some_value_with_no_keys'\n"
    )
    rows = system_health.run_health_checks()
    by_name = {r["check_name"]: r for r in rows}
    r = by_name["no_api_keys_in_tracked_files"]
    assert r["status"] == system_health.STATUS_PASS


def test_system_health_detects_hardcoded_api_key(tmp_path):
    """A literal `KRAKEN_API_KEY = "actual_key"` in src/ must FAIL with
    critical severity."""
    (config.REPO_ROOT / "src" / "leaky.py").write_text(
        'KRAKEN_API_KEY = "AAAAAAAA-BBBBBBBB-CCCCCCCC-DDDDDDDD"\n'
    )
    rows = system_health.run_health_checks()
    by_name = {r["check_name"]: r for r in rows}
    r = by_name["no_api_keys_in_tracked_files"]
    assert r["status"] == system_health.STATUS_FAIL
    assert r["severity"] == "critical"


def test_system_health_status_set_is_documented():
    rows = system_health.run_health_checks()
    for r in rows:
        assert r["status"] in (
            system_health.STATUS_PASS,
            system_health.STATUS_WARN,
            system_health.STATUS_FAIL,
        )


# ---------------------------------------------------------------------------
# unlock_procedure document
# ---------------------------------------------------------------------------
def _real_repo_root() -> Path:
    """Find the repo root regardless of test redirection (the autouse
    fixture redirects config.REPO_ROOT to tmp_path)."""
    return Path(__file__).resolve().parents[1]


def test_unlock_procedure_document_exists():
    p = _real_repo_root() / "docs" / "unlock_procedure.md"
    assert p.exists(), f"missing: {p}"
    text = p.read_text()
    assert len(text) > 1000  # substantive document


def test_unlock_procedure_says_no_env_var_can_unlock():
    p = _real_repo_root() / "docs" / "unlock_procedure.md"
    text = p.read_text().lower()
    # The document must explicitly state no env var / config flag /
    # secrets entry can unlock trading.
    found = any(phrase in text for phrase in (
        "no environment variable", "no config flag",
        "no env var", "environment variable, secrets",
    ))
    assert found, "doc must say no environment variable can unlock trading"


def test_unlock_procedure_lists_required_gates():
    p = _real_repo_root() / "docs" / "unlock_procedure.md"
    text = p.read_text().lower()
    # Each numbered gate in the spec must be findable somewhere in the
    # document. We do a soft string match — full prose isn't required.
    for keyword in ("strategy reaches pass", "beats btc",
                     "beats the equal-weight basket", "beats the placebo",
                     "stability", "dry-run", "paper-trading",
                     "kraken connector", "withdrawals disabled",
                     "explicit code change"):
        assert keyword in text, f"unlock procedure missing: {keyword!r}"


# ---------------------------------------------------------------------------
# Cross-cutting safety
# ---------------------------------------------------------------------------
def test_no_observability_module_imports_a_broker():
    """No bot-observability module imports ccxt or order plumbing."""
    import importlib
    for modname in ("bot_status_history", "alert_history",
                     "decision_journal", "system_health"):
        m = importlib.import_module(f"src.{modname}")
        src = inspect.getsource(m)
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
        for pat in (r"\bimport\s+ccxt\b", r"\bfrom\s+ccxt\b",
                     r"\bplace_order\s*\(", r"\bsubmit_order\s*\(",
                     r"\bcreate_order\s*\("):
            assert not re.search(pat, code), (modname, pat)


def test_no_observability_module_reads_api_key_value(monkeypatch):
    """Setting env vars must not change behavior beyond the boolean
    `api_keys_loaded` flag in bot_status. Importing each module is
    side-effect-free."""
    monkeypatch.setenv("KRAKEN_API_KEY", "fake-but-non-empty")
    import importlib
    for modname in ("bot_status_history", "alert_history",
                     "decision_journal", "system_health"):
        m = importlib.import_module(f"src.{modname}")
        # Grep source for direct reads of the actual key VALUE (not
        # a presence-only `os.environ.get(...)` call which is fine).
        code = inspect.getsource(m)
        # Direct subscription of the env value is suspicious.
        assert "os.environ['KRAKEN_API_KEY']" not in code, modname
        assert 'os.environ["KRAKEN_API_KEY"]' not in code, modname


def test_safety_lock_remains_locked_after_observability_runs():
    """Running every observability module must NOT change the lock state."""
    bot_status_history.record_status(save=True)
    alert_history.record_alerts(save=True)
    decision_journal.record_decision(save=True)
    system_health.write_health(save=True)
    assert safety_lock.is_execution_allowed() is False
    assert safety_lock.safety_lock_status() == "locked"
