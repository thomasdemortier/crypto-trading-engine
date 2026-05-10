"""Tests for `src/research_dashboard.py`.

These run entirely offline. They verify:
    * Archived branch metadata covers every known archived branch.
    * Every archived entry carries `merge_allowed=False` (locked).
    * No next-step action recommends trading / Kraken / API keys.
    * Module contains no broker imports, no Kraken private endpoints,
      no order-placement strings, no API-key reads.
    * Helpers degrade gracefully when results files are missing.
    * Strategy registry continues to expose paper/live=False.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from src import config, research_dashboard as rd, strategy_registry


# ---------------------------------------------------------------------------
# Archived branch metadata
# ---------------------------------------------------------------------------
EXPECTED_ARCHIVED_BRANCHES = {
    "research/fail-1-funding-derivatives",
    "research/strategy-2-market-structure",
    "research/strategy-3-sentiment-fear-greed",
    "research/strategy-4-drawdown-targeted-btc",
    "research/strategy-5-paid-positioning-data-audit",
    "research/strategy-6-funding-basis-carry",
    "research/strategy-7-relative-value-btc-eth",
    "research/strategy-8-paid-data-decision-audit",
    "research/strategy-9-free-open-data-reaudit",
    "research/portfolio-rebalancing-strategy-v1",
}


def test_archived_branches_cover_every_known_research_branch():
    actual = {b.branch for b in rd.ARCHIVED_BRANCHES}
    assert EXPECTED_ARCHIVED_BRANCHES <= actual, (
        f"missing branches: "
        f"{EXPECTED_ARCHIVED_BRANCHES - actual}"
    )


def test_every_archived_branch_has_merge_allowed_false():
    """The single most important test in this file. A future edit must
    not flip this flag for any archived branch — the dashboard renders
    a hard-red banner on it."""
    for entry in rd.ARCHIVED_BRANCHES:
        assert entry.merge_allowed is False, (
            f"merge_allowed must be False for {entry.branch}"
        )


def test_archived_branch_kinds_are_in_locked_vocabulary():
    for entry in rd.ARCHIVED_BRANCHES:
        assert entry.kind in rd.ARCHIVED_KINDS, (
            f"unknown kind {entry.kind!r} for {entry.branch}"
        )


def test_archived_branch_verdicts_are_in_locked_vocabulary():
    for entry in rd.ARCHIVED_BRANCHES:
        assert entry.verdict in rd.ARCHIVED_VERDICTS, (
            f"unknown verdict {entry.verdict!r} for {entry.branch}"
        )


def test_archived_timeline_dataframe_columns_locked():
    df = rd.archived_timeline_dataframe()
    assert list(df.columns) == [
        "branch", "kind", "verdict", "one_line_reason",
        "report_path", "merge_allowed",
        "paper_trading_allowed", "live_trading_allowed",
        "archive_commit", "archive_tag",
    ]
    assert (~df["merge_allowed"].astype(bool)).all()
    assert (~df["paper_trading_allowed"].astype(bool)).all()
    assert (~df["live_trading_allowed"].astype(bool)).all()
    assert len(df) == len(rd.ARCHIVED_BRANCHES)


# ---------------------------------------------------------------------------
# Portfolio rebalancing FAIL — archive row invariants
# ---------------------------------------------------------------------------
def _portfolio_rebalancing_entry() -> rd.ArchivedBranch:
    matches = [b for b in rd.ARCHIVED_BRANCHES
                if b.branch == "research/portfolio-rebalancing-strategy-v1"]
    assert len(matches) == 1, (
        "exactly one archive row expected for portfolio rebalancing"
    )
    return matches[0]


def test_portfolio_rebalancing_archive_entry_present():
    e = _portfolio_rebalancing_entry()
    assert e.kind == rd.KIND_STRATEGY
    assert e.verdict == "FAIL"


def test_portfolio_rebalancing_archive_merge_allowed_false():
    e = _portfolio_rebalancing_entry()
    assert e.merge_allowed is False


def test_portfolio_rebalancing_archive_paper_and_live_false():
    e = _portfolio_rebalancing_entry()
    assert e.paper_trading_allowed is False
    assert e.live_trading_allowed is False


def test_portfolio_rebalancing_archive_carries_commit_and_tag():
    e = _portfolio_rebalancing_entry()
    assert e.archive_commit == "19b1268"
    assert e.archive_tag == "fail-portfolio-rebalancing-strategy-v1"


def test_portfolio_rebalancing_archive_reason_mentions_locked_facts():
    """The reason text must record both gates that failed (DD pp + placebo
    return) — those are the load-bearing facts of this archive entry."""
    e = _portfolio_rebalancing_entry()
    low = e.one_line_reason.lower()
    assert "10.78" in low
    assert "15 pp" in low
    assert "placebo" in low


def test_portfolio_rebalancing_archive_report_path_marked_archived():
    """Strategy code + report live ONLY on the archived branch — the
    report_path should explicitly say so so a checkout of `main`
    doesn't render a misleading 'file not found' as the canonical
    error message."""
    e = _portfolio_rebalancing_entry()
    assert e.report_path is not None
    low = e.report_path.lower()
    assert "archived" in low or "archive" in low


def test_strategy_implementation_files_not_present_on_main():
    """Sanity check: this archive branch must NOT have copied the
    strategy implementation across. Failed strategy code stays only
    on the archived branch."""
    repo_root = Path(__file__).resolve().parents[1]
    forbidden = (
        repo_root / "src" / "strategies"
                   / "portfolio_rebalancing_allocator.py",
        repo_root / "src" / "portfolio_rebalancing_research.py",
    )
    for p in forbidden:
        assert not p.exists(), (
            f"failed strategy file leaked into main: {p}"
        )


# ---------------------------------------------------------------------------
# Executive state
# ---------------------------------------------------------------------------
def test_executive_state_keys_locked():
    s = rd.executive_state()
    expected = {
        "project_mode", "active_strategy", "production_baseline",
        "safety_lock_status", "execution_allowed", "kraken_connected",
        "paper_trading", "live_trading", "next_allowed_action",
        "reasons_blocked",
    }
    assert set(s.keys()) == expected


def test_executive_state_reflects_locked_safety():
    s = rd.executive_state()
    # The safety lock must remain locked through this dashboard.
    assert s["safety_lock_status"] == "locked"
    assert s["execution_allowed"] is False
    assert s["kraken_connected"] is False
    assert s["paper_trading"] is False
    assert s["live_trading"] is False
    assert s["active_strategy"] == "none"
    assert s["production_baseline"] == "BTC buy-and-hold"


# ---------------------------------------------------------------------------
# Strategy verdict board + registry flags
# ---------------------------------------------------------------------------
def test_strategy_verdict_board_has_registry_columns():
    df = rd.strategy_verdict_board()
    if df.empty:
        return
    assert set(strategy_registry.REGISTRY_COLUMNS) <= set(df.columns)


def test_strategy_verdict_board_paper_and_live_off_for_every_row():
    df = rd.strategy_verdict_board()
    if df.empty:
        return
    assert rd.all_strategy_flags_off(df) is True
    assert (~df["paper_trading_allowed"].astype(bool)).all()
    assert (~df["live_trading_allowed"].astype(bool)).all()


def test_all_strategy_flags_off_predicate_handles_empty():
    assert rd.all_strategy_flags_off(pd.DataFrame()) is True


def test_all_strategy_flags_off_predicate_returns_false_when_flag_set():
    df = pd.DataFrame([{
        "paper_trading_allowed": True,
        "live_trading_allowed": False,
    }])
    assert rd.all_strategy_flags_off(df) is False


# ---------------------------------------------------------------------------
# Risk dashboard helpers — fail-soft
# ---------------------------------------------------------------------------
def test_latest_results_state_handles_missing_results_dir(tmp_path):
    s = rd.latest_results_state(results_dir=tmp_path)
    assert s["results_dir_exists"] is True
    # No expected files present.
    assert s["files_present"] == []
    assert set(rd.RISK_RESULT_FILES) <= set(s["files_missing"])
    assert s["equity_n_bars"] == 0


def test_latest_results_state_picks_up_present_files(tmp_path):
    # Drop a stub equity_curve.csv into the temp results dir.
    eq_path = tmp_path / "equity_curve.csv"
    pd.DataFrame({
        "timestamp": [1700000000000, 1700086400000],
        "datetime": ["2023-11-14T00:00:00+00:00",
                       "2023-11-15T00:00:00+00:00"],
        "equity": [10000.0, 10100.0],
    }).to_csv(eq_path, index=False)
    s = rd.latest_results_state(results_dir=tmp_path)
    assert "equity_curve.csv" in s["files_present"]
    assert s["equity_n_bars"] == 2
    assert s["equity_freshness_days"] is not None
    assert s["equity_freshness_days"] >= 0


def test_baseline_metrics_default_when_missing(tmp_path):
    out = rd.baseline_metrics(results_dir=tmp_path)
    assert out["available"] is False
    assert out["n_bars"] == 0


def test_baseline_metrics_compute_on_synthetic_curve(tmp_path):
    pd.DataFrame({
        "timestamp": [1700000000000, 1700086400000, 1700172800000],
        "equity": [10000.0, 11000.0, 9500.0],
    }).to_csv(tmp_path / "equity_curve.csv", index=False)
    out = rd.baseline_metrics(results_dir=tmp_path)
    assert out["available"] is True
    assert out["n_bars"] == 3
    assert out["total_return_pct"] == pytest.approx(-5.0)
    # Drawdown from 11000 down to 9500 = -13.6%.
    assert out["max_drawdown_pct"] < 0


def test_decision_journal_latest_row_handles_missing(tmp_path):
    assert rd.decision_journal_latest_row(results_dir=tmp_path) is None


def test_alert_history_latest_rows_handles_missing(tmp_path):
    out = rd.alert_history_latest_rows(results_dir=tmp_path)
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_report_text_handles_missing_path():
    out = rd.report_text("does/not/exist.md")
    assert "not found" in out.lower()


def test_report_text_handles_empty_path():
    out = rd.report_text("")
    assert "no report" in out.lower()


# ---------------------------------------------------------------------------
# Action policy — no recommendation must endorse trading
# ---------------------------------------------------------------------------
def test_no_allowed_action_endorses_trading():
    forbidden = ("paper trade", "paper-trade", "live trade",
                  "live-trade", "place order", "submit order",
                  "execute trade", "kraken connect")
    for action in rd.ALLOWED_NEXT_ACTIONS:
        lower = action.lower()
        for bad in forbidden:
            assert bad not in lower, (
                f"allowed action {action!r} contains forbidden token "
                f"{bad!r}"
            )


def test_forbidden_actions_listed_explicitly():
    """The dashboard must explicitly enumerate what NOT to do, so a
    future maintainer cannot quietly add an action to ALLOWED."""
    expected = {
        "paper trading", "live trading", "Kraken connection",
        "API key entry into the dashboard", "order placement",
        "any execution surface",
    }
    assert set(rd.FORBIDDEN_NEXT_ACTIONS) == expected


# ---------------------------------------------------------------------------
# Source-level safety invariants
# ---------------------------------------------------------------------------
_HELPER_SOURCE = (Path(__file__).resolve().parents[1]
                    / "src" / "research_dashboard.py").read_text()


def test_no_broker_imports():
    """The helper may legitimately reference safety-lock functions whose
    names mention 'kraken' (e.g. `is_kraken_connection_allowed`). What
    we forbid is *broker SDK imports* — i.e. literal import lines."""
    bad_import_patterns = (
        re.compile(r"^\s*import\s+ccxt\b", re.MULTILINE),
        re.compile(r"^\s*from\s+ccxt", re.MULTILINE),
        re.compile(r"^\s*import\s+kraken\b", re.MULTILINE),
        re.compile(r"^\s*from\s+kraken\b", re.MULTILINE),
        re.compile(r"\bbinance\.client\b"),
        re.compile(r"\bbybit\.client\b"),
        re.compile(r"\bderibit\.client\b"),
        re.compile(r"\bokx\.client\b"),
    )
    for pat in bad_import_patterns:
        assert pat.search(_HELPER_SOURCE) is None, pat.pattern


def test_no_kraken_private_endpoints():
    forbidden = (
        re.compile(r"\bkraken\.private\b"),
        re.compile(r"\bkraken_private\b"),
        re.compile(r"['\"]/0/private/"),
        re.compile(r"\bAddOrder\b"),
        re.compile(r"\bCancelOrder\b"),
    )
    for pat in forbidden:
        assert pat.search(_HELPER_SOURCE) is None, pat.pattern


def test_no_api_key_reads():
    forbidden = (
        re.compile(r"\bos\.environ\.get\([\"'](?:[A-Z_]*API[_-]?KEY)[\"']"),
        re.compile(r"\bos\.environ\[[\"'](?:[A-Z_]*API[_-]?KEY)[\"']"),
        re.compile(r"\bos\.getenv\([\"'](?:[A-Z_]*API[_-]?KEY)[\"']"),
        re.compile(r"\bAPI[_-]?KEY\s*=\s*['\"][A-Za-z0-9_\-]{8,}['\"]"),
    )
    for pat in forbidden:
        assert pat.search(_HELPER_SOURCE) is None, pat.pattern


def test_no_order_placement():
    forbidden = (
        re.compile(r"\bcreate_order\s*\("),
        re.compile(r"\bplace_order\s*\("),
        re.compile(r"\bsubmit_order\s*\("),
        re.compile(r"\bsend_order\s*\("),
        re.compile(r"\bcancel_order\s*\("),
        re.compile(r"\bdef\s+(?:create|place|submit|send|cancel)_order\b"),
    )
    for pat in forbidden:
        assert pat.search(_HELPER_SOURCE) is None, pat.pattern


def test_no_network_calls_in_helper():
    bad = ("urllib.request", "urllib.urlopen", "requests.get(",
            "httpx.get(", "aiohttp")
    for s in bad:
        assert s not in _HELPER_SOURCE, s


def test_no_file_writes_in_helper():
    """The helper is read-only. No `to_csv` / `write_text` etc."""
    bad = (".to_csv(", ".to_json(", ".write_text(",
            ".write_bytes(", "open(", ".write(")
    for s in bad:
        assert s not in _HELPER_SOURCE, (
            f"research_dashboard helper must be read-only: {s!r}"
        )
