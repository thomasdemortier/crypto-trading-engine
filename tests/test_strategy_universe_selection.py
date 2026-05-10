"""Tests for `src/strategy_universe_selection.py`. All offline."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from src import safety_lock, strategy_universe_selection as sus


# ---------------------------------------------------------------------------
# Schema + universes covered
# ---------------------------------------------------------------------------
def test_table_columns_locked():
    assert sus.TABLE_COLUMNS == [
        "universe", "score", "rank",
        "data_score", "execution_score", "edge_score",
        "complexity_score", "risk_score",
        "recommended_next_action", "decision_status", "notes",
    ]


def test_decision_status_vocabulary_locked():
    assert sus.DECISION_STATUSES == (
        "RECOMMENDED", "WATCHLIST", "NOT_NOW", "REJECTED",
    )


def test_all_four_universes_present():
    expected = {
        "Forex",
        "Crypto spot",
        "Crypto derivatives / perps",
        "Portfolio rebalancing / risk allocation",
    }
    actual = {a.universe for a in sus.assessments()}
    assert expected == actual


def test_each_assessment_has_locked_fields():
    for a in sus.assessments():
        assert isinstance(a.data_score, int)
        assert 0 <= a.data_score <= 10
        assert 0 <= a.execution_score <= 10
        assert 0 <= a.edge_score <= 10
        assert 0 <= a.complexity_score <= 10
        assert 0 <= a.risk_score <= 10
        assert a.decision_status in sus.DECISION_STATUSES
        assert a.recommended_next_action
        assert a.notes
        assert a.pass_criteria
        assert a.fail_criteria


# ---------------------------------------------------------------------------
# Composite score + ranking determinism
# ---------------------------------------------------------------------------
def test_composite_score_is_equal_weighted_average():
    for a in sus.assessments():
        expected = (
            a.data_score + a.execution_score + a.edge_score
            + a.complexity_score + a.risk_score
        ) / 5.0
        assert sus.composite_score(a) == pytest.approx(expected)


def test_rank_universes_returns_all_four_rows():
    df = sus.rank_universes()
    assert len(df) == 4
    assert set(df["universe"]) == {a.universe for a in sus.assessments()}


def test_rank_universes_columns_locked():
    df = sus.rank_universes()
    assert list(df.columns) == sus.TABLE_COLUMNS


def test_rank_is_deterministic():
    """Calling rank_universes twice must yield the same ordering."""
    a = sus.rank_universes()
    b = sus.rank_universes()
    pd.testing.assert_frame_equal(a, b)


def test_ranks_are_unique_and_consecutive():
    df = sus.rank_universes()
    assert sorted(df["rank"].tolist()) == [1, 2, 3, 4]


def test_recommended_universe_appears_in_table():
    df = sus.rank_universes()
    rec = df[df["decision_status"] == "RECOMMENDED"]
    # Project policy: exactly one RECOMMENDED universe.
    assert len(rec) == 1
    assert rec.iloc[0]["universe"] == (
        "Portfolio rebalancing / risk allocation"
    )


def test_rejected_universe_is_crypto_spot():
    df = sus.rank_universes()
    rej = df[df["decision_status"] == "REJECTED"]
    assert "Crypto spot" in rej["universe"].tolist()


def test_top_recommendation_returns_recommended_row():
    out = sus.top_recommendation()
    assert out["decision_status"] == "RECOMMENDED"
    assert out["no_strategy_today"] is False
    assert out["universe"] == "Portfolio rebalancing / risk allocation"


# ---------------------------------------------------------------------------
# Recommendation-language safety — must NEVER contain trade-action words
# ---------------------------------------------------------------------------
def test_every_action_contains_research():
    """Every recommendation must point at a RESEARCH branch — not at
    trading, paper trading, or broker connection."""
    for a in sus.assessments():
        assert "research" in a.recommended_next_action.lower(), (
            f"{a.universe} recommendation does not mention "
            f"'research': {a.recommended_next_action!r}"
        )


def test_no_action_contains_forbidden_tokens():
    for a in sus.assessments():
        low = a.recommended_next_action.lower()
        for bad in sus.FORBIDDEN_RECOMMENDATION_TOKENS:
            assert bad not in low, (
                f"{a.universe} recommendation contains forbidden "
                f"token {bad!r}: {a.recommended_next_action!r}"
            )


def test_recommendation_is_clean_predicate():
    assert sus.recommendation_is_clean(
        "Open research/portfolio-rebalancing-strategy-v1 next.",
    )
    # Missing 'research' → fails.
    assert not sus.recommendation_is_clean("Buy BTC and hold.")
    # Contains forbidden token → fails.
    assert not sus.recommendation_is_clean(
        "Research phase recommends: place order on Kraken.",
    )


def test_top_recommendation_action_is_clean():
    out = sus.top_recommendation()
    assert sus.recommendation_is_clean(out["recommended_next_action"])


# ---------------------------------------------------------------------------
# Source-level safety
# ---------------------------------------------------------------------------
_SOURCE = (Path(__file__).resolve().parents[1]
              / "src" / "strategy_universe_selection.py").read_text()


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


def test_no_strategy_imports():
    """The module must not pull in actual strategy code — it only
    DESCRIBES strategy universes."""
    bad = (
        "from .strategies", "from src.strategies",
        "import strategies",
        # Backtester / signal modules must also be absent — this is
        # purely a decision module.
        "from . import backtester", "from . import portfolio_backtester",
    )
    for s in bad:
        assert s not in _SOURCE, s


def test_no_network_calls():
    bad = ("urllib.request", "urllib.urlopen", "requests.get(",
            "httpx.get(", "aiohttp")
    for s in bad:
        assert s not in _SOURCE, s


# ---------------------------------------------------------------------------
# Safety lock continues to be locked
# ---------------------------------------------------------------------------
def test_safety_lock_remains_locked():
    assert safety_lock.is_execution_allowed() is False
    assert safety_lock.is_paper_trading_allowed() is False
    assert safety_lock.is_kraken_connection_allowed() is False
    assert safety_lock.safety_lock_status() == "locked"
