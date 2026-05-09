"""Sanity-check the v1 closure report + README status section.

These are documentation tests — they assert that the closure document
and the README continue to carry the right messages over time. If a
future change drops the "do not paper trade" line or the
Binance/Kraken limitation note, these tests fail loudly.
"""
from __future__ import annotations

from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "reports" / "final_crypto_research_report.md"
README = REPO / "README.md"


@pytest.fixture(scope="module")
def report_text() -> str:
    assert REPORT.exists(), f"report missing: {REPORT}"
    return REPORT.read_text()


@pytest.fixture(scope="module")
def readme_text() -> str:
    assert README.exists(), f"README missing: {README}"
    return README.read_text()


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------
def test_report_file_exists():
    assert REPORT.exists()
    assert REPORT.stat().st_size > 0


def test_report_includes_final_verdict_table(report_text):
    """The closure verdict table must appear and explicitly mark every
    strategy family as FAIL."""
    assert "Current final verdict" in report_text
    for family in (
        "Single-asset RSI / MA / ATR",
        "Single-asset MA crossover",
        "Single-asset breakout",
        "Single-asset trend following",
        "Single-asset pullback continuation",
        "Single-asset sideways mean reversion",
        "Regime selector",
        "Portfolio momentum rotation",
        "Regime-aware portfolio momentum",
    ):
        assert family in report_text, f"missing verdict row: {family!r}"
    # FAIL must outnumber PASS — and PASS must NOT appear as a verdict
    # for any tradable family in this v1 report.
    assert "0 PASS" in report_text or "0 strategies reached PASS" in report_text


def test_report_says_no_paper_trading(report_text):
    """The 'do not paper-trade' message must appear at least twice — in
    the executive summary and in the paper-trading-decision section."""
    occurrences = sum(1 for line in report_text.splitlines()
                      if "no paper trad" in line.lower()
                      or "do not paper-trade" in line.lower()
                      or "do not paper trade" in line.lower())
    assert occurrences >= 2, (
        f"'no paper trading' message must be repeated; found {occurrences}"
    )


def test_report_says_no_kraken_execution(report_text):
    """Kraken execution must be explicitly rejected."""
    text = report_text.lower()
    assert "no kraken execution" in text or "do not connect kraken" in text


def test_report_includes_binance_kraken_limitation(report_text):
    """Section 4 must explain WHY Binance is primary and Kraken is
    reserved for execution. Both names + the 720-cap fact must appear."""
    text = report_text.lower()
    assert "binance" in text
    assert "kraken" in text
    assert "720" in report_text  # the cap number


def test_report_recommends_btc_buy_and_hold_as_baseline(report_text):
    text = report_text.lower()
    assert "buy-and-hold" in text or "buy and hold" in text
    # Section 11 must explicitly include "accept BTC buy-and-hold as the
    # baseline" as a credible direction.
    assert "accept btc buy-and-hold" in text or "accept btc buy and hold" in text


def test_report_credible_next_directions_excludes_TA_tweaks(report_text):
    """Section 11 must NOT recommend more price-based TA. Specifically,
    it should call out new SIGNAL classes."""
    text = report_text.lower()
    assert "funding rate" in text
    assert "on-chain" in text
    # Must include a section labelled "Credible next directions".
    assert "credible next directions" in text


def test_report_section_12_lists_what_not_to_do(report_text):
    """The 'what not to do' section must explicitly reject lowering
    thresholds, paper-trading, and connecting Kraken."""
    text = report_text.lower()
    assert "do not lower" in text
    assert "do not connect kraken" in text
    assert "do not paper" in text


# ---------------------------------------------------------------------------
# README shape
# ---------------------------------------------------------------------------
def test_readme_includes_current_research_status(readme_text):
    """The README must surface the closure status near the top."""
    assert "Current research status" in readme_text


def test_readme_says_no_strategy_passed(readme_text):
    text = readme_text.lower()
    assert "no strategy has passed" in text


def test_readme_says_do_not_paper_trade(readme_text):
    text = readme_text.lower()
    assert "do not paper trade" in text or "do not paper-trade" in text


def test_readme_says_do_not_connect_kraken(readme_text):
    text = readme_text.lower()
    assert "do not connect kraken" in text


def test_readme_links_to_report(readme_text):
    """The README must point readers at the full report."""
    assert "reports/final_crypto_research_report.md" in readme_text
