"""Tests for `src/portfolio_risk.py`. All offline."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from src import portfolio_risk as pr
from src import safety_lock


# ---------------------------------------------------------------------------
# Synthetic portfolio fixtures
# ---------------------------------------------------------------------------
def _df(*rows: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _row(asset: str, qty: float, avg_cost: float, current_price: float,
          currency: str = "USD") -> Dict[str, Any]:
    return {
        "asset": asset, "quantity": qty, "average_cost": avg_cost,
        "currency": currency, "current_price": current_price,
        "price_source": "manual", "notes": "",
    }


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
def test_required_columns_locked():
    assert pr.REQUIRED_COLUMNS == (
        "asset", "quantity", "average_cost", "currency",
        "current_price", "price_source", "notes",
    )


def test_validate_schema_accepts_all_required_columns():
    df = _df(_row("BTC", 1.0, 30000.0, 60000.0))
    status = pr.validate_portfolio_schema(df)
    assert status.ok is True
    assert status.missing_columns == ()
    assert status.n_rows == 1


def test_validate_schema_fails_when_columns_missing():
    df = pd.DataFrame([{"asset": "BTC", "quantity": 1.0}])
    status = pr.validate_portfolio_schema(df)
    assert status.ok is False
    assert "average_cost" in status.missing_columns
    assert "current_price" in status.missing_columns


def test_validate_schema_handles_extra_columns():
    df = _df(_row("BTC", 1.0, 30000.0, 60000.0))
    df["custom_field"] = "x"
    status = pr.validate_portfolio_schema(df)
    assert status.ok is True
    assert "custom_field" in status.extra_columns


# ---------------------------------------------------------------------------
# load_portfolio_holdings — fail-soft
# ---------------------------------------------------------------------------
def test_load_missing_csv_returns_empty_with_warning(tmp_path):
    df, warning = pr.load_portfolio_holdings(tmp_path / "nope.csv")
    assert df.empty
    assert list(df.columns) == list(pr.REQUIRED_COLUMNS)
    assert warning is not None
    assert "data/portfolio_holdings" in warning


def test_load_existing_csv_succeeds(tmp_path):
    p = tmp_path / "portfolio_holdings.csv"
    _df(_row("BTC", 1.0, 30000.0, 60000.0)).to_csv(p, index=False)
    df, warning = pr.load_portfolio_holdings(p)
    assert warning is None
    assert len(df) == 1
    assert df.iloc[0]["asset"] == "BTC"


# ---------------------------------------------------------------------------
# Position values
# ---------------------------------------------------------------------------
def test_calculate_position_values_basic():
    df = _df(
        _row("BTC", 1.0, 30000.0, 60000.0),
        _row("ETH", 10.0, 2000.0, 3000.0),
    )
    out = pr.calculate_position_values(df)
    btc = out[out["asset"] == "BTC"].iloc[0]
    eth = out[out["asset"] == "ETH"].iloc[0]
    assert btc["position_value"] == pytest.approx(60000.0)
    assert btc["cost_basis"] == pytest.approx(30000.0)
    assert btc["unrealized_pnl"] == pytest.approx(30000.0)
    assert btc["unrealized_pnl_percent"] == pytest.approx(100.0)
    assert eth["position_value"] == pytest.approx(30000.0)


def test_calculate_position_values_weights_sum_to_one():
    df = _df(
        _row("BTC", 1.0, 30000.0, 60000.0),
        _row("ETH", 10.0, 2000.0, 3000.0),
        _row("USDC", 10000.0, 1.0, 1.0),
    )
    out = pr.calculate_position_values(df)
    assert out["portfolio_weight"].sum() == pytest.approx(1.0)


def test_calculate_position_values_zero_total_safe():
    df = _df(_row("BTC", 0.0, 0.0, 0.0))
    out = pr.calculate_position_values(df)
    assert (out["portfolio_weight"] == 0.0).all()


# ---------------------------------------------------------------------------
# Portfolio summary
# ---------------------------------------------------------------------------
def test_summary_aggregates_correctly():
    df = _df(
        _row("BTC", 1.0, 30000.0, 60000.0),   # +30k pnl
        _row("ETH", 10.0, 2000.0, 3000.0),     # +10k pnl
        _row("USDC", 10000.0, 1.0, 1.0),       # 0 pnl
    )
    s = pr.calculate_portfolio_summary(df)
    assert s["total_market_value"] == pytest.approx(100000.0)
    assert s["total_cost_basis"] == pytest.approx(60000.0)
    assert s["total_unrealized_pnl"] == pytest.approx(40000.0)
    assert s["largest_position"] == "BTC"
    assert s["largest_position_weight"] == pytest.approx(0.6)
    assert s["asset_count"] == 3
    assert s["crypto_exposure"] == pytest.approx(0.9)
    assert s["stablecoin_exposure"] == pytest.approx(0.1)


def test_summary_empty_frame_is_safe():
    s = pr.calculate_portfolio_summary(pd.DataFrame())
    assert s["total_market_value"] == 0.0
    assert s["asset_count"] == 0
    assert s["largest_position"] is None


# ---------------------------------------------------------------------------
# Drawdown scenarios
# ---------------------------------------------------------------------------
def test_scenarios_apply_only_to_crypto():
    df = _df(
        _row("BTC", 1.0, 30000.0, 60000.0),   # 60k crypto
        _row("USDC", 40000.0, 1.0, 1.0),       # 40k stablecoin
    )
    out = pr.calculate_drawdown_scenarios(df)
    assert set(out["scenario"]) == {
        "minus_10_percent", "minus_20_percent",
        "minus_30_percent", "minus_50_percent",
    }
    minus_10 = out[out["scenario"] == "minus_10_percent"].iloc[0]
    # Only BTC (60k) gets shocked: -6k.
    assert minus_10["portfolio_loss"] == pytest.approx(-6000.0)
    assert minus_10["new_portfolio_value"] == pytest.approx(94000.0)
    assert minus_10["largest_loss_asset"] == "BTC"

    minus_50 = out[out["scenario"] == "minus_50_percent"].iloc[0]
    # 50% of 60k crypto = -30k. Total mv 100k → 70k.
    assert minus_50["portfolio_loss"] == pytest.approx(-30000.0)
    assert minus_50["new_portfolio_value"] == pytest.approx(70000.0)


def test_scenarios_handle_no_crypto():
    df = _df(_row("USDC", 100000.0, 1.0, 1.0))
    out = pr.calculate_drawdown_scenarios(df)
    for _, row in out.iterrows():
        assert row["portfolio_loss"] == pytest.approx(0.0)
        assert row["largest_loss_asset"] is None


def test_scenarios_largest_loss_asset_picks_dollar_largest():
    df = _df(
        _row("BTC", 1.0, 30000.0, 60000.0),    # 60k
        _row("ETH", 1.0, 2000.0, 3000.0),       # 3k
    )
    out = pr.calculate_drawdown_scenarios(df)
    for _, row in out.iterrows():
        assert row["largest_loss_asset"] == "BTC"


# ---------------------------------------------------------------------------
# BTC baseline comparison
# ---------------------------------------------------------------------------
def test_btc_baseline_when_btc_present():
    df = _df(
        _row("BTC", 1.0, 30000.0, 60000.0),
        _row("ETH", 10.0, 2000.0, 3000.0),
    )
    out = pr.compare_to_btc_baseline(df)
    assert out["btc_present"] is True
    # 60k BTC / 90k total = 0.667
    assert out["btc_weight"] == pytest.approx(60000.0 / 90000.0)
    assert out["non_btc_weight"] == pytest.approx(30000.0 / 90000.0)
    assert out["warning"] is None


def test_btc_baseline_when_btc_absent():
    df = _df(
        _row("ETH", 10.0, 2000.0, 3000.0),
        _row("SOL", 100.0, 50.0, 100.0),
    )
    out = pr.compare_to_btc_baseline(df)
    assert out["btc_present"] is False
    assert out["btc_weight"] == 0.0
    assert "BTC absent" in out["warning"]


def test_btc_baseline_empty_frame_is_safe():
    out = pr.compare_to_btc_baseline(pd.DataFrame())
    assert out["btc_present"] is False
    assert "warning" in out


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------
def test_classify_unknown_when_no_data():
    assert pr.classify_portfolio_risk({}) == "UNKNOWN"
    assert pr.classify_portfolio_risk(
        {"total_market_value": 0.0}) == "UNKNOWN"


def test_classify_extreme_at_85pct_largest():
    s = {"total_market_value": 100000.0,
          "largest_position_weight": 0.85,
          "crypto_exposure": 0.85}
    assert pr.classify_portfolio_risk(s) == "EXTREME"


def test_classify_high_at_70pct_largest():
    s = {"total_market_value": 100000.0,
          "largest_position_weight": 0.70,
          "crypto_exposure": 0.70}
    assert pr.classify_portfolio_risk(s) == "HIGH"


def test_classify_high_when_crypto_exposure_above_90pct():
    s = {"total_market_value": 100000.0,
          "largest_position_weight": 0.30,
          "crypto_exposure": 0.95}
    assert pr.classify_portfolio_risk(s) == "HIGH"


def test_classify_moderate_at_50pct_largest():
    s = {"total_market_value": 100000.0,
          "largest_position_weight": 0.50,
          "crypto_exposure": 0.50}
    assert pr.classify_portfolio_risk(s) == "MODERATE"


def test_classify_low_at_30pct_largest_with_normal_crypto():
    s = {"total_market_value": 100000.0,
          "largest_position_weight": 0.30,
          "crypto_exposure": 0.50}
    assert pr.classify_portfolio_risk(s) == "LOW"


# ---------------------------------------------------------------------------
# Recommendation phrasing — must NEVER contain trade-action language
# ---------------------------------------------------------------------------
def test_recommendation_phrases_locked():
    assert pr.RECOMMENDATION_PHRASES == (
        "hold risk steady", "review concentration",
        "reduce concentration", "data missing",
        "do nothing until data is complete",
    )


def test_recommendation_for_extreme_says_reduce():
    s = {"total_market_value": 100000.0,
          "largest_position_weight": 0.90,
          "crypto_exposure": 0.95}
    rec = pr.generate_risk_recommendation(s, pd.DataFrame())
    assert rec["action"] == "reduce concentration"


def test_recommendation_for_high_says_review():
    s = {"total_market_value": 100000.0,
          "largest_position_weight": 0.70,
          "crypto_exposure": 0.70}
    rec = pr.generate_risk_recommendation(s, pd.DataFrame())
    assert rec["action"] == "review concentration"


def test_recommendation_for_low_says_hold_steady():
    s = {"total_market_value": 100000.0,
          "largest_position_weight": 0.20,
          "crypto_exposure": 0.30}
    rec = pr.generate_risk_recommendation(s, pd.DataFrame())
    assert rec["action"] == "hold risk steady"


def test_recommendation_for_no_data_says_do_nothing():
    rec = pr.generate_risk_recommendation({}, pd.DataFrame())
    assert rec["action"] == "do nothing until data is complete"
    assert rec["category"] == "data missing"


def test_recommendation_never_contains_trade_action_language():
    """Iterate the locked phrase list. None may contain forbidden
    trade-action language."""
    forbidden = (
        "buy", "sell", "enter trade", "open position",
        "place order", "submit order", "close trade",
        "connect broker", "go long", "go short",
        "long position", "short position",
    )
    for phrase in pr.RECOMMENDATION_PHRASES:
        lower = phrase.lower()
        for bad in forbidden:
            assert bad not in lower, (
                f"forbidden trade-action token {bad!r} found in "
                f"recommendation phrase {phrase!r}"
            )


# ---------------------------------------------------------------------------
# Dashboard state — entry point
# ---------------------------------------------------------------------------
def test_dashboard_state_works_with_missing_file(tmp_path):
    state = pr.get_portfolio_risk_dashboard_state(tmp_path / "absent.csv")
    assert state["holdings"].empty
    assert state["schema_status"].ok is True  # empty frame has all cols
    assert state["risk_classification"] == "UNKNOWN"
    assert state["recommendation"]["action"] == "do nothing until data is complete"
    assert any("portfolio_holdings" in w for w in state["warnings"])


def test_dashboard_state_with_holdings(tmp_path):
    p = tmp_path / "portfolio_holdings.csv"
    _df(
        _row("BTC", 1.0, 30000.0, 60000.0),
        _row("USDC", 40000.0, 1.0, 1.0),
    ).to_csv(p, index=False)
    state = pr.get_portfolio_risk_dashboard_state(p)
    assert not state["holdings"].empty
    assert state["summary"]["total_market_value"] == pytest.approx(100000.0)
    assert state["btc_baseline"]["btc_present"] is True
    assert state["risk_classification"] in pr.RISK_CLASSES
    assert state["recommendation"]["action"] in pr.RECOMMENDATION_PHRASES


def test_dashboard_state_warns_when_btc_missing(tmp_path):
    p = tmp_path / "portfolio_holdings.csv"
    _df(_row("ETH", 10.0, 2000.0, 3000.0)).to_csv(p, index=False)
    state = pr.get_portfolio_risk_dashboard_state(p)
    assert state["btc_baseline"]["btc_present"] is False
    assert any("BTC absent" in w for w in state["warnings"])


def test_dashboard_state_with_bad_schema(tmp_path):
    p = tmp_path / "portfolio_holdings.csv"
    pd.DataFrame([{"asset": "BTC", "quantity": 1.0}]).to_csv(p,
                                                                    index=False)
    state = pr.get_portfolio_risk_dashboard_state(p)
    assert state["schema_status"].ok is False
    assert state["risk_classification"] == "UNKNOWN"
    assert state["recommendation"]["action"] == "do nothing until data is complete"


# ---------------------------------------------------------------------------
# Safety lock continues to be locked
# ---------------------------------------------------------------------------
def test_safety_lock_remains_locked():
    assert safety_lock.is_execution_allowed() is False
    assert safety_lock.is_paper_trading_allowed() is False
    assert safety_lock.is_kraken_connection_allowed() is False
    assert safety_lock.safety_lock_status() == "locked"


# ---------------------------------------------------------------------------
# Source-level safety invariants
# ---------------------------------------------------------------------------
_SOURCE = (Path(__file__).resolve().parents[1]
              / "src" / "portfolio_risk.py").read_text()


def test_no_broker_imports():
    bad_patterns = (
        re.compile(r"^\s*import\s+ccxt\b", re.MULTILINE),
        re.compile(r"^\s*from\s+ccxt", re.MULTILINE),
        re.compile(r"^\s*import\s+kraken\b", re.MULTILINE),
        re.compile(r"^\s*from\s+kraken\b", re.MULTILINE),
        re.compile(r"\bbinance\.client\b"),
        re.compile(r"\bbybit\.client\b"),
        re.compile(r"\balpaca\b"),  # spec said ignore Alpaca completely
    )
    for pat in bad_patterns:
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


def test_no_paper_trading_strings():
    """The portfolio risk module must not mention paper / live
    trading vocabulary anywhere."""
    forbidden = (
        re.compile(r"\bpaper_trade\b", re.IGNORECASE),
        re.compile(r"\blive_trade\b", re.IGNORECASE),
        re.compile(r"\bgo_live\b", re.IGNORECASE),
    )
    for pat in forbidden:
        assert pat.search(_SOURCE) is None, pat.pattern


def test_no_network_calls():
    bad = ("urllib.request", "urllib.urlopen", "requests.get(",
            "httpx.get(", "aiohttp")
    for s in bad:
        assert s not in _SOURCE, s


def test_no_file_writes_in_module():
    """The helper is read-only. Locked patterns: `to_csv`, `to_json`,
    `write_text`, `write_bytes`, `open(.., 'w')`."""
    bad = (".to_csv(", ".to_json(",
            ".write_text(", ".write_bytes(")
    for s in bad:
        assert s not in _SOURCE, (
            f"portfolio_risk module must be read-only: {s!r}"
        )
    assert re.search(r"open\([^)]*['\"]w['\"]", _SOURCE) is None
