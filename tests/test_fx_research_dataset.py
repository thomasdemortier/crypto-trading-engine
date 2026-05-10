"""Tests for `src/fx_research_dataset.py`. All offline.

Exercises schema, derivation logic, return computation, soft network
failure behaviour, write-path safety, and the locked safety invariants
(no broker / API key / execution / paper-trading / live-trading code).
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src import (
    config,
    fx_research_dataset as fxd,
    safety_lock,
    utils,
)


REPO_ROOT = config.REPO_ROOT


# ---------------------------------------------------------------------------
# Stub HTTP helpers
# ---------------------------------------------------------------------------
def _ok(payload: Any = None, text: str = "") -> fxd._Response:
    return fxd._Response(True, 200, payload, text or "", None)


def _err(error: str = "simulated") -> fxd._Response:
    return fxd._Response(False, None, None, None,
                          f"network_error: {error}")


def _ecb_csv(currency: str, rows: list[tuple[str, str]]) -> str:
    """Build a synthetic ECB SDMX CSV payload."""
    header = ("KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,"
                "TIME_PERIOD,OBS_VALUE\n")
    body = "".join(
        f"EXR.D.{currency}.EUR.SP00.A,D,{currency},EUR,SP00,A,"
        f"{date},{value}\n"
        for date, value in rows
    )
    return header + body


def _build_ecb_stub(payloads: dict[str, str]):
    """Return an http_get stub keyed on the ISO currency in the URL."""
    def stub(url: str) -> fxd._Response:
        for ccy, csv_text in payloads.items():
            if f"D.{ccy}.EUR" in url:
                return _ok(text=csv_text)
        return _err("currency not in stub")
    return stub


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_dataset_columns_locked():
    assert fxd.DATASET_COLUMNS == [
        "date", "asset", "source", "base", "quote", "close",
        "return_1d", "log_return_1d", "is_derived", "source_pair",
        "data_quality_status", "notes",
    ]


def test_validate_schema_accepts_empty_frame():
    fxd.validate_fx_dataset_schema(fxd._empty_long_frame())


def test_validate_schema_rejects_missing_column():
    df = fxd._empty_long_frame().drop(columns=["return_1d"])
    with pytest.raises(fxd.FxDatasetSchemaError):
        fxd.validate_fx_dataset_schema(df)


def test_validate_schema_rejects_extra_column():
    df = fxd._empty_long_frame()
    df["extra"] = []
    with pytest.raises(fxd.FxDatasetSchemaError):
        fxd.validate_fx_dataset_schema(df)


def test_validate_schema_rejects_bad_status():
    df = pd.DataFrame([{
        "date": pd.Timestamp("2020-01-01"),
        "asset": "EUR/USD", "source": "ecb_sdmx",
        "base": "EUR", "quote": "USD", "close": 1.1,
        "return_1d": float("nan"), "log_return_1d": float("nan"),
        "is_derived": False, "source_pair": "",
        "data_quality_status": "totally_made_up",
        "notes": "",
    }])
    with pytest.raises(fxd.FxDatasetSchemaError):
        fxd.validate_fx_dataset_schema(df[fxd.DATASET_COLUMNS])


# ---------------------------------------------------------------------------
# ECB parser
# ---------------------------------------------------------------------------
def test_parse_ecb_csv_skips_dot_holidays():
    csv_text = _ecb_csv("USD", [
        ("2020-01-02", "1.10"),
        ("2020-01-03", "."),
        ("2020-01-06", "1.12"),
    ])
    df, warnings = fxd._parse_ecb_csv(csv_text, "EUR/USD", "USD")
    assert len(df) == 2
    assert df["close"].tolist() == [1.10, 1.12]
    assert any("TARGET-holiday" in w for w in warnings)


def test_parse_ecb_csv_drops_unparseable_rows():
    csv_text = (
        "KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,"
        "TIME_PERIOD,OBS_VALUE\n"
        "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,not-a-date,1.10\n"
        "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2020-01-02,abc\n"
        "EXR.D.USD.EUR.SP00.A,D,USD,EUR,SP00,A,2020-01-03,1.11\n"
    )
    df, warnings = fxd._parse_ecb_csv(csv_text, "EUR/USD", "USD")
    assert len(df) == 1
    assert df["close"].iloc[0] == pytest.approx(1.11)
    assert any("unparseable" in w for w in warnings)


def test_parse_ecb_csv_empty_returns_empty_frame():
    csv_text = (
        "KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,"
        "TIME_PERIOD,OBS_VALUE\n"
    )
    df, warnings = fxd._parse_ecb_csv(csv_text, "EUR/USD", "USD")
    assert df.empty
    assert any("0 usable rows" in w for w in warnings)


# ---------------------------------------------------------------------------
# Soft network failure behaviour
# ---------------------------------------------------------------------------
def test_fetch_ecb_rates_soft_fails_on_network_error():
    df, warnings = fxd.fetch_ecb_rates(http_get=lambda url: _err("boom"))
    assert not df.empty  # placeholder rows
    assert (df["data_quality_status"] == fxd.DATA_QUALITY_MISSING).all()
    assert set(df["asset"]) == {"EUR/USD", "EUR/GBP", "EUR/JPY", "EUR/CHF"}
    assert all("ECB fetch failed" in w for w in warnings)


def test_fetch_ecb_rates_returns_data_on_success():
    payloads = {
        "USD": _ecb_csv("USD", [("2020-01-02", "1.10"),
                                  ("2020-01-03", "1.11")]),
        "GBP": _ecb_csv("GBP", [("2020-01-02", "0.85")]),
        "JPY": _ecb_csv("JPY", [("2020-01-02", "120.0")]),
        "CHF": _ecb_csv("CHF", [("2020-01-02", "1.07")]),
    }
    df, warnings = fxd.fetch_ecb_rates(http_get=_build_ecb_stub(payloads))
    ok = df[df["data_quality_status"] == fxd.DATA_QUALITY_OK]
    assert set(ok["asset"]) == {"EUR/USD", "EUR/GBP", "EUR/JPY", "EUR/CHF"}
    eur_usd = ok[ok["asset"] == "EUR/USD"].sort_values("date")
    assert eur_usd["close"].tolist() == [1.10, 1.11]
    assert (eur_usd["base"] == "EUR").all()
    assert (eur_usd["quote"] == "USD").all()
    assert (~eur_usd["is_derived"]).all()


def test_fetch_lbma_gold_soft_fails_on_network_error():
    df, warnings = fxd.fetch_lbma_gold(http_get=lambda url: _err("boom"))
    assert len(df) == 1
    assert df.iloc[0]["data_quality_status"] == fxd.DATA_QUALITY_MISSING
    assert df.iloc[0]["asset"] == "XAU/USD"
    assert any("LBMA fetch failed" in w for w in warnings)


def test_fetch_lbma_gold_parses_payload():
    payload = [
        {"d": "2020-01-02", "v": [1500.0, 1100.0, 1340.0]},
        {"d": "2020-01-03", "v": [1510.0, 1110.0, 1345.0]},
    ]
    df, warnings = fxd.fetch_lbma_gold(http_get=lambda url: _ok(payload))
    assert (df["asset"] == "XAU/USD").all()
    assert df["close"].tolist() == [1500.0, 1510.0]
    assert (df["data_quality_status"] == fxd.DATA_QUALITY_OK).all()


# ---------------------------------------------------------------------------
# Derived crosses
# ---------------------------------------------------------------------------
def test_derive_fx_pairs_computes_correct_ratios():
    direct = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-02"), "asset": "EUR/USD",
         "source": "ecb_sdmx", "base": "EUR", "quote": "USD",
         "close": 1.10, "is_derived": False, "source_pair": "",
         "data_quality_status": fxd.DATA_QUALITY_OK,
         "return_1d": float("nan"), "log_return_1d": float("nan"),
         "notes": ""},
        {"date": pd.Timestamp("2020-01-02"), "asset": "EUR/JPY",
         "source": "ecb_sdmx", "base": "EUR", "quote": "JPY",
         "close": 120.0, "is_derived": False, "source_pair": "",
         "data_quality_status": fxd.DATA_QUALITY_OK,
         "return_1d": float("nan"), "log_return_1d": float("nan"),
         "notes": ""},
        {"date": pd.Timestamp("2020-01-02"), "asset": "EUR/CHF",
         "source": "ecb_sdmx", "base": "EUR", "quote": "CHF",
         "close": 1.07, "is_derived": False, "source_pair": "",
         "data_quality_status": fxd.DATA_QUALITY_OK,
         "return_1d": float("nan"), "log_return_1d": float("nan"),
         "notes": ""},
        {"date": pd.Timestamp("2020-01-02"), "asset": "EUR/GBP",
         "source": "ecb_sdmx", "base": "EUR", "quote": "GBP",
         "close": 0.85, "is_derived": False, "source_pair": "",
         "data_quality_status": fxd.DATA_QUALITY_OK,
         "return_1d": float("nan"), "log_return_1d": float("nan"),
         "notes": ""},
    ])
    derived, warnings = fxd.derive_fx_pairs(direct)
    by_asset = {r["asset"]: r for _, r in derived.iterrows()}
    # USD/JPY = EUR/JPY / EUR/USD = 120 / 1.10
    assert by_asset["USD/JPY"]["close"] == pytest.approx(120.0 / 1.10)
    # USD/CHF = EUR/CHF / EUR/USD = 1.07 / 1.10
    assert by_asset["USD/CHF"]["close"] == pytest.approx(1.07 / 1.10)
    # GBP/USD = EUR/USD / EUR/GBP = 1.10 / 0.85
    assert by_asset["GBP/USD"]["close"] == pytest.approx(1.10 / 0.85)
    assert all(by_asset[a]["is_derived"] for a in
                 ("USD/JPY", "USD/CHF", "GBP/USD"))
    assert by_asset["USD/JPY"]["source_pair"] == "EUR/JPY÷EUR/USD"
    assert warnings == []


def test_derive_fx_pairs_drops_dates_missing_a_leg():
    """No silent forward-fill — derived rows only exist where both
    legs share a date."""
    direct = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-02"), "asset": "EUR/USD",
         "source": "ecb_sdmx", "base": "EUR", "quote": "USD",
         "close": 1.10, "is_derived": False, "source_pair": "",
         "data_quality_status": fxd.DATA_QUALITY_OK,
         "return_1d": float("nan"), "log_return_1d": float("nan"),
         "notes": ""},
        # EUR/JPY missing on 2020-01-02; only present on 2020-01-03
        {"date": pd.Timestamp("2020-01-03"), "asset": "EUR/JPY",
         "source": "ecb_sdmx", "base": "EUR", "quote": "JPY",
         "close": 120.0, "is_derived": False, "source_pair": "",
         "data_quality_status": fxd.DATA_QUALITY_OK,
         "return_1d": float("nan"), "log_return_1d": float("nan"),
         "notes": ""},
    ])
    derived, _ = fxd.derive_fx_pairs(direct)
    usd_jpy = derived[derived["asset"] == "USD/JPY"]
    # Legs never overlap → placeholder missing row
    assert (usd_jpy["data_quality_status"]
            == fxd.DATA_QUALITY_MISSING).all()


def test_derive_fx_pairs_handles_empty_input():
    df, warnings = fxd.derive_fx_pairs(fxd._empty_long_frame())
    assert df.empty
    assert any("empty" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------
def test_compute_fx_returns_simple_and_log():
    df = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-02"), "asset": "EUR/USD",
         "source": "ecb_sdmx", "base": "EUR", "quote": "USD",
         "close": 1.10, "is_derived": False, "source_pair": "",
         "data_quality_status": fxd.DATA_QUALITY_OK,
         "return_1d": float("nan"), "log_return_1d": float("nan"),
         "notes": ""},
        {"date": pd.Timestamp("2020-01-03"), "asset": "EUR/USD",
         "source": "ecb_sdmx", "base": "EUR", "quote": "USD",
         "close": 1.21, "is_derived": False, "source_pair": "",
         "data_quality_status": fxd.DATA_QUALITY_OK,
         "return_1d": float("nan"), "log_return_1d": float("nan"),
         "notes": ""},
    ])
    out = fxd.compute_fx_returns(df)
    out_sorted = out.sort_values("date").reset_index(drop=True)
    # First row has no prior close
    assert math.isnan(out_sorted.iloc[0]["return_1d"])
    assert math.isnan(out_sorted.iloc[0]["log_return_1d"])
    # Second row: 1.21 / 1.10 - 1 = 0.10; log(1.21/1.10) = log(1.1)
    assert out_sorted.iloc[1]["return_1d"] == pytest.approx(0.10)
    assert out_sorted.iloc[1]["log_return_1d"] == pytest.approx(
        math.log(1.21 / 1.10)
    )


def test_compute_fx_returns_independent_per_asset():
    df = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-02"), "asset": "EUR/USD",
         "source": "ecb_sdmx", "base": "EUR", "quote": "USD",
         "close": 1.10, "is_derived": False, "source_pair": "",
         "data_quality_status": fxd.DATA_QUALITY_OK,
         "return_1d": float("nan"), "log_return_1d": float("nan"),
         "notes": ""},
        {"date": pd.Timestamp("2020-01-02"), "asset": "EUR/JPY",
         "source": "ecb_sdmx", "base": "EUR", "quote": "JPY",
         "close": 120.0, "is_derived": False, "source_pair": "",
         "data_quality_status": fxd.DATA_QUALITY_OK,
         "return_1d": float("nan"), "log_return_1d": float("nan"),
         "notes": ""},
        {"date": pd.Timestamp("2020-01-03"), "asset": "EUR/JPY",
         "source": "ecb_sdmx", "base": "EUR", "quote": "JPY",
         "close": 132.0, "is_derived": False, "source_pair": "",
         "data_quality_status": fxd.DATA_QUALITY_OK,
         "return_1d": float("nan"), "log_return_1d": float("nan"),
         "notes": ""},
    ])
    out = fxd.compute_fx_returns(df)
    eur_jpy = out[out["asset"] == "EUR/JPY"].sort_values("date")
    assert math.isnan(eur_jpy.iloc[0]["return_1d"])  # first JPY row
    assert eur_jpy.iloc[1]["return_1d"] == pytest.approx(0.10)


def test_compute_fx_returns_skips_missing_rows():
    df = fxd._missing_placeholder("EUR/USD", "ecb_sdmx", "fetch failed")
    out = fxd.compute_fx_returns(df)
    assert math.isnan(out.iloc[0]["return_1d"])
    assert out.iloc[0]["data_quality_status"] == fxd.DATA_QUALITY_MISSING


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def test_summarize_empty_dataset_lists_all_assets_missing():
    summary = fxd.summarize_fx_dataset(fxd._empty_long_frame())
    assert summary["row_count"] == 0
    assert summary["asset_count"] == 0
    assert set(summary["assets_missing"]) == {
        "EUR/USD", "EUR/GBP", "EUR/JPY", "EUR/CHF",
        "USD/JPY", "USD/CHF", "GBP/USD", "XAU/USD",
    }
    assert summary["coverage_by_asset"] == {}


def test_summarize_reports_per_asset_coverage():
    df = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-02"), "asset": "EUR/USD",
         "source": "ecb_sdmx", "base": "EUR", "quote": "USD",
         "close": 1.10, "is_derived": False, "source_pair": "",
         "data_quality_status": fxd.DATA_QUALITY_OK,
         "return_1d": float("nan"), "log_return_1d": float("nan"),
         "notes": ""},
        {"date": pd.Timestamp("2020-01-03"), "asset": "EUR/USD",
         "source": "ecb_sdmx", "base": "EUR", "quote": "USD",
         "close": 1.11, "is_derived": False, "source_pair": "",
         "data_quality_status": fxd.DATA_QUALITY_OK,
         "return_1d": float("nan"), "log_return_1d": float("nan"),
         "notes": ""},
    ])
    summary = fxd.summarize_fx_dataset(df, warnings=["w1"])
    assert summary["row_count"] == 2
    assert summary["asset_count"] == 1
    assert summary["assets_available"] == ["EUR/USD"]
    assert "EUR/JPY" in summary["assets_missing"]
    assert summary["coverage_by_asset"]["EUR/USD"]["rows"] == 2
    assert summary["data_quality_warnings"] == ["w1"]


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def _minimal_valid_frame() -> pd.DataFrame:
    return pd.DataFrame([{
        "date": pd.Timestamp("2020-01-02"),
        "asset": "EUR/USD", "source": "ecb_sdmx",
        "base": "EUR", "quote": "USD", "close": 1.10,
        "return_1d": float("nan"), "log_return_1d": float("nan"),
        "is_derived": False, "source_pair": "",
        "data_quality_status": fxd.DATA_QUALITY_OK,
        "notes": "",
    }], columns=fxd.DATASET_COLUMNS)


def test_write_rejects_path_outside_data_fx(tmp_path):
    df = _minimal_valid_frame()
    bad = tmp_path / "not_in_repo.parquet"
    with pytest.raises(fxd.FxDatasetWritePathError):
        fxd.write_fx_dataset(df, parquet_path=bad, csv_path=None)


def test_write_rejects_repo_path_outside_data_fx():
    df = _minimal_valid_frame()
    bad = config.RESULTS_DIR / "fx_daily_v1.parquet"
    with pytest.raises(fxd.FxDatasetWritePathError):
        fxd.write_fx_dataset(df, parquet_path=bad, csv_path=None)


def test_write_csv_inside_data_fx(tmp_path, monkeypatch):
    df = _minimal_valid_frame()
    target_dir = config.REPO_ROOT / "data" / "fx"
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_target = target_dir / "fx_test_only_v1.csv"
    parquet_target = target_dir / "fx_test_only_v1.parquet"
    try:
        written = fxd.write_fx_dataset(
            df, parquet_path=parquet_target, csv_path=csv_target,
        )
        assert "csv" in written
        assert csv_target.exists()
        loaded = pd.read_csv(csv_target)
        assert list(loaded.columns) == fxd.DATASET_COLUMNS
    finally:
        for p in (csv_target, parquet_target):
            if p.exists():
                p.unlink()


def test_write_validates_schema_before_writing(tmp_path):
    target_dir = config.REPO_ROOT / "data" / "fx"
    target_dir.mkdir(parents=True, exist_ok=True)
    bad = _minimal_valid_frame().drop(columns=["close"])
    csv_target = target_dir / "fx_schema_reject_v1.csv"
    parquet_target = target_dir / "fx_schema_reject_v1.parquet"
    try:
        with pytest.raises(fxd.FxDatasetSchemaError):
            fxd.write_fx_dataset(
                bad, parquet_path=parquet_target, csv_path=csv_target,
            )
        assert not csv_target.exists()
        assert not parquet_target.exists()
    finally:
        for p in (csv_target, parquet_target):
            if p.exists():
                p.unlink()


# ---------------------------------------------------------------------------
# End-to-end build (offline, using stubs)
# ---------------------------------------------------------------------------
def test_build_fx_daily_dataset_offline_with_stubs():
    ecb_payloads = {
        "USD": _ecb_csv("USD", [("2020-01-02", "1.10"),
                                  ("2020-01-03", "1.21")]),
        "GBP": _ecb_csv("GBP", [("2020-01-02", "0.85"),
                                  ("2020-01-03", "0.86")]),
        "JPY": _ecb_csv("JPY", [("2020-01-02", "120.0"),
                                  ("2020-01-03", "132.0")]),
        "CHF": _ecb_csv("CHF", [("2020-01-02", "1.07"),
                                  ("2020-01-03", "1.08")]),
    }
    lbma_payload = [
        {"d": "2020-01-02", "v": [1500.0, 1100.0, 1340.0]},
        {"d": "2020-01-03", "v": [1510.0, 1110.0, 1345.0]},
    ]
    df, warnings = fxd.build_fx_daily_dataset(
        ecb_http_get=_build_ecb_stub(ecb_payloads),
        lbma_http_get=lambda url: _ok(lbma_payload),
    )
    fxd.validate_fx_dataset_schema(df)
    assert set(df["asset"].unique()) == {
        "EUR/USD", "EUR/GBP", "EUR/JPY", "EUR/CHF",
        "USD/JPY", "USD/CHF", "GBP/USD", "XAU/USD",
    }
    derived = df[df["is_derived"]]
    assert set(derived["asset"]) == {"USD/JPY", "USD/CHF", "GBP/USD"}
    # Returns populated for the second observation per asset
    eur_usd = df[df["asset"] == "EUR/USD"].sort_values("date")
    assert eur_usd.iloc[1]["return_1d"] == pytest.approx(0.10)


def test_build_fx_daily_dataset_soft_fails_when_all_sources_down():
    df, warnings = fxd.build_fx_daily_dataset(
        ecb_http_get=lambda url: _err(),
        lbma_http_get=lambda url: _err(),
    )
    fxd.validate_fx_dataset_schema(df)
    assert (df["data_quality_status"] == fxd.DATA_QUALITY_MISSING).all()
    assert any("ECB fetch failed" in w for w in warnings)
    assert any("LBMA fetch failed" in w for w in warnings)


# ---------------------------------------------------------------------------
# Safety invariants — the module must not introduce strategy / broker /
# execution / paper-trading / live-trading / API-key code.
# ---------------------------------------------------------------------------
_MODULE_PATH = Path(fxd.__file__)
_SCRIPT_PATH = REPO_ROOT / "scripts" / "build_fx_dataset.py"
_TRACKED_FX_FILES: tuple[Path, ...] = (_MODULE_PATH, _SCRIPT_PATH)


def _read(path: Path) -> str:
    return path.read_text(errors="ignore")


def test_no_broker_imports_in_fx_module():
    for p in _TRACKED_FX_FILES:
        text = _read(p)
        assert "ccxt" not in text
        assert "alpaca" not in text.lower()
        assert "kraken" not in text.lower()
        assert "ig_bank" not in text.lower()
        assert "oanda" not in text.lower()


def test_no_api_key_reads_in_fx_module():
    for p in _TRACKED_FX_FILES:
        text = _read(p)
        assert "API_KEY" not in text
        assert "API_SECRET" not in text
        assert "os.environ" not in text
        assert "os.getenv" not in text


def test_no_order_placement_strings_in_fx_module():
    forbidden = (
        "create_order", "place_order", "submit_order",
        "send_order", "cancel_order", "AddOrder", "CancelOrder",
    )
    for p in _TRACKED_FX_FILES:
        text = _read(p)
        for token in forbidden:
            assert token not in text, f"{token} found in {p.name}"


def test_no_paper_or_live_trading_enablement_in_fx_module():
    for p in _TRACKED_FX_FILES:
        text = _read(p)
        assert "LIVE_TRADING_ENABLED = True" not in text
        assert "paper_trading_allowed = True" not in text
        assert "execution_allowed = True" not in text
        assert "ENABLE_LIVE" not in text
        assert "UNLOCK_TRADING" not in text
        assert "FORCE_TRADE" not in text


def test_no_strategy_code_in_fx_module():
    """The dataset module must not register or implement a strategy."""
    text = _read(_MODULE_PATH)
    forbidden_substrings = (
        "strategy_registry", "scorecard", "backtester",
        "strategy.register", "register_strategy", "Strategy(",
        "from src import backtester", "from src import paper_trader",
    )
    for token in forbidden_substrings:
        assert token not in text, f"strategy-leak token {token!r} present"


def test_safety_lock_remains_locked():
    s = safety_lock.status()
    assert not s.execution_allowed
    assert not s.paper_trading_allowed
    assert not s.kraken_connection_allowed
    assert s.safety_lock_status == "locked"


# ---------------------------------------------------------------------------
# .gitignore must exclude generated parquet/CSV in data/fx/
# ---------------------------------------------------------------------------
def test_gitignore_excludes_generated_fx_files():
    text = (REPO_ROOT / ".gitignore").read_text()
    assert "data/fx/*.parquet" in text
    assert "data/fx/*.csv" in text


# ---------------------------------------------------------------------------
# Module-level smoke: assert_paper_only is invoked by the builder
# ---------------------------------------------------------------------------
def test_build_calls_assert_paper_only(monkeypatch):
    called = {"n": 0}

    def fake_assert():
        called["n"] += 1

    monkeypatch.setattr(utils, "assert_paper_only", fake_assert)
    monkeypatch.setattr(fxd.utils, "assert_paper_only", fake_assert)
    fxd.build_fx_daily_dataset(
        ecb_http_get=lambda url: _err(),
        lbma_http_get=lambda url: _err(),
    )
    assert called["n"] >= 1
