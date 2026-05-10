"""Tests for `src/fx_data_quality.py`. All offline.

Covers per-check pass/fail behaviour, the soft-fail on missing
dataset, the verdict combiner, the writer's path restriction, and
the locked safety invariants (no broker / API key / execution /
strategy / paper-trading / live-trading code).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from src import (
    config,
    fx_data_quality as fxq,
    fx_research_dataset as fxd,
    safety_lock,
    utils,
)


REPO_ROOT = config.REPO_ROOT


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _row(date: str, asset: str, close: float, *,
         source: str = "ecb_sdmx",
         is_derived: bool = False,
         source_pair: str = "",
         status: str = fxd.DATA_QUALITY_OK,
         return_1d: float = float("nan"),
         log_return_1d: float = float("nan"),
         notes: str = "") -> dict:
    base, quote = asset.split("/")
    return {
        "date": pd.Timestamp(date),
        "asset": asset, "source": source,
        "base": base, "quote": quote, "close": close,
        "return_1d": return_1d, "log_return_1d": log_return_1d,
        "is_derived": is_derived, "source_pair": source_pair,
        "data_quality_status": status, "notes": notes,
    }


def _eight_asset_frame() -> pd.DataFrame:
    """A tiny but internally consistent 8-asset frame: 3 dates, all
    derived rows match their EUR-leg formulas exactly, and returns
    are recomputed from close so consistency checks pass."""
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    direct = {
        "EUR/USD": [1.10, 1.12, 1.11],
        "EUR/GBP": [0.85, 0.86, 0.84],
        "EUR/JPY": [120.0, 121.5, 119.0],
        "EUR/CHF": [1.07, 1.08, 1.06],
    }
    rows: List[dict] = []
    for asset, closes in direct.items():
        for d, c in zip(dates, closes):
            rows.append(_row(d, asset, c, source="ecb_sdmx"))
    derived_specs = (
        ("USD/JPY", "EUR/JPY", "EUR/USD"),
        ("USD/CHF", "EUR/CHF", "EUR/USD"),
        ("GBP/USD", "EUR/USD", "EUR/GBP"),
    )
    for target, num, den in derived_specs:
        for i, d in enumerate(dates):
            close = direct[num][i] / direct[den][i]
            rows.append(_row(d, target, close,
                              source="derived", is_derived=True,
                              source_pair=f"{num}÷{den}"))
    for d, c in zip(dates, [1500.0, 1510.0, 1495.0]):
        rows.append(_row(d, "XAU/USD", c, source="lbma"))
    df = pd.DataFrame(rows, columns=fxd.DATASET_COLUMNS)
    df = fxd.compute_fx_returns(df)
    return df


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def test_load_fx_dataset_missing_raises():
    with pytest.raises(fxq.FxDatasetMissingError):
        fxq.load_fx_dataset(REPO_ROOT / "data" / "fx"
                              / "definitely_not_there.parquet")


def test_load_fx_dataset_csv_fallback(tmp_path):
    target_dir = config.REPO_ROOT / "data" / "fx"
    target_dir.mkdir(parents=True, exist_ok=True)
    parquet = target_dir / "fx_test_loader_v1.parquet"
    csv = target_dir / "fx_test_loader_v1.csv"
    df_in = _eight_asset_frame().head(3)
    try:
        df_in.to_csv(csv, index=False)
        # Parquet absent → CSV fallback path
        df_out = fxq.load_fx_dataset(parquet)
        assert list(df_out.columns) == list(fxd.DATASET_COLUMNS)
        assert len(df_out) == 3
    finally:
        for p in (csv, parquet):
            if p.exists():
                p.unlink()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_validate_schema_pass():
    df = _eight_asset_frame()
    r = fxq.validate_fx_quality_schema(df)
    assert r.status == fxq.PASS


def test_validate_schema_fail_missing_column():
    df = _eight_asset_frame().drop(columns=["close"])
    r = fxq.validate_fx_quality_schema(df)
    assert r.status == fxq.FAIL
    assert "close" in r.message


def test_validate_schema_fail_extra_column():
    df = _eight_asset_frame().assign(extra=1)
    r = fxq.validate_fx_quality_schema(df)
    assert r.status == fxq.FAIL
    assert "extra" in r.message


# ---------------------------------------------------------------------------
# Asset coverage
# ---------------------------------------------------------------------------
def test_asset_coverage_pass_on_eight_asset_frame():
    df = _eight_asset_frame()
    r = fxq.check_asset_coverage(df)
    assert r.status == fxq.PASS


def test_asset_coverage_fail_when_missing_an_asset():
    df = _eight_asset_frame()
    df = df[df["asset"] != "XAU/USD"]
    r = fxq.check_asset_coverage(df)
    assert r.status == fxq.FAIL
    assert "XAU/USD" in r.message


# ---------------------------------------------------------------------------
# Date monotonicity
# ---------------------------------------------------------------------------
def test_date_monotonicity_pass():
    df = _eight_asset_frame()
    r = fxq.check_date_monotonicity(df)
    assert r.status == fxq.PASS


def test_date_monotonicity_fail():
    df = _eight_asset_frame()
    # Reverse the EUR/USD slice so its dates go down.
    eur_usd = df[df["asset"] == "EUR/USD"].iloc[::-1]
    others = df[df["asset"] != "EUR/USD"]
    bad_df = pd.concat([eur_usd, others], ignore_index=True)
    r = fxq.check_date_monotonicity(bad_df)
    assert r.status == fxq.FAIL
    assert "EUR/USD" in r.message


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------
def test_duplicate_rows_pass():
    df = _eight_asset_frame()
    r = fxq.check_duplicate_rows(df)
    assert r.status == fxq.PASS


def test_duplicate_rows_fail():
    df = _eight_asset_frame()
    dup = df.iloc[[0]].copy()
    df_bad = pd.concat([df, dup], ignore_index=True)
    r = fxq.check_duplicate_rows(df_bad)
    assert r.status == fxq.FAIL
    assert "duplicate" in r.message.lower()


# ---------------------------------------------------------------------------
# Missing close
# ---------------------------------------------------------------------------
def test_missing_close_pass():
    df = _eight_asset_frame()
    r = fxq.check_missing_close(df)
    assert r.status == fxq.PASS


def test_missing_close_fail():
    df = _eight_asset_frame()
    df.loc[df.index[0], "close"] = float("nan")
    r = fxq.check_missing_close(df)
    assert r.status == fxq.FAIL


# ---------------------------------------------------------------------------
# Return consistency
# ---------------------------------------------------------------------------
def test_return_consistency_pass():
    df = _eight_asset_frame()
    r = fxq.check_return_consistency(df)
    assert r.status == fxq.PASS


def test_return_consistency_fail_when_returns_corrupted():
    df = _eight_asset_frame()
    # Find a row with an existing return and bump it well past tol.
    idx = df.index[df["return_1d"].notna()][0]
    df.loc[idx, "return_1d"] += 0.5
    r = fxq.check_return_consistency(df)
    assert r.status == fxq.FAIL
    assert "tol" in r.message


# ---------------------------------------------------------------------------
# Derived pair sanity
# ---------------------------------------------------------------------------
def test_derived_pair_sanity_pass():
    df = _eight_asset_frame()
    r = fxq.check_derived_pair_sanity(df)
    assert r.status == fxq.PASS


def test_derived_pair_sanity_fail_when_derived_close_corrupted():
    df = _eight_asset_frame()
    mask = (df["asset"] == "USD/JPY") \
              & (df["date"] == pd.Timestamp("2020-01-02"))
    idx = df.index[mask][0]
    df.loc[idx, "close"] = df.loc[idx, "close"] * 1.5  # blatantly wrong
    r = fxq.check_derived_pair_sanity(df)
    assert r.status == fxq.FAIL
    assert "USD/JPY" in r.message


# ---------------------------------------------------------------------------
# Extreme returns
# ---------------------------------------------------------------------------
def test_extreme_returns_pass_on_clean_frame():
    df = _eight_asset_frame()
    r = fxq.check_extreme_returns(df)
    assert r.status == fxq.PASS


def test_extreme_returns_warns_when_fx_jumps_above_5pct():
    # Build a frame with a single huge return.
    df = pd.DataFrame([
        _row("2020-01-02", "EUR/USD", 1.00),
        _row("2020-01-03", "EUR/USD", 1.10),  # +10%, > 5%
    ], columns=fxd.DATASET_COLUMNS)
    df = fxd.compute_fx_returns(df)
    r = fxq.check_extreme_returns(df)
    assert r.status == fxq.WARNING
    assert r.detail["flagged"] >= 1


def test_extreme_returns_xau_threshold_higher():
    df = pd.DataFrame([
        _row("2020-01-02", "XAU/USD", 1500.0, source="lbma"),
        _row("2020-01-03", "XAU/USD", 1605.0, source="lbma"),  # +7%, < 10%
    ], columns=fxd.DATASET_COLUMNS)
    df = fxd.compute_fx_returns(df)
    r = fxq.check_extreme_returns(df)
    assert r.status == fxq.PASS  # below 10% threshold


# ---------------------------------------------------------------------------
# Coverage gaps
# ---------------------------------------------------------------------------
def test_coverage_gaps_pass_on_consecutive_dates():
    df = pd.DataFrame([
        _row("2020-01-02", "EUR/USD", 1.10),
        _row("2020-01-03", "EUR/USD", 1.11),
        _row("2020-01-06", "EUR/USD", 1.12),  # weekend gap = 3 days; OK
    ], columns=fxd.DATASET_COLUMNS)
    r = fxq.check_coverage_gaps(df)
    assert r.status == fxq.PASS


def test_coverage_gaps_warns_above_threshold():
    df = pd.DataFrame([
        _row("2020-01-02", "EUR/USD", 1.10),
        _row("2020-02-01", "EUR/USD", 1.12),  # 30-day gap
    ], columns=fxd.DATASET_COLUMNS)
    r = fxq.check_coverage_gaps(df)
    assert r.status == fxq.WARNING
    assert r.detail["flagged"] >= 1


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------
def test_run_inconclusive_when_dataset_missing(tmp_path):
    bogus = tmp_path / "missing_fx.parquet"
    report = fxq.run_fx_data_quality_checks(bogus)
    assert report.verdict == fxq.INCONCLUSIVE
    assert any(c.name == "load_dataset" for c in report.checks)


def test_run_overall_pass_on_clean_frame(tmp_path):
    target = config.REPO_ROOT / "data" / "fx" / "fx_test_runner_pass_v1.csv"
    parquet = target.with_suffix(".parquet")
    df = _eight_asset_frame()
    try:
        df.to_csv(target, index=False)
        report = fxq.run_fx_data_quality_checks(parquet)
        assert report.verdict in (fxq.PASS, fxq.WARNING)
        # On this synthetic frame the only legitimate WARNING source
        # would be coverage gaps (Jan 3 → Jan 6 is fine, weekend), so
        # we expect PASS.
        assert report.verdict == fxq.PASS
    finally:
        for p in (target, parquet):
            if p.exists():
                p.unlink()


def test_run_overall_fail_when_close_missing(tmp_path):
    target_dir = config.REPO_ROOT / "data" / "fx"
    target_dir.mkdir(parents=True, exist_ok=True)
    csv = target_dir / "fx_test_runner_fail_v1.csv"
    parquet = csv.with_suffix(".parquet")
    df = _eight_asset_frame()
    df.loc[df.index[0], "close"] = float("nan")
    try:
        df.to_csv(csv, index=False)
        report = fxq.run_fx_data_quality_checks(parquet)
        assert report.verdict == fxq.FAIL
        names = [c.name for c in report.checks]
        assert "missing_close" in names
    finally:
        for p in (csv, parquet):
            if p.exists():
                p.unlink()


def test_run_overall_warning_on_extreme_return(tmp_path):
    """Extreme returns flag a WARNING but everything else PASSes."""
    target_dir = config.REPO_ROOT / "data" / "fx"
    target_dir.mkdir(parents=True, exist_ok=True)
    csv = target_dir / "fx_test_runner_warn_v1.csv"
    parquet = csv.with_suffix(".parquet")
    df = _eight_asset_frame()
    # Force a >5% jump in EUR/USD between row 0 and row 1, then
    # repropagate returns and rebuild derived rows so the rest of the
    # frame stays internally consistent.
    df.loc[(df["asset"] == "EUR/USD")
              & (df["date"] == pd.Timestamp("2020-01-03")), "close"] = 1.50
    direct = df[~df["is_derived"]].copy()
    derived, _ = fxd.derive_fx_pairs(direct)
    df = pd.concat([direct, derived], ignore_index=True)
    df = df[df["data_quality_status"] == fxd.DATA_QUALITY_OK]
    df = df.sort_values(["asset", "date"]).reset_index(drop=True)
    df = fxd.compute_fx_returns(df)
    try:
        df.to_csv(csv, index=False)
        report = fxq.run_fx_data_quality_checks(parquet)
        assert report.verdict == fxq.WARNING
        extreme = [c for c in report.checks if c.name == "extreme_returns"]
        assert extreme and extreme[0].status == fxq.WARNING
    finally:
        for p in (csv, parquet):
            if p.exists():
                p.unlink()


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def _empty_report() -> fxq.QualityReport:
    return fxq.QualityReport(
        verdict=fxq.PASS, checks=[
            fxq.CheckResult("schema", fxq.PASS, "ok"),
        ], summary={"rows": 0},
    )


def test_writer_rejects_path_outside_results(tmp_path):
    report = _empty_report()
    bad = tmp_path / "report.csv"
    with pytest.raises(fxq.FxQualityReportPathError):
        fxq.write_fx_data_quality_report(report, csv_path=bad,
                                            json_path=None)


def test_writer_rejects_repo_path_outside_results():
    report = _empty_report()
    bad = config.REPO_ROOT / "data" / "fx" / "report.csv"
    with pytest.raises(fxq.FxQualityReportPathError):
        fxq.write_fx_data_quality_report(report, csv_path=bad,
                                            json_path=None)


def test_writer_writes_csv_inside_results():
    report = _empty_report()
    csv_target = config.RESULTS_DIR / "fx_data_quality_report_test.csv"
    json_target = config.RESULTS_DIR / "fx_data_quality_report_test.json"
    try:
        out = fxq.write_fx_data_quality_report(
            report, csv_path=csv_target, json_path=json_target,
        )
        assert "csv" in out
        assert csv_target.exists()
        loaded = pd.read_csv(csv_target)
        assert "check_name" in loaded.columns
        assert "status" in loaded.columns
    finally:
        for p in (csv_target, json_target):
            if p.exists():
                p.unlink()


# ---------------------------------------------------------------------------
# Safety invariants — the module must not introduce strategy / broker /
# execution / paper-trading / live-trading / API-key code.
# ---------------------------------------------------------------------------
_MODULE_PATH = Path(fxq.__file__)


def _read(path: Path) -> str:
    return path.read_text(errors="ignore")


def test_no_broker_imports_in_quality_module():
    text = _read(_MODULE_PATH)
    assert "ccxt" not in text
    assert "alpaca" not in text.lower()
    assert "kraken" not in text.lower()
    assert "ig_bank" not in text.lower()
    assert "oanda" not in text.lower()


def test_no_api_key_reads_in_quality_module():
    text = _read(_MODULE_PATH)
    assert "API_KEY" not in text
    assert "API_SECRET" not in text
    assert "os.environ" not in text
    assert "os.getenv" not in text


def test_no_order_placement_strings_in_quality_module():
    forbidden = (
        "create_order", "place_order", "submit_order",
        "send_order", "cancel_order", "AddOrder", "CancelOrder",
    )
    text = _read(_MODULE_PATH)
    for token in forbidden:
        assert token not in text, f"{token} found in quality module"


def test_no_paper_or_live_trading_enablement():
    text = _read(_MODULE_PATH)
    assert "LIVE_TRADING_ENABLED = True" not in text
    assert "paper_trading_allowed = True" not in text
    assert "execution_allowed = True" not in text
    assert "ENABLE_LIVE" not in text
    assert "UNLOCK_TRADING" not in text
    assert "FORCE_TRADE" not in text


def test_no_strategy_code_in_quality_module():
    text = _read(_MODULE_PATH)
    forbidden = (
        "strategy_registry", "scorecard", "backtester",
        "register_strategy", "from src import paper_trader",
        "from src import backtester",
    )
    for token in forbidden:
        assert token not in text, f"strategy-leak token {token!r}"


def test_safety_lock_remains_locked_after_quality_import():
    s = safety_lock.status()
    assert not s.execution_allowed
    assert not s.paper_trading_allowed
    assert not s.kraken_connection_allowed
    assert s.safety_lock_status == "locked"


# ---------------------------------------------------------------------------
# .gitignore must exclude the generated quality report CSV
# ---------------------------------------------------------------------------
def test_gitignore_excludes_results_csv():
    text = (REPO_ROOT / ".gitignore").read_text()
    # The existing generic rule covers the report.
    assert "results/*.csv" in text
    assert "results/*.json" in text
