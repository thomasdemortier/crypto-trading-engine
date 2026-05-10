"""
FX research dataset — data quality checks (read-only).

Consumes the v1 FX dataset produced by `src/fx_research_dataset.py`
(`data/fx/fx_daily_v1.parquet`) and runs a fixed battery of checks:

    1.  Schema             (locked column set + dtypes loosely)
    2.  Asset coverage     (every expected pair present)
    3.  Date monotonicity  (per-asset ascending dates)
    4.  Duplicate rows     (no duplicate (asset, date) pairs)
    5.  Missing close      (no NaN close on OK rows)
    6.  Return consistency (return_1d + log_return_1d ≈ recomputed)
    7.  Derived sanity     (USD/JPY, USD/CHF, GBP/USD ≈ EUR-leg ratios)
    8.  Extreme returns    (|FX|>5%, |XAU|>10% — flagged, not deleted)
    9.  Coverage gaps      (per-asset gaps > 7 calendar days)
    10. Source completeness (per-asset row count, dates, coverage_days)

Hard rules (locked):
    * Read-only. No network. No broker. No API keys. No execution.
    * No strategy code, no backtest, no paper / live trading.
    * Output `results/fx_data_quality_report.csv` is gitignored under
      the existing `results/*.csv` rule.
    * `write_fx_data_quality_report` refuses to write outside
      `results/`.
    * No row is deleted, no value is imputed. Findings are recorded
      and surfaced; the caller decides what to do.

Exit semantics:
    The verdict is one of {PASS, WARNING, FAIL, INCONCLUSIVE}. The
    CLI exits 0 on PASS / WARNING and 1 on FAIL; INCONCLUSIVE
    (dataset missing) exits 2 so a CI step can distinguish "dataset
    not built yet" from "dataset is broken".
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config, fx_research_dataset, utils

logger = utils.get_logger("cte.fx_data_quality")


# ---------------------------------------------------------------------------
# Locked constants
# ---------------------------------------------------------------------------
DEFAULT_DATASET_PATH: Path = fx_research_dataset.DATASET_PARQUET
RESULTS_REPORT_PATH: Path = config.RESULTS_DIR / "fx_data_quality_report.csv"
RESULTS_REPORT_JSON_PATH: Path = (
    config.RESULTS_DIR / "fx_data_quality_report.json"
)

EXPECTED_ASSETS: Tuple[str, ...] = (
    "EUR/USD", "EUR/GBP", "EUR/JPY", "EUR/CHF",
    "USD/JPY", "USD/CHF", "GBP/USD", "XAU/USD",
)

REQUIRED_COLUMNS: Tuple[str, ...] = tuple(fx_research_dataset.DATASET_COLUMNS)

# Tolerances (locked).
RETURN_RECOMPUTE_TOL = 1e-9
DERIVED_RATIO_REL_TOL = 1e-9
EXTREME_FX_THRESHOLD = 0.05      # 5 percent
EXTREME_XAU_THRESHOLD = 0.10     # 10 percent
COVERAGE_GAP_DAYS = 7            # weekend gaps OK; > 7d flagged

MIN_ROWS_FOR_QUALITY = 10        # below this, INCONCLUSIVE

# Verdict severity (lower wins on combine).
SEVERITY_RANK = {
    "PASS": 0,
    "WARNING": 1,
    "FAIL": 2,
    "INCONCLUSIVE": 3,
}
PASS, WARNING, FAIL, INCONCLUSIVE = "PASS", "WARNING", "FAIL", "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    """One row in the quality report."""
    name: str
    status: str  # PASS / WARNING / FAIL / INCONCLUSIVE
    message: str
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Bundle of per-check results plus an overall verdict."""
    verdict: str
    checks: List[CheckResult]
    summary: Dict[str, Any]

    def to_records(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for r in self.checks:
            row = {
                "check_name": r.name,
                "status": r.status,
                "message": r.message,
                "detail_json": json.dumps(r.detail, default=str),
            }
            rows.append(row)
        rows.append({
            "check_name": "overall_verdict",
            "status": self.verdict,
            "message": (f"overall verdict from {len(self.checks)} "
                        f"check(s)"),
            "detail_json": json.dumps(self.summary, default=str),
        })
        return rows

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "checks": [asdict(c) for c in self.checks],
            "summary": self.summary,
        }


def _combine_verdict(checks: Sequence[CheckResult]) -> str:
    """Reduce per-check statuses to one overall verdict."""
    if not checks:
        return INCONCLUSIVE
    worst = max(SEVERITY_RANK[c.status] for c in checks)
    for k, v in SEVERITY_RANK.items():
        if v == worst:
            return k
    return INCONCLUSIVE


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
class FxDatasetMissingError(FileNotFoundError):
    """Raised when the FX dataset parquet/CSV cannot be located."""


def load_fx_dataset(path: Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load the v1 FX dataset. Tries parquet first, then a same-stem
    CSV fallback. Raises `FxDatasetMissingError` if neither is found
    or if the file exists but is empty / unreadable. Never imputes."""
    path = Path(path)
    if path.exists():
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            csv_fallback = path.with_suffix(".csv")
            if csv_fallback.exists():
                logger.warning(
                    "parquet unreadable (%s); falling back to CSV at %s",
                    type(exc).__name__, csv_fallback,
                )
                df = pd.read_csv(csv_fallback)
            else:
                raise FxDatasetMissingError(
                    f"could not read {path} ({type(exc).__name__}: {exc}) "
                    f"and no CSV fallback at {csv_fallback}"
                ) from exc
    else:
        csv_fallback = path.with_suffix(".csv")
        if csv_fallback.exists():
            df = pd.read_csv(csv_fallback)
        else:
            raise FxDatasetMissingError(
                f"FX dataset not found at {path}. "
                "Run `python main.py build_fx_dataset` first."
            )
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=False, errors="coerce")
    if "is_derived" in df.columns:
        df["is_derived"] = df["is_derived"].astype(bool)
    return df


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def validate_fx_quality_schema(df: pd.DataFrame) -> CheckResult:
    if df is None or not isinstance(df, pd.DataFrame):
        return CheckResult("schema", FAIL, "input is not a DataFrame")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if missing:
        return CheckResult(
            "schema", FAIL,
            f"missing required columns: {missing}",
            {"missing": missing, "extra": extra},
        )
    if extra:
        return CheckResult(
            "schema", FAIL,
            f"unexpected extra columns: {extra}",
            {"missing": missing, "extra": extra},
        )
    return CheckResult(
        "schema", PASS,
        f"all {len(REQUIRED_COLUMNS)} required columns present",
        {"columns": list(REQUIRED_COLUMNS)},
    )


def check_asset_coverage(df: pd.DataFrame) -> CheckResult:
    if "asset" not in df.columns:
        return CheckResult("asset_coverage", FAIL,
                            "asset column missing")
    ok = df[df["data_quality_status"]
             == fx_research_dataset.DATA_QUALITY_OK] \
        if "data_quality_status" in df.columns else df
    have = sorted(ok["asset"].dropna().unique().tolist())
    missing = [a for a in EXPECTED_ASSETS if a not in have]
    extra = [a for a in have if a not in EXPECTED_ASSETS]
    if missing:
        return CheckResult(
            "asset_coverage", FAIL,
            f"missing expected asset(s): {missing}",
            {"have": have, "missing": missing, "extra": extra},
        )
    return CheckResult(
        "asset_coverage", PASS,
        f"all {len(EXPECTED_ASSETS)} expected asset(s) present",
        {"have": have, "extra": extra},
    )


def check_date_monotonicity(df: pd.DataFrame) -> CheckResult:
    if "date" not in df.columns or "asset" not in df.columns:
        return CheckResult("date_monotonicity", FAIL,
                            "date or asset column missing")
    ok = df.dropna(subset=["date"])
    bad: List[str] = []
    for asset, sub in ok.groupby("asset", sort=True):
        if not sub["date"].is_monotonic_increasing:
            sub_sorted = sub.sort_values("date")
            n_violations = int(
                (sub["date"].values != sub_sorted["date"].values).sum()
            )
            bad.append(f"{asset} ({n_violations} out-of-order)")
    if bad:
        return CheckResult(
            "date_monotonicity", FAIL,
            f"non-monotonic dates: {bad[:5]}",
            {"violations": bad},
        )
    return CheckResult(
        "date_monotonicity", PASS,
        "every asset has strictly ascending dates",
    )


def check_duplicate_rows(df: pd.DataFrame) -> CheckResult:
    if "date" not in df.columns or "asset" not in df.columns:
        return CheckResult("duplicate_rows", FAIL,
                            "date or asset column missing")
    sub = df.dropna(subset=["date"])
    dups = sub.duplicated(subset=["asset", "date"], keep=False)
    n = int(dups.sum())
    if n > 0:
        sample = sub.loc[dups, ["asset", "date"]].head(5).to_dict("records")
        return CheckResult(
            "duplicate_rows", FAIL,
            f"{n} duplicate (asset, date) row(s) detected",
            {"duplicate_count": n, "sample": sample},
        )
    return CheckResult(
        "duplicate_rows", PASS,
        "no duplicate (asset, date) rows",
    )


def check_missing_close(df: pd.DataFrame) -> CheckResult:
    if "close" not in df.columns:
        return CheckResult("missing_close", FAIL, "close column missing")
    ok_mask = (df.get("data_quality_status",
                          fx_research_dataset.DATA_QUALITY_OK)
                  == fx_research_dataset.DATA_QUALITY_OK)
    ok = df[ok_mask]
    n_missing = int(ok["close"].isna().sum())
    if n_missing > 0:
        sample = ok[ok["close"].isna()][["asset", "date"]].head(5).to_dict(
            "records"
        )
        return CheckResult(
            "missing_close", FAIL,
            f"{n_missing} OK row(s) have a missing close",
            {"missing_count": n_missing, "sample": sample},
        )
    return CheckResult(
        "missing_close", PASS,
        "no missing close values on OK rows",
    )


def check_return_consistency(df: pd.DataFrame,
                                tol: float = RETURN_RECOMPUTE_TOL,
                                ) -> CheckResult:
    """Recompute return_1d and log_return_1d from close and verify
    every existing value is within tolerance."""
    needed = {"asset", "date", "close", "return_1d", "log_return_1d"}
    if not needed.issubset(df.columns):
        return CheckResult("return_consistency", FAIL,
                            "required columns missing")
    sub = df.dropna(subset=["date"]).sort_values(["asset", "date"]).copy()
    if sub.empty:
        return CheckResult("return_consistency", INCONCLUSIVE,
                            "no priced rows to verify")
    grouped = sub.groupby("asset", sort=False)
    prev_close = grouped["close"].shift(1)
    expected_simple = sub["close"] / prev_close - 1.0
    expected_log = np.log(sub["close"]) - np.log(prev_close)
    have_simple = sub["return_1d"]
    have_log = sub["log_return_1d"]
    # Compare only where both expected and stored are finite.
    simple_mask = expected_simple.notna() & have_simple.notna()
    log_mask = expected_log.notna() & have_log.notna()
    simple_resid = (have_simple[simple_mask]
                       - expected_simple[simple_mask]).abs()
    log_resid = (have_log[log_mask] - expected_log[log_mask]).abs()
    max_simple = float(simple_resid.max()) if not simple_resid.empty else 0.0
    max_log = float(log_resid.max()) if not log_resid.empty else 0.0
    n_simple_bad = int((simple_resid > tol).sum())
    n_log_bad = int((log_resid > tol).sum())
    detail = {
        "tolerance": tol,
        "max_residual_simple": max_simple,
        "max_residual_log": max_log,
        "n_simple_violations": n_simple_bad,
        "n_log_violations": n_log_bad,
        "n_simple_compared": int(simple_mask.sum()),
        "n_log_compared": int(log_mask.sum()),
    }
    if n_simple_bad or n_log_bad:
        return CheckResult(
            "return_consistency", FAIL,
            f"{n_simple_bad} simple + {n_log_bad} log return(s) "
            f"exceed tol={tol}",
            detail,
        )
    return CheckResult(
        "return_consistency", PASS,
        f"max residual: simple={max_simple:.2e}, log={max_log:.2e}",
        detail,
    )


def check_derived_pair_sanity(df: pd.DataFrame,
                                 rel_tol: float = DERIVED_RATIO_REL_TOL,
                                 ) -> CheckResult:
    """For each derived pair (USD/JPY, USD/CHF, GBP/USD) recompute
    close from the EUR legs on the same date and verify the residual
    is within relative tolerance."""
    needed = {"asset", "date", "close", "is_derived"}
    if not needed.issubset(df.columns):
        return CheckResult("derived_pair_sanity", FAIL,
                            "required columns missing")
    ok = df[df["data_quality_status"]
             == fx_research_dataset.DATA_QUALITY_OK] \
        if "data_quality_status" in df.columns else df
    wide = ok.pivot_table(
        index="date", columns="asset", values="close", aggfunc="last",
    )
    spec: Tuple[Tuple[str, str, str], ...] = (
        ("USD/JPY", "EUR/JPY", "EUR/USD"),
        ("USD/CHF", "EUR/CHF", "EUR/USD"),
        ("GBP/USD", "EUR/USD", "EUR/GBP"),
    )
    detail: Dict[str, Any] = {"per_pair": {}}
    worst_status = PASS
    msgs: List[str] = []
    for target, num, den in spec:
        if any(p not in wide.columns for p in (target, num, den)):
            detail["per_pair"][target] = {
                "status": INCONCLUSIVE,
                "reason": "missing leg",
            }
            if SEVERITY_RANK[INCONCLUSIVE] > SEVERITY_RANK[worst_status]:
                worst_status = INCONCLUSIVE
            msgs.append(f"{target}: missing leg")
            continue
        sub = wide[[target, num, den]].dropna()
        if sub.empty:
            detail["per_pair"][target] = {
                "status": INCONCLUSIVE,
                "reason": "no overlapping dates",
            }
            if SEVERITY_RANK[INCONCLUSIVE] > SEVERITY_RANK[worst_status]:
                worst_status = INCONCLUSIVE
            msgs.append(f"{target}: no overlapping dates")
            continue
        expected = sub[num] / sub[den]
        rel = (sub[target] - expected).abs() / expected.abs()
        max_rel = float(rel.max())
        n_violations = int((rel > rel_tol).sum())
        per = {
            "status": PASS if n_violations == 0 else FAIL,
            "max_relative_residual": max_rel,
            "n_violations": n_violations,
            "n_compared": int(len(sub)),
            "rel_tol": rel_tol,
            "formula": f"{num} / {den}",
        }
        detail["per_pair"][target] = per
        if per["status"] == FAIL:
            worst_status = FAIL
            msgs.append(
                f"{target}: {n_violations} row(s) > rel_tol "
                f"(max={max_rel:.2e})"
            )
        else:
            msgs.append(f"{target}: max_rel={max_rel:.2e}")
    return CheckResult(
        "derived_pair_sanity", worst_status,
        "; ".join(msgs) if msgs else "no derived pairs to check",
        detail,
    )


def check_extreme_returns(df: pd.DataFrame,
                             fx_threshold: float = EXTREME_FX_THRESHOLD,
                             xau_threshold: float = EXTREME_XAU_THRESHOLD,
                             ) -> CheckResult:
    """Flag — never delete — daily returns above the per-asset
    threshold."""
    if "return_1d" not in df.columns or "asset" not in df.columns:
        return CheckResult("extreme_returns", FAIL,
                            "return_1d or asset column missing")
    sub = df.dropna(subset=["return_1d"]).copy()
    if sub.empty:
        return CheckResult("extreme_returns", INCONCLUSIVE,
                            "no return rows to scan")
    threshold = np.where(
        sub["asset"].eq("XAU/USD"), xau_threshold, fx_threshold,
    )
    flagged_mask = sub["return_1d"].abs() > pd.Series(
        threshold, index=sub.index
    )
    n = int(flagged_mask.sum())
    if n == 0:
        return CheckResult(
            "extreme_returns", PASS,
            f"no |return_1d| above thresholds "
            f"(FX>{fx_threshold}, XAU>{xau_threshold})",
            {"fx_threshold": fx_threshold,
             "xau_threshold": xau_threshold,
             "flagged": 0},
        )
    by_asset = (
        sub.loc[flagged_mask].groupby("asset")["return_1d"]
           .agg(["count", "min", "max"]).to_dict("index")
    )
    sample = sub.loc[flagged_mask, ["asset", "date", "return_1d"]] \
                .head(10).to_dict("records")
    return CheckResult(
        "extreme_returns", WARNING,
        f"{n} extreme daily return(s) flagged (not deleted)",
        {
            "fx_threshold": fx_threshold,
            "xau_threshold": xau_threshold,
            "flagged": n,
            "by_asset": by_asset,
            "sample": sample,
        },
    )


def check_coverage_gaps(df: pd.DataFrame,
                            gap_days: int = COVERAGE_GAP_DAYS,
                            ) -> CheckResult:
    """Per-asset: detect calendar-day gaps strictly greater than
    `gap_days` between successive observations."""
    if "date" not in df.columns or "asset" not in df.columns:
        return CheckResult("coverage_gaps", FAIL,
                            "date or asset column missing")
    sub = df.dropna(subset=["date"]).copy()
    if sub.empty:
        return CheckResult("coverage_gaps", INCONCLUSIVE,
                            "no dated rows")
    sub = sub.sort_values(["asset", "date"])
    sub["gap_days"] = sub.groupby("asset")["date"].diff().dt.days
    flagged = sub[sub["gap_days"] > gap_days]
    n = int(len(flagged))
    by_asset = (
        flagged.groupby("asset")["gap_days"]
               .agg(["count", "max"]).to_dict("index")
    )
    if n == 0:
        return CheckResult(
            "coverage_gaps", PASS,
            f"no per-asset gaps > {gap_days} calendar day(s)",
            {"gap_days_threshold": gap_days, "flagged": 0},
        )
    sample = flagged[["asset", "date", "gap_days"]].head(10).to_dict(
        "records"
    )
    return CheckResult(
        "coverage_gaps", WARNING,
        f"{n} gap(s) > {gap_days} calendar day(s) across "
        f"{len(by_asset)} asset(s)",
        {
            "gap_days_threshold": gap_days,
            "flagged": n,
            "by_asset": by_asset,
            "sample": sample,
        },
    )


def summarize_asset_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    """Per-asset row count, start, end, coverage_days. Tolerant of
    missing columns."""
    if "asset" not in df.columns or "date" not in df.columns:
        return {}
    ok_mask = (df.get("data_quality_status",
                          fx_research_dataset.DATA_QUALITY_OK)
                  == fx_research_dataset.DATA_QUALITY_OK)
    ok = df[ok_mask].dropna(subset=["date"])
    out: Dict[str, Any] = {}
    for asset, sub in ok.groupby("asset", sort=True):
        out[str(asset)] = {
            "rows": int(len(sub)),
            "start_date": pd.Timestamp(sub["date"].min()).date().isoformat(),
            "end_date": pd.Timestamp(sub["date"].max()).date().isoformat(),
            "coverage_days": int(
                (sub["date"].max() - sub["date"].min()).days
            ),
            "is_derived": bool(sub["is_derived"].iloc[0])
                if "is_derived" in sub.columns else False,
            "source": str(sub["source"].iloc[0])
                if "source" in sub.columns else "",
        }
    return out


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------
def run_fx_data_quality_checks(
    path: Path = DEFAULT_DATASET_PATH,
) -> QualityReport:
    """Run every check and return a `QualityReport`. Fail-soft on
    missing dataset → returns an INCONCLUSIVE report (no crash)."""
    utils.assert_paper_only()
    try:
        df = load_fx_dataset(path)
    except FxDatasetMissingError as exc:
        return QualityReport(
            verdict=INCONCLUSIVE,
            checks=[CheckResult(
                "load_dataset", INCONCLUSIVE, str(exc),
                {"path": str(path)},
            )],
            summary={"path": str(path), "rows": 0,
                     "loaded": False, "error": str(exc)},
        )

    if len(df) < MIN_ROWS_FOR_QUALITY:
        return QualityReport(
            verdict=INCONCLUSIVE,
            checks=[CheckResult(
                "load_dataset", INCONCLUSIVE,
                f"only {len(df)} row(s) — below minimum "
                f"{MIN_ROWS_FOR_QUALITY}",
                {"rows": int(len(df))},
            )],
            summary={"path": str(path), "rows": int(len(df)),
                     "loaded": True,
                     "error": "insufficient rows"},
        )

    checks: List[CheckResult] = [
        validate_fx_quality_schema(df),
    ]
    # Only continue if schema is at least structurally valid; otherwise
    # downstream checks will FAIL with confusing messages.
    if checks[0].status == PASS:
        checks.extend([
            check_asset_coverage(df),
            check_date_monotonicity(df),
            check_duplicate_rows(df),
            check_missing_close(df),
            check_return_consistency(df),
            check_derived_pair_sanity(df),
            check_extreme_returns(df),
            check_coverage_gaps(df),
        ])
    verdict = _combine_verdict(checks)
    summary = {
        "path": str(path),
        "rows": int(len(df)),
        "loaded": True,
        "asset_coverage": summarize_asset_coverage(df),
        "verdict": verdict,
    }
    return QualityReport(verdict=verdict, checks=checks, summary=summary)


# ---------------------------------------------------------------------------
# Writer — restricted to results/
# ---------------------------------------------------------------------------
class FxQualityReportPathError(ValueError):
    """Raised if the writer is asked to write outside `results/`."""


def _assert_inside_results_dir(path: Path) -> None:
    resolved = path.resolve()
    results_root = config.RESULTS_DIR.resolve()
    try:
        resolved.relative_to(results_root)
    except ValueError as exc:
        raise FxQualityReportPathError(
            f"refusing to write outside results/: {resolved}"
        ) from exc


def write_fx_data_quality_report(
    report: QualityReport,
    csv_path: Path = RESULTS_REPORT_PATH,
    json_path: Optional[Path] = RESULTS_REPORT_JSON_PATH,
) -> Dict[str, str]:
    """Persist the report to `results/`. Both targets must be inside
    `results/`; anything else is rejected. Both outputs are gitignored
    by the existing `results/*.csv` and `results/*.json` rules."""
    utils.assert_paper_only()
    csv_path = Path(csv_path)
    _assert_inside_results_dir(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(report.to_records())
    df.to_csv(csv_path, index=False)
    written: Dict[str, str] = {"csv": str(csv_path)}
    if json_path is not None:
        json_path = Path(json_path)
        _assert_inside_results_dir(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report.to_dict(), indent=2, default=str)
        )
        written["json"] = str(json_path)
    return written
