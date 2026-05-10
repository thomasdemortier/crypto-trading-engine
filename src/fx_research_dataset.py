"""
FX research dataset v1 — reproducible daily FX + gold dataset.

Builds the first reproducible FX dataset for future research, drawing
ONLY from free, public, no-key sources that passed the FX/crypto
source audit (see `src/fx_crypto_source_audit.py`).

Hard rules (locked):
    * Public endpoints only. NO API keys ever read or required.
    * No private endpoints, no broker, no order routing, no balances.
    * No execution, no paper trading, no live trading.
    * Daily frequency only — no intraday data.
    * No tradable spread assumptions, no volume assumptions.
    * Derived crosses must be flagged with `is_derived=True` and a
      `source_pair` describing how they were computed.
    * Network failures NEVER crash the build — every fetch is wrapped
      in a guarded try/except and the offending source degrades to
      `data_quality_status="missing"` with a `notes` explanation.
    * Generated parquet/CSV at `data/fx/` is gitignored under the
      existing `data/fx/*` rule. Writes are restricted to `data/fx/`.
    * No silent forward-fill. If a row is missing, it stays missing.

Sources used (only the audit's PASS-tier, no-key sources):
    1. ECB SDMX — daily reference rates, EUR-quoted; covers 1999→
       (~9,986 days). Pairs: EUR/USD, EUR/GBP, EUR/JPY, EUR/CHF.
    2. LBMA London PM gold fix — daily, free JSON, since 1968.
       Pair: XAU/USD.
    3. Frankfurter — wraps the ECB feed, used as an explicit fallback
       only if the SDMX endpoint is unreachable.

Derived pairs (computed via EUR cross, both legs same date):
    USD/JPY  =  EUR/JPY  /  EUR/USD
    USD/CHF  =  EUR/CHF  /  EUR/USD
    GBP/USD  =  EUR/USD  /  EUR/GBP

Output:
    data/fx/fx_daily_v1.parquet  (primary, gitignored)
    data/fx/fx_daily_v1.csv      (companion for inspection, gitignored)

Schema (locked):
    date                 (datetime64[ns], UTC, daily)
    asset                (str, e.g. "EUR/USD", "XAU/USD")
    source               (str, e.g. "ecb_sdmx", "lbma", "derived")
    base                 (str, ISO/asset code, e.g. "EUR", "XAU")
    quote                (str, ISO code, e.g. "USD")
    close                (float, the daily reference rate / fix)
    return_1d            (float, simple daily return; NaN at series start)
    log_return_1d        (float, natural-log daily return; NaN at start)
    is_derived           (bool, True for cross-derived pairs)
    source_pair          (str, "EUR/JPY÷EUR/USD" for derived, else "")
    data_quality_status  (str: "ok" | "missing" | "stale" | "warning")
    notes                (str, free-form provenance / quality note)
"""
from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config, utils

logger = utils.get_logger("cte.fx_research_dataset")


# ---------------------------------------------------------------------------
# Locked constants
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT = 12.0
DEFAULT_USER_AGENT = "cte-fx-research-dataset/0.1"

# Output paths — writes are restricted to this directory.
FX_DATA_DIR: Path = config.REPO_ROOT / "data" / "fx"
DATASET_PARQUET: Path = FX_DATA_DIR / "fx_daily_v1.parquet"
DATASET_CSV: Path = FX_DATA_DIR / "fx_daily_v1.csv"

# Schema (locked).
DATASET_COLUMNS: List[str] = [
    "date",
    "asset",
    "source",
    "base",
    "quote",
    "close",
    "return_1d",
    "log_return_1d",
    "is_derived",
    "source_pair",
    "data_quality_status",
    "notes",
]

DATA_QUALITY_OK = "ok"
DATA_QUALITY_MISSING = "missing"
DATA_QUALITY_STALE = "stale"
DATA_QUALITY_WARNING = "warning"
DATA_QUALITY_STATUSES: Tuple[str, ...] = (
    DATA_QUALITY_OK, DATA_QUALITY_MISSING,
    DATA_QUALITY_STALE, DATA_QUALITY_WARNING,
)

SOURCE_ECB = "ecb_sdmx"
SOURCE_FRANKFURTER = "frankfurter_app"
SOURCE_LBMA = "lbma"
SOURCE_DERIVED = "derived"

# ECB SDMX — direct EUR-quoted daily reference rate. Each entry is
# (asset_pair_label, ISO currency code expected by ECB).
ECB_DIRECT_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("EUR/USD", "USD"),
    ("EUR/GBP", "GBP"),
    ("EUR/JPY", "JPY"),
    ("EUR/CHF", "CHF"),
)

# Derived crosses: target_pair, numerator_pair, denominator_pair, label.
DERIVED_PAIRS: Tuple[Tuple[str, str, str, str], ...] = (
    ("USD/JPY", "EUR/JPY", "EUR/USD", "EUR/JPY÷EUR/USD"),
    ("USD/CHF", "EUR/CHF", "EUR/USD", "EUR/CHF÷EUR/USD"),
    ("GBP/USD", "EUR/USD", "EUR/GBP", "EUR/USD÷EUR/GBP"),
)


# ---------------------------------------------------------------------------
# Fail-soft HTTP helper (mirrors the audit module's contract)
# ---------------------------------------------------------------------------
@dataclass
class _Response:
    ok: bool
    status_code: Optional[int]
    payload: Optional[Any]
    text: Optional[str]
    error: Optional[str]


def _http_get(url: str, timeout: float = DEFAULT_TIMEOUT,
                user_agent: str = DEFAULT_USER_AGENT,
                parse_json: bool = True) -> _Response:
    """Public, unauthenticated GET. Never raises — returns
    `_Response(ok=False, ...)` on any failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            text = raw.decode("utf-8", errors="replace")
            payload: Any = None
            if parse_json:
                try:
                    payload = json.loads(text)
                except ValueError:
                    payload = None
            return _Response(True, r.status, payload, text, None)
    except urllib.error.HTTPError as exc:
        return _Response(False, exc.code, None, None,
                          f"http_error: {exc.reason}")
    except urllib.error.URLError as exc:
        return _Response(False, None, None, None,
                          f"url_error: {exc.reason}")
    except (TimeoutError, OSError) as exc:
        return _Response(False, None, None, None, f"network_error: {exc}")
    except Exception as exc:  # noqa: BLE001 — fail-soft policy
        return _Response(False, None, None, None,
                          f"unexpected: {type(exc).__name__}")


HttpGet = Callable[[str], _Response]


# ---------------------------------------------------------------------------
# ECB SDMX
# ---------------------------------------------------------------------------
def _parse_ecb_csv(text: str, pair_label: str, ccy: str
                    ) -> Tuple[pd.DataFrame, List[str]]:
    """Parse an ECB SDMX CSV payload into a long-format DataFrame.

    Returns (df, warnings). ECB returns "." for non-fixing days
    (TARGET holidays); those rows are dropped and counted as a
    quality warning. Never raises — malformed lines are skipped."""
    warnings: List[str] = []
    rows: List[Dict[str, Any]] = []
    skipped_dot = 0
    skipped_parse = 0
    base, quote = pair_label.split("/", 1)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("KEY,"):
            continue
        parts = line.split(",")
        if len(parts) < 8:
            skipped_parse += 1
            continue
        date_str = parts[6].strip()
        value_str = parts[7].strip()
        if not date_str:
            skipped_parse += 1
            continue
        if value_str in ("", "."):
            skipped_dot += 1
            continue
        try:
            d = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
        except ValueError:
            skipped_parse += 1
            continue
        try:
            close = float(value_str)
        except ValueError:
            skipped_parse += 1
            continue
        if close <= 0 or not math.isfinite(close):
            skipped_parse += 1
            continue
        rows.append({
            "date": pd.Timestamp(d).normalize(),
            "asset": pair_label,
            "source": SOURCE_ECB,
            "base": base,
            "quote": quote,
            "close": close,
            "is_derived": False,
            "source_pair": "",
            "data_quality_status": DATA_QUALITY_OK,
            "notes": ("ECB SDMX EUR-quoted daily reference rate "
                      "(holiday gaps not forward-filled)"),
        })
    if skipped_dot:
        warnings.append(
            f"{pair_label}: skipped {skipped_dot} TARGET-holiday rows "
            f"(value='.')"
        )
    if skipped_parse:
        warnings.append(
            f"{pair_label}: skipped {skipped_parse} unparseable rows"
        )
    if not rows:
        warnings.append(f"{pair_label}: ECB returned 0 usable rows")
        return _empty_long_frame(), warnings
    df = pd.DataFrame(rows)
    df = df.sort_values("date").drop_duplicates(
        subset=["date", "asset"], keep="last"
    ).reset_index(drop=True)
    return df, warnings


def _empty_long_frame() -> pd.DataFrame:
    """Return an empty DataFrame with the locked schema."""
    return pd.DataFrame(columns=DATASET_COLUMNS)


def fetch_ecb_rates(http_get: HttpGet = _http_get) -> Tuple[
    pd.DataFrame, List[str]
]:
    """Fetch the four direct ECB EUR-quoted reference-rate series
    (USD, GBP, JPY, CHF). Fail-soft: on any error, the missing pair
    is recorded as a `missing` placeholder row and a warning is
    emitted. Returns (df_long, warnings)."""
    frames: List[pd.DataFrame] = []
    warnings: List[str] = []
    for pair_label, ccy in ECB_DIRECT_PAIRS:
        url = (f"https://data-api.ecb.europa.eu/service/data/EXR/"
               f"D.{ccy}.EUR.SP00.A?format=csvdata")
        r = http_get(url)
        if not r.ok or not r.text:
            warnings.append(
                f"{pair_label}: ECB fetch failed — "
                f"{r.error or 'no payload'}"
            )
            frames.append(_missing_placeholder(
                pair_label, SOURCE_ECB,
                f"ECB fetch failed: {r.error or 'no payload'}",
            ))
            continue
        df, warns = _parse_ecb_csv(r.text, pair_label, ccy)
        warnings.extend(warns)
        if df.empty:
            frames.append(_missing_placeholder(
                pair_label, SOURCE_ECB,
                "ECB returned 0 usable rows",
            ))
        else:
            frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames \
            else _empty_long_frame()
    return _ensure_columns(out), warnings


def _missing_placeholder(pair_label: str, source: str, note: str
                            ) -> pd.DataFrame:
    """A single-row placeholder used when a source is unreachable.
    The row carries no price (close=NaN) and `data_quality_status="missing"`
    so the caller can surface the gap without silently dropping the asset."""
    base, quote = pair_label.split("/", 1)
    row = {
        "date": pd.NaT,
        "asset": pair_label,
        "source": source,
        "base": base,
        "quote": quote,
        "close": float("nan"),
        "return_1d": float("nan"),
        "log_return_1d": float("nan"),
        "is_derived": False,
        "source_pair": "",
        "data_quality_status": DATA_QUALITY_MISSING,
        "notes": note,
    }
    return pd.DataFrame([row], columns=DATASET_COLUMNS)


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee every locked column exists and is in the right order."""
    out = df.copy()
    for col in DATASET_COLUMNS:
        if col not in out.columns:
            if col in ("return_1d", "log_return_1d", "close"):
                out[col] = float("nan")
            elif col == "is_derived":
                out[col] = False
            elif col == "data_quality_status":
                out[col] = DATA_QUALITY_OK
            else:
                out[col] = ""
    return out[DATASET_COLUMNS]


# ---------------------------------------------------------------------------
# Derived crosses
# ---------------------------------------------------------------------------
def derive_fx_pairs(direct_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Compute USD/JPY, USD/CHF, GBP/USD as ratios of EUR-quoted legs.

        USD/JPY  =  EUR/JPY  /  EUR/USD
        USD/CHF  =  EUR/CHF  /  EUR/USD
        GBP/USD  =  EUR/USD  /  EUR/GBP

    Both legs must be present on the same date; missing-date pairs
    are dropped (no forward-fill). Returns (df_long, warnings)."""
    warnings: List[str] = []
    if direct_df is None or direct_df.empty:
        warnings.append("derive_fx_pairs: input frame is empty")
        return _empty_long_frame(), warnings

    # Build a wide lookup keyed by date.
    direct_clean = direct_df[
        direct_df["data_quality_status"] == DATA_QUALITY_OK
    ].copy()
    if direct_clean.empty:
        warnings.append("derive_fx_pairs: no OK rows to derive from")
        return _empty_long_frame(), warnings

    wide = direct_clean.pivot_table(
        index="date", columns="asset", values="close", aggfunc="last",
    )

    frames: List[pd.DataFrame] = []
    for target_pair, num_pair, den_pair, label in DERIVED_PAIRS:
        if num_pair not in wide.columns or den_pair not in wide.columns:
            warnings.append(
                f"{target_pair}: cannot derive — missing leg "
                f"({num_pair} or {den_pair})"
            )
            frames.append(_missing_placeholder(
                target_pair, SOURCE_DERIVED,
                f"derived from {label}; leg unavailable",
            ))
            continue
        sub = wide[[num_pair, den_pair]].dropna()
        if sub.empty:
            warnings.append(
                f"{target_pair}: no overlapping dates between "
                f"{num_pair} and {den_pair}"
            )
            frames.append(_missing_placeholder(
                target_pair, SOURCE_DERIVED,
                f"derived from {label}; no overlapping dates",
            ))
            continue
        ratio = sub[num_pair] / sub[den_pair]
        base, quote = target_pair.split("/", 1)
        df = pd.DataFrame({
            "date": sub.index,
            "asset": target_pair,
            "source": SOURCE_DERIVED,
            "base": base,
            "quote": quote,
            "close": ratio.values,
            "is_derived": True,
            "source_pair": label,
            "data_quality_status": DATA_QUALITY_OK,
            "notes": (f"derived cross: {label}; both legs ECB-sourced; "
                      f"holiday gaps inherit from underlying legs"),
        })
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames \
            else _empty_long_frame()
    return _ensure_columns(out), warnings


# ---------------------------------------------------------------------------
# LBMA gold
# ---------------------------------------------------------------------------
def _parse_lbma_payload(payload: Any) -> Tuple[pd.DataFrame, List[str]]:
    """Parse an LBMA `gold_pm.json` payload. Each entry has `d`
    (ISO date) and `v` (list of fixings: USD, GBP, EUR). We take
    `v[0]` as the USD fix. Fail-soft: skip malformed entries."""
    warnings: List[str] = []
    if not isinstance(payload, list) or not payload:
        warnings.append("LBMA: empty / non-list payload")
        return _empty_long_frame(), warnings
    rows: List[Dict[str, Any]] = []
    skipped = 0
    for entry in payload:
        if not isinstance(entry, dict):
            skipped += 1
            continue
        date_str = entry.get("d")
        values = entry.get("v")
        if not date_str or not isinstance(values, (list, tuple)) or not values:
            skipped += 1
            continue
        try:
            d = datetime.fromisoformat(str(date_str)).replace(
                tzinfo=timezone.utc)
        except (ValueError, TypeError):
            skipped += 1
            continue
        try:
            close = float(values[0])
        except (ValueError, TypeError):
            skipped += 1
            continue
        if close <= 0 or not math.isfinite(close):
            skipped += 1
            continue
        rows.append({
            "date": pd.Timestamp(d).normalize(),
            "asset": "XAU/USD",
            "source": SOURCE_LBMA,
            "base": "XAU",
            "quote": "USD",
            "close": close,
            "is_derived": False,
            "source_pair": "",
            "data_quality_status": DATA_QUALITY_OK,
            "notes": ("LBMA London PM gold fix (USD); daily, free, "
                      "since 1968; weekend/holiday gaps not filled"),
        })
    if skipped:
        warnings.append(f"XAU/USD: skipped {skipped} malformed LBMA entries")
    if not rows:
        warnings.append("XAU/USD: LBMA returned 0 usable rows")
        return _empty_long_frame(), warnings
    df = pd.DataFrame(rows).sort_values("date").drop_duplicates(
        subset=["date", "asset"], keep="last"
    ).reset_index(drop=True)
    return df, warnings


def fetch_lbma_gold(http_get: HttpGet = _http_get) -> Tuple[
    pd.DataFrame, List[str]
]:
    """Fetch the LBMA gold PM fix series. Fail-soft: returns a
    `missing` placeholder row plus a warning if the endpoint cannot
    be reached or the payload is unparseable."""
    url = "https://prices.lbma.org.uk/json/gold_pm.json"
    r = http_get(url)
    if not r.ok:
        return _missing_placeholder(
            "XAU/USD", SOURCE_LBMA,
            f"LBMA fetch failed: {r.error or 'no payload'}",
        ), [f"XAU/USD: LBMA fetch failed — {r.error or 'no payload'}"]
    df, warns = _parse_lbma_payload(r.payload)
    if df.empty:
        return _missing_placeholder(
            "XAU/USD", SOURCE_LBMA,
            "LBMA payload empty / unparseable",
        ), warns
    return _ensure_columns(df), warns


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------
def compute_fx_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add `return_1d` and `log_return_1d` columns, computed per
    asset on date-sorted closes. The first observation per asset is
    NaN (no prior close). Missing-row placeholders are passed
    through untouched."""
    if df is None or df.empty:
        return _empty_long_frame()
    out = _ensure_columns(df).copy()
    ok_mask = (out["data_quality_status"] == DATA_QUALITY_OK) \
                & out["close"].notna() & out["date"].notna()
    if not ok_mask.any():
        return out
    ok = out.loc[ok_mask].sort_values(["asset", "date"]).copy()
    grouped = ok.groupby("asset", sort=False)
    prev_close = grouped["close"].shift(1)
    ok["return_1d"] = ok["close"] / prev_close - 1.0
    ok["log_return_1d"] = np.log(ok["close"]) - np.log(prev_close)
    out.loc[ok.index, "return_1d"] = ok["return_1d"].values
    out.loc[ok.index, "log_return_1d"] = ok["log_return_1d"].values
    return out


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
class FxDatasetSchemaError(ValueError):
    """Raised by validate_fx_dataset_schema on a structural failure."""


def validate_fx_dataset_schema(df: pd.DataFrame) -> None:
    """Check the locked schema. Raises FxDatasetSchemaError on any
    structural problem; this is meant to be called BEFORE writing
    the parquet, so a bad build never lands on disk."""
    if df is None:
        raise FxDatasetSchemaError("dataset is None")
    if not isinstance(df, pd.DataFrame):
        raise FxDatasetSchemaError(
            f"expected pandas DataFrame, got {type(df).__name__}")
    missing = [c for c in DATASET_COLUMNS if c not in df.columns]
    if missing:
        raise FxDatasetSchemaError(f"missing required columns: {missing}")
    extra = [c for c in df.columns if c not in DATASET_COLUMNS]
    if extra:
        raise FxDatasetSchemaError(f"unexpected extra columns: {extra}")
    if list(df.columns) != DATASET_COLUMNS:
        raise FxDatasetSchemaError(
            "column order does not match locked schema "
            f"(got {list(df.columns)})"
        )
    bad_status = sorted(set(df["data_quality_status"].dropna().unique())
                          - set(DATA_QUALITY_STATUSES))
    if bad_status:
        raise FxDatasetSchemaError(
            f"unexpected data_quality_status values: {bad_status}"
        )
    if not df["is_derived"].dropna().map(
        lambda x: isinstance(x, (bool, np.bool_))
    ).all():
        raise FxDatasetSchemaError("is_derived column must be bool")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_fx_daily_dataset(
    ecb_http_get: HttpGet = _http_get,
    lbma_http_get: HttpGet = _http_get,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build the v1 dataset end-to-end. Fail-soft on every source —
    a network failure produces a `missing` placeholder row, never a
    crash. Returns (df_long, warnings)."""
    utils.assert_paper_only()
    warnings: List[str] = []
    direct_df, ecb_warns = fetch_ecb_rates(http_get=ecb_http_get)
    warnings.extend(ecb_warns)
    derived_df, derive_warns = derive_fx_pairs(direct_df)
    warnings.extend(derive_warns)
    gold_df, lbma_warns = fetch_lbma_gold(http_get=lbma_http_get)
    warnings.extend(lbma_warns)
    combined = pd.concat([direct_df, derived_df, gold_df], ignore_index=True)
    combined = _ensure_columns(combined)
    combined = combined.sort_values(
        ["asset", "date"], na_position="last"
    ).reset_index(drop=True)
    combined = compute_fx_returns(combined)
    validate_fx_dataset_schema(combined)
    return combined, warnings


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summarize_fx_dataset(df: pd.DataFrame,
                            warnings: Optional[Sequence[str]] = None,
                            ) -> Dict[str, Any]:
    """Return summary statistics + per-asset coverage. Safe to call
    on an empty / partial frame."""
    warnings = list(warnings or [])
    expected_assets = sorted({a for a, _ in ECB_DIRECT_PAIRS}
                                | {p[0] for p in DERIVED_PAIRS}
                                | {"XAU/USD"})
    if df is None or df.empty:
        return {
            "row_count": 0,
            "asset_count": 0,
            "start_date": None,
            "end_date": None,
            "coverage_days": 0,
            "missing_value_count": 0,
            "assets_available": [],
            "assets_missing": expected_assets,
            "data_quality_warnings": warnings,
            "coverage_by_asset": {},
        }

    ok = df[df["data_quality_status"] == DATA_QUALITY_OK]
    have_close = ok["close"].notna() & ok["date"].notna()
    ok_priced = ok[have_close]
    assets_available = sorted(ok_priced["asset"].unique().tolist())
    assets_missing = [a for a in expected_assets
                        if a not in set(assets_available)]
    if not ok_priced.empty:
        start_date = pd.Timestamp(ok_priced["date"].min()).date().isoformat()
        end_date = pd.Timestamp(ok_priced["date"].max()).date().isoformat()
        coverage_days = int(
            (ok_priced["date"].max() - ok_priced["date"].min()).days
        )
    else:
        start_date = None
        end_date = None
        coverage_days = 0
    coverage_by_asset: Dict[str, Dict[str, Any]] = {}
    for asset, sub in ok_priced.groupby("asset", sort=True):
        coverage_by_asset[str(asset)] = {
            "rows": int(len(sub)),
            "start_date": pd.Timestamp(sub["date"].min()).date().isoformat(),
            "end_date": pd.Timestamp(sub["date"].max()).date().isoformat(),
            "coverage_days": int(
                (sub["date"].max() - sub["date"].min()).days
            ),
            "is_derived": bool(sub["is_derived"].iloc[0]),
            "source": str(sub["source"].iloc[0]),
        }
    missing_value_count = int(df["close"].isna().sum())
    return {
        "row_count": int(len(df)),
        "asset_count": int(df["asset"].nunique()),
        "start_date": start_date,
        "end_date": end_date,
        "coverage_days": coverage_days,
        "missing_value_count": missing_value_count,
        "assets_available": assets_available,
        "assets_missing": assets_missing,
        "data_quality_warnings": warnings,
        "coverage_by_asset": coverage_by_asset,
    }


# ---------------------------------------------------------------------------
# Writer — restricted to data/fx/
# ---------------------------------------------------------------------------
class FxDatasetWritePathError(ValueError):
    """Raised if a caller tries to write the dataset outside data/fx/."""


def _assert_inside_fx_dir(path: Path) -> None:
    """Refuse to write anywhere except `data/fx/`."""
    resolved = path.resolve()
    fx_root = FX_DATA_DIR.resolve()
    try:
        resolved.relative_to(fx_root)
    except ValueError as exc:
        raise FxDatasetWritePathError(
            f"refusing to write outside data/fx/: {resolved}"
        ) from exc


def write_fx_dataset(df: pd.DataFrame,
                        parquet_path: Path = DATASET_PARQUET,
                        csv_path: Optional[Path] = DATASET_CSV,
                        ) -> Dict[str, str]:
    """Write the dataset to `data/fx/`. The companion CSV is
    optional (pass `csv_path=None` to skip). Both targets must be
    inside `data/fx/`; any other path is rejected. Validates schema
    before writing — a bad frame never lands on disk."""
    utils.assert_paper_only()
    if df is None:
        raise FxDatasetSchemaError("cannot write None dataset")
    validate_fx_dataset_schema(df)
    parquet_path = Path(parquet_path)
    _assert_inside_fx_dir(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    try:
        df.to_parquet(parquet_path, index=False)
        written["parquet"] = str(parquet_path)
    except Exception as exc:  # noqa: BLE001
        # Parquet engines (pyarrow/fastparquet) are optional in some
        # environments. Fall back to CSV-only without crashing the
        # build; the caller still receives an audited dataset.
        logger.warning("parquet write failed (%s); falling back to CSV",
                          type(exc).__name__)
        written["parquet_error"] = f"{type(exc).__name__}: {exc}"
    if csv_path is not None:
        csv_path = Path(csv_path)
        _assert_inside_fx_dir(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        written["csv"] = str(csv_path)
    return written


# ---------------------------------------------------------------------------
# Convenience: full build + write
# ---------------------------------------------------------------------------
def build_and_write(parquet_path: Path = DATASET_PARQUET,
                       csv_path: Optional[Path] = DATASET_CSV,
                       ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """End-to-end: fetch → derive → returns → validate → write.
    Always returns a (df, summary) tuple even on partial success
    (missing sources surface as `missing` rows + warnings)."""
    df, warnings = build_fx_daily_dataset()
    summary = summarize_fx_dataset(df, warnings=warnings)
    written = write_fx_dataset(df, parquet_path=parquet_path,
                                  csv_path=csv_path)
    summary["written"] = written
    return df, summary
