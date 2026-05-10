"""
Portfolio risk dashboard helpers.

Read-only computation layer for the Streamlit Portfolio Risk section.
The user supplies a local `data/portfolio_holdings.csv` file; this
module loads it, validates the schema, computes simple position
metrics + drawdown scenarios, classifies portfolio risk, and emits a
non-trading recommendation.

Hard rules (locked):
    * No network calls.
    * No broker imports.
    * No API key reads.
    * No order placement strings (no buy / sell / submit_order, etc.).
    * No paper-trading / live-trading imports.
    * No file writes — every helper here is a READER. The dashboard
      is the only thing that renders, and even the dashboard never
      writes the user's portfolio.
    * The recommendation vocabulary is locked: only the phrases in
      `RECOMMENDATION_PHRASES` are emitted. Any future edit that
      introduces "buy", "sell", "trade", "open position", or
      "connect broker" will fail the unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Locked schema + asset taxonomy.
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "asset", "quantity", "average_cost", "currency",
    "current_price", "price_source", "notes",
)

# Asset symbols treated as stablecoins / cash. Drawdown shocks do NOT
# apply to these — they are assumed to hold value through a crypto
# crash. (USDT depegs etc. are out of scope for this module.)
STABLECOIN_TICKERS: Tuple[str, ...] = (
    "USDT", "USDC", "BUSD", "DAI", "FRAX", "TUSD", "GUSD", "PYUSD",
    "LUSD", "USDD",
)
CASH_TICKERS: Tuple[str, ...] = (
    "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "CASH",
)

# Default portfolio CSV path. Users supply this file locally; it is
# gitignored.
DEFAULT_PORTFOLIO_PATH: Path = config.REPO_ROOT / "data" / "portfolio_holdings.csv"


# ---------------------------------------------------------------------------
# Locked recommendation vocabulary. The CLI/dashboard MUST emit one
# of these phrases verbatim — no other recommendation language is
# allowed.
# ---------------------------------------------------------------------------
RECOMMENDATION_HOLD = "hold risk steady"
RECOMMENDATION_REVIEW = "review concentration"
RECOMMENDATION_REDUCE = "reduce concentration"
RECOMMENDATION_DATA_MISSING = "data missing"
RECOMMENDATION_DO_NOTHING = "do nothing until data is complete"

RECOMMENDATION_PHRASES: Tuple[str, ...] = (
    RECOMMENDATION_HOLD,
    RECOMMENDATION_REVIEW,
    RECOMMENDATION_REDUCE,
    RECOMMENDATION_DATA_MISSING,
    RECOMMENDATION_DO_NOTHING,
)

# Risk-class vocabulary.
RISK_LOW = "LOW"
RISK_MODERATE = "MODERATE"
RISK_HIGH = "HIGH"
RISK_EXTREME = "EXTREME"
RISK_UNKNOWN = "UNKNOWN"

RISK_CLASSES: Tuple[str, ...] = (
    RISK_LOW, RISK_MODERATE, RISK_HIGH, RISK_EXTREME, RISK_UNKNOWN,
)

# Drawdown scenario shocks. Values are decimals (negative).
SCENARIOS: Tuple[Tuple[str, float], ...] = (
    ("minus_10_percent", -0.10),
    ("minus_20_percent", -0.20),
    ("minus_30_percent", -0.30),
    ("minus_50_percent", -0.50),
)

# Risk-class thresholds (locked, never tuned).
THRESH_EXTREME_LARGEST = 0.80   # > 80 % single position
THRESH_HIGH_LARGEST = 0.60       # > 60 % single position
THRESH_HIGH_CRYPTO = 0.90        # > 90 % crypto exposure
THRESH_MODERATE_LARGEST = 0.40   # > 40 % single position


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SchemaStatus:
    ok: bool
    missing_columns: Tuple[str, ...]
    extra_columns: Tuple[str, ...]
    n_rows: int
    n_cols: int


# ---------------------------------------------------------------------------
# 1) load_portfolio_holdings
# ---------------------------------------------------------------------------
def load_portfolio_holdings(
    path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Safely load the user's portfolio CSV. If the file is absent or
    unreadable, return an empty DataFrame plus a warning string. Never
    raises."""
    p = Path(path) if path is not None else DEFAULT_PORTFOLIO_PATH
    if not p.exists():
        return (
            pd.DataFrame(columns=list(REQUIRED_COLUMNS)),
            f"No portfolio file at {p}. "
            "Add a local data/portfolio_holdings.csv using the "
            "template documented in docs/portfolio_risk_dashboard.md. "
            "This file is gitignored and must not be committed.",
        )
    try:
        df = pd.read_csv(p)
    except Exception as exc:  # noqa: BLE001 — fail-soft per spec
        return (
            pd.DataFrame(columns=list(REQUIRED_COLUMNS)),
            f"portfolio CSV at {p} unreadable: "
            f"{type(exc).__name__}",
        )
    return df, None


# ---------------------------------------------------------------------------
# 2) validate_portfolio_schema
# ---------------------------------------------------------------------------
def validate_portfolio_schema(df: pd.DataFrame) -> SchemaStatus:
    """Return a structured schema status. Never raises."""
    cols = list(df.columns) if df is not None else []
    missing = tuple(c for c in REQUIRED_COLUMNS if c not in cols)
    extra = tuple(c for c in cols if c not in REQUIRED_COLUMNS)
    return SchemaStatus(
        ok=(not missing),
        missing_columns=missing,
        extra_columns=extra,
        n_rows=int(len(df)) if df is not None else 0,
        n_cols=len(cols),
    )


# ---------------------------------------------------------------------------
# Internal asset-class helpers
# ---------------------------------------------------------------------------
def _is_stable(asset: str) -> bool:
    return str(asset).strip().upper() in STABLECOIN_TICKERS


def _is_cash(asset: str) -> bool:
    return str(asset).strip().upper() in CASH_TICKERS


def _is_crypto(asset: str) -> bool:
    a = str(asset).strip().upper()
    return bool(a) and (a not in STABLECOIN_TICKERS) and (a not in CASH_TICKERS)


# ---------------------------------------------------------------------------
# 3) calculate_position_values
# ---------------------------------------------------------------------------
def calculate_position_values(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-position columns. Returns a NEW DataFrame; the caller's
    frame is never mutated."""
    out = df.copy()
    for c in ("quantity", "average_cost", "current_price"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        else:
            out[c] = 0.0
    out["position_value"] = (out["quantity"] * out["current_price"]).astype(float)
    out["cost_basis"] = (out["quantity"] * out["average_cost"]).astype(float)
    out["unrealized_pnl"] = out["position_value"] - out["cost_basis"]
    safe_cb = out["cost_basis"].where(out["cost_basis"] != 0.0)
    out["unrealized_pnl_percent"] = (
        out["unrealized_pnl"] / safe_cb * 100.0
    ).fillna(0.0)
    total_value = float(out["position_value"].sum())
    if total_value > 0:
        out["portfolio_weight"] = out["position_value"] / total_value
    else:
        out["portfolio_weight"] = 0.0
    return out


# ---------------------------------------------------------------------------
# 4) calculate_portfolio_summary
# ---------------------------------------------------------------------------
def calculate_portfolio_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate-level metrics. Accepts either a raw schema-valid frame
    or a frame already augmented by `calculate_position_values`."""
    if df is None or df.empty:
        return {
            "total_market_value": 0.0,
            "total_cost_basis": 0.0,
            "total_unrealized_pnl": 0.0,
            "total_unrealized_pnl_percent": 0.0,
            "largest_position": None,
            "largest_position_weight": 0.0,
            "asset_count": 0,
            "cash_count": 0,
            "crypto_exposure": 0.0,
            "stablecoin_exposure": 0.0,
        }
    pos = (df if "position_value" in df.columns
              else calculate_position_values(df))
    total_mv = float(pos["position_value"].sum())
    total_cb = float(pos["cost_basis"].sum())
    total_pnl = total_mv - total_cb
    total_pnl_pct = (total_pnl / total_cb * 100.0) if total_cb > 0 else 0.0

    if total_mv > 0:
        idx = pos["position_value"].idxmax()
        largest_asset = str(pos.loc[idx, "asset"])
        largest_weight = float(pos.loc[idx, "portfolio_weight"])
    else:
        largest_asset = None
        largest_weight = 0.0

    crypto_mask = pos["asset"].apply(_is_crypto)
    stable_mask = pos["asset"].apply(_is_stable)
    cash_mask = pos["asset"].apply(_is_cash)
    crypto_value = float(pos.loc[crypto_mask, "position_value"].sum())
    stable_value = float(pos.loc[stable_mask, "position_value"].sum())

    return {
        "total_market_value": total_mv,
        "total_cost_basis": total_cb,
        "total_unrealized_pnl": total_pnl,
        "total_unrealized_pnl_percent": total_pnl_pct,
        "largest_position": largest_asset,
        "largest_position_weight": largest_weight,
        "asset_count": int(pos["asset"].nunique()),
        "cash_count": int(cash_mask.sum()),
        "crypto_exposure": (crypto_value / total_mv) if total_mv > 0 else 0.0,
        "stablecoin_exposure": (stable_value / total_mv) if total_mv > 0
                                  else 0.0,
    }


# ---------------------------------------------------------------------------
# 5) calculate_drawdown_scenarios
# ---------------------------------------------------------------------------
def calculate_drawdown_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute portfolio loss under each scenario shock. Stablecoins
    and cash are NOT shocked — only crypto positions are repriced."""
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "scenario", "shock_pct", "new_portfolio_value",
            "portfolio_loss", "portfolio_loss_percent",
            "largest_loss_asset",
        ])
    pos = (df if "position_value" in df.columns
              else calculate_position_values(df))
    base_mv = float(pos["position_value"].sum())
    crypto_mask = pos["asset"].apply(_is_crypto)
    crypto_value = float(pos.loc[crypto_mask, "position_value"].sum())
    rows: List[Dict[str, Any]] = []
    for label, shock in SCENARIOS:
        portfolio_loss = crypto_value * shock
        new_mv = base_mv + portfolio_loss
        if crypto_mask.any():
            crypto_only = pos.loc[crypto_mask, ["asset", "position_value"]
                                     ].copy()
            crypto_only["loss"] = crypto_only["position_value"] * shock
            largest_loss = str(
                crypto_only.loc[crypto_only["loss"].idxmin(), "asset"],
            )
        else:
            largest_loss = None
        rows.append({
            "scenario": label,
            "shock_pct": shock * 100.0,
            "new_portfolio_value": float(new_mv),
            "portfolio_loss": float(portfolio_loss),
            "portfolio_loss_percent": (
                float(portfolio_loss / base_mv * 100.0)
                if base_mv > 0 else 0.0
            ),
            "largest_loss_asset": largest_loss,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6) compare_to_btc_baseline
# ---------------------------------------------------------------------------
def compare_to_btc_baseline(df: pd.DataFrame) -> Dict[str, Any]:
    """If BTC is in the portfolio, surface its weight and the non-BTC
    weight; otherwise return a warning."""
    if df is None or df.empty:
        return {
            "btc_present": False, "btc_weight": 0.0,
            "non_btc_weight": 0.0,
            "concentration_vs_baseline": "n/a",
            "warning": "no holdings — baseline comparison unavailable",
        }
    pos = (df if "position_value" in df.columns
              else calculate_position_values(df))
    total_mv = float(pos["position_value"].sum())
    if total_mv <= 0:
        return {
            "btc_present": False, "btc_weight": 0.0,
            "non_btc_weight": 0.0,
            "concentration_vs_baseline": "n/a",
            "warning": "total market value is zero — "
                          "baseline comparison unavailable",
        }
    btc_rows = pos[pos["asset"].astype(str).str.upper() == "BTC"]
    if btc_rows.empty:
        return {
            "btc_present": False, "btc_weight": 0.0,
            "non_btc_weight": 1.0,
            "concentration_vs_baseline": "n/a",
            "warning": "BTC absent from holdings — baseline "
                          "comparison limited",
        }
    btc_value = float(btc_rows["position_value"].sum())
    btc_weight = btc_value / total_mv
    non_btc_weight = 1.0 - btc_weight
    if btc_weight >= 0.999:
        concentration = "matches BTC baseline"
    elif btc_weight >= 0.50:
        concentration = ("BTC majority but more diversified than "
                            "BTC baseline")
    else:
        concentration = ("BTC minority — portfolio less BTC-concentrated "
                            "than BTC baseline")
    return {
        "btc_present": True,
        "btc_weight": btc_weight,
        "non_btc_weight": non_btc_weight,
        "concentration_vs_baseline": concentration,
        "warning": None,
    }


# ---------------------------------------------------------------------------
# 7) classify_portfolio_risk
# ---------------------------------------------------------------------------
def classify_portfolio_risk(summary: Dict[str, Any]) -> str:
    """Locked decision rules — never tuned after seeing portfolio data.

    UNKNOWN  no holdings or zero market value
    EXTREME  any single position > 80 %
    HIGH     any single position > 60 %, OR crypto exposure > 90 %
    MODERATE any single position > 40 %
    LOW      otherwise
    """
    if not summary or summary.get("total_market_value", 0.0) <= 0:
        return RISK_UNKNOWN
    largest = float(summary.get("largest_position_weight", 0.0))
    crypto_exp = float(summary.get("crypto_exposure", 0.0))
    if largest > THRESH_EXTREME_LARGEST:
        return RISK_EXTREME
    if largest > THRESH_HIGH_LARGEST or crypto_exp > THRESH_HIGH_CRYPTO:
        return RISK_HIGH
    if largest > THRESH_MODERATE_LARGEST:
        return RISK_MODERATE
    return RISK_LOW


# ---------------------------------------------------------------------------
# 8) generate_risk_recommendation
# ---------------------------------------------------------------------------
def generate_risk_recommendation(
    summary: Dict[str, Any],
    scenarios: pd.DataFrame,
) -> Dict[str, str]:
    """Map the risk class to one of `RECOMMENDATION_PHRASES`. NEVER
    emits trade-action language. The phrasing is deliberately
    operational — `hold risk steady` is not "buy and hold the
    market" advice; it is "the dashboard does not see a reason for
    the user to change their concentration today"."""
    if not summary or summary.get("total_market_value", 0.0) <= 0:
        return {
            "category": RECOMMENDATION_DATA_MISSING,
            "action": RECOMMENDATION_DO_NOTHING,
        }
    risk = classify_portfolio_risk(summary)
    if risk == RISK_EXTREME:
        return {"category": RISK_EXTREME, "action": RECOMMENDATION_REDUCE}
    if risk == RISK_HIGH:
        return {"category": RISK_HIGH, "action": RECOMMENDATION_REVIEW}
    if risk in (RISK_MODERATE, RISK_LOW):
        return {"category": risk, "action": RECOMMENDATION_HOLD}
    return {
        "category": RECOMMENDATION_DATA_MISSING,
        "action": RECOMMENDATION_DO_NOTHING,
    }


# ---------------------------------------------------------------------------
# 9) get_portfolio_risk_dashboard_state
# ---------------------------------------------------------------------------
def get_portfolio_risk_dashboard_state(
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """One safe entry point. Always returns a dict the dashboard can
    render; never raises."""
    holdings, load_warning = load_portfolio_holdings(path)
    schema = validate_portfolio_schema(holdings)
    warnings: List[str] = []
    if load_warning:
        warnings.append(load_warning)

    if not schema.ok:
        if schema.n_cols > 0:
            warnings.append(
                "portfolio CSV is missing required columns: "
                f"{list(schema.missing_columns)}",
            )
        return {
            "holdings": holdings,
            "schema_status": schema,
            "summary": calculate_portfolio_summary(pd.DataFrame()),
            "scenarios": pd.DataFrame(),
            "btc_baseline": compare_to_btc_baseline(pd.DataFrame()),
            "risk_classification": RISK_UNKNOWN,
            "recommendation": {
                "category": RECOMMENDATION_DATA_MISSING,
                "action": RECOMMENDATION_DO_NOTHING,
            },
            "warnings": warnings,
        }

    if holdings.empty:
        return {
            "holdings": holdings,
            "schema_status": schema,
            "summary": calculate_portfolio_summary(pd.DataFrame()),
            "scenarios": pd.DataFrame(),
            "btc_baseline": compare_to_btc_baseline(pd.DataFrame()),
            "risk_classification": RISK_UNKNOWN,
            "recommendation": {
                "category": RECOMMENDATION_DATA_MISSING,
                "action": RECOMMENDATION_DO_NOTHING,
            },
            "warnings": warnings + ["portfolio CSV has no rows"],
        }

    pos = calculate_position_values(holdings)
    summary = calculate_portfolio_summary(pos)
    scenarios = calculate_drawdown_scenarios(pos)
    btc_baseline = compare_to_btc_baseline(pos)
    risk = classify_portfolio_risk(summary)
    rec = generate_risk_recommendation(summary, scenarios)
    if btc_baseline.get("warning"):
        warnings.append(btc_baseline["warning"])
    return {
        "holdings": pos,
        "schema_status": schema,
        "summary": summary,
        "scenarios": scenarios,
        "btc_baseline": btc_baseline,
        "risk_classification": risk,
        "recommendation": rec,
        "warnings": warnings,
    }
