"""
Research dashboard helpers.

Pure, read-only support module for the Streamlit Research Dashboard.
This module exists so the dashboard's data layer is testable and so
the project has a single source of truth for archived branch metadata.

Hard rules:
    * No network calls.
    * No broker imports.
    * No API key reads.
    * No execution / order placement.
    * No file writes — every helper here is a READER. The Streamlit
      app is the only thing that renders.
    * Every reader fails softly: if a results file is missing, return
      a typed default (empty DataFrame, None, or a safe dict).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from . import config, safety_lock, strategy_registry


# ---------------------------------------------------------------------------
# Archived branch metadata — single source of truth.
#
# Every entry is a research branch that has been ARCHIVED (not merged into
# main). The `merge_allowed` flag is locked False for every entry; a test
# asserts this so a future edit cannot quietly enable a merge path.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArchivedBranch:
    branch: str
    kind: str            # "strategy" | "data_audit" | "infrastructure"
    verdict: str          # "FAIL" | "INCONCLUSIVE" | "NO_GO" | "archived"
    one_line_reason: str
    report_path: Optional[str] = None
    merge_allowed: bool = False
    # Optional commit + tag pointers for archived research that we
    # never want to merge but want to be able to revisit. Both fields
    # default to None so existing entries keep their original schema.
    archive_commit: Optional[str] = None
    archive_tag: Optional[str] = None
    # Per-row trading flags. Both default False; a unit test asserts
    # that no row in the ledger flips either to True, regardless of
    # verdict.
    paper_trading_allowed: bool = False
    live_trading_allowed: bool = False


KIND_STRATEGY = "strategy"
KIND_DATA_AUDIT = "data_audit"
KIND_INFRASTRUCTURE = "infrastructure"

ARCHIVED_KINDS: Tuple[str, ...] = (
    KIND_STRATEGY, KIND_DATA_AUDIT, KIND_INFRASTRUCTURE,
)
ARCHIVED_VERDICTS: Tuple[str, ...] = (
    "FAIL", "INCONCLUSIVE", "NO_GO", "archived",
)


ARCHIVED_BRANCHES: Tuple[ArchivedBranch, ...] = (
    ArchivedBranch(
        branch="research/fail-1-funding-derivatives",
        kind=KIND_STRATEGY,
        verdict="INCONCLUSIVE",
        one_line_reason=(
            "Funding-only rotation; funding+OI joint inconclusive "
            "because public OI history is capped to ~30 days."
        ),
        report_path="reports/funding_research_report.md",
    ),
    ArchivedBranch(
        branch="research/strategy-2-market-structure",
        kind=KIND_STRATEGY,
        verdict="FAIL",
        one_line_reason=(
            "DefiLlama TVL + on-chain + alt-breadth allocator "
            "(vol-target variant): tighter drawdown but failed BTC "
            "and stability gates."
        ),
        report_path="reports/market_structure_vol_target_report.md",
    ),
    ArchivedBranch(
        branch="research/strategy-3-sentiment-fear-greed",
        kind=KIND_STRATEGY,
        verdict="FAIL",
        one_line_reason=(
            "Fear & Greed overlay on the prior-best vol-target "
            "allocator; failed scorecard."
        ),
        report_path="reports/sentiment_research_report.md",
    ),
    ArchivedBranch(
        branch="research/strategy-4-drawdown-targeted-btc",
        kind=KIND_STRATEGY,
        verdict="FAIL",
        one_line_reason=(
            "BTC drawdown buckets + 200d MA cap; risk control, not "
            "alpha — loses to BTC in 8/14 OOS windows."
        ),
        report_path="reports/drawdown_targeted_btc_report.md",
    ),
    ArchivedBranch(
        branch="research/strategy-5-paid-positioning-data-audit",
        kind=KIND_DATA_AUDIT,
        verdict="archived",
        one_line_reason=(
            "Audited public OI / liquidations / long-short stack — "
            "all FAIL on free public endpoints (≤ 30–500 days)."
        ),
        report_path="reports/positioning_data_audit_report.md",
    ),
    ArchivedBranch(
        branch="research/strategy-6-funding-basis-carry",
        kind=KIND_STRATEGY,
        verdict="FAIL",
        one_line_reason=(
            "Funding + futures basis carry on BTC/ETH; beats placebo, "
            "tighter drawdown, but loses to BTC in 11/14 OOS windows."
        ),
        report_path="reports/funding_basis_carry_report.md",
    ),
    ArchivedBranch(
        branch="research/strategy-7-relative-value-btc-eth",
        kind=KIND_STRATEGY,
        verdict="FAIL",
        one_line_reason=(
            "BTC/ETH long-only relative-value rotation; loses to BTC "
            "in 10/14 windows and basket in 10/14 — no genuine "
            "dispersion alpha without a short leg."
        ),
        report_path="reports/relative_value_btc_eth_report.md",
    ),
    ArchivedBranch(
        branch="research/strategy-8-paid-data-decision-audit",
        kind=KIND_DATA_AUDIT,
        verdict="archived",
        one_line_reason=(
            "Vendor desk audit of CoinGlass / CryptoQuant / Velo / "
            "Glassnode / Kaiko / others; named CoinGlass the cheapest "
            "verifiable first paid trial."
        ),
        report_path="reports/paid_data_vendor_decision_report.md",
    ),
    ArchivedBranch(
        branch="research/strategy-9-free-open-data-reaudit",
        kind=KIND_DATA_AUDIT,
        verdict="NO_GO",
        one_line_reason=(
            "Free public stack has multi-year depth on 5 useful field "
            "classes — but every one was already used in a failed "
            "branch. No new untested signal class on free data."
        ),
        report_path="reports/free_open_data_reaudit_report.md",
    ),
    ArchivedBranch(
        branch="research/portfolio-rebalancing-strategy-v1",
        kind=KIND_STRATEGY,
        verdict="FAIL",
        one_line_reason=(
            "Sharpe was within 0.10 of BTC and the rebalance count "
            "passed, but max drawdown improvement was only 10.78 pp "
            "vs the required 15 pp, and the strategy lost to the "
            "placebo median return."
        ),
        # Report and strategy code live ONLY on the archived branch —
        # they are never merged into main. The dashboard's
        # `report_text()` falls back gracefully when this path is
        # absent on the current checkout.
        report_path=(
            "reports/portfolio_rebalancing_report.md "
            "(on archived branch — checkout "
            "research/portfolio-rebalancing-strategy-v1 to read inline)"
        ),
        archive_commit="19b1268",
        archive_tag="fail-portfolio-rebalancing-strategy-v1",
    ),
)


# ---------------------------------------------------------------------------
# Executive state — pure read-through over safety_lock.
# ---------------------------------------------------------------------------
PROJECT_MODE = "research-only"
ACTIVE_STRATEGY = "none"
PRODUCTION_BASELINE = "BTC buy-and-hold"
NEXT_ALLOWED_ACTION = "research / risk reporting only"


def executive_state() -> Dict[str, Any]:
    """Return a flat dict the dashboard can render directly. All values
    are derived live from the safety lock + project constants."""
    return {
        "project_mode": PROJECT_MODE,
        "active_strategy": ACTIVE_STRATEGY,
        "production_baseline": PRODUCTION_BASELINE,
        "safety_lock_status": safety_lock.safety_lock_status(),
        "execution_allowed": bool(safety_lock.is_execution_allowed()),
        "kraken_connected": bool(safety_lock.is_kraken_connection_allowed()),
        "paper_trading": bool(safety_lock.is_paper_trading_allowed()),
        "live_trading": bool(safety_lock.is_execution_allowed()),
        "next_allowed_action": NEXT_ALLOWED_ACTION,
        "reasons_blocked": list(safety_lock.reasons_blocked()),
    }


# ---------------------------------------------------------------------------
# Strategy verdict board — wraps `strategy_registry.build_registry()`.
# ---------------------------------------------------------------------------
def strategy_verdict_board() -> pd.DataFrame:
    """Return a typed DataFrame for the verdict board. Every row carries
    paper/live=False — the dashboard renders FAIL / INCONCLUSIVE rows
    visually red and the column ordering is locked."""
    try:
        rows = strategy_registry.build_registry()
    except Exception:  # noqa: BLE001 — fail soft for the dashboard
        rows = []
    df = pd.DataFrame(rows, columns=strategy_registry.REGISTRY_COLUMNS)
    return df


def all_strategy_flags_off(df: pd.DataFrame) -> bool:
    """Defensive predicate. The dashboard surfaces a hard-red banner if
    any registry row has paper/live trading allowed."""
    if df.empty:
        return True
    paper_col = df.get("paper_trading_allowed",
                          pd.Series([False] * len(df)))
    live_col = df.get("live_trading_allowed",
                         pd.Series([False] * len(df)))
    paper_off = (~paper_col.astype(bool)).all()
    live_off = (~live_col.astype(bool)).all()
    return bool(paper_off and live_off)


# ---------------------------------------------------------------------------
# Archived timeline — pure metadata; renders identically every load.
# ---------------------------------------------------------------------------
def archived_timeline_dataframe() -> pd.DataFrame:
    """Render the archived branch metadata as a DataFrame for the
    dashboard table. Column order is locked."""
    rows = [{
        "branch": b.branch,
        "kind": b.kind,
        "verdict": b.verdict,
        "one_line_reason": b.one_line_reason,
        "report_path": b.report_path or "",
        "merge_allowed": bool(b.merge_allowed),
        "paper_trading_allowed": bool(b.paper_trading_allowed),
        "live_trading_allowed": bool(b.live_trading_allowed),
        "archive_commit": b.archive_commit or "",
        "archive_tag": b.archive_tag or "",
    } for b in ARCHIVED_BRANCHES]
    cols = ["branch", "kind", "verdict", "one_line_reason",
             "report_path", "merge_allowed",
             "paper_trading_allowed", "live_trading_allowed",
             "archive_commit", "archive_tag"]
    return pd.DataFrame(rows, columns=cols)


def report_text(branch_or_path: str) -> str:
    """Read a report markdown file; return a friendly placeholder if
    missing. Never raises."""
    if not branch_or_path:
        return "(no report path on this entry)"
    p = (config.REPO_ROOT / branch_or_path
          if not branch_or_path.startswith("/")
          else Path(branch_or_path))
    if not p.exists() or not p.is_file():
        return f"(report file not found: {branch_or_path})"
    try:
        return p.read_text()
    except OSError as exc:
        return f"(report file unreadable: {exc})"


# ---------------------------------------------------------------------------
# Risk dashboard placeholder — read existing equity curve + scorecards.
# All readers fail softly.
# ---------------------------------------------------------------------------
RISK_RESULT_FILES: Tuple[str, ...] = (
    "equity_curve.csv",
    "summary_metrics.csv",
    "per_asset_metrics.csv",
    "scorecard.csv",
    "portfolio_scorecard.csv",
    "regime_aware_portfolio_scorecard.csv",
)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def latest_results_state(
    results_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Inventory the `results/` directory: which expected files are
    present, which are missing, and (optionally) the freshness of the
    most-recent equity curve. Returns a flat dict."""
    base = results_dir or config.RESULTS_DIR
    present: List[str] = []
    missing: List[str] = []
    for fname in RISK_RESULT_FILES:
        if (base / fname).exists() and (base / fname).stat().st_size > 0:
            present.append(fname)
        else:
            missing.append(fname)

    eq_path = base / "equity_curve.csv"
    eq = _safe_read_csv(eq_path)
    summary = {
        "results_dir_exists": base.exists(),
        "files_present": present,
        "files_missing": missing,
        "equity_first_row": None,
        "equity_last_row": None,
        "equity_n_bars": 0,
        "equity_freshness_days": None,
    }
    if not eq.empty and "equity" in eq.columns:
        summary["equity_n_bars"] = int(len(eq))
        if "datetime" in eq.columns:
            try:
                first = pd.to_datetime(eq["datetime"].iloc[0],
                                          utc=True)
                last = pd.to_datetime(eq["datetime"].iloc[-1],
                                         utc=True)
                summary["equity_first_row"] = first.isoformat()
                summary["equity_last_row"] = last.isoformat()
                summary["equity_freshness_days"] = float(
                    (pd.Timestamp.utcnow().tz_convert("UTC") - last)
                    .total_seconds() / 86400.0
                )
            except Exception:  # noqa: BLE001
                pass
    return summary


def baseline_metrics(
    results_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compute simple BTC buy-and-hold style metrics from the existing
    equity curve. The metric set is deliberately small — the dashboard
    is honest about what's there. Returns NaN-style defaults if the
    equity file is missing."""
    base = results_dir or config.RESULTS_DIR
    eq = _safe_read_csv(base / "equity_curve.csv")
    if eq.empty or "equity" not in eq.columns:
        return {
            "available": False, "starting_capital": float("nan"),
            "final_value": float("nan"),
            "total_return_pct": float("nan"),
            "max_drawdown_pct": float("nan"),
            "vol_pct_annualised": float("nan"),
            "n_bars": 0,
        }
    eq_ser = eq["equity"].astype(float).reset_index(drop=True)
    if eq_ser.empty:
        return baseline_metrics()  # repeat the empty branch
    start = float(eq_ser.iloc[0])
    final = float(eq_ser.iloc[-1])
    if start <= 0:
        return baseline_metrics()
    rets = eq_ser.pct_change().dropna()
    running_max = eq_ser.cummax()
    dd_pct = float(((eq_ser / running_max) - 1.0).min() * 100.0) \
        if not eq_ser.empty else float("nan")
    vol_pct = (float(rets.std(ddof=1) * (365.0 ** 0.5) * 100.0)
                if len(rets) > 1 else float("nan"))
    return {
        "available": True,
        "starting_capital": start,
        "final_value": final,
        "total_return_pct": (final / start - 1.0) * 100.0,
        "max_drawdown_pct": dd_pct,
        "vol_pct_annualised": vol_pct,
        "n_bars": int(len(eq_ser)),
    }


# ---------------------------------------------------------------------------
# Next allowed actions — only safe, read-only research activities.
# ---------------------------------------------------------------------------
ALLOWED_NEXT_ACTIONS: Tuple[str, ...] = (
    "Run system health: `python main.py system_health`",
    "Refresh historical data: `python main.py download`",
    "Review archived reports under `reports/`",
    "Re-run archived audits via their CLI commands "
    "(read-only, no execution)",
    "Improve this dashboard or supporting documentation",
    "Run a paid-data audit ONLY if the user supplies an API key in "
    "their shell — never committed",
    "Open a long/short pair-research module ONLY with explicit "
    "spec approval (out of scope for the current research-only policy)",
)

FORBIDDEN_NEXT_ACTIONS: Tuple[str, ...] = (
    "paper trading",
    "live trading",
    "Kraken connection",
    "API key entry into the dashboard",
    "order placement",
    "any execution surface",
)


# ---------------------------------------------------------------------------
# Decision journal + alert history readers — fail-soft.
# ---------------------------------------------------------------------------
def decision_journal_latest_row(
    results_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Return the most recent decision_journal row, or None if absent."""
    base = results_dir or config.RESULTS_DIR
    df = _safe_read_csv(base / "decision_journal.csv")
    if df.empty:
        return None
    row = df.iloc[-1].to_dict()
    return {k: (v if not (isinstance(v, float) and pd.isna(v)) else None)
            for k, v in row.items()}


def alert_history_latest_rows(
    results_dir: Optional[Path] = None,
    limit: int = 10,
) -> pd.DataFrame:
    """Return the most-recent alert rows for the dashboard table."""
    base = results_dir or config.RESULTS_DIR
    df = _safe_read_csv(base / "alert_history.csv")
    if df.empty:
        return df
    return df.tail(int(max(1, limit))).reset_index(drop=True)
