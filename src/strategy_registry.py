"""
Strategy registry — every researched strategy family with its current
verdict and an explicit "trading allowed?" gate.

The registry is the single read-through summary the Bot Control Center
shows, the alert engine consumes, and the safety lock cross-checks.
Every entry **must** carry:

    paper_trading_allowed = False
    live_trading_allowed  = False

on this branch. The flag is wired through `safety_lock`; the registry
does not invent its own bypass. If you ever flip an entry's flag, the
unit test `test_no_strategy_is_paper_or_live_allowed` will fail and the
test suite will break.

Data sources (all read-only):

    * `results/*_scorecard.csv` — verdict + headline metrics if present.
    * Hard-coded family metadata (branch name, report path) for the
      handful of families researched on this project.
    * `git rev-parse HEAD` if available, else "(not_git_or_unknown)".

Outputs:

    results/strategy_registry_snapshot.csv
    results/strategy_registry_snapshot.json
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import config, safety_lock, utils


# ---------------------------------------------------------------------------
# Family metadata — hand-curated.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _FamilyMeta:
    family: str
    branch: str
    report_path: str
    scorecard_files: tuple
    headline: str       # short canonical description used in the registry


_FAMILIES: List[_FamilyMeta] = [
    _FamilyMeta(
        family="single_asset_ta",
        branch="main (v1-research-closure)",
        report_path="reports/final_crypto_research_report.md",
        scorecard_files=("scorecard.csv", "strategy_scorecard.csv"),
        headline=("Single-asset RSI/MA/ATR + breakout + trend-following + "
                   "pullback + mean-reversion families"),
    ),
    _FamilyMeta(
        family="regime_selector",
        branch="main (v1-research-closure)",
        report_path="reports/final_crypto_research_report.md",
        scorecard_files=("scorecard.csv",),
        headline="Regime-aware single-asset selector",
    ),
    _FamilyMeta(
        family="portfolio_momentum",
        branch="main (v1-research-closure)",
        report_path="reports/final_crypto_research_report.md",
        scorecard_files=("portfolio_scorecard.csv",),
        headline="Top-N weekly portfolio momentum + regime-aware variant",
    ),
    _FamilyMeta(
        family="derivatives_funding",
        branch="research/fail-1-funding-derivatives",
        report_path="reports/funding_research_report.md",
        scorecard_files=("funding_scorecard.csv",
                          "derivatives_scorecard.csv"),
        headline=("Funding-only rotation (4 yr funding) + funding+OI joint "
                   "(INCONCLUSIVE on data length)"),
    ),
    _FamilyMeta(
        family="market_structure",
        branch="research/strategy-2-market-structure",
        report_path="reports/market_structure_vol_target_report.md",
        scorecard_files=("market_structure_vol_target_scorecard.csv",
                          "market_structure_scorecard.csv"),
        headline=("DefiLlama TVL + stables + Blockchain.com on-chain + "
                   "alt breadth allocator (vol-target variant — closest result)"),
    ),
    _FamilyMeta(
        family="sentiment_overlay",
        branch="research/strategy-3-sentiment-fear-greed",
        report_path="reports/sentiment_research_report.md",
        scorecard_files=("sentiment_scorecard.csv",),
        headline=("Fear & Greed overlay on the prior-best vol-target "
                   "allocator"),
    ),
    _FamilyMeta(
        family="drawdown_targeted_btc",
        branch="research/strategy-4-drawdown-targeted-btc",
        report_path="reports/drawdown_targeted_btc_report.md",
        scorecard_files=("drawdown_targeted_btc_scorecard.csv",),
        headline=("Continuous BTC drawdown-targeted allocator + 200d MA "
                   "regime + optional alt overlay"),
    ),
]


# ---------------------------------------------------------------------------
# Registry row schema (locked here so tests can assert)
# ---------------------------------------------------------------------------
REGISTRY_COLUMNS: List[str] = [
    "strategy_family", "branch", "latest_commit_if_available",
    "verdict", "best_result_summary", "benchmark_result", "placebo_result",
    "scorecard_status", "paper_trading_allowed", "live_trading_allowed",
    "reason_blocked", "report_path",
]


# ---------------------------------------------------------------------------
# Helpers (pure / read-only)
# ---------------------------------------------------------------------------
def _git_head_short() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(config.REPO_ROOT),
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return "(not_git_or_unknown)"


def _read_scorecard(meta: _FamilyMeta) -> Optional[pd.DataFrame]:
    for fname in meta.scorecard_files:
        p = config.RESULTS_DIR / fname
        if p.exists() and p.stat().st_size > 0:
            try:
                df = pd.read_csv(p)
                if not df.empty:
                    return df
            except Exception:  # noqa: BLE001
                continue
    return None


def _summarise(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if df is None or df.empty:
        return {
            "verdict": "UNKNOWN",
            "scorecard_status": "missing",
            "best_result_summary": "no scorecard CSV in results/",
            "benchmark_result": "n/a",
            "placebo_result": "n/a",
        }
    row = df.iloc[0].to_dict()
    verdict = str(row.get("verdict", "UNKNOWN")).upper()
    parts = []
    for k in ("avg_oos_return_pct", "stability_score_pct",
              "pct_windows_beat_btc", "pct_windows_beat_basket",
              "pct_windows_beat_simple_momentum"):
        if k in row and pd.notna(row[k]):
            parts.append(f"{k}={float(row[k]):.2f}")
    bench = "; ".join(parts) if parts else "n/a"
    plac_parts = []
    for k in ("strategy_full_return_pct", "placebo_median_return_pct",
              "beats_placebo_median"):
        if k in row and pd.notna(row[k]):
            plac_parts.append(f"{k}={row[k]}")
    placebo = "; ".join(plac_parts) if plac_parts else "n/a"
    cks = (f"{int(row.get('checks_passed', 0))}"
            f"/{int(row.get('checks_total', 0))}"
            if "checks_passed" in row and pd.notna(row.get("checks_passed"))
            else "n/a")
    summary = (f"verdict={verdict}; checks_passed={cks}"
                + (f"; reason={row.get('reason')}" if "reason" in row else ""))
    return {
        "verdict": verdict,
        "scorecard_status": "present",
        "best_result_summary": summary,
        "benchmark_result": bench,
        "placebo_result": placebo,
    }


def _trading_allowed(verdict: str) -> Dict[str, Any]:
    """Both paper and live MUST be False on this branch. We check the
    safety lock on top so a (theoretical) future PASS verdict alone
    isn't enough — the lock has to release as well."""
    pass_verdict = (verdict == "PASS")
    paper = pass_verdict and safety_lock.is_paper_trading_allowed()
    live = pass_verdict and safety_lock.is_execution_allowed()
    if not pass_verdict:
        reason = "verdict is not PASS"
    elif not safety_lock.is_execution_allowed():
        reason = safety_lock.reason_blocked()
    else:
        reason = ""
    return {
        "paper_trading_allowed": bool(paper),
        "live_trading_allowed": bool(live),
        "reason_blocked": reason,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_registry() -> List[Dict[str, Any]]:
    head = _git_head_short()
    rows: List[Dict[str, Any]] = []
    for meta in _FAMILIES:
        df = _read_scorecard(meta)
        s = _summarise(df)
        gate = _trading_allowed(s["verdict"])
        rows.append({
            "strategy_family": meta.family,
            "branch": meta.branch,
            "latest_commit_if_available": head,
            "verdict": s["verdict"],
            "best_result_summary": s["best_result_summary"],
            "benchmark_result": s["benchmark_result"],
            "placebo_result": s["placebo_result"],
            "scorecard_status": s["scorecard_status"],
            "paper_trading_allowed": gate["paper_trading_allowed"],
            "live_trading_allowed": gate["live_trading_allowed"],
            "reason_blocked": gate["reason_blocked"],
            "report_path": meta.report_path,
        })
    return rows


def write_snapshot(save: bool = True) -> pd.DataFrame:
    rows = build_registry()
    out = pd.DataFrame(rows, columns=REGISTRY_COLUMNS)
    if save:
        utils.write_df(
            out, config.RESULTS_DIR / "strategy_registry_snapshot.csv",
        )
        json_path = config.RESULTS_DIR / "strategy_registry_snapshot.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(rows, indent=2))
    return out
