"""
Bot status object — a clean, documented snapshot of "what is the
engine doing right now and is it allowed to do anything stateful?"

Reads (read-only):
    * `safety_lock` — execution / paper / Kraken gates.
    * `strategy_registry` — current verdict snapshot per family.
    * `results/*` — file presence (NOT contents).
    * `data/raw/*_1d.csv` — file mtime to detect stale data.

Writes:
    results/bot_status.json
    results/bot_status.csv

Defaults (always, on this branch):
    bot_mode                    = "research_only"
    execution_enabled           = False
    paper_trading_enabled       = False
    kraken_connected            = False
    api_keys_loaded             = False
    safety_lock_status          = "locked"
    reason_execution_blocked    = "no strategy has passed"
    next_allowed_step           = "continue research or run dry-run analysis only"

These defaults are computed, not hard-coded — they fall through from
`safety_lock` and `strategy_registry`. If a future PASS strategy lands
AND the lock releases AND a real execution module exists, the values
update — but every cross-check has to clear, not just one.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import config, safety_lock, strategy_registry, utils


STATUS_COLUMNS: List[str] = [
    "bot_mode", "execution_enabled", "paper_trading_enabled",
    "kraken_connected", "api_keys_loaded",
    "active_strategy", "active_strategy_verdict",
    "latest_research_report", "latest_scorecard_status",
    "latest_data_update", "stale_data_warning",
    "generated_results_available",
    "safety_lock_status", "reason_execution_blocked", "next_allowed_step",
]

DEFAULT_NEXT_STEP_RESEARCH = (
    "continue research or run dry-run analysis only"
)


# ---------------------------------------------------------------------------
# Helpers — pure / read-only
# ---------------------------------------------------------------------------
def _api_keys_loaded() -> bool:
    """Return True only if explicit broker-key env vars are set. The
    engine does not READ the values — it only reports presence so the
    Bot Control Center can warn the user if they accidentally set one
    locally. The current branch must always show `False`."""
    candidates = (
        "KRAKEN_API_KEY", "KRAKEN_API_SECRET", "KRAKEN_KEY",
        "BINANCE_API_KEY", "BINANCE_API_SECRET",
    )
    for name in candidates:
        v = os.environ.get(name)
        if v is not None and len(v) > 0:
            return True
    return False


def _kraken_connected() -> bool:
    """We never integrate Kraken broker code. The check is conservative
    and fail-closed: if no Kraken broker module is importable, return
    False. This module never opens a network connection."""
    return safety_lock._kraken_blocked() is None  # noqa: SLF001


def _generated_results_available() -> bool:
    p = config.RESULTS_DIR
    if not p.exists():
        return False
    for f in p.glob("*.csv"):
        try:
            if f.stat().st_size > 0:
                return True
        except OSError:
            continue
    return False


def _latest_data_update_iso() -> Optional[str]:
    p = config.DATA_RAW_DIR
    if not p.exists():
        return None
    candidates = list(p.glob("*_1d.csv"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda f: f.stat().st_mtime)
    return pd.to_datetime(latest.stat().st_mtime, unit="s",
                            utc=True).isoformat()


def _stale_data_warning(stale_after_days: int = 7) -> bool:
    iso = _latest_data_update_iso()
    if iso is None:
        return True   # no data == stale
    last = pd.to_datetime(iso)
    age_days = (pd.Timestamp.utcnow() - last).total_seconds() / 86400.0
    return age_days > stale_after_days


def _active_strategy() -> Dict[str, Any]:
    """Pick the highest-scoring (most recent / best-verdict) strategy
    from the registry. With no PASS, "active strategy" is `None`."""
    reg = strategy_registry.build_registry()
    pass_rows = [r for r in reg if r["verdict"] == "PASS"]
    if pass_rows:
        r = pass_rows[0]
        return {"name": r["strategy_family"], "verdict": "PASS"}
    # Otherwise none is "active" for trading purposes.
    return {"name": None, "verdict": "no_pass_strategy"}


def _latest_research_report() -> Optional[str]:
    p = config.REPO_ROOT / "reports"
    if not p.exists():
        return None
    candidates = list(p.glob("*.md"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda f: f.stat().st_mtime)
    # Return path relative to repo root for readability.
    try:
        return str(latest.relative_to(config.REPO_ROOT))
    except ValueError:
        return str(latest)


def _latest_scorecard_status() -> str:
    p = config.RESULTS_DIR
    if not p.exists():
        return "results_dir_missing"
    has_any = False
    has_pass = False
    for f in p.glob("*scorecard*.csv"):
        has_any = True
        try:
            df = pd.read_csv(f)
        except Exception:  # noqa: BLE001
            continue
        if df.empty or "verdict" not in df.columns:
            continue
        if (df["verdict"].astype(str).str.upper() == "PASS").any():
            has_pass = True
            break
    if has_pass:
        return "pass_present"
    return "no_pass_among_scorecards" if has_any else "no_scorecards_present"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BotStatus:
    bot_mode: str
    execution_enabled: bool
    paper_trading_enabled: bool
    kraken_connected: bool
    api_keys_loaded: bool
    active_strategy: Optional[str]
    active_strategy_verdict: str
    latest_research_report: Optional[str]
    latest_scorecard_status: str
    latest_data_update: Optional[str]
    stale_data_warning: bool
    generated_results_available: bool
    safety_lock_status: str
    reason_execution_blocked: str
    next_allowed_step: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_status() -> BotStatus:
    sl = safety_lock.status()
    active = _active_strategy()
    # Surface ALL active block reasons so the user sees every gate at
    # once — not just the first. Defence in depth: also append a
    # "no strategy has passed" line if no PASS scorecard exists, even
    # if the safety lock has already mentioned it.
    reasons = list(sl.reasons_blocked)
    if active["verdict"] != "PASS" and not any(
        "no strategy" in r.lower() for r in reasons
    ):
        reasons.append("no strategy has passed")
    reason = "; ".join(reasons)
    next_step = (
        "continue research or run dry-run analysis only"
        if not sl.execution_allowed else
        "review PASS strategy carefully before any trading consideration"
    )
    return BotStatus(
        bot_mode="research_only" if not sl.execution_allowed else "trading_pending_review",
        execution_enabled=sl.execution_allowed,
        paper_trading_enabled=sl.paper_trading_allowed,
        kraken_connected=_kraken_connected(),
        api_keys_loaded=_api_keys_loaded(),
        active_strategy=active["name"],
        active_strategy_verdict=active["verdict"],
        latest_research_report=_latest_research_report(),
        latest_scorecard_status=_latest_scorecard_status(),
        latest_data_update=_latest_data_update_iso(),
        stale_data_warning=_stale_data_warning(),
        generated_results_available=_generated_results_available(),
        safety_lock_status=sl.safety_lock_status,
        reason_execution_blocked=reason,
        next_allowed_step=next_step,
    )


def write_status(save: bool = True) -> pd.DataFrame:
    s = compute_status().to_dict()
    df = pd.DataFrame([s], columns=STATUS_COLUMNS)
    if save:
        utils.write_df(df, config.RESULTS_DIR / "bot_status.csv")
        (config.RESULTS_DIR / "bot_status.json").parent.mkdir(
            parents=True, exist_ok=True,
        )
        (config.RESULTS_DIR / "bot_status.json").write_text(
            json.dumps(s, indent=2, default=str),
        )
    return df
