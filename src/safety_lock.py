"""
Single source of truth for "is execution allowed on this engine?".

Every other bot-shell module (`bot_status`, `strategy_registry`,
`alert_engine`, `dry_run_planner`, the Streamlit Bot Control Center)
asks this module — they do **not** decide independently.

Current behaviour (locked):
    is_execution_allowed()         -> False
    is_paper_trading_allowed()     -> False
    is_kraken_connection_allowed() -> False
    reason_blocked()               -> non-empty string

There is **no** environment variable, config flag, or bypass parameter
that unlocks any of these from the outside. Future unlocks must be
explicit code changes on top of:

    1. A scorecard with verdict == "PASS" being present in
       `results/`. Until then `is_execution_allowed()` returns False
       regardless of any config.
    2. Hard-coded `LIVE_TRADING_ENABLED = False` in `src/config.py`.
       This module additionally enforces it (defence in depth).

`assert_execution_blocked()` raises `ExecutionBlocked` if any caller
attempts to act on an unlocked path. This is the function broker /
order code must call before doing anything stateful — and since none
of those paths exist on this branch, it is currently only used by tests
and bot-shell code.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import config

REASON_NO_PASS_STRATEGY = "no strategy has passed the conservative scorecard"
REASON_LIVE_FLAG_FALSE = (
    "config.LIVE_TRADING_ENABLED is hard-coded False (v1 invariant)"
)
REASON_KRAKEN_NOT_INTEGRATED = "no Kraken execution module exists"
REASON_NO_PAPER_TRADER = "paper trading remains disabled by spec"


class ExecutionBlocked(RuntimeError):
    """Raised when caller attempts to take an action that the safety
    lock currently forbids. The message lists every reason the path is
    blocked so logs are explicit, not silent."""


# ---------------------------------------------------------------------------
# Status object
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SafetyStatus:
    execution_allowed: bool
    paper_trading_allowed: bool
    kraken_connection_allowed: bool
    safety_lock_status: str
    reasons_blocked: List[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Stable serialisation for JSON / CSV.
        d["reasons_blocked"] = list(self.reasons_blocked)
        return d


# ---------------------------------------------------------------------------
# Internal checks (each one is a pure boolean function — no side effects)
# ---------------------------------------------------------------------------
def _live_flag_blocked() -> Optional[str]:
    """The v1 hard-coded `LIVE_TRADING_ENABLED = False` invariant. We
    READ the flag — if a future maintainer flips it, this still
    returns a block until a real execution module is wired AND a PASS
    strategy exists. Defence in depth."""
    if getattr(config, "LIVE_TRADING_ENABLED", False):
        return None
    return REASON_LIVE_FLAG_FALSE


def _no_pass_strategy_blocked() -> Optional[str]:
    """Walk every `*_scorecard.csv` under results/ and look for a row
    whose `verdict` column is `PASS`. If none exist, execution is
    blocked. Missing files == blocked (the lock is fail-closed)."""
    results = config.RESULTS_DIR
    if not results.exists():
        return REASON_NO_PASS_STRATEGY
    found_pass = False
    for p in results.glob("*scorecard*.csv"):
        try:
            df = pd.read_csv(p)
        except Exception:  # noqa: BLE001
            continue
        if df.empty or "verdict" not in df.columns:
            continue
        if (df["verdict"].astype(str).str.upper() == "PASS").any():
            found_pass = True
            break
    return None if found_pass else REASON_NO_PASS_STRATEGY


def _kraken_blocked() -> Optional[str]:
    """We never integrated Kraken on this engine. The check verifies
    that no `kraken_*` execution module is importable; if one is ever
    added, it must come with explicit unlocking code on top of every
    other check, not as a passive side effect."""
    repo_src = Path(__file__).resolve().parent
    for candidate in ("kraken_execution.py", "kraken_broker.py",
                       "kraken_client.py", "execution.py"):
        if (repo_src / candidate).exists():
            return None
    return REASON_KRAKEN_NOT_INTEGRATED


def _paper_trader_blocked() -> Optional[str]:
    """Paper trading exists as a per-tick simulator (`paper_trader.py`)
    but the spec for this branch is `paper_trading_enabled = False`.
    The lock states that explicitly so callers don't accidentally
    re-enable it. Note: `paper_trader.py` itself only writes JSON state;
    no order leaves the process."""
    return REASON_NO_PAPER_TRADER


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def reasons_blocked() -> List[str]:
    """Return every active blocking reason. Empty list = nothing
    blocked (which is impossible on this branch by construction)."""
    out: List[str] = []
    for fn in (_live_flag_blocked, _no_pass_strategy_blocked,
                _kraken_blocked, _paper_trader_blocked):
        msg = fn()
        if msg:
            out.append(msg)
    return out


def is_execution_allowed() -> bool:
    return len(reasons_blocked()) == 0


def is_paper_trading_allowed() -> bool:
    """Paper trading is independent of "live" execution but still
    requires a PASS strategy AND removal of the spec-level paper-disable
    rule. Both are blocked on this branch."""
    return _no_pass_strategy_blocked() is None and _paper_trader_blocked() is None


def is_kraken_connection_allowed() -> bool:
    """Connecting to ANY broker (Kraken or otherwise) requires an
    execution module to exist AND a PASS strategy."""
    return _kraken_blocked() is None and _no_pass_strategy_blocked() is None


def reason_blocked() -> str:
    rs = reasons_blocked()
    return "; ".join(rs) if rs else ""


def safety_lock_status() -> str:
    return "locked" if not is_execution_allowed() else "unlocked"


def status() -> SafetyStatus:
    return SafetyStatus(
        execution_allowed=is_execution_allowed(),
        paper_trading_allowed=is_paper_trading_allowed(),
        kraken_connection_allowed=is_kraken_connection_allowed(),
        safety_lock_status=safety_lock_status(),
        reasons_blocked=reasons_blocked(),
    )


def assert_execution_blocked(action: str = "execution") -> None:
    """Raise `ExecutionBlocked` whenever any caller asks the engine to
    actually do something stateful. Tests assert that this raises in
    the current configuration."""
    if is_execution_allowed():
        return  # not reached on this branch
    rs = reasons_blocked()
    raise ExecutionBlocked(
        f"{action} is blocked by the safety lock: " + "; ".join(rs)
    )
