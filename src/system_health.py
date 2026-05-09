"""
System health checks — read-only operational sanity for the research
engine.

Every check returns one row with `(check_name, status, severity,
message, recommended_action)`. Status ∈ {PASS, WARNING, FAIL}.

These checks NEVER modify state, NEVER call brokers, NEVER read API
keys. They verify the engine is in a healthy research-only posture.

Output:
    results/system_health.csv
    results/system_health.json
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from . import bot_status, config, safety_lock, utils

logger = utils.get_logger("cte.system_health")


HEALTH_COLUMNS: List[str] = [
    "check_name", "status", "severity", "message", "recommended_action",
]
STATUS_PASS = "PASS"
STATUS_WARN = "WARNING"
STATUS_FAIL = "FAIL"

OUTPUT_CSV = "system_health.csv"
OUTPUT_JSON = "system_health.json"

REC_RESEARCH_ONLY = "continue research; do not trade"
REC_REFRESH_DATA = "refresh local data caches"
REC_RUN_RESEARCH = "run a research command to populate"


# ---------------------------------------------------------------------------
# Helper: result row builder
# ---------------------------------------------------------------------------
def _row(name: str, status: str, severity: str, message: str,
          recommended_action: str = REC_RESEARCH_ONLY) -> Dict[str, Any]:
    if status not in (STATUS_PASS, STATUS_WARN, STATUS_FAIL):
        raise ValueError(f"unknown status {status!r}")
    return {
        "check_name": name,
        "status": status,
        "severity": severity,
        "message": message,
        "recommended_action": recommended_action,
    }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def _check_python_version() -> Dict[str, Any]:
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 9):
        return _row(
            "python_version", STATUS_FAIL, "high",
            f"Python {v.major}.{v.minor} is too old (need ≥ 3.9).",
            "upgrade Python to 3.12 (Streamlit Cloud) or 3.10+ locally",
        )
    if v.major == 3 and v.minor == 9:
        return _row(
            "python_version", STATUS_WARN, "low",
            f"Python 3.9 still works but 3.12 is the supported target.",
            "upgrade to 3.12 when convenient",
        )
    return _row(
        "python_version", STATUS_PASS, "info",
        f"Python {v.major}.{v.minor}.{v.micro}",
        REC_RESEARCH_ONLY,
    )


def _check_required_folders() -> Dict[str, Any]:
    missing = []
    for p in (config.DATA_RAW_DIR, config.DATA_PROCESSED_DIR,
                config.LOGS_DIR, config.RESULTS_DIR,
                config.REPO_ROOT / "src", config.REPO_ROOT / "tests"):
        if not p.exists():
            missing.append(str(p.relative_to(config.REPO_ROOT)
                                if p.is_relative_to(config.REPO_ROOT)
                                else p))
    if missing:
        return _row(
            "required_folders", STATUS_FAIL, "high",
            "Missing required folders: " + ", ".join(missing),
            "create the listed folders or re-clone the repo",
        )
    return _row(
        "required_folders", STATUS_PASS, "info",
        "all required folders present",
        REC_RESEARCH_ONLY,
    )


def _check_results_writable() -> Dict[str, Any]:
    p = config.RESULTS_DIR
    if not p.exists():
        return _row(
            "results_writable", STATUS_FAIL, "high",
            f"results dir does not exist: {p}",
            "create the results/ folder",
        )
    test = p / ".__health_check"
    try:
        test.write_text("ok")
        test.unlink()
    except Exception as e:  # noqa: BLE001
        return _row(
            "results_writable", STATUS_FAIL, "high",
            f"results dir not writable ({e})",
            "fix filesystem permissions on results/",
        )
    return _row(
        "results_writable", STATUS_PASS, "info",
        f"results dir is writable: {p}",
        REC_RESEARCH_ONLY,
    )


def _check_data_raw_present() -> Dict[str, Any]:
    """Cache state. In CI this is expected to be empty / missing — both
    cases are WARNING (the engine still loads fine; downstream commands
    that need data will say so themselves)."""
    p = config.DATA_RAW_DIR
    if not p.exists():
        return _row(
            "data_raw_present", STATUS_WARN, "low",
            f"data/raw directory missing: {p}",
            "create data/raw (or rely on `.gitkeep`) and "
            "`python main.py download` when needed",
        )
    csv_count = sum(1 for _ in p.glob("*.csv"))
    if csv_count == 0:
        return _row(
            "data_raw_present", STATUS_WARN, "low",
            "data/raw exists but contains no CSVs (expected in fresh CI)",
            "run `python main.py download` locally to populate",
        )
    return _row(
        "data_raw_present", STATUS_PASS, "info",
        f"data/raw contains {csv_count} CSV file(s)",
        REC_RESEARCH_ONLY,
    )


def _check_latest_research_report() -> Dict[str, Any]:
    p = config.REPO_ROOT / "reports"
    if not p.exists():
        return _row(
            "latest_research_report", STATUS_WARN, "low",
            "reports/ directory missing — no closure documents",
            REC_RUN_RESEARCH,
        )
    md = list(p.glob("*.md"))
    if not md:
        return _row(
            "latest_research_report", STATUS_WARN, "low",
            "no .md research reports under reports/",
            REC_RUN_RESEARCH,
        )
    return _row(
        "latest_research_report", STATUS_PASS, "info",
        f"{len(md)} research report(s) under reports/",
        REC_RESEARCH_ONLY,
    )


def _check_latest_bot_status() -> Dict[str, Any]:
    p = config.RESULTS_DIR / "bot_status.csv"
    if not p.exists() or p.stat().st_size == 0:
        return _row(
            "latest_bot_status", STATUS_WARN, "low",
            "results/bot_status.csv missing — run `python main.py bot_status`",
            "run `python main.py bot_status` to refresh",
        )
    return _row(
        "latest_bot_status", STATUS_PASS, "info",
        "bot_status.csv present", REC_RESEARCH_ONLY,
    )


def _check_generated_csvs_gitignored() -> Dict[str, Any]:
    """Sanity-check that `.gitignore` lists the generated-data folders.
    We grep the file rather than running git so the check works without
    a git repo (e.g. Streamlit Cloud)."""
    gi = config.REPO_ROOT / ".gitignore"
    if not gi.exists():
        return _row(
            "generated_csvs_gitignored", STATUS_FAIL, "high",
            ".gitignore missing — generated CSVs may be tracked",
            "create .gitignore with results/*.csv etc.",
        )
    text = gi.read_text()
    expected = ("results/*.csv", "data/raw/*.csv", "logs/*.csv")
    missing = [e for e in expected if e not in text]
    if missing:
        return _row(
            "generated_csvs_gitignored", STATUS_WARN, "medium",
            ".gitignore is missing entries: " + ", ".join(missing),
            "extend .gitignore with the missing patterns",
        )
    return _row(
        "generated_csvs_gitignored", STATUS_PASS, "info",
        "generated-CSV folders are gitignored",
        REC_RESEARCH_ONLY,
    )


_API_KEY_PATTERNS = (
    r"\bKRAKEN_API_KEY\s*=\s*['\"][A-Za-z0-9_\-]{8,}['\"]",
    r"\bKRAKEN_API_SECRET\s*=\s*['\"][A-Za-z0-9_\-/+=]{8,}['\"]",
    r"\bBINANCE_API_KEY\s*=\s*['\"][A-Za-z0-9_\-]{8,}['\"]",
    r"\bBINANCE_API_SECRET\s*=\s*['\"][A-Za-z0-9_\-/+=]{8,}['\"]",
    r"\bX-MBX-APIKEY\s*[:=]",
)


def _check_no_api_keys_in_tracked_files() -> Dict[str, Any]:
    """Greps every tracked source file for what looks like a hardcoded
    API key. Doesn't require git — just walks the working tree."""
    suspicious: List[str] = []
    for p in (config.REPO_ROOT / "src").rglob("*.py"):
        try:
            text = p.read_text(errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        for pat in _API_KEY_PATTERNS:
            if re.search(pat, text):
                suspicious.append(f"{p.name}: {pat}")
                break
    # main.py + tests/ scan too
    for p in [config.REPO_ROOT / "main.py"]:
        if p.exists():
            try:
                text = p.read_text(errors="ignore")
            except Exception:  # noqa: BLE001
                continue
            for pat in _API_KEY_PATTERNS:
                if re.search(pat, text):
                    suspicious.append(f"{p.name}: {pat}")
                    break
    if suspicious:
        return _row(
            "no_api_keys_in_tracked_files", STATUS_FAIL, "critical",
            "possible hardcoded broker keys: " + "; ".join(suspicious),
            "remove the literal key value from source code",
        )
    return _row(
        "no_api_keys_in_tracked_files", STATUS_PASS, "info",
        "no broker-key literals found in tracked source",
        REC_RESEARCH_ONLY,
    )


def _check_no_kraken_execution_module() -> Dict[str, Any]:
    src = config.REPO_ROOT / "src"
    if not src.exists():
        return _row(
            "no_kraken_execution_module", STATUS_PASS, "info",
            "src/ missing; trivially no execution module",
            REC_RESEARCH_ONLY,
        )
    forbidden = ("kraken_execution.py", "kraken_broker.py",
                  "kraken_client.py", "execution.py")
    found = [f for f in forbidden if (src / f).exists()]
    if found:
        return _row(
            "no_kraken_execution_module", STATUS_FAIL, "critical",
            "execution-style module(s) present in src/: " + ", ".join(found),
            "remove the file(s) or audit their contents before any unlock",
        )
    return _row(
        "no_kraken_execution_module", STATUS_PASS, "info",
        "no kraken/execution modules in src/",
        REC_RESEARCH_ONLY,
    )


def _check_safety_lock_locked() -> Dict[str, Any]:
    s = safety_lock.status()
    if s.execution_allowed:
        return _row(
            "safety_lock_locked", STATUS_FAIL, "critical",
            "safety lock reports execution_allowed=True — every gate "
            "released. Review immediately.",
            "review safety_lock.reasons_blocked() and the unlock procedure",
        )
    return _row(
        "safety_lock_locked", STATUS_PASS, "info",
        f"safety lock is {s.safety_lock_status} ("
        f"{len(s.reasons_blocked)} active reason(s))",
        REC_RESEARCH_ONLY,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_CHECKS = (
    _check_python_version,
    _check_required_folders,
    _check_results_writable,
    _check_data_raw_present,
    _check_latest_research_report,
    _check_latest_bot_status,
    _check_generated_csvs_gitignored,
    _check_no_api_keys_in_tracked_files,
    _check_no_kraken_execution_module,
    _check_safety_lock_locked,
)


def run_health_checks() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for check in _CHECKS:
        try:
            rows.append(check())
        except Exception as e:  # noqa: BLE001
            rows.append(_row(
                check.__name__.replace("_check_", ""),
                STATUS_FAIL, "high",
                f"check raised: {e}",
                "investigate the listed exception",
            ))
    return rows


def write_health(save: bool = True) -> pd.DataFrame:
    rows = run_health_checks()
    df = pd.DataFrame(rows, columns=HEALTH_COLUMNS)
    if save:
        utils.write_df(df, config.RESULTS_DIR / OUTPUT_CSV)
        (config.RESULTS_DIR / OUTPUT_JSON).parent.mkdir(
            parents=True, exist_ok=True,
        )
        (config.RESULTS_DIR / OUTPUT_JSON).write_text(
            json.dumps(rows, indent=2, default=str),
        )
    return df
