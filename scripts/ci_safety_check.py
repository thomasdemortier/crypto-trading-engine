#!/usr/bin/env python3
"""
Repository safety check — CI gate for the crypto_trading_engine.

This script enforces every safety invariant the project has accumulated
across the v1 closure + the bot-shell + the observability layer. It is
the **last line of defence** before code lands on `main`. If any check
fails, CI fails the build.

Run locally:

    python scripts/ci_safety_check.py

Run in CI: see `.github/workflows/safety-ci.yml`.

The script:
    * Is deterministic (no time-of-day dependency, no random).
    * Requires no network.
    * Requires no API keys.
    * Does not modify any repo file.
    * Reads `git ls-files` if available, otherwise falls back to a
      filesystem walk (any tests pass `tracked_files` directly).

Exit codes:
    0 — every check passed.
    Non-zero — at least one check failed; details on stderr/stdout.

The check list (matches the spec in the platform-hardening prompt):
    1.  Safety lock must be locked.
    2.  `safety_status` exposes execution_allowed=False,
        paper_trading_allowed=False, kraken_connection_allowed=False.
    3.  `system_health` returns no FAIL rows.
    4.  No tracked generated CSVs (results/*.csv except .gitkeep).
    5.  No tracked generated JSONs (results/*.json).
    6.  No tracked raw-data CSVs (data/raw/*.csv).
    7.  No tracked sentiment / futures / market_structure CSVs.
    8.  No API-key literals in any tracked source file.
    9.  No Kraken private-endpoint code in `src/`.
    10. No order-placement function definition in `src/`.
    11. No `create_order|place_order|submit_order|send_order|cancel_order`
        broker call sites.
    12. No environment-variable unlock pattern (ENABLE_LIVE,
        UNLOCK_TRADING, FORCE_TRADE, LIVE_TRADING_ENABLED override).
    13. No new execution module in `src/` (kraken_execution, etc.).
    14. No Streamlit API-key input field.
    15. No `st.text_input(... API key ...)` UI capture.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Repo root resolution
# ---------------------------------------------------------------------------
def _find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for p in (here, *here.parents):
        if (p / ".git").exists() or (p / "src").exists():
            return p
    return here


REPO_ROOT = _find_repo_root()


# ---------------------------------------------------------------------------
# Result row
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    message: str

    def line(self) -> str:
        prefix = "PASS" if self.passed else "FAIL"
        return f"[{prefix}] {self.name}: {self.message}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _list_tracked_files(repo_root: Path) -> List[str]:
    """Use `git ls-files`. If git isn't available (rare in CI), fall
    back to a manual walk of paths that COULD be tracked (excluding the
    standard ignored dirs)."""
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=str(repo_root),
            capture_output=True, text=True, timeout=30, check=False,
        )
        if out.returncode == 0:
            return [line for line in out.stdout.splitlines() if line.strip()]
    except Exception:  # noqa: BLE001
        pass
    # Manual fallback — walk repo and assume `.gitignore` is respected.
    SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules",
                  "external", "models", "hf_cache", ".cache", ".pytest_cache",
                  ".mypy_cache"}
    files: List[str] = []
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(repo_root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        files.append(str(rel))
    return files


def _read_text_safe(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:  # noqa: BLE001
        return ""


def _strip_docstrings_and_comments(src: str) -> str:
    """Remove Python docstrings (triple-quoted blocks) and `#` comment
    lines. The grep checks below are about real code, not prose."""
    code_lines: List[str] = []
    in_doc = False
    for raw in src.splitlines():
        s = raw.strip()
        if s.startswith('"""') or s.startswith("'''"):
            in_doc = not in_doc
            continue
        if in_doc or s.startswith("#"):
            continue
        code_lines.append(raw)
    return "\n".join(code_lines)


# ---------------------------------------------------------------------------
# Allowed exceptions
# ---------------------------------------------------------------------------
_ALLOWED_TRACKED_CSVS = {
    # Empty-folder placeholders — these MUST be tracked.
    "data/raw/.gitkeep", "data/processed/.gitkeep",
    "logs/.gitkeep", "results/.gitkeep",
}
_ALLOWED_TRACKED_JSONS: set = set()


# Paths that legitimately discuss the forbidden tokens (docs, this
# script itself, the safety-lock module that DECLARES the env-var
# names so the lock can read them, the test files that simulate
# violations).
_DESCRIPTIVE_FILE_PATTERNS = (
    re.compile(r"^docs/.*\.md$"),
    re.compile(r"^scripts/ci_safety_check\.py$"),
    re.compile(r"^src/safety_lock\.py$"),
    re.compile(r"^src/system_health\.py$"),
    re.compile(r"^tests/.*\.py$"),
    re.compile(r"^reports/.*\.md$"),
    re.compile(r"^README\.md$"),
)


def _is_descriptive(rel: str) -> bool:
    """True if a tracked file is allowed to MENTION dangerous tokens
    (docs, tests, the safety-lock module that declares env-var names
    so the lock itself can recognise them)."""
    return any(p.match(rel) for p in _DESCRIPTIVE_FILE_PATTERNS)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_safety_lock_locked(repo_root: Path) -> CheckResult:
    """Import the safety lock and verify every gate is closed."""
    sys.path.insert(0, str(repo_root))
    try:
        from src import safety_lock  # type: ignore
    except Exception as e:  # noqa: BLE001
        return CheckResult("safety_lock_locked", False,
                            f"could not import safety_lock: {e}")
    finally:
        if str(repo_root) in sys.path:
            sys.path.remove(str(repo_root))

    s = safety_lock.status()
    if s.execution_allowed or s.paper_trading_allowed \
            or s.kraken_connection_allowed:
        return CheckResult(
            "safety_lock_locked", False,
            f"lock is NOT fully locked: execution={s.execution_allowed}, "
            f"paper={s.paper_trading_allowed}, "
            f"kraken={s.kraken_connection_allowed}",
        )
    return CheckResult(
        "safety_lock_locked", True,
        f"locked ({len(s.reasons_blocked)} reason(s))",
    )


def check_system_health_no_fails(repo_root: Path) -> CheckResult:
    sys.path.insert(0, str(repo_root))
    try:
        from src import system_health  # type: ignore
    except Exception as e:  # noqa: BLE001
        return CheckResult("system_health_no_fails", False,
                            f"could not import system_health: {e}")
    finally:
        if str(repo_root) in sys.path:
            sys.path.remove(str(repo_root))

    rows = system_health.run_health_checks()
    fails = [r for r in rows if r["status"] == system_health.STATUS_FAIL]
    if fails:
        names = ", ".join(r["check_name"] for r in fails)
        return CheckResult(
            "system_health_no_fails", False,
            f"{len(fails)} FAIL row(s): {names}",
        )
    n_warn = sum(1 for r in rows
                  if r["status"] == system_health.STATUS_WARN)
    return CheckResult(
        "system_health_no_fails", True,
        f"{len(rows)} check(s) ran; {n_warn} WARNING, 0 FAIL",
    )


def check_no_tracked_generated_csvs(
    repo_root: Path, tracked: Sequence[str],
) -> CheckResult:
    bad: List[str] = []
    for rel in tracked:
        p = rel.replace("\\", "/")
        if not p.endswith(".csv"):
            continue
        if p in _ALLOWED_TRACKED_CSVS:
            continue
        if (p.startswith("results/") or p.startswith("logs/")
                or p.startswith("data/raw/")
                or p.startswith("data/processed/")
                or p.startswith("data/futures/")
                or p.startswith("data/market_structure/")
                or p.startswith("data/sentiment/")):
            bad.append(p)
    if bad:
        return CheckResult(
            "no_tracked_generated_csvs", False,
            f"tracked generated CSVs: {bad[:10]}"
            + ("…" if len(bad) > 10 else ""),
        )
    return CheckResult(
        "no_tracked_generated_csvs", True,
        "no generated CSVs in the tracked tree",
    )


def check_no_tracked_generated_jsons(
    repo_root: Path, tracked: Sequence[str],
) -> CheckResult:
    bad: List[str] = []
    for rel in tracked:
        p = rel.replace("\\", "/")
        if not p.endswith(".json"):
            continue
        if p in _ALLOWED_TRACKED_JSONS:
            continue
        if p.startswith("results/") or p.startswith("logs/"):
            bad.append(p)
    if bad:
        return CheckResult(
            "no_tracked_generated_jsons", False,
            f"tracked generated JSONs: {bad[:10]}",
        )
    return CheckResult(
        "no_tracked_generated_jsons", True,
        "no generated JSONs in the tracked tree",
    )


# Patterns for hardcoded API keys.
_API_KEY_PATTERNS = (
    re.compile(r"\bKRAKEN_API_KEY\s*=\s*['\"][A-Za-z0-9_\-]{8,}['\"]"),
    re.compile(r"\bKRAKEN_API_SECRET\s*=\s*['\"][A-Za-z0-9_\-/+=]{8,}['\"]"),
    re.compile(r"\bBINANCE_API_KEY\s*=\s*['\"][A-Za-z0-9_\-]{8,}['\"]"),
    re.compile(r"\bBINANCE_API_SECRET\s*=\s*['\"][A-Za-z0-9_\-/+=]{8,}['\"]"),
    re.compile(r"\bX-MBX-APIKEY\s*[:=]"),
)


def check_no_api_key_literals(
    repo_root: Path, tracked: Sequence[str],
) -> CheckResult:
    """Search every tracked .py / .toml / .yaml file for hardcoded
    broker key literals. Docs and the tests are allowed to discuss
    the env-var names but not put a literal value next to `=`."""
    bad: List[str] = []
    for rel in tracked:
        p = rel.replace("\\", "/")
        if not p.endswith((".py", ".toml", ".yaml", ".yml", ".env")):
            continue
        if _is_descriptive(p):
            continue
        text = _read_text_safe(repo_root / p)
        if not text:
            continue
        code = _strip_docstrings_and_comments(text)
        for pat in _API_KEY_PATTERNS:
            if pat.search(code):
                bad.append(f"{p}:{pat.pattern}")
                break
    if bad:
        return CheckResult(
            "no_api_key_literals", False,
            f"hardcoded broker keys: {bad[:5]}",
        )
    return CheckResult(
        "no_api_key_literals", True,
        "no broker-key literals in tracked source",
    )


_ORDER_CALL_PATTERNS = (
    re.compile(r"\bcreate_order\s*\("),
    re.compile(r"\bplace_order\s*\("),
    re.compile(r"\bsubmit_order\s*\("),
    re.compile(r"\bsend_order\s*\("),
    re.compile(r"\bcancel_order\s*\("),
)
_ORDER_DEF_PATTERNS = (
    re.compile(r"\bdef\s+(?:create|place|submit|send|cancel)_order\b"),
)


def check_no_order_placement_code(
    repo_root: Path, tracked: Sequence[str],
) -> CheckResult:
    bad: List[str] = []
    for rel in tracked:
        p = rel.replace("\\", "/")
        if not p.startswith(("src/", "main.py", "streamlit_app.py")):
            continue
        if not p.endswith(".py"):
            continue
        if _is_descriptive(p):
            continue
        text = _read_text_safe(repo_root / p)
        if not text:
            continue
        code = _strip_docstrings_and_comments(text)
        for pat in (*_ORDER_CALL_PATTERNS, *_ORDER_DEF_PATTERNS):
            if pat.search(code):
                bad.append(f"{p}: {pat.pattern}")
                break
    if bad:
        return CheckResult(
            "no_order_placement_code", False,
            f"order-placement plumbing detected: {bad[:5]}",
        )
    return CheckResult(
        "no_order_placement_code", True,
        "no order-placement function calls or definitions",
    )


_KRAKEN_PRIVATE_PATTERNS = (
    re.compile(r"\bkraken\.private\b"),
    re.compile(r"\bkraken_private\b"),
    re.compile(r"['\"]/0/private/"),
    re.compile(r"\bAddOrder\b"),
    re.compile(r"\bCancelOrder\b"),
    re.compile(r"\bWithdraw\b"),
)


def check_no_kraken_private_endpoint_code(
    repo_root: Path, tracked: Sequence[str],
) -> CheckResult:
    bad: List[str] = []
    for rel in tracked:
        p = rel.replace("\\", "/")
        if not p.startswith(("src/", "main.py", "streamlit_app.py")):
            continue
        if not p.endswith(".py"):
            continue
        if _is_descriptive(p):
            continue
        text = _read_text_safe(repo_root / p)
        if not text:
            continue
        code = _strip_docstrings_and_comments(text)
        for pat in _KRAKEN_PRIVATE_PATTERNS:
            if pat.search(code):
                bad.append(f"{p}: {pat.pattern}")
                break
    if bad:
        return CheckResult(
            "no_kraken_private_endpoint_code", False,
            f"kraken-private patterns detected: {bad[:5]}",
        )
    return CheckResult(
        "no_kraken_private_endpoint_code", True,
        "no kraken private-endpoint code in src/",
    )


_ENV_UNLOCK_PATTERNS = (
    re.compile(r"\bos\.environ\[\s*['\"]ENABLE_LIVE['\"]\s*\]"),
    re.compile(r"\bos\.getenv\s*\(\s*['\"]ENABLE_LIVE['\"]"),
    re.compile(r"\bos\.environ\[\s*['\"]UNLOCK_TRADING['\"]\s*\]"),
    re.compile(r"\bos\.getenv\s*\(\s*['\"]UNLOCK_TRADING['\"]"),
    re.compile(r"\bos\.environ\[\s*['\"]FORCE_TRADE['\"]\s*\]"),
    re.compile(r"\bos\.getenv\s*\(\s*['\"]FORCE_TRADE['\"]"),
    # An explicit override of `LIVE_TRADING_ENABLED` from env.
    re.compile(r"LIVE_TRADING_ENABLED\s*=\s*os\."),
    re.compile(r"LIVE_TRADING_ENABLED\s*=\s*True"),
)


def check_no_env_unlock(
    repo_root: Path, tracked: Sequence[str],
) -> CheckResult:
    """Reject any code that READS an env var named ENABLE_LIVE /
    UNLOCK_TRADING / FORCE_TRADE, or that sets `LIVE_TRADING_ENABLED`
    to True from env or literal True."""
    bad: List[str] = []
    for rel in tracked:
        p = rel.replace("\\", "/")
        if not p.endswith(".py"):
            continue
        if _is_descriptive(p):
            continue
        text = _read_text_safe(repo_root / p)
        if not text:
            continue
        code = _strip_docstrings_and_comments(text)
        for pat in _ENV_UNLOCK_PATTERNS:
            if pat.search(code):
                bad.append(f"{p}: {pat.pattern}")
                break
    if bad:
        return CheckResult(
            "no_env_unlock", False,
            f"env-var unlock patterns detected: {bad[:5]}",
        )
    return CheckResult(
        "no_env_unlock", True,
        "no env-var unlock pattern in tracked source",
    )


_FORBIDDEN_EXECUTION_FILES = (
    "src/kraken_execution.py", "src/kraken_broker.py",
    "src/kraken_client.py", "src/kraken_private.py",
    "src/execution.py", "src/broker.py",
)


def check_no_new_execution_module(
    repo_root: Path, tracked: Sequence[str],
) -> CheckResult:
    bad = [rel.replace("\\", "/") for rel in tracked
            if rel.replace("\\", "/") in _FORBIDDEN_EXECUTION_FILES]
    if bad:
        return CheckResult(
            "no_new_execution_module", False,
            f"forbidden execution module(s) tracked: {bad}",
        )
    return CheckResult(
        "no_new_execution_module", True,
        "no execution-style module in src/",
    )


_STREAMLIT_KEY_INPUT_PATTERNS = (
    re.compile(r"\bst\.text_input\s*\([^)]*api[\s_]?key", re.IGNORECASE),
    re.compile(r"\bst\.text_input\s*\([^)]*secret", re.IGNORECASE),
    re.compile(r"\bst\.password_input\s*\(", re.IGNORECASE),
)


def check_no_streamlit_key_input(
    repo_root: Path, tracked: Sequence[str],
) -> CheckResult:
    bad: List[str] = []
    for rel in tracked:
        p = rel.replace("\\", "/")
        if not p.endswith(".py"):
            continue
        if _is_descriptive(p):
            continue
        text = _read_text_safe(repo_root / p)
        if not text:
            continue
        # Only check files that actually import streamlit.
        if "import streamlit" not in text:
            continue
        code = _strip_docstrings_and_comments(text)
        for pat in _STREAMLIT_KEY_INPUT_PATTERNS:
            if pat.search(code):
                bad.append(f"{p}: {pat.pattern}")
                break
    if bad:
        return CheckResult(
            "no_streamlit_key_input", False,
            f"Streamlit key-input field detected: {bad[:5]}",
        )
    return CheckResult(
        "no_streamlit_key_input", True,
        "no Streamlit API-key input field",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_all_checks(
    repo_root: Optional[Path] = None,
    tracked_files: Optional[Sequence[str]] = None,
) -> List[CheckResult]:
    """Run every check. `tracked_files` lets tests inject a synthetic
    file list; in normal use we read it from `git ls-files`."""
    repo_root = repo_root or REPO_ROOT
    tracked = list(tracked_files) if tracked_files is not None \
                else _list_tracked_files(repo_root)
    results: List[CheckResult] = [
        check_safety_lock_locked(repo_root),
        check_system_health_no_fails(repo_root),
        check_no_tracked_generated_csvs(repo_root, tracked),
        check_no_tracked_generated_jsons(repo_root, tracked),
        check_no_api_key_literals(repo_root, tracked),
        check_no_order_placement_code(repo_root, tracked),
        check_no_kraken_private_endpoint_code(repo_root, tracked),
        check_no_env_unlock(repo_root, tracked),
        check_no_new_execution_module(repo_root, tracked),
        check_no_streamlit_key_input(repo_root, tracked),
    ]
    return results


def main() -> int:
    print("=" * 72)
    print(f"crypto_trading_engine — repository safety check")
    print(f"repo root: {REPO_ROOT}")
    print("=" * 72)
    results = run_all_checks()
    n_pass = sum(1 for r in results if r.passed)
    n_fail = sum(1 for r in results if not r.passed)
    for r in results:
        print(r.line())
    print("=" * 72)
    print(f"summary: {n_pass} PASS / {n_fail} FAIL "
          f"({len(results)} checks total)")
    if n_fail:
        print("\nCI gate FAILED. Address every FAIL row above before merging.")
        return 1
    print("\nCI gate PASSED. Safety invariants intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
