"""Tests for the repository safety CI script.

The script under test is `scripts/ci_safety_check.py`. It exposes
`run_all_checks(repo_root, tracked_files)` so tests can inject a
synthetic tracked-file list without mutating the real repository.

Each violation test creates a tiny file under `tmp_path` and points
the relevant check at it via `tracked_files=`. The clean-repo test
runs the real checks on the actual repository tree.
"""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Iterable, List

import pytest


# ---------------------------------------------------------------------------
# Load the script as a module (it lives outside `src/`, so no normal
# `from src import ...` path).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "ci_safety_check.py"


def _load_ci_script():
    """Load `scripts/ci_safety_check.py` as a module. Register it in
    `sys.modules` BEFORE executing so its `@dataclass` resolves its own
    module name correctly (Python 3.9 dataclasses look themselves up
    via `sys.modules[cls.__module__]`)."""
    import sys
    name = "ci_safety_check"
    spec = importlib.util.spec_from_file_location(name, str(_SCRIPT_PATH))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)   # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def ci():
    return _load_ci_script()


# ---------------------------------------------------------------------------
# Clean repo
# ---------------------------------------------------------------------------
def test_ci_safety_passes_on_clean_repo(ci):
    """Run the real checks against the actual repository tree. Every
    check must PASS — anything else is the headline failure of this
    branch's safety story."""
    results = ci.run_all_checks()
    fails = [r for r in results if not r.passed]
    assert not fails, "; ".join(r.line() for r in fails)


def test_ci_safety_main_exit_zero_on_clean_repo(ci):
    """The CLI entry point must return exit code 0 on a clean repo."""
    rc = ci.main()
    assert rc == 0


# ---------------------------------------------------------------------------
# Violation simulations (tracked-files based — no repo mutation)
# ---------------------------------------------------------------------------
def _real_results(ci, tracked: Iterable[str]):
    """Run only the *file-list-based* checks against a synthetic tracked
    set. The runtime checks (safety lock, system_health) run against the
    REAL repo state and are unaffected."""
    return ci.run_all_checks(tracked_files=list(tracked))


def test_ci_fails_on_tracked_generated_csv(ci):
    """A tracked `results/foo.csv` must FAIL the CSV check."""
    results = _real_results(ci, [
        "src/safety_lock.py", "results/strategy_scorecard.csv",
    ])
    csv_check = next(r for r in results
                       if r.name == "no_tracked_generated_csvs")
    assert not csv_check.passed
    assert "strategy_scorecard.csv" in csv_check.message


def test_ci_fails_on_tracked_data_raw_csv(ci):
    results = _real_results(ci, [
        "src/safety_lock.py", "data/raw/BTC_USDT_1d.csv",
    ])
    csv_check = next(r for r in results
                       if r.name == "no_tracked_generated_csvs")
    assert not csv_check.passed
    assert "BTC_USDT_1d.csv" in csv_check.message


def test_ci_fails_on_tracked_sentiment_cache(ci):
    results = _real_results(ci, [
        "src/safety_lock.py", "data/sentiment/fear_greed.csv",
    ])
    csv_check = next(r for r in results
                       if r.name == "no_tracked_generated_csvs")
    assert not csv_check.passed


def test_ci_fails_on_tracked_results_json(ci):
    results = _real_results(ci, [
        "src/safety_lock.py", "results/bot_status.json",
    ])
    json_check = next(r for r in results
                        if r.name == "no_tracked_generated_jsons")
    assert not json_check.passed
    assert "bot_status.json" in json_check.message


def test_ci_allows_gitkeep(ci):
    """`results/.gitkeep` etc. are tracked by design — the CSV check
    must NOT fire on them."""
    results = _real_results(ci, [
        "src/safety_lock.py",
        "results/.gitkeep", "data/raw/.gitkeep",
        "data/processed/.gitkeep", "logs/.gitkeep",
    ])
    csv_check = next(r for r in results
                       if r.name == "no_tracked_generated_csvs")
    assert csv_check.passed


# ---- API-key literals -------------------------------------------------------
def test_ci_fails_on_hardcoded_api_key(ci, tmp_path, monkeypatch):
    """A `KRAKEN_API_KEY = "ACTUAL-KEY"` line in tracked source must
    FAIL the API-key check."""
    fake_root = tmp_path
    src_dir = fake_root / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "leaky.py").write_text(
        'KRAKEN_API_KEY = "AAAAAAAA-BBBBBBBB-CCCCCCCC-DDDDDDDD"\n'
    )
    # Run only the file-list check by passing a custom root + tracked list.
    results = ci.check_no_api_key_literals(
        fake_root, ["src/leaky.py"],
    )
    assert not results.passed
    assert "leaky.py" in results.message


def test_ci_passes_when_safety_lock_describes_envvar_names(ci):
    """`safety_lock.py` declares string literals like
    `'paper trading remains disabled by spec'` and must NOT trip the
    API-key heuristic. The descriptive-files allowlist covers it."""
    # Real repo run: `src/safety_lock.py` is in the allowlist.
    real_root = _REPO_ROOT
    tracked = ["src/safety_lock.py"]
    out = ci.check_no_api_key_literals(real_root, tracked)
    assert out.passed


# ---- Order-placement code --------------------------------------------------
def test_ci_fails_on_order_placement_call(ci, tmp_path):
    """A `client.create_order(...)` call site in tracked source must
    FAIL the order-placement check."""
    fake_root = tmp_path
    src_dir = fake_root / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "rogue.py").write_text(
        "import some_broker\n"
        "client = some_broker.Client()\n"
        "client.create_order(symbol='BTCUSDT', side='BUY', quantity=1)\n"
    )
    out = ci.check_no_order_placement_code(
        fake_root, ["src/rogue.py"],
    )
    assert not out.passed
    assert "rogue.py" in out.message
    assert "create_order" in out.message


def test_ci_fails_on_order_placement_definition(ci, tmp_path):
    """A `def place_order(...)` definition must FAIL too."""
    fake_root = tmp_path
    src_dir = fake_root / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "rogue_def.py").write_text(
        "def place_order(symbol, side, qty):\n"
        "    return symbol\n"
    )
    out = ci.check_no_order_placement_code(
        fake_root, ["src/rogue_def.py"],
    )
    assert not out.passed


# ---- Env-var unlock --------------------------------------------------------
def test_ci_fails_on_env_unlock_pattern(ci, tmp_path):
    """A `os.environ["UNLOCK_TRADING"]` read in tracked source must
    FAIL the env-unlock check."""
    fake_root = tmp_path
    src_dir = fake_root / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "sneaky.py").write_text(
        "import os\n"
        "if os.environ['UNLOCK_TRADING'] == '1':\n"
        "    enabled = True\n"
    )
    out = ci.check_no_env_unlock(fake_root, ["src/sneaky.py"])
    assert not out.passed
    assert "UNLOCK_TRADING" in out.message


def test_ci_fails_on_live_trading_enabled_set_to_true(ci, tmp_path):
    """Setting `LIVE_TRADING_ENABLED = True` literally is a violation."""
    fake_root = tmp_path
    src_dir = fake_root / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "config_override.py").write_text(
        "LIVE_TRADING_ENABLED = True\n"
    )
    out = ci.check_no_env_unlock(
        fake_root, ["src/config_override.py"],
    )
    assert not out.passed


# ---- Streamlit key input ---------------------------------------------------
def test_ci_fails_on_streamlit_api_key_text_input(ci, tmp_path):
    """A `st.text_input("API key")` field must FAIL."""
    fake_root = tmp_path
    src_dir = fake_root
    (src_dir / "streamlit_app.py").write_text(
        "import streamlit as st\n"
        "key = st.text_input('Enter your Kraken API key')\n"
    )
    out = ci.check_no_streamlit_key_input(
        fake_root, ["streamlit_app.py"],
    )
    assert not out.passed
    assert "streamlit_app.py" in out.message


def test_ci_fails_on_streamlit_password_input(ci, tmp_path):
    """A `st.password_input(...)` is also forbidden — it's a key
    capture by another name."""
    fake_root = tmp_path
    (fake_root / "panel.py").write_text(
        "import streamlit as st\n"
        "secret = st.password_input('Kraken secret')\n"
    )
    out = ci.check_no_streamlit_key_input(
        fake_root, ["panel.py"],
    )
    assert not out.passed


def test_ci_passes_on_neutral_streamlit_input(ci, tmp_path):
    """`st.text_input("Asset name")` and similar non-key inputs MUST
    NOT trip the check."""
    fake_root = tmp_path
    (fake_root / "panel.py").write_text(
        "import streamlit as st\n"
        "asset = st.text_input('Asset symbol')\n"
    )
    out = ci.check_no_streamlit_key_input(
        fake_root, ["panel.py"],
    )
    assert out.passed


# ---- New execution module --------------------------------------------------
def test_ci_fails_on_new_kraken_execution_module(ci):
    out = ci.check_no_new_execution_module(
        _REPO_ROOT, ["src/safety_lock.py", "src/kraken_execution.py"],
    )
    assert not out.passed
    assert "kraken_execution.py" in out.message


def test_ci_fails_on_new_broker_module(ci):
    out = ci.check_no_new_execution_module(
        _REPO_ROOT, ["src/broker.py"],
    )
    assert not out.passed


# ---- Kraken private endpoint -----------------------------------------------
def test_ci_fails_on_kraken_private_path(ci, tmp_path):
    fake_root = tmp_path
    src_dir = fake_root / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "rogue_kraken.py").write_text(
        "import requests\n"
        "url = 'https://api.kraken.com/0/private/AddOrder'\n"
    )
    out = ci.check_no_kraken_private_endpoint_code(
        fake_root, ["src/rogue_kraken.py"],
    )
    assert not out.passed


# ---------------------------------------------------------------------------
# GitHub workflow file presence
# ---------------------------------------------------------------------------
def test_workflow_file_exists():
    p = _REPO_ROOT / ".github" / "workflows" / "safety-ci.yml"
    assert p.exists(), f"missing: {p}"
    text = p.read_text()
    assert len(text) > 200


def test_workflow_runs_pytest():
    p = _REPO_ROOT / ".github" / "workflows" / "safety-ci.yml"
    text = p.read_text()
    assert "pytest -q" in text


def test_workflow_runs_safety_status():
    p = _REPO_ROOT / ".github" / "workflows" / "safety-ci.yml"
    text = p.read_text()
    assert "python main.py safety_status" in text


def test_workflow_runs_system_health():
    p = _REPO_ROOT / ".github" / "workflows" / "safety-ci.yml"
    text = p.read_text()
    assert "python main.py system_health" in text


def test_workflow_runs_ci_safety_check():
    p = _REPO_ROOT / ".github" / "workflows" / "safety-ci.yml"
    text = p.read_text()
    assert "python scripts/ci_safety_check.py" in text


def test_workflow_triggers_on_push_main_and_research(_=None):
    p = _REPO_ROOT / ".github" / "workflows" / "safety-ci.yml"
    text = p.read_text()
    assert "pull_request" in text
    assert "- main" in text
    assert "- 'research/**'" in text or "- research/**" in text


# ---------------------------------------------------------------------------
# Cross-cutting: the script itself never opens the network or reads keys
# ---------------------------------------------------------------------------
def test_ci_script_reads_no_network_imports(ci):
    src = inspect.getsource(ci)
    # No HTTP libs, no broker SDKs.
    forbidden = ("import requests", "from requests", "import ccxt",
                  "from ccxt", "import urllib", "from urllib")
    for token in forbidden:
        assert token not in src, f"ci_safety_check imports {token!r}"


def test_ci_script_does_not_read_api_key_values(ci):
    src = inspect.getsource(ci)
    assert "os.environ['KRAKEN_API_KEY']" not in src
    assert 'os.environ["KRAKEN_API_KEY"]' not in src
