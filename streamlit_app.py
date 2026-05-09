"""
Streamlit dashboard.

Run locally:
    streamlit run streamlit_app.py

Deployable to Streamlit Community Cloud — no API keys, all paths relative.

This module is the UI layer only. All trading, backtesting, risk, and
performance logic lives in `src/`. The hard `LIVE_TRADING_ENABLED = False`
safety lock is enforced by `src/utils.assert_paper_only()` and is not
overridable from the dashboard.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from src import (
    config, backtester, data_collector, paper_trader,
    performance, plotting, research, utils,
)


# ---------------------------------------------------------------------------
# Page setup + premium dark theme CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Crypto Trading Engine",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)


_CTE_CSS = """
<style>
:root {
  --cte-bg: #0a0e1a;
  --cte-card: #111827;
  --cte-card-2: #0f172a;
  --cte-border: #1f2937;
  --cte-border-soft: rgba(148,163,184,0.10);
  --cte-text: #e5e7eb;
  --cte-muted: #9ca3af;
  --cte-accent: #38bdf8;
  --cte-accent-soft: rgba(56,189,248,0.12);
  --cte-pos: #10b981;
  --cte-pos-soft: rgba(16,185,129,0.14);
  --cte-neg: #f43f5e;
  --cte-neg-soft: rgba(244,63,94,0.14);
  --cte-warn: #f59e0b;
  --cte-warn-soft: rgba(245,158,11,0.14);
}

/* Page background */
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
  background: var(--cte-bg) !important;
}
[data-testid="stHeader"] { backdrop-filter: none; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: var(--cte-card-2) !important;
  border-right: 1px solid var(--cte-border);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
  color: var(--cte-text) !important;
  letter-spacing: 0.01em;
}
[data-testid="stSidebar"] [data-testid="stExpander"] {
  border: 1px solid var(--cte-border) !important;
  border-radius: 10px !important;
  background: rgba(15,23,42,0.6);
  margin-bottom: 0.6rem;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
  font-weight: 600;
  color: var(--cte-text) !important;
  padding: 0.55rem 0.7rem !important;
}
/* Sidebar buttons — explicit contrast against the sidebar surface so they
   never disappear into the background. */
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stDownloadButton > button {
  background: #1e293b !important;
  color: var(--cte-text) !important;
  border: 1px solid #334155 !important;
  font-weight: 600 !important;
  min-height: 2.4rem;
}
[data-testid="stSidebar"] .stButton > button:hover {
  border-color: var(--cte-accent) !important;
  background: rgba(56,189,248,0.10) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: var(--cte-accent) !important;
  color: #0a0e1a !important;
  border-color: var(--cte-accent) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
  background: #0ea5e9 !important; border-color: #0ea5e9 !important;
}

/* Native bordered containers — our "cards". Keep the chrome subtle so it
   never collapses or hides inner widgets. */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--cte-card);
  border: 1px solid var(--cte-border) !important;
  border-radius: 14px !important;
  box-shadow: 0 1px 0 rgba(255,255,255,0.02), 0 6px 22px rgba(0,0,0,0.18);
}
/* Padding only on the direct content wrapper inside a bordered container —
   never use !important padding on the wrapper itself, which can clip inner
   tab panels and dataframes. */
[data-testid="stVerticalBlockBorderWrapper"] > div {
  padding: 0.85rem 1rem;
}

/* Native st.metric polishing */
[data-testid="stMetric"] {
  background: transparent;
  border-radius: 10px;
  padding: 0.25rem 0.1rem;
}
[data-testid="stMetricLabel"] p {
  font-size: 0.78rem !important;
  color: var(--cte-muted) !important;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-weight: 600;
}
[data-testid="stMetricValue"] {
  font-size: 1.55rem !important;
  font-weight: 700 !important;
  color: var(--cte-text) !important;
  letter-spacing: -0.01em;
}
[data-testid="stMetricDelta"] svg { display: none; }
[data-testid="stMetricDelta"] {
  font-weight: 600 !important;
  font-size: 0.85rem !important;
}

/* Custom header */
.cte-header {
  display: flex; align-items: center; justify-content: space-between;
  gap: 1rem; flex-wrap: wrap;
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
  border: 1px solid var(--cte-border);
  border-radius: 16px;
  padding: 1rem 1.3rem;
  margin-bottom: 1.1rem;
}
.cte-title { font-size: 1.45rem; font-weight: 700; color: var(--cte-text);
             letter-spacing: -0.01em; line-height: 1.1; }
.cte-subtitle { color: var(--cte-muted); font-size: 0.92rem; margin-top: 0.18rem; }
.cte-pills { display: flex; gap: 0.45rem; flex-wrap: wrap; align-items: center; }
.pill {
  display: inline-flex; align-items: center; gap: 0.35rem;
  font-size: 0.78rem; font-weight: 600; letter-spacing: 0.02em;
  padding: 0.28rem 0.65rem; border-radius: 999px;
  border: 1px solid transparent;
}
.pill-blue { color: var(--cte-accent); background: var(--cte-accent-soft);
             border-color: rgba(56,189,248,0.35); }
.pill-amber { color: var(--cte-warn); background: var(--cte-warn-soft);
              border-color: rgba(245,158,11,0.35); }
.pill-pos  { color: var(--cte-pos); background: var(--cte-pos-soft);
             border-color: rgba(16,185,129,0.35); }
.pill-neg  { color: var(--cte-neg); background: var(--cte-neg-soft);
             border-color: rgba(244,63,94,0.35); }
.pill-grey { color: var(--cte-muted); background: rgba(148,163,184,0.10);
             border-color: rgba(148,163,184,0.25); }
.pill-dot { width: 6px; height: 6px; border-radius: 999px; background: currentColor; }

/* Hero metric cards */
.hero-grid { display: grid; grid-template-columns: repeat(4, minmax(0,1fr));
             gap: 0.9rem; margin-bottom: 1.1rem; }
@media (max-width: 1100px) { .hero-grid { grid-template-columns: repeat(2,1fr); } }
@media (max-width:  640px) { .hero-grid { grid-template-columns: 1fr; } }
.hero-card {
  background: var(--cte-card);
  border: 1px solid var(--cte-border);
  border-radius: 14px;
  padding: 1rem 1.1rem;
  box-shadow: 0 6px 22px rgba(0,0,0,0.18);
  position: relative; overflow: hidden;
}
.hero-card::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: var(--cte-accent); opacity: 0.55;
}
.hero-card.pos::before { background: var(--cte-pos); }
.hero-card.neg::before { background: var(--cte-neg); }
.hero-label { font-size: 0.72rem; color: var(--cte-muted);
              text-transform: uppercase; letter-spacing: 0.07em; font-weight: 700; }
.hero-value { font-size: 1.85rem; font-weight: 700; color: var(--cte-text);
              margin-top: 0.25rem; letter-spacing: -0.02em; line-height: 1.1; }
.hero-value.pos { color: var(--cte-pos); }
.hero-value.neg { color: var(--cte-neg); }
.hero-context { font-size: 0.82rem; color: var(--cte-muted); margin-top: 0.4rem; }

/* Scope grid */
.scope-grid { display: grid; grid-template-columns: repeat(4, minmax(0,1fr));
              gap: 0.9rem; }
@media (max-width: 1100px) { .scope-grid { grid-template-columns: repeat(2,1fr); } }
.scope-cell { background: var(--cte-card-2); border: 1px solid var(--cte-border-soft);
              border-radius: 10px; padding: 0.7rem 0.85rem; }
.scope-cell .l { font-size: 0.7rem; color: var(--cte-muted);
                 text-transform: uppercase; letter-spacing: 0.07em; font-weight: 700; }
.scope-cell .v { font-size: 0.98rem; color: var(--cte-text); font-weight: 600;
                 margin-top: 0.2rem; word-break: break-word; }

/* Section header */
.section-h {
  font-size: 1.05rem; font-weight: 700; color: var(--cte-text);
  margin: 0 0 0.6rem 0; display: flex; align-items: center; gap: 0.6rem;
}
.section-h .dot { width: 6px; height: 6px; border-radius: 999px;
                  background: var(--cte-accent); }
.section-sub { color: var(--cte-muted); font-size: 0.82rem; margin: -0.3rem 0 0.7rem 0; }

/* Asset card rows: compact key/value lines */
.kv-row { display: grid; grid-template-columns: 1fr auto; gap: 0.6rem;
          padding: 0.4rem 0; border-bottom: 1px dashed var(--cte-border-soft); }
.kv-row:last-child { border-bottom: 0; }
.kv-row .k { color: var(--cte-muted); font-size: 0.85rem; }
.kv-row .v { color: var(--cte-text); font-size: 0.92rem; font-weight: 600; }
.kv-row .v.pos { color: var(--cte-pos); }
.kv-row .v.neg { color: var(--cte-neg); }

/* Tabs — explicit underline indicator on the active tab so users can
   actually see which panel is showing. */
[data-baseweb="tab-list"] {
  border-bottom: 1px solid var(--cte-border) !important;
  gap: 0.4rem;
}
[data-baseweb="tab"] {
  color: var(--cte-muted) !important;
  font-weight: 600 !important;
  font-size: 0.93rem !important;
  padding: 0.55rem 0.9rem !important;
  background: transparent !important;
  border-radius: 8px 8px 0 0 !important;
}
[data-baseweb="tab"]:hover { color: var(--cte-text) !important; }
[data-baseweb="tab"][aria-selected="true"] {
  color: var(--cte-text) !important;
  background: rgba(56,189,248,0.06) !important;
  box-shadow: inset 0 -2px 0 var(--cte-accent);
}
[data-baseweb="tab-highlight"] { background: var(--cte-accent) !important; }
[data-baseweb="tab-panel"] { padding-top: 0.9rem !important; }

/* Selectbox & expander — give them visible borders on the dark surface. */
[data-baseweb="select"] > div {
  background: var(--cte-card-2) !important;
  border: 1px solid var(--cte-border) !important;
  border-radius: 8px !important;
  color: var(--cte-text) !important;
}
[data-baseweb="select"] svg { color: var(--cte-muted) !important; }
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
  background: rgba(56,189,248,0.12) !important;
  color: var(--cte-accent) !important;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
  background: var(--cte-card) !important;
  color: var(--cte-text) !important;
  border: 1px solid var(--cte-border) !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  transition: border-color 120ms, background 120ms, transform 80ms;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--cte-accent) !important;
  background: rgba(56,189,248,0.06) !important;
}
.stButton > button[kind="primary"] {
  background: var(--cte-accent) !important;
  color: #0a0e1a !important;
  border-color: var(--cte-accent) !important;
}
.stButton > button[kind="primary"]:hover {
  background: #0ea5e9 !important; border-color: #0ea5e9 !important;
}
.stButton > button:disabled, .stDownloadButton > button:disabled {
  opacity: 0.45 !important;
}

/* Tables — DO NOT set overflow:hidden on the dataframe container, it clips
   the virtualised scrolling region and hides the rows. Border only. */
[data-testid="stDataFrame"] {
  border: 1px solid var(--cte-border);
  border-radius: 10px;
}
[data-testid="stDataFrame"] [role="grid"] {
  background: var(--cte-card-2) !important;
}

/* Inline alerts */
[data-testid="stAlert"] {
  border-radius: 10px !important;
  border: 1px solid var(--cte-border-soft) !important;
}

/* Subtle separator */
hr { border-color: var(--cte-border) !important; }

/* Footer */
.cte-footer { color: var(--cte-muted); font-size: 0.8rem; text-align: center;
              margin-top: 1.4rem; padding: 0.8rem 0;
              border-top: 1px solid var(--cte-border); }
</style>
"""
st.markdown(_CTE_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_candles_cached(asset: str, timeframe: str, mtime: float) -> pd.DataFrame:
    return data_collector.load_candles(asset, timeframe)


def _csv_mtime(asset: str, timeframe: str) -> float:
    p = utils.csv_path_for(asset, timeframe)
    return p.stat().st_mtime if p.exists() else 0.0


def _safe_load_candles(asset: str, timeframe: str) -> Optional[pd.DataFrame]:
    try:
        return _load_candles_cached(asset, timeframe, _csv_mtime(asset, timeframe))
    except FileNotFoundError:
        return None


def _read_artifact(name: str, subdir: str = "results") -> Optional[pd.DataFrame]:
    base = config.RESULTS_DIR if subdir == "results" else config.LOGS_DIR
    p = base / name
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def _read_meta() -> Optional[Dict]:
    p = config.RESULTS_DIR / "backtest_meta.json"
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _fmt_money(v: float, decimals: int = 2) -> str:
    return f"{v:,.{decimals}f}"


def _fmt_pct(v: float, decimals: int = 2, signed: bool = True) -> str:
    fmt = f"{{:+.{decimals}f}}%" if signed else f"{{:.{decimals}f}}%"
    return fmt.format(v)


def _sign_class(v: float) -> str:
    if v > 0: return "pos"
    if v < 0: return "neg"
    return ""


def _short_iso(s: Optional[str]) -> str:
    if not s: return "—"
    try:
        return pd.to_datetime(s, utc=True).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(s)


def _now_utc_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _queue_flash(kind: str, text: str) -> None:
    """Store a one-shot message that survives the next st.rerun() and is
    displayed once at the top of the main area before being cleared."""
    st.session_state["flash_message"] = (kind, text)


# Initialise the dashboard-refresh timestamp once per browser session.
if "last_dashboard_refresh_iso" not in st.session_state:
    st.session_state["last_dashboard_refresh_iso"] = _now_utc_iso()


# ---------------------------------------------------------------------------
# Sidebar — Status block + controls grouped into expandable sections
# ---------------------------------------------------------------------------

# Look up status-block facts before sidebar renders. We'll re-read meta later
# in the script, but the sidebar status block needs them up-front.
_meta_for_status = _read_meta()
_status_last_run_iso = (_meta_for_status or {}).get("run_timestamp_iso")
_status_fresh_iso = st.session_state.get("fresh_run_iso")
if not _meta_for_status:
    _status_source = ("amber", "No backtest yet")
elif _status_fresh_iso and _status_fresh_iso == _status_last_run_iso:
    _status_source = ("pos", "Fresh this session")
else:
    _status_source = ("grey", "Loaded from saved files")

_status_pill_class = {
    "pos": "pill-pos", "amber": "pill-amber", "grey": "pill-grey",
}[_status_source[0]]

st.sidebar.markdown(
    f"""
    <div style='background:rgba(15,23,42,0.7);border:1px solid #1f2937;
                border-radius:10px;padding:0.7rem 0.85rem;margin-bottom:0.8rem;'>
      <div style='font-size:0.72rem;color:#9ca3af;text-transform:uppercase;
                  letter-spacing:0.07em;font-weight:700;'>App status</div>
      <div style='margin-top:0.45rem;font-size:0.82rem;color:#e5e7eb;'>
        <div><span style='color:#9ca3af;'>Last refresh:</span>
             {_short_iso(st.session_state.get('last_dashboard_refresh_iso'))}</div>
        <div style='margin-top:0.2rem;'><span style='color:#9ca3af;'>Last backtest:</span>
             {_short_iso(_status_last_run_iso) if _status_last_run_iso else 'no run yet'}</div>
        <div style='margin-top:0.5rem;'>
          <span class='pill {_status_pill_class}'>
            <span class='pill-dot'></span>{_status_source[1]}
          </span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Simple ↔ Advanced mode toggle ---------------------------------
# Simple mode hides every technical slider behind a small set of presets so
# a non-technical user can run a research test in 3 clicks. Advanced mode
# preserves every existing control with no regression.
DASHBOARD_MODE = st.sidebar.radio(
    "Dashboard mode",
    options=["Simple", "Advanced"],
    horizontal=True,
    help=("Simple hides technical sliders behind presets. Advanced shows "
          "every parameter exactly as before — useful for tuning."),
)
SIMPLE_MODE = (DASHBOARD_MODE == "Simple")

# ---------- Risk presets (Simple Mode) ------------------------------------
# Each preset maps to a dict of risk-engine parameter values. They MUST be
# named "research" so the user does not read this as a trading recommendation.
RISK_PRESETS: Dict[str, Dict[str, float]] = {
    "Conservative research": {
        "max_position_pct": 0.05, "risk_per_trade_pct": 0.005,
        "max_daily_loss_pct": 0.01, "stop_loss_pct": 0.05,
    },
    "Balanced research": {
        "max_position_pct": 0.10, "risk_per_trade_pct": 0.01,
        "max_daily_loss_pct": 0.02, "stop_loss_pct": 0.05,
    },
    "Aggressive research": {
        "max_position_pct": 0.20, "risk_per_trade_pct": 0.02,
        "max_daily_loss_pct": 0.04, "stop_loss_pct": 0.08,
    },
}
# Single short caption shared by every preset — keeps the sidebar terse.
RISK_PRESET_CAPTION = "Research setting only. Not a trading recommendation."
STRATEGY_CAPTION = "Rule-based strategy used for this backtest."

# (key, display label) — the third "long description" field was removed in the
# decluttering patch. The Research Lab is where strategies are described in
# detail; the sidebar dropdown only needs to name them.
STRATEGY_OPTIONS: List[Tuple[str, str]] = [
    ("rsi_ma_atr", "Current RSI trend strategy"),
    ("ma_cross", "Moving average trend strategy"),
    ("breakout", "Breakout strategy"),
    ("buy_and_hold", "Buy and hold baseline"),
]

st.sidebar.markdown(
    "<div style='font-size:1.05rem;font-weight:700;color:#e5e7eb;"
    "padding:0.2rem 0 0.6rem 0;'>Controls</div>",
    unsafe_allow_html=True,
)

with st.sidebar.expander("Assets & timeframe", expanded=True):
    asset_choice = st.multiselect(
        "Assets",
        options=config.ASSETS, default=config.ASSETS,
        help=("Which crypto assets to test. Both BTC/USDT and ETH/USDT "
              "are selected by default."),
    )
    timeframe = st.selectbox(
        "Timeframe",
        options=config.TIMEFRAMES,
        index=config.TIMEFRAMES.index(config.DEFAULT_TIMEFRAME),
        help=("How long each candle covers. 1h = hourly bars, "
              "4h = 4-hour bars, 1d = daily bars. Shorter timeframes "
              "produce more trades but more noise."),
    )

if SIMPLE_MODE:
    # ---------- Simple Mode controls --------------------------------------
    with st.sidebar.expander("Money & risk", expanded=True):
        starting_capital = st.number_input(
            "Starting capital (USDT)",
            min_value=100.0, max_value=10_000_000.0,
            value=10_000.0, step=100.0,
            help=("How much simulated money the strategy starts with. "
                  "Bigger numbers do not change strategy behaviour, only "
                  "the dollar amounts in the result."),
        )
        risk_preset_label = st.selectbox(
            "Risk preset",
            options=list(RISK_PRESETS.keys()),
            index=1,  # Balanced research
            help=("Bundle of position-size, daily-loss and stop-loss "
                  "settings. Conservative is smallest, Aggressive is "
                  "largest. Research configurations only — not trading "
                  "recommendations."),
        )
        st.caption(RISK_PRESET_CAPTION)
        # Pull preset values into the same variable names the rest of the
        # script already uses, so the action handlers don't need to care
        # which mode is active.
        _preset = RISK_PRESETS[risk_preset_label]
        max_position_pct = _preset["max_position_pct"]
        risk_per_trade_pct = _preset["risk_per_trade_pct"]
        max_daily_loss_pct = _preset["max_daily_loss_pct"]
        stop_loss_pct = _preset["stop_loss_pct"]
        # Fees and slippage stay at the repo defaults in Simple Mode.
        fee_pct = float(config.RISK.fee_pct)
        slippage_pct = float(config.RISK.slippage_pct)

    with st.sidebar.expander("Strategy style", expanded=True):
        strategy_label_to_key = {label: key for key, label in STRATEGY_OPTIONS}
        strategy_label = st.selectbox(
            "Strategy",
            options=[label for _, label in STRATEGY_OPTIONS],
            index=0,
            help=("Which trading rule to use. Risk engine, fees and "
                  "slippage are identical across strategies — only the "
                  "buy/sell logic differs."),
        )
        strategy_key = strategy_label_to_key[strategy_label]
        st.caption(STRATEGY_CAPTION)
        # Strategy parameters in Simple Mode = repo defaults.
        rsi_buy = int(config.STRATEGY.rsi_buy_threshold)
        rsi_sell = int(config.STRATEGY.rsi_sell_threshold)
        atr_pct_max = float(config.STRATEGY.atr_pct_max)

    st.sidebar.caption("Switch to Advanced for RSI, ATR, fees, slippage, stop-loss.")
else:
    # ---------- Advanced Mode controls — every original control intact ----
    with st.sidebar.expander("Strategy parameters", expanded=False):
        rsi_buy = st.slider(
            "RSI buy threshold", 5, 50, int(config.STRATEGY.rsi_buy_threshold),
            help=("RSI is a momentum score from 0 to 100. Lower numbers "
                  "mean the asset may be oversold. A buy threshold of 35 "
                  "means the strategy looks for weakness before buying."),
        )
        rsi_sell = st.slider(
            "RSI sell threshold", 50, 95, int(config.STRATEGY.rsi_sell_threshold),
            help=("Higher RSI means price may be overheated. A sell "
                  "threshold of 65 means the strategy exits when momentum "
                  "looks stretched."),
        )
        atr_pct_max = st.slider(
            "ATR % max (skip new entries above)", 1.0, 15.0,
            float(config.STRATEGY.atr_pct_max), 0.5,
            help=("ATR measures volatility. A high ATR means price is "
                  "moving violently. This filter avoids opening trades "
                  "when the market is too unstable."),
        )

    with st.sidebar.expander("Risk parameters", expanded=False):
        starting_capital = st.number_input(
            "Starting capital (USDT)",
            min_value=100.0, max_value=10_000_000.0,
            value=float(config.RISK.starting_capital), step=100.0,
            help="How much simulated money the strategy starts with.",
        )
        max_position_pct = st.slider(
            "Max position size (%)", 1.0, 25.0,
            float(config.RISK.max_position_pct * 100), 0.5,
            help="The biggest share of the portfolio that can go into one asset.",
        ) / 100.0
        risk_per_trade_pct = st.slider(
            "Risk per trade (%)", 0.1, 5.0,
            float(config.RISK.risk_per_trade_pct * 100), 0.1,
            help="The maximum amount the strategy is allowed to lose on one trade.",
        ) / 100.0
        max_daily_loss_pct = st.slider(
            "Max daily loss (%)", 0.5, 10.0,
            float(config.RISK.max_daily_loss_pct * 100), 0.1,
            help=("If simulated losses hit this level in one day, the "
                  "system stops opening new trades."),
        ) / 100.0
        fee_pct = st.number_input(
            "Fee (% per trade)", min_value=0.0, max_value=2.0,
            value=float(config.RISK.fee_pct * 100), step=0.01, format="%.3f",
            help="Estimated exchange trading cost per trade side.",
        ) / 100.0
        slippage_pct = st.number_input(
            "Slippage (% per trade)", min_value=0.0, max_value=2.0,
            value=float(config.RISK.slippage_pct * 100), step=0.01, format="%.3f",
            help=("Estimated price difference between expected and actual "
                  "execution price."),
        ) / 100.0
        stop_loss_pct = st.slider(
            "Stop-loss distance (%)", 1.0, 25.0,
            float(config.RISK.stop_loss_pct * 100), 0.5,
            help=("How far price can move against the trade before the "
                  "simulated position is closed."),
        ) / 100.0

    # In Advanced mode we keep the existing strategy (RSI/MA/ATR) — same as before.
    strategy_key = "rsi_ma_atr"
    strategy_label = "Current RSI trend strategy"

with st.sidebar.expander("Actions", expanded=True):
    do_backtest = st.button(
        "Run backtest", type="primary", use_container_width=True,
        help="Run the backtester for the selected assets and timeframe.",
    )
    do_refresh_dashboard = st.button(
        "Refresh dashboard", use_container_width=True,
        help=("Reload saved artifacts from disk and refresh UI state. "
              "Use this if numbers look stale. Does not re-download data."),
    )
    do_refresh_data = st.button(
        "Refresh market data", use_container_width=True,
        help="Re-download OHLCV candles from the exchange (public endpoints only).",
    )
    do_paper_tick = st.button(
        "Run paper tick", use_container_width=True,
        help="Run one paper-trade evaluation on the latest candle.",
    )
    do_clear = st.button(
        "Clear saved results", use_container_width=True,
        help=("Delete saved backtest artifacts. Cached candles and "
              "paper-trader state are not touched."),
    )

with st.sidebar.expander("Research lab", expanded=False):
    do_run_research = st.button(
        "Run research lab", use_container_width=True,
        help=("Run timeframe comparison, walk-forward, strategy comparison, "
              "robustness checks, and Monte Carlo. Saves CSVs to results/."),
    )
    do_refresh_research = st.button(
        "Refresh research results", use_container_width=True,
        help="Reload saved research CSVs from disk into the dashboard.",
    )
    research_tfs = st.multiselect(
        "Timeframes to research", options=config.TIMEFRAMES,
        default=[config.DEFAULT_TIMEFRAME],
        help="Pick which cached timeframes the research lab should use.",
    )

st.sidebar.caption("Paper only. No live trading.")


# ---------------------------------------------------------------------------
# Action handlers (logic identical to previous version)
# ---------------------------------------------------------------------------
def _build_runtime_configs():
    strat_cfg = replace(
        config.STRATEGY,
        rsi_buy_threshold=float(rsi_buy),
        rsi_sell_threshold=float(rsi_sell),
        atr_pct_max=float(atr_pct_max),
    )
    risk_cfg = replace(
        config.RISK,
        starting_capital=float(starting_capital),
        max_position_pct=float(max_position_pct),
        risk_per_trade_pct=float(risk_per_trade_pct),
        max_daily_loss_pct=float(max_daily_loss_pct),
        fee_pct=float(fee_pct),
        slippage_pct=float(slippage_pct),
        stop_loss_pct=float(stop_loss_pct),
    )
    return strat_cfg, risk_cfg


if do_refresh_dashboard:
    # Reload saved artifacts + clear caches; do NOT re-download market data.
    try:
        _load_candles_cached.clear()
    except Exception:
        pass
    try:
        st.cache_data.clear()
    except Exception:
        pass
    st.session_state["last_dashboard_refresh_iso"] = _now_utc_iso()
    _queue_flash("success", "Dashboard refreshed from saved artifacts.")
    st.rerun()

if do_refresh_data:
    if not asset_choice:
        _queue_flash("warning", "Pick at least one asset to refresh.")
        st.rerun()
    else:
        with st.spinner("Downloading candles…"):
            try:
                paths = data_collector.download_all(
                    assets=asset_choice, timeframes=[timeframe], refresh=True,
                )
                try:
                    _load_candles_cached.clear()
                except Exception:
                    pass
                _queue_flash(
                    "success",
                    f"Market data refreshed successfully — {len(paths)} dataset(s).",
                )
                st.rerun()
            except Exception as e:
                _queue_flash("error", f"Market data refresh failed: {e}")
                st.rerun()

if do_backtest:
    if not asset_choice:
        _queue_flash("warning", "Pick at least one asset to backtest.")
        st.rerun()
    else:
        strat_cfg, risk_cfg = _build_runtime_configs()
        with st.spinner("Running backtest…"):
            try:
                data_collector.download_all(
                    assets=asset_choice, timeframes=[timeframe], refresh=False,
                )
                # Resolve the chosen strategy (Simple Mode lets the user
                # pick; Advanced Mode keeps the incumbent RSI/MA/ATR for
                # back-compat). The risk engine, fees, slippage, and fill
                # model are identical across strategies — see src/strategies.
                from src.strategies import REGISTRY as _STRATEGY_REGISTRY
                _strategy_cls = _STRATEGY_REGISTRY.get(
                    strategy_key, _STRATEGY_REGISTRY["rsi_ma_atr"]
                )
                art = backtester.run_backtest(
                    assets=asset_choice, timeframe=timeframe,
                    risk_cfg=risk_cfg, strat_cfg=strat_cfg, save=True,
                    strategy=_strategy_cls(),
                )
                metrics = performance.compute_metrics(
                    art.equity_curve, art.trades, art.asset_close_curves,
                    starting_capital=risk_cfg.starting_capital,
                )
                performance.save_metrics(metrics)
                pa_df = performance.per_asset_metrics(
                    art.trades, art.asset_close_curves,
                    starting_capital=risk_cfg.starting_capital,
                    equity_curve=art.equity_curve,
                )
                performance.save_per_asset_metrics(pa_df)
                st.session_state["fresh_run_iso"] = art.meta.get("run_timestamp_iso")
                st.session_state["last_dashboard_refresh_iso"] = _now_utc_iso()
                _queue_flash(
                    "success",
                    f"Backtest completed successfully — final equity "
                    f"{metrics.final_portfolio_value:,.2f} USDT.",
                )
                st.rerun()
            except Exception as e:
                _queue_flash("error", f"Backtest failed: {e}")
                st.rerun()

if do_paper_tick:
    with st.spinner("Running paper tick…"):
        try:
            paper_trader.run_tick(
                timeframe=timeframe, assets=asset_choice, refresh=True,
            )
            _queue_flash("success", "Paper tick completed.")
            st.rerun()
        except Exception as e:
            _queue_flash("error", f"Paper tick failed: {e}")
            st.rerun()

if do_clear:
    targets: List[Path] = [
        config.RESULTS_DIR / "equity_curve.csv",
        config.RESULTS_DIR / "summary_metrics.csv",
        config.RESULTS_DIR / "per_asset_metrics.csv",
        config.RESULTS_DIR / "backtest_meta.json",
        config.LOGS_DIR / "trades.csv",
        config.LOGS_DIR / "decisions.csv",
    ]
    targets += list(config.RESULTS_DIR.glob("price_*.csv"))
    removed = 0
    errors: List[str] = []
    for p in targets:
        try:
            if p.exists():
                p.unlink()
                removed += 1
        except Exception as e:
            errors.append(f"{p.name}: {e}")
    st.session_state.pop("fresh_run_iso", None)
    try:
        _load_candles_cached.clear()
    except Exception:
        pass
    st.session_state["last_dashboard_refresh_iso"] = _now_utc_iso()
    if errors:
        _queue_flash(
            "error",
            f"Cleared {removed} file(s). Errors: {'; '.join(errors)}",
        )
    else:
        _queue_flash(
            "success",
            f"Saved results cleared ({removed} file(s)). "
            "Run a new backtest.",
        )
    st.rerun()

if do_run_research:
    if not research_tfs:
        _queue_flash("warning",
                     "Pick at least one timeframe in the Research lab section.")
        st.rerun()
    else:
        with st.spinner("Running research lab… this may take a minute."):
            try:
                research.run_all(
                    assets=tuple(asset_choice or config.ASSETS),
                    timeframes=tuple(research_tfs),
                )
                st.session_state["last_research_run_iso"] = _now_utc_iso()
                _queue_flash("success",
                             "Research lab completed — results saved to "
                             "results/research_*.csv.")
                st.rerun()
            except Exception as e:
                _queue_flash("error", f"Research lab failed: {e}")
                st.rerun()

if do_refresh_research:
    _queue_flash("success", "Research results reloaded from saved CSVs.")
    st.rerun()


# ---------------------------------------------------------------------------
# Load latest artifacts
# ---------------------------------------------------------------------------
def _df_or_empty(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    return df if df is not None else pd.DataFrame()


equity_df = _df_or_empty(_read_artifact("equity_curve.csv"))
trades_df = _df_or_empty(_read_artifact("trades.csv", subdir="logs"))
decisions_df = _df_or_empty(_read_artifact("decisions.csv", subdir="logs"))
metrics_df = _df_or_empty(_read_artifact("summary_metrics.csv"))
per_asset_df = _df_or_empty(_read_artifact("per_asset_metrics.csv"))
meta = _read_meta()

meta_assets: List[str] = list((meta or {}).get("assets") or [])
meta_timeframe = str((meta or {}).get("timeframe") or timeframe)
last_run_iso = (meta or {}).get("run_timestamp_iso")

fresh_iso = st.session_state.get("fresh_run_iso")
is_fresh = bool(fresh_iso) and fresh_iso == last_run_iso


# ---------------------------------------------------------------------------
# One-shot flash message (queued by an action handler before st.rerun)
# ---------------------------------------------------------------------------
_flash = st.session_state.pop("flash_message", None)
if _flash:
    _kind, _text = _flash
    if _kind == "success":   st.success(_text, icon="✅")
    elif _kind == "error":   st.error(_text, icon="🚫")
    elif _kind == "warning": st.warning(_text, icon="⚠️")
    else:                    st.info(_text)


# ---------------------------------------------------------------------------
# Stale-state banner — compare current sidebar controls against the saved
# backtest metadata. If anything differs, show ONE prominent warning so the
# user can never silently look at stale numbers without noticing.
# ---------------------------------------------------------------------------
def _stale_state_mismatches() -> List[str]:
    if meta is None:
        return []
    rows: List[str] = []

    sidebar_assets = sorted(set(asset_choice or []))
    saved_assets = sorted(set(meta_assets))
    if sidebar_assets and sidebar_assets != saved_assets:
        rows.append(
            f"**Assets** — selected `{sidebar_assets}` vs displayed "
            f"`{saved_assets}`"
        )

    saved_tf = str(meta.get("timeframe") or "")
    if saved_tf and timeframe != saved_tf:
        rows.append(
            f"**Timeframe** — selected `{timeframe}` vs displayed `{saved_tf}`"
        )

    try:
        saved_capital = float(meta.get("starting_capital") or 0.0)
        if saved_capital and abs(float(starting_capital) - saved_capital) > 1e-6:
            rows.append(
                f"**Starting capital** — selected "
                f"`{float(starting_capital):,.2f} USDT` vs displayed "
                f"`{saved_capital:,.2f} USDT`"
            )
    except (TypeError, ValueError):
        pass

    saved_strategy = str(meta.get("strategy_name") or "")
    if saved_strategy and strategy_key and strategy_key != saved_strategy:
        rows.append(
            f"**Strategy** — selected `{strategy_key}` vs displayed "
            f"`{saved_strategy}`"
        )
    return rows


_mismatches = _stale_state_mismatches()
if _mismatches:
    st.warning(
        "**Current controls do not match the displayed result. "
        "Click Run backtest to update.**\n\n"
        + "\n".join(f"- {row}" for row in _mismatches),
        icon="⚠️",
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
last_run_label = _short_iso(last_run_iso) if last_run_iso else "no run yet"
fresh_pill = (
    "<span class='pill pill-pos'><span class='pill-dot'></span>Fresh</span>"
    if is_fresh and last_run_iso else
    "<span class='pill pill-grey'><span class='pill-dot'></span>Saved</span>"
    if last_run_iso else ""
)
st.markdown(
    f"""
    <div class="cte-header">
      <div>
        <div class="cte-title">Crypto Trading Engine</div>
        <div class="cte-subtitle">Research-only backtesting and paper-trading dashboard for BTC and ETH</div>
      </div>
      <div class="cte-pills">
        <span class="pill pill-blue"><span class="pill-dot"></span>Paper only</span>
        <span class="pill pill-amber"><span class="pill-dot"></span>No live trading</span>
        {fresh_pill}
        <span class="pill pill-grey">Last backtest: {last_run_label}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Quick guide — collapsed by default; opens to 4 short bullets only.
# ---------------------------------------------------------------------------
with st.expander("Quick guide", expanded=False):
    st.markdown(
        "- This is a research dashboard, not live trading.\n"
        "- *Strategy vs buy and hold* shows whether trading added value.\n"
        "- Low drawdown can be misleading if the strategy barely trades.\n"
        "- Run **Research Lab** for stronger validation."
    )


# ---------------------------------------------------------------------------
# Hero metrics — Final portfolio / Strategy return / B&H / vs B&H
# ---------------------------------------------------------------------------
def _hero_card(label: str, value: str, context: str,
               variant: str = "", value_class: str = "") -> str:
    return (
        f"<div class='hero-card {variant}'>"
        f"  <div class='hero-label'>{label}</div>"
        f"  <div class='hero-value {value_class}'>{value}</div>"
        f"  <div class='hero-context'>{context}</div>"
        f"</div>"
    )


if metrics_df.empty:
    st.info(
        "No backtest run yet. Open **Actions** in the sidebar and click "
        "**Run backtest** to populate the dashboard.",
        icon="ℹ️",
    )
else:
    m = metrics_df.iloc[0].to_dict()
    final_v = float(m["final_portfolio_value"])
    start_v = float(m["starting_capital"])
    tot_ret = float(m["total_return_pct"])
    bh_ret = float(m["buy_and_hold_return_pct"])
    vs_bh = float(m["strategy_vs_bh_pct"])

    cards = [
        _hero_card(
            "Final portfolio",
            f"{_fmt_money(final_v)} USDT",
            f"Starting capital {_fmt_money(start_v, 0)} USDT",
            variant=_sign_class(tot_ret),
        ),
        _hero_card(
            "Strategy total return",
            _fmt_pct(tot_ret),
            f"Net of fees & slippage over the tested window",
            variant=_sign_class(tot_ret),
            value_class=_sign_class(tot_ret),
        ),
        _hero_card(
            "Buy & hold benchmark",
            _fmt_pct(bh_ret),
            "Equal-weight allocation across selected assets",
            variant=_sign_class(bh_ret),
            value_class=_sign_class(bh_ret),
        ),
        _hero_card(
            "Strategy vs B&H",
            _fmt_pct(vs_bh),
            "Outperformed benchmark" if vs_bh > 0 else
            "Underperformed benchmark" if vs_bh < 0 else
            "Matched benchmark",
            variant=_sign_class(vs_bh),
            value_class=_sign_class(vs_bh),
        ),
    ]
    st.markdown(f"<div class='hero-grid'>{''.join(cards)}</div>",
                unsafe_allow_html=True)

    # ---- Interpretation card (max 3 bullets) ----------------------------
    def _interpret(m: dict) -> Tuple[str, List[str]]:
        tot = float(m.get("total_return_pct", 0.0))
        bh = float(m.get("buy_and_hold_return_pct", 0.0))
        diff = float(m.get("strategy_vs_bh_pct", 0.0))
        n_trades = int(m.get("num_trades", 0))
        exposure = float(m.get("exposure_time_pct", 0.0))

        bits: List[str] = [
            f"Strategy returned **{tot:+.2f}%** vs buy and hold **{bh:+.2f}%**.",
        ]
        if n_trades < 10 or exposure < 10:
            bits.append(
                f"Only **{n_trades}** closed trade(s) and **{exposure:.1f}%** "
                f"exposure — evidence is statistically weak."
            )

        if diff < -1 or n_trades < 10:
            verdict = "**Verdict: not worth trading as configured.**"
        elif diff > 1 and n_trades >= 10:
            verdict = "**Verdict: encouraging in this window — needs Research Lab validation.**"
        else:
            verdict = "**Verdict: inconclusive — run Research Lab.**"
        return verdict, bits

    with st.container(border=True):
        st.markdown(
            "<div class='section-h'><span class='dot'></span>"
            "Interpretation</div>",
            unsafe_allow_html=True,
        )
        verdict_md, bits_md = _interpret(m)
        for line in bits_md:
            st.markdown(f"- {line}")
        st.markdown(verdict_md)


# ---------------------------------------------------------------------------
# Backtest scope card
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("<div class='section-h'><span class='dot'></span>Backtest scope</div>",
                unsafe_allow_html=True)
    if meta is None:
        st.markdown(
            "<div class='section-sub'>No saved scope. Run a backtest to "
            "populate this card.</div>",
            unsafe_allow_html=True,
        )
    else:
        scope_items = [
            ("Selected assets", ", ".join(meta_assets) if meta_assets else "—"),
            ("Timeframe", meta_timeframe),
            ("Candles used", str(int((meta or {}).get("num_candles_used") or 0))),
            ("Date range",
             f"{_short_iso((meta or {}).get('first_candle_iso'))}"
             f" → {_short_iso((meta or {}).get('last_candle_iso'))}"),
            ("Starting capital",
             f"{_fmt_money(float((meta or {}).get('starting_capital') or 0.0))} USDT"),
            ("Result source",
             "Fresh (this session)" if is_fresh else "Loaded from saved files"),
            ("Last run", _short_iso(last_run_iso)),
            ("Fill model",
             "Next-bar open (honest)" if not (meta or {}).get("fill_on_signal_close")
             else "Same-bar close (optimistic)"),
        ]
        cells = "".join(
            f"<div class='scope-cell'><div class='l'>{k}</div>"
            f"<div class='v'>{v}</div></div>"
            for k, v in scope_items
        )
        st.markdown(f"<div class='scope-grid'>{cells}</div>",
                    unsafe_allow_html=True)

    # The scope mismatch banner is rendered higher up in the page (above the
    # hero metrics) so the user spots stale state before reading the numbers.


# ---------------------------------------------------------------------------
# Per-asset cards + Combined portfolio card
# ---------------------------------------------------------------------------
def _kv(k: str, v: str, sign: str = "") -> str:
    return f"<div class='kv-row'><div class='k'>{k}</div><div class='v {sign}'>{v}</div></div>"


def _per_asset_card_html(asset: str, row: pd.Series) -> str:
    realized = float(row["realized_pnl"])
    rr_alloc = float(row["realized_return_on_allocation_pct"])
    bh = float(row["buy_and_hold_return_pct"])
    vs_bh = float(row["strategy_vs_bh_pct"])
    rows_html = "".join([
        _kv("Realized P&L",
            f"{_fmt_money(realized)} USDT", _sign_class(realized)),
        _kv("Return on allocation",
            _fmt_pct(rr_alloc), _sign_class(rr_alloc)),
        _kv("Buy & hold", _fmt_pct(bh), _sign_class(bh)),
        _kv("Strategy vs B&H", _fmt_pct(vs_bh), _sign_class(vs_bh)),
        _kv("Win rate",
            f"{float(row['win_rate_pct']):.1f}% "
            f"<span style='color:var(--cte-muted);font-weight:500;'>"
            f"({int(row['num_trades'])} trades)</span>"),
        _kv("Profit factor", f"{float(row['profit_factor']):.2f}"),
        _kv("Fees paid", f"{_fmt_money(float(row['fees_paid']))} USDT"),
        _kv("Allocated capital",
            f"{_fmt_money(float(row['allocated_capital']))} USDT"),
    ])
    return rows_html


if not metrics_df.empty:
    n_assets = len(meta_assets)

    if n_assets >= 2 and not per_asset_df.empty:
        cols = st.columns(min(n_assets, 2))
        for i, asset in enumerate(meta_assets[:2]):
            with cols[i]:
                with st.container(border=True):
                    st.markdown(
                        f"<div class='section-h'><span class='dot'></span>"
                        f"{asset} Results</div>",
                        unsafe_allow_html=True,
                    )
                    rows = per_asset_df[per_asset_df["asset"] == asset]
                    if rows.empty:
                        st.markdown(
                            "<div class='section-sub'>No per-asset data — "
                            "re-run backtest.</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            _per_asset_card_html(asset, rows.iloc[0]),
                            unsafe_allow_html=True,
                        )
    elif n_assets == 1:
        only = meta_assets[0]
        with st.container(border=True):
            st.markdown(
                f"<div class='section-h'><span class='dot'></span>"
                f"{only} Results</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='section-sub'>Single asset selected — combined "
                "portfolio metrics below ARE the per-asset results.</div>",
                unsafe_allow_html=True,
            )

    # Combined portfolio card
    m = metrics_df.iloc[0].to_dict()
    with st.container(border=True):
        st.markdown(
            "<div class='section-h'><span class='dot'></span>"
            "Combined portfolio</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='section-sub'>Portfolio-level metrics use shared "
            "cash across selected assets.</div>",
            unsafe_allow_html=True,
        )
        c = st.columns(5)
        c[0].metric(
            "Final value",
            f"{_fmt_money(float(m['final_portfolio_value']), 0)} USDT",
            _fmt_pct(float(m['total_return_pct'])),
            help=("Where the simulated portfolio ended up. The delta below "
                  "is the total percentage return over the tested window."),
        )
        c[1].metric(
            "Max drawdown", _fmt_pct(float(m['max_drawdown_pct'])),
            help=("The biggest peak-to-trough drop the portfolio suffered. "
                  "A small drawdown can be misleading if the strategy "
                  "barely traded."),
        )
        c[2].metric(
            "Win rate", f"{float(m['win_rate_pct']):.1f}%",
            f"{int(m['num_trades'])} round-trips",
            help=("Share of closed trades that finished green. With fewer "
                  "than 10 trades this number is statistically very weak."),
        )
        c[3].metric(
            "Profit factor", f"{float(m['profit_factor']):.2f}",
            help=("Total profit divided by total loss. Above 1 means "
                  "profitable before context. Below 1 is bad."),
        )
        c[4].metric(
            "Fees paid", f"{_fmt_money(float(m['fees_paid']))} USDT",
            help="Sum of all simulated exchange fees over the test window.",
        )
        c2 = st.columns(5)
        c2[0].metric(
            "Sharpe", f"{float(m['sharpe_ratio']):.2f}",
            help=("Return compared to volatility. Higher is better. "
                  "Unreliable with too few trades."),
        )
        c2[1].metric(
            "Sortino", f"{float(m['sortino_ratio']):.2f}",
            help=("Like Sharpe, but focuses more on downside volatility "
                  "(big losses count more than big gains)."),
        )
        c2[2].metric(
            "Calmar", f"{float(m['calmar_ratio']):.2f}",
            help="Return compared to max drawdown. Higher is better.",
        )
        c2[3].metric(
            "Slippage cost",
            f"{_fmt_money(float(m['slippage_cost']))} USDT",
            help=("Sum of estimated price-impact costs across all "
                  "simulated fills."),
        )
        c2[4].metric(
            "Exposure time",
            f"{float(m.get('exposure_time_pct', 0.0)):.1f}%",
            help=("Percentage of time the strategy actually had money in "
                  "the market. Low exposure means low risk AND no real "
                  "test of an edge."),
        )

        if float(m.get("strategy_vs_bh_pct", 0.0)) < 0:
            st.warning(
                "Strategy underperformed buy and hold over this exact "
                "tested window."
            )


# ---------------------------------------------------------------------------
# Charts — inside one card with internal tabs
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        "<div class='section-h'><span class='dot'></span>Charts</div>",
        unsafe_allow_html=True,
    )
    chart_tabs = st.tabs([
        "Equity", "Drawdown", "Price + trades",
        "Cumulative P&L", "Monthly returns",
    ])

    with chart_tabs[0]:
        if equity_df.empty:
            st.info("No equity curve. Run a backtest to populate.")
        else:
            st.plotly_chart(
                plotting.equity_curve_fig(equity_df, float(starting_capital)),
                use_container_width=True,
            )
    with chart_tabs[1]:
        if equity_df.empty:
            st.info("No drawdown data. Run a backtest first.")
        else:
            st.plotly_chart(plotting.drawdown_fig(equity_df),
                            use_container_width=True)
    with chart_tabs[2]:
        if not asset_choice:
            st.info("Pick at least one asset in the sidebar.")
        else:
            sel_col, _ = st.columns([1, 3])
            with sel_col:
                which = st.selectbox(
                    "Asset for price chart",
                    options=asset_choice, key="price_asset",
                    help="Pick which asset's price + trade markers to plot.",
                )
            price_df = _read_artifact(f"price_{utils.safe_symbol(which)}.csv")
            if price_df is None:
                price_df = _safe_load_candles(which, timeframe)
            if price_df is None:
                st.info(
                    f"No price data for {which} at {timeframe}. "
                    "Click **Refresh data** in the sidebar."
                )
            else:
                if trades_df.empty:
                    st.caption(
                        f"Showing {which} closes only — no trades generated "
                        "under current settings."
                    )
                elif trades_df[trades_df["asset"] == which].empty:
                    st.caption(
                        f"Showing {which} closes only — no trades recorded "
                        f"for {which} in the last backtest."
                    )
                st.plotly_chart(
                    plotting.price_with_trades_fig(price_df, trades_df, which),
                    use_container_width=True,
                )
    with chart_tabs[3]:
        if trades_df.empty:
            st.info(
                "No cumulative P&L. Reason: no trades generated under "
                "current settings."
            )
        elif trades_df[trades_df["side"] == "SELL"].empty:
            st.info(
                "No cumulative P&L. Reason: only OPEN trades exist — no "
                "round-trips closed yet."
            )
        else:
            st.plotly_chart(plotting.cumulative_pnl_fig(trades_df),
                            use_container_width=True)
    with chart_tabs[4]:
        if equity_df.empty:
            st.info("No monthly returns. Run a backtest first.")
        elif len(equity_df) < 30:
            st.info(
                f"Not enough history — only {len(equity_df)} bars. "
                "Use a longer history window."
            )
        else:
            st.plotly_chart(plotting.monthly_returns_fig(equity_df),
                            use_container_width=True)


# ---------------------------------------------------------------------------
# Tables — Asset comparison / Trade history / Decision log / Downloads
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        "<div class='section-h'><span class='dot'></span>Tables &amp; downloads</div>",
        unsafe_allow_html=True,
    )
    tab_compare, tab_trades, tab_decisions, tab_downloads = st.tabs([
        "Asset comparison", "Trade history", "Decision log", "Downloads",
    ])

    with tab_compare:
        if per_asset_df.empty:
            st.info("Run a backtest with at least one asset to see the comparison.")
        else:
            st.markdown(
                f"<div class='section-sub'><b>{len(per_asset_df)} asset "
                f"rows loaded</b> &middot; per-asset breakdown derived from "
                f"closed trades.</div>",
                unsafe_allow_html=True,
            )
            comp_cols = [
                "asset", "allocated_capital", "realized_pnl",
                "realized_return_on_allocation_pct", "buy_and_hold_return_pct",
                "strategy_vs_bh_pct", "num_trades", "win_rate_pct",
                "profit_factor", "fees_paid", "slippage_cost",
            ]
            comp_cols = [c for c in comp_cols if c in per_asset_df.columns]
            st.dataframe(
                per_asset_df[comp_cols],
                use_container_width=True, hide_index=True,
                height=min(80 + 38 * len(per_asset_df), 320),
            )

    with tab_trades:
        if trades_df.empty:
            st.info("No trades generated under current settings.")
        else:
            display_cols = [
                "datetime_iso", "asset", "side", "price", "size",
                "fee", "slippage_cost", "realized_pnl",
                "portfolio_value", "reason",
            ]
            display_cols = [c for c in display_cols if c in trades_df.columns]
            n_assets_in_trades = int(trades_df['asset'].nunique())
            st.markdown(
                f"<div class='section-sub'><b>{len(trades_df)} trade rows "
                f"loaded</b> &middot; across {n_assets_in_trades} asset(s).</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(
                trades_df[display_cols],
                use_container_width=True, height=360, hide_index=True,
            )

    with tab_decisions:
        if decisions_df.empty:
            st.info("No decisions logged under current settings.")
        else:
            recent = decisions_df.tail(500)
            st.markdown(
                f"<div class='section-sub'><b>Showing latest {len(recent)} "
                f"of {len(decisions_df)} decisions</b> &middot; full log "
                f"available in <b>Downloads</b>.</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(
                recent, use_container_width=True, height=360, hide_index=True,
            )

    with tab_downloads:
        st.markdown(
            "<div class='section-sub'><b>Download exports</b> &middot; "
            "current saved artifacts as CSV.</div>",
            unsafe_allow_html=True,
        )
        download_specs = [
            ("Metrics", "summary_metrics.csv", metrics_df,
             "Combined-portfolio summary metrics.",
             "Run a backtest to enable.", "dl_metrics"),
            ("Trades", "trades.csv", trades_df,
             "Per-trade execution log (BUY/SELL fills).",
             "No trades to export — none generated under current settings.",
             "dl_trades"),
            ("Decisions", "decisions.csv", decisions_df,
             "Every BUY / SELL / HOLD / SKIP / REJECT decision with reason.",
             "No decisions to export — run a backtest first.",
             "dl_decisions"),
            ("Equity curve", "equity_curve.csv", equity_df,
             "Per-bar portfolio equity, cash, and exposure.",
             "Run a backtest to enable.", "dl_equity"),
            ("Asset comparison", "per_asset_metrics.csv", per_asset_df,
             "Per-asset breakdown (P&L, win rate, B&H, fees, …).",
             "Run a multi-asset backtest to enable.", "dl_compare"),
        ]
        for chunk_start in range(0, len(download_specs), 2):
            cols = st.columns(2)
            for i, spec in enumerate(download_specs[chunk_start:chunk_start + 2]):
                label, fname, df, desc, missing_msg, key = spec
                with cols[i]:
                    st.markdown(f"**{label}** &nbsp; `{fname}`")
                    st.caption(desc)
                    if df is not None and not df.empty:
                        st.download_button(
                            label=f"Download {label.lower()} CSV",
                            data=_df_to_csv_bytes(df),
                            file_name=fname,
                            mime="text/csv",
                            key=key,
                            use_container_width=True,
                        )
                    else:
                        st.button(
                            f"Download {label.lower()} CSV (unavailable)",
                            disabled=True,
                            key=key + "_disabled",
                            use_container_width=True,
                        )
                        st.caption(f":grey[{missing_msg}]")


# ---------------------------------------------------------------------------
# Research Lab — phases 1-6 surfaced as a single card with 6 tabs
# ---------------------------------------------------------------------------
def _research_csv(name: str) -> pd.DataFrame:
    return _df_or_empty(_read_artifact(name))


def _color_vs_bh(val: float) -> str:
    """Inline CSS background for the strategy_vs_bh_pct cell."""
    if pd.isna(val): return ""
    if val > 0:  return "background-color: rgba(16,185,129,0.18); color:#a7f3d0;"
    if val < 0:  return "background-color: rgba(244,63,94,0.18); color:#fecdd3;"
    return ""


def _style_vs_bh(df: pd.DataFrame, col: str = "strategy_vs_bh_pct"):
    if col not in df.columns:
        return df
    return df.style.map(_color_vs_bh, subset=[col])


with st.container(border=True):
    st.markdown(
        "<div class='section-h'><span class='dot'></span>Research lab</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-sub'>Five lenses on the strategy. "
        "Run <b>Run research lab</b> in the sidebar to populate.</div>",
        unsafe_allow_html=True,
    )

    rl_tabs = st.tabs([
        "Timeframe comparison", "Walk-forward", "Strategy comparison",
        "Robustness checks", "Monte Carlo", "Research summary",
        "Kronos confirmation",
    ])

    rl_tf_df = _research_csv("research_timeframe_comparison.csv")
    rl_wf_df = _research_csv("walk_forward_results.csv")
    rl_sc_df = _research_csv("strategy_comparison.csv")
    rl_rb_df = _research_csv("robustness_results.csv")
    rl_mc_df = _research_csv("monte_carlo_results.csv")
    rl_mcs_df = _research_csv("monte_carlo_simulations.csv")

    # One-line explainer shown at the top of each Research Lab tab.
    RL_TAB_EXPLAINERS: Dict[str, str] = {
        "timeframe": "Checks whether results hold across 1h, 4h and 1d.",
        "walk_forward": "Tests future windows instead of one big backtest.",
        "strategy": "Compares rule-based strategies using the same risk engine.",
        "robustness": "Checks whether small parameter changes break the result.",
        "monte_carlo": "Stress-tests trade-result randomness.",
        "summary": "Plain-English PASS / FAIL verdict.",
        "kronos": "Optional ML confirmation layer. Local install only.",
    }

    def _explainer(key: str) -> None:
        st.markdown(
            f"<div class='section-sub'>{RL_TAB_EXPLAINERS[key]}</div>",
            unsafe_allow_html=True,
        )

    # ---- Timeframe comparison -----------------------------------------
    with rl_tabs[0]:
        _explainer("timeframe")
        if rl_tf_df.empty:
            st.info("No timeframe-comparison results yet. Run **Run research "
                    "lab** in the sidebar.")
        else:
            ok = rl_tf_df[rl_tf_df["error"].isna()] if "error" in rl_tf_df.columns else rl_tf_df
            n_ok = len(ok)
            n_skipped = len(rl_tf_df) - n_ok
            wins = int((ok["strategy_vs_bh_pct"] > 0).sum()) if not ok.empty else 0
            st.markdown(
                f"<div class='section-sub'><b>{n_ok} runs ok · "
                f"{n_skipped} skipped</b> &middot; strategy beat B&amp;H in "
                f"<b>{wins}/{n_ok}</b> combinations.</div>",
                unsafe_allow_html=True,
            )
            low_trades = ok[ok["num_trades"] < research.MIN_TRADES_FOR_CONFIDENCE]
            if not low_trades.empty:
                st.warning(
                    f"{len(low_trades)} of {n_ok} runs have fewer than "
                    f"{research.MIN_TRADES_FOR_CONFIDENCE} closed trades — "
                    f"those rows are statistically thin.",
                )
            cols = [c for c in ["asset", "timeframe", "total_return_pct",
                    "buy_and_hold_return_pct", "strategy_vs_bh_pct",
                    "max_drawdown_pct", "win_rate_pct", "num_trades",
                    "profit_factor", "fees_paid", "exposure_time_pct",
                    "sharpe_ratio", "sortino_ratio", "calmar_ratio"]
                    if c in ok.columns]
            st.dataframe(
                _style_vs_bh(ok[cols].copy()),
                use_container_width=True, hide_index=True, height=320,
            )
            st.download_button(
                "Download research_timeframe_comparison.csv",
                _df_to_csv_bytes(rl_tf_df),
                file_name="research_timeframe_comparison.csv",
                mime="text/csv", key="dl_research_tf",
            )

    # ---- Walk-forward --------------------------------------------------
    with rl_tabs[1]:
        _explainer("walk_forward")
        if rl_wf_df.empty:
            st.info("No walk-forward results yet. Run **Run research lab**.")
        else:
            ok = rl_wf_df[rl_wf_df["error"].isna()] if "error" in rl_wf_df.columns else rl_wf_df
            n_ok = len(ok)
            if n_ok == 0:
                st.warning("All walk-forward windows were skipped — see "
                           "the saved CSV for reasons.")
            else:
                wins = int(((ok["strategy_return_pct"] > 0)
                            & (ok["strategy_vs_bh_pct"] > 0)).sum())
                stability = wins / n_ok * 100.0
                avg_oos = float(ok["strategy_return_pct"].mean())
                worst = float(ok["strategy_return_pct"].min())
                best = float(ok["strategy_return_pct"].max())
                cwf = st.columns(5)
                cwf[0].metric("OOS windows", str(n_ok))
                cwf[1].metric("Avg OOS return", f"{avg_oos:+.2f}%")
                cwf[2].metric("Worst OOS", f"{worst:+.2f}%")
                cwf[3].metric("Best OOS", f"{best:+.2f}%")
                cwf[4].metric(
                    "Stability score", f"{stability:.0f}%",
                    "% windows profitable AND beat B&H",
                )
                if stability < 50:
                    st.warning(
                        f"Stability score {stability:.0f}% — strategy is "
                        f"NOT robust across out-of-sample periods.",
                    )
                cols = [c for c in ["asset", "timeframe", "window",
                        "oos_start_iso", "oos_end_iso",
                        "strategy_return_pct", "buy_and_hold_return_pct",
                        "strategy_vs_bh_pct", "max_drawdown_pct",
                        "win_rate_pct", "num_trades", "profit_factor"]
                        if c in ok.columns]
                st.dataframe(
                    _style_vs_bh(ok[cols].copy()),
                    use_container_width=True, hide_index=True, height=340,
                )
            st.download_button(
                "Download walk_forward_results.csv",
                _df_to_csv_bytes(rl_wf_df),
                file_name="walk_forward_results.csv",
                mime="text/csv", key="dl_research_wf",
            )

    # ---- Strategy comparison -------------------------------------------
    with rl_tabs[2]:
        _explainer("strategy")
        if rl_sc_df.empty:
            st.info("No strategy-comparison results yet.")
        else:
            ok = rl_sc_df[rl_sc_df["error"].isna()] if "error" in rl_sc_df.columns else rl_sc_df
            st.markdown(
                f"<div class='section-sub'><b>{len(ok)} rows</b> &middot; "
                f"all strategies use the same risk engine, fees, slippage, "
                f"and next-bar-open fills.</div>",
                unsafe_allow_html=True,
            )
            if not ok.empty:
                # Aggregate per strategy
                agg = ok.groupby("strategy").agg(
                    mean_return=("total_return_pct", "mean"),
                    mean_vs_bh=("strategy_vs_bh_pct", "mean"),
                    mean_dd=("max_drawdown_pct", "mean"),
                    mean_sharpe=("sharpe_ratio", "mean"),
                    total_trades=("num_trades", "sum"),
                ).reset_index()
                best_sharpe = agg.loc[agg["mean_sharpe"].idxmax()]
                best_dd = agg.loc[agg["mean_dd"].idxmax()]  # closer to 0
                best_vs_bh = agg.loc[agg["mean_vs_bh"].idxmax()]
                cs = st.columns(3)
                cs[0].metric(
                    "Best risk-adjusted (mean Sharpe)",
                    str(best_sharpe["strategy"]),
                    f"{best_sharpe['mean_sharpe']:.2f}",
                )
                cs[1].metric(
                    "Smallest mean drawdown",
                    str(best_dd["strategy"]),
                    f"{best_dd['mean_dd']:.2f}%",
                )
                cs[2].metric(
                    "Best mean vs B&H",
                    str(best_vs_bh["strategy"]),
                    f"{best_vs_bh['mean_vs_bh']:+.2f}%",
                )
                low_trade_strats = agg[
                    agg["total_trades"] / max(agg.shape[0], 1)
                    < research.MIN_TRADES_FOR_CONFIDENCE
                ]
                if not low_trade_strats.empty:
                    st.warning(
                        "Some strategies have very few total trades — "
                        "their rankings above are not statistically robust.",
                    )
            cols = [c for c in ["strategy", "asset", "timeframe",
                    "total_return_pct", "buy_and_hold_return_pct",
                    "strategy_vs_bh_pct", "max_drawdown_pct",
                    "win_rate_pct", "num_trades", "profit_factor",
                    "fees_paid", "exposure_time_pct", "sharpe_ratio",
                    "sortino_ratio", "calmar_ratio"]
                    if c in ok.columns]
            st.dataframe(
                _style_vs_bh(ok[cols].copy()),
                use_container_width=True, hide_index=True, height=340,
            )
            st.download_button(
                "Download strategy_comparison.csv",
                _df_to_csv_bytes(rl_sc_df),
                file_name="strategy_comparison.csv",
                mime="text/csv", key="dl_research_sc",
            )

    # ---- Robustness ----------------------------------------------------
    with rl_tabs[3]:
        _explainer("robustness")
        if rl_rb_df.empty:
            st.info("No robustness results yet.")
        else:
            ok = rl_rb_df[rl_rb_df["error"].isna()] if "error" in rl_rb_df.columns else rl_rb_df
            st.markdown(
                "<div class='section-sub'>Small parameter variations per "
                "strategy family. The point is fragility testing — "
                "<b>do not pick the best variant and call it the strategy</b>."
                "</div>",
                unsafe_allow_html=True,
            )
            if not ok.empty:
                fam_summary = ok.groupby("family").agg(
                    n=("variant", "count"),
                    median_ret=("total_return_pct", "median"),
                    worst=("total_return_pct", "min"),
                    best=("total_return_pct", "max"),
                    mean_dd=("max_drawdown_pct", "mean"),
                    beats_bh=("strategy_vs_bh_pct",
                              lambda s: int((s > 0).sum())),
                ).reset_index()
                fam_summary["beats_bh_pct"] = (
                    fam_summary["beats_bh"] / fam_summary["n"] * 100.0
                )
                st.dataframe(
                    fam_summary, use_container_width=True, hide_index=True,
                    height=180,
                )
                fragile = fam_summary[
                    (fam_summary["best"] > 0) & (fam_summary["worst"] < -2)
                ]
                if not fragile.empty:
                    st.warning(
                        f"Fragile families (best variant > 0 but worst "
                        f"variant < -2%): {list(fragile['family'])}",
                    )
            cols = [c for c in ["family", "variant", "asset", "timeframe",
                    "total_return_pct", "buy_and_hold_return_pct",
                    "strategy_vs_bh_pct", "max_drawdown_pct",
                    "num_trades", "sharpe_ratio"]
                    if c in ok.columns]
            st.dataframe(
                _style_vs_bh(ok[cols].copy()),
                use_container_width=True, hide_index=True, height=340,
            )
            st.download_button(
                "Download robustness_results.csv",
                _df_to_csv_bytes(rl_rb_df),
                file_name="robustness_results.csv",
                mime="text/csv", key="dl_research_rb",
            )

    # ---- Monte Carlo --------------------------------------------------
    with rl_tabs[4]:
        _explainer("monte_carlo")
        if rl_mc_df.empty:
            st.info(
                "No Monte Carlo results yet. Run a backtest (so trades.csv "
                "exists) then **Run research lab**."
            )
        else:
            ok_row = rl_mc_df.iloc[0].to_dict()
            n_trades = int(ok_row.get("n_trades", 0))
            if n_trades < research.MIN_TRADES_FOR_MONTE_CARLO:
                st.warning("Too few trades for meaningful Monte Carlo analysis.")
            cmc = st.columns(5)
            cmc[0].metric("Closed trades", str(n_trades))
            cmc[1].metric(
                "Median final value",
                f"{float(ok_row.get('median_final_value', 0)):,.2f}",
            )
            cmc[2].metric(
                "5th pct final",
                f"{float(ok_row.get('p05_final_value', 0)):,.2f}",
            )
            cmc[3].metric(
                "95th pct final",
                f"{float(ok_row.get('p95_final_value', 0)):,.2f}",
            )
            cmc[4].metric(
                "Probability of loss",
                f"{float(ok_row.get('prob_loss', 0)) * 100:.1f}%",
            )
            cmc2 = st.columns(2)
            cmc2[0].metric(
                "Worst simulated drawdown",
                f"{float(ok_row.get('worst_drawdown_pct', 0)):.2f}%",
            )
            cmc2[1].metric(
                "Mean simulated drawdown",
                f"{float(ok_row.get('mean_drawdown_pct', 0)):.2f}%",
            )
            if not rl_mcs_df.empty:
                import plotly.graph_objects as _go
                fig = _go.Figure()
                fig.add_trace(_go.Histogram(
                    x=rl_mcs_df["return_pct"], nbinsx=40,
                    marker_color="#38bdf8",
                    marker_line_color="#0a0e1a", marker_line_width=1,
                ))
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=320,
                    margin=dict(t=30, b=20, l=20, r=20),
                    xaxis_title="Simulated return (%)",
                    yaxis_title="Count",
                )
                st.plotly_chart(fig, use_container_width=True)
            st.download_button(
                "Download monte_carlo_results.csv",
                _df_to_csv_bytes(rl_mc_df),
                file_name="monte_carlo_results.csv",
                mime="text/csv", key="dl_research_mc",
            )

    # ---- Research summary ---------------------------------------------
    with rl_tabs[5]:
        _explainer("summary")
        rl_summary_df = _research_csv("research_summary.csv")
        if rl_summary_df.empty:
            st.info("No research summary yet. Run **Run research lab**.")
        else:
            VERDICT_PLAIN: Dict[str, str] = {
                "PASS": "Passed — needs more validation.",
                "FAIL": "Failed — evidence is against it.",
                "INCONCLUSIVE": "Not reliable — too few trades or too little data.",
            }
            for _, r in rl_summary_df.iterrows():
                v = str(r["verdict"])
                cls = ("pill-pos" if v == "PASS"
                       else "pill-neg" if v == "FAIL"
                       else "pill-amber")
                plain = VERDICT_PLAIN.get(v, "")
                st.markdown(
                    f"<div style='display:flex;gap:0.6rem;align-items:flex-start;"
                    f"padding:0.5rem 0;border-bottom:1px dashed "
                    f"rgba(148,163,184,0.10);'>"
                    f"<span class='pill {cls}'>"
                    f"<span class='pill-dot'></span>{v}</span>"
                    f"<div><b>{r['check']}</b><br>"
                    f"<span style='color:#9ca3af;font-size:0.88rem;'>"
                    f"{plain} <i>{r['message']}</i></span></div></div>",
                    unsafe_allow_html=True,
                )
            st.download_button(
                "Download research_summary.csv",
                _df_to_csv_bytes(rl_summary_df),
                file_name="research_summary.csv",
                mime="text/csv", key="dl_research_summary",
            )

    # ---- Kronos confirmation (optional, local only) -------------------
    with rl_tabs[6]:
        _explainer("kronos")
        st.warning(
            "Kronos is experimental. It confirms or rejects strategy signals. "
            "It is not a trading oracle.",
            icon="⚠️",
        )
        # `src.ml.kronos_adapter` is safe to import — every heavy dep is lazy.
        from src.ml import kronos_adapter as _kronos
        kronos_status = _kronos.get_kronos_status()
        if not kronos_status["available"]:
            st.error("Kronos is not installed locally.", icon="🚫")
            st.markdown(
                "**Setup**\n\n"
                "1. Clone Kronos outside this project:\n"
                "   ```bash\n"
                "   git clone https://github.com/shiyu-coder/Kronos.git external/Kronos\n"
                "   # or set:\n"
                "   export KRONOS_REPO_PATH=\"/path/to/Kronos\"\n"
                "   ```\n"
                "2. Install optional ML dependencies:\n"
                "   ```bash\n"
                "   pip install -r requirements-ml.txt\n"
                "   ```\n"
                "3. Verify:\n"
                "   ```bash\n"
                "   python main.py kronos_status\n"
                "   ```"
            )
            with st.expander("Diagnostic details", expanded=False):
                st.json(kronos_status)
        else:
            st.success(
                f"Kronos available — model: {kronos_status['default_model']}",
                icon="✅",
            )
            kc1, kc2, kc3 = st.columns(3)
            kronos_model = kc1.selectbox(
                "Model", options=kronos_status["supported_models"], index=0,
                help="Kronos-mini is the smallest and fastest. Use it first.",
            )
            kronos_device = kc2.selectbox(
                "Device", options=["cpu", "mps", "cuda:0"], index=0,
                help="CPU works everywhere but is slow.",
            )
            kronos_asset = kc3.selectbox(
                "Asset", options=asset_choice or list(config.ASSETS),
                help="Which asset to forecast.",
            )
            kc4, kc5, kc6, kc7 = st.columns(4)
            kronos_tf = kc4.selectbox(
                "Timeframe", options=config.TIMEFRAMES,
                index=config.TIMEFRAMES.index(config.DEFAULT_TIMEFRAME),
            )
            kronos_lookback = kc5.number_input(
                "Lookback candles", min_value=50, max_value=2048,
                value=400, step=50,
            )
            kronos_pred_len = kc6.number_input(
                "Prediction length", min_value=4, max_value=128,
                value=24, step=1,
            )
            kronos_buy_thr = kc7.number_input(
                "Buy confirm threshold (%)", min_value=0.0, max_value=10.0,
                value=1.0, step=0.5,
                help=("Forecast return % required to CONFIRM a BUY. "
                      "Below 0 = REJECT, between 0 and threshold = NEUTRAL."),
            )

            st.caption(
                "All Kronos work runs synchronously on CPU by default. The "
                "first run downloads model weights to the Hugging Face cache."
            )
            kbtn1, kbtn2, kbtn3 = st.columns(3)
            do_kronos_eval = kbtn1.button(
                "Run forecast evaluation", use_container_width=True,
                key="kronos_eval_btn",
            )
            do_kronos_confirm = kbtn2.button(
                "Generate confirmations", use_container_width=True,
                key="kronos_confirm_btn",
            )
            do_kronos_compare = kbtn3.button(
                "Compare base vs Kronos-confirmed", use_container_width=True,
                key="kronos_compare_btn",
            )

            if do_kronos_eval:
                from src.ml import forecast_evaluation as _fe
                with st.spinner("Running Kronos evaluation… (CPU, slow)"):
                    try:
                        df = _fe.evaluate_kronos_forecasts(
                            asset=kronos_asset, timeframe=kronos_tf,
                            model_name=kronos_model,
                            lookback=int(kronos_lookback),
                            pred_len=int(kronos_pred_len),
                            step=int(kronos_pred_len),
                            max_windows=10, device=kronos_device,
                        )
                        _queue_flash("success",
                                     f"Kronos evaluation complete — {len(df)} windows.")
                        st.rerun()
                    except Exception as e:
                        _queue_flash("error", f"Kronos evaluation failed: {e}")
                        st.rerun()

            if do_kronos_confirm:
                from src.ml import kronos_confirmation as _kc
                fe_path = config.RESULTS_DIR / "kronos_forecast_evaluation.csv"
                if not fe_path.exists():
                    _queue_flash("warning",
                                 "Run **Run forecast evaluation** first.")
                    st.rerun()
                else:
                    with st.spinner("Generating confirmations…"):
                        try:
                            from src.strategies import REGISTRY as _STRATS
                            art = backtester.run_backtest(
                                assets=[kronos_asset], timeframe=kronos_tf,
                                save=False, strategy=_STRATS["rsi_ma_atr"](),
                            )
                            decisions = art.decisions.copy()
                            decisions["timeframe"] = kronos_tf
                            base_signals = decisions[
                                ["timestamp_ms", "asset", "timeframe", "action"]
                            ]
                            fe_df = pd.read_csv(fe_path)
                            _kc.generate_kronos_confirmations(
                                base_signals_df=base_signals,
                                forecast_eval_df=fe_df,
                                buy_confirm_threshold_pct=float(kronos_buy_thr),
                            )
                            _queue_flash("success",
                                         "Kronos confirmations saved.")
                            st.rerun()
                        except Exception as e:
                            _queue_flash("error",
                                         f"Confirmation generation failed: {e}")
                            st.rerun()

            if do_kronos_compare:
                from src.ml import forecast_evaluation as _fe
                with st.spinner("Running base vs Kronos-confirmed backtest…"):
                    try:
                        _fe.compare_base_vs_kronos_confirmed(
                            asset=kronos_asset, timeframe=kronos_tf,
                            base_strategy_name="rsi_ma_atr",
                        )
                        _queue_flash("success",
                                     "Base vs Kronos comparison saved.")
                        st.rerun()
                    except Exception as e:
                        _queue_flash("error", f"Comparison failed: {e}")
                        st.rerun()

        # ---- Tables + downloads (always visible if files exist) ----------
        kronos_eval_df = _research_csv("kronos_forecast_evaluation.csv")
        kronos_confirm_df = _research_csv("kronos_confirmations.csv")
        kronos_compare_df = _research_csv("kronos_confirmation_comparison.csv")
        kronos_forecast_df = _research_csv("kronos_forecast.csv")

        if not kronos_eval_df.empty:
            from src.ml.forecast_evaluation import summarise_forecast_evaluation
            summary = summarise_forecast_evaluation(kronos_eval_df)
            if summary.get("ok"):
                kc = st.columns(4)
                kc[0].metric("Windows", str(summary["n_windows"]))
                kc[1].metric(
                    "Direction accuracy",
                    f"{summary['direction_accuracy_pct']:.1f}%",
                )
                kc[2].metric(
                    "MAPE", f"{summary['mape_pct']:.2f}%",
                    help="Mean absolute % error of forecast vs actual return.",
                )
                kc[3].metric(
                    "Worst miss", f"{summary['worst_abs_error_pct']:.2f}%",
                )
            st.markdown("**Forecast evaluation**")
            st.dataframe(
                kronos_eval_df, use_container_width=True,
                hide_index=True, height=240,
            )

        if not kronos_confirm_df.empty:
            st.markdown("**Confirmations**")
            counts = kronos_confirm_df["confirmation"].value_counts().to_dict()
            st.caption(f"Verdict counts: {counts}")
            st.dataframe(
                kronos_confirm_df, use_container_width=True,
                hide_index=True, height=240,
            )

        if not kronos_compare_df.empty:
            st.markdown("**Base vs Kronos-confirmed**")
            cols = [c for c in ["variant", "total_return_pct",
                    "buy_and_hold_return_pct", "strategy_vs_bh_pct",
                    "max_drawdown_pct", "num_trades", "profit_factor",
                    "fees_paid", "exposure_time_pct", "sharpe_ratio"]
                    if c in kronos_compare_df.columns]
            st.dataframe(
                _style_vs_bh(kronos_compare_df[cols].copy()),
                use_container_width=True, hide_index=True, height=180,
            )
            # Quick honest readout: did Kronos help?
            try:
                base_row = kronos_compare_df.iloc[0]
                k_row = kronos_compare_df.iloc[1]
                base_vs = float(base_row["strategy_vs_bh_pct"])
                k_vs = float(k_row["strategy_vs_bh_pct"])
                k_trades = int(k_row["num_trades"])
                base_trades = int(base_row["num_trades"])
                if k_vs > base_vs and k_trades > 0:
                    st.success(
                        f"Kronos confirmation improved vs B&H by "
                        f"{k_vs - base_vs:+.2f}%.",
                    )
                else:
                    st.warning(
                        "Kronos did not improve vs B&H. Reduced trades "
                        f"({base_trades} → {k_trades}) without an edge gain.",
                    )
            except Exception:
                pass

        kdl = st.columns(4)
        for i, (label, name, df) in enumerate([
            ("Forecast", "kronos_forecast.csv", kronos_forecast_df),
            ("Evaluation", "kronos_forecast_evaluation.csv", kronos_eval_df),
            ("Confirmations", "kronos_confirmations.csv", kronos_confirm_df),
            ("Comparison", "kronos_confirmation_comparison.csv",
             kronos_compare_df),
        ]):
            with kdl[i]:
                if df is not None and not df.empty:
                    st.download_button(
                        f"{label} CSV", _df_to_csv_bytes(df),
                        file_name=name, mime="text/csv",
                        key=f"dl_kronos_{i}", use_container_width=True,
                    )
                else:
                    st.button(
                        f"{label} CSV (unavailable)", disabled=True,
                        key=f"dl_kronos_{i}_disabled",
                        use_container_width=True,
                    )


# ---------------------------------------------------------------------------
# Paper-trader state + risk dashboard
# ---------------------------------------------------------------------------
paper_state = paper_trader.get_state_for_display()
positions = paper_state.get("positions", {})

with st.container(border=True):
    st.markdown(
        "<div class='section-h'><span class='dot'></span>"
        "Simulated positions &amp; risk</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-sub'>Paper-trader state — separate from the "
        "backtest. Updated by <b>Run paper tick</b>.</div>",
        unsafe_allow_html=True,
    )

    if not positions:
        st.info("No open paper positions. Click **Run paper tick** in the sidebar.")
    else:
        rows = []
        for asset, pos in positions.items():
            candle_df = _safe_load_candles(asset, timeframe)
            last_close = (float(candle_df["close"].iloc[-1])
                          if candle_df is not None and not candle_df.empty
                          else float(pos["entry_price"]))
            size = float(pos["size"])
            entry = float(pos["entry_price"])
            upnl = size * (last_close - entry)
            rows.append({
                "asset": asset,
                "entry_price": entry,
                "current_price": last_close,
                "size": size,
                "stop_loss": float(pos["stop_loss_price"]),
                "unrealized_pnl": upnl,
                "allocation_%": (
                    (size * last_close)
                    / max(paper_state.get("cash", 0.0) + size * last_close, 1e-9)
                    * 100.0
                ),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    total_eq = paper_state.get("cash", 0.0)
    exposure = 0.0
    for asset, pos in positions.items():
        candle_df = _safe_load_candles(asset, timeframe)
        last_close = (float(candle_df["close"].iloc[-1])
                      if candle_df is not None and not candle_df.empty
                      else float(pos["entry_price"]))
        exposure += float(pos["size"]) * last_close
        total_eq += float(pos["size"]) * last_close
    max_allowed = total_eq * float(max_position_pct) * len(asset_choice or [1])

    cols_r = st.columns(4)
    cols_r[0].metric("Cash (paper)",
                     f"{_fmt_money(paper_state.get('cash', 0.0))} USDT")
    cols_r[1].metric("Current exposure", f"{_fmt_money(exposure)} USDT")
    cols_r[2].metric("Max allowed exposure",
                     f"{_fmt_money(max_allowed)} USDT")
    cols_r[3].metric("Rejected (paper)",
                     str(int(paper_state.get("rejected_count", 0))))

    risk_decisions = pd.DataFrame(paper_state.get("decisions", []))
    if not risk_decisions.empty:
        rejected_recent = risk_decisions[risk_decisions["accepted"] == False]
        if not rejected_recent.empty:
            st.caption("Recent paper rejections / risk-offs:")
            st.dataframe(
                rejected_recent.tail(20)[
                    ["datetime_iso", "asset", "action", "reason"]
                ],
                use_container_width=True, hide_index=True,
            )


# ---------------------------------------------------------------------------
# Debug & audit (collapsed by default — hidden by default to reduce clutter)
# ---------------------------------------------------------------------------
with st.expander("Debug & audit", expanded=False):
    st.write("**Config snapshot**")
    st.json(config.summary())

    st.write("**Data freshness**")
    rows = []
    for asset in config.ASSETS:
        for tf in config.TIMEFRAMES:
            p = utils.csv_path_for(asset, tf)
            if p.exists():
                df = _safe_load_candles(asset, tf)
                last = df["datetime"].iloc[-1] if df is not None and not df.empty else None
                rows.append({
                    "asset": asset, "timeframe": tf,
                    "candles": len(df) if df is not None else 0,
                    "last_candle_utc": str(last),
                    "missing": (int(df.isna().any(axis=1).sum())
                                if df is not None else 0),
                })
            else:
                rows.append({"asset": asset, "timeframe": tf,
                             "candles": 0, "last_candle_utc": None, "missing": 0})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if config.LIVE_TRADING_ENABLED:
        st.error(
            "LIVE_TRADING_ENABLED is True. This must be False in v1. "
            "The app refuses to run trading actions in this state."
        )
    else:
        st.success(
            "Safety: LIVE_TRADING_ENABLED = False. No live execution module exists."
        )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='cte-footer'>"
    "v1 is research-only. No exchange API keys are used or accepted. "
    "All execution is simulated."
    "</div>",
    unsafe_allow_html=True,
)
