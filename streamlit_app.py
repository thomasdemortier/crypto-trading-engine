"""
Streamlit dashboard.

Run locally:
    streamlit run streamlit_app.py

Deployable to Streamlit Community Cloud — no API keys, all paths relative.
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from src import (
    config, backtester, data_collector, indicators, paper_trader,
    performance, plotting, strategy, utils,
)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Crypto Trading Engine",
    page_icon=":bar_chart:",
    layout="wide",
)

st.markdown(
    """
    <style>
      .metric-card {padding: 0.6rem 0.9rem; border:1px solid #eee;
                    border-radius:8px; background:#fafafa;}
      .warn-box {background:#fff8e1; border:1px solid #ffd54f;
                 border-radius:6px; padding:0.6rem 0.9rem;}
      .small {font-size: 0.85rem; color:#666;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Crypto Trading Engine")
st.caption("Backtesting and paper trading dashboard for BTC and ETH")
st.markdown(
    '<div class="warn-box"><b>Research only.</b> No live trading. '
    'No financial advice. Past performance is not indicative of future results.'
    '</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_candles_cached(asset: str, timeframe: str, mtime: float) -> pd.DataFrame:
    """Cache by (asset, timeframe, file mtime) so the cache invalidates on refresh."""
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


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Controls")

asset_choice = st.sidebar.multiselect(
    "Assets", options=config.ASSETS, default=config.ASSETS,
)
timeframe = st.sidebar.selectbox(
    "Timeframe", options=config.TIMEFRAMES,
    index=config.TIMEFRAMES.index(config.DEFAULT_TIMEFRAME),
)

st.sidebar.subheader("Strategy parameters")
rsi_buy = st.sidebar.slider(
    "RSI buy threshold", 5, 50, int(config.STRATEGY.rsi_buy_threshold),
)
rsi_sell = st.sidebar.slider(
    "RSI sell threshold", 50, 95, int(config.STRATEGY.rsi_sell_threshold),
)
atr_pct_max = st.sidebar.slider(
    "ATR % max (skip new entries above)", 1.0, 15.0,
    float(config.STRATEGY.atr_pct_max), 0.5,
)

st.sidebar.subheader("Risk parameters")
starting_capital = st.sidebar.number_input(
    "Starting capital (USDT)", min_value=100.0, max_value=10_000_000.0,
    value=float(config.RISK.starting_capital), step=100.0,
)
max_position_pct = st.sidebar.slider(
    "Max position size (%)", 1.0, 25.0,
    float(config.RISK.max_position_pct * 100), 0.5,
) / 100.0
risk_per_trade_pct = st.sidebar.slider(
    "Risk per trade (%)", 0.1, 5.0,
    float(config.RISK.risk_per_trade_pct * 100), 0.1,
) / 100.0
max_daily_loss_pct = st.sidebar.slider(
    "Max daily loss (%)", 0.5, 10.0,
    float(config.RISK.max_daily_loss_pct * 100), 0.1,
) / 100.0
fee_pct = st.sidebar.number_input(
    "Fee (% per trade)", min_value=0.0, max_value=2.0,
    value=float(config.RISK.fee_pct * 100), step=0.01, format="%.3f",
) / 100.0
slippage_pct = st.sidebar.number_input(
    "Slippage (% per trade)", min_value=0.0, max_value=2.0,
    value=float(config.RISK.slippage_pct * 100), step=0.01, format="%.3f",
) / 100.0
stop_loss_pct = st.sidebar.slider(
    "Stop-loss distance (%)", 1.0, 25.0,
    float(config.RISK.stop_loss_pct * 100), 0.5,
) / 100.0

st.sidebar.subheader("Actions")
do_backtest = st.sidebar.button(
    "Run backtest", type="primary", use_container_width=True,
    help="Run the backtester for the selected assets and timeframe.",
)
do_refresh = st.sidebar.button(
    "Refresh data", use_container_width=True,
    help="Re-download candles from the exchange.",
)
do_paper_tick = st.sidebar.button(
    "Run paper tick", use_container_width=True,
    help="Run one paper-trade evaluation on the latest candle.",
)
do_clear = st.sidebar.button(
    "Clear saved results", use_container_width=True,
    help=(
        "Delete saved backtest artifacts (equity curve, trades, decisions, "
        "metrics, scope metadata). Cached candles and paper-trader state "
        "are NOT touched."
    ),
)

# ---------------------------------------------------------------------------
# Run actions
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


if do_refresh:
    if not asset_choice:
        st.warning("Pick at least one asset to refresh.")
    else:
        with st.spinner("Downloading candles…"):
            try:
                paths = data_collector.download_all(
                    assets=asset_choice, timeframes=[timeframe], refresh=True,
                )
                st.success(f"Refreshed {len(paths)} dataset(s).")
            except Exception as e:
                st.error(f"Refresh failed: {e}")

if do_backtest:
    if not asset_choice:
        st.warning("Pick at least one asset to backtest.")
    else:
        strat_cfg, risk_cfg = _build_runtime_configs()
        with st.spinner("Running backtest…"):
            try:
                # Make sure data exists; do a non-refreshing download first.
                data_collector.download_all(
                    assets=asset_choice, timeframes=[timeframe], refresh=False,
                )
                art = backtester.run_backtest(
                    assets=asset_choice, timeframe=timeframe,
                    risk_cfg=risk_cfg, strat_cfg=strat_cfg, save=True,
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
                # Mark the just-written run as fresh in this session.
                st.session_state["fresh_run_iso"] = art.meta.get("run_timestamp_iso")
                st.success(
                    f"Backtest complete — final equity: "
                    f"{metrics.final_portfolio_value:,.2f} USDT"
                )
            except Exception as e:
                st.error(f"Backtest failed: {e}")

if do_paper_tick:
    strat_cfg, risk_cfg = _build_runtime_configs()
    with st.spinner("Running paper tick…"):
        try:
            result = paper_trader.run_tick(
                timeframe=timeframe, assets=asset_choice, refresh=True,
            )
            st.success("Paper tick complete.")
            st.json(result)
        except Exception as e:
            st.error(f"Paper tick failed: {e}")

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
    for p in targets:
        try:
            if p.exists():
                p.unlink()
                removed += 1
        except Exception as e:  # noqa: BLE001
            st.warning(f"Could not delete {p.name}: {e}")
    # Drop the freshness sentinel and the cached candle reads.
    st.session_state.pop("fresh_run_iso", None)
    try:
        _load_candles_cached.clear()
    except Exception:
        pass
    st.success(f"Cleared {removed} saved result file(s). Cached candles kept.")


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


# ---------------------------------------------------------------------------
# Section 1 — Backtest Scope panel
# ---------------------------------------------------------------------------
st.subheader("Backtest scope")

if meta is None:
    st.info(
        "No backtest has been run on this machine yet. Configure the sidebar "
        "and click **Run backtest** to populate this dashboard."
    )
    meta_assets: List[str] = []
    meta_timeframe = timeframe
else:
    meta_assets = list(meta.get("assets") or [])
    meta_timeframe = str(meta.get("timeframe") or timeframe)

    fresh_iso = st.session_state.get("fresh_run_iso")
    run_iso = meta.get("run_timestamp_iso")
    is_fresh = bool(fresh_iso) and fresh_iso == run_iso
    freshness_label = (
        "Fresh (this session)" if is_fresh else "Loaded from saved files"
    )
    freshness_badge_color = "#1b5e20" if is_fresh else "#6d4c41"

    # Render scope as a 4-column grid of metric cards so the values are large
    # and unmissable. Each cell uses the .metric-card style defined above.
    def _scope_card(label: str, value: str) -> str:
        return (
            f"<div class='metric-card'>"
            f"<div style='font-size:0.78rem;color:#666;"
            f"text-transform:uppercase;letter-spacing:0.04em;'>{label}</div>"
            f"<div style='font-size:1.05rem;font-weight:600;"
            f"margin-top:0.15rem;word-break:break-word;'>{value}</div>"
            f"</div>"
        )

    scope_items = [
        ("Selected assets", ", ".join(meta_assets) if meta_assets else "—"),
        ("Timeframe", meta_timeframe),
        ("Starting capital", f"{float(meta.get('starting_capital', 0.0)):,.2f} USDT"),
        ("Candles used", str(int(meta.get("num_candles_used") or 0))),
        ("First candle (UTC)", str(meta.get("first_candle_iso") or "—")),
        ("Last candle (UTC)", str(meta.get("last_candle_iso") or "—")),
        ("Last backtest run (UTC)", str(run_iso or "—")),
        ("Result source", freshness_label),
    ]
    # Two rows of 4 columns each so all 8 fields are visible without scrolling.
    for chunk_start in (0, 4):
        cols = st.columns(4)
        for i, (label, value) in enumerate(scope_items[chunk_start:chunk_start + 4]):
            with cols[i]:
                st.markdown(_scope_card(label, value), unsafe_allow_html=True)

    # Explicit freshness badge so the status is obvious at a glance.
    st.markdown(
        f"<div style='margin-top:0.6rem;'>"
        f"<span style='background:{freshness_badge_color};color:white;"
        f"padding:0.2rem 0.6rem;border-radius:999px;font-size:0.85rem;'>"
        f"Status: {freshness_label}</span></div>",
        unsafe_allow_html=True,
    )

    # Show a clear banner if the sidebar selection differs from the loaded run.
    sidebar_set = set(asset_choice or [])
    meta_set = set(meta_assets)
    if sidebar_set and sidebar_set != meta_set:
        st.warning(
            f"Sidebar selection ({sorted(sidebar_set)}) differs from the saved "
            f"backtest scope ({sorted(meta_set)}). Click **Run backtest** to "
            f"regenerate results for the current selection. Numbers below "
            f"reflect the saved scope."
        )
    if timeframe != meta_timeframe:
        st.warning(
            f"Sidebar timeframe ({timeframe}) differs from the saved backtest "
            f"timeframe ({meta_timeframe}). Numbers below reflect the saved "
            f"timeframe."
        )


# ---------------------------------------------------------------------------
# Section 2 — Results
# ---------------------------------------------------------------------------
def _render_metric_block(label_prefix: str, m: dict) -> None:
    cols = st.columns(4)
    cols[0].metric(
        f"{label_prefix} — Final portfolio",
        f"{m['final_portfolio_value']:,.0f} USDT",
        f"{m['total_return_pct']:+.2f}%",
    )
    cols[1].metric(
        f"{label_prefix} — Buy & hold",
        f"{m['buy_and_hold_return_pct']:+.2f}%",
        f"{m['strategy_vs_bh_pct']:+.2f}% vs B&H",
    )
    cols[2].metric(f"{label_prefix} — Max drawdown",
                   f"{m['max_drawdown_pct']:+.2f}%")
    cols[3].metric(f"{label_prefix} — Win rate",
                   f"{m['win_rate_pct']:.1f}%",
                   f"{int(m['num_trades'])} round-trips")

    cols2 = st.columns(4)
    cols2[0].metric(f"{label_prefix} — Profit factor", f"{m['profit_factor']:.2f}")
    cols2[1].metric(f"{label_prefix} — Fees paid",
                    f"{m['fees_paid']:,.2f} USDT")
    cols2[2].metric(f"{label_prefix} — Sharpe", f"{m['sharpe_ratio']:.2f}")
    cols2[3].metric(f"{label_prefix} — Sortino / Calmar",
                    f"{m['sortino_ratio']:.2f} / {m['calmar_ratio']:.2f}")


def _render_per_asset_block(label_prefix: str, row: pd.Series) -> None:
    cols = st.columns(4)
    cols[0].metric(
        f"{label_prefix} — Realized P&L",
        f"{row['realized_pnl']:+,.2f} USDT",
        f"{row['realized_return_on_allocation_pct']:+.2f}% on allocated share",
    )
    cols[1].metric(
        f"{label_prefix} — Buy & hold",
        f"{row['buy_and_hold_return_pct']:+.2f}%",
        f"{row['strategy_vs_bh_pct']:+.2f}% vs B&H",
    )
    cols[2].metric(
        f"{label_prefix} — Win rate",
        f"{row['win_rate_pct']:.1f}%",
        f"{int(row['num_trades'])} round-trips",
    )
    cols[3].metric(
        f"{label_prefix} — Profit factor",
        f"{row['profit_factor']:.2f}",
    )

    cols2 = st.columns(4)
    cols2[0].metric(f"{label_prefix} — Avg win",
                    f"{row['avg_winning_trade']:+,.2f} USDT")
    cols2[1].metric(f"{label_prefix} — Avg loss",
                    f"{row['avg_losing_trade']:+,.2f} USDT")
    cols2[2].metric(f"{label_prefix} — Fees paid",
                    f"{row['fees_paid']:,.2f} USDT")
    cols2[3].metric(f"{label_prefix} — Allocated capital",
                    f"{row['allocated_capital']:,.2f} USDT")


if metrics_df.empty:
    st.subheader("Results")
    st.info("No backtest run yet. Configure the sidebar and click **Run backtest**.")
else:
    m = metrics_df.iloc[0].to_dict()
    n_assets = len(meta_assets)

    # ----- B&H underperformance warning (always shown when applicable) -----
    if float(m.get("strategy_vs_bh_pct", 0.0)) < 0:
        st.warning(
            "Strategy underperformed buy and hold over this exact tested window."
        )

    # ----- Per-asset sections -----
    if n_assets <= 1:
        only = meta_assets[0] if meta_assets else "(asset)"
        st.subheader(f"{only} Results")
        st.caption(
            "Single asset selected — combined portfolio metrics below ARE the "
            f"{only} results (one shared cash pool, one asset)."
        )
        _render_metric_block(only, m)
    else:
        # Both assets selected: separate per-asset sections + a combined section
        # explicitly labelled, plus a side-by-side comparison table.
        if per_asset_df.empty:
            st.warning(
                "Per-asset breakdown is missing from the saved results. "
                "Click **Run backtest** to regenerate."
            )
        else:
            for asset in meta_assets:
                rows = per_asset_df[per_asset_df["asset"] == asset]
                if rows.empty:
                    st.subheader(f"{asset} Results")
                    st.info(f"No saved per-asset data for {asset}.")
                    continue
                st.subheader(f"{asset} Results")
                st.caption(
                    f"Trade-derived metrics for {asset} alone. Allocated capital "
                    f"is starting_capital ÷ number of selected assets."
                )
                _render_per_asset_block(asset, rows.iloc[0])

        st.subheader("Combined Portfolio Results")
        st.caption(
            "Shared cash pool across all selected assets. Sharpe, Sortino, "
            "max drawdown and exposure-time are PORTFOLIO-LEVEL only — they "
            "cannot be cleanly split per asset and so are not shown above."
        )
        _render_metric_block("Portfolio", m)

        if not per_asset_df.empty:
            st.subheader("Asset comparison")
            comp_cols = [
                "asset", "allocated_capital", "realized_pnl",
                "realized_return_on_allocation_pct", "buy_and_hold_return_pct",
                "strategy_vs_bh_pct", "num_trades", "win_rate_pct",
                "profit_factor", "fees_paid", "slippage_cost",
            ]
            comp_cols = [c for c in comp_cols if c in per_asset_df.columns]
            st.dataframe(
                per_asset_df[comp_cols], use_container_width=True, hide_index=True,
            )


# ---------------------------------------------------------------------------
# Section 3 — CSV downloads
# ---------------------------------------------------------------------------
st.subheader("Downloads")
st.caption("Export the most recent saved artifacts as CSV.")

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

# Render two per row so labels and buttons stay legible at typical widths.
for chunk_start in range(0, len(download_specs), 2):
    cols = st.columns(2)
    for i, spec in enumerate(download_specs[chunk_start:chunk_start + 2]):
        label, fname, df, desc, missing_msg, key = spec
        with cols[i]:
            st.markdown(f"**{label}** — `{fname}`")
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
# Section 4 — Charts
# ---------------------------------------------------------------------------
st.subheader("Charts")
chart_tabs = st.tabs(["Equity", "Drawdown", "Price + trades",
                      "Cumulative P&L", "Monthly returns"])

with chart_tabs[0]:
    if equity_df.empty:
        st.info(
            "No equity curve to plot. Reason: no backtest has been run yet — "
            "click **Run backtest** in the sidebar."
        )
    else:
        st.plotly_chart(
            plotting.equity_curve_fig(equity_df, float(starting_capital)),
            use_container_width=True,
        )
with chart_tabs[1]:
    if equity_df.empty:
        st.info(
            "No drawdown to plot. Reason: equity curve is empty — run a "
            "backtest first."
        )
    else:
        st.plotly_chart(plotting.drawdown_fig(equity_df), use_container_width=True)
with chart_tabs[2]:
    if not asset_choice:
        st.info(
            "No price chart to plot. Reason: no asset is selected in the "
            "sidebar. Pick BTC/USDT and/or ETH/USDT."
        )
    else:
        which = st.selectbox("Asset for price chart", options=asset_choice,
                             key="price_asset")
        price_df = _read_artifact(f"price_{utils.safe_symbol(which)}.csv")
        if price_df is None:
            price_df = _safe_load_candles(which, timeframe)
        if price_df is None:
            st.info(
                f"No price data found for {which} at {timeframe}. Reason: "
                f"the candle CSV does not exist locally — click **Refresh "
                f"data** in the sidebar to download it."
            )
        else:
            if trades_df.empty:
                st.caption(
                    f"Showing {which} closes only — no trade markers because "
                    "no trades were generated under current settings."
                )
            elif trades_df[trades_df["asset"] == which].empty:
                st.caption(
                    f"Showing {which} closes only — no trades recorded for "
                    f"{which} in the last backtest."
                )
            st.plotly_chart(
                plotting.price_with_trades_fig(price_df, trades_df, which),
                use_container_width=True,
            )
with chart_tabs[3]:
    if trades_df.empty:
        st.info(
            "No cumulative P&L to plot. Reason: no trades were generated "
            "under current settings — try lowering the RSI buy threshold or "
            "raising the ATR% max in the sidebar."
        )
    elif trades_df[trades_df["side"] == "SELL"].empty:
        st.info(
            "No cumulative P&L to plot. Reason: only OPEN trades exist — no "
            "round-trips have been closed yet."
        )
    else:
        st.plotly_chart(
            plotting.cumulative_pnl_fig(trades_df), use_container_width=True,
        )
with chart_tabs[4]:
    if equity_df.empty:
        st.info(
            "No monthly returns to plot. Reason: equity curve is empty — "
            "run a backtest first."
        )
    elif len(equity_df) < 30:
        st.info(
            f"Not enough history to compute monthly returns — only "
            f"{len(equity_df)} bars available. Use a longer history window."
        )
    else:
        st.plotly_chart(
            plotting.monthly_returns_fig(equity_df), use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Section 5 — Trade history
# ---------------------------------------------------------------------------
st.subheader("Trade history")
if trades_df.empty:
    st.info("No trades generated under current settings.")
else:
    display_cols = ["datetime_iso", "asset", "side", "price", "size",
                    "fee", "slippage_cost", "realized_pnl",
                    "portfolio_value", "reason"]
    display_cols = [c for c in display_cols if c in trades_df.columns]
    st.caption(
        f"{len(trades_df)} trade row(s) across "
        f"{trades_df['asset'].nunique()} asset(s)."
    )
    st.dataframe(trades_df[display_cols], use_container_width=True, height=320)
    st.download_button("Download trades.csv", _df_to_csv_bytes(trades_df),
                       "trades.csv", "text/csv", key="dl_trades_section")


# ---------------------------------------------------------------------------
# Section 6 — Decision log
# ---------------------------------------------------------------------------
st.subheader("Decision log")
if decisions_df.empty:
    st.info("No decisions logged under current settings.")
else:
    st.caption(
        f"{len(decisions_df)} decision row(s) — showing the most recent 500."
    )
    st.dataframe(decisions_df.tail(500), use_container_width=True, height=320)
    st.download_button("Download decisions.csv", _df_to_csv_bytes(decisions_df),
                       "decisions.csv", "text/csv", key="dl_decisions_section")


# ---------------------------------------------------------------------------
# Section 7 — Current simulated positions
# ---------------------------------------------------------------------------
st.subheader("Simulated positions (paper-trader state)")
paper_state = paper_trader.get_state_for_display()
positions = paper_state.get("positions", {})
if not positions:
    st.info("No open paper positions. Run **Paper tick** from the sidebar.")
else:
    rows = []
    for asset, pos in positions.items():
        candle_df = _safe_load_candles(asset, timeframe)
        last_close = float(candle_df["close"].iloc[-1]) if candle_df is not None and not candle_df.empty else float(pos["entry_price"])
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
            "allocation_%": (size * last_close) / max(paper_state.get("cash", 0.0) + size * last_close, 1e-9) * 100.0,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ---------------------------------------------------------------------------
# Section 8 — Risk dashboard
# ---------------------------------------------------------------------------
st.subheader("Risk dashboard")
total_eq = paper_state.get("cash", 0.0)
exposure = 0.0
for asset, pos in paper_state.get("positions", {}).items():
    candle_df = _safe_load_candles(asset, timeframe)
    last_close = float(candle_df["close"].iloc[-1]) if candle_df is not None and not candle_df.empty else float(pos["entry_price"])
    exposure += float(pos["size"]) * last_close
    total_eq += float(pos["size"]) * last_close
max_allowed = total_eq * float(max_position_pct) * len(asset_choice or [1])

risk_decisions = pd.DataFrame(paper_state.get("decisions", []))
rejected_recent = risk_decisions[risk_decisions["accepted"] == False] if not risk_decisions.empty else pd.DataFrame()

cols_r = st.columns(4)
cols_r[0].metric("Cash (paper)", f"{paper_state.get('cash', 0.0):,.2f} USDT")
cols_r[1].metric("Current exposure", f"{exposure:,.2f} USDT")
cols_r[2].metric("Max allowed exposure", f"{max_allowed:,.2f} USDT")
cols_r[3].metric("Rejected (paper)", str(int(paper_state.get("rejected_count", 0))))

if not rejected_recent.empty:
    st.write("Recent rejections / risk-offs:")
    st.dataframe(rejected_recent.tail(20)[["datetime_iso", "asset", "action", "reason"]],
                 use_container_width=True)


# ---------------------------------------------------------------------------
# Section 9 — Debug & audit panel
# ---------------------------------------------------------------------------
with st.expander("Debug & audit"):
    cfg_summary = config.summary()
    st.write("**Config snapshot**")
    st.json(cfg_summary)

    st.write("**Data freshness**")
    rows = []
    for asset in config.ASSETS:
        for tf in config.TIMEFRAMES:
            p = utils.csv_path_for(asset, tf)
            if p.exists():
                df = _safe_load_candles(asset, tf)
                last = df["datetime"].iloc[-1] if df is not None and not df.empty else None
                rows.append({"asset": asset, "timeframe": tf,
                             "candles": len(df) if df is not None else 0,
                             "last_candle_utc": str(last),
                             "missing": int(df.isna().any(axis=1).sum()) if df is not None else 0})
            else:
                rows.append({"asset": asset, "timeframe": tf,
                             "candles": 0, "last_candle_utc": None, "missing": 0})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if config.LIVE_TRADING_ENABLED:
        st.error("LIVE_TRADING_ENABLED is True. This must be False in v1. "
                 "The app refuses to run trading actions in this state.")
    else:
        st.success("Safety: LIVE_TRADING_ENABLED = False. No live execution module exists.")

st.markdown(
    '<p class="small">v1 is research-only. No exchange API keys are used or accepted. '
    'All execution is simulated.</p>',
    unsafe_allow_html=True,
)
