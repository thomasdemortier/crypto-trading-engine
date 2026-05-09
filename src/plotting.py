"""Plotly chart builders used by the Streamlit dashboard.

Plain functions so they can also be imported from notebooks. Every function
gracefully handles empty inputs and returns a figure with a placeholder
annotation rather than raising — this keeps the dashboard from crashing when
no backtest has run yet.

All figures use a dark, transparent-background style so they sit cleanly
inside the dashboard's dark card components.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import performance


# Shared dark palette — keeps all charts consistent with the dashboard chrome.
_BG = "rgba(0,0,0,0)"           # transparent, so the card colour shows through
_FG = "#e5e7eb"
_MUTED = "#9ca3af"
_GRID = "#1f2937"
_ACCENT = "#38bdf8"
_POS = "#10b981"
_NEG = "#f43f5e"


def _apply_dark_layout(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color=_FG, size=12),
        height=height,
        margin=dict(t=30, b=30, l=20, r=20),
        legend=dict(
            orientation="h", y=1.08, x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(color=_MUTED, size=11),
        ),
        xaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID,
                   tickfont=dict(color=_MUTED), linecolor=_GRID),
        yaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID,
                   tickfont=dict(color=_MUTED), linecolor=_GRID),
    )
    return fig


def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=13, color=_MUTED),
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        height=320, margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def equity_curve_fig(equity_df: pd.DataFrame, starting_capital: float) -> go.Figure:
    if equity_df.empty:
        return _empty_fig("No equity curve yet — run a backtest from the sidebar.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(equity_df["datetime"], utc=True, errors="coerce"),
        y=equity_df["equity"], mode="lines", name="Portfolio equity",
        line=dict(width=2.2, color=_ACCENT),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.08)",
    ))
    fig.add_hline(
        y=starting_capital, line_dash="dash", line_color=_MUTED,
        annotation_text=f"start: {starting_capital:,.0f}",
        annotation_position="bottom right",
        annotation_font=dict(color=_MUTED, size=10),
    )
    _apply_dark_layout(fig, height=380)
    fig.update_layout(yaxis_title="USDT", xaxis_title="")
    return fig


def drawdown_fig(equity_df: pd.DataFrame) -> go.Figure:
    if equity_df.empty:
        return _empty_fig("No drawdown data yet.")
    dd = performance.drawdown_curve(equity_df["equity"].astype(float)) * 100.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(equity_df["datetime"], utc=True, errors="coerce"),
        y=dd, fill="tozeroy", mode="lines",
        line=dict(color=_NEG, width=1.6),
        fillcolor="rgba(244,63,94,0.18)",
        name="Drawdown %",
    ))
    _apply_dark_layout(fig, height=300)
    fig.update_layout(yaxis_title="Drawdown (%)", xaxis_title="")
    return fig


def price_with_trades_fig(price_df: pd.DataFrame,
                          trades_df: pd.DataFrame,
                          asset: str) -> go.Figure:
    if price_df.empty:
        return _empty_fig(f"No price data for {asset}.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(price_df["datetime"], utc=True, errors="coerce"),
        y=price_df["close"], mode="lines", name=f"{asset} close",
        line=dict(width=1.5, color=_ACCENT),
    ))
    if not trades_df.empty:
        sub = trades_df[trades_df["asset"] == asset]
        buys = sub[sub["side"] == "BUY"]
        sells = sub[sub["side"] == "SELL"]
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(buys["datetime_iso"], utc=True, errors="coerce"),
                y=buys["price"], mode="markers", name="BUY",
                marker=dict(symbol="triangle-up", size=11, color=_POS,
                            line=dict(color="#0a0e1a", width=1)),
            ))
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(sells["datetime_iso"], utc=True, errors="coerce"),
                y=sells["price"], mode="markers", name="SELL",
                marker=dict(symbol="triangle-down", size=11, color=_NEG,
                            line=dict(color="#0a0e1a", width=1)),
            ))
    _apply_dark_layout(fig, height=380)
    fig.update_layout(yaxis_title="Price (USDT)", xaxis_title="")
    return fig


def cumulative_pnl_fig(trades_df: pd.DataFrame) -> go.Figure:
    if trades_df.empty:
        return _empty_fig("No trades yet.")
    sells = trades_df[trades_df["side"] == "SELL"].copy()
    if sells.empty:
        return _empty_fig("No closed trades yet.")
    sells = sells.sort_values("timestamp_ms")
    sells["cum_pnl"] = sells["realized_pnl"].astype(float).cumsum()
    final = float(sells["cum_pnl"].iloc[-1])
    line_color = _POS if final >= 0 else _NEG
    fill_color = "rgba(16,185,129,0.10)" if final >= 0 else "rgba(244,63,94,0.10)"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(sells["datetime_iso"], utc=True, errors="coerce"),
        y=sells["cum_pnl"], mode="lines+markers", name="Cumulative P&L",
        line=dict(width=2, color=line_color),
        marker=dict(size=6, color=line_color),
        fill="tozeroy", fillcolor=fill_color,
    ))
    _apply_dark_layout(fig, height=320)
    fig.update_layout(yaxis_title="USDT", xaxis_title="")
    return fig


def monthly_returns_fig(equity_df: pd.DataFrame) -> go.Figure:
    if equity_df.empty:
        return _empty_fig("No equity data for monthly returns.")
    df = equity_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime")
    if len(df) < 30:
        return _empty_fig("Not enough history to compute monthly returns.")
    monthly = df["equity"].resample("ME").last().pct_change().dropna() * 100.0
    if monthly.empty:
        return _empty_fig("Not enough history to compute monthly returns.")
    colors = [_POS if v >= 0 else _NEG for v in monthly.values]
    fig = go.Figure(go.Bar(
        x=monthly.index.strftime("%Y-%m"), y=monthly.values,
        marker_color=colors,
        marker_line_width=0,
    ))
    _apply_dark_layout(fig, height=320)
    fig.update_layout(yaxis_title="Monthly return (%)", xaxis_title="")
    return fig
