"""Plotly chart builders used by the Streamlit dashboard.

Plain functions so they can also be imported from notebooks. Every function
gracefully handles empty inputs and returns a figure with a placeholder
annotation rather than raising — this keeps the dashboard from crashing when
no backtest has run yet.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import performance


def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False, font=dict(size=14))
    fig.update_layout(template="plotly_white", height=320,
                      margin=dict(t=20, b=20, l=20, r=20))
    return fig


def equity_curve_fig(equity_df: pd.DataFrame, starting_capital: float) -> go.Figure:
    if equity_df.empty:
        return _empty_fig("No equity curve yet. Run a backtest from the sidebar.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(equity_df["datetime"], utc=True, errors="coerce"),
        y=equity_df["equity"], mode="lines", name="Portfolio equity",
        line=dict(width=2),
    ))
    fig.add_hline(y=starting_capital, line_dash="dash", line_color="gray",
                  annotation_text=f"start: {starting_capital:,.0f}",
                  annotation_position="bottom right")
    fig.update_layout(template="plotly_white", height=380,
                      margin=dict(t=30, b=20, l=20, r=20),
                      yaxis_title="USDT", xaxis_title="")
    return fig


def drawdown_fig(equity_df: pd.DataFrame) -> go.Figure:
    if equity_df.empty:
        return _empty_fig("No drawdown data yet.")
    dd = performance.drawdown_curve(equity_df["equity"].astype(float)) * 100.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(equity_df["datetime"], utc=True, errors="coerce"),
        y=dd, fill="tozeroy", mode="lines",
        line=dict(color="#c0504d", width=1.5),
        name="Drawdown %",
    ))
    fig.update_layout(template="plotly_white", height=300,
                      margin=dict(t=30, b=20, l=20, r=20),
                      yaxis_title="Drawdown (%)", xaxis_title="")
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
        line=dict(width=1.4),
    ))
    if not trades_df.empty:
        sub = trades_df[trades_df["asset"] == asset]
        buys = sub[sub["side"] == "BUY"]
        sells = sub[sub["side"] == "SELL"]
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(buys["datetime_iso"], utc=True, errors="coerce"),
                y=buys["price"], mode="markers", name="BUY",
                marker=dict(symbol="triangle-up", size=10, color="#2ca02c"),
            ))
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(sells["datetime_iso"], utc=True, errors="coerce"),
                y=sells["price"], mode="markers", name="SELL",
                marker=dict(symbol="triangle-down", size=10, color="#d62728"),
            ))
    fig.update_layout(template="plotly_white", height=380,
                      margin=dict(t=30, b=20, l=20, r=20),
                      yaxis_title="Price (USDT)", xaxis_title="",
                      legend=dict(orientation="h", y=1.05))
    return fig


def cumulative_pnl_fig(trades_df: pd.DataFrame) -> go.Figure:
    if trades_df.empty:
        return _empty_fig("No trades yet.")
    sells = trades_df[trades_df["side"] == "SELL"].copy()
    if sells.empty:
        return _empty_fig("No closed trades yet.")
    sells = sells.sort_values("timestamp_ms")
    sells["cum_pnl"] = sells["realized_pnl"].astype(float).cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(sells["datetime_iso"], utc=True, errors="coerce"),
        y=sells["cum_pnl"], mode="lines+markers", name="Cumulative P&L",
        line=dict(width=2),
    ))
    fig.update_layout(template="plotly_white", height=320,
                      margin=dict(t=30, b=20, l=20, r=20),
                      yaxis_title="USDT", xaxis_title="")
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
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in monthly.values]
    fig = go.Figure(go.Bar(
        x=monthly.index.strftime("%Y-%m"), y=monthly.values, marker_color=colors,
    ))
    fig.update_layout(template="plotly_white", height=320,
                      margin=dict(t=30, b=20, l=20, r=20),
                      yaxis_title="Monthly return (%)", xaxis_title="")
    return fig
