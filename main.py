"""
CLI entrypoint.

Usage:
    python main.py download
    python main.py backtest
    python main.py paper
    python main.py clean_logs
    python main.py status
    python main.py research_timeframes
    python main.py walk_forward
    python main.py compare_strategies
    python main.py robustness
    python main.py monte_carlo
    python main.py research_all
    python main.py kronos_status
    python main.py kronos_forecast
    python main.py kronos_evaluate
    python main.py kronos_confirm
    python main.py kronos_compare
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from src import (
    backtester, config, data_collector, paper_trader, performance,
    research, utils,
)

logger = utils.get_logger("cte.cli")


def cmd_download(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    paths = data_collector.download_all(
        assets=args.assets, timeframes=args.timeframes,
        days=args.days, refresh=args.refresh,
    )
    print(f"downloaded {len(paths)} dataset(s)")
    for p in paths:
        print(f"  - {p}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    art = backtester.run_backtest(
        assets=args.assets, timeframe=args.timeframe, save=True,
    )
    metrics = performance.compute_metrics(
        art.equity_curve, art.trades, art.asset_close_curves,
        starting_capital=config.RISK.starting_capital,
    )
    performance.save_metrics(metrics)
    print(f"start equity   : {metrics.starting_capital:,.2f}")
    print(f"final equity   : {metrics.final_portfolio_value:,.2f}")
    print(f"total return   : {metrics.total_return_pct:+.2f}%")
    print(f"buy & hold     : {metrics.buy_and_hold_return_pct:+.2f}%")
    print(f"strategy vs B&H: {metrics.strategy_vs_bh_pct:+.2f}%")
    print(f"max drawdown   : {metrics.max_drawdown_pct:.2f}%")
    print(f"trades (closed): {metrics.num_trades}")
    print(f"win rate       : {metrics.win_rate_pct:.1f}%")
    print(f"profit factor  : {metrics.profit_factor:.2f}")
    print(f"sharpe         : {metrics.sharpe_ratio:.2f}")
    print(f"fees paid      : {metrics.fees_paid:,.2f}")
    print(f"slippage cost  : {metrics.slippage_cost:,.2f}")
    return 0


def cmd_paper(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    out = paper_trader.run_tick(timeframe=args.timeframe, assets=args.assets,
                                refresh=args.refresh)
    for asset, info in out.items():
        if "error" in info:
            print(f"  {asset}: ERROR — {info['error']}")
        else:
            print(f"  {asset}: {info['action']} @ {info['price']:.2f} "
                  f"({info['reason']})  eq={info['portfolio_value']:.2f}")
    return 0


def cmd_clean_logs(args: argparse.Namespace) -> int:
    """Wipe logs/ and results/ contents (keeps the directories)."""
    confirm = args.yes or _confirm("Delete all files in logs/ and results/?")
    if not confirm:
        print("aborted.")
        return 1
    for d in (config.LOGS_DIR, config.RESULTS_DIR):
        if d.exists():
            for child in d.iterdir():
                if child.name == ".gitkeep":
                    continue
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
    print("logs/ and results/ cleaned.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    print("=== crypto_trading_engine status ===")
    print(f"LIVE_TRADING_ENABLED : {config.LIVE_TRADING_ENABLED}  (must be False)")
    print(f"primary exchange     : {config.PRIMARY_EXCHANGE}")
    print(f"assets               : {config.ASSETS}")
    print(f"timeframes           : {config.TIMEFRAMES}")
    print(f"starting capital     : {config.RISK.starting_capital:,.2f} {config.RISK.base_currency}")
    print()
    print("Cached datasets:")
    for asset in config.ASSETS:
        for tf in config.TIMEFRAMES:
            p = utils.csv_path_for(asset, tf)
            if p.exists():
                size_kb = p.stat().st_size // 1024
                print(f"  ✓ {asset:<10} {tf:<3}  {size_kb:>5} KB  {p.name}")
            else:
                print(f"  · {asset:<10} {tf:<3}  not downloaded")
    return 0


def _confirm(prompt: str) -> bool:
    try:
        ans = input(f"{prompt} [y/N] ").strip().lower()
        return ans in ("y", "yes")
    except EOFError:
        return False


# ---------------------------------------------------------------------------
# Research CLI commands
# ---------------------------------------------------------------------------
def _print_summary_block(label: str, df) -> None:
    if df is None or df.empty:
        print(f"\n{label}: no rows produced")
        return
    if "error" in df.columns:
        ok = df[df["error"].isna()]
        bad = df[df["error"].notna()]
    else:
        ok, bad = df, df.iloc[0:0]
    print(f"\n{label}: {len(ok)} ok, {len(bad)} skipped")
    if not bad.empty:
        for _, r in bad.head(5).iterrows():
            ident_cols = [c for c in ("strategy", "family", "variant", "asset",
                                      "timeframe", "window") if c in r.index]
            ident = " ".join(f"{c}={r[c]}" for c in ident_cols)
            print(f"  skip [{ident}]: {r['error']}")


def cmd_research_timeframes(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    df = research.timeframe_comparison(
        assets=args.assets, timeframes=args.timeframes, save=True,
    )
    _print_summary_block("timeframe_comparison", df)
    if "error" in df.columns:
        ok = df[df["error"].isna()]
    else:
        ok = df
    if not ok.empty:
        cols = ["asset", "timeframe", "total_return_pct",
                "buy_and_hold_return_pct", "strategy_vs_bh_pct",
                "max_drawdown_pct", "num_trades", "sharpe_ratio"]
        cols = [c for c in cols if c in ok.columns]
        print(ok[cols].to_string(index=False))
    print(f"\nSaved → results/research_timeframe_comparison.csv")
    return 0


def cmd_walk_forward(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    df = research.walk_forward(
        assets=args.assets, timeframes=args.timeframes,
        in_sample_days=args.in_sample_days, oos_days=args.oos_days,
        step_days=args.step_days, save=True,
    )
    _print_summary_block("walk_forward", df)
    if "error" in df.columns:
        ok = df[df["error"].isna()]
        if not ok.empty:
            wins = ((ok["strategy_return_pct"] > 0)
                    & (ok["strategy_vs_bh_pct"] > 0)).sum()
            print(f"  stability score: {wins}/{len(ok)} OOS windows "
                  f"({wins/len(ok)*100:.1f}%) profitable AND beat B&H")
    print(f"\nSaved → results/walk_forward_results.csv")
    return 0


def cmd_compare_strategies(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    df = research.strategy_comparison(
        assets=args.assets, timeframes=args.timeframes, save=True,
    )
    _print_summary_block("strategy_comparison", df)
    if "error" in df.columns:
        ok = df[df["error"].isna()]
    else:
        ok = df
    if not ok.empty:
        cols = ["strategy", "asset", "timeframe", "total_return_pct",
                "buy_and_hold_return_pct", "strategy_vs_bh_pct",
                "max_drawdown_pct", "num_trades"]
        cols = [c for c in cols if c in ok.columns]
        print(ok[cols].to_string(index=False))
    print(f"\nSaved → results/strategy_comparison.csv")
    return 0


def cmd_robustness(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    df = research.robustness(
        assets=args.assets, timeframes=args.timeframes, save=True,
    )
    _print_summary_block("robustness", df)
    if "error" in df.columns:
        ok = df[df["error"].isna()]
    else:
        ok = df
    if not ok.empty:
        for fam in sorted(ok["family"].unique()):
            sub = ok[ok["family"] == fam]
            beats = (sub["strategy_vs_bh_pct"] > 0).sum()
            print(f"  {fam:<14} median_ret={sub['total_return_pct'].median():+.2f}% "
                  f"worst={sub['total_return_pct'].min():+.2f}% "
                  f"best={sub['total_return_pct'].max():+.2f}% "
                  f"beats_BH={beats}/{len(sub)}")
    print(f"\nSaved → results/robustness_results.csv")
    return 0


def cmd_monte_carlo(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    trades_path = config.LOGS_DIR / "trades.csv"
    if not trades_path.exists() or trades_path.stat().st_size == 0:
        print("No saved trades.csv yet — run a backtest first.")
        return 1
    tdf = pd.read_csv(trades_path)
    summary = research.monte_carlo_from_trades(
        tdf, starting_capital=config.RISK.starting_capital,
        n_sim=args.n_sim, save=True,
    )
    if not summary.get("ok"):
        print(f"Monte Carlo not run: {summary.get('reason')}")
        return 0
    print(f"Monte Carlo on {summary['n_trades']} closed trades, "
          f"{summary['n_simulations']} simulations:")
    print(f"  starting capital     : {summary['starting_capital']:,.2f}")
    print(f"  actual final value   : {summary['actual_final_value']:,.2f}")
    print(f"  median final value   : {summary['median_final_value']:,.2f}")
    print(f"  5th  pct final value : {summary['p05_final_value']:,.2f}")
    print(f"  95th pct final value : {summary['p95_final_value']:,.2f}")
    print(f"  prob. of loss        : {summary['prob_loss'] * 100:.1f}%")
    print(f"  worst sim drawdown   : {summary['worst_drawdown_pct']:.2f}%")
    print(f"\nSaved → results/monte_carlo_results.csv "
          f"and results/monte_carlo_simulations.csv")
    return 0


def cmd_research_all(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    bundle = research.run_all(
        assets=args.assets, timeframes=args.timeframes, n_sim=args.n_sim,
    )
    print("\n=== research summary ===")
    for check, v in bundle["summary"]["checks"].items():
        print(f"  [{v['verdict']:<12}] {check}: {v['message']}")
    return 0


# ---------------------------------------------------------------------------
# Optional Kronos commands. These import the ML adapter lazily so users
# without `requirements-ml.txt` installed can still see `python main.py
# --help` and the non-Kronos commands.
# ---------------------------------------------------------------------------
def _print_kronos_install_hint() -> None:
    print("To enable Kronos locally:")
    print("  pip install -r requirements-ml.txt")
    print("  git clone https://github.com/shiyu-coder/Kronos.git external/Kronos")
    print("  python main.py kronos_status")


def cmd_kronos_status(args: argparse.Namespace) -> int:
    from src.ml import kronos_adapter
    status = kronos_adapter.get_kronos_status()
    print("=== Kronos status ===")
    for k, v in status.items():
        print(f"  {k:<28} {v}")
    if not status["available"]:
        print()
        _print_kronos_install_hint()
    return 0


def cmd_kronos_forecast(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    from src.ml import kronos_adapter
    if not kronos_adapter.kronos_available():
        print("Kronos not available. Status:")
        for k, v in kronos_adapter.get_kronos_status().items():
            print(f"  {k:<28} {v}")
        _print_kronos_install_hint()
        return 1
    candles = data_collector.load_candles(args.asset, args.timeframe)
    pred_df = kronos_adapter.run_kronos_forecast(
        candles=candles, timeframe=args.timeframe,
        model_name=args.model, lookback=args.lookback,
        pred_len=args.pred_len, device=args.device,
    )
    out_path = config.RESULTS_DIR / "kronos_forecast.csv"
    utils.write_df(pred_df, out_path)
    print(f"Forecast saved to {out_path} ({len(pred_df)} rows).")
    if "close" in pred_df.columns and not pred_df.empty:
        last_close = float(candles["close"].iloc[-1])
        f_close = float(pred_df["close"].iloc[-1])
        print(f"  current close : {last_close:.2f}")
        print(f"  forecast close: {f_close:.2f}")
        print(f"  forecast Δ%   : {(f_close / last_close - 1) * 100:+.2f}%")
    return 0


def cmd_kronos_evaluate(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    from src.ml import kronos_adapter, forecast_evaluation
    if not kronos_adapter.kronos_available():
        print("Kronos not available — aborting evaluation.")
        _print_kronos_install_hint()
        return 1
    df = forecast_evaluation.evaluate_kronos_forecasts(
        asset=args.asset, timeframe=args.timeframe,
        model_name=args.model, lookback=args.lookback,
        pred_len=args.pred_len, step=args.step,
        max_windows=args.max_windows, device=args.device,
    )
    summary = forecast_evaluation.summarise_forecast_evaluation(df)
    print(f"\nKronos evaluation: {summary}")
    print(f"Saved → results/kronos_forecast_evaluation.csv ({len(df)} windows).")
    return 0


def cmd_kronos_confirm(args: argparse.Namespace) -> int:
    """Generate confirmations from the saved forecast evaluation. This step
    does NOT require Kronos to be installed — it only needs an existing
    forecast_evaluation CSV and a base-signals CSV (auto-generated from a
    fresh backtest if missing)."""
    utils.assert_paper_only()
    from src.ml import kronos_confirmation
    fe_path = config.RESULTS_DIR / "kronos_forecast_evaluation.csv"
    if not fe_path.exists():
        print(f"Forecast evaluation CSV not found at {fe_path}. "
              "Run `python main.py kronos_evaluate` first.")
        return 1
    fc_eval_df = pd.read_csv(fe_path)

    # Build a base-signals frame by re-running the backtester for the chosen
    # base strategy and reading its decisions log.
    from src.strategies import REGISTRY as _STRATS
    if args.base_strategy not in _STRATS:
        print(f"Unknown base strategy {args.base_strategy!r}. "
              f"Known: {list(_STRATS)}")
        return 2
    art = backtester.run_backtest(
        assets=[args.asset], timeframe=args.timeframe, save=False,
        strategy=_STRATS[args.base_strategy](),
    )
    decisions = art.decisions.copy()
    if decisions.empty:
        print("Base backtest produced no decisions to confirm.")
        return 3
    decisions = decisions.rename(columns={"timestamp_ms": "timestamp_ms"})
    decisions["timeframe"] = args.timeframe
    base_signals = decisions[["timestamp_ms", "asset", "timeframe", "action"]].copy()
    confirmations = kronos_confirmation.generate_kronos_confirmations(
        base_signals_df=base_signals,
        forecast_eval_df=fc_eval_df,
    )
    print(f"Generated {len(confirmations)} confirmations.")
    print(confirmations["confirmation"].value_counts().to_string())
    print(f"Saved → results/kronos_confirmations.csv")
    return 0


def cmd_kronos_compare(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    from src.ml import forecast_evaluation
    df = forecast_evaluation.compare_base_vs_kronos_confirmed(
        asset=args.asset, timeframe=args.timeframe,
        base_strategy_name=args.base_strategy,
    )
    print("\nBase vs Kronos-confirmed:")
    cols = ["variant", "total_return_pct", "buy_and_hold_return_pct",
            "strategy_vs_bh_pct", "max_drawdown_pct", "num_trades",
            "profit_factor", "fees_paid"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))
    print(f"\nSaved → results/kronos_confirmation_comparison.csv")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="crypto_trading_engine",
                                description="Research-only BTC/ETH backtester.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("download", help="Download OHLCV data.")
    sp.add_argument("--assets", nargs="+", default=config.ASSETS)
    sp.add_argument("--timeframes", nargs="+", default=config.TIMEFRAMES)
    sp.add_argument("--days", type=int, default=config.DEFAULT_HISTORY_DAYS)
    sp.add_argument("--refresh", action="store_true",
                    help="Re-download even if cached.")
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("backtest", help="Run a backtest.")
    sp.add_argument("--assets", nargs="+", default=config.ASSETS)
    sp.add_argument("--timeframe", default=config.DEFAULT_TIMEFRAME)
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("paper", help="Run a paper-trading tick (simulated only).")
    sp.add_argument("--assets", nargs="+", default=config.ASSETS)
    sp.add_argument("--timeframe", default=config.DEFAULT_TIMEFRAME)
    sp.add_argument("--refresh", action="store_true", default=True)
    sp.set_defaults(func=cmd_paper)

    sp = sub.add_parser("clean_logs", help="Delete logs/ and results/ contents.")
    sp.add_argument("-y", "--yes", action="store_true",
                    help="Don't prompt for confirmation.")
    sp.set_defaults(func=cmd_clean_logs)

    sp = sub.add_parser("status", help="Print configuration and cache status.")
    sp.set_defaults(func=cmd_status)

    # ----- Research commands ------------------------------------------------
    research_assets_default = ["BTC/USDT", "ETH/USDT"]
    research_tfs_default = ["1h", "4h", "1d"]

    sp = sub.add_parser("research_timeframes",
                        help="Run incumbent strategy across assets × timeframes.")
    sp.add_argument("--assets", nargs="+", default=research_assets_default)
    sp.add_argument("--timeframes", nargs="+", default=research_tfs_default)
    sp.set_defaults(func=cmd_research_timeframes)

    sp = sub.add_parser("walk_forward",
                        help="Walk-forward analysis over rolling OOS windows.")
    sp.add_argument("--assets", nargs="+", default=research_assets_default)
    sp.add_argument("--timeframes", nargs="+", default=research_tfs_default)
    sp.add_argument("--in-sample-days", type=int, default=90, dest="in_sample_days")
    sp.add_argument("--oos-days", type=int, default=30, dest="oos_days")
    sp.add_argument("--step-days", type=int, default=30, dest="step_days")
    sp.set_defaults(func=cmd_walk_forward)

    sp = sub.add_parser("compare_strategies",
                        help="Run multiple strategies through the same risk engine.")
    sp.add_argument("--assets", nargs="+", default=research_assets_default)
    sp.add_argument("--timeframes", nargs="+", default=research_tfs_default)
    sp.set_defaults(func=cmd_compare_strategies)

    sp = sub.add_parser("robustness",
                        help="Sweep small parameter variations per strategy family.")
    sp.add_argument("--assets", nargs="+", default=research_assets_default)
    sp.add_argument("--timeframes", nargs="+", default=research_tfs_default)
    sp.set_defaults(func=cmd_robustness)

    sp = sub.add_parser("monte_carlo",
                        help="Trade-order Monte Carlo on saved trades.csv.")
    sp.add_argument("--n-sim", type=int, default=1000, dest="n_sim")
    sp.set_defaults(func=cmd_monte_carlo)

    sp = sub.add_parser("research_all",
                        help="Run every research lens and save results + summary.")
    sp.add_argument("--assets", nargs="+", default=research_assets_default)
    sp.add_argument("--timeframes", nargs="+", default=research_tfs_default)
    sp.add_argument("--n-sim", type=int, default=1000, dest="n_sim")
    sp.set_defaults(func=cmd_research_all)

    # ----- Optional Kronos commands (lazy import; no-op if unavailable) ----
    sp = sub.add_parser("kronos_status",
                        help="Check optional Kronos availability and config.")
    sp.set_defaults(func=cmd_kronos_status)

    sp = sub.add_parser("kronos_forecast",
                        help="Run one Kronos forecast on the latest candles.")
    sp.add_argument("--asset", default="BTC/USDT")
    sp.add_argument("--timeframe", default=config.DEFAULT_TIMEFRAME)
    sp.add_argument("--model", default="Kronos-mini")
    sp.add_argument("--lookback", type=int, default=400)
    sp.add_argument("--pred-len", type=int, default=24, dest="pred_len")
    sp.add_argument("--device", default="cpu")
    sp.set_defaults(func=cmd_kronos_forecast)

    sp = sub.add_parser("kronos_evaluate",
                        help="Rolling-window Kronos forecast evaluation.")
    sp.add_argument("--asset", default="BTC/USDT")
    sp.add_argument("--timeframe", default=config.DEFAULT_TIMEFRAME)
    sp.add_argument("--model", default="Kronos-mini")
    sp.add_argument("--lookback", type=int, default=400)
    sp.add_argument("--pred-len", type=int, default=24, dest="pred_len")
    sp.add_argument("--step", type=int, default=24)
    sp.add_argument("--max-windows", type=int, default=20, dest="max_windows")
    sp.add_argument("--device", default="cpu")
    sp.set_defaults(func=cmd_kronos_evaluate)

    sp = sub.add_parser("kronos_confirm",
                        help="Generate kronos_confirmations.csv from saved evaluation.")
    sp.add_argument("--asset", default="BTC/USDT")
    sp.add_argument("--timeframe", default=config.DEFAULT_TIMEFRAME)
    sp.add_argument("--base-strategy", default="rsi_ma_atr",
                    dest="base_strategy")
    sp.set_defaults(func=cmd_kronos_confirm)

    sp = sub.add_parser("kronos_compare",
                        help="Backtest base vs Kronos-confirmed wrapper.")
    sp.add_argument("--asset", default="BTC/USDT")
    sp.add_argument("--timeframe", default=config.DEFAULT_TIMEFRAME)
    sp.add_argument("--base-strategy", default="rsi_ma_atr",
                    dest="base_strategy")
    sp.set_defaults(func=cmd_kronos_compare)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
