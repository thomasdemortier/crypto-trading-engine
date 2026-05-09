"""
CLI entrypoint.

Usage:
    python main.py download
    python main.py backtest
    python main.py paper
    python main.py clean_logs
    python main.py status
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src import (
    backtester, config, data_collector, paper_trader, performance, utils,
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

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
