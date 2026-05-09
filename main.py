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
    python main.py regimes
    python main.py scorecard
    python main.py audit_oos
    python main.py placebo_audit
    python main.py portfolio_momentum
    python main.py portfolio_walk_forward
    python main.py portfolio_placebo
    python main.py portfolio_scorecard
    python main.py research_all_portfolio
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from src import (
    backtester, config, crypto_regime_signals, data_collector,
    market_structure_data_audit, market_structure_data_collector,
    market_structure_research, market_structure_signals,
    oos_audit, paper_trader, performance, portfolio_audit, portfolio_research,
    research, utils,
)

logger = utils.get_logger("cte.cli")


def cmd_download(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    # `--lookback-days` is the spec name; `--days` is the older alias.
    lookback = getattr(args, "lookback_days", None) or args.days
    paths = data_collector.download_all(
        assets=args.assets, timeframes=args.timeframes,
        days=lookback, refresh=args.refresh,
    )
    print(f"downloaded {len(paths)} dataset(s)")
    for p in paths:
        print(f"  - {p}")
    # Always emit the coverage audit so the user immediately sees whether
    # the requested lookback was actually achievable.
    coverage = research.data_coverage_audit(
        assets=args.assets, timeframes=args.timeframes,
        requested_lookback_days=lookback,
    )
    if not coverage.empty:
        print("\nData coverage:")
        cols = [c for c in ["asset", "timeframe", "candle_count",
                            "coverage_days", "enough_for_walk_forward",
                            "notes"] if c in coverage.columns]
        print(coverage[cols].to_string(index=False))
    print("\nSaved → results/data_coverage.csv")
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
    stages = list(args.stage) if args.stage else ["all"]
    bundle = research.run_stages(
        stages=stages,
        assets=args.assets, timeframes=args.timeframes, n_sim=args.n_sim,
        skip_robustness=bool(getattr(args, "skip_robustness", False)),
        strategy_filter=getattr(args, "strategy", None),
    )
    summary = bundle.get("summary")
    if summary is not None:
        print("\n=== research summary ===")
        for check, v in summary["checks"].items():
            print(f"  [{v['verdict']:<12}] {check}: {v['message']}")
    sc = bundle.get("scorecard_df")
    if sc is not None and not sc.empty:
        print("\n=== top scorecard rows ===")
        cols = ["strategy_name", "asset", "timeframe", "total_score", "verdict"]
        cols = [c for c in cols if c in sc.columns]
        print(sc.head(10)[cols].to_string(index=False))
    print(f"\nstages completed: {bundle.get('summary') is not None and 'all' or '(partial)'}")
    print(f"see results/research_run_state.json for the per-stage timeline.")
    return 0


def cmd_regimes(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    df = research.regime_analysis(assets=args.assets, timeframes=args.timeframes)
    if df.empty:
        print("No regime rows produced — check candle availability.")
        return 0
    cols = [c for c in ["asset", "timeframe", "n_bars", "pct_bull",
                        "pct_bear", "pct_sideways", "pct_high_vol",
                        "pct_low_vol"] if c in df.columns]
    print(df[cols].to_string(index=False))
    print(f"\nSaved → results/regime_summary.csv "
          f"(+ per-asset regime_<sym>_<tf>.csv files)")
    return 0


def _print_portfolio_metrics_dict(label: str, m: dict) -> None:
    print(f"  {label:<30} return={m['total_return_pct']:+.2f}% "
          f"dd={m['max_drawdown_pct']:.2f}% "
          f"sharpe={m['sharpe_ratio']:.2f}")


def cmd_portfolio_momentum(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    out = portfolio_research.run_portfolio_momentum(
        assets=args.assets, timeframe=args.timeframe,
    )
    if not out.get("ok"):
        print(f"portfolio_momentum: {out.get('reason')}")
        return 1
    print("\n=== portfolio momentum vs benchmarks ===")
    _print_portfolio_metrics_dict("momentum_rotation", out["metrics"])
    for name, m in out["bench_metrics"].items():
        _print_portfolio_metrics_dict(name, m)
    print("\nSaved → results/portfolio_momentum_*.csv")
    return 0


def cmd_portfolio_walk_forward(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    df = portfolio_research.portfolio_walk_forward(
        assets=args.assets, timeframe=args.timeframe,
        in_sample_days=args.in_sample_days, oos_days=args.oos_days,
        step_days=args.step_days,
    )
    if df.empty:
        print("portfolio_walk_forward: no rows produced.")
        return 1
    cols = [c for c in ["window", "oos_start_iso", "oos_end_iso",
            "oos_return_pct", "oos_max_drawdown_pct", "btc_oos_return_pct",
            "basket_oos_return_pct", "beats_btc", "beats_basket",
            "n_rebalances", "avg_holdings"] if c in df.columns]
    print(df[cols].to_string(index=False))
    print(f"\nSaved → results/portfolio_walk_forward.csv ({len(df)} windows).")
    return 0


def cmd_portfolio_placebo(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    df = portfolio_research.portfolio_placebo(
        assets=args.assets, timeframe=args.timeframe,
        seeds=tuple(range(args.n_seeds)),
    )
    if df.empty:
        print("portfolio_placebo: no rows produced.")
        return 1
    summary = df.iloc[0].to_dict()
    print("\n=== portfolio placebo summary ===")
    for k in ("strategy_return_pct", "placebo_median_return_pct",
              "strategy_max_drawdown_pct", "placebo_median_drawdown_pct",
              "strategy_beats_median_return",
              "strategy_beats_median_drawdown"):
        if k in summary:
            print(f"  {k:<35} {summary[k]}")
    print(f"\nSaved → results/portfolio_placebo_comparison.csv "
          f"({args.n_seeds} seeds).")
    return 0


def cmd_portfolio_scorecard(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    df = portfolio_research.portfolio_scorecard()
    if df.empty:
        print("portfolio_scorecard: no rows.")
        return 1
    row = df.iloc[0].to_dict()
    print("\n=== portfolio scorecard ===")
    for k in ("strategy_name", "n_windows", "verdict",
              "avg_oos_return_pct", "avg_oos_drawdown_pct",
              "pct_windows_beat_btc", "pct_windows_beat_basket",
              "stability_score_pct", "total_rebalances",
              "beats_placebo_median", "checks_passed", "checks_total",
              "reason"):
        if k in row:
            print(f"  {k:<30} {row[k]}")
    print(f"\nSaved → results/portfolio_scorecard.csv")
    return 0


def cmd_crypto_regime_signals(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    df = crypto_regime_signals.compute_regime_signals(
        timeframe=args.timeframe, save=True,
    )
    if df.empty:
        print("crypto_regime_signals: no rows produced (BTC data missing?)")
        return 1
    dist = crypto_regime_signals.regime_distribution(df)
    print(f"crypto_regime_signals: {len(df)} rows.")
    print(f"  span: {pd.to_datetime(df['datetime'].iloc[0]).date()} -> "
          f"{pd.to_datetime(df['datetime'].iloc[-1]).date()}")
    print(f"  risk_state distribution (% of bars): {dist}")
    print(f"\nSaved → results/crypto_regime_signals.csv")
    return 0


def cmd_regime_aware_portfolio(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    out = portfolio_research.run_regime_aware_portfolio(
        assets=args.assets, timeframe=args.timeframe,
    )
    if not out.get("ok"):
        print(f"regime_aware_portfolio: {out.get('reason')}")
        return 1
    print("\n=== regime-aware portfolio vs benchmarks (full window) ===")
    _print_portfolio_metrics_dict(
        "regime_aware_momentum_rotation", out["metrics"],
    )
    _print_portfolio_metrics_dict(
        "momentum_rotation_simple", out["simple_metrics"],
    )
    for name, m in out["bench_metrics"].items():
        _print_portfolio_metrics_dict(name, m)
    print("\nSaved → results/regime_aware_portfolio_*.csv")
    return 0


def cmd_regime_aware_portfolio_walk_forward(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    df = portfolio_research.regime_aware_portfolio_walk_forward(
        assets=args.assets, timeframe=args.timeframe,
        in_sample_days=args.in_sample_days, oos_days=args.oos_days,
        step_days=args.step_days,
    )
    if df.empty:
        print("regime_aware_portfolio_walk_forward: no rows produced.")
        return 1
    cols = [c for c in ["window", "oos_start_iso", "oos_end_iso",
            "oos_return_pct", "btc_oos_return_pct", "basket_oos_return_pct",
            "simple_oos_return_pct", "beats_btc", "beats_basket",
            "beats_simple_momentum", "n_rebalances"] if c in df.columns]
    print(df[cols].to_string(index=False))
    print(f"\nSaved → results/regime_aware_portfolio_walk_forward.csv "
          f"({len(df)} windows).")
    return 0


def cmd_regime_aware_portfolio_placebo(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    df = portfolio_research.regime_aware_portfolio_placebo(
        assets=args.assets, timeframe=args.timeframe,
        seeds=tuple(range(args.n_seeds)),
    )
    if df.empty:
        print("regime_aware_portfolio_placebo: no rows.")
        return 1
    summary = df.iloc[0].to_dict()
    print("\n=== regime-aware placebo summary ===")
    for k in ("strategy_return_pct", "placebo_median_return_pct",
              "strategy_max_drawdown_pct", "placebo_median_drawdown_pct",
              "strategy_beats_median_return",
              "strategy_beats_median_drawdown"):
        if k in summary:
            print(f"  {k:<35} {summary[k]}")
    print(f"\nSaved → results/regime_aware_portfolio_placebo.csv")
    return 0


def cmd_regime_aware_portfolio_scorecard(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    df = portfolio_research.regime_aware_portfolio_scorecard()
    if df.empty:
        print("regime_aware_portfolio_scorecard: no rows.")
        return 1
    row = df.iloc[0].to_dict()
    print("\n=== regime-aware portfolio scorecard ===")
    for k in ("strategy_name", "n_windows", "verdict",
              "avg_oos_return_pct", "avg_oos_drawdown_pct",
              "pct_windows_beat_btc", "pct_windows_beat_basket",
              "pct_windows_beat_simple_momentum",
              "stability_score_pct", "total_rebalances",
              "beats_placebo_median", "checks_passed", "checks_total",
              "reason"):
        if k in row:
            print(f"  {k:<32} {row[k]}")
    print(f"\nSaved → results/regime_aware_portfolio_scorecard.csv")
    return 0


def cmd_audit_portfolio(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    summary = portfolio_audit.audit_all_portfolio()
    cf = summary["cash_filter"]
    print("=== portfolio cash filter audit ===")
    if not cf.get("ok"):
        print(f"  not ok: {cf.get('reason')}")
        return 1
    print(f"  BTC data span:               {cf['btc_data_first']} -> {cf['btc_data_last']}")
    print(f"  Evaluable bars (post warmup): {cf['n_eval_bars_after_warmup']}")
    print(f"  Bearish bars (close < 200d): {cf['n_bearish_bars']} ({cf['pct_bearish']}%)")
    print(f"  First / last bearish:        {cf['first_bearish_date']} / {cf['last_bearish_date']}")
    print(f"  Contiguous bearish spans:    {cf['n_contiguous_bearish_spans']} (longest {cf['longest_bearish_span_days']} days)")
    print(f"  Weights file present:        {cf['weights_file_present']}")
    for row in cf.get("cross_check", []):
        print(f"  cross-check: {row['metric']:<60} {row['value']}")
    print(f"\n=== benchmark alignment audit ===")
    print(f"  rows audited:                {summary['benchmark_alignment_rows']}")
    print(f"  BTC drift within tolerance:  {summary['benchmark_alignment_ok_btc']}")
    print(f"  Basket drift within tol:     {summary['benchmark_alignment_ok_basket']}")
    print(f"\n=== rebalance logic audit ===")
    print(f"  all checks ok:               {summary['rebalance_audit_all_ok']}")
    print(f"\nSaved → results/portfolio_cash_filter_audit.csv")
    print(f"Saved → results/portfolio_benchmark_alignment_audit.csv")
    print(f"Saved → results/portfolio_rebalance_audit.csv")
    return 0


def cmd_research_all_portfolio(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    out = portfolio_research.run_all_portfolio(
        assets=args.assets, timeframe=args.timeframe,
        in_sample_days=args.in_sample_days, oos_days=args.oos_days,
        step_days=args.step_days, seeds=tuple(range(args.n_seeds)),
    )
    sc = out.get("scorecard")
    if sc is not None and not sc.empty:
        print("\n=== portfolio scorecard ===")
        print(sc.to_string(index=False))
    return 0


def cmd_audit_oos(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    audit_df, summary_df = oos_audit.audit_walk_forward(save=True)
    if audit_df.empty:
        print("No walk_forward_by_strategy.csv found. "
              "Run `python main.py research_all` first.")
        return 1
    print(f"OOS audit: {len(audit_df)} per-window rows, "
          f"{len(summary_df)} (strategy, asset, timeframe) groups.")
    if not summary_df.empty:
        cols = ["strategy_name", "asset", "timeframe", "n_windows",
                "stability_score_pct", "mean_trades_per_window",
                "windows_overlap", "enough_windows_for_confidence", "notes"]
        cols = [c for c in cols if c in summary_df.columns]
        print(summary_df[cols].to_string(index=False))
    print(f"\nSaved → results/oos_audit.csv "
          f"and results/oos_audit_summary.csv")
    return 0


def cmd_placebo_audit(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    df = research.placebo_comparison(save=True)
    if df.empty:
        print("No placebo comparison rows. Run `python main.py research_all` "
              "first to populate walk_forward_by_strategy.csv.")
        return 1
    cols = ["strategy_name", "asset", "timeframe",
            "strategy_oos_stability", "placebo_oos_stability",
            "strategy_mean_oos_return", "placebo_mean_oos_return",
            "strategy_beats_placebo", "notes"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))
    n_beat = int(df["strategy_beats_placebo"].sum())
    print(f"\n{n_beat}/{len(df)} (strategy, asset, timeframe) rows beat the placebo.")
    print("Saved → results/placebo_comparison.csv")
    return 0


def cmd_scorecard(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    sc = research.build_scorecard_from_saved(save=True)
    if sc.empty:
        print("No scorecard rows. Run `python main.py compare_strategies` "
              "first (and ideally walk_forward + robustness too).")
        return 1
    cols = ["strategy_name", "asset", "timeframe", "total_score", "verdict",
            "benchmark_score", "walk_forward_score", "robustness_score",
            "trade_count_score", "drawdown_score"]
    cols = [c for c in cols if c in sc.columns]
    print(sc.head(40)[cols].to_string(index=False))
    print(f"\nSaved → results/strategy_scorecard.csv ({len(sc)} rows).")
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


def cmd_download_market_structure_data(args: argparse.Namespace) -> int:
    """Download every free market-structure dataset, write coverage CSV."""
    utils.assert_paper_only()
    res = market_structure_data_collector.download_all_market_structure(
        refresh=args.refresh,
        sleep_seconds=args.sleep_seconds,
        lookback_days=args.lookback_days,
    )
    cov = res["coverage_df"]
    print(f"download_market_structure_data: {len(res['paths'])} datasets "
          f"persisted under data/market_structure/")
    if not cov.empty:
        cols = [c for c in ["source", "dataset", "row_count",
                            "coverage_days", "enough_for_research", "notes"]
                if c in cov.columns]
        print(cov[cols].to_string(index=False))
    print("\nSaved → results/market_structure_data_coverage.csv")
    return 0


def cmd_market_structure_signals(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    df = market_structure_signals.compute_market_structure_signals(save=True)
    if df.empty:
        print("market_structure_signals: no rows produced.")
        return 1
    by_state = df["market_structure_state"].value_counts().to_dict()
    print(f"market_structure_signals: {len(df)} rows.")
    print(f"  state distribution: {by_state}")
    print("\nSaved → results/market_structure_signals.csv")
    return 0


def cmd_market_structure_allocator(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    out = market_structure_research.run_market_structure_allocator(
        timeframe=args.timeframe,
    )
    if not out.get("ok"):
        print(f"market_structure_allocator: {out.get('reason')}")
        return 1
    print("\n=== market structure allocator vs benchmarks ===")
    _print_portfolio_metrics_dict("market_structure_allocator", out["metrics"])
    for name, m in out["bench_metrics"].items():
        _print_portfolio_metrics_dict(name, m)
    print("\nSaved → results/market_structure_allocator_*.csv")
    return 0


def cmd_market_structure_walk_forward(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    df = market_structure_research.market_structure_walk_forward(
        timeframe=args.timeframe,
        in_sample_days=args.in_sample_days,
        oos_days=args.oos_days, step_days=args.step_days,
    )
    if df.empty:
        print("market_structure_walk_forward: no windows fit available history.")
        return 1
    cols = [c for c in ["window", "oos_start_iso", "oos_end_iso",
                         "oos_return_pct", "btc_oos_return_pct",
                         "basket_oos_return_pct", "simple_oos_return_pct",
                         "beats_btc", "beats_basket",
                         "beats_simple_momentum", "n_rebalances"]
            if c in df.columns]
    print(df[cols].to_string(index=False))
    print(f"\nSaved → results/market_structure_walk_forward.csv "
          f"({len(df)} windows).")
    return 0


def cmd_market_structure_placebo(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    df = market_structure_research.market_structure_placebo(
        timeframe=args.timeframe,
        seeds=tuple(range(args.n_seeds)),
    )
    if df.empty:
        print("market_structure_placebo: no rows.")
        return 1
    summary = df.iloc[0].to_dict()
    print("\n=== market structure placebo summary ===")
    for k in ("strategy_return_pct", "placebo_median_return_pct",
              "strategy_max_drawdown_pct", "placebo_median_drawdown_pct",
              "strategy_beats_median_return",
              "strategy_beats_median_drawdown"):
        if k in summary:
            print(f"  {k:<35} {summary[k]}")
    print(f"\nSaved → results/market_structure_placebo.csv "
          f"({args.n_seeds} seeds).")
    return 0


def cmd_market_structure_scorecard(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    df = market_structure_research.market_structure_scorecard()
    if df.empty:
        print("market_structure_scorecard: no rows.")
        return 1
    row = df.iloc[0].to_dict()
    print("\n=== market structure scorecard ===")
    for k in ("strategy_name", "n_windows", "verdict",
              "avg_oos_return_pct", "avg_oos_drawdown_pct",
              "pct_windows_beat_btc", "pct_windows_beat_basket",
              "pct_windows_beat_simple_momentum",
              "stability_score_pct", "total_rebalances",
              "strategy_full_drawdown_pct", "btc_full_drawdown_pct",
              "dd_gap_pp", "beats_placebo_median",
              "coverage_note", "checks_passed", "checks_total", "reason"):
        if k in row:
            print(f"  {k:<32} {row[k]}")
    print("\nSaved → results/market_structure_scorecard.csv")
    return 0


def cmd_market_structure_vol_target(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    out = market_structure_research.run_market_structure_vol_target(
        timeframe=args.timeframe,
    )
    if not out.get("ok"):
        print(f"market_structure_vol_target: {out.get('reason')}")
        return 1
    print("\n=== vol-target allocator vs benchmarks ===")
    _print_portfolio_metrics_dict("market_structure_vol_target",
                                    out["metrics"])
    for name, m in out["bench_metrics"].items():
        _print_portfolio_metrics_dict(name, m)
    print("\nSaved → results/market_structure_vol_target_*.csv")
    return 0


def cmd_market_structure_vol_target_walk_forward(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    df = market_structure_research.market_structure_vol_target_walk_forward(
        timeframe=args.timeframe,
        in_sample_days=args.in_sample_days,
        oos_days=args.oos_days, step_days=args.step_days,
    )
    if df.empty:
        print("vol-target walk-forward: no windows fit available history.")
        return 1
    cols = [c for c in ["window", "oos_start_iso", "oos_end_iso",
            "oos_return_pct", "btc_oos_return_pct", "basket_oos_return_pct",
            "simple_oos_return_pct", "original_allocator_oos_return_pct",
            "beats_btc", "beats_basket", "beats_simple_momentum",
            "beats_original_allocator", "n_rebalances"]
            if c in df.columns]
    print(df[cols].to_string(index=False))
    print(f"\nSaved → results/market_structure_vol_target_walk_forward.csv "
          f"({len(df)} windows).")
    return 0


def cmd_market_structure_vol_target_placebo(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    df = market_structure_research.market_structure_vol_target_placebo(
        timeframe=args.timeframe,
        seeds=tuple(range(args.n_seeds)),
    )
    if df.empty:
        print("vol-target placebo: no rows.")
        return 1
    summary = df.iloc[0].to_dict()
    print("\n=== vol-target placebo summary ===")
    for k in ("strategy_return_pct", "placebo_median_return_pct",
              "strategy_max_drawdown_pct", "placebo_median_drawdown_pct",
              "strategy_beats_median_return",
              "strategy_beats_median_drawdown"):
        if k in summary:
            print(f"  {k:<35} {summary[k]}")
    print(f"\nSaved → results/market_structure_vol_target_placebo.csv "
          f"({args.n_seeds} seeds).")
    return 0


def cmd_market_structure_vol_target_scorecard(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    df = market_structure_research.market_structure_vol_target_scorecard()
    if df.empty:
        print("vol-target scorecard: no rows.")
        return 1
    row = df.iloc[0].to_dict()
    print("\n=== vol-target scorecard ===")
    for k in ("strategy_name", "n_windows", "verdict",
              "avg_oos_return_pct", "avg_oos_drawdown_pct",
              "pct_windows_beat_btc", "pct_windows_beat_basket",
              "pct_windows_beat_simple_momentum",
              "pct_windows_beat_original_allocator",
              "stability_score_pct", "total_rebalances",
              "strategy_full_drawdown_pct", "btc_full_drawdown_pct",
              "dd_gap_pp", "beats_placebo_median",
              "coverage_note", "checks_passed", "checks_total", "reason"):
        if k in row:
            print(f"  {k:<38} {row[k]}")
    print("\nSaved → results/market_structure_vol_target_scorecard.csv")
    return 0


def cmd_research_all_market_structure(args: argparse.Namespace) -> int:
    utils.assert_paper_only()
    out = market_structure_research.run_all_market_structure(
        timeframe=args.timeframe,
        in_sample_days=args.in_sample_days,
        oos_days=args.oos_days, step_days=args.step_days,
        seeds=tuple(range(args.n_seeds)),
    )
    sc = out.get("scorecard")
    if sc is None or sc.empty:
        print("research_all_market_structure: scorecard empty.")
        return 1
    row = sc.iloc[0].to_dict()
    print("\n=== research_all_market_structure FINAL ===")
    print(f"  verdict:       {row.get('verdict')}")
    print(f"  reason:        {row.get('reason')}")
    print(f"  n_windows:     {row.get('n_windows')}")
    print(f"  beats_placebo: {row.get('beats_placebo_median')}")
    print(f"  coverage:      {row.get('coverage_note')}")
    return 0


def cmd_audit_market_structure_data(args: argparse.Namespace) -> int:
    """Probe free public market-structure data sources, write
    `results/market_structure_data_audit.csv`, print a verdict."""
    utils.assert_paper_only()
    df = market_structure_data_audit.audit_market_structure_data(save=True)
    if df.empty:
        print("audit_market_structure_data: no rows produced.")
        return 1
    cols = [c for c in ["source", "dataset", "row_count",
                         "coverage_days", "usable_for_research", "notes"]
            if c in df.columns]
    print(df[cols].to_string(index=False))
    n_usable = int(df["usable_for_research"].sum())
    print(f"\n{n_usable} of {len(df)} sources are usable "
          f"(≥{market_structure_data_audit.MIN_DAYS_FOR_RESEARCH}d daily).")
    print("\nSaved → results/market_structure_data_audit.csv")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="crypto_trading_engine",
                                description="Research-only BTC/ETH backtester.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("download", help="Download OHLCV data.")
    sp.add_argument("--assets", nargs="+", default=config.ASSETS)
    sp.add_argument("--timeframes", nargs="+", default=config.TIMEFRAMES)
    sp.add_argument("--days", type=int, default=config.DEFAULT_HISTORY_DAYS)
    sp.add_argument("--lookback-days", type=int, default=None,
                    dest="lookback_days",
                    help="Alias for --days; takes precedence if both set.")
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
    sp.add_argument(
        "--stage", nargs="+", default=None,
        help=("Run only specific stages. Choices: data_coverage, regimes, "
              "strategy_comparison, walk_forward, robustness, scorecard, "
              "monte_carlo, oos_audit, placebo_audit, summary. "
              "Default = all stages in order. Repeat to run multiple."),
    )
    sp.add_argument(
        "--skip-robustness", action="store_true", dest="skip_robustness",
        help=("Skip the robustness sweep. Use this when iterating quickly "
              "or when 1h data makes the sweep too slow."),
    )
    sp.add_argument(
        "--strategy", default=None,
        help=("Limit comparison + walk-forward + robustness to a single "
              "strategy / family name (e.g. 'regime_selector')."),
    )
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

    # ----- Regime + scorecard -----------------------------------------------
    sp = sub.add_parser("regimes",
                        help="Per (asset, timeframe) market regime distribution.")
    sp.add_argument("--assets", nargs="+", default=research_assets_default)
    sp.add_argument("--timeframes", nargs="+", default=research_tfs_default)
    sp.set_defaults(func=cmd_regimes)

    sp = sub.add_parser("scorecard",
                        help="Rebuild strategy scorecard from saved CSVs.")
    sp.set_defaults(func=cmd_scorecard)

    # ----- Portfolio momentum rotation (multi-asset research) ---------------
    portfolio_assets_default = list(config.EXPANDED_UNIVERSE)
    portfolio_tf_default = "1d"

    sp = sub.add_parser("portfolio_momentum",
                        help="Run a single-window momentum rotation backtest "
                             "over the expanded asset universe.")
    sp.add_argument("--assets", nargs="+", default=portfolio_assets_default)
    sp.add_argument("--timeframe", default=portfolio_tf_default)
    sp.set_defaults(func=cmd_portfolio_momentum)

    sp = sub.add_parser("portfolio_walk_forward",
                        help="Walk-forward the momentum rotation strategy.")
    sp.add_argument("--assets", nargs="+", default=portfolio_assets_default)
    sp.add_argument("--timeframe", default=portfolio_tf_default)
    sp.add_argument("--in-sample-days", type=int, default=180,
                    dest="in_sample_days")
    sp.add_argument("--oos-days", type=int, default=90, dest="oos_days")
    sp.add_argument("--step-days", type=int, default=90, dest="step_days")
    sp.set_defaults(func=cmd_portfolio_walk_forward)

    sp = sub.add_parser("portfolio_placebo",
                        help="Compare momentum rotation vs random-rotation "
                             "placebo across N seeds.")
    sp.add_argument("--assets", nargs="+", default=portfolio_assets_default)
    sp.add_argument("--timeframe", default=portfolio_tf_default)
    sp.add_argument("--n-seeds", type=int, default=20, dest="n_seeds")
    sp.set_defaults(func=cmd_portfolio_placebo)

    sp = sub.add_parser("portfolio_scorecard",
                        help="Build the portfolio scorecard from saved CSVs.")
    sp.set_defaults(func=cmd_portfolio_scorecard)

    sp = sub.add_parser("audit_portfolio",
                        help="Audit cash filter, benchmark alignment, and "
                             "rebalance logic of the saved portfolio results.")
    sp.set_defaults(func=cmd_audit_portfolio)

    # ----- Phase A: cross-asset regime + regime-aware momentum -------------
    sp = sub.add_parser("crypto_regime_signals",
                        help="Compute cross-asset regime signals over the "
                             "expanded universe.")
    sp.add_argument("--timeframe", default=portfolio_tf_default)
    sp.set_defaults(func=cmd_crypto_regime_signals)

    sp = sub.add_parser("regime_aware_portfolio",
                        help="Single-window regime-aware portfolio backtest "
                             "vs benchmarks.")
    sp.add_argument("--assets", nargs="+", default=portfolio_assets_default)
    sp.add_argument("--timeframe", default=portfolio_tf_default)
    sp.set_defaults(func=cmd_regime_aware_portfolio)

    sp = sub.add_parser("regime_aware_portfolio_walk_forward",
                        help="Walk-forward the regime-aware portfolio.")
    sp.add_argument("--assets", nargs="+", default=portfolio_assets_default)
    sp.add_argument("--timeframe", default=portfolio_tf_default)
    sp.add_argument("--in-sample-days", type=int, default=180,
                    dest="in_sample_days")
    sp.add_argument("--oos-days", type=int, default=90, dest="oos_days")
    sp.add_argument("--step-days", type=int, default=90, dest="step_days")
    sp.set_defaults(func=cmd_regime_aware_portfolio_walk_forward)

    sp = sub.add_parser("regime_aware_portfolio_placebo",
                        help="Compare regime-aware vs regime-aware random "
                             "placebo over N seeds.")
    sp.add_argument("--assets", nargs="+", default=portfolio_assets_default)
    sp.add_argument("--timeframe", default=portfolio_tf_default)
    sp.add_argument("--n-seeds", type=int, default=20, dest="n_seeds")
    sp.set_defaults(func=cmd_regime_aware_portfolio_placebo)

    sp = sub.add_parser("regime_aware_portfolio_scorecard",
                        help="Build the regime-aware portfolio scorecard "
                             "from saved CSVs.")
    sp.set_defaults(func=cmd_regime_aware_portfolio_scorecard)

    sp = sub.add_parser("research_all_portfolio",
                        help="Run portfolio momentum + walk-forward + "
                             "placebo + scorecard end to end.")
    sp.add_argument("--assets", nargs="+", default=portfolio_assets_default)
    sp.add_argument("--timeframe", default=portfolio_tf_default)
    sp.add_argument("--in-sample-days", type=int, default=180,
                    dest="in_sample_days")
    sp.add_argument("--oos-days", type=int, default=90, dest="oos_days")
    sp.add_argument("--step-days", type=int, default=90, dest="step_days")
    sp.add_argument("--n-seeds", type=int, default=20, dest="n_seeds")
    sp.set_defaults(func=cmd_research_all_portfolio)

    sp = sub.add_parser(
        "download_market_structure_data",
        help=("Download free market-structure datasets (DefiLlama, "
              "Blockchain.com). No API keys."),
    )
    sp.add_argument("--lookback-days", type=int, default=1500,
                    dest="lookback_days")
    sp.add_argument("--refresh", action="store_true")
    sp.add_argument("--sleep-seconds", type=float, default=0.25,
                    dest="sleep_seconds")
    sp.set_defaults(func=cmd_download_market_structure_data)

    sp = sub.add_parser(
        "audit_market_structure_data",
        help=("Audit free public market-structure data sources. Writes "
              "results/market_structure_data_audit.csv. No API keys, "
              "no paid endpoints."),
    )
    sp.set_defaults(func=cmd_audit_market_structure_data)

    sp = sub.add_parser("market_structure_signals",
                         help="Compute daily market-structure signals.")
    sp.set_defaults(func=cmd_market_structure_signals)

    sp = sub.add_parser("market_structure_allocator",
                         help="Single-window market-structure allocator backtest.")
    sp.add_argument("--timeframe", default=portfolio_tf_default)
    sp.set_defaults(func=cmd_market_structure_allocator)

    sp = sub.add_parser("market_structure_walk_forward",
                         help="Walk-forward the market-structure allocator.")
    sp.add_argument("--timeframe", default=portfolio_tf_default)
    sp.add_argument("--in-sample-days", type=int, default=180,
                     dest="in_sample_days")
    sp.add_argument("--oos-days", type=int, default=90, dest="oos_days")
    sp.add_argument("--step-days", type=int, default=90, dest="step_days")
    sp.set_defaults(func=cmd_market_structure_walk_forward)

    sp = sub.add_parser("market_structure_placebo",
                         help="Compare allocator vs random state-picker placebo.")
    sp.add_argument("--timeframe", default=portfolio_tf_default)
    sp.add_argument("--n-seeds", type=int, default=20, dest="n_seeds")
    sp.set_defaults(func=cmd_market_structure_placebo)

    sp = sub.add_parser("market_structure_scorecard",
                         help="Build market-structure scorecard from saved CSVs.")
    sp.set_defaults(func=cmd_market_structure_scorecard)

    sp = sub.add_parser("market_structure_vol_target",
                         help="Single-window vol-target market-structure backtest.")
    sp.add_argument("--timeframe", default=portfolio_tf_default)
    sp.set_defaults(func=cmd_market_structure_vol_target)

    sp = sub.add_parser("market_structure_vol_target_walk_forward",
                         help="Walk-forward the vol-target allocator.")
    sp.add_argument("--timeframe", default=portfolio_tf_default)
    sp.add_argument("--in-sample-days", type=int, default=180,
                     dest="in_sample_days")
    sp.add_argument("--oos-days", type=int, default=90, dest="oos_days")
    sp.add_argument("--step-days", type=int, default=90, dest="step_days")
    sp.set_defaults(func=cmd_market_structure_vol_target_walk_forward)

    sp = sub.add_parser("market_structure_vol_target_placebo",
                         help="Vol-target allocator vs 5-state random placebo.")
    sp.add_argument("--timeframe", default=portfolio_tf_default)
    sp.add_argument("--n-seeds", type=int, default=20, dest="n_seeds")
    sp.set_defaults(func=cmd_market_structure_vol_target_placebo)

    sp = sub.add_parser("market_structure_vol_target_scorecard",
                         help="Build vol-target scorecard from saved CSVs.")
    sp.set_defaults(func=cmd_market_structure_vol_target_scorecard)

    sp = sub.add_parser("research_all_market_structure",
                         help="End-to-end market-structure pipeline.")
    sp.add_argument("--timeframe", default=portfolio_tf_default)
    sp.add_argument("--in-sample-days", type=int, default=180,
                     dest="in_sample_days")
    sp.add_argument("--oos-days", type=int, default=90, dest="oos_days")
    sp.add_argument("--step-days", type=int, default=90, dest="step_days")
    sp.add_argument("--n-seeds", type=int, default=20, dest="n_seeds")
    sp.set_defaults(func=cmd_research_all_market_structure)

    sp = sub.add_parser("audit_oos",
                        help="Audit walk-forward window mechanics.")
    sp.set_defaults(func=cmd_audit_oos)

    sp = sub.add_parser("placebo_audit",
                        help="Compare every strategy's OOS stability and "
                             "return against the random placebo.")
    sp.set_defaults(func=cmd_placebo_audit)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
