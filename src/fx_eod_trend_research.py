"""
FX end-of-day trend research orchestrator (read-only, no broker).

Glues:

    `data/fx/fx_daily_v1.parquet` (built by `fx_research_dataset`)
    → quality guard via `fx_data_quality.run_fx_data_quality_checks`
    → `FXEODTrendStrategy` (locked rule, no parameter search)
    → full-window backtest, walk-forward, 20-seed placebo, scorecard.

Hard rules (locked):
    * Read-only. The only filesystem reads are the validated parquet
      and (optionally) its CSV companion. The orchestrator never
      touches the network.
    * No broker imports, no API keys, no order placement, no
      execution, no paper trading, no live trading.
    * The strategy rule is locked before any output is produced
      (200-day SMA, long-or-cash, EUR/USD, daily). No optimiser.
    * The placebo design is locked: 20 deterministic seeds, each
      generates a uniform-random binary position with mean exposure
      matched to the strategy's realised exposure on the same series
      (no parameter search, no regeneration after seeing results).
    * If the dataset's quality verdict is FAIL or INCONCLUSIVE the
      orchestrator refuses to run. WARNING is acceptable because
      v1's only WARNING source is real historical FX shocks.
    * Generated CSVs at `results/fx_eod_trend_*.csv` are gitignored
      under the existing `results/*.csv` rule. The writer refuses to
      write outside `results/`.

Outputs:

    results/fx_eod_trend_backtest.csv
    results/fx_eod_trend_walk_forward.csv
    results/fx_eod_trend_placebo.csv
    results/fx_eod_trend_scorecard.csv
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config, fx_data_quality, fx_research_dataset, utils
from .strategies import fx_eod_trend


logger = utils.get_logger("cte.fx_eod_trend_research")


# ---------------------------------------------------------------------------
# Locked output paths
# ---------------------------------------------------------------------------
BACKTEST_PATH: Path = config.RESULTS_DIR / "fx_eod_trend_backtest.csv"
WALK_FORWARD_PATH: Path = config.RESULTS_DIR / "fx_eod_trend_walk_forward.csv"
PLACEBO_PATH: Path = config.RESULTS_DIR / "fx_eod_trend_placebo.csv"
SCORECARD_PATH: Path = config.RESULTS_DIR / "fx_eod_trend_scorecard.csv"

# Locked design parameters (NEVER tuned by results)
DEFAULT_N_PLACEBO_SEEDS = 20
DEFAULT_N_WALK_FORWARD_WINDOWS = 5
MIN_TRADE_COUNT_FOR_PASS = 20
DRAWDOWN_TIGHTER_PP = 0.05  # 5 percentage points

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"
ACCEPTABLE_QUALITY_VERDICTS: Tuple[str, ...] = ("PASS", "WARNING")


# ---------------------------------------------------------------------------
# Result schemas
# ---------------------------------------------------------------------------
WALK_FORWARD_COLUMNS: List[str] = [
    "window_id", "start_date", "end_date", "n_rows",
    "strategy_total_return", "benchmark_total_return",
    "strategy_sharpe", "benchmark_sharpe",
    "strategy_max_drawdown", "benchmark_max_drawdown",
    "drawdown_improvement_pp",
    "trade_count", "exposure_pct",
    "strategy_beats_benchmark_return",
    "strategy_beats_benchmark_sharpe",
]


PLACEBO_COLUMNS: List[str] = [
    "seed", "exposure_pct", "trade_count",
    "total_return", "sharpe", "max_drawdown",
    "design",
]


SCORECARD_COLUMNS: List[str] = [
    "strategy_name", "asset", "verdict",
    "total_return", "benchmark_total_return", "cash_total_return",
    "sharpe", "benchmark_sharpe",
    "max_drawdown", "benchmark_max_drawdown",
    "drawdown_improvement_pp",
    "placebo_return_percentile", "placebo_drawdown_percentile",
    "placebo_median_return", "placebo_median_drawdown",
    "trade_count", "exposure_percent",
    "pass_positive_return", "pass_sharpe_beats_benchmark",
    "pass_drawdown_tighter", "pass_beats_placebo_return",
    "pass_beats_placebo_drawdown", "pass_min_trade_count",
    "no_leverage", "no_shorts", "no_lookahead",
    "checks_passed", "checks_total",
    "data_quality_verdict",
    "n_placebo_seeds", "n_walk_forward_windows",
    "notes",
]


# ---------------------------------------------------------------------------
# Quality guard
# ---------------------------------------------------------------------------
class FxEodTrendBlockedError(RuntimeError):
    """Raised when the dataset's quality verdict blocks research, or
    when the dataset is unreadable / missing."""


def load_and_guard(path: Path = fx_research_dataset.DATASET_PARQUET
                      ) -> Tuple[pd.DataFrame, fx_data_quality.QualityReport]:
    """Run the FX data-quality checks first; if the verdict is FAIL
    or INCONCLUSIVE, raise. Otherwise load and return the dataset
    plus the report. WARNING is acceptable."""
    utils.assert_paper_only()
    report = fx_data_quality.run_fx_data_quality_checks(path)
    if report.verdict not in ACCEPTABLE_QUALITY_VERDICTS:
        raise FxEodTrendBlockedError(
            f"FX data quality verdict is {report.verdict}; "
            "research blocked. Acceptable verdicts: "
            f"{ACCEPTABLE_QUALITY_VERDICTS}. "
            "Run `python main.py check_fx_data_quality` for details."
        )
    df = fx_data_quality.load_fx_dataset(path)
    return df, report


# ---------------------------------------------------------------------------
# Backtest, walk-forward, placebo
# ---------------------------------------------------------------------------
def run_full_window_backtest(
    cfg: fx_eod_trend.FXEODTrendConfig,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Single full-window backtest. Asserts long-cash + no-lookahead
    invariants before returning."""
    utils.assert_paper_only()
    bt = fx_eod_trend.FXEODTrendStrategy.compute_backtest(df, cfg)
    if bt.empty:
        return bt
    fx_eod_trend.assert_long_cash_only(bt["position"])
    prep = fx_eod_trend.FXEODTrendStrategy.prepare_close_series(df, cfg)
    sig = fx_eod_trend.FXEODTrendStrategy.generate_signals(
        prep["close"], cfg
    )
    fx_eod_trend.assert_no_lookahead(prep["close"], sig["sma"], cfg)
    return bt


def _window_metrics(window_df: pd.DataFrame,
                       window_id: int) -> Dict[str, Any]:
    """Compute per-window metrics from a slice of the full backtest."""
    if window_df.empty:
        return {c: None for c in WALK_FORWARD_COLUMNS}
    s_total = fx_eod_trend.total_return_from_returns(
        window_df["strategy_return"]
    )
    b_total = fx_eod_trend.total_return_from_returns(
        window_df["benchmark_buyhold_return"]
    )
    # Equity rebased to 1.0 at window start so drawdown is window-local.
    s_eq = (1.0 + window_df["strategy_return"].fillna(0.0)).cumprod()
    b_eq = (1.0 + window_df["benchmark_buyhold_return"].fillna(0.0)).cumprod()
    s_dd = fx_eod_trend.max_drawdown_from_equity(s_eq)
    b_dd = fx_eod_trend.max_drawdown_from_equity(b_eq)
    s_sharpe = fx_eod_trend.annualised_sharpe(window_df["strategy_return"])
    b_sharpe = fx_eod_trend.annualised_sharpe(
        window_df["benchmark_buyhold_return"]
    )
    return {
        "window_id": int(window_id),
        "start_date": pd.Timestamp(
            window_df["date"].min()
        ).date().isoformat(),
        "end_date": pd.Timestamp(
            window_df["date"].max()
        ).date().isoformat(),
        "n_rows": int(len(window_df)),
        "strategy_total_return": float(s_total),
        "benchmark_total_return": float(b_total),
        "strategy_sharpe": float(s_sharpe),
        "benchmark_sharpe": float(b_sharpe),
        "strategy_max_drawdown": float(s_dd),
        "benchmark_max_drawdown": float(b_dd),
        "drawdown_improvement_pp": float(s_dd - b_dd),
        "trade_count": int(
            fx_eod_trend.trade_count_from_position(window_df["position"])
        ),
        "exposure_pct": float(
            fx_eod_trend.exposure_pct_from_position(window_df["position"])
        ),
        "strategy_beats_benchmark_return": bool(s_total > b_total),
        "strategy_beats_benchmark_sharpe": bool(s_sharpe > b_sharpe),
    }


def run_walk_forward(
    cfg: fx_eod_trend.FXEODTrendConfig,
    df: pd.DataFrame,
    n_windows: int = DEFAULT_N_WALK_FORWARD_WINDOWS,
) -> pd.DataFrame:
    """Split the full-window backtest into `n_windows` contiguous,
    equal-length chronological windows and report per-window metrics.

    There is **no in-sample optimisation**: the rule is locked, so
    walk-forward here is an OOS *stability* report, not a parameter
    search."""
    utils.assert_paper_only()
    bt = run_full_window_backtest(cfg, df)
    if bt.empty:
        return pd.DataFrame(columns=WALK_FORWARD_COLUMNS)
    if n_windows <= 0:
        return pd.DataFrame(columns=WALK_FORWARD_COLUMNS)
    n = len(bt)
    if n < n_windows:
        return pd.DataFrame(columns=WALK_FORWARD_COLUMNS)
    boundaries = np.linspace(0, n, n_windows + 1, dtype=int)
    rows: List[Dict[str, Any]] = []
    for i in range(n_windows):
        sub = bt.iloc[boundaries[i]:boundaries[i + 1]]
        rows.append(_window_metrics(sub, window_id=i))
    return pd.DataFrame(rows, columns=WALK_FORWARD_COLUMNS)


def _matched_exposure_position(rng: np.random.Generator,
                                  n: int,
                                  exposure_frac: float) -> np.ndarray:
    """Generate a length-n binary position with mean exactly equal to
    `round(n * exposure_frac) / n`. Achieves matched exposure by
    permuting a fixed-count vector, so the placebo's gross market
    time matches the strategy's exactly."""
    if n <= 0:
        return np.zeros(0, dtype=float)
    k = int(round(n * float(exposure_frac)))
    k = max(0, min(n, k))
    arr = np.zeros(n, dtype=float)
    arr[:k] = 1.0
    rng.shuffle(arr)
    return arr


def run_placebo(
    cfg: fx_eod_trend.FXEODTrendConfig,
    df: pd.DataFrame,
    n_seeds: int = DEFAULT_N_PLACEBO_SEEDS,
) -> pd.DataFrame:
    """20 deterministic seeded placebos with matched exposure.

    Design (locked, declared before results):

        For seed s in range(n_seeds):
            rng = numpy.random.default_rng(s)
            generate a random binary position with mean exposure
            equal to the strategy's realised exposure on the same
            backtest period; compute the same metrics.

    Matched-exposure permutation removes the trivial "more-time-in-
    market beats less-time-in-market" effect."""
    utils.assert_paper_only()
    if n_seeds < DEFAULT_N_PLACEBO_SEEDS:
        raise ValueError(
            f"n_seeds={n_seeds} below the locked minimum "
            f"{DEFAULT_N_PLACEBO_SEEDS}; the placebo design is locked."
        )
    bt = run_full_window_backtest(cfg, df)
    if bt.empty:
        return pd.DataFrame(columns=PLACEBO_COLUMNS)
    raw_returns = bt["benchmark_buyhold_return"].fillna(0.0).values
    exposure_frac = float(bt["position"].mean())
    n = len(raw_returns)
    rows: List[Dict[str, Any]] = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        pos = _matched_exposure_position(rng, n, exposure_frac)
        ret = pos * raw_returns
        equity = (1.0 + ret).cumprod()
        rows.append({
            "seed": int(seed),
            "exposure_pct": float(100.0 * pos.mean()),
            "trade_count": int(
                fx_eod_trend.trade_count_from_position(pd.Series(pos))
            ),
            "total_return": float(
                fx_eod_trend.total_return_from_returns(pd.Series(ret))
            ),
            "sharpe": float(
                fx_eod_trend.annualised_sharpe(pd.Series(ret))
            ),
            "max_drawdown": float(
                fx_eod_trend.max_drawdown_from_equity(pd.Series(equity))
            ),
            "design": "matched_exposure_random_binary_seeded",
        })
    return pd.DataFrame(rows, columns=PLACEBO_COLUMNS)


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
def _percentile_of(value: float, sample: Sequence[float]) -> float:
    """Empirical percentile of `value` against `sample` ∈ [0, 100]."""
    if sample is None or len(sample) == 0:
        return float("nan")
    arr = np.asarray(sample, dtype=float)
    return float(100.0 * (arr < value).mean())


def compute_scorecard(
    cfg: fx_eod_trend.FXEODTrendConfig,
    backtest: pd.DataFrame,
    placebo: pd.DataFrame,
    walk_forward: Optional[pd.DataFrame] = None,
    quality_verdict: str = "WARNING",
    notes: str = "",
) -> pd.DataFrame:
    """Compute the locked scorecard. Verdict logic is the spec from
    the branch prompt — never tuned after results."""
    utils.assert_paper_only()
    if backtest is None or backtest.empty:
        row = {c: None for c in SCORECARD_COLUMNS}
        row.update({
            "strategy_name": cfg.strategy_name,
            "asset": cfg.asset,
            "verdict": VERDICT_INCONCLUSIVE,
            "data_quality_verdict": quality_verdict,
            "notes": (notes
                       or "backtest is empty (insufficient data)"),
            "n_placebo_seeds": int(len(placebo)) if placebo is not None
                                  else 0,
            "n_walk_forward_windows": int(len(walk_forward))
                                          if walk_forward is not None
                                          else 0,
        })
        return pd.DataFrame([row], columns=SCORECARD_COLUMNS)

    s_returns = backtest["strategy_return"]
    b_returns = backtest["benchmark_buyhold_return"]
    s_equity = backtest["strategy_equity"]
    b_equity = backtest["benchmark_buyhold_equity"]
    cash_total = 0.0  # cash is held at constant `initial_cash`
    s_total = fx_eod_trend.total_return_from_returns(s_returns)
    b_total = fx_eod_trend.total_return_from_returns(b_returns)
    s_sharpe = fx_eod_trend.annualised_sharpe(s_returns)
    b_sharpe = fx_eod_trend.annualised_sharpe(b_returns)
    s_dd = fx_eod_trend.max_drawdown_from_equity(s_equity)
    b_dd = fx_eod_trend.max_drawdown_from_equity(b_equity)
    dd_improvement_pp = float(s_dd - b_dd)  # positive = strat tighter
    n_trades = fx_eod_trend.trade_count_from_position(backtest["position"])
    exposure_pct = fx_eod_trend.exposure_pct_from_position(
        backtest["position"]
    )

    if placebo is not None and not placebo.empty:
        p_returns = placebo["total_return"].astype(float).values
        p_dds = placebo["max_drawdown"].astype(float).values
        p_ret_pct = _percentile_of(s_total, p_returns)
        p_dd_pct = _percentile_of(s_dd, p_dds)  # higher = tighter
        p_ret_median = float(np.median(p_returns))
        p_dd_median = float(np.median(p_dds))
    else:
        p_ret_pct = float("nan")
        p_dd_pct = float("nan")
        p_ret_median = float("nan")
        p_dd_median = float("nan")

    pass_positive_return = bool(s_total > 0)
    pass_sharpe_beats_benchmark = bool(s_sharpe > b_sharpe)
    pass_drawdown_tighter = bool(
        dd_improvement_pp >= DRAWDOWN_TIGHTER_PP
    )
    pass_beats_placebo_return = bool(
        not np.isnan(p_ret_median) and s_total > p_ret_median
    )
    pass_beats_placebo_drawdown = bool(
        not np.isnan(p_dd_median) and s_dd > p_dd_median
    )
    pass_min_trade_count = bool(n_trades >= MIN_TRADE_COUNT_FOR_PASS)
    no_leverage = bool(backtest["position"].dropna().max() <= 1.0)
    no_shorts = bool(backtest["position"].dropna().min() >= 0.0)
    no_lookahead = True  # by construction; tests assert separately

    pass_flags = [
        pass_positive_return, pass_sharpe_beats_benchmark,
        pass_drawdown_tighter, pass_beats_placebo_return,
        pass_beats_placebo_drawdown, pass_min_trade_count,
        no_leverage, no_shorts, no_lookahead,
    ]
    checks_total = len(pass_flags)
    checks_passed = int(sum(1 for f in pass_flags if f))
    verdict = VERDICT_PASS if checks_passed == checks_total else VERDICT_FAIL
    failed_reasons: List[str] = []
    if not pass_positive_return:
        failed_reasons.append(f"total_return={s_total:+.4%} ≤ 0")
    if not pass_sharpe_beats_benchmark:
        failed_reasons.append(
            f"sharpe={s_sharpe:.3f} ≤ benchmark={b_sharpe:.3f}"
        )
    if not pass_drawdown_tighter:
        failed_reasons.append(
            f"drawdown_improvement_pp={dd_improvement_pp:+.4f} "
            f"< {DRAWDOWN_TIGHTER_PP:+.2f}"
        )
    if not pass_beats_placebo_return:
        failed_reasons.append(
            f"total_return={s_total:+.4%} ≤ placebo_median="
            f"{p_ret_median:+.4%}"
        )
    if not pass_beats_placebo_drawdown:
        failed_reasons.append(
            f"max_drawdown={s_dd:+.4%} ≤ placebo_median="
            f"{p_dd_median:+.4%}"
        )
    if not pass_min_trade_count:
        failed_reasons.append(
            f"trade_count={n_trades} < min={MIN_TRADE_COUNT_FOR_PASS}"
        )
    note_text = notes
    if failed_reasons:
        note_text = ((notes + "; ") if notes else "") \
                       + "FAIL reasons: " + " | ".join(failed_reasons)

    row = {
        "strategy_name": cfg.strategy_name,
        "asset": cfg.asset,
        "verdict": verdict,
        "total_return": float(s_total),
        "benchmark_total_return": float(b_total),
        "cash_total_return": float(cash_total),
        "sharpe": float(s_sharpe),
        "benchmark_sharpe": float(b_sharpe),
        "max_drawdown": float(s_dd),
        "benchmark_max_drawdown": float(b_dd),
        "drawdown_improvement_pp": float(dd_improvement_pp),
        "placebo_return_percentile": float(p_ret_pct),
        "placebo_drawdown_percentile": float(p_dd_pct),
        "placebo_median_return": float(p_ret_median),
        "placebo_median_drawdown": float(p_dd_median),
        "trade_count": int(n_trades),
        "exposure_percent": float(exposure_pct),
        "pass_positive_return": pass_positive_return,
        "pass_sharpe_beats_benchmark": pass_sharpe_beats_benchmark,
        "pass_drawdown_tighter": pass_drawdown_tighter,
        "pass_beats_placebo_return": pass_beats_placebo_return,
        "pass_beats_placebo_drawdown": pass_beats_placebo_drawdown,
        "pass_min_trade_count": pass_min_trade_count,
        "no_leverage": no_leverage,
        "no_shorts": no_shorts,
        "no_lookahead": no_lookahead,
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "data_quality_verdict": quality_verdict,
        "n_placebo_seeds": (int(len(placebo))
                              if placebo is not None else 0),
        "n_walk_forward_windows": (int(len(walk_forward))
                                       if walk_forward is not None
                                       else 0),
        "notes": note_text,
    }
    return pd.DataFrame([row], columns=SCORECARD_COLUMNS)


# ---------------------------------------------------------------------------
# Writer (restricted to results/)
# ---------------------------------------------------------------------------
class FxEodTrendWritePathError(ValueError):
    """Raised if the writer is asked to write outside results/."""


def _assert_inside_results_dir(path: Path) -> None:
    resolved = path.resolve()
    results_root = config.RESULTS_DIR.resolve()
    try:
        resolved.relative_to(results_root)
    except ValueError as exc:
        raise FxEodTrendWritePathError(
            f"refusing to write outside results/: {resolved}"
        ) from exc


def write_csv(df: pd.DataFrame, path: Path) -> str:
    _assert_inside_results_dir(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
@dataclass
class ResearchBundle:
    """Everything `run_full_research` produces, in one struct."""
    cfg: fx_eod_trend.FXEODTrendConfig
    backtest: pd.DataFrame
    walk_forward: pd.DataFrame
    placebo: pd.DataFrame
    scorecard: pd.DataFrame
    quality_verdict: str
    written: Dict[str, str] = field(default_factory=dict)


def run_full_research(
    cfg: Optional[fx_eod_trend.FXEODTrendConfig] = None,
    n_walk_forward_windows: int = DEFAULT_N_WALK_FORWARD_WINDOWS,
    n_placebo_seeds: int = DEFAULT_N_PLACEBO_SEEDS,
    save: bool = True,
) -> ResearchBundle:
    """End-to-end: guard → backtest → walk-forward → placebo →
    scorecard → optional save."""
    utils.assert_paper_only()
    cfg = cfg or fx_eod_trend.FXEODTrendConfig()
    df, q_report = load_and_guard()
    bt = run_full_window_backtest(cfg, df)
    wf = run_walk_forward(cfg, df, n_windows=n_walk_forward_windows)
    pb = run_placebo(cfg, df, n_seeds=n_placebo_seeds)
    sc = compute_scorecard(cfg, bt, pb, walk_forward=wf,
                              quality_verdict=q_report.verdict)
    written: Dict[str, str] = {}
    if save:
        written["backtest"] = write_csv(bt, BACKTEST_PATH)
        written["walk_forward"] = write_csv(wf, WALK_FORWARD_PATH)
        written["placebo"] = write_csv(pb, PLACEBO_PATH)
        written["scorecard"] = write_csv(sc, SCORECARD_PATH)
    return ResearchBundle(
        cfg=cfg, backtest=bt, walk_forward=wf, placebo=pb,
        scorecard=sc, quality_verdict=q_report.verdict, written=written,
    )
