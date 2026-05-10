"""
Strategy universe selection — research decision module.

A read-only ranking of candidate universes the project COULD pursue
next:

    * Forex (FX major pairs)
    * Crypto spot
    * Crypto derivatives / perps
    * Portfolio rebalancing / risk allocation

For each universe this module records a deterministic per-axis score
(data, execution, edge, complexity, risk), a composite score, a
decision status, and a `recommended_next_action` that points at the
**next research branch** — never at trading, paper-trading, broker
connection, or order placement.

Hard rules (locked):
    * No network calls.
    * No broker SDK imports of any kind.
    * No API key reads.
    * No order placement strings.
    * No paper-trading or live-trading enablement.
    * No file writes from the module — purely a reader / ranker.
    * The recommendation vocabulary is locked: every
      `recommended_next_action` must contain the substring
      "research" and must NOT contain forbidden trade-action language.
      Unit tests enforce both rules at the source level.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Locked vocabulary.
# ---------------------------------------------------------------------------
DECISION_RECOMMENDED = "RECOMMENDED"
DECISION_WATCHLIST = "WATCHLIST"
DECISION_NOT_NOW = "NOT_NOW"
DECISION_REJECTED = "REJECTED"

DECISION_STATUSES: Tuple[str, ...] = (
    DECISION_RECOMMENDED, DECISION_WATCHLIST,
    DECISION_NOT_NOW, DECISION_REJECTED,
)

UNIVERSE_FOREX = "Forex"
UNIVERSE_CRYPTO_SPOT = "Crypto spot"
UNIVERSE_CRYPTO_DERIVATIVES = "Crypto derivatives / perps"
UNIVERSE_PORTFOLIO_REBALANCING = "Portfolio rebalancing / risk allocation"

UNIVERSES: Tuple[str, ...] = (
    UNIVERSE_FOREX, UNIVERSE_CRYPTO_SPOT,
    UNIVERSE_CRYPTO_DERIVATIVES, UNIVERSE_PORTFOLIO_REBALANCING,
)


# ---------------------------------------------------------------------------
# Per-universe assessment dataclass.
#
# Scores are integer 0-10 and were assigned BEFORE running the rank
# function — they are not tuned to make any universe rank highest.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UniverseAssessment:
    universe: str
    data_score: int            # 0-10: depth + freshness + free availability
    execution_score: int       # 0-10: fits this engine's no-broker policy
    edge_score: int            # 0-10: realistic alpha potential for retail
    complexity_score: int      # 0-10: 10 = simple to ship, 0 = huge build
    risk_score: int            # 0-10: 10 = lowest blow-up risk, 0 = highest

    decision_status: str       # one of DECISION_STATUSES
    recommended_next_action: str  # MUST mention "research"; no trade verbs
    notes: str

    pass_criteria: str
    fail_criteria: str

    # For each per-axis score, a one-line "why this score" caption used
    # in the report and dashboard.
    data_caption: str
    execution_caption: str
    edge_caption: str
    complexity_caption: str
    risk_caption: str


# ---------------------------------------------------------------------------
# Locked per-universe assessments.
# ---------------------------------------------------------------------------
_ASSESSMENTS: Tuple[UniverseAssessment, ...] = (
    UniverseAssessment(
        universe=UNIVERSE_FOREX,
        data_score=6,
        execution_score=3,
        edge_score=3,
        complexity_score=4,
        risk_score=4,
        decision_status=DECISION_NOT_NOW,
        recommended_next_action=(
            "Defer. The next research branch on this universe would be "
            "research/forex-data-audit-v1 — but only if a broker "
            "integration policy change is approved separately. Out of "
            "scope under the current no-broker policy."
        ),
        notes=(
            "OANDA / Dukascopy / HistData expose multi-year free OHLC "
            "for major FX pairs, but a real backtest needs a spread + "
            "swap model and a demo broker for paper paths. Adding a "
            "broker is forbidden on this project. Retail FX edge is "
            "extremely competitive."
        ),
        pass_criteria=(
            "Free FX historical data loads at >= 5y with verified "
            "spreads; a strategy beats EUR/USD buy-and-hold in "
            ">50% of OOS windows under fee + spread modelling."
        ),
        fail_criteria=(
            "Free data shallow, spread model unverifiable, or "
            "edge fails to clear the placebo on any major pair."
        ),
        data_caption=(
            "free multi-year OHLC available, but tick-level depth "
            "and verified spread modelling typically need paid data"
        ),
        execution_caption=(
            "would require a broker / demo integration; current "
            "engine policy forbids any broker"
        ),
        edge_caption=(
            "retail FX is one of the most-arbitraged retail markets; "
            "edge is structurally low without institutional liquidity"
        ),
        complexity_caption=(
            "moderate: new asset class, new pricing model, new "
            "spread / swap accounting, new universe loader"
        ),
        risk_caption=(
            "leverage cultures + tight spreads + 24x5 sessions = "
            "elevated blow-up risk without strict risk caps"
        ),
    ),
    UniverseAssessment(
        universe=UNIVERSE_CRYPTO_SPOT,
        data_score=9,
        execution_score=7,
        edge_score=2,
        complexity_score=8,
        risk_score=6,
        decision_status=DECISION_REJECTED,
        recommended_next_action=(
            "Closed. Do not open another long-only research branch on "
            "this universe; eight prior branches failed under the "
            "locked scorecard. The free public stack adds no new "
            "untested signal class. Future research on this universe "
            "is only justified if a paid positioning data plan is "
            "purchased."
        ),
        notes=(
            "Binance public OHLCV provides multi-year deep history; "
            "the existing portfolio_backtester already runs against "
            "it. But the failed-strategy track record is dispositive: "
            "drawdown-targeted, funding+basis carry, BTC/ETH "
            "relative-value, market-structure, sentiment overlay, "
            "funding-only — every long-only allocator on this "
            "universe failed against BTC buy-and-hold in OOS windows."
        ),
        pass_criteria=(
            "A new long-only allocator beats BTC in >50% of 14 OOS "
            "windows under the locked scorecard, with stability >= 60%."
        ),
        fail_criteria=(
            "Already FAILED — see the archived research branches on "
            "the registry."
        ),
        data_caption=(
            "Binance public OHLCV, funding history, and futures basis "
            "klines all PASS at 1500+ days"
        ),
        execution_caption=(
            "fits the existing portfolio_backtester; no new "
            "infrastructure required"
        ),
        edge_caption=(
            "EXHAUSTED. Eight branches failed against BTC b&h on "
            "this exact universe; the free-data verdict is closed"
        ),
        complexity_caption=(
            "lowest build cost — every signal stack already exists "
            "in archived branches"
        ),
        risk_caption=(
            "long-only, no leverage; risk is not blow-up but "
            "opportunity cost vs BTC b&h"
        ),
    ),
    UniverseAssessment(
        universe=UNIVERSE_CRYPTO_DERIVATIVES,
        data_score=4,
        execution_score=3,
        edge_score=5,
        complexity_score=4,
        risk_score=4,
        decision_status=DECISION_NOT_NOW,
        recommended_next_action=(
            "Defer until a paid positioning data trial is purchased. "
            "The next research branch is "
            "research/strategy-9-coinglass-keyed-data-audit (already "
            "shipped, currently INCONCLUSIVE pending COINGLASS_API_KEY) "
            "OR a vendor-keyed re-audit on CryptoQuant / Velo Data."
        ),
        notes=(
            "Public OI / liquidations / long-short ratios are FAIL "
            "on every free venue (capped 30-500 days — "
            "documented in the strategy-5 and strategy-9 free-data "
            "audits). Paid plans (CoinGlass / CryptoQuant / Velo) "
            "from $29-$99/mo could unlock the missing depth, but "
            "execution would also need a perp / margin backtester "
            "this engine does not have. Two distinct prerequisites."
        ),
        pass_criteria=(
            ">= 1460 days of OI + liquidations + long-short ratios "
            "on a paid plan, AND a perp backtester with funding + "
            "basis cost modelling, AND a strategy passes the locked "
            "scorecard against BTC b&h."
        ),
        fail_criteria=(
            "Paid data still capped or one of the prerequisites "
            "remains unbuilt."
        ),
        data_caption=(
            "free public stack FAILS for OI / liquidations / "
            "long-short; paid is the only verified path"
        ),
        execution_caption=(
            "requires a perp + margin backtester (not yet built) plus "
            "a future broker for any paper / live phase"
        ),
        edge_caption=(
            "best signal-class potential of the four — crowded "
            "longs + liquidation cascades are real patterns — but "
            "data-gated"
        ),
        complexity_caption=(
            "two prerequisite branches before a strategy: keyed "
            "data audit, then perp backtester; not cheap"
        ),
        risk_caption=(
            "leverage, funding, liquidation mechanics — the "
            "highest blow-up risk of the four if executed badly"
        ),
    ),
    UniverseAssessment(
        universe=UNIVERSE_PORTFOLIO_REBALANCING,
        data_score=9,
        execution_score=8,
        edge_score=4,
        complexity_score=9,
        risk_score=9,
        decision_status=DECISION_RECOMMENDED,
        recommended_next_action=(
            "Open research/portfolio-rebalancing-strategy-v1 next. "
            "The strategy is fixed-weight monthly rebalancing — e.g. "
            "60/30/10 BTC/ETH/USDC — benchmarked against BTC b&h "
            "and an equal-weight basket. PASS criteria are NOT "
            "'beat BTC' — they are 'comparable risk-adjusted return "
            "with materially lower drawdown'. This is a research "
            "branch, not a trading branch."
        ),
        notes=(
            "Fits the existing engine end-to-end: portfolio_risk "
            "module supplies position math, portfolio_backtester "
            "supplies the rebalance simulator, the registry supplies "
            "verdict tracking. No new data sources, no new broker, "
            "no leverage, no shorting. The locked PASS criteria are "
            "different from the failed allocator branches: rebalance "
            "research is judged on Sharpe / drawdown-adjusted "
            "metrics, not on raw return vs BTC."
        ),
        pass_criteria=(
            "Sharpe ratio within 0.10 of BTC b&h AND max drawdown "
            "at least 15 pp tighter than BTC AND beats placebo on "
            "both axes AND >= 24 rebalances over the OOS window. "
            "If beating BTC outright on return is required, this "
            "becomes a FAIL universe like the others."
        ),
        fail_criteria=(
            "Cannot tighten drawdown materially OR fails the placebo "
            "OR introduces concentration that violates the locked "
            "portfolio_risk thresholds (largest position > 80%)."
        ),
        data_caption=(
            "uses the existing local 1d cache — no new data; "
            "every input has multi-year coverage"
        ),
        execution_caption=(
            "fits the existing portfolio_backtester directly; no "
            "broker required for research"
        ),
        edge_caption=(
            "not a high-return alpha class, but a real, measurable "
            "drawdown / Sharpe improvement vs unrebalanced BTC b&h"
        ),
        complexity_caption=(
            "lowest complexity of the four; reuses every existing "
            "module; smallest new test surface"
        ),
        risk_caption=(
            "no leverage, no shorting, fixed weights — the lowest "
            "overfitting and blow-up risk of the four universes"
        ),
    ),
)


# ---------------------------------------------------------------------------
# Composite scoring + ranking.
# ---------------------------------------------------------------------------
SCORE_AXES: Tuple[str, ...] = (
    "data_score", "execution_score", "edge_score",
    "complexity_score", "risk_score",
)


def composite_score(a: UniverseAssessment) -> float:
    """Equal-weighted average of the five per-axis scores. Locked at
    equal weighting; not tuned per universe."""
    return (
        a.data_score + a.execution_score + a.edge_score
        + a.complexity_score + a.risk_score
    ) / 5.0


def assessments() -> Tuple[UniverseAssessment, ...]:
    """Public read-only accessor for the locked ledger."""
    return _ASSESSMENTS


# Output table column order — locked.
TABLE_COLUMNS: List[str] = [
    "universe", "score", "rank",
    "data_score", "execution_score", "edge_score",
    "complexity_score", "risk_score",
    "recommended_next_action", "decision_status", "notes",
]


def rank_universes() -> pd.DataFrame:
    """Build the ranked DataFrame. Sort key: composite score desc, then
    decision_status priority (RECOMMENDED > WATCHLIST > NOT_NOW >
    REJECTED) so a tied score still surfaces the actionable choice
    first. Deterministic — same inputs always yield the same order."""
    rows: List[Dict[str, Any]] = []
    status_priority = {
        DECISION_RECOMMENDED: 0,
        DECISION_WATCHLIST: 1,
        DECISION_NOT_NOW: 2,
        DECISION_REJECTED: 3,
    }
    scored = [(a, composite_score(a),
                 status_priority.get(a.decision_status, 99))
                for a in _ASSESSMENTS]
    scored.sort(key=lambda x: (-x[1], x[2]))
    for rank, (a, score, _) in enumerate(scored, start=1):
        rows.append({
            "universe": a.universe,
            "score": float(round(score, 3)),
            "rank": int(rank),
            "data_score": int(a.data_score),
            "execution_score": int(a.execution_score),
            "edge_score": int(a.edge_score),
            "complexity_score": int(a.complexity_score),
            "risk_score": int(a.risk_score),
            "recommended_next_action": a.recommended_next_action,
            "decision_status": a.decision_status,
            "notes": a.notes,
        })
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def top_recommendation() -> Dict[str, Any]:
    """Return the recommended universe (first row whose decision_status
    is RECOMMENDED). If no universe is RECOMMENDED, return the
    highest-scored row plus a flag that the project should not start
    new strategy work."""
    table = rank_universes()
    recommended = table[table["decision_status"] == DECISION_RECOMMENDED]
    if not recommended.empty:
        row = recommended.iloc[0]
        return {
            "universe": str(row["universe"]),
            "score": float(row["score"]),
            "rank": int(row["rank"]),
            "decision_status": DECISION_RECOMMENDED,
            "recommended_next_action": str(row["recommended_next_action"]),
            "no_strategy_today": False,
        }
    row = table.iloc[0]
    return {
        "universe": str(row["universe"]),
        "score": float(row["score"]),
        "rank": int(row["rank"]),
        "decision_status": str(row["decision_status"]),
        "recommended_next_action": (
            "No universe currently RECOMMENDED — do not open another "
            "strategy branch yet. Continue using the engine as a "
            "research and risk dashboard."
        ),
        "no_strategy_today": True,
    }


# ---------------------------------------------------------------------------
# Recommendation-language safety.
# ---------------------------------------------------------------------------
# Tokens forbidden anywhere in `recommended_next_action`. Tests enforce.
FORBIDDEN_RECOMMENDATION_TOKENS: Tuple[str, ...] = (
    "buy ", "sell ", "place order", "submit order",
    "connect broker", "enter trade", "open position", "close position",
    "go long", "go short", "paper trade now", "start trading",
    "place a trade", "execute trade", "go live",
)


def recommendation_is_clean(action: str) -> bool:
    """Predicate the test suite uses; also guards the dashboard."""
    if not action:
        return False
    low = action.lower()
    if "research" not in low:
        return False
    for bad in FORBIDDEN_RECOMMENDATION_TOKENS:
        if bad in low:
            return False
    return True
