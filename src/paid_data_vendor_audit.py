"""
Paid data vendor decision audit.

This module is a structured, manually-maintained ledger of crypto
positioning data vendors. It does NOT call private APIs, hold API
keys, scrape pricing pages, or download paid data. Every row in the
ledger is a desk-research entry that names public sources we can
verify by visiting the vendor's published documentation/marketing
pages.

Output:
    `results/paid_data_vendor_audit.csv` — one row per vendor.

Decision rule (locked here, not tuned):

    PASS_CANDIDATE  Vendor clearly offers >= 2 of the priority data
                    classes (OI, liquidations, long/short ratios,
                    funding, basis, exchange flows), AND has API or
                    bulk-export access, AND multi-year historical
                    depth is claimed/documented, AND there is a
                    non-enterprise plan with verifiable pricing.
    WATCHLIST       Useful data exists but pricing is unclear or the
                    vendor only covers a subset.
    SALES_REQUIRED  Data probably exists; pricing requires sales
                    contact.
    DOCS_GATED      API docs require login/demo to verify field set.
    INCONCLUSIVE    Cannot reach a verdict without further research.
    FAIL            No relevant historical positioning data, or
                    research-only snapshots, or clearly unsuitable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from . import config, utils

logger = utils.get_logger("cte.paid_data_vendor_audit")


# ---------------------------------------------------------------------------
# Output schema (locked).
# ---------------------------------------------------------------------------
AUDIT_COLUMNS: List[str] = [
    "vendor",
    "category",
    "website",
    "docs_url",
    "pricing_url",
    "api_docs_public",
    "pricing_public",
    "pricing_model",
    "lowest_visible_plan_usd",
    "free_trial_available",
    "requires_sales_call",
    "historical_open_interest",
    "historical_liquidations",
    "historical_long_short_ratios",
    "historical_funding",
    "historical_basis",
    "exchange_flows_reserves",
    "asset_coverage",
    "granularity_claimed",
    "history_depth_claimed",
    "api_access",
    "download_access",
    "terms_notes",
    "research_usability",
    "decision_status",
    "recommended_next_action",
    "confidence",
    "notes",
]

# Locked decision-status vocabulary.
DECISION_STATUSES: tuple = (
    "PASS_CANDIDATE", "WATCHLIST", "SALES_REQUIRED",
    "DOCS_GATED", "INCONCLUSIVE", "FAIL",
)
CONFIDENCE_LEVELS: tuple = ("high", "medium", "low")
TERNARY: tuple = ("yes", "no", "unknown", "limited", "snapshot_only")


# ---------------------------------------------------------------------------
# Decision classifier — pure, schema-driven.
# ---------------------------------------------------------------------------
PRIORITY_FIELDS: tuple = (
    "historical_open_interest",
    "historical_liquidations",
    "historical_long_short_ratios",
    "historical_funding",
    "historical_basis",
    "exchange_flows_reserves",
)


def classify_decision(row: Dict[str, Any]) -> str:
    """Return one of `DECISION_STATUSES`. Pure function — all inputs
    come from the row itself."""
    yes_count = sum(1 for f in PRIORITY_FIELDS
                     if str(row.get(f, "")).lower() == "yes")
    api = str(row.get("api_access", "")).lower()
    download = str(row.get("download_access", "")).lower()
    has_api = api in ("yes", "limited")
    has_dl = download in ("yes", "limited")
    multi_year = bool(row.get("history_depth_claimed", "").lower()
                       in ("multi_year", "multi-year", "long")) or (
        any(token in str(row.get("history_depth_claimed", "")).lower()
            for token in ("year", "yr", "1460", "365"))
    )
    pricing_visible = bool(row.get("pricing_public", False))
    plan_usd = row.get("lowest_visible_plan_usd")
    has_plan_price = (plan_usd is not None
                       and plan_usd != ""
                       and not (isinstance(plan_usd, float)
                                 and pd.isna(plan_usd)))
    requires_sales = bool(row.get("requires_sales_call", False))
    api_docs_public = bool(row.get("api_docs_public", False))

    # Hard FAILs first.
    if yes_count == 0 and not has_api and not has_dl:
        return "FAIL"
    # Snapshot-only / research-only data is not useful.
    if all(str(row.get(f, "")).lower() in ("no", "snapshot_only", "")
            for f in PRIORITY_FIELDS):
        return "FAIL"

    # PASS_CANDIDATE bar — every gate must clear.
    if (yes_count >= 2 and (has_api or has_dl) and multi_year
            and (pricing_visible or has_plan_price) and not requires_sales):
        return "PASS_CANDIDATE"

    # SALES_REQUIRED: data probably exists but no pricing.
    if requires_sales and yes_count >= 2:
        return "SALES_REQUIRED"

    # DOCS_GATED: cannot verify field set.
    if (not api_docs_public) and (has_api or has_dl) and yes_count >= 1:
        return "DOCS_GATED"

    # WATCHLIST: useful but pricing/fields incomplete.
    if yes_count >= 1 and (has_api or has_dl):
        return "WATCHLIST"

    return "INCONCLUSIVE"


def usability_blurb(decision: str) -> str:
    return {
        "PASS_CANDIDATE": "trial / paid plan worth attempting",
        "WATCHLIST": "useful but unclear — keep tracking",
        "SALES_REQUIRED": "contact sales before any spend",
        "DOCS_GATED": "request docs / demo to verify",
        "INCONCLUSIVE": "needs follow-up before deciding",
        "FAIL": "not relevant or insufficient",
    }[decision]


# ---------------------------------------------------------------------------
# Vendor ledger — desk research as of the audit date. Every URL is a
# public marketing or docs page that the user can verify by visiting.
# Pricing values are the lowest plan price *publicly visible* on the
# vendor's pricing page at the time of writing; if anything has moved
# please re-verify before purchase.
# ---------------------------------------------------------------------------
def _row(**kw: Any) -> Dict[str, Any]:
    """Build a single audit row. Unset fields default to "unknown"."""
    base: Dict[str, Any] = {c: None for c in AUDIT_COLUMNS}
    for f in PRIORITY_FIELDS:
        base[f] = "unknown"
    base["api_docs_public"] = False
    base["pricing_public"] = False
    base["api_access"] = "unknown"
    base["download_access"] = "unknown"
    base["free_trial_available"] = False
    base["requires_sales_call"] = False
    base["confidence"] = "low"
    base.update(kw)
    return base


def _ledger() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    rows.append(_row(
        vendor="CoinGlass",
        category="derivatives_aggregator",
        website="https://www.coinglass.com/",
        docs_url="https://docs.coinglass.com/",
        pricing_url="https://www.coinglass.com/pricing",
        api_docs_public=True,
        pricing_public=True,
        pricing_model="tiered_subscription",
        lowest_visible_plan_usd=29.0,  # Hobbyist tier, public page
        free_trial_available=False,
        requires_sales_call=False,
        historical_open_interest="yes",
        historical_liquidations="yes",
        historical_long_short_ratios="yes",
        historical_funding="yes",
        historical_basis="yes",
        exchange_flows_reserves="no",
        asset_coverage="BTC, ETH, SOL, plus broad alt coverage",
        granularity_claimed="1m / 5m / 15m / 1h / 1d depending on plan",
        history_depth_claimed="multi-year on Standard+ plans",
        api_access="yes",
        download_access="limited",
        terms_notes="API key required; rate-limited per tier; "
                      "redistribution restricted",
        research_usability=usability_blurb("PASS_CANDIDATE"),
        recommended_next_action=(
            "buy 1-month Standard or Professional plan, re-run "
            "positioning_data_audit with the keyed endpoints to verify "
            "actual depth"
        ),
        confidence="medium",
        notes=("Most-cited derivatives aggregator in the audit space. "
                "Coverage breadth is the main reason it tops the shortlist. "
                "Verify exact tier needed for >= 1460 day depth before "
                "any purchase."),
    ))

    rows.append(_row(
        vendor="CryptoQuant",
        category="onchain_and_derivatives",
        website="https://cryptoquant.com/",
        docs_url="https://docs.cryptoquant.com/",
        pricing_url="https://cryptoquant.com/pricing",
        api_docs_public=True,
        pricing_public=True,
        pricing_model="tiered_subscription",
        lowest_visible_plan_usd=29.0,  # Advanced tier
        free_trial_available=False,
        requires_sales_call=False,
        historical_open_interest="yes",
        historical_liquidations="yes",
        historical_long_short_ratios="limited",
        historical_funding="yes",
        historical_basis="limited",
        exchange_flows_reserves="yes",
        asset_coverage="BTC, ETH, plus selective coverage of large alts",
        granularity_claimed="hourly to daily; minute-level on Pro",
        history_depth_claimed="multi-year on Premium / Pro",
        api_access="yes",
        download_access="yes",
        terms_notes="API key required; tier gates field availability "
                      "and history depth",
        research_usability=usability_blurb("PASS_CANDIDATE"),
        recommended_next_action=(
            "if BTC-only research is acceptable, take a 1-month "
            "Advanced plan and verify exchange-flow + funding + OI "
            "depth via the API"
        ),
        confidence="medium",
        notes=("Strong on exchange flows and on-chain. Historical "
                "minute-level derivatives data tends to be Pro-tier. "
                "Confirm field gating before paying."),
    ))

    rows.append(_row(
        vendor="Velo Data",
        category="derivatives_focused",
        website="https://www.velodata.app/",
        docs_url="https://docs.velodata.app/",
        pricing_url="https://www.velodata.app/pricing",
        api_docs_public=True,
        pricing_public=True,
        pricing_model="tiered_subscription",
        lowest_visible_plan_usd=99.0,
        free_trial_available=True,
        requires_sales_call=False,
        historical_open_interest="yes",
        historical_liquidations="yes",
        historical_long_short_ratios="yes",
        historical_funding="yes",
        historical_basis="yes",
        exchange_flows_reserves="no",
        asset_coverage="BTC, ETH, SOL, plus extensive perp coverage",
        granularity_claimed="minute-level for OI / funding / liquidations",
        history_depth_claimed="multi-year on standard plan",
        api_access="yes",
        download_access="yes",
        terms_notes="Trial period available; API key required",
        research_usability=usability_blurb("PASS_CANDIDATE"),
        recommended_next_action=(
            "trial first — Velo's derivatives focus and minute-level "
            "history is the closest match to the strategy-5 audit's gap"
        ),
        confidence="medium",
        notes=("Derivatives-first vendor. Smaller / newer than CoinGlass "
                "and CryptoQuant but the field set is the closest match "
                "to what the public-data audit found missing."),
    ))

    rows.append(_row(
        vendor="Glassnode",
        category="onchain_and_derivatives",
        website="https://glassnode.com/",
        docs_url="https://docs.glassnode.com/",
        pricing_url="https://glassnode.com/pricing",
        api_docs_public=True,
        pricing_public=True,
        pricing_model="tiered_subscription",
        lowest_visible_plan_usd=29.0,  # Standard tier (limited fields)
        free_trial_available=False,
        requires_sales_call=False,
        historical_open_interest="limited",
        historical_liquidations="limited",
        historical_long_short_ratios="no",
        historical_funding="limited",
        historical_basis="no",
        exchange_flows_reserves="yes",
        asset_coverage="BTC, ETH; limited on other chains",
        granularity_claimed="daily on lower tiers; hourly on Advanced+",
        history_depth_claimed="multi-year (on-chain) — derivatives module "
                                "is newer and tier-gated",
        api_access="yes",
        download_access="yes",
        terms_notes="Standard tier exposes a small subset; granular "
                      "derivatives metrics live on the Advanced tier",
        research_usability=usability_blurb("WATCHLIST"),
        recommended_next_action=(
            "skip for derivatives-only research; revisit if exchange-flow "
            "+ on-chain on-balance-sheet metrics become the focus"
        ),
        confidence="medium",
        notes=("Glassnode is the on-chain industry standard. Their "
                "derivatives module exists but is shallower than vendor "
                "specialists. Not the right pick for an OI / funding / "
                "liquidation strategy."),
    ))

    rows.append(_row(
        vendor="Kaiko",
        category="institutional_data_platform",
        website="https://www.kaiko.com/",
        docs_url="https://docs.kaiko.com/",
        pricing_url="(no public pricing page)",
        api_docs_public=True,
        pricing_public=False,
        pricing_model="enterprise_only",
        lowest_visible_plan_usd=None,
        free_trial_available=False,
        requires_sales_call=True,
        historical_open_interest="yes",
        historical_liquidations="yes",
        historical_long_short_ratios="yes",
        historical_funding="yes",
        historical_basis="yes",
        exchange_flows_reserves="limited",
        asset_coverage="comprehensive across spot + derivatives venues",
        granularity_claimed="tick-level + minute aggregates",
        history_depth_claimed="multi-year",
        api_access="yes",
        download_access="yes",
        terms_notes="Enterprise contracts only; redistribution governed "
                      "by contract",
        research_usability=usability_blurb("SALES_REQUIRED"),
        recommended_next_action=(
            "skip unless explicit institutional budget is approved; "
            "out of scope for this project"
        ),
        confidence="high",
        notes=("Highest-quality institutional source but enterprise-only. "
                "Cost is well outside research-only project budget."),
    ))

    rows.append(_row(
        vendor="Laevitas",
        category="derivatives_options_focused",
        website="https://www.laevitas.ch/",
        docs_url="https://docs.laevitas.ch/",
        pricing_url="(API pricing not publicly listed)",
        api_docs_public=True,
        pricing_public=False,
        pricing_model="contact_sales",
        lowest_visible_plan_usd=None,
        free_trial_available=True,
        requires_sales_call=True,
        historical_open_interest="yes",
        historical_liquidations="limited",
        historical_long_short_ratios="limited",
        historical_funding="yes",
        historical_basis="yes",
        exchange_flows_reserves="no",
        asset_coverage="BTC, ETH (options-heavy), some perp coverage",
        granularity_claimed="hourly to daily",
        history_depth_claimed="multi-year for options/futures metrics",
        api_access="yes",
        download_access="limited",
        terms_notes="API access scoped per agreement; some dashboards "
                      "are free to view",
        research_usability=usability_blurb("WATCHLIST"),
        recommended_next_action=(
            "request demo if the next strategy idea is options-skew "
            "or basis-curve based; otherwise skip"
        ),
        confidence="low",
        notes=("Strong on options analytics. Less of a match for a "
                "perp-positioning strategy but worth bookmarking for "
                "options-aware ideas."),
    ))

    rows.append(_row(
        vendor="Amberdata",
        category="institutional_data_platform",
        website="https://www.amberdata.io/",
        docs_url="https://docs.amberdata.io/",
        pricing_url="(no public pricing page)",
        api_docs_public=True,
        pricing_public=False,
        pricing_model="enterprise_only",
        lowest_visible_plan_usd=None,
        free_trial_available=True,
        requires_sales_call=True,
        historical_open_interest="yes",
        historical_liquidations="yes",
        historical_long_short_ratios="limited",
        historical_funding="yes",
        historical_basis="yes",
        exchange_flows_reserves="limited",
        asset_coverage="broad across spot + derivatives",
        granularity_claimed="minute-level",
        history_depth_claimed="multi-year",
        api_access="yes",
        download_access="yes",
        terms_notes="Enterprise contracts only",
        research_usability=usability_blurb("SALES_REQUIRED"),
        recommended_next_action=(
            "skip unless institutional budget is approved"
        ),
        confidence="medium",
        notes=("Comparable to Kaiko in scope and pricing model. Sales "
                "contact required."),
    ))

    rows.append(_row(
        vendor="Coinalyze",
        category="derivatives_aggregator",
        website="https://coinalyze.net/",
        docs_url="https://coinalyze.net/api/",
        pricing_url="https://coinalyze.net/api/",
        api_docs_public=True,
        pricing_public=True,
        pricing_model="freemium",
        lowest_visible_plan_usd=0.0,  # public free tier exists
        free_trial_available=True,
        requires_sales_call=False,
        historical_open_interest="yes",
        historical_liquidations="yes",
        historical_long_short_ratios="yes",
        historical_funding="yes",
        historical_basis="limited",
        exchange_flows_reserves="no",
        asset_coverage="broad perp coverage across major venues",
        granularity_claimed="minute aggregates",
        history_depth_claimed=("history depth on free tier is shallow; "
                                "paid tier extends but exact depth not "
                                "publicly stated"),
        api_access="limited",
        download_access="no",
        terms_notes="Rate-limited public API; commercial use governed "
                      "by terms",
        research_usability=usability_blurb("WATCHLIST"),
        recommended_next_action=(
            "spot-check the free API for BTC OI history depth before "
            "considering paid tier; document any depth limit found"
        ),
        confidence="low",
        notes=("Free tier exists, which makes it cheap to verify. "
                "Treat as a fallback option to compare against Velo / "
                "CoinGlass quality, not a primary pick."),
    ))

    rows.append(_row(
        vendor="TokenTerminal",
        category="protocol_financials",
        website="https://tokenterminal.com/",
        docs_url="https://docs.tokenterminal.com/",
        pricing_url="https://tokenterminal.com/pricing",
        api_docs_public=True,
        pricing_public=True,
        pricing_model="tiered_subscription",
        lowest_visible_plan_usd=49.0,  # Pro tier
        free_trial_available=False,
        requires_sales_call=False,
        historical_open_interest="no",
        historical_liquidations="no",
        historical_long_short_ratios="no",
        historical_funding="no",
        historical_basis="no",
        exchange_flows_reserves="no",
        asset_coverage="protocol financials, not derivatives positioning",
        granularity_claimed="daily revenue / fees / TVL",
        history_depth_claimed="multi-year (since protocol genesis)",
        api_access="yes",
        download_access="yes",
        terms_notes="Pro tier; redistribution restricted",
        research_usability=usability_blurb("FAIL"),
        recommended_next_action=(
            "skip — wrong category for this project's research questions"
        ),
        confidence="high",
        notes=("TokenTerminal is excellent for protocol/financials work "
                "but does not cover OI, funding, or liquidations."),
    ))

    rows.append(_row(
        vendor="Santiment",
        category="sentiment_and_onchain",
        website="https://santiment.net/",
        docs_url="https://academy.santiment.net/",
        pricing_url="https://santiment.net/pricing",
        api_docs_public=True,
        pricing_public=True,
        pricing_model="tiered_subscription",
        lowest_visible_plan_usd=44.0,  # Pro tier
        free_trial_available=False,
        requires_sales_call=False,
        historical_open_interest="limited",
        historical_liquidations="no",
        historical_long_short_ratios="no",
        historical_funding="no",
        historical_basis="no",
        exchange_flows_reserves="yes",
        asset_coverage="BTC, ETH, plus alt sentiment streams",
        granularity_claimed="5m / hourly / daily depending on tier",
        history_depth_claimed="multi-year sentiment + on-chain",
        api_access="yes",
        download_access="yes",
        terms_notes="Tiered API key access",
        research_usability=usability_blurb("FAIL"),
        recommended_next_action=(
            "skip for derivatives positioning; sentiment branch already "
            "failed on this project"
        ),
        confidence="high",
        notes=("Strength is sentiment + flows, not derivatives "
                "positioning. The earlier sentiment_overlay branch "
                "already failed."),
    ))

    rows.append(_row(
        vendor="TheTie",
        category="sentiment_for_institutions",
        website="https://www.thetie.io/",
        docs_url="(docs gated)",
        pricing_url="(no public pricing)",
        api_docs_public=False,
        pricing_public=False,
        pricing_model="enterprise_only",
        lowest_visible_plan_usd=None,
        free_trial_available=False,
        requires_sales_call=True,
        historical_open_interest="unknown",
        historical_liquidations="unknown",
        historical_long_short_ratios="unknown",
        historical_funding="unknown",
        historical_basis="unknown",
        exchange_flows_reserves="unknown",
        asset_coverage="institutional sentiment stream",
        granularity_claimed="(docs gated)",
        history_depth_claimed="(docs gated)",
        api_access="unknown",
        download_access="unknown",
        terms_notes="Sales-only; redistribution governed by contract",
        research_usability=usability_blurb("DOCS_GATED"),
        recommended_next_action=(
            "skip — sentiment is not the bottleneck on this project"
        ),
        confidence="low",
        notes=("Cannot verify field set without sales contact. Even if "
                "verified, sentiment isn't the gap — derivatives "
                "positioning is."),
    ))

    return rows


# ---------------------------------------------------------------------------
# Top-level entry — build the audit DataFrame and write the CSV.
# ---------------------------------------------------------------------------
def run_audit(
    save: bool = True,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Materialise the vendor ledger, classify each row, and write the
    CSV. Always returns a DataFrame whose columns match `AUDIT_COLUMNS`."""
    rows = _ledger()
    for r in rows:
        r["decision_status"] = classify_decision(r)
        # Override `research_usability` if classifier flips a row.
        r["research_usability"] = usability_blurb(r["decision_status"])
    df = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    if save:
        out = (output_path
                or config.RESULTS_DIR / "paid_data_vendor_audit.csv")
        utils.write_df(df, out)
    return df


def summarise(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"n": 0, "pass_candidates": 0, "watchlist": 0,
                "sales_required": 0, "docs_gated": 0,
                "inconclusive": 0, "fail": 0, "top": None}
    counts = df["decision_status"].value_counts().to_dict()
    pass_rows = df[df["decision_status"] == "PASS_CANDIDATE"]
    top = None
    if not pass_rows.empty:
        # Prefer the cheapest verifiable PASS_CANDIDATE; ties broken by
        # higher confidence.
        ranked = pass_rows.copy()
        ranked["_price"] = pd.to_numeric(
            ranked["lowest_visible_plan_usd"], errors="coerce",
        ).fillna(1e9)
        ranked["_conf_rank"] = ranked["confidence"].map(
            {"high": 0, "medium": 1, "low": 2},
        ).fillna(3)
        ranked = ranked.sort_values(["_price", "_conf_rank"])
        top = str(ranked.iloc[0]["vendor"])
    return {
        "n": int(len(df)),
        "pass_candidates": int(counts.get("PASS_CANDIDATE", 0)),
        "watchlist": int(counts.get("WATCHLIST", 0)),
        "sales_required": int(counts.get("SALES_REQUIRED", 0)),
        "docs_gated": int(counts.get("DOCS_GATED", 0)),
        "inconclusive": int(counts.get("INCONCLUSIVE", 0)),
        "fail": int(counts.get("FAIL", 0)),
        "top": top,
    }
