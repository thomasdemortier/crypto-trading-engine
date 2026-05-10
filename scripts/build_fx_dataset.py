#!/usr/bin/env python3
"""
Build the v1 FX research dataset — a thin wrapper around
`src.fx_research_dataset.build_and_write`.

Usage:
    python scripts/build_fx_dataset.py

Outputs (gitignored):
    data/fx/fx_daily_v1.parquet
    data/fx/fx_daily_v1.csv

Hard rules (locked):
    * Read-only public endpoints; no API keys ever read.
    * No broker, no execution, no paper trading, no live trading.
    * Network failures degrade to `missing` placeholder rows + a
      printed warning; the script never crashes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _add_repo_root_to_path() -> Path:
    """Allow running the script directly from the repo root."""
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def main() -> int:
    _add_repo_root_to_path()
    from src import fx_research_dataset  # noqa: E402 — late import

    df, summary = fx_research_dataset.build_and_write()
    print("=== FX research dataset v1 ===")
    print(f"  rows                : {summary['row_count']}")
    print(f"  assets              : {summary['asset_count']}")
    print(f"  start_date          : {summary['start_date']}")
    print(f"  end_date            : {summary['end_date']}")
    print(f"  coverage_days       : {summary['coverage_days']}")
    print(f"  missing_value_count : {summary['missing_value_count']}")
    print(f"  assets_available    : {summary['assets_available']}")
    print(f"  assets_missing      : {summary['assets_missing']}")
    print()
    print("  per-asset coverage:")
    cov = summary.get("coverage_by_asset", {})
    if cov:
        for asset in sorted(cov):
            row = cov[asset]
            kind = "derived" if row["is_derived"] else "direct"
            print(f"    {asset:<10} src={row['source']:<13} {kind:<7} "
                  f"rows={row['rows']:<6} "
                  f"{row['start_date']} → {row['end_date']} "
                  f"({row['coverage_days']}d)")
    else:
        print("    (none)")
    warnings = summary.get("data_quality_warnings", []) or []
    if warnings:
        print()
        print(f"  {len(warnings)} data-quality warning(s):")
        for w in warnings:
            print(f"    - {w}")
    written = summary.get("written", {}) or {}
    print()
    if "parquet" in written:
        print(f"Saved → {written['parquet']}")
    if "parquet_error" in written:
        print(f"Parquet engine unavailable: {written['parquet_error']}")
    if "csv" in written:
        print(f"Saved → {written['csv']}")
    # Also dump a JSON summary alongside (gitignored under data/fx/*.json).
    try:
        summary_path = Path(written.get("parquet")
                              or written.get("csv")).with_name(
            "fx_daily_v1_summary.json"
        )
        summary_for_json = {k: v for k, v in summary.items()
                              if k != "written"}
        summary_path.write_text(
            json.dumps(summary_for_json, indent=2, default=str)
        )
        print(f"Saved → {summary_path}")
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
