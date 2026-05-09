"""Shared helpers: paths, logging, IO, and the paper-only safety guard."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Hard safety guard
# ---------------------------------------------------------------------------
class LiveTradingForbiddenError(RuntimeError):
    """Raised if any module attempts a privileged operation while
    LIVE_TRADING_ENABLED is True. Version 1 has no execution module — this
    flag must remain False."""


def assert_paper_only() -> None:
    """Call this at the top of any function that simulates trades or fetches
    data. If someone has flipped the safety lock without writing a separate
    audited execution module, we refuse to run."""
    if config.LIVE_TRADING_ENABLED:
        raise LiveTradingForbiddenError(
            "LIVE_TRADING_ENABLED is True but no execution module exists in "
            "version 1. Refusing to run. Revert config.LIVE_TRADING_ENABLED "
            "to False, or implement and audit a separate execution module."
        )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_INITIALISED = False


def get_logger(name: str = "cte") -> logging.Logger:
    """Return a singleton, console-friendly logger."""
    global _LOG_INITIALISED
    logger = logging.getLogger(name)
    if not _LOG_INITIALISED:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        # Avoid duplicate handlers if Streamlit reloads the module.
        if not logger.handlers:
            logger.addHandler(h)
        logger.propagate = False
        _LOG_INITIALISED = True
    return logger


# ---------------------------------------------------------------------------
# File system
# ---------------------------------------------------------------------------
def ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def safe_symbol(symbol: str) -> str:
    """Convert 'BTC/USDT' to a filesystem-safe stem like 'BTC_USDT'."""
    return symbol.replace("/", "_").replace(":", "_")


def csv_path_for(symbol: str, timeframe: str) -> Path:
    return config.DATA_RAW_DIR / f"{safe_symbol(symbol)}_{timeframe}.csv"


def write_df(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV, creating parent dirs and warning before
    overwrite (we only warn — backtests are reproducible)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        get_logger().info("overwriting %s (%d existing rows)", path.name,
                          sum(1 for _ in path.open()) - 1)
    df.to_csv(path, index=False)


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path)


def parse_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 'timestamp' is int64 ms and 'datetime' is a UTC pandas Timestamp."""
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce").astype("Int64")
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
    return out
